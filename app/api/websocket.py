"""
WebSocket endpoint for real-time audio streaming and analysis.
Employs bounded sliding audio windows, bounded queues, and offloaded model inference.
"""
import json
import time
import asyncio
import base64
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.deepfake_service import AnalysisOrchestrator
from app.services.audio_processor import AudioProcessor
from app.services.streaming_processor import StreamingAnalysisSession
from app.utils.config import config
from app.utils.logging import logger

router = APIRouter(prefix="/api/v1", tags=["websocket"])

orchestrator = AnalysisOrchestrator()
audio_processor = AudioProcessor()


def decode_audio_payload(payload):
    """Decode audio data from either a base64 string or a numeric Python/JS array."""
    if payload is None:
        return np.array([], dtype=np.float32)

    if isinstance(payload, str):
        try:
            decoded = base64.b64decode(payload, validate=True)
            return np.frombuffer(decoded, dtype=np.float32)
        except Exception:
            try:
                return np.asarray(json.loads(payload), dtype=np.float32)
            except Exception:
                return np.array([], dtype=np.float32)

    if isinstance(payload, (bytes, bytearray)):
        return np.frombuffer(payload, dtype=np.float32)

    if isinstance(payload, list):
        return np.asarray(payload, dtype=np.float32)

    if isinstance(payload, np.ndarray):
        return payload.astype(np.float32)

    return np.array([], dtype=np.float32)


@router.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio analysis with bounded buffers and queueing.
    
    Client sends:
    {
        "type": "audio_chunk",
        "data": <base64_encoded_audio>,
        "sample_rate": 16000,
        "is_final": false,
        "timestamp": 123456789.0
    }
    
    Server responds with:
    {
        "type": "status|result|error",
        "data": {...}
    }
    """
    await websocket.accept()
    logger.info("WebSocket analysis connection established")
    
    stream_session = StreamingAnalysisSession(
        orchestrator=orchestrator,
        sample_rate=config.SAMPLE_RATE,
        max_queue_size=30,
        asr_interval_sec=5.0
    )
    
    chunks_received = 0
    total_bytes = 0
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            message_type = message.get("type", "")
            
            if message_type == "audio_chunk":
                try:
                    audio_payload = message.get("data", "")
                    audio_chunk = decode_audio_payload(audio_payload)
                    if audio_chunk.size == 0:
                        raise ValueError("Empty audio chunk received")

                    chunks_received += 1
                    total_bytes += len(audio_payload) if isinstance(audio_payload, str) else audio_chunk.nbytes
                    client_ts = message.get("timestamp", time.time())

                    # Enqueue chunk in bounded queue with backpressure handling
                    enqueued = stream_session.enqueue_chunk(audio_chunk, client_timestamp=client_ts)

                    # Send immediate status acknowledgement
                    await websocket.send_json({
                        "type": "status",
                        "data": {
                            "chunks_received": chunks_received,
                            "total_bytes": total_bytes,
                            "queue_depth": stream_session.queue.qsize(),
                            "dropped_chunks": stream_session.dropped_chunks_count,
                            "state": stream_session.processing_state,
                            "message": "Chunk enqueued" if enqueued else "Backpressure active: chunk dropped"
                        }
                    })

                    is_final = bool(message.get("is_final", False))
                    # Trigger analysis every 3 chunks (~0.75s at 4096 samples/chunk) or on final chunk
                    if is_final or (chunks_received % 3 == 0):
                        result = await stream_session.process_next(is_final=is_final)
                        if result:
                            await websocket.send_json({
                                "type": "result",
                                "data": result
                            })

                except Exception as e:
                    logger.error(f"Error processing audio chunk: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "data": {"error": str(e)}
                    })
            
            elif message_type == "process":
                try:
                    result = await stream_session.process_next(is_final=True)
                    if result:
                        await websocket.send_json({
                            "type": "result",
                            "data": result
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "data": {"error": "Insufficient audio data to process"}
                        })
                except Exception as e:
                    logger.error(f"Error in process command: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "data": {"error": str(e)}
                    })
            
            elif message_type == "reset":
                stream_session.close()
                stream_session = StreamingAnalysisSession(
                    orchestrator=orchestrator,
                    sample_rate=config.SAMPLE_RATE,
                    max_queue_size=30,
                    asr_interval_sec=5.0
                )
                chunks_received = 0
                total_bytes = 0
                await websocket.send_json({
                    "type": "status",
                    "data": {"message": "Session reset"}
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "data": {"error": f"Unknown message type: {message_type}"}
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket analysis connection closed by client")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"error": str(e)}
            })
        except Exception:
            pass
    finally:
        stream_session.close()
