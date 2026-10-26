"""
WebSocket endpoint for real-time audio streaming and analysis.
"""
import json
import asyncio
import base64
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.deepfake_service import AnalysisOrchestrator
from app.services.audio_processor import AudioProcessor
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
    WebSocket endpoint for streaming audio analysis.
    
    Client sends:
    {
        "type": "audio_chunk",
        "data": <base64_encoded_audio>,
        "sample_rate": 16000,
        "is_final": false
    }
    
    Server responds with:
    {
        "type": "status|result|error",
        "data": {...}
    }
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    audio_chunks = []
    sample_rate = None
    session_data = {
        "total_bytes": 0,
        "chunks_received": 0
    }
    
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

                    audio_chunks.append(audio_chunk)

                    if sample_rate is None:
                        sample_rate = message.get("sample_rate", config.SAMPLE_RATE)

                    session_data["chunks_received"] += 1
                    session_data["total_bytes"] += len(audio_payload) if isinstance(audio_payload, str) else audio_chunk.nbytes

                    await websocket.send_json({
                        "type": "status",
                        "data": {
                            "chunks_received": session_data["chunks_received"],
                            "total_bytes": session_data["total_bytes"],
                            "message": "Chunk received"
                        }
                    })

                    is_final = bool(message.get("is_final", False))
                    if is_final or session_data["chunks_received"] % 10 == 0:
                        if audio_chunks:
                            full_audio = np.concatenate(audio_chunks)
                            await websocket.send_json({
                                "type": "status",
                                "data": {"message": "Processing audio..."}
                            })

                            result = orchestrator.analyze(full_audio, sample_rate)
                            await websocket.send_json({
                                "type": "result",
                                "data": result
                            })

                            if is_final:
                                audio_chunks = []
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "data": {"error": str(e)}
                    })
            
            elif message_type == "process":
                if audio_chunks:
                    full_audio = np.concatenate(audio_chunks)
                    result = orchestrator.analyze(full_audio, sample_rate)
                    await websocket.send_json({
                        "type": "result",
                        "data": result
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"error": "No audio data to process"}
                    })
            
            elif message_type == "reset":
                audio_chunks = []
                sample_rate = None
                session_data = {"total_bytes": 0, "chunks_received": 0}
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
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"error": str(e)}
            })
        except Exception:
            pass
