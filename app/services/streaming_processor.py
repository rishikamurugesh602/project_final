"""
Streaming audio processor with bounded sliding windows, bounded queues, and backpressure handling.
Ensures real-time low-latency analysis without ever-growing buffers or event-loop blocking.
"""
import time
import asyncio
from typing import Dict, Optional, Tuple, Any
import numpy as np
from app.utils.config import config
from app.utils.logging import logger


class BoundedAudioBuffer:
    """
    Sliding-window bounded audio buffer.
    Maintains only the most recent N seconds of audio for deepfake acoustic analysis,
    preventing infinite buffer growth.
    """
    def __init__(self, max_duration_sec: float = 3.5, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        self.max_samples = int(max_duration_sec * sample_rate)
        self._buffer = np.array([], dtype=np.float32)
        self.total_samples_received = 0
        self.chunks_received = 0

    def add_chunk(self, chunk: np.ndarray) -> int:
        """
        Append chunk and bound buffer to max_samples.
        Returns the number of samples in the bounded buffer.
        """
        if chunk is None or chunk.size == 0:
            return len(self._buffer)

        chunk_float = chunk.astype(np.float32)
        if self._buffer.size == 0:
            self._buffer = chunk_float
        else:
            self._buffer = np.concatenate((self._buffer, chunk_float))

        self.total_samples_received += len(chunk_float)
        self.chunks_received += 1

        # Bound to sliding window
        if len(self._buffer) > self.max_samples:
            self._buffer = self._buffer[-self.max_samples:]

        return len(self._buffer)

    def get_window(self) -> np.ndarray:
        """Return the current bounded sliding audio window."""
        return self._buffer.copy()

    def has_minimum_samples(self, min_samples: int = 4000) -> bool:
        """Check if buffer has enough audio for acoustic feature extraction (e.g. 0.25s)."""
        return len(self._buffer) >= min_samples

    def reset(self):
        """Reset buffer state."""
        self._buffer = np.array([], dtype=np.float32)
        self.total_samples_received = 0
        self.chunks_received = 0


class StreamingAnalysisSession:
    """
    Manages an active streaming analysis session over WebSocket.
    Coordinates bounded queues, sliding audio window, backpressure handling,
    and throttled ASR.
    """
    def __init__(
        self,
        orchestrator: Any = None,
        sample_rate: int = 16000,
        max_queue_size: int = 25,
        asr_interval_sec: float = 5.0
    ):
        self.orchestrator = orchestrator
        self.sample_rate = sample_rate
        self.max_queue_size = max_queue_size
        self.asr_interval_sec = asr_interval_sec

        self.audio_buffer = BoundedAudioBuffer(max_duration_sec=3.5, sample_rate=sample_rate)
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        
        self.dropped_chunks_count = 0
        self.last_asr_time = 0.0
        self.latest_transcript = ""
        self.latest_result: Optional[Dict] = None
        self.processing_state = "idle"
        self._is_active = True

    def enqueue_chunk(self, chunk: np.ndarray, client_timestamp: Optional[float] = None) -> bool:
        """
        Push incoming chunk into bounded queue with backpressure handling.
        Returns True if enqueued, False if dropped due to backpressure.
        """
        if not self._is_active:
            return False

        item = {
            "chunk": chunk,
            "received_at": time.time(),
            "client_timestamp": client_timestamp or time.time()
        }

        if self.queue.full():
            # Backpressure handling: drop oldest unhandled chunk to keep latency bounded
            try:
                self.queue.get_nowait()
                self.dropped_chunks_count += 1
                logger.warning("Streaming queue full: dropped oldest chunk to relieve backpressure")
            except (asyncio.QueueEmpty, ValueError):
                pass

        try:
            self.queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            self.dropped_chunks_count += 1
            return False

    async def process_next(self, is_final: bool = False) -> Optional[Dict]:
        """
        Process the next chunk or batch in the queue off FastAPI's event loop.
        Returns analysis result dictionary or None.
        """
        if self.queue.empty() and not is_final:
            return None

        items_processed = 0
        latest_client_ts = None
        receive_time = time.time()

        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                self.audio_buffer.add_chunk(item["chunk"])
                latest_client_ts = item["client_timestamp"]
                receive_time = item["received_at"]
                items_processed += 1
            except asyncio.QueueEmpty:
                break

        if not self.audio_buffer.has_minimum_samples(min_samples=4000):
            return None

        window_audio = self.audio_buffer.get_window()
        current_time = time.time()

        should_run_asr = is_final or (current_time - self.last_asr_time >= self.asr_interval_sec)

        self.processing_state = "analyzing"
        inference_start = time.time()

        if self.orchestrator is not None:
            # Run CPU/GPU heavy inference in thread pool off event loop
            result = await asyncio.to_thread(
                self.orchestrator.analyze,
                audio=window_audio,
                sr=self.sample_rate,
                run_asr=should_run_asr,
                existing_transcript=self.latest_transcript
            )
        else:
            result = {
                "deepfake_probability": 0.5,
                "deepfake_prediction": "Demo",
                "scam_probability": 0.0,
                "scam_indicators": [],
                "transcript": self.latest_transcript,
                "overall_risk_score": 0.25,
                "risk_category": "LOW",
                "recommendation": "Mock test analysis",
                "timing_metrics": {"total_processing_time": 0.01}
            }

        inference_end = time.time()
        self.processing_state = "idle"

        if should_run_asr and result.get("asr_success"):
            self.last_asr_time = inference_end
            self.latest_transcript = result.get("transcript", "")

        end_to_end_latency_ms = (inference_end - (latest_client_ts or receive_time)) * 1000.0

        result["latency_metrics"] = {
            "end_to_end_latency_ms": round(end_to_end_latency_ms, 2),
            "inference_duration_ms": round((inference_end - inference_start) * 1000.0, 2),
            "queue_depth": self.queue.qsize(),
            "dropped_chunks": self.dropped_chunks_count,
            "bounded_buffer_seconds": round(len(window_audio) / self.sample_rate, 2),
            "asr_executed": should_run_asr
        }
        result["processing_state"] = self.processing_state

        self.latest_result = result
        return result

    def close(self):
        """Close the session and clear resources."""
        self._is_active = False
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break
        self.audio_buffer.reset()
