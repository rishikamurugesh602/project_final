"""
Unit tests for streaming buffer, queue bounds, backpressure, WebRTC signaling,
protocol validation, and privacy/history clearing.
"""
import os
import sys
import tempfile
import numpy as np
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.streaming_processor import BoundedAudioBuffer, StreamingAnalysisSession
from app.services.call_manager import CallSessionManager
from app.database.database import SessionLocal, init_database
from app.database.crud import AnalysisResultCRUD
from training.validate_protocol import validate_asvspoof_protocol


class TestBoundedAudioBuffer:
    """Test sliding window audio buffer and chunk boundary handling."""

    def test_buffer_bounds_max_samples(self):
        """Ensure buffer never grows beyond max_duration_sec * sample_rate."""
        sr = 16000
        max_duration = 2.0
        max_samples = int(max_duration * sr)
        buffer = BoundedAudioBuffer(max_duration_sec=max_duration, sample_rate=sr)

        # Add 3 seconds of audio in 1-second chunks (total 48,000 samples)
        chunk1 = np.ones(sr, dtype=np.float32) * 1.0
        chunk2 = np.ones(sr, dtype=np.float32) * 2.0
        chunk3 = np.ones(sr, dtype=np.float32) * 3.0

        buffer.add_chunk(chunk1)
        assert len(buffer.get_window()) == sr

        buffer.add_chunk(chunk2)
        assert len(buffer.get_window()) == 2 * sr

        buffer.add_chunk(chunk3)
        # Bounded at 2.0 seconds (32,000 samples)
        window = buffer.get_window()
        assert len(window) == max_samples
        # Oldest chunk1 must have been pushed out: window should contain chunk2 and chunk3
        assert np.all(window[:sr] == 2.0)
        assert np.all(window[sr:] == 3.0)

    def test_chunk_boundaries_and_empty_chunks(self):
        """Verify handling of empty or varying chunk sizes across boundaries."""
        buffer = BoundedAudioBuffer(max_duration_sec=3.0, sample_rate=16000)

        # Add empty chunk
        buffer.add_chunk(np.array([], dtype=np.float32))
        assert len(buffer.get_window()) == 0

        # Add variable length chunks
        c1 = np.arange(100, dtype=np.float32)
        c2 = np.arange(100, 250, dtype=np.float32)
        buffer.add_chunk(c1)
        buffer.add_chunk(c2)

        window = buffer.get_window()
        assert len(window) == 250
        assert np.array_equal(window[:100], c1)
        assert np.array_equal(window[100:], c2)

    def test_buffer_reset(self):
        """Verify reset clears buffer and sample counters."""
        buffer = BoundedAudioBuffer(max_duration_sec=2.0, sample_rate=16000)
        buffer.add_chunk(np.ones(1000, dtype=np.float32))
        assert len(buffer.get_window()) == 1000

        buffer.reset()
        assert len(buffer.get_window()) == 0
        assert buffer.total_samples_received == 0
        assert buffer.chunks_received == 0


class TestStreamingAnalysisSessionQueueAndBackpressure:
    """Test bounded queue limits, backpressure handling, and disconnect cleanup."""

    def test_queue_limit_and_backpressure_drop(self):
        """Verify queue maxsize is respected and backpressure drops oldest chunks."""
        max_q = 3
        session = StreamingAnalysisSession(
            orchestrator=None,
            sample_rate=16000,
            max_queue_size=max_q
        )

        dummy_chunk = np.zeros(1000, dtype=np.float32)

        # Fill queue to capacity
        for i in range(max_q):
            enqueued = session.enqueue_chunk(dummy_chunk, client_timestamp=float(i))
            assert enqueued is True

        assert session.queue.qsize() == max_q
        assert session.dropped_chunks_count == 0

        # Push additional chunks beyond capacity
        session.enqueue_chunk(dummy_chunk, client_timestamp=10.0)
        # Should have dropped the oldest chunk to relieve backpressure
        assert session.queue.qsize() == max_q
        assert session.dropped_chunks_count == 1

        session.close()

    def test_session_close_and_disconnect(self):
        """Verify session close drains queue, resets audio buffer, and marks inactive."""
        session = StreamingAnalysisSession(
            orchestrator=None,
            sample_rate=16000,
            max_queue_size=5
        )

        session.enqueue_chunk(np.zeros(2000, dtype=np.float32))
        assert session.queue.qsize() == 1

        session.close()
        assert session._is_active is False
        assert session.queue.empty()
        assert len(session.audio_buffer.get_window()) == 0

        # Further enqueues after close must be rejected
        accepted = session.enqueue_chunk(np.zeros(1000, dtype=np.float32))
        assert accepted is False


class TestCallSessionManager:
    """Test WebRTC call session tracking and mutual consent."""

    def test_session_creation_roles_and_consent(self):
        mgr = CallSessionManager()
        call_id = "CALL-TEST-01"

        session = mgr.get_or_create_session(call_id)
        assert session.call_id == call_id

        class MockWS:
            pass

        ws1 = MockWS()
        ws2 = MockWS()

        p1 = session.add_participant("peer_alice", ws1)
        assert p1.role == "caller"
        assert session.both_consented() is False

        p2 = session.add_participant("peer_bob", ws2)
        assert p2.role == "callee"
        assert session.both_consented() is False

        # Alice grants consent
        session.set_consent("peer_alice", True)
        assert session.both_consented() is False

        # Bob grants consent
        session.set_consent("peer_bob", True)
        assert session.both_consented() is True

        # Alice revokes consent
        session.set_consent("peer_alice", False)
        assert session.both_consented() is False


class TestHistoryClearing:
    """Test database history deletion endpoint and CRUD."""

    def test_clear_all_records(self):
        init_database()
        db = SessionLocal()
        try:
            # Create a sample record
            dummy_result = {
                "session_id": "test-session-uuid",
                "deepfake_probability": 0.12,
                "deepfake_prediction": "Genuine",
                "transcript": "hello world",
                "scam_probability": 0.05,
                "scam_indicators": [],
                "overall_risk_score": 0.08,
                "risk_category": "LOW",
                "risk_reasons": ["Authentic voice patterns"],
                "total_processing_time": 0.25
            }
            AnalysisResultCRUD.create(db, dummy_result)

            # Clear all
            count = AnalysisResultCRUD.clear_all(db)
            assert count >= 1

            # Verify empty
            recent = AnalysisResultCRUD.get_recent(db, limit=10)
            assert len(recent) == 0
        finally:
            db.close()


class TestASVspoofProtocolValidation:
    """Test ASVspoof protocol validator with synthetic mock files."""

    def test_protocol_validator_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir) / "flac"
            audio_dir.mkdir()

            (audio_dir / "utt_001.flac").write_bytes(b"dummy")
            (audio_dir / "utt_002.flac").write_bytes(b"dummy")

            proto_file = Path(tmpdir) / "protocol.txt"
            proto_file.write_text(
                "SPK01 utt_001 - - bonafide\n"
                "SPK02 utt_002 - - spoof\n",
                encoding="utf-8"
            )

            res = validate_asvspoof_protocol(str(proto_file), str(audio_dir))
            assert res["valid"] is True
            assert res["total_protocol_entries"] == 2
            assert res["class_counts"]["bonafide"] == 1
            assert res["class_counts"]["spoof"] == 1
            assert res["missing_audio_files_count"] == 0

    def test_protocol_validator_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir) / "flac"
            audio_dir.mkdir()

            (audio_dir / "utt_001.flac").write_bytes(b"dummy")

            proto_file = Path(tmpdir) / "protocol.txt"
            proto_file.write_text(
                "SPK01 utt_001 - - bonafide\n"
                "SPK02 utt_002 - - spoof\n",
                encoding="utf-8"
            )

            res = validate_asvspoof_protocol(str(proto_file), str(audio_dir))
            assert res["valid"] is False
            assert res["found_audio_files"] == 1
            assert res["missing_audio_files_count"] == 1
            assert "utt_002" in res["missing_examples"]
