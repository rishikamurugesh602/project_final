"""
Unit tests for the deepfake voice detection system.
"""
import pytest
import numpy as np
import torch
import tempfile
import os

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.audio_processor import AudioProcessor
from app.models.deepfake_detector import DeepfakeDetectorModel, DeepfakeDetector
from app.models.scam_detector import ScamDetector
from app.services.risk_engine import RiskEngine
from app.utils.config import config


class TestAudioProcessor:
    """Test audio preprocessing."""
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        processor = AudioProcessor()
        
        # Create test audio
        audio = np.array([0.5, 1.0, -0.8, 0.3])
        normalized = processor.normalize_audio(audio)
        
        # Check range
        assert np.max(np.abs(normalized)) <= 1.0
        assert len(normalized) == len(audio)
    
    def test_extract_mel_spectrogram(self):
        """Test Mel spectrogram extraction."""
        processor = AudioProcessor()
        
        # Create synthetic audio
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        mel_spec = processor.extract_mel_spectrogram(audio)
        
        # Check shape
        assert mel_spec.shape[0] == 128  # n_mels
        assert mel_spec.shape[1] > 0  # time steps
    
    def test_chunk_audio(self):
        """Test audio chunking."""
        processor = AudioProcessor()
        
        # Create synthetic audio
        duration = 5.0
        sr = 16000
        audio = np.random.randn(int(sr * duration))
        
        chunks = processor.chunk_audio(audio)
        
        # Check chunks
        assert len(chunks) > 0
        assert all(len(chunk) == processor.chunk_samples for chunk in chunks)


class TestDeepfakeDetector:
    """Test deepfake detector model."""
    
    def test_model_creation(self):
        """Test model initialization."""
        model = DeepfakeDetectorModel()
        assert model is not None
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = DeepfakeDetectorModel()
        model.eval()
        
        # Create dummy input
        batch_size = 2
        n_mels = 128
        time_steps = 100
        x = torch.randn(batch_size, 1, n_mels, time_steps)
        
        with torch.no_grad():
            output = model(x)
        
        # Check output shape and values
        assert output.shape == (batch_size, 2)
        assert torch.all(torch.isfinite(output))
    
    def test_detector_predict(self):
        """Test detector prediction."""
        detector = DeepfakeDetector(device='cpu')
        
        # Create dummy mel spectrogram
        mel_spec = np.random.randn(128, 100)
        
        result = detector.predict(mel_spec)
        
        # Check result structure
        assert 'deepfake_probability' in result
        assert 'is_deepfake' in result
        assert 'prediction_class' in result
        
        # Check value ranges
        assert 0 <= result['deepfake_probability'] <= 1


class TestScamDetector:
    """Test scam detection."""
    
    def test_suspicious_indicators(self):
        """Test detection of suspicious keywords."""
        detector = ScamDetector()
        
        # Test with suspicious text
        text = "Please send me your bank account password immediately for verification"
        result = detector.predict(text)
        
        assert result['scam_probability'] > 0
        assert len(result['indicators']) > 0
    
    def test_normal_text(self):
        """Test with normal text."""
        detector = ScamDetector()
        
        text = "Hello, how are you today?"
        result = detector.predict(text)
        
        assert result['scam_probability'] < 0.5 or result['scam_probability'] == 0
    
    def test_empty_text(self):
        """Test with empty text."""
        detector = ScamDetector()
        
        result = detector.predict("")
        
        assert result['scam_probability'] == 0.0
        assert len(result['indicators']) == 0


class TestRiskEngine:
    """Test risk decision engine."""
    
    def test_risk_calculation_low(self):
        """Test LOW risk calculation."""
        engine = RiskEngine()
        
        result = engine.calculate_risk(0.1, 0.2)
        
        assert result['risk_category'] == 'LOW'
        assert result['overall_risk_score'] <= config.RISK_LOW_THRESHOLD
    
    def test_risk_calculation_high(self):
        """Test HIGH risk calculation."""
        engine = RiskEngine()
        
        result = engine.calculate_risk(0.9, 0.8)
        
        assert result['risk_category'] == 'HIGH'
        assert result['overall_risk_score'] >= config.RISK_MEDIUM_THRESHOLD
    
    def test_risk_calculation_medium(self):
        """Test MEDIUM risk calculation."""
        engine = RiskEngine()
        
        result = engine.calculate_risk(0.5, 0.5)
        
        assert result['risk_category'] == 'MEDIUM'
    
    def test_get_recommendation(self):
        """Test recommendation generation."""
        engine = RiskEngine()
        
        for category in ['LOW', 'MEDIUM', 'HIGH']:
            rec = engine.get_recommendation(category)
            assert isinstance(rec, str)
            assert len(rec) > 0
    
    def test_should_alert(self):
        """Test alert triggering."""
        engine = RiskEngine()
        
        assert not engine.should_alert('LOW')
        assert engine.should_alert('MEDIUM')
        assert engine.should_alert('HIGH')


class TestIntegration:
    """Integration tests."""
    
    def test_audio_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        processor = AudioProcessor()
        
        # Create synthetic audio
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        
        # Preprocess
        processed = processor.preprocess(audio, sr)
        
        # Check output
        assert len(processed) > 0
        assert np.all(np.isfinite(processed))
        assert np.max(np.abs(processed)) <= 1.0 + 0.1  # Allow small tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
