"""
Automatic Speech Recognition (ASR) using OpenAI Whisper.
"""
import numpy as np
import whisper
from typing import Optional, Dict
from app.utils.config import config
from app.utils.logging import logger


class ASRService:
    """
    Speech-to-text conversion using Whisper.
    """
    
    def __init__(
        self,
        model_name: str = config.WHISPER_MODEL,
        device: str = config.MODEL_DEVICE
    ):
        """
        Initialize ASR service.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
            device: Device to use (cpu/cuda)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.is_loaded = False
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            self.is_loaded = True
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            self.is_loaded = False
    
    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = config.SAMPLE_RATE,
        language: str = "en"
    ) -> Dict:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio array
            sr: Sample rate
            language: Language code (e.g., 'en')
        
        Returns:
            Dictionary with transcription and metadata
        """
        if not self.is_loaded:
            return {
                "text": "",
                "error": "Whisper model not loaded",
                "success": False
            }
        
        try:
            # Ensure float32 dtype for PyTorch compatibility
            audio = np.asarray(audio, dtype=np.float32)

            # Whisper expects 16kHz mono audio
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Transcribe
            logger.info("Starting transcription")
            result = self.model.transcribe(
                audio,
                language=language,
                verbose=False
            )
            
            transcript = result.get("text", "").strip()
            logger.info(f"Transcription complete: {len(transcript)} characters")
            
            return {
                "text": transcript,
                "language": result.get("language", language),
                "segments": result.get("segments", []),
                "success": True,
                "model_name": self.model_name
            }
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
    
    def transcribe_from_file(
        self,
        audio_path: str,
        language: str = "en"
    ) -> Dict:
        """
        Transcribe audio from file.
        
        Args:
            audio_path: Path to audio file
            language: Language code
        
        Returns:
            Dictionary with transcription
        """
        try:
            logger.info(f"Transcribing file: {audio_path}")
            result = self.model.transcribe(
                audio_path,
                language=language,
                verbose=False
            )
            
            transcript = result.get("text", "").strip()
            logger.info(f"File transcription complete: {len(transcript)} characters")
            
            return {
                "text": transcript,
                "language": result.get("language", language),
                "segments": result.get("segments", []),
                "success": True,
                "model_name": self.model_name
            }
        except Exception as e:
            logger.error(f"Error transcribing file {audio_path}: {str(e)}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
