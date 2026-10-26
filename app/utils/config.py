"""
Application configuration management.
Loads settings from environment variables and .env file.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""
    
    # Application
    APP_NAME: str = os.getenv("APP_NAME", "Real-Time Deepfake Voice Interceptor")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # ML Model
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "cpu")
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    
    # Thresholds and Weights
    DEEPFAKE_THRESHOLD: float = float(os.getenv("DEEPFAKE_THRESHOLD", "0.5"))
    DEEPFAKE_WEIGHT: float = float(os.getenv("DEEPFAKE_WEIGHT", "0.5"))
    SCAM_THRESHOLD: float = float(os.getenv("SCAM_THRESHOLD", "0.5"))
    SCAM_WEIGHT: float = float(os.getenv("SCAM_WEIGHT", "0.5"))
    
    # Risk Thresholds
    RISK_LOW_THRESHOLD: float = float(os.getenv("RISK_LOW_THRESHOLD", "0.39"))
    RISK_MEDIUM_THRESHOLD: float = float(os.getenv("RISK_MEDIUM_THRESHOLD", "0.69"))
    RISK_HIGH_THRESHOLD: float = float(os.getenv("RISK_HIGH_THRESHOLD", "1.0"))
    
    # Audio Processing
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
    CHUNK_DURATION: float = float(os.getenv("CHUNK_DURATION", "1.0"))
    SILENCE_THRESHOLD: float = float(os.getenv("SILENCE_THRESHOLD", "0.02"))
    ENABLE_NOISE_REDUCTION: bool = os.getenv("ENABLE_NOISE_REDUCTION", "true").lower() == "true"
    
    # Model Paths
    DEEPFAKE_MODEL_PATH: str = os.getenv("DEEPFAKE_MODEL_PATH", "models/checkpoints/deepfake_detector_best.pt")
    SCAM_MODEL_PATH: str = os.getenv("SCAM_MODEL_PATH", "models/checkpoints/scam_detector.pt")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
    ENABLE_AUDIO_STORAGE: bool = os.getenv("ENABLE_AUDIO_STORAGE", "false").lower() == "true"
    
    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    
    @classmethod
    def validate(cls) -> None:
        """Validate critical configuration settings."""
        if cls.SAMPLE_RATE not in [8000, 16000, 22050, 44100]:
            raise ValueError(f"Invalid SAMPLE_RATE: {cls.SAMPLE_RATE}")
        if not 0 <= cls.DEEPFAKE_WEIGHT <= 1:
            raise ValueError(f"DEEPFAKE_WEIGHT must be between 0 and 1, got {cls.DEEPFAKE_WEIGHT}")
        if not 0 <= cls.SCAM_WEIGHT <= 1:
            raise ValueError(f"SCAM_WEIGHT must be between 0 and 1, got {cls.SCAM_WEIGHT}")


# Create config instance
config = Config()
config.validate()
