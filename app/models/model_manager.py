"""
Model manager for lazy loading and caching ML models.
"""
from typing import Optional
from app.models.deepfake_detector import DeepfakeDetector
from app.models.scam_detector import ScamDetector
from app.services.asr_service import ASRService
from app.utils.config import config
from app.utils.logging import logger


class ModelManager:
    """Manages all ML models as singletons."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize model manager."""
        if self._initialized:
            return
        
        self._initialized = True
        self.deepfake_detector: Optional[DeepfakeDetector] = None
        self.scam_detector: Optional[ScamDetector] = None
        self.asr_service: Optional[ASRService] = None
        
        logger.info("ModelManager initialized")
    
    def get_deepfake_detector(self, force_reload: bool = False) -> DeepfakeDetector:
        """
        Get or create deepfake detector.
        
        Args:
            force_reload: Force reload even if already loaded
        
        Returns:
            DeepfakeDetector instance
        """
        if self.deepfake_detector is None or force_reload:
            logger.info("Loading deepfake detector")
            self.deepfake_detector = DeepfakeDetector(
                device=config.MODEL_DEVICE,
                model_path=config.DEEPFAKE_MODEL_PATH
            )
        return self.deepfake_detector
    
    def get_scam_detector(self, force_reload: bool = False) -> ScamDetector:
        """
        Get or create scam detector.
        
        Args:
            force_reload: Force reload even if already loaded
        
        Returns:
            ScamDetector instance
        """
        if self.scam_detector is None or force_reload:
            logger.info("Loading scam detector")
            self.scam_detector = ScamDetector(device=config.MODEL_DEVICE)
        return self.scam_detector
    
    def get_asr_service(self, force_reload: bool = False) -> ASRService:
        """
        Get or create ASR service.
        
        Args:
            force_reload: Force reload even if already loaded
        
        Returns:
            ASRService instance
        """
        if self.asr_service is None or force_reload:
            logger.info("Loading ASR service")
            self.asr_service = ASRService(
                model_name=config.WHISPER_MODEL,
                device=config.MODEL_DEVICE
            )
        return self.asr_service
    
    def get_models_status(self) -> dict:
        """
        Get status of all loaded models.
        
        Returns:
            Dictionary with model status
        """
        return {
            "deepfake_detector": {
                "loaded": self.deepfake_detector is not None,
                "model_ready": self.deepfake_detector.is_loaded if self.deepfake_detector else False
            },
            "scam_detector": {
                "loaded": self.scam_detector is not None,
                "model_ready": self.scam_detector.model_loaded if self.scam_detector else False
            },
            "asr_service": {
                "loaded": self.asr_service is not None,
                "model_ready": self.asr_service.is_loaded if self.asr_service else False
            }
        }
    
    def unload_all(self) -> None:
        """Unload all models to free memory."""
        logger.info("Unloading all models")
        self.deepfake_detector = None
        self.scam_detector = None
        self.asr_service = None
