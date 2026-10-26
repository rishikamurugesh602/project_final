"""
Analysis orchestration service.
Coordinates the full analysis pipeline from audio input to risk assessment.
"""
import time
import uuid
from datetime import datetime
from typing import Dict, Tuple, Optional
import numpy as np
from app.services.audio_processor import AudioProcessor
from app.services.asr_service import ASRService
from app.services.risk_engine import RiskEngine
from app.models.deepfake_detector import DeepfakeDetector
from app.models.scam_detector import ScamDetector
from app.models.model_manager import ModelManager
from app.utils.config import config
from app.utils.logging import logger


class AnalysisOrchestrator:
    """
    Orchestrates the complete analysis pipeline.
    
    Pipeline:
    1. Audio Loading & Preprocessing
    2. Deepfake Detection
    3. Speech Recognition (ASR)
    4. Scam Intent Analysis
    5. Risk Assessment
    6. Alert Generation
    """
    
    def __init__(self):
        """Initialize analysis orchestrator."""
        self.audio_processor = AudioProcessor()
        self.risk_engine = RiskEngine()
        self.model_manager = ModelManager()
        logger.info("AnalysisOrchestrator initialized")
    
    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        audio_filename: Optional[str] = None
    ) -> Dict:
        """
        Complete audio analysis pipeline.
        
        Args:
            audio: Audio array
            sr: Sample rate
            audio_filename: Optional filename for logging
        
        Returns:
            Dictionary with complete analysis results
        """
        session_id = str(uuid.uuid4())
        logger.info(f"Starting analysis pipeline - session: {session_id}")
        
        timing = {}
        
        try:
            # Audio preprocessing
            t0 = time.time()
            audio_preprocessed = self.audio_processor.preprocess(audio, sr)
            timing['preprocessing'] = time.time() - t0
            logger.info(f"Audio preprocessing completed in {timing['preprocessing']:.2f}s")
            
            # Extract features for deepfake detection
            mel_spec = self.audio_processor.extract_mel_spectrogram(audio_preprocessed)
            
            # Deepfake detection
            t0 = time.time()
            deepfake_detector = self.model_manager.get_deepfake_detector()
            deepfake_result = deepfake_detector.predict(mel_spec)
            deepfake_probability = deepfake_result.get("deepfake_probability", 0.5)
            timing['deepfake'] = time.time() - t0
            logger.info(f"Deepfake detection completed in {timing['deepfake']:.2f}s")
            
            # ASR (Speech Recognition)
            t0 = time.time()
            asr_service = self.model_manager.get_asr_service()
            asr_result = asr_service.transcribe(audio_preprocessed, sr=config.SAMPLE_RATE)
            transcript = asr_result.get("text", "")
            timing['asr'] = time.time() - t0
            logger.info(f"ASR completed in {timing['asr']:.2f}s")
            
            # Scam Intent Analysis
            t0 = time.time()
            scam_detector = self.model_manager.get_scam_detector()
            scam_result = scam_detector.predict(transcript)
            scam_probability = scam_result.get("scam_probability", 0.0)
            timing['scam'] = time.time() - t0
            logger.info(f"Scam analysis completed in {timing['scam']:.2f}s")
            
            # Risk Assessment
            risk_result = self.risk_engine.calculate_risk(
                deepfake_probability,
                scam_probability
            )
            overall_risk_score = risk_result.get("overall_risk_score", 0.0)
            risk_category = risk_result.get("risk_category", "UNKNOWN")
            
            # Calculate total time
            total_time = sum(timing.values())
            
            # Build complete response
            analysis_response = {
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                
                # Deepfake results
                "deepfake_probability": deepfake_probability,
                "deepfake_prediction": deepfake_result.get("prediction_class", "Unknown"),
                "deepfake_model_loaded": deepfake_result.get("model_loaded", False),
                
                # ASR results
                "transcript": transcript,
                "asr_success": asr_result.get("success", False),
                
                # Scam results
                "scam_probability": scam_probability,
                "scam_indicators": scam_result.get("indicators", []),
                "scam_detection_method": scam_result.get("detection_method", "unknown"),
                
                # Risk results
                "overall_risk_score": overall_risk_score,
                "risk_category": risk_category,
                "risk_reasons": risk_result.get("reasons", []),
                "recommendation": self.risk_engine.get_recommendation(risk_category),
                "should_alert": self.risk_engine.should_alert(risk_category),
                
                # Metadata
                "audio_filename": audio_filename,
                "audio_duration": len(audio_preprocessed) / config.SAMPLE_RATE,
                "sample_rate": config.SAMPLE_RATE,
                
                # Timing
                "preprocessing_time": timing['preprocessing'],
                "deepfake_time": timing['deepfake'],
                "asr_time": timing['asr'],
                "scam_time": timing['scam'],
                "total_processing_time": total_time
            }
            
            logger.info(
                f"Analysis complete - Risk: {risk_category}, "
                f"Deepfake: {deepfake_probability:.2f}, "
                f"Scam: {scam_probability:.2f}, "
                f"Total time: {total_time:.2f}s"
            )
            
            return analysis_response
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}", exc_info=True)
            return {
                "session_id": session_id,
                "error": str(e),
                "success": False
            }
    
    def analyze_from_file(self, audio_path: str) -> Dict:
        """
        Analyze audio from file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Analysis results
        """
        try:
            logger.info(f"Loading audio file: {audio_path}")
            audio, sr = self.audio_processor.load_audio(audio_path)
            
            import os
            filename = os.path.basename(audio_path)
            
            return self.analyze(audio, sr, audio_filename=filename)
        except Exception as e:
            logger.error(f"Error analyzing file {audio_path}: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
