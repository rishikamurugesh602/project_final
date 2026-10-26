"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class AnalysisRequest(BaseModel):
    """Request model for audio analysis."""
    audio_filename: Optional[str] = Field(None, description="Original audio filename")
    audio_duration: Optional[float] = Field(None, description="Duration in seconds")


class DeepfakeResult(BaseModel):
    """Deepfake detection result."""
    deepfake_probability: float = Field(..., ge=0, le=1)
    prediction_class: str
    is_deepfake: bool
    confidence_scores: dict
    model_loaded: bool


class ScamResult(BaseModel):
    """Scam detection result."""
    scam_probability: float = Field(..., ge=0, le=1)
    risk_category: str
    indicators: List[str]
    detection_method: str


class RiskResult(BaseModel):
    """Overall risk assessment result."""
    overall_risk_score: float = Field(..., ge=0, le=1)
    risk_category: str
    severity_level: int
    reasons: List[str]
    recommendation: str


class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    session_id: str
    timestamp: datetime
    
    # Deepfake analysis
    deepfake_result: DeepfakeResult
    
    # Speech recognition
    transcript: str
    asr_confidence: Optional[float] = None
    
    # Scam analysis
    scam_result: ScamResult
    
    # Risk assessment
    risk_result: RiskResult
    
    # Performance metrics
    preprocessing_time: Optional[float] = None
    deepfake_time: Optional[float] = None
    asr_time: Optional[float] = None
    scam_time: Optional[float] = None
    total_processing_time: Optional[float] = None


class AnalysisHistoryItem(BaseModel):
    """Single item in analysis history."""
    id: int
    session_id: str
    timestamp: datetime
    audio_filename: Optional[str]
    deepfake_probability: float
    scam_probability: float
    overall_risk_score: float
    risk_category: str
    transcript: Optional[str]


class AnalysisStatistics(BaseModel):
    """System statistics."""
    total_analyses: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    high_risk_percentage: float
    average_processing_time: float
    average_deepfake_probability: float
    average_scam_probability: float


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database_connected: bool
    models_loaded: dict


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
