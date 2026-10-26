"""
FastAPI REST endpoints for audio analysis.
"""
import os
import uuid
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import numpy as np
import librosa

from app.schemas.analysis import (
    AnalysisResponse, AnalysisHistoryItem, AnalysisStatistics, 
    HealthCheckResponse, ErrorResponse
)
from app.services.deepfake_service import AnalysisOrchestrator
from app.services.audio_processor import AudioProcessor
from app.database.database import get_db
from app.database.crud import AnalysisResultCRUD
from app.models.model_manager import ModelManager
from app.utils.config import config
from app.utils.logging import logger

router = APIRouter(prefix="/api/v1", tags=["analysis"])

# Initialize services
orchestrator = AnalysisOrchestrator()
audio_processor = AudioProcessor()
model_manager = ModelManager()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint.
    
    Returns:
        Health status and model information
    """
    try:
        # Check database connection
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}", exc_info=True)
        db_connected = False
    
    models_status = model_manager.get_models_status()
    
    return HealthCheckResponse(
        status="healthy" if db_connected else "degraded",
        version=config.APP_VERSION,
        database_connected=db_connected,
        models_loaded=models_status
    )


@router.post("/analyze/audio", response_model=dict)
async def analyze_audio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Analyze uploaded audio file.
    
    Args:
        file: Audio file (WAV, MP3, etc.)
        db: Database session
    
    Returns:
        Complete analysis results
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Validate file extension
    valid_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Supported: {', '.join(valid_extensions)}"
        )
    
    try:
        logger.info(f"Analyzing uploaded audio: {file.filename}")
        
        # Read audio data
        audio_data = await file.read()
        
        # Save temporarily
        temp_path = f"temp_{uuid.uuid4()}.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_data)
        
        # Load audio
        audio, sr = librosa.load(temp_path, sr=None, mono=True)
        
        # Run analysis
        result = orchestrator.analyze(audio, sr, audio_filename=file.filename)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Store in database
        db_result = AnalysisResultCRUD.create(db, result)

        if isinstance(result.get("timestamp"), str):
            result["timestamp"] = result["timestamp"]
        elif result.get("timestamp") is not None:
            result["timestamp"] = result["timestamp"].isoformat()
        
        return JSONResponse(content=result, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/transcript")
async def analyze_transcript(
    transcript_data: dict,
    db: Session = Depends(get_db)
):
    """
    Analyze just the transcript for scam intent.
    
    Args:
        transcript_data: Dict with 'transcript' key
        db: Database session
    
    Returns:
        Scam analysis results
    """
    transcript = transcript_data.get("transcript", "").strip()
    
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is required")
    
    try:
        from app.models.scam_detector import ScamDetector
        scam_detector = model_manager.get_scam_detector()
        result = scam_detector.predict(transcript)
        
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        logger.error(f"Error analyzing transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{session_id}", response_model=dict)
async def get_result(session_id: str, db: Session = Depends(get_db)):
    """
    Retrieve analysis result by session ID.
    
    Args:
        session_id: Session ID from analysis
        db: Database session
    
    Returns:
        Analysis result
    """
    try:
        result = AnalysisResultCRUD.get_by_session_id(db, session_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        
        return {
            "id": result.id,
            "session_id": result.session_id,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "audio_filename": result.audio_filename,
            "deepfake_probability": result.deepfake_probability,
            "deepfake_prediction": result.deepfake_prediction,
            "transcript": result.transcript,
            "scam_probability": result.scam_probability,
            "scam_indicators": result.scam_indicators,
            "overall_risk_score": result.overall_risk_score,
            "risk_category": result.risk_category,
            "risk_reasons": result.risk_reasons,
            "total_processing_time": result.total_processing_time
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving result: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results", response_model=list)
async def list_results(
    limit: int = 10,
    risk_category: str = None,
    db: Session = Depends(get_db)
):
    """
    List recent analysis results.
    
    Args:
        limit: Number of results to return
        risk_category: Filter by risk category (LOW, MEDIUM, HIGH)
        db: Database session
    
    Returns:
        List of analysis results
    """
    try:
        if risk_category:
            results = AnalysisResultCRUD.get_by_risk_category(db, risk_category, limit)
        else:
            results = AnalysisResultCRUD.get_recent(db, limit)
        
        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "audio_filename": r.audio_filename,
                "deepfake_probability": r.deepfake_probability,
                "scam_probability": r.scam_probability,
                "overall_risk_score": r.overall_risk_score,
                "risk_category": r.risk_category,
                "transcript": r.transcript[:100] if r.transcript else None  # Truncate
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"Error listing results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=AnalysisStatistics)
async def get_statistics(
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """
    Get system statistics.
    
    Args:
        hours: Number of hours to look back
        db: Database session
    
    Returns:
        System statistics
    """
    try:
        stats = AnalysisResultCRUD.get_statistics(db, hours)
        return AnalysisStatistics(**stats)
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
