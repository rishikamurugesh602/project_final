"""
Database CRUD operations.
"""
import json
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.database.database import AnalysisResult, SystemMetrics
from app.utils.logging import logger


class AnalysisResultCRUD:
    """CRUD operations for AnalysisResult."""
    
    @staticmethod
    def create(db: Session, result_data: dict) -> AnalysisResult:
        """
        Create a new analysis result.
        
        Args:
            db: Database session
            result_data: Dictionary with analysis data
        
        Returns:
            AnalysisResult instance
        """
        try:
            # Convert lists to JSON strings
            scam_indicators = result_data.get("scam_indicators", [])
            if isinstance(scam_indicators, list):
                scam_indicators = json.dumps(scam_indicators)
            
            risk_reasons = result_data.get("risk_reasons", [])
            if isinstance(risk_reasons, list):
                risk_reasons = json.dumps(risk_reasons)
            
            model_versions = result_data.get("model_versions", {})
            if isinstance(model_versions, dict):
                model_versions = json.dumps(model_versions)

            timestamp = result_data.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    timestamp = datetime.utcnow()
            elif isinstance(timestamp, (int, float)):
                try:
                    timestamp = datetime.fromtimestamp(timestamp)
                except (TypeError, ValueError, OSError):
                    timestamp = datetime.utcnow()
            elif timestamp is None:
                timestamp = datetime.utcnow()
            
            # Create record
            db_result = AnalysisResult(
                session_id=result_data.get("session_id"),
                timestamp=timestamp,
                audio_filename=result_data.get("audio_filename"),
                audio_duration=result_data.get("audio_duration"),
                sample_rate=result_data.get("sample_rate"),
                deepfake_probability=result_data.get("deepfake_probability", 0.0),
                deepfake_prediction=result_data.get("deepfake_prediction", "Unknown"),
                transcript=result_data.get("transcript", ""),
                scam_probability=result_data.get("scam_probability", 0.0),
                scam_indicators=scam_indicators,
                overall_risk_score=result_data.get("overall_risk_score", 0.0),
                risk_category=result_data.get("risk_category", "UNKNOWN"),
                risk_reasons=risk_reasons,
                preprocessing_time=result_data.get("preprocessing_time"),
                deepfake_time=result_data.get("deepfake_time"),
                asr_time=result_data.get("asr_time"),
                scam_time=result_data.get("scam_time"),
                total_processing_time=result_data.get("total_processing_time"),
                model_versions=model_versions
            )
            
            db.add(db_result)
            db.commit()
            db.refresh(db_result)
            logger.info(f"Created analysis result: {db_result.session_id}")
            return db_result
        except Exception as e:
            logger.error(f"Error creating analysis result: {str(e)}")
            db.rollback()
            raise
    
    @staticmethod
    def get_by_id(db: Session, result_id: int) -> AnalysisResult:
        """Get analysis result by ID."""
        return db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
    
    @staticmethod
    def get_by_session_id(db: Session, session_id: str) -> AnalysisResult:
        """Get analysis result by session ID."""
        return db.query(AnalysisResult).filter(
            AnalysisResult.session_id == session_id
        ).first()
    
    @staticmethod
    def get_recent(db: Session, limit: int = 10) -> list:
        """Get recent analysis results."""
        return db.query(AnalysisResult).order_by(
            AnalysisResult.timestamp.desc()
        ).limit(limit).all()
    
    @staticmethod
    def get_by_risk_category(db: Session, category: str, limit: int = 10) -> list:
        """Get analysis results by risk category."""
        return db.query(AnalysisResult).filter(
            AnalysisResult.risk_category == category
        ).order_by(AnalysisResult.timestamp.desc()).limit(limit).all()
    
    @staticmethod
    def clear_all(db: Session) -> int:
        """Delete all analysis history records from the database."""
        try:
            count = db.query(AnalysisResult).delete()
            db.commit()
            logger.info(f"Cleared {count} analysis history records")
            return count
        except Exception as e:
            logger.error(f"Error clearing analysis history: {str(e)}")
            db.rollback()
            raise

    @staticmethod
    def get_statistics(db: Session, hours: int = 24) -> dict:
        """
        Get statistics for the last N hours.
        
        Args:
            db: Database session
            hours: Number of hours to look back
        
        Returns:
            Dictionary with statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        results = db.query(AnalysisResult).filter(
            AnalysisResult.timestamp >= cutoff_time
        ).all()
        
        if not results:
            return {
                "total_analyses": 0,
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "low_risk_count": 0,
                "average_processing_time": 0.0,
                "average_deepfake_probability": 0.0,
                "average_scam_probability": 0.0
            }
        
        total = len(results)
        high_risk = sum(1 for r in results if r.risk_category == "HIGH")
        medium_risk = sum(1 for r in results if r.risk_category == "MEDIUM")
        low_risk = sum(1 for r in results if r.risk_category == "LOW")
        
        avg_processing_time = sum(
            r.total_processing_time for r in results if r.total_processing_time
        ) / total if total > 0 else 0.0
        
        avg_deepfake = sum(
            r.deepfake_probability for r in results
        ) / total if total > 0 else 0.0
        
        avg_scam = sum(
            r.scam_probability for r in results
        ) / total if total > 0 else 0.0
        
        return {
            "total_analyses": total,
            "high_risk_count": high_risk,
            "medium_risk_count": medium_risk,
            "low_risk_count": low_risk,
            "high_risk_percentage": (high_risk / total * 100) if total > 0 else 0.0,
            "average_processing_time": avg_processing_time,
            "average_deepfake_probability": avg_deepfake,
            "average_scam_probability": avg_scam
        }


class SystemMetricsCRUD:
    """CRUD operations for SystemMetrics."""
    
    @staticmethod
    def create(db: Session, metrics_data: dict) -> SystemMetrics:
        """Create a new system metrics record."""
        try:
            db_metrics = SystemMetrics(
                timestamp=metrics_data.get("timestamp", datetime.utcnow()),
                cpu_usage=metrics_data.get("cpu_usage"),
                memory_usage=metrics_data.get("memory_usage"),
                model_loaded=metrics_data.get("model_loaded", False),
                total_analyses=metrics_data.get("total_analyses", 0),
                high_risk_count=metrics_data.get("high_risk_count", 0),
                average_processing_time=metrics_data.get("average_processing_time")
            )
            
            db.add(db_metrics)
            db.commit()
            db.refresh(db_metrics)
            return db_metrics
        except Exception as e:
            logger.error(f"Error creating system metrics: {str(e)}")
            db.rollback()
            raise
    
    @staticmethod
    def get_recent(db: Session, limit: int = 1) -> list:
        """Get recent system metrics."""
        return db.query(SystemMetrics).order_by(
            SystemMetrics.timestamp.desc()
        ).limit(limit).all()
