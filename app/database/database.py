"""
Database configuration and models.
Using SQLAlchemy with SQLite.
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.utils.config import config
from app.utils.logging import logger

# Create database URL
DATABASE_URL = config.DATABASE_URL

# Create database directory if needed
db_dir = os.path.dirname(DATABASE_URL.replace("sqlite:///./", ""))
if db_dir:
    os.makedirs(db_dir, exist_ok=True)

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class AnalysisResult(Base):
    """
    Stores analysis results for audio files.
    """
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Audio metadata
    audio_filename = Column(String(255), nullable=True)
    audio_duration = Column(Float, nullable=True)  # seconds
    sample_rate = Column(Integer, nullable=True)  # Hz
    
    # Deepfake detection
    deepfake_probability = Column(Float)
    deepfake_prediction = Column(String(50))  # "Genuine" or "AI-Generated"
    
    # ASR
    transcript = Column(Text)
    
    # Scam detection
    scam_probability = Column(Float)
    scam_indicators = Column(Text)  # JSON
    
    # Risk assessment
    overall_risk_score = Column(Float)
    risk_category = Column(String(20))  # LOW, MEDIUM, HIGH
    risk_reasons = Column(Text)  # JSON
    
    # Processing metrics
    preprocessing_time = Column(Float, nullable=True)  # seconds
    deepfake_time = Column(Float, nullable=True)
    asr_time = Column(Float, nullable=True)
    scam_time = Column(Float, nullable=True)
    total_processing_time = Column(Float, nullable=True)
    
    # Metadata
    model_versions = Column(Text, nullable=True)  # JSON
    
    def __repr__(self):
        return f"<AnalysisResult(session_id={self.session_id}, risk={self.risk_category})>"


class SystemMetrics(Base):
    """
    Stores system performance metrics.
    """
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Performance metrics
    cpu_usage = Column(Float, nullable=True)  # percent
    memory_usage = Column(Float, nullable=True)  # percent
    model_loaded = Column(Boolean, default=False)
    
    # Count metrics
    total_analyses = Column(Integer, default=0)
    high_risk_count = Column(Integer, default=0)
    
    # Timing metrics
    average_processing_time = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<SystemMetrics(timestamp={self.timestamp})>"


def init_database():
    """Initialize database - create all tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise


def get_db():
    """
    Database session dependency for FastAPI.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
