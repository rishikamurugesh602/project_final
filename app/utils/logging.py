"""
Logging configuration for the application.
"""
import logging
import os
from app.utils.config import config


def setup_logging() -> logging.Logger:
    """
    Set up application logging.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("deepfake_interceptor")
    logger.setLevel(config.LOG_LEVEL)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.LOG_LEVEL)
    
    # Create file handler
    log_file = os.path.join(config.LOG_DIR, "app.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(config.LOG_LEVEL)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Initialize logger
logger = setup_logging()
