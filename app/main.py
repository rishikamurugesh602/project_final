"""
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.utils.config import config
from app.utils.logging import logger
from app.database.database import init_database
from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.api.live_call import router as live_call_router


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {config.APP_NAME} v{config.APP_VERSION}")
    init_database()
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")


# Create FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="Real-Time Deepfake Voice Interceptor & Audio Authenticator",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router)
app.include_router(ws_router)
app.include_router(live_call_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": config.APP_NAME,
        "version": config.APP_VERSION,
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG
    )
