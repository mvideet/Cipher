"""
AutoML Desktop - Main FastAPI Application
Entry point for the backend API server
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import orchestrator
from .api.websocket_manager import websocket_manager
from .core.config import settings
from .database import init_db


# Configure better logging for development
def configure_logging():
    """Configure structured logging with better readability"""
    
    # Set log levels to reduce noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    
    # Configure structlog with readable format
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True)  # Pretty colors for console
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Configure logging first
    configure_logging()
    
    logger.info("Starting AutoML Desktop")
    
    # Create necessary directories
    settings.TEMP_DIR.mkdir(exist_ok=True)
    settings.MODELS_DIR.mkdir(exist_ok=True)
    settings.RUNS_DIR.mkdir(exist_ok=True)
    
    # Initialize database
    await init_db()
    
    yield
    
    logger.info("Shutting down AutoML Desktop")


# Create FastAPI app
app = FastAPI(
    title="AutoML Desktop API",
    description="Backend API for AutoML Desktop Application",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware for Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(orchestrator.router, prefix="/api/v1")

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time training progress updates"""
    await websocket_manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(session_id)
        logger.info("WebSocket disconnected", session_id=session_id)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "0.1.0"}


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_config=None  # Use structlog instead
    ) 