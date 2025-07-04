"""
Cipher Desktop - Main FastAPI Application
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
    """Configure structured logging with better readability and file output"""
    
    # Set log levels to reduce noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    
    # Create logs directory
    from pathlib import Path
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Setup file handler for detailed logging
    from datetime import datetime
    log_filename = logs_dir / f"cipher_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure root logger to write to file
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # File handler with detailed format
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler with colors (existing behavior)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Configure structlog with file and console output
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    print(f"📝 Logging to file: {log_filename}")
    print(f"📝 Logging to console: INFO level and above")
    print(f"📝 File logging: DEBUG level and above (includes all debug messages)")

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Configure logging first
    configure_logging()
    
    logger.info("Starting Cipher Desktop")
    
    # Create necessary directories
    settings.TEMP_DIR.mkdir(exist_ok=True)
    settings.MODELS_DIR.mkdir(exist_ok=True)
    settings.RUNS_DIR.mkdir(exist_ok=True)
    
    # Initialize database
    await init_db()
    
    yield
    
    logger.info("Shutting down Cipher Desktop")


# Create FastAPI app
app = FastAPI(
    title="Cipher Desktop API",
    description="Backend API for Cipher Desktop Application",
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
    logger.info("🔗 WebSocket connection attempt", 
               session_id=session_id,
               client_host=websocket.client.host if websocket.client else "unknown",
               client_port=websocket.client.port if websocket.client else "unknown")
    
    try:
        # Attempt to accept the connection
        await websocket_manager.connect(websocket, session_id)
        logger.info("✅ WebSocket connection established successfully", 
                   session_id=session_id,
                   websocket_state=websocket.application_state if hasattr(websocket, 'application_state') else "unknown")
        
        # Don't send initial message immediately - let the connection stabilize first
        logger.info("🔗 WebSocket connection ready", session_id=session_id)
        
        # Keep connection alive and log periodic status
        message_count = 0
        while True:
            try:
                # Keep connection alive with timeout to prevent hanging
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message_count += 1
                logger.debug("📨 WebSocket message received", 
                           session_id=session_id, 
                           message_count=message_count,
                           message_preview=message[:100] if len(message) > 100 else message)
                
                # Echo back a keepalive response
                if message.strip() == "ping":
                    await websocket.send_text('{"type": "pong", "data": {}}')
                    logger.debug("🏓 Ping-pong keepalive", session_id=session_id)
                    
            except asyncio.TimeoutError:
                # Send keepalive message every 30 seconds
                try:
                    await websocket.send_text('{"type": "keepalive", "data": {"timestamp": "' + str(datetime.now().isoformat()) + '"}}')
                    logger.debug("💓 Keepalive sent", session_id=session_id, message_count=message_count)
                except Exception as keepalive_error:
                    logger.error("❌ Failed to send keepalive", 
                               session_id=session_id, 
                               error=str(keepalive_error))
                    break
                    
    except WebSocketDisconnect as disconnect_error:
        websocket_manager.disconnect(session_id)
        logger.info("🔌 WebSocket disconnected normally", 
                   session_id=session_id,
                   disconnect_code=getattr(disconnect_error, 'code', 'unknown'),
                   disconnect_reason=getattr(disconnect_error, 'reason', 'unknown'))
        
    except Exception as connection_error:
        logger.error("❌ WebSocket connection error", 
                    session_id=session_id,
                    error_type=type(connection_error).__name__,
                    error_message=str(connection_error))
        # Make sure to cleanup
        websocket_manager.disconnect(session_id)
        
        # Try to close the websocket gracefully
        try:
            await websocket.close(code=1011, reason="Server error")
        except Exception as close_error:
            logger.error("❌ Failed to close WebSocket gracefully", 
                        session_id=session_id, 
                        close_error=str(close_error))


# Move health check to API router
@app.get("/api/v1/health")
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