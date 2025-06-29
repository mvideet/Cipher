"""
WebSocket connection manager for real-time updates
"""

import json
from typing import Dict
from fastapi import WebSocket
import structlog

logger = structlog.get_logger()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info("WebSocket connected", session_id=session_id)
    
    def disconnect(self, session_id: str):
        """Remove WebSocket connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info("WebSocket disconnected", session_id=session_id)
    
    async def send_personal_message(self, message: dict, session_id: str):
        """Send message to specific session"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(
                    json.dumps(message)
                )
            except Exception as e:
                logger.error("Failed to send WebSocket message", 
                           session_id=session_id, error=str(e))
                self.disconnect(session_id)
    
    async def broadcast_trial_update(self, session_id: str, event_data: dict):
        """Broadcast trial update to session"""
        message = {
            "type": "trial_update",
            "data": event_data
        }
        await self.send_personal_message(message, session_id)
    
    async def broadcast_family_completion(self, session_id: str, event_data: dict):
        """Broadcast family completion to session"""
        message = {
            "type": "family_complete",
            "data": event_data
        }
        await self.send_personal_message(message, session_id)
    
    async def broadcast_training_complete(self, session_id: str, result_data: dict):
        """Broadcast training completion to session"""
        message = {
            "type": "training_complete",
            "data": result_data
        }
        await self.send_personal_message(message, session_id)
    
    async def broadcast_error(self, session_id: str, error_msg: str):
        """Broadcast error message to session"""
        message = {
            "type": "error",
            "data": {"message": error_msg}
        }
        await self.send_personal_message(message, session_id)


# Global connection manager instance
websocket_manager = ConnectionManager() 