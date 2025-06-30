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

    async def broadcast_training_status(self, session_id: str, status_data: dict):
        """Broadcast training status update to session"""
        message = {
            "type": "training_status",
            "data": status_data
        }
        await self.send_personal_message(message, session_id)

    async def broadcast_to_session(self, session_id: str, event_data: dict):
        """Broadcast generic event data to session"""
        await self.send_personal_message(event_data, session_id)
    
    async def broadcast_pytorch_training_start(self, session_id: str, pytorch_data: dict):
        """Broadcast PyTorch training start to session"""
        message = {
            "type": "pytorch_training_start", 
            "data": pytorch_data
        }
        await self.send_personal_message(message, session_id)
    
    async def broadcast_pytorch_training_complete(self, session_id: str, pytorch_data: dict):
        """Broadcast PyTorch training completion to session"""
        message = {
            "type": "pytorch_training_complete",
            "data": pytorch_data
        }
        await self.send_personal_message(message, session_id)
    
    async def broadcast_epoch_update(self, session_id: str, epoch_data: dict):
        """Broadcast epoch training update to session"""
        message = {
            "type": "epoch_update",
            "data": epoch_data
        }
        await self.send_personal_message(message, session_id)
    
    async def broadcast_model_improvement(self, session_id: str, improvement_data: dict):
        """Broadcast model improvement notification to session"""
        message = {
            "type": "model_improvement",
            "data": improvement_data
        }
        await self.send_personal_message(message, session_id)
    
    async def broadcast_early_stopping(self, session_id: str, stopping_data: dict):
        """Broadcast early stopping notification to session"""
        message = {
            "type": "early_stopping",
            "data": stopping_data
        }
        await self.send_personal_message(message, session_id)


# Global connection manager instance
websocket_manager = ConnectionManager() 