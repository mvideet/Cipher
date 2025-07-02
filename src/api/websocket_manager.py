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
        self._cleanup_task = None
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection"""
        # Start cleanup task if not already running
        if self._cleanup_task is None:
            self._start_cleanup_task()
            
        logger.info("🔗 Attempting to accept WebSocket connection", 
                   session_id=session_id,
                   current_connections=len(self.active_connections),
                   existing_connection=session_id in self.active_connections)
        
        # If there's already a connection for this session, close the old one
        if session_id in self.active_connections:
            logger.info("🔄 Replacing existing WebSocket connection", session_id=session_id)
            old_websocket = self.active_connections[session_id]
            
            # Force remove the old connection first to prevent blocking
            del self.active_connections[session_id]
            logger.info("🗑️ Removed old connection from registry", session_id=session_id)
            
            # Try to close the old connection, but don't wait if it fails
            try:
                # Use wait_for with timeout to prevent hanging on phantom connections
                import asyncio
                await asyncio.wait_for(
                    old_websocket.close(code=1000, reason="Replaced by new connection"),
                    timeout=1.0  # 1 second timeout
                )
                logger.info("✅ Old WebSocket connection closed", session_id=session_id)
            except asyncio.TimeoutError:
                logger.warning("⏰ Timeout closing old WebSocket connection (phantom connection)", 
                              session_id=session_id)
            except Exception as close_error:
                logger.warning("⚠️ Failed to close old WebSocket connection", 
                              session_id=session_id, 
                              error=str(close_error))
        
        try:
            await websocket.accept()
            logger.info("✅ WebSocket connection accepted successfully", session_id=session_id)
            
            self.active_connections[session_id] = websocket
            logger.info("🔗 WebSocket connection stored", 
                       session_id=session_id,
                       total_connections=len(self.active_connections),
                       connection_list=list(self.active_connections.keys()))
            
        except Exception as accept_error:
            logger.error("❌ Failed to accept WebSocket connection", 
                        session_id=session_id,
                        error_type=type(accept_error).__name__,
                        error_message=str(accept_error))
            raise accept_error
    
    def disconnect(self, session_id: str):
        """Remove WebSocket connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info("🔌 WebSocket connection removed", 
                       session_id=session_id,
                       remaining_connections=len(self.active_connections),
                       remaining_list=list(self.active_connections.keys()))
        else:
            logger.warning("⚠️ Attempted to disconnect non-existent WebSocket", 
                          session_id=session_id,
                          existing_connections=list(self.active_connections.keys()))
    
    async def send_personal_message(self, message: dict, session_id: str):
        """Send message to specific session"""
        logger.debug("📤 Attempting to send WebSocket message", 
                    session_id=session_id,
                    message_type=message.get('type', 'unknown'),
                    has_connection=session_id in self.active_connections)
        
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            
            # Check if WebSocket is still open before sending
            try:
                # First check the client state
                if hasattr(websocket, 'client_state') and websocket.client_state.value != 1:  # 1 = CONNECTED
                    logger.warning("🔌 WebSocket client_state not connected, removing phantom connection", 
                                 session_id=session_id,
                                 client_state=websocket.client_state.value if hasattr(websocket, 'client_state') else 'unknown')
                    self.disconnect(session_id)
                    return
                    
                # Try to send the message
                message_json = json.dumps(message)
                await websocket.send_text(message_json)
                logger.debug("✅ WebSocket message sent successfully", 
                           session_id=session_id,
                           message_type=message.get('type', 'unknown'),
                           message_size=len(message_json))
            except Exception as e:
                logger.error("❌ Failed to send WebSocket message", 
                           session_id=session_id, 
                           error_type=type(e).__name__,
                           error_message=str(e),
                           message_type=message.get('type', 'unknown'))
                # Remove phantom connection
                self.disconnect(session_id)
        else:
            logger.warning("⚠️ Attempted to send message to non-existent connection", 
                          session_id=session_id,
                          message_type=message.get('type', 'unknown'),
                          active_connections=list(self.active_connections.keys()))
    
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
    
    # PyTorch broadcasting methods removed
    
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


    def _start_cleanup_task(self):
        """Start periodic cleanup of phantom connections"""
        import asyncio
        
        async def cleanup_phantom_connections():
            while True:
                try:
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                    # Check all connections
                    phantom_sessions = []
                    for session_id, websocket in self.active_connections.items():
                        try:
                            # Check if connection is still alive
                            if hasattr(websocket, 'client_state') and websocket.client_state.value != 1:
                                phantom_sessions.append(session_id)
                                logger.info("🧹 Found phantom connection", session_id=session_id)
                        except Exception as e:
                            phantom_sessions.append(session_id)
                            logger.warning("🧹 Error checking connection, marking as phantom", 
                                         session_id=session_id, error=str(e))
                    
                    # Clean up phantom connections
                    for session_id in phantom_sessions:
                        self.disconnect(session_id)
                        logger.info("🧹 Cleaned up phantom connection", session_id=session_id)
                        
                    if phantom_sessions:
                        logger.info("🧹 Cleanup complete", 
                                   cleaned_connections=len(phantom_sessions),
                                   remaining_connections=len(self.active_connections))
                                   
                except Exception as e:
                    logger.error("🧹 Cleanup task error", error=str(e))
        
        # Create and start the cleanup task
        self._cleanup_task = asyncio.create_task(cleanup_phantom_connections())
        logger.info("🧹 Started phantom connection cleanup task")


# Global connection manager instance
websocket_manager = ConnectionManager() 