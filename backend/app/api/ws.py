"""
SentinelSite — WebSockets Router
Supervisor dashboard real-time alerts.
"""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)

router = APIRouter(tags=["ws"])

@router.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket, site_id: str):
    """
    Real-time alert pushes to the supervisor dashboard.
    (Mocked pub/sub loop for real-time events)
    """
    await websocket.accept()
    log.info(f"WebSocket connected for site {site_id}")
    try:
        while True:
            # In a real impl, this would listen to Redis pub/sub channel for `site_id`
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected for site {site_id}")
