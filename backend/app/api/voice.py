from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_db
from app.core.services import voice_service
from app.schemas import VoiceQueryRequest, VoiceQueryResponse

router = APIRouter(tags=["voice"])

@router.post("/voice/query", response_model=VoiceQueryResponse)
async def voice_query(
    request: VoiceQueryRequest,
    db: AsyncSession = Depends(get_async_db)
):
    try:
        return await voice_service.handle_query(db, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
