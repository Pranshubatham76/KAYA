from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from uuid import UUID

from app.db.session import get_async_db
from app.core.services import event_service
from app.schemas import NearMissEventIngest, NearMissEventRead, NearMissEventReview, EventListResponse

router = APIRouter(tags=["events"])

@router.post("/events", response_model=NearMissEventRead)
async def ingest_event(
    payload: str = Form(...),
    audio: UploadFile = File(None),
    frame: UploadFile = File(None),
    db: AsyncSession = Depends(get_async_db)
):
    import json
    try:
        data = json.loads(payload)
        ingest_data = NearMissEventIngest(**data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    audio_bytes = await audio.read() if audio else None
    frame_bytes = await frame.read() if frame else None

    event = await event_service.ingest_event(db, ingest_data, audio_bytes, frame_bytes)
    return event

@router.get("/events/{site_id}", response_model=EventListResponse)
async def list_events(
    site_id: str,
    status: str = None,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_async_db)
):
    events, total = await event_service.list_events(db, site_id, status, page, page_size)
    return EventListResponse(items=events, total=total, page=page, page_size=page_size)

@router.put("/events/{event_id}/review", response_model=NearMissEventRead)
async def review_event(
    event_id: str,
    review: NearMissEventReview,
    reviewer_id: str = "admin",  # In real app, get from auth context
    db: AsyncSession = Depends(get_async_db)
):
    try:
        return await event_service.review_event(db, event_id, reviewer_id, review)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/events/{event_id}/detail", response_model=dict)
async def get_event_detail(
    event_id: str,
    db: AsyncSession = Depends(get_async_db)
):
    try:
        return await event_service.get_event_with_urls(db, event_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
