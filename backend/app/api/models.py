"""
SentinelSite — Models API Router
Device-facing OTA update check and accuracy reporting.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.model_service import model_service

router = APIRouter(tags=["models"])

@router.get("/models/latest")
async def get_latest_model(
    site_id: str,
    model_type: str,
    device_current_version: int = None,
    db: AsyncSession = Depends(get_db)
):
    """Check for OTA model update (cold start)."""
    try:
        result = await model_service.check_for_update(db, site_id, model_type, device_current_version)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/accuracy-report")
async def report_accuracy(
    site_id: str,
    model_type: str,
    version: int,
    reported_accuracy: float,
    db: AsyncSession = Depends(get_db)
):
    """Report post-deploy accuracy (triggers auto-rollback if drops)."""
    try:
        result = await model_service.report_post_deploy_accuracy(db, site_id, model_type, version, reported_accuracy)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
