from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.db.session import get_async_db
from app.schemas import TrainingJobTrigger

router = APIRouter(tags=["training"])

@router.post("/admin/train")
async def trigger_training(
    request: TrainingJobTrigger,
    db: AsyncSession = Depends(get_async_db)
):
    from app.training.scheduler import _enqueue_training_job
    try:
        job_id = _enqueue_training_job(
            db=db,
            site_id=str(request.site_id),
            model_type=request.model_type.value,
            trigger="manual",
            replay_strategy=request.replay_strategy
        )
        await db.commit()
        return {"status": "queued", "job_id": job_id}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
