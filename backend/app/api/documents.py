"""
SentinelSite — Documents API Router
PDF upload and RAG indexing.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.services import document_service
from app.db.models import DocumentType

router = APIRouter(tags=["documents"])

@router.post("/documents")
async def upload_document(
    site_id: str = Form(...),
    doc_type: DocumentType = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload document to S3 and trigger Celery indexing task."""
    try:
        file_bytes = await file.read()
        doc = await document_service.upload_document(
            db=db,
            site_id=site_id,
            doc_type=doc_type,
            filename=file.filename,
            file_bytes=file_bytes
        )
        return {"doc_id": doc.id, "status": doc.status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{doc_id}/status")
async def get_document_status(doc_id: str, site_id: str, db: AsyncSession = Depends(get_db)):
    """Poll ingestion status."""
    try:
        status = await document_service.get_document_status(db, doc_id, site_id)
        return status
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
