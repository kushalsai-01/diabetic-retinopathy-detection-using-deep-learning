from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pathlib import Path

from backend.database import get_db
from backend.models import Prediction
from backend.schemas import PredictionResponse, PaginatedHistory, HistoryItem

router = APIRouter()

@router.get("/history", response_model=PaginatedHistory)
async def get_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PaginatedHistory:
    offset = (page - 1) * page_size
    
    # Get total count
    total_query = select(func.count()).select_from(Prediction)
    total = await db.scalar(total_query) or 0
    
    # Get items
    items_query = select(Prediction).order_by(Prediction.created_at.desc()).offset(offset).limit(page_size)
    result = await db.execute(items_query)
    predictions = result.scalars().all()
    
    items = []
    for p in predictions:
        items.append(HistoryItem(
            prediction_id=str(p.id),
            image_filename=p.image_filename,
            patient_name=p.patient_name,
            grade=p.grade,
            grade_name=p.grade_name,
            urgency=p.urgency,
            created_at=p.created_at
        ))
        
    return PaginatedHistory(items=items, total=total, page=page, page_size=page_size)

@router.get("/history/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: str,
    db: AsyncSession = Depends(get_db),
) -> PredictionResponse:
    try:
        p_id = Path(prediction_id).name # sanitise / check
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid prediction ID format")

    result = await db.execute(select(Prediction).where(Prediction.id == prediction_id))
    p = result.scalar_one_or_none()
    
    if p is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
        
    heatmap_url = f"/heatmaps/{Path(p.heatmap_path).name}" if p.heatmap_path else None
    pdf_url = f"/reports/{Path(p.pdf_path).name}" if p.pdf_path else None

    image_url = None
    for ext in [".png", ".jpg", ".jpeg"]:
        if Path(f"backend/uploads/{p.id}{ext}").exists():
            image_url = f"/uploads/{p.id}{ext}"
            break

    return PredictionResponse(
        prediction_id=str(p.id),
        grade=p.grade if p.grade is not None else -1,
        grade_name=p.grade_name or "Invalid Image",
        probabilities=p.probabilities or [],
        recommendation=p.recommendation or "",
        urgency=p.urgency or "routine",
        heatmap_url=heatmap_url,
        pdf_url=pdf_url,
        image_url=image_url,
        quality_passed=bool(p.quality_passed),
        quality_reason=p.quality_reason,
        created_at=p.created_at
    )

@router.delete("/history/{prediction_id}", status_code=204)
async def delete_prediction(
    prediction_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(select(Prediction).where(Prediction.id == prediction_id))
    p = result.scalar_one_or_none()
    
    if p is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
        
    await db.delete(p)
    await db.commit()
