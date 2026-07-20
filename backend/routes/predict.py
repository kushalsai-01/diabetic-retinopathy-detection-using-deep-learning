import uuid
import shutil
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models import Prediction
from backend.schemas import PredictionResponse
from inference.predictor import predict
from inference.recommendations import get_urgency
from inference.report import generate_report, PatientInfo

router = APIRouter()

UPLOAD_DIR = Path("backend/uploads")
REPORT_DIR = Path("backend/reports")

@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = File(...),
    patient_name: str = Form(None),
    patient_dob: str = Form(None),
    patient_id: str = Form(None),
    db: AsyncSession = Depends(get_db),
) -> PredictionResponse:
    # 1. Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Only PNG and JPEG supported.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Save original uploaded file
    pred_uuid = uuid.uuid4()
    unique_filename = f"{pred_uuid}{ext}"
    file_path = UPLOAD_DIR / unique_filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. Predict pipeline
    result = predict(str(file_path))
    
    # 4. Generate report
    pdf_path = None
    if result.quality_passed:
        patient_info = PatientInfo(
            name=patient_name or "Anonymous",
            dob=patient_dob or "N/A",
            patient_id=patient_id or "N/A",
            exam_date=datetime.utcnow().strftime("%Y-%m-%d")
        )
        pdf_filename = f"{pred_uuid}.pdf"
        pdf_path = REPORT_DIR / pdf_filename
        
        generate_report(
            patient=patient_info,
            grade=result.grade,
            grade_name=result.grade_name,
            probabilities=result.probabilities,
            recommendation=result.recommendation,
            urgency=get_urgency(result.grade),
            fundus_path=str(file_path),
            heatmap_path=result.heatmap_path,
            output_path=str(pdf_path)
        )

    # 5. Database Save
    db_record = Prediction(
        id=pred_uuid,
        image_filename=file.filename,
        patient_name=patient_name,
        patient_id=patient_id,
        grade=result.grade if result.quality_passed else None,
        grade_name=result.grade_name if result.quality_passed else None,
        probabilities=result.probabilities if result.quality_passed else None,
        recommendation=result.recommendation,
        urgency=get_urgency(result.grade) if result.quality_passed else None,
        heatmap_path=result.heatmap_path,
        pdf_path=str(pdf_path) if pdf_path else None,
        quality_passed=1 if result.quality_passed else 0,
        quality_reason=result.quality_reason
    )
    
    db.add(db_record)
    await db.commit()

    heatmap_url = f"/heatmaps/{Path(result.heatmap_path).name}" if result.heatmap_path else None
    pdf_url = f"/reports/{pdf_path.name}" if pdf_path else None
    image_url = f"/uploads/{unique_filename}"

    return PredictionResponse(
        prediction_id=str(pred_uuid),
        grade=result.grade,
        grade_name=result.grade_name,
        probabilities=result.probabilities,
        recommendation=result.recommendation,
        urgency=get_urgency(result.grade),
        heatmap_url=heatmap_url,
        pdf_url=pdf_url,
        image_url=image_url,
        quality_passed=result.quality_passed,
        quality_reason=result.quality_reason,
        created_at=datetime.utcnow()
    )
