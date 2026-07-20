"""
schemas.py
----------
Why it exists:
    Pydantic schemas validate request/response data and generate
    OpenAPI docs automatically. Keeping them separate from ORM models
    avoids coupling the database layer to the API layer.

What it does:
    - PredictRequest: Form fields for patient metadata (name, DOB, ID).
    - PredictionResponse: API response with all prediction fields.
    - HistoryItem: Lightweight row for the history list endpoint.
    - PaginatedHistory: Wraps history list with pagination metadata.

Imported by:
    - backend/routes/predict.py
    - backend/routes/history.py
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class PredictRequest(BaseModel):
    """
    Optional patient metadata sent alongside the image upload.

    All fields are optional — predictions can be made anonymously.
    """
    patient_name: Optional[str] = Field(None, example="Jane Doe")
    patient_dob: Optional[str] = Field(None, example="1985-03-14")
    patient_id: Optional[str] = Field(None, example="PT-00123")


class PredictionResponse(BaseModel):
    """
    Full API response for a single prediction request.

    Returned by POST /api/predict.
    """
    prediction_id: str
    grade: int
    grade_name: str
    probabilities: list[float]
    recommendation: str
    urgency: str
    heatmap_url: Optional[str]
    pdf_url: Optional[str]
    image_url: Optional[str] = None
    quality_passed: bool
    quality_reason: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class HistoryItem(BaseModel):
    """
    Lightweight row in the prediction history list.

    Returned by GET /api/history.
    """
    prediction_id: str
    image_filename: str
    patient_name: Optional[str]
    grade: Optional[int]
    grade_name: Optional[str]
    urgency: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedHistory(BaseModel):
    """
    Paginated response for the history endpoint.
    """
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int
