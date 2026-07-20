import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID

from backend.database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_filename = Column(String, nullable=False)
    patient_name = Column(String, nullable=True)
    patient_id = Column(String, nullable=True)
    grade = Column(Integer, nullable=True)
    grade_name = Column(String, nullable=True)
    probabilities = Column(JSON, nullable=True)
    recommendation = Column(String, nullable=True)
    urgency = Column(String, nullable=True)
    heatmap_path = Column(String, nullable=True)
    pdf_path = Column(String, nullable=True)
    quality_passed = Column(Integer, nullable=False, default=1)
    quality_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
