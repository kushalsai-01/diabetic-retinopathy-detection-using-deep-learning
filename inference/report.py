from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from pathlib import Path
from dataclasses import dataclass

@dataclass
class PatientInfo:
    name: str
    dob: str
    patient_id: str
    exam_date: str

def draw_header(c: canvas.Canvas, patient: PatientInfo) -> None:
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2*cm, 27*cm, "Diabetic Retinopathy Screening Report")
    c.setLineWidth(1)
    c.line(2*cm, 26.5*cm, 19*cm, 26.5*cm)
    
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, 25.8*cm, f"Patient Name: {patient.name}")
    c.drawString(2*cm, 25.2*cm, f"Date of Birth: {patient.dob}")
    c.drawString(11*cm, 25.8*cm, f"Patient ID: {patient.patient_id}")
    c.drawString(11*cm, 25.2*cm, f"Exam Date: {patient.exam_date}")
    c.line(2*cm, 24.7*cm, 19*cm, 24.7*cm)

def draw_images(
    c: canvas.Canvas,
    fundus_path: str,
    heatmap_path: str | None,
    y_position: float,
) -> None:
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y_position, "Retinal Images Analysis")
    
    img_y = y_position - 6.5*cm
    c.drawImage(fundus_path, 2*cm, img_y, width=6*cm, height=6*cm)
    c.setFont("Helvetica", 8)
    c.drawString(2*cm, img_y - 0.4*cm, "Original Fundus Photograph")

    if heatmap_path and Path(heatmap_path).exists():
        c.drawImage(heatmap_path, 11*cm, img_y, width=6*cm, height=6*cm)
        c.drawString(11*cm, img_y - 0.4*cm, "Grad-CAM Heatmap Visualization")

def draw_result(
    c: canvas.Canvas,
    grade: int,
    grade_name: str,
    probabilities: list[float],
    recommendation: str,
    urgency: str,
    y_position: float,
) -> None:
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y_position, "Diagnostic Evaluation")
    
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y_position - 0.8*cm, f"Predicted Severity: {grade_name} (Grade {grade})")
    c.drawString(2*cm, y_position - 1.4*cm, f"Referral Urgency: {urgency.upper()}")
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, y_position - 2.2*cm, "Clinical Recommendation:")
    c.setFont("Helvetica", 10)
    
    # Simple word wrap for recommendation
    words = recommendation.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        if len(" ".join(current_line)) > 75:
            lines.append(" ".join(current_line[:-1]))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
        
    y_offset = y_position - 2.7*cm
    for line in lines:
        c.drawString(2*cm, y_offset, line)
        y_offset -= 0.5*cm

def draw_disclaimer(c: canvas.Canvas) -> None:
    c.line(2*cm, 3*cm, 19*cm, 3*cm)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(2*cm, 2.5*cm, "Disclaimer: This report is AI-assisted and should not replace professional medical advice.")
    c.drawString(2*cm, 2.1*cm, "Always consult a qualified ophthalmologist for clinical decisions.")

def generate_report(
    patient: PatientInfo,
    grade: int,
    grade_name: str,
    probabilities: list[float],
    recommendation: str,
    urgency: str,
    fundus_path: str,
    heatmap_path: str | None,
    output_path: str,
) -> str:
    c = canvas.Canvas(output_path, pagesize=A4)
    draw_header(c, patient)
    draw_images(c, fundus_path, heatmap_path, y_position=23.5*cm)
    draw_result(c, grade, grade_name, probabilities, recommendation, urgency, y_position=15.0*cm)
    draw_disclaimer(c)
    c.save()
    return str(Path(output_path).resolve())
