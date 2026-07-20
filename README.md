# RetinaGuard AI — Diabetic Retinopathy Detection System

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://reactjs.org)

An end-to-end clinical AI screening system for Diabetic Retinopathy from retinal fundus photographs. Built with EfficientNetV2-S, GradCAM explainability, and a full-stack FastAPI + React interface.

---

## Model Performance

| Metric | Score |
|---|---|
| Validation QWK | **0.8181** |
| Test QWK | **0.8081** |
| Model | EfficientNetV2-S + GeM Pooling |
| Dataset | APTOS 2019 Blindness Detection |

---

## Architecture

```
Fundus Image
    -> Image Quality Check
    -> Black Border Removal + Circular Crop
    -> Ben Graham Preprocessing + CLAHE
    -> Albumentations (Resize 320px, Normalize)
    -> EfficientNetV2-S Backbone
    -> GeM Pooling -> Dropout -> MLP Classifier
    -> Softmax -> Ordinal Grade Prediction
    -> GradCAM Heatmap Overlay
    -> Clinical Recommendation
    -> PDF Report + API JSON Response
```

---

## Setup

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

## Usage

```bash
# Preprocess dataset
python preprocessing/run_preprocessing.py --split all

# Train model (auto-resumes from checkpoint)
python training/train.py

# Evaluate model with TTA
python training/evaluate.py

# Generate GradCAM visualizations
python scripts/generate_gradcam.py

# Start backend API
uvicorn backend.main:app --reload --port 8000

# Start frontend
cd frontend && npm run dev
```

## Docker

```bash
docker compose up --build
```

Frontend: http://localhost:3000
API Docs: http://localhost:8000/docs

---

## DR Severity Classes

| Grade | Label | Action |
|---|---|---|
| 0 | No DR | Routine annual screening |
| 1 | Mild | Follow-up in 12 months |
| 2 | Moderate | Refer within 3 months |
| 3 | Severe | Refer within 1 month |
| 4 | Proliferative DR | Immediate referral |

---

## Dataset

APTOS 2019 Blindness Detection (Kaggle): Train: 2930 / Val: 366 / Test: 366

## API Endpoints

- POST /api/predict — Predict DR grade from fundus image
- GET /api/history — Paginated prediction history
- GET /api/history/{id} — Get single prediction
- DELETE /api/history/{id} — Delete prediction record

---

## Results

Evaluation outputs saved to reports/evaluation/:
- confusion_matrix.png, roc_curves.png, pr_curves.png
- metrics_summary.json, classification_report.json

GradCAM visualizations saved to reports/gradcam/
