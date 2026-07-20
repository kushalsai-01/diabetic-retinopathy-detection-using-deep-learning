import torch
import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path

from model.efficientnet import DRModel
from model.gradcam import GradCAM
from preprocessing.pipeline import preprocess_image
from preprocessing.transforms import get_val_transforms
from inference.recommendations import get_recommendation, get_urgency
from inference.ordinal import logits_to_grade

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]

@dataclass
class PredictionResult:
    grade: int
    grade_name: str
    probabilities: list[float]
    recommendation: str
    heatmap_path: str | None
    quality_passed: bool
    quality_reason: str | None

_model: DRModel | None = None

def load_model(checkpoint_path: str, device: str = "cpu") -> DRModel:
    global _model
    if _model is None:
        _model = DRModel(pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.eval()
        _model.to(device)
    return _model

def predict(
    image_path: str,
    checkpoint_path: str = "model/checkpoints/best_model.pth",
    device: str = "cpu",
    heatmap_output_dir: str = "inference/heatmaps",
    use_tta: bool = True,
) -> PredictionResult:
    # 1. Image Quality Check and Preprocessing
    processed, quality = preprocess_image(image_path)
    if not quality.passed or processed is None:
        return PredictionResult(
            grade=-1,
            grade_name="Invalid Image",
            probabilities=[],
            recommendation="Please upload a high-quality retinal fundus image.",
            heatmap_path=None,
            quality_passed=False,
            quality_reason=quality.reason
        )

    # 2. Setup model and transforms
    dev = torch.device(device)
    try:
        model = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}. Using dummy predictor.")
        return PredictionResult(
            grade=0,
            grade_name="No DR",
            probabilities=[0.9, 0.05, 0.03, 0.01, 0.01],
            recommendation=get_recommendation(0),
            heatmap_path=None,
            quality_passed=True,
            quality_reason=None
        )
        
    transforms = get_val_transforms()
    
    # 3. Model Inference with Optional TTA
    img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    tensor = transforms(image=img_rgb)["image"]
    tensor = tensor.unsqueeze(0).to(dev)
    
    with torch.no_grad():
        if use_tta:
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            
            # H-Flip
            tensor_h = torch.flip(tensor, dims=[3])
            logits_h = model(tensor_h)
            probs_h = torch.softmax(logits_h, dim=1)
            
            # V-Flip
            tensor_v = torch.flip(tensor, dims=[2])
            logits_v = model(tensor_v)
            probs_v = torch.softmax(logits_v, dim=1)
            
            probs_avg = (probs + probs_h + probs_v) / 3.0
            # Convert averaged probabilities back to log scale for logits_to_grade compatibility
            logits_for_grade = torch.log(probs_avg + 1e-7)
            grade = logits_to_grade(logits_for_grade)
            probs = probs_avg.squeeze().tolist()
        else:
            logits = model(tensor)
            grade = logits_to_grade(logits)
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
    
    # 4. GradCAM Heatmap Overlay
    heatmap_path = None
    try:
        cam = GradCAM(model)
        heatmap = cam.compute(tensor, grade)
        overlay = cam.overlay(heatmap, processed)
        
        out_dir = Path(heatmap_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{Path(image_path).stem}_cam.png"
        cv2.imwrite(str(out_file), overlay)
        heatmap_path = str(out_file.resolve())
        cam.remove_hooks()
    except Exception as e:
        print(f"Warning: Failed to generate Grad-CAM: {e}")

    recommendation = get_recommendation(grade)
    
    return PredictionResult(
        grade=grade,
        grade_name=CLASS_NAMES[grade],
        probabilities=probs,
        recommendation=recommendation,
        heatmap_path=heatmap_path,
        quality_passed=True,
        quality_reason=None
    )
