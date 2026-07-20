import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import cv2
import torch
import pandas as pd
from model.efficientnet import DRModel
from model.gradcam import GradCAM
from preprocessing.transforms import get_val_transforms
from training.config import DEFAULT_CONFIG

def main() -> None:
    output_dir = Path("reports/gradcam")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(DEFAULT_CONFIG.model_dir) / "best_model.pth"
    
    if not checkpoint_path.exists():
        print("No checkpoint found to run GradCAM. Skipping generation.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DRModel(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    val_df = pd.read_csv(DEFAULT_CONFIG.val_csv)
    transforms = get_val_transforms()
    cam = GradCAM(model)
    
    # Save a few sample visualizations
    samples = val_df.sample(min(5, len(val_df)), random_state=42)
    
    for idx, row in samples.iterrows():
        id_code = row["id_code"]
        diagnosis = int(row["diagnosis"])
        img_path = Path(DEFAULT_CONFIG.val_image_dir) / f"{id_code}.png"
        
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = transforms(image=img_rgb)["image"]
        input_tensor = transformed.unsqueeze(0).to(device)
        
        heatmap = cam.compute(input_tensor, diagnosis)
        overlay = cam.overlay(heatmap, img)
        
        # Save side by side
        combined = cv2.hconcat([img, overlay])
        cv2.imwrite(str(output_dir / f"gradcam_{id_code}_class{diagnosis}.png"), combined)
        
    cam.remove_hooks()
    print("GradCAM generation complete. Visualizations saved in reports/gradcam/.")

if __name__ == "__main__":
    main()
