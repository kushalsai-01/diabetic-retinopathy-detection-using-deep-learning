"""
Multimodal DR Prediction Script

Demonstrates how to use the trained multimodal model for inference.
Supports both modes:
1. With clinical data (best accuracy)
2. Without clinical data (image only)
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import json
from typing import Optional, Dict, Any

from src.models.multimodal import MultimodalDRClassifier
from src.data.transforms import get_inference_transforms


# DR severity labels
DR_LABELS = {
    0: "No DR (Healthy)",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

# Clinical feature order (must match training)
CLINICAL_FEATURES = [
    'age', 'gender', 'diabetes_duration', 'hba1c',
    'bp_sys', 'bp_dia', 'bmi', 'smoking', 'insulin'
]


def load_model(checkpoint_path: str, device: str = 'cuda') -> MultimodalDRClassifier:
    """Load trained multimodal model."""
    print(f"📂 Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = MultimodalDRClassifier(
        num_classes=5,
        backbone="resnet50",
        num_tabular_features=9,
    )
    
    # Load weights
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model


def prepare_image(image_path: str, image_size: int = 512, crop_size: int = 448) -> torch.Tensor:
    """Load and preprocess image."""
    transform = get_inference_transforms(image_size, crop_size)
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image=np.array(image))['image']
    
    return image_tensor.unsqueeze(0)  # Add batch dimension


def prepare_clinical_data(clinical_dict: Optional[Dict[str, float]]) -> Optional[torch.Tensor]:
    """Prepare clinical data tensor from dictionary."""
    if clinical_dict is None:
        return None
    
    # Extract features in correct order
    clinical_values = []
    for feature in CLINICAL_FEATURES:
        if feature not in clinical_dict:
            raise ValueError(f"Missing clinical feature: {feature}")
        clinical_values.append(clinical_dict[feature])
    
    # Convert to tensor
    clinical_tensor = torch.tensor([clinical_values], dtype=torch.float32)
    
    return clinical_tensor


def predict(
    model: MultimodalDRClassifier,
    image_path: str,
    clinical_data: Optional[Dict[str, float]] = None,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """
    Make prediction on a single image.
    
    Args:
        model: Trained multimodal model
        image_path: Path to fundus image
        clinical_data: Dictionary of clinical features (optional)
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with prediction results
    """
    # Prepare inputs
    image_tensor = prepare_image(image_path).to(device)
    
    if clinical_data is not None:
        clinical_tensor = prepare_clinical_data(clinical_data).to(device)
    else:
        clinical_tensor = None
    
    # Predict
    with torch.no_grad():
        logits = model(image_tensor, clinical_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prediction = logits.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item()
    
    # Results
    results = {
        'image_path': str(image_path),
        'prediction': prediction,
        'prediction_label': DR_LABELS[prediction],
        'confidence': confidence,
        'probabilities': {
            DR_LABELS[i]: probabilities[0, i].item()
            for i in range(5)
        },
        'used_clinical_data': clinical_data is not None,
    }
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Pretty print prediction results."""
    print("\n" + "="*60)
    print("🔬 DIABETIC RETINOPATHY PREDICTION RESULTS")
    print("="*60)
    
    print(f"\n📸 Image: {results['image_path']}")
    print(f"💊 Clinical Data: {'Yes' if results['used_clinical_data'] else 'No (Image Only)'}")
    
    print(f"\n🎯 Prediction: {results['prediction_label']}")
    print(f"📊 Confidence: {results['confidence']:.2%}")
    
    print("\n📈 Class Probabilities:")
    for label, prob in results['probabilities'].items():
        bar = "█" * int(prob * 40)
        print(f"  {label:20s} {prob:6.2%} {bar}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Multimodal DR Prediction")
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to fundus image'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='experiments/checkpoints/best.ckpt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--clinical',
        type=str,
        default=None,
        help='Path to JSON file with clinical data (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Check image exists
    if not Path(args.image).exists():
        print(f"❌ Error: Image not found: {args.image}")
        return
    
    # Load clinical data if provided
    clinical_data = None
    if args.clinical:
        if not Path(args.clinical).exists():
            print(f"❌ Error: Clinical data file not found: {args.clinical}")
            return
        
        with open(args.clinical, 'r') as f:
            clinical_data = json.load(f)
        
        print(f"📊 Loaded clinical data from: {args.clinical}")
    else:
        print("⚠️  No clinical data provided - using image-only mode")
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Make prediction
    print(f"\n🔍 Analyzing image: {args.image}")
    results = predict(model, args.image, clinical_data, args.device)
    
    # Display results
    print_results(results)


if __name__ == "__main__":
    # Example usage (when running as script)
    import sys
    
    if len(sys.argv) == 1:
        # Demo mode
        print("📚 Multimodal DR Prediction Script")
        print("\nUsage examples:\n")
        
        print("1. With clinical data:")
        print("   python predict_multimodal.py --image patient.jpg --clinical clinical.json\n")
        
        print("2. Image only (no clinical data):")
        print("   python predict_multimodal.py --image patient.jpg\n")
        
        print("\nClinical data JSON format:")
        clinical_example = {
            "age": 54.0,
            "gender": 1,
            "diabetes_duration": 8.0,
            "hba1c": 7.2,
            "bp_sys": 140.0,
            "bp_dia": 90.0,
            "bmi": 28.5,
            "smoking": 0,
            "insulin": 1
        }
        print(json.dumps(clinical_example, indent=2))
        
        print("\n" + "-"*60)
        print("Run with --help for full options")
    else:
        main()
