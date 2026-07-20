import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, cohen_kappa_score
)

from training.config import TrainConfig, DEFAULT_CONFIG
from training.dataset import create_dataloader
from training.metrics import compute_qwk, compute_accuracy
from model.efficientnet import DRModel
from preprocessing.transforms import get_val_transforms

def load_checkpoint(checkpoint_path: str, config: TrainConfig) -> DRModel:
    model = DRModel(num_classes=config.num_classes, dropout_rate=config.dropout_rate, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

@torch.no_grad()
def run_evaluation(
    model: DRModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_tta: bool = True,
) -> tuple[list[int], list[int], np.ndarray]:
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]
        
        if use_tta:
            # 3-pass Test-Time Augmentation (Original, H-Flip, V-Flip)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            images_h = torch.flip(images, dims=[3])
            logits_h = model(images_h)
            probs_h = torch.softmax(logits_h, dim=1)
            
            images_v = torch.flip(images, dims=[2])
            logits_v = model(images_v)
            probs_v = torch.softmax(logits_v, dim=1)
            
            probs = (probs + probs_h + probs_v) / 3.0
            preds = probs.argmax(dim=1)
        else:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        
    return all_labels, all_preds, np.array(all_probs)

def plot_and_save_metrics(
    y_true: list[int],
    y_pred: list[int],
    probs: np.ndarray,
    output_dir: Path,
    num_classes: int = 5
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curves (One-vs-Rest)
    plt.figure(figsize=(8, 6))
    y_true_bin = np.eye(num_classes)[y_true]
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png")
    plt.close()
    
    # 3. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
        plt.plot(recall, precision, label=f"Class {i}")
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves.png")
    plt.close()

def main(config: TrainConfig = DEFAULT_CONFIG) -> None:
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    checkpoint_path = Path(config.model_dir) / "best_model.pth"
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}. Evaluation aborted.")
        return
        
    model = load_checkpoint(str(checkpoint_path), config).to(device)
    test_loader = create_dataloader(
        config.test_csv, config.test_image_dir, get_val_transforms(), config, shuffle=False
    )
    
    y_true, y_pred, probs = run_evaluation(model, test_loader, device, use_tta=True)
    
    qwk = compute_qwk(y_true, y_pred)
    acc = compute_accuracy(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"\nTest Set Accuracy (TTA): {acc:.4f}")
    print(f"Test Set QWK (TTA): {qwk:.4f}")
    print(f"Test Set Cohen's Kappa (TTA): {cohen_kappa:.4f}")
    
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    output_dir = Path("reports/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=4)
        
    plot_and_save_metrics(y_true, y_pred, probs, output_dir, config.num_classes)
    
    summary = {
        "accuracy": acc,
        "qwk": qwk,
        "cohen_kappa": cohen_kappa,
        "precision_macro": report_dict["macro avg"]["precision"],
        "recall_macro": report_dict["macro avg"]["recall"],
        "f1_macro": report_dict["macro avg"]["f1-score"],
        "precision_weighted": report_dict["weighted avg"]["precision"],
        "recall_weighted": report_dict["weighted avg"]["recall"],
        "f1_weighted": report_dict["weighted avg"]["f1-score"]
    }
    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("Evaluation completed. Reports and curves saved in reports/evaluation/.")

if __name__ == "__main__":
    main()
