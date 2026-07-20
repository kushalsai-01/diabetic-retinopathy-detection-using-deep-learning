import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from dataclasses import dataclass

@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    qwk: float
    per_class_acc: list[float]

def compute_qwk(y_true: list[int], y_pred: list[int]) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

def compute_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return float(np.mean(np.array(y_true) == np.array(y_pred)))

def compute_per_class_accuracy(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int = 5,
) -> list[float]:
    if not y_true:
        return [0.0] * num_classes
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sums = cm.sum(axis=1)
    accs = []
    for i in range(num_classes):
        if sums[i] > 0:
            accs.append(float(cm[i, i] / sums[i]))
        else:
            accs.append(0.0)
    return accs

class MetricsTracker:
    def __init__(self) -> None:
        self.losses: list[float] = []
        self.all_preds: list[int] = []
        self.all_labels: list[int] = []

    def update(self, loss: float, preds: list[int], labels: list[int]) -> None:
        self.losses.append(loss)
        self.all_preds.extend(preds)
        self.all_labels.extend(labels)

    def compute(self) -> EpochMetrics:
        avg_loss = float(np.mean(self.losses)) if self.losses else 0.0
        acc = compute_accuracy(self.all_labels, self.all_preds)
        qwk = compute_qwk(self.all_labels, self.all_preds)
        per_class = compute_per_class_accuracy(self.all_labels, self.all_preds)
        return EpochMetrics(loss=avg_loss, accuracy=acc, qwk=qwk, per_class_acc=per_class)
