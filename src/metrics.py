"""IoU, per-class F1, confusion matrix utilities."""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)

from .utils import TERRAIN_CLASSES, NUM_CLASSES


def compute_confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES):
    """Compute confusion matrix, handling missing classes."""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def per_class_iou(cm):
    """Compute per-class IoU from a confusion matrix."""
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = np.where(union > 0, intersection / union, 0.0)
    return iou


def mean_iou(cm):
    """Mean IoU across classes with at least one sample."""
    iou = per_class_iou(cm)
    mask = (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)) > 0
    return iou[mask].mean() if mask.any() else 0.0


def compute_all_metrics(y_true, y_pred, num_classes=NUM_CLASSES):
    """Compute a full metrics dictionary."""
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    iou = per_class_iou(cm)
    miou = mean_iou(cm)

    class_names = [TERRAIN_CLASSES.get(i, f"class_{i}") for i in range(num_classes)]

    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "mean_iou": float(miou),
        "per_class_iou": {class_names[i]: float(iou[i]) for i in range(num_classes)},
        "per_class_f1": {
            class_names[i]: report[class_names[i]]["f1-score"]
            for i in range(num_classes)
        },
        "per_class_precision": {
            class_names[i]: report[class_names[i]]["precision"]
            for i in range(num_classes)
        },
        "per_class_recall": {
            class_names[i]: report[class_names[i]]["recall"]
            for i in range(num_classes)
        },
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def print_metrics_table(metrics):
    """Pretty-print a metrics summary."""
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Mean IoU:         {metrics['mean_iou']:.4f}")
    print()
    header = f"{'Class':<15} {'IoU':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}"
    print(header)
    print("-" * len(header))
    for cls in metrics["per_class_iou"]:
        print(
            f"{cls:<15} "
            f"{metrics['per_class_iou'][cls]:>6.3f} "
            f"{metrics['per_class_f1'][cls]:>6.3f} "
            f"{metrics['per_class_precision'][cls]:>6.3f} "
            f"{metrics['per_class_recall'][cls]:>6.3f}"
        )
