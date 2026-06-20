"""
Tính Accuracy, Precision, Recall, F1-score, Confusion Matrix.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: list = None) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1_score":  f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def print_metrics(metrics: dict, model_name: str = ""):
    print(f"\n{'='*55}")
    print(f"  KẾT QUẢ: {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1_score']:.4f}")
    print(f"\n{metrics['classification_report']}")
