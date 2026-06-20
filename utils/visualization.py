"""
Vẽ Loss/Accuracy curves và Confusion Matrix.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(history: dict, model_name: str, save_dir: str):
    """history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}"""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History — {model_name}", fontsize=13, fontweight="bold")

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="tomato")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="steelblue")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc",   color="tomato")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                          model_name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    # Normalize để dễ đọc
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")

    short_names = [n[:12] for n in class_names]

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Số lượng", "Tỷ lệ (%)"],
        ["d", ".2f"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=short_names, yticklabels=short_names,
                    ax=ax, linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45)

    path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def plot_comparison_table(results: dict, save_dir: str):
    """results = {'CNN': {'accuracy':...,'precision':...,'recall':...,'f1_score':...}, ...}"""
    os.makedirs(save_dir, exist_ok=True)

    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-score"]

    x = np.arange(len(metrics))
    width = 0.25
    colors = ["steelblue", "tomato", "mediumseagreen"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        values = [results[model][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model, color=color, alpha=0.85)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("So sánh hiệu suất các mô hình trên PathMNIST", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(save_dir, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")
