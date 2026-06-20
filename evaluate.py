"""
Evaluate tất cả mô hình trên tập test và xuất bảng so sánh.

Cách dùng:
    python evaluate.py                  # đánh giá cả 3 mô hình
    python evaluate.py --model cnn      # chỉ đánh giá CNN
"""

import os
import argparse
import json
import yaml
import numpy as np
import torch
from tqdm import tqdm

from data.dataloader import get_dataloaders, CLASS_NAMES
from models import SimpleCNN, ResNet18PathMNIST, VisionTransformer
from utils.metrics import compute_metrics, print_metrics
from utils.visualization import plot_confusion_matrix, plot_comparison_table
from train import build_model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="  Predict", leave=False):
        images = images.to(device)
        preds  = model(images).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.squeeze(1).numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def evaluate_model(model_name: str, cfg: dict, test_loader, device):
    ckpt = os.path.join(cfg["paths"]["checkpoints_dir"], f"{model_name}_best.pth")
    if not os.path.exists(ckpt):
        print(f"  [SKIP] Chưa có checkpoint: {ckpt}")
        return None

    model = build_model(model_name, cfg, device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    preds, labels = predict(model, test_loader, device)
    metrics = compute_metrics(labels, preds, CLASS_NAMES)
    print_metrics(metrics, model_name.upper())

    plot_confusion_matrix(
        metrics["confusion_matrix"], CLASS_NAMES,
        model_name, cfg["paths"]["figures_dir"]
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["cnn", "resnet18", "vit", "all"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["paths"]["figures_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(
        data_dir=cfg["paths"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
        image_size=cfg["dataset"]["image_size"],
    )

    models_to_eval = ["cnn", "resnet18", "vit"] if args.model == "all" else [args.model]
    results = {}

    for m in models_to_eval:
        print(f"\nEvaluating {m.upper()}...")
        metrics = evaluate_model(m, cfg, test_loader, device)
        if metrics:
            results[m.upper()] = {
                "accuracy":  metrics["accuracy"],
                "precision": metrics["precision"],
                "recall":    metrics["recall"],
                "f1_score":  metrics["f1_score"],
            }

    if len(results) > 1:
        plot_comparison_table(results, cfg["paths"]["figures_dir"])

    # Lưu JSON để dùng trong báo cáo
    json_path = os.path.join(cfg["paths"]["results_dir"], "metrics_summary.json")
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {json_path}")

    # In bảng so sánh
    if results:
        print(f"\n{'='*55}")
        print("  BẢNG SO SÁNH TỔNG HỢP")
        print(f"{'='*55}")
        print(f"  {'Model':<12} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print(f"  {'-'*50}")
        for m, v in results.items():
            print(f"  {m:<12} {v['accuracy']:>9.4f} {v['precision']:>10.4f} "
                  f"{v['recall']:>8.4f} {v['f1_score']:>8.4f}")


if __name__ == "__main__":
    main()
