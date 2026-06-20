"""
Training script chính.

Cách dùng:
    python train.py --model cnn
    python train.py --model resnet18
    python train.py --model vit
    python train.py --model all          # train cả 3
"""

import os
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.dataloader import get_dataloaders
from models import SimpleCNN, ResNet18PathMNIST, VisionTransformer
from utils.visualization import plot_training_curves


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, cfg: dict, device: torch.device):
    n = cfg["dataset"]["num_classes"]
    if model_name == "cnn":
        return SimpleCNN(num_classes=n, dropout=cfg["models"]["cnn"]["dropout"]).to(device)
    elif model_name == "resnet18":
        m = cfg["models"]["resnet18"]
        return ResNet18PathMNIST(num_classes=n, pretrained=m["pretrained"],
                                  freeze_backbone=m["freeze_backbone"]).to(device)
    elif model_name == "vit":
        v = cfg["models"]["vit"]
        return VisionTransformer(
            image_size=v["image_size"], patch_size=v["patch_size"],
            num_classes=n, embed_dim=v["embed_dim"], num_heads=v["num_heads"],
            num_layers=v["num_layers"], mlp_ratio=v["mlp_ratio"],
            dropout=v["dropout"]
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        labels = labels.squeeze(1).long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total         += images.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze(1).long().to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss    += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total         += images.size(0)

    return total_loss / total, total_correct / total


def train_model(model_name: str, cfg: dict, device: torch.device):
    print(f"\n{'='*55}")
    print(f"  TRAINING: {model_name.upper()}")
    print(f"{'='*55}")

    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["figures_dir"],     exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=cfg["paths"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
        image_size=cfg["dataset"]["image_size"],
    )

    model     = build_model(model_name, cfg, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    epochs   = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    best_acc = 0.0
    wait     = 0
    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    ckpt_path = os.path.join(cfg["paths"]["checkpoints_dir"], f"{model_name}_best.pth")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        flag = ""
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), ckpt_path)
            wait = 0
            flag = " ← best"
        else:
            wait += 1

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"Loss: {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc: {tr_acc:.4f}/{vl_acc:.4f}{flag}")

        if wait >= patience:
            print(f"  Early stopping tại epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(f"\n  Best Val Accuracy : {best_acc:.4f}")
    print(f"  Thời gian train   : {elapsed/60:.1f} phút")

    plot_training_curves(history, model_name, cfg["paths"]["figures_dir"])
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["cnn", "resnet18", "vit", "all"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models_to_train = ["cnn", "resnet18", "vit"] if args.model == "all" else [args.model]
    for m in models_to_train:
        train_model(m, cfg, device)

    print("\nHoàn tất training!")


if __name__ == "__main__":
    main()
