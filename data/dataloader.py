"""
DataLoader cho PathMNIST.
Mean/Std tính từ tập train: R(0.7406, 0.2086), G(0.5331, 0.2530), B(0.7059, 0.2168)
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST


# Thống kê pixel tính từ tập train PathMNIST
MEAN = (0.7406, 0.5331, 0.7059)
STD  = (0.2086, 0.2530, 0.2168)

NUM_CLASSES = 9
CLASS_NAMES = [
    "Adipose",
    "Background",
    "Debris",
    "Lymphocytes",
    "Mucus",
    "Smooth Muscle",
    "Normal Colon Mucosa",
    "Cancer-Associated Stroma",
    "Colorectal Adenocarcinoma Epithelium",
]


def get_transforms(split: str, image_size: int = 28):
    """
    Train: random flip + color jitter + normalize
    Val/Test: chỉ normalize
    """
    normalize = transforms.Normalize(mean=MEAN, std=STD)

    if split == "train":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders(data_dir: str = "./data/raw",
                    batch_size: int = 64,
                    num_workers: int = 2,
                    image_size: int = 28):
    """Trả về (train_loader, val_loader, test_loader)."""

    train_ds = PathMNIST(split="train", transform=get_transforms("train", image_size),
                         download=True, root=data_dir, size=image_size)
    val_ds   = PathMNIST(split="val",   transform=get_transforms("val",   image_size),
                         download=True, root=data_dir, size=image_size)
    test_ds  = PathMNIST(split="test",  transform=get_transforms("test",  image_size),
                         download=True, root=data_dir, size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader
