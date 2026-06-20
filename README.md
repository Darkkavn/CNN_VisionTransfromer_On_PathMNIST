# Phân loại mô bệnh học bằng CNN và Vision Transformer trên PathMNIST

Tiểu luận thạc sĩ — Deep Learning  
**Dataset**: [PathMNIST](https://medmnist.com/) (MedMNIST v2)  
**Task**: Multi-class classification (9 lớp mô đại trực tràng)

---

## Giới thiệu

PathMNIST được xây dựng từ ảnh mô bệnh học H&E của ung thư đại trực tràng (NCT-CRC-HE-100K), gồm 107,180 ảnh 28×28 RGB chia thành 9 loại mô:

| ID | Nhãn |
|----|------|
| 0 | Adipose |
| 1 | Background |
| 2 | Debris |
| 3 | Lymphocytes |
| 4 | Mucus |
| 5 | Smooth Muscle |
| 6 | Normal Colon Mucosa |
| 7 | Cancer-Associated Stroma |
| 8 | Colorectal Adenocarcinoma Epithelium |

**Phân chia dữ liệu**: Train 89,996 | Val 10,004 | Test 7,180

---

## Các mô hình

| Mô hình | Mô tả | Kỳ vọng |
|---------|-------|---------|
| **SimpleCNN** | 3 khối Conv-BN-ReLU + FC | 80–88% |
| **ResNet18** | Transfer learning từ ImageNet | 85–92% |
| **ViT** | Vision Transformer từ đầu (patch 4×4) | 88–94% |

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Sử dụng

### Train
```bash
python train.py --model cnn        # CNN baseline
python train.py --model resnet18   # ResNet18 transfer learning
python train.py --model vit        # Vision Transformer
python train.py --model all        # Train cả 3 mô hình
```

### Evaluate
```bash
python evaluate.py                 # Đánh giá cả 3 mô hình
python evaluate.py --model cnn     # Chỉ đánh giá CNN
```

---

## Cấu trúc project

```
CNN_VisionTransfromer_On_PathMNIST/
├── configs/
│   └── config.yaml          # Cấu hình hyperparameters
├── data/
│   └── dataloader.py        # PathMNIST DataLoader + augmentation
├── models/
│   ├── cnn.py               # CNN baseline
│   ├── resnet.py            # ResNet18 transfer learning
│   └── vit.py               # Vision Transformer
├── utils/
│   ├── metrics.py           # Accuracy, Precision, Recall, F1, CM
│   ├── visualization.py     # Loss/Acc curves, Confusion Matrix, So sánh
│   └── gradcam.py           # Grad-CAM giải thích CNN
├── train.py                 # Script training
├── evaluate.py              # Script đánh giá & so sánh
├── requirements.txt
└── results/                 # Kết quả training (checkpoints, figures)
```

---

## Kết quả (cập nhật sau khi train)

| Mô hình | Accuracy | Precision | Recall | F1-score |
|---------|----------|-----------|--------|----------|
| CNN | — | — | — | — |
| ResNet18 | — | — | — | — |
| ViT | — | — | — | — |

---

## Tài liệu tham khảo

- Kather et al. (2019). *Predicting survival from colorectal cancer histology slides.* PLOS Medicine.
- Yang et al. (2023). *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data.
- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
- He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
