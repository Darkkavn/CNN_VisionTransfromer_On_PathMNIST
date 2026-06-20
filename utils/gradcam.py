"""
Grad-CAM để giải thích quyết định của mô hình CNN/ResNet.
Chỉ áp dụng cho mô hình có conv layers (CNN, ResNet18).
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP over spatial
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def visualize_gradcam(model, target_layer, images, labels, class_names,
                      save_dir, n_samples=9, device="cpu"):
    """Hiển thị Grad-CAM cho n_samples ảnh."""
    os.makedirs(save_dir, exist_ok=True)
    gradcam = GradCAM(model, target_layer)
    model.to(device)

    # Denormalize về [0,1] để hiển thị
    MEAN = torch.tensor([0.7406, 0.5331, 0.7059]).view(3, 1, 1)
    STD  = torch.tensor([0.2086, 0.2530, 0.2168]).view(3, 1, 1)

    fig, axes = plt.subplots(n_samples, 3, figsize=(9, n_samples * 3))
    fig.suptitle("Grad-CAM Visualization", fontsize=13, fontweight="bold")

    for i in range(n_samples):
        img = images[i].unsqueeze(0).to(device)
        label = labels[i].item()

        heatmap = gradcam.generate(img)

        # Denormalize
        img_show = (images[i] * STD + MEAN).permute(1, 2, 0).numpy().clip(0, 1)
        overlay = cm.jet(heatmap)[..., :3] * 0.4 + img_show * 0.6

        with torch.no_grad():
            pred = model(img).argmax(dim=1).item()

        axes[i][0].imshow(img_show)
        axes[i][0].set_title(f"True: {class_names[label][:12]}", fontsize=8)
        axes[i][0].axis("off")

        axes[i][1].imshow(heatmap, cmap="jet")
        axes[i][1].set_title("Heatmap", fontsize=8)
        axes[i][1].axis("off")

        axes[i][2].imshow(overlay)
        axes[i][2].set_title(f"Pred: {class_names[pred][:12]}", fontsize=8,
                              color="green" if pred == label else "red")
        axes[i][2].axis("off")

    path = os.path.join(save_dir, "gradcam.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")
