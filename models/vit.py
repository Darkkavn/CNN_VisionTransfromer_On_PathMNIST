"""
Vision Transformer (ViT) từ đầu cho PathMNIST (28x28, patch_size=4 → 49 patches).
Dựa trên bài báo: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020).
"""

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    """Chia ảnh thành các patch và project lên embed_dim."""

    def __init__(self, image_size: int, patch_size: int,
                 in_channels: int, embed_dim: int):
        super().__init__()
        assert image_size % patch_size == 0, "image_size phải chia hết cho patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)                 # (B, embed_dim, H/P, W/P)
        x = rearrange(x, "b c h w -> b (h w) c")  # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return x + residual


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = embed_dim * mlp_ratio
        self.norm = nn.LayerNorm(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff   = FeedForward(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x


class VisionTransformer(nn.Module):
    """
    ViT nhỏ phù hợp ảnh 28x28.
    Mặc định: patch_size=4, embed_dim=128, num_heads=4, num_layers=6
    → 49 patches + 1 CLS token = 50 sequence length
    """

    def __init__(self,
                 image_size: int = 28,
                 patch_size: int = 4,
                 in_channels: int = 3,
                 num_classes: int = 9,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 6,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.pos_drop  = nn.Dropout(dropout)

        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                               # (B, num_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)                        # (B, num_patches+1, embed_dim)
        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer(x)
        x = self.norm(x[:, 0])                                # lấy CLS token
        return self.head(x)
