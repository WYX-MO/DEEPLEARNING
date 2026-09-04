#Vit.py
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, d_model=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: (B, 3, 224, 224)
        out: (B, 196, 768)
        """
        x = self.proj(x)       # (B,768,14,14)
        x = x.flatten(2)       # (B,768,196)
        x = x.transpose(1, 2)  # (B,196,768)
        return x