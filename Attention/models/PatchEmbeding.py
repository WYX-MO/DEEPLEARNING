# PatchEmbeding.py
import torch
import torch.nn as nn

class PatchEmbeding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, d_model=192):
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
        x: (B, 3, H, H)
        out: (B, (H/patch_size)**2, d_model)
        """
        x = self.proj(x)       # (B,d_model,H/patch_size,H/patch_size)
        x = x.flatten(2)       # (B,d_model,(H/patch_size)**2)
        x = x.transpose(1, 2)  # (B,(H/patch_size)**2,d_model)
        #add cls

        #position
        
        return x
    