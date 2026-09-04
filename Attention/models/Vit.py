#Vit.py
import torch
import torch.nn as nn
import os
import sys 
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
import models.Transformer_block
import models.CLSToken
import models.PositionEncoding
import models.PatchEncoding

class VitionTransformer(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, d_model=192, num_layers=12, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.patch_encoding = models.PatchEncoding(in_channels=in_channels, patch_size=patch_size, d_model=d_model)
        self.cls_token = models.CLSToken(d_model=d_model)
        self.position_encoding = models.PositionEncoding(d_model=d_model)
        self.transformer_blocks = nn.ModuleList([
            models.Transformer_block.TransformerBlock(d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.patch_encoding(x)  # (B, num_patches, d_model)
        x = self.cls_token(x)       # (B, num_patches+1, d_model)
        x = self.position_encoding(x)  # (B, num_patches+1, d_model)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)  # (B, num_patches+1, d_model)
        return x
