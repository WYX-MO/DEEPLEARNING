#Vit.py
import torch
import torch.nn as nn
import os
import sys 
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))

from Attention.models.Transformer_block import TransformerBlock
from Attention.models.CLSToken import CLStoken_Generator
from Attention.models.PositionEncoding import PositionEncoding
from Attention.models.PatchEmbeding import PatchEmbeding

class VitionTransformer(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, d_model=192, num_layers=12, num_heads=3, mlp_ratio=4.0, num_classes=10):
        super().__init__()
        self.patch_encoding = PatchEmbeding(in_channels=in_channels, patch_size=patch_size, d_model=d_model)
        self.cls_token = CLStoken_Generator(d_model=d_model)
        self.position_encoding = PositionEncoding(d_model=d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])

        self.clsHead = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.patch_encoding(x)  # (B, num_patches, d_model)
        x = self.cls_token(x)       # (B, num_patches+1, d_model)
        x = self.position_encoding(x)  # (B, num_patches+1, d_model)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)  # (B, num_patches+1, d_model)
        x = self.clsHead(x[:, 0])  # Use the CLS token for classification
        return x

if __name__ == "__main__":
    model = VitionTransformer(in_channels=3, patch_size=4, d_model=192, num_layers=12, num_heads=3, mlp_ratio=4.0, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)