# PositionEncoding.py
import torch
import torch.nn as nn

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048):
       super().__init__()
       self.pe = torch.zeros(1, max_seq_len, d_model)
       self.pe = nn.init.trunc_normal_(self.pe, mean=0.0, std=0.02)
       self.pos_emb = nn.Parameter(self.pe)

    def forward(self, x):
       return x + self.pos_emb[:, :x.size(1), :]  # Add positional encoding