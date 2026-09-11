#GPT_block.py

import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
from Attention.models.attention import MultiHeadAttention
from Attention.models.feedforward import FeedForward

class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        attn_output = self.attention(x, mask)[0]  # [batch_size, seq_len, d_model]
        x = self.norm1(x + attn_output)  # Residual connection + LayerNorm
        ff_output = self.feed_forward(x)  # [batch_size, seq_len, d_model]
        x = self.norm2(x + ff_output)  # Residual connection + LayerNorm
        return x  # [batch_size, seq_len, d_model]