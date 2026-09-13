#GPT_block.py

import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..'))
from Attention.common.attention import MultiHeadAttention,MHAttnWithCache
from Attention.common.feedforward import FeedForward
from Attention.common.RMSNorm import RMSNorm
from Attention.common.SwiGLU import SwiGLU


class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.attention_with_cache = MHAttnWithCache(d_model, num_heads)
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

class ModernGPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads,rope = True)
        self.attention_with_cache = MHAttnWithCache(d_model, num_heads)
        self.RMSNorm1 = RMSNorm(d_model)
        self.RMSNorm2 = RMSNorm(d_model)
        self.RMSNorm3 = RMSNorm(d_model)
        self.SwiGLU = SwiGLU(d_model, d_ff)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        rms1 = self.RMSNorm1(x)
        attn_output = self.attention(rms1, mask)[0]  # [batch_size, seq_len, d_model]
        x = self.RMSNorm2(x+attn_output )  # Residual connection + RMSNorm
        ff_output = self.SwiGLU(x)  # [batch_size, seq_len, d_model]
        x = x + ff_output  # Residual connection + RMSNorm
        return x  # [batch_size, seq_len, d_model]
