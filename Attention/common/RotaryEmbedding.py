#RotaryEmbedding.py
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=128):
        super().__init__()

        freqs = 1.0 / (
            10000 ** (
                torch.arange(0, head_dim, 2).float()
                / head_dim
            )
        )

        positions = torch.arange(max_seq_len).float()

        angles = positions[:, None] * freqs[None, :]

        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x, start_pos=0):
        # x shape: [B, num_heads, seq_len, head_dim]
        B, nh, seq_len, hd = x.shape

        # 取从 start_pos 开始的seq_len个位置的cos/sin
        cos = self.cos[start_pos : start_pos + seq_len]
        sin = self.sin[start_pos : start_pos + seq_len]

        # 广播到匹配x的维度 [1,1,seq_len,hd//2]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot  = x_even * sin + x_odd * cos

        x_rotated = torch.stack([x_even_rot, x_odd_rot], dim=-1)
        x_rotated = x_rotated.flatten(-2)

        return x_rotated