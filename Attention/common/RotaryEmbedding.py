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

        self.register_buffer(
            "cos",
            torch.cos(angles)
        )

        self.register_buffer(
            "sin",
            torch.sin(angles)
        )

    def forward(self, x):

        seq_len = x.size(-2)

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rot = (
            x_even * cos
            - x_odd * sin
        )

        x_odd_rot = (
            x_even * sin
            + x_odd * cos
        )

        x_rotated = torch.stack(
            [x_even_rot, x_odd_rot],
            dim=-1
        )

        x_rotated = x_rotated.flatten(-2)

        return x_rotated