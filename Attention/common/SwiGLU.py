#SwiGLU.py
import torch
import torch.nn as nn

import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if self.d_ff == None:
            self.d_ff = d_model * 4
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x1 = self.w1(x)
        x2 = self.w2(x)
        x1 = torch.nn.functional.silu(x1)
        x = x1 * x2
        return self.w3(x)