#RMSNorm.py

import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self,d_model,eps = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.w =  nn.Parameter(torch.ones(self.d_model))

    def forward(self,x):
        # x: [batch_size, seq_len, d_model]
        x_s = torch.square(x)
        mean = x_s.mean(dim=-1, keepdim=True)
        down = (mean+self.eps).sqrt()
        out = x / down
        return out * self.w

if __name__ == "__main__":
    x = torch.randn(2, 5, 192)

    model = RMSNorm(192)

    y = model(x)

    print(x.shape)
    print(y.shape)