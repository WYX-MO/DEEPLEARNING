# CLSToken.py
import torch
import torch.nn as nn



class CLStoken_Generator(nn.Module):
    def __init__(self, d_model=192):   
        """
        Add CLS token to the input tensor.
        x: (B, seq_len, d_model)
        out: (B, seq_len + 1, d_model)
        """
        super().__init__()
        self.cls_tokens = torch.zeros(1, 1, d_model)
        self.cls_tokens = nn.init.trunc_normal_(self.cls_tokens, mean=0.0, std=0.02).detach()
        self.cls_tokens = nn.Parameter(self.cls_tokens.requires_grad_(True))
        # self.register_parameter('cls_tokens', self.cls_tokens)

    def forward(self, x):
        cls_tokens = self.cls_tokens.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x