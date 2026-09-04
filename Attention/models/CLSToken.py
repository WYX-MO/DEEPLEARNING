# CLSToken.py
import torch
import torch.nn as nn



class CLStoken_Generator():
    def __init__(self, x):
        """
        Add CLS token to the input tensor.
        x: (B, seq_len, d_model)
        out: (B, seq_len + 1, d_model)
        """
        cls_tokens = torch.zeros(1, 1, self.proj.out_channels).to(x.device)
        cls_tokens = cls_tokens.expand(x.size(0), -1, -1)  # (B, 1, d_model)
        cls_tokens = nn.init.trunc_normal_(cls_tokens, mean=0.0, std=0.02).detach()
        cls_tokens = nn.Parameter(cls_tokens.requires_grad_(True))
        self.register_parameter('cls_tokens', cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, seq_len + 1, d_model)
        return x
