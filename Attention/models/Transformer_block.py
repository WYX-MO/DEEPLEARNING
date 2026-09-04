#Transformer_block.py
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
from Attention.models.attention import MultiHeadAttention
from Attention.models.feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.attention = MultiHeadAttention(d_model,  num_heads)
        self.feed_forward = FeedForward(d_model, int(d_model * mlp_ratio))
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        attention_output, _ = self.attention(x)
        #print(x)
        x = self.layer_norm1(x + attention_output)  # Residual connection + LayerNorm
        #print(x)
        feed_forward_output = self.feed_forward(x)
        x = self.layer_norm2(x + feed_forward_output)  # Residual connection + LayerNorm
        return x

if __name__ == "__main__":
    # Test the TransformerBlock
    torch.manual_seed(42)
    x = torch.randn(2, 5, 8)  # [batch_size, seq_len, d_model]
    transformer_block = TransformerBlock(d_model=8, num_heads=2, mlp_ratio=4.0)
    output = transformer_block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)