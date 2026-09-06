# Decoder_block.py

import torch.nn as nn
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention import MultiHeadAttention
from CrossAttention import MultiHeadCrossAttention
from feedforward import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, d_model_img,d_model_seq, num_heads,mlp_ratio =4.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model_seq, num_heads)
        self.self_cross_attn = MultiHeadCrossAttention(d_model_img, d_model_seq, num_heads)
        self.layer_norm = nn.LayerNorm(d_model_seq)
        self.cross_layer_norm = nn.LayerNorm(d_model_seq)
        self.ffn = FeedForward(d_model_seq, int(d_model_seq * mlp_ratio))
        #self.mask = torch.triu(torch.ones(1, 1, requires_grad=False), diagonal=1).bool()
        self.ffn_norm = nn.LayerNorm(d_model_seq)
    def forward(self, img_,seq_,mask):
        self_attn_output, _ = self.self_attn(seq_, mask=mask)
        seq_ = self.layer_norm(seq_ + self_attn_output)

        cross_attn_output, _ = self.self_cross_attn(img_, seq_)
        seq_ = self.cross_layer_norm(seq_ + cross_attn_output)

        ffn_output = self.ffn(seq_)
        seq_ = self.ffn_norm(seq_ + ffn_output)
        return seq_
        