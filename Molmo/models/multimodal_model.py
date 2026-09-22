#multimodal_model.py

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..'))
from Attention.gpt.models.modern_gpt import ModernGPT
from Molmo.models.connector import VisionProjector
from Attention.common.Tokenizer import Tokenizer
from Attention.common.vision.Vit import VisionTransformer
class MultimodalModel(nn.Module):
    def __init__(self, vision_dim, llm_dim, vocab_size, max_seq_len, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        _,self.vision_encoder = VisionTransformer(vision_dim, llm_dim, cls= False)
        self.vision_projector = VisionProjector(vision_dim, llm_dim)
        self.llm = ModernGPT(vocab_size, max_seq_len, d_model, num_heads, d_ff, num_layers)

    def forward(self, x, text_feature):
        #get image feature
        x = self.vision_encoder(x)
        #project to llm dim
        x = self.vision_projector(x) # [B, seq_len, llm_dim]
        #concat image feature and text feature
        x = torch.cat([x, text_feature], dim=1)
        #pass to llm
        x = self.llm(x)
        return x    