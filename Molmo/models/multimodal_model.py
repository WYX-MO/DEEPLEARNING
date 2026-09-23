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

    def __init__(
        self,
        vision_dim,
        llm_dim,
        vocab_size,
        max_seq_len,
        d_model,
        num_heads,
        d_ff,
        num_layers
    ):
        super().__init__()

        self.vision_encoder = VisionTransformer(
            d_model = vision_dim,
            cls=False
        )

        self.vision_projector = VisionProjector(
            vision_dim,
            llm_dim
        )

        self.llm = ModernGPT(
            vocab_size,
            max_seq_len,
            d_model,
            num_heads,
            d_ff,
            num_layers
        )

    def forward(self, image, text_feature):
        # image → visual features
        image_feature = self.vision_encoder(image)

        # visual features → LLM dimension
        visual_tokens = self.vision_projector(image_feature)

        # image tokens + text tokens
        multimodal_embeds = torch.cat([visual_tokens, text_feature], dim=1)

        # directly enter Transformer
        logits = self.llm(
            inputs_embeds=multimodal_embeds
        )

        return logits

