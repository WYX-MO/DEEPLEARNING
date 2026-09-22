#clip.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
sys.path.insert(0, os.path.join (os.path.dirname(os.path.abspath(__file__)),'..','..','..'))
from Attention.common.vision.Vit import VisionTransformer



class EasyTextEncoder(nn.Module):
    def __init__(self, vocab_size,d_model,embed_dim,max_seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.linear = nn.Linear(d_model, embed_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape
        position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).to(x.device)
        x = self.embedding(x) + self.position_embedding(position_ids)
        x = torch.mean(x, dim=1) # (B, d_model)
        x = self.linear(x)
        return x # (B, embed_dim)

class CLIP(nn.Module):
    def __init__(self, vocab_size, d_model, embed_dim, same_dim=256, max_seq_len=128):
        super().__init__()
        image_encoder = VisionTransformer(patch_size=8,cls = True,num_classes = 0)
        self.image_encoder = image_encoder
        self.text_encoder = EasyTextEncoder(vocab_size, d_model, embed_dim, max_seq_len)
        self.linear_img = nn.Linear(embed_dim, same_dim)
        self.linear_text = nn.Linear(embed_dim, same_dim)
        # 可学习的温度：logit_scale = log(1/τ)，初始 τ=0.07（CLIP 原文设置）。
        # 用 log 存是为了保证 exp() 之后恒为正；它是参数，会被 loss 一起优化。
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    def forward(self, image, text):
        batch_size = image.shape[0]
        _,image_features = self.image_encoder(image)
        text_features = self.text_encoder(text) # (B, embed_dim)
        
        text_emb = self.linear_text(text_features) # (B, same_dim)
        image_emb = self.linear_img(image_features) # (B, same_dim)

        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        similarity = image_emb @ text_emb.T # (B, B)

        # label = torch.arange(batch_size).to(image_emb.device)
        # loss_i = F.cross_entropy(similarity, label)
        # loss_t = F.cross_entropy(similarity.T, label)
        # loss =  (loss_i + loss_t)/2
        return similarity

if __name__ == "__main__":
    vocab_size = 26
    image = torch.randn(4, 3, 32, 32)
    text = torch.randint(0, vocab_size, (4, 20))
    model = CLIP(vocab_size, d_model=192, embed_dim=192)
    similarity = model(image, text)
    
    print(similarity.shape)

