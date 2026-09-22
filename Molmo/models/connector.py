#connector.py
import torch
import torch.nn as nn



class VisionProjector(nn.Module):
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.projection = nn.Linear(vision_dim, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)
    
    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        return x
