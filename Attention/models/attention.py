#attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

seed = 42
random.seed(seed)

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_k)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # TODO
        # 1. QK^T
        # 2. scale
        # 3. softmax
        # 4. attention @ V
        attention = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5), dim=-1)
        output = torch.matmul(attention, V)

        return output, attention

class discrete_MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.heads = nn.ModuleList([SelfAttention(d_model, d_k) for _ in range(num_heads)])
        self.W_o = nn.Linear(num_heads * d_k, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]

        outputs = []
        attentions = []

        for head in self.heads:
            output, attention = head(x)
            outputs.append(output)
            attentions.append(attention)

        # Concatenate the outputs of all heads
        concatenated_output = torch.cat(outputs, dim=-1)
        final_output = self.W_o(concatenated_output)

        return final_output, attentions

class MultiHeadAttention(nn.Module):
    def __init__ (self, d_model,  num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W = nn.Linear(d_model, num_heads * self.d_k * 3)  # For Q, K, V
        self.W_o = nn.Linear(num_heads * self.d_k, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()

        # Linear projection to get Q, K, V for all heads
        qkv = self.W(x)  # [batch_size, seq_len, num_heads * d_k * 3]
        qkv = qkv.view(batch_size, seq_len, self.num_heads, -1)  # [batch_size, seq_len, num_heads, 3 * d_k]
        Q, K, V = torch.chunk(qkv, 3, dim=-1)  # Each of shape [batch_size, seq_len, num_heads, d_k]

        # Transpose to get the shape [batch_size, num_heads, seq_len, d_k]
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = K.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        V = V.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # [batch_size, num_heads, seq
        attention_scores = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]                                                            
        attention = torch.matmul(attention_scores, V)  # [batch_size, num_heads, seq_len, d_k]
        attention = attention.transpose(1,2)
        attention = attention.reshape(batch_size, seq_len, -1)
        attention = self.W_o(attention)  # [batch_size, seq_len, d_model]
        
        return attention,attention_scores  # [batch_size, num_heads, seq_len, d_k]