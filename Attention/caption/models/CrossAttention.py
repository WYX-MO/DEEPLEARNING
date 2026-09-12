# CrossAttention.py
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from Attention.common.attention import MultiHeadAttention

class CrossAttention(nn.Module):
    '''
    Cross attention module.
    image generate the key and value;
    query is generated from the input sequence.
    '''
    def __init__(self, d_model_image, d_model_seq, d_k):
        super().__init__()
        self.W_k = nn.Linear(d_model_image, d_k)
        self.W_v = nn.Linear(d_model_image, d_k)
        self.W_q = nn.Linear(d_model_seq, d_k)

    def forward(self, x, x_image):
        # x: [batch_size, seq_len, d_model]

        Q = self.W_q(x)
        K = self.W_k(x_image)
        V = self.W_v(x_image)

        # TODO
        # 1. QK^T
        # 2. scale
        # 3. softmax
        # 4. attention @ V
        attention = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5), dim=-1)
        output = torch.matmul(attention, V)

        return output, attention

class MultiHeadCrossAttention(nn.Module):
    def __init__ (self, d_model_img,d_model_seq, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model_seq // num_heads
        assert d_model_seq % num_heads == 0, "d_model_seq must be divisible by num_heads"
        self.W4image = nn.Linear(d_model_img, num_heads * self.d_k * 2)  # For Q, K
        self.W_o = nn.Linear(num_heads * self.d_k, d_model_seq)
        self.W4seq = nn.Linear(d_model_seq, num_heads * self.d_k)  # For seq Q
        # regularization
        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, img, seq):
        # x: [batch_size, seq_len, d_model]
        batch_size, img_len, _ = img.size()
        _2 , seq_len, _ = seq.size()

        # Linear projection to get Q, K, V for all heads
        kv = self.W4image(img)  # [batch_size, img_len, num_heads * d_k * 2]
        kv  = kv.view(batch_size, img_len, self.num_heads, -1)  # [batch_size, img_len, num_heads, 2 * d_k]
        K,V = torch.chunk(kv, 2, dim=-1)  # Each of shape [batch_size, img_len]
        Q = self.W4seq(seq)  # [batch_size， seq_len， num_heads * d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, -1)  # [batch_size， seq_len， num_heads，    d_k]
        # Transpose to get the shape [batch_size, num_heads, seq_len, d_k]
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = K.transpose(1, 2)  # [batch_size, num_heads, img_len, d_k]
        V = V.transpose(1, 2)  # [batch_size, num_heads, img_len, d_k]

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # [batch_size, num_heads, seq_len, img_len]
        
        attention_scores = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, img_len]
        attention_scores = self.attn_dropout(attention_scores)
        
        attention = torch.matmul(attention_scores, V)  # [batch_size, num_heads, seq_len, d_k]
        attention = attention.transpose(1,2)
        attention = attention.reshape(batch_size, seq_len, -1)
        attention = self.W_o(attention)  # [batch_size, seq_len, d_model]
        
        return attention,attention_scores  # [batch_size, num_heads, seq_len, d_k]

if __name__ == "__main__":
    x = torch.randn(2, 5, 192)

    mask = torch.tril(torch.ones(5, 5))

    model = MultiHeadAttention(192, 8)
    output, attention = model(x, mask)

    print(output.shape)
    print(attention.shape)
