#GPT.py

import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..'))
from Attention.gpt.models.GPT_block import GPTBlock
from Attention.common.text_embedding import TextEmbedding
class GPT(nn.Module):
    def __init__(self, vocab_size,max_seq_len,d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.text_embeding = TextEmbedding(vocab_size, d_model, max_seq_len)
        self.layers = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, mask=None):
        B,T = x.shape
        x = self.text_embeding(x)
        # 因果掩码：1 = 允许看（含自己），0 = 屏蔽未来。不传就自己造。
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.lm_head(x)  # [batch_size, seq_len, vocab_size]

    @torch.no_grad()
    def generator(self,idx,max_new_len,temperature=0.7):
        for _ in range(max_new_len):
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits/temperature, dim=-1)

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat([idx, next_token], dim=1)

        return idx

    @torch.no_grad()
    def generator_cache(self, idx, max_new_len, temperature=0.7):
        # idx: [B, T] prompt序列
        cache_k = None
        cache_v = None

        for _ in range(max_new_len):
            if cache_k is None:
                # Prefill阶段：第一次，输入完整prompt片段，生成初始KV cache
                idx_cond = idx[:, -self.max_seq_len:]
                logits, cache_k, cache_v = self(idx_cond, k_cache=cache_k, v_cache=cache_v)
            else:
                # Decode阶段：只取上一步最后1个token输入，不再传整段序列！
                idx_cond = idx[:, -1:]
                logits, cache_k, cache_v = self(idx_cond, k_cache=cache_k, v_cache=cache_v)

            # 只取最后一个token的logits
            logits = logits[:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

if __name__ == "__main__":
    model = GPT(
        vocab_size=10000,
        max_seq_len=64,
        d_model=192,
        num_heads=4,
        d_ff=768,
        num_layers=6
    )
    x = torch.randint(0, 10000, (2, 32))
    output = model(x)
    print(x.shape)
    print(output.shape)