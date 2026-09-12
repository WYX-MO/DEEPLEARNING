# text_embedding.py

import torch
import torch.nn as nn

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
    def forward(self, x,rope = False):
        batch_size, seq_len = x.shape
        if rope == False:
            position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).to(x.device)
            emd = self.embedding(x) + self.position_embedding(position_ids)
        else:
            emd = self.embedding(x)
        return emd

if __name__ == "__main__":
    embedding = TextEmbedding(
    vocab_size=10000,
    d_model=192,
    max_seq_len=64
    )

    x = torch.randint(0, 10000, (2, 10))

    output = embedding(x)

    print(x.shape)
    print(output.shape)

