#gptGenerator.py

import torch
import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
from Attention.models.GPT import GPT
from Attention.models.Tokenizer import Tokenizer
from Attention.datasets.shakespeare import ShakespeareDataset

HERE = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(HERE, 'data', 'shakespeare.txt')

# 必须与 train4gpt.py 保持一致
MAX_SEQ_LEN = 32
D_MODEL = 192
NUM_HEADS = 4
D_FF = 768
NUM_LAYERS = 6


if __name__ == "__main__":

    ds = ShakespeareDataset(file_path,MAX_SEQ_LEN)
    vocab_size = ds.chars
    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS
    )

    model.eval()
    model.load_state_dict(torch.load("gpt_model_10.pth"))
    prompt = input("请输入提示：")
    ids = [ds.char2idx[c] for c in prompt if c in ds.char2idx] 
    ids = torch.tensor(ids)
    generated = model.generator(ids,max_new_len=100,temperature=0.5)
    out = "".join([ds.idx2char[_] for _ in generated[0].tolist()])
    print(out)
