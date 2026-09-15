#gptGenerator.py

import torch
import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
from Attention.gpt.models.GPT import GPT
from Attention.gpt.models.modern_gpt import ModernGPT
from Attention.common.Tokenizer import Tokenizer
from Attention.gpt.datasets.shakespeare import ShakespeareDataset
from Attention.gpt.datasets.sanguo import SanGuoDataset
import sentencepiece as spm
HERE = os.path.dirname(os.path.abspath(__file__))            # Attention/gpt
_ATTN_DIR = os.path.dirname(HERE)                            # Attention
CKPT = os.path.join(_ATTN_DIR, 'checkpoints', 'gpt_sanguo_model_99.pth')
file_path = os.path.join(_ATTN_DIR, 'data', 'sanGuo','all.txt')
SPM_MODEL = os.path.join(_ATTN_DIR , 'data' , 'sanGuo' ,'zh.model' )     


# 必须与 gpt/train.py 保持一致
MAX_SEQ_LEN = 128
D_MODEL = 192
NUM_HEADS = 4
D_FF = 768
NUM_LAYERS = 6


if __name__ == "__main__":

    ds = ShakespeareDataset(file_path,MAX_SEQ_LEN)

    sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL))
    ds_sanguo = SanGuoDataset(file_path,MAX_SEQ_LEN,sp)
    vocab_size = ds_sanguo.vocab_size
    model = ModernGPT(
        vocab_size=vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS
    )

    model.eval()
    model.load_state_dict(torch.load(CKPT, map_location='cpu'))
    prompt = input("请输入提示：")
    #ids = [ds.char2idx[c] for c in prompt if c in ds.char2idx] 
    ids = sp.EncodeAsIds(prompt)
    ids = ids[:MAX_SEQ_LEN]
    ids = torch.tensor(ids).unsqueeze(0)
    generated = model.generator(ids,max_new_len=100,temperature=0.5)
    #out = "".join([ds.idx2char[_] for _ in generated[0].tolist()])
    out = sp.decode(generated[0].tolist())
    out = "".join(out)
    
    print(out)
