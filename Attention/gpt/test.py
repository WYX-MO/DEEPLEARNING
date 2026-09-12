# test_gpt.py
# 交互式测试：输入一段 seq，模型输出下一个字符的 top-5 概率，并续写一段文本。
# 用法: python test_gpt.py [--gen 100] [--temperature 1.0] [--greedy]

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn.functional as F
from Attention.gpt.models.GPT import GPT
from Attention.gpt.datasets.shakespeare import ShakespeareDataset

# 必须与 gpt/train.py 保持一致
MAX_SEQ_LEN = 32
D_MODEL = 192
NUM_HEADS = 4
D_FF = 768
NUM_LAYERS = 6

HERE = os.path.dirname(os.path.abspath(__file__))            # Attention/gpt
_ATTN_DIR = os.path.dirname(HERE)                            # Attention
CKPT = os.path.join(_ATTN_DIR, 'checkpoints', 'gpt_model_10.pth')
DATA = os.path.join(_ATTN_DIR, 'data', 'shakespeare.txt')


def load_model():
    # 复用 Dataset 重建词表，保证 char2idx / idx2char 与训练时完全一致
    ds = ShakespeareDataset(DATA, MAX_SEQ_LEN)
    model = GPT(ds.chars, MAX_SEQ_LEN, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
    state = torch.load(CKPT, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model, ds


def encode(ds, seq):
    """字符 -> id，丢弃词表外的字符，并截断到 max_seq_len。"""
    ids = [ds.char2idx[c] for c in seq if c in ds.char2idx]
    if not ids:
        raise ValueError('seq 中没有任何词表内的字符')
    return ids[-MAX_SEQ_LEN:]


@torch.no_grad()
def next_char_probs(model, ds, ids, topk=5):
    x = torch.tensor([ids], dtype=torch.long)
    logits = model(x)[0, -1]                       # [vocab]
    probs = F.softmax(logits, dim=-1)
    vals, idx = probs.topk(topk)
    return [(ds.idx2char[int(i)], float(v)) for v, i in zip(vals, idx)]


@torch.no_grad()
def generate(model, ds, ids, n, temperature=1.0, greedy=False):
    ids = list(ids)
    for _ in range(n):
        x = torch.tensor([ids[-MAX_SEQ_LEN:]], dtype=torch.long)
        logits = model(x)[0, -1]
        if greedy:
            nxt = int(logits.argmax())
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        ids.append(nxt)
    return ''.join(ds.idx2char[i] for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', type=int, default=100, help='续写字符数，0 表示不续写')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--greedy', action='store_true', help='贪心解码（默认按 temperature 采样）')
    ap.add_argument('--seq', type=str, default=None, help='直接给定 seq，非交互模式')
    args = ap.parse_args()

    model, ds = load_model()
    print(f'loaded {CKPT}  |  vocab={ds.chars}  max_seq_len={MAX_SEQ_LEN}')
    print('输入 seq 回车查看输出；直接回车退出。\n')

    while True:
        if args.seq is not None:
            seq = args.seq
        else:
            try:
                seq = input('seq> ')
            except (EOFError, KeyboardInterrupt):
                break
            if not seq.strip():
                break

        try:
            ids = encode(ds, seq)
        except ValueError as e:
            print(f'  ! {e}\n')
            continue

        print('\n  top-5 下一个字符:')
        for ch, p in next_char_probs(model, ds, ids):
            shown = repr(ch)                       # 换行/空格等不可见字符便于查看
            print(f'    {shown:>6}  {p:.4f}')

        if args.gen > 0:
            out = generate(model, ds, ids, args.gen,
                           temperature=args.temperature, greedy=args.greedy)
            print(f'\n  续写 ({args.gen} chars):')
            print('  ' + out.replace('\n', '\n  '))
        print()

        if args.seq is not None:
            break


if __name__ == '__main__':
    main()
