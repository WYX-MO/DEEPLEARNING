# Attention/gpt/datasets/sanguo.py
# 三国演义 语料 -> 语言模型训练样本 (x, y)。
#
# 流程（和 gpt/datasets/shakespeare.py 是同一套，只是把“建字表+查表”换成 sentencepiece 分词器）：
#   整段文本 --sp.encode--> 一长串 id --切成定长窗口--> (x, y)
#   x = ids[i     : i+L]
#   y = ids[i + 1 : i + L + 1]   # 右移一位 = “预测下一个 token”
#
# 前置条件：
#   1) 语料文件 Attention/data/sanguo.txt（UTF-8，全文，每回可另起一行）
#   2) 一个能用的 sentencepiece 模型（下面 SPM_MODEL 指向的路径）
#      —— 注意：仓库里现有的 30-/zh.model 是用单章语料训的，实测 27% unk，
#         换成正式训练的词表再指过来。

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

# ---- 路径 ----
# sanguo.py 位于 Attention/gpt/datasets/，往上 3 层 = Attention
_ATTN = Path(__file__).resolve().parents[2]
CORPUS = _ATTN / 'data' / 'sanguo.txt'      # 原始语料
SPM_MODEL = _ATTN.parent / 'zh.model'       # 分词器（30-/zh.model，可改成新训的）


class SanGuoDataset(Dataset):
    """把三国演义全文编码成 id 序列，切成 (输入, 目标) 定长窗口。"""

    def __init__(self, corpus_path, seq_len, sp):
        # 1) 读全文（中文语料基本是 UTF-8）
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 2) 整段文本只 encode 一次 -> 直接存成 long tensor。
        #    之后切片 data[i:i+L] 出来就已经是 tensor，不必再转。
        ids = sp.encode(text)
        self.data = torch.tensor(ids, dtype=torch.long)

        # 3) GPT 用得到的词表大小（必须用分词器给的，不能再用 len(set(text))）
        self.vocab_size = sp.get_piece_size()

        self.seq_len = seq_len
        self.sp = sp

    def __len__(self):
        # 目标要右移一位，最后一个窗口会越界，所以减 seq_len
        return len(self.data) - self.seq_len

    def __getitem__(self, i):
        x = self.data[i     : i + self.seq_len]        # [L]  long
        y = self.data[i + 1 : i + 1 + self.seq_len]    # [L]  long
        return x, y

    def decode(self, ids):
        """id -> 文本，方便肉眼检查样本。"""
        return self.sp.decode(ids)


if __name__ == "__main__":
    # 自检：跑通说明 语料 + 分词器 + 切窗 三件事都对
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL))
    ds = SanGuoDataset(CORPUS, seq_len=128, sp=sp)

    print("corpus :", CORPUS)
    print("spm    :", SPM_MODEL)
    print("vocab  :", ds.vocab_size, "| 样本数:", len(ds))

    x, y = ds[0]
    print("x:", tuple(x.shape), x.dtype)
    print("y:", tuple(y.shape), y.dtype)
    print("x 解码:", ds.decode(x.tolist())[:60])
    print("y 解码:", ds.decode(y.tolist())[:60])
