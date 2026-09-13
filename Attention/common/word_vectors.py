# word_vectors.py
# 中文词表 -> 空间向量：把词表里的每个 token(字 / 词 / subword) 映射成 d_model 维稠密向量。
#
# 和 common/text_embedding.py 的区别：
#   TextEmbedding —— GPT 内部的一层，输入一批 id，输出“向量 + 位置编码”；
#   本文件        —— “表”本身：可查、可存、可比相似度，独立于模型结构。
#
# 词表从哪来（三选一）：
#   A. from_sentencepiece(zh.model)  sp 自带 id_to_piece，id 顺序和训练时一致（推荐）
#   B. from_vocab_file(zh.vocab)     逐行读 "piece\tscore"，id = 行号(0 起)
#   C. 直接传 id2piece 列表           自己搭的小词表也能跑
#
# 向量的“语义”从哪来：
#   默认随机初始化 —— 此刻向量只是随机点，语义要靠训练把它们推开/拉近
#   （这正是 nn.Embedding 那层可学习权重的意义）。
#   想让向量一上来就带语义：用 vectors= 载入预训练向量(fastText/GloVe 文本格式)，
#   或者另外单独跑 word2vec 训练。本文件只负责“表”的构建 + 查表 + 相似度，
#   训练 embedding 是 GPT.train.py 的事。

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordVectors(nn.Module):
    """id <-> token 的双向映射 + 一张 [vocab_size, d_model] 的向量表。"""

    def __init__(self, id2piece, d_model, vectors=None):
        super().__init__()
        self.id2piece = list(id2piece)
        self.piece2id = {p: i for i, p in enumerate(self.id2piece)}
        self.vocab_size = len(self.id2piece)
        self.d_model = d_model

        self.embedding = nn.Embedding(self.vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)  # 小随机数，别用全 0
        if vectors is not None:
            hit = self.load_vectors(vectors)
            print(f"载入预训练向量: 命中 {hit}/{self.vocab_size}")

    # ---------- 三种构造入口 ----------
    @classmethod
    def from_sentencepiece(cls, model_path, d_model, vectors=None):
        """读 sp 的 .model；id 顺序由分词器给出，和训练时完全一致。"""
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        id2piece = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        return cls(id2piece, d_model, vectors)

    @classmethod
    def from_vocab_file(cls, vocab_path, d_model, vectors=None):
        """读 .vocab 文本：每行 'piece\\tscore'，id 就是行号(0 起)。"""
        id2piece = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                id2piece.append(line.split('\t')[0])
        return cls(id2piece, d_model, vectors)

    # ---------- 查表 ----------
    def forward(self, ids):
        """ids: [B, L] long -> [B, L, d_model]。直接当 nn.Embedding 用。"""
        return self.embedding(ids)

    def id_of(self, piece):
        """token -> id；不在表里返回 None（别用 [] 硬取，会 KeyError）。"""
        return self.piece2id.get(piece)

    @torch.no_grad()
    def vector(self, piece):
        """token -> 单个向量 [d_model]。"""
        i = self.piece2id.get(piece)
        if i is None:
            raise KeyError(f"词表里没有 {piece!r}")
        return self.embedding.weight[i]

    @torch.no_grad()
    def nearest(self, piece, k=5):
        """余弦相似度找最近邻 —— 这是“空间向量”唯一能被眼睛验证的方式。"""
        i = self.piece2id.get(piece)
        if i is None:
            raise KeyError(f"词表里没有 {piece!r}")

        W = F.normalize(self.embedding.weight, dim=-1)  # 归一化后点积 = 余弦
        sim = W @ W[i]                                  # [vocab_size]
        vals, idx = sim.topk(min(k + 1, self.vocab_size))  # +1 是因为第一个是自己
        out = [(self.id2piece[j], s) for s, j in zip(vals.tolist(), idx.tolist()) if j != i]
        return out[:k]

    # ---------- 预训练向量 / 存取 ----------
    def load_vectors(self, path):
        """外部预训练向量（每行: piece v1 v2 ...）填进表里，命中多少填多少。"""
        hit = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                i = self.piece2id.get(parts[0])
                if i is None or len(parts) - 1 != self.d_model:
                    continue
                vec = torch.tensor([float(v) for v in parts[1:]], dtype=torch.float)
                with torch.no_grad():
                    self.embedding.weight[i] = vec
                hit += 1
        return hit

    def save(self, path):
        torch.save({'id2piece': self.id2piece,
                    'd_model': self.d_model,
                    'weight': self.embedding.weight.detach().cpu()}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location='cpu')
        obj = cls(ckpt['id2piece'], ckpt['d_model'])
        with torch.no_grad():
            obj.embedding.weight.copy_(ckpt['weight'])
        return obj


if __name__ == "__main__":
    # 自检：优先用仓库里的 sp 模型；没有就手搓几个 token，保证这段能跑通
    SPM = Path(__file__).resolve().parents[2] / 'zh.model'   # 30-/zh.model

    if SPM.exists():
        wv = WordVectors.from_sentencepiece(SPM, d_model=192)
        print("来源:", SPM)
    else:
        wv = WordVectors(['<pad>', '<unk>', '曹', '操', '诸', '葛', '亮', '三', '国'], d_model=192)
        print("来源: 内置小词表（没找到 zh.model）")

    print("vocab:", wv.vocab_size, "| d_model:", wv.d_model)

    probe = '曹' if '曹' in wv.piece2id else wv.id2piece[-1]
    print("查表:", repr(probe), "-> id", wv.id_of(probe))

    v = wv.vector(probe)
    print("向量:", tuple(v.shape), v.dtype, "| 前 5 维:", [round(t, 3) for t in v[:5].tolist()])

    ids = torch.tensor([[wv.id_of(probe), 1]])
    print("forward:", tuple(wv(ids).shape), "  # [B, L, d_model]")

    # 注意：随机初始化的向量之间相似度没有意义，这一步只是验证代码通路
    print("最近邻(随机初始化, 仅演示代码):", wv.nearest(probe, k=3))
