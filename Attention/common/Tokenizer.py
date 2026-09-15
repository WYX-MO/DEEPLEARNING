#Tokenizer.py

import os
from tokenizers import Tokenizer as Tkr
from tokenizers import models,trainers,pre_tokenizers
import sentencepiece as spm

class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        self.unk_idx = self.word2idx.get('<unk>', None)
    def encode(self, text):
        return [self.word2idx[word] if word in self.word2idx else self.unk_idx for word in text.split() ]

    def decode(self, indices):
        return ' '.join([self.idx2word[idx] for idx in indices if idx in self.idx2word])

def build_vocab(captions, min_freq=1):
    from collections import Counter
    counter = Counter()
    for caption in captions:
        counter.update(caption.split())
    vocab = [word for word, freq in counter.items() if freq >= min_freq]
    vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + vocab  # Add special tokens
    return vocab

# class ZHTokenizer:
#     def __init__(self,vocab):
#         self.tok = Tkr(models.BPE(unk_token = "<unk>"))
#         self.tok.pre_tokenizer = pre_tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
#         self.trainer = trainers.BpeTrainer(vocab_size=len(vocab), special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"])

#     def train_by_hf(self, vocab):
#         self.tok.train_from_iterator(vocab, trainer=self.trainer)
#         self.tok.save("zhtokenizer.json")

#     def encode(self,text):
#         return self.tok.encode(text)

#     def decode(self,indices):
#         return self.tok.decode(indices)

def train_by_spm(corpus_path, vocab_size, model_prefix="zh",
                 normalization_rule_tsv=None):
    """训练中文 BPE 分词器。

    为什么不用 SentencePiece 的默认归一化 "nmt_nfkc"：
    它按 NFKC 把全角标点 ，：！？（） 折叠成 ASCII ,:!?()，而且 decode 无法还原 ——
    结果是生成的中文台词里混进英文标点。

    为什么需要 normalization_rule_tsv：
    SentencePiece 训练时是**按行读取**语料的，换行符 \n 永远进不了训练样本，
    所以它永远学不到 \n。不显式映射的话，编码含换行的文本会把 \n 变成 <unk>
    （实测 26570 次）。nl2space.tsv 只做一件事：把 \n 映射成空格，其余字符保持原样。

    remove_extra_whitespaces=False：
    让连续空白不被压缩。这样换行 \n 变成单个 ▁，而空行 \n\n 变成 ▁▁ ——
    「空行 = 场景结束」这个结构才在 token 层面可区分。
    """
    kwargs = dict(
        input=corpus_path, model_prefix=model_prefix,
        vocab_size=vocab_size, model_type="bpe",
        character_coverage=0.9995,          # 中文关键参数
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3)
    if normalization_rule_tsv:
        kwargs["normalization_rule_tsv"] = normalization_rule_tsv
    spm.SentencePieceTrainer.train(**kwargs)

if __name__ == "__main__":
    corpus_path = "/mnt/data/ML/dl/Attention/data/sanGuo/all.txt"
    model_prefix = "/mnt/data/ML/dl/Attention/data/sanGuo/zh"
    rule_tsv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nl2space.tsv")
    vocab_size = 10000
    train_by_spm(corpus_path, vocab_size=vocab_size, model_prefix=model_prefix,
                 normalization_rule_tsv=rule_tsv)
    print("done")
