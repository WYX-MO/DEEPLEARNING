#Tokenizer.py

from tokenizers import Tokenizer,models,trainers,pre_tokenizers
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

class ZHTokenizer:
    def __init__(self,vocab):
        self.tok = Tokenizer(models.BPE(unk_token = "<unk>"))
        self.tok.pre_tokenizer = pre_tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.trainer = trainers.BpeTrainer(vocab_size=len(vocab), special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"])

    def train_by_hf(self, vocab):
        self.tok.train_from_iterator(vocab, trainer=self.trainer)
        self.tok.save("zhtokenizer.json")

def train_by_spm(corpus_path, vocab_size):
    spm.SentencePieceTrainer.train(                                         
    input=corpus_path, model_prefix="zh",                              
    vocab_size=vocab_size, model_type="bpe",                                 
    character_coverage=0.9995,          # 中文关键参数                  
    pad_id=0, unk_id=1, bos_id=2, eos_id=3)   

def build_vocab_zh(vocab_path):
    with open("vocab_path",'w') as f:
        pass

if __name__ == "__main__":
    corpus_path = "/mnt/data/ML/dl/Attention/data/sanGuo/cap1.txt"
    vocab_size = 5770
    train_by_spm(corpus_path, vocab_size=vocab_size)
    print("done")