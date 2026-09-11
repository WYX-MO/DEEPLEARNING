#Tokenizer.py

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