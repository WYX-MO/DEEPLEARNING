#shakespeare.py

from torch.utils.data import Dataset
import torch
import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..' ))

class ShakespeareDataset(Dataset):
    def __init__(self,text_file,max_seq_len):
        with open(text_file,'r') as f:
            self.text = f.read()
        self.max_seq_len = max_seq_len
        self.vocab = sorted(set(self.text))
        self.chars = len(self.vocab)
        self.char2idx = {ch:i for i,ch in enumerate(self.vocab)}
        self.idx2char = {i:ch for i,ch in enumerate(self.vocab)}
        self.data = [self.char2idx[ch] for ch in self.text]

    def __len__(self):
        return len(self.data) - self.max_seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.max_seq_len],dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.max_seq_len+1],dtype=torch.long)
        return x,y

if __name__ == "__main__":

    dataset = ShakespeareDataset(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'shakespeare.txt'),
        max_seq_len=32
    )

    x, y = dataset[0]

    print("dataset length:", len(dataset))
    print("x shape:", x.shape)
    print("y shape:", y.shape)

    print("x:", x)
    print("y:", y)