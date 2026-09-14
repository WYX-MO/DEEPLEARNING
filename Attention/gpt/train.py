#train4gpt
import os
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # Attention/gpt
_ATTN_DIR = os.path.dirname(_THIS_DIR)                       # Attention
sys.path.insert(0, os.path.dirname(_ATTN_DIR))               # 30-
CKPT_DIR = os.path.join(_ATTN_DIR, 'checkpoints')
import torch
import logging
from Attention.gpt.datasets.shakespeare import ShakespeareDataset
from Attention.gpt.datasets.sanguo import SanGuoDataset
from Attention.gpt.models.GPT import GPT
from Attention.gpt.models.modern_gpt import ModernGPT
from torch.utils.data import DataLoader
from logger import get_logger
import sentencepiece as spm
from pathlib import Path
import subprocess

_dl = Path(__file__).resolve().parents[2]
SPM_MODEL = _dl / 'Attention' / 'data' / 'sanGuo' / 'zh.model'      


def train_model(model, epochs, train_loader,val_loader,learning_rate, device):
    logger = get_logger()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        
        for  b_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            ppl = 0.0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
                total_loss += loss.item()
            avg_loss = total_loss / len(val_loader)
            logger.info(f"Validation Loss: {avg_loss:.4f}, ppl: {torch.exp(torch.tensor(avg_loss)).item():.4f}")
        if epoch%10 == 0:
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"gpt_sanguo_model_{epoch}.pth"))

def shape_test(model,device,test = False):
    # x : [batch_size, seq_len, d_model]
    x = torch.randint(0, model.vocab_size, (2, 5)).to(device)
    print("shapetest using device:",device)
    print(f"x.shape: {x.shape}")
    y = model(x)
    print(f"y.shape: {y.shape}")
    print("shape test done")
    if test:
        sys.exit(0)

if __name__ == "__main__":

    
    # Hyperparameters
    max_seq_len = 128 #单次输入最大tokens数
    d_model = 192   #词向量维度
    num_heads = 4
    d_ff = 768  #ffn隐藏层维度
    num_layers = 6 #gpt层数
    batch_size = 64
    epochs = 100
    learning_rate = 1e-4

    # Prepare dataset and dataloader
    data_path_shakes = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","data", "shakespeare.txt")
    dataset_shakes = ShakespeareDataset(data_path_shakes, max_seq_len)

    #sanGuo dataset
    data_path_sanguo = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data","sanGuo","all.txt")
    sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL))
    dataset_sanguo = SanGuoDataset(data_path_sanguo,max_seq_len,sp)


    dataset = dataset_sanguo
    #build valid dataset
    train_size = int(0.9*len(dataset))
    val_size = len(dataset)-train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
    )

    vocab_size = dataset.vocab_size  # Adjust based on dataset
    train_loader = DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset= val_dataset,batch_size = batch_size,shuffle=False)
    # Initialize model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModernGPT(vocab_size, max_seq_len, d_model, num_heads, d_ff, num_layers).to(device)
    shape_test(model,device,False)

    print(f"Using device: {device}")
    print("start training;exp1")
    train_model(model, epochs, train_loader,val_loader ,learning_rate, device)
    #subprocess.run(["shutdown","-h","now"],check=True)