# train4caption
import torch
import os
import sys
# 让 `import Attention` 生效：把 Attention 的父目录加入 sys.path（不依赖运行 cwd）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))  # 父目录 = 30-，让 import Attention 生效
import torch.nn as nn
from Attention.models.ImageCaptioningModel import ImageCaptioningModel
from Attention.datasets.Flickr8k import get_data_loaders, get_vocab_size, CAPTION_MAX_LEN


def train_models(model,epoch=100,device=None,data_loader = None,data_loader_test = None):
    model = model.to(device)
    PAD_ID = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(epoch):  # example number of epochs
        model.train()
        total_loss = 0
        cur = 0
        total = 0
        for img, cap in data_loader:
            img = img.to(device)
            cap = cap.to(device)
            optimizer.zero_grad()
            cap_in,cap_tar = cap[:, :-1], cap[:, 1:]
            outputs = model(img, cap_in)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), cap_tar.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = outputs.argmax(-1).reshape(-1)
            tgt = cap_tar.reshape(-1)
            real = tgt!=0
            cur += ((pred == tgt)&real).sum()
            total += real.sum()
        print(f"Epoch {epoch+1}, Accuracy: {cur/total if total > 0 else 0}%")
        #scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}%")
        model.eval()
        cur_test = 0
        total_test = 0
        with torch.no_grad():
            for img, cap in data_loader_test:
                img = img.to(device)
                cap = cap.to(device)
                cap_in,cap_tar = cap[:, :-1], cap[:, 1:]
                outputs = model(img, cap_in)
                _, predicted = torch.max(outputs.reshape(-1, outputs.shape[-1]), 1)
                cur_test += ((predicted == cap_tar.reshape(-1))&(cap_tar.reshape(-1)!=0)).sum()
                total_test += (cap_tar.reshape(-1)!=0).sum()

        print(f"Epoch {epoch+1}, Test Accuracy: {cur_test/total_test if total_test > 0 else 0}%")
        print(f"Epoch {epoch+1}%")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }, f"Attention/checkpoints/vit_model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
        vocab_size = get_vocab_size()
        data_loader, data_loader_test = get_data_loaders()
        model = ImageCaptioningModel(vocab_size=vocab_size, max_seq_len=CAPTION_MAX_LEN, patch_size=14,decoder_num = 12)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("using device:", device)
        print("start training;exp0")
        train_models( model, epoch=100, device=device, data_loader=data_loader, data_loader_test=data_loader_test)

