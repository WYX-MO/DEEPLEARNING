#train.py
# TODO: 参照 ../CNN/train.py 的结构自己写训练入口
# 固定 seed -> 加载数据 -> 建模型 -> 训练循环（train/valid）-> 记录结果到 experiments/experiments.csv
import torch
import os
import sys
sys.path.insert(0,os.path.join( os.path.dirname(os.path.abspath(__file__)),'..','..'))
import torch.nn as nn
from Attention.models.ImageCaptioningModel import ImageCaptioningModel
from Attention.data import get_data_loaders


def train_models(model,epoch=100,device=None,data_loader = None,data_loader_test = None):
    model = model.to(device)
    PAD_ID = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(epoch):  # example number of epochs
        model.train()
        total_loss = 0
        correct = 0
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
            _, predicted = torch.max(outputs.reshape(-1, outputs.shape[-1]), 1)
            total += cap_tar.reshape(-1).size(0)
            correct += (predicted == cap_tar.reshape(-1)).sum().item()

        #scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}, \nTrain Accuracy: {100 * correct / total}%")
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for img, cap in data_loader_test:
                img = img.to(device)
                cap = cap.to(device)
                cap_in,cap_tar = cap[:, :-1], cap[:, 1:]
                outputs = model(img, cap_in)
                _, predicted = torch.max(outputs.reshape(-1, outputs.shape[-1]), 1)
                total += cap_tar.reshape(-1).size(0)
                correct += (predicted == cap_tar.reshape(-1)).sum().item()
        print(f"Epoch {epoch+1}, Test Accuracy: {100 * correct / total}%")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }, f"checkpoints/vit_model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
        data_loader, data_loader_test = get_data_loaders()
        model = ImageCaptioningModel(vocab_size=10000, max_seq_len=32)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("using device:", device)
        print("start training;exp2")
        train_models( model, epoch=100, device=device, data_loader=data_loader, data_loader_test=data_loader_test)

