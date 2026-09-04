# train4vitCls.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch.nn as nn
import torch
from Attention.datasets.cifar10 import get_data_loaders
from Attention.models.Vit import VitionTransformer

def train_models(model,epoch=10,device=None,data_loader = None,data_loader_test = None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(epoch):  # example number of epochs
        model.train()
        total_loss = 0
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader_test:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1}, Test Accuracy: {100 * correct / total}%")


if __name__ == "__main__":
    data_loader, data_loader_test = get_data_loaders()
    model = VitionTransformer(in_channels=3, patch_size=4, d_model=192, num_layers=12, num_heads=3, mlp_ratio=4.0, num_classes=10)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_models( model, epoch=10, device=device, data_loader=data_loader, data_loader_test=data_loader_test)
