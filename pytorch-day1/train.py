import torch
import torch.nn as nn
import torch.optim as optim
from datasets.cifar10 import get_data_loaders
from models.CNN import MyCNN

def train_model(model,data_loader,data_loader_test,epoch= 10,learning_rate = 0.01):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("start training")
    for e in range(epoch):
        model.train()
        for i, (images,labels) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(images)
            l = loss(outputs,labels)
            l.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"epoch: {e}, step: {i}, loss: {l.item()}")
if __name__ == "__main__":
    data_loader, data_loader_test = get_data_loaders()
    model = MyCNN()
    train_model(model,data_loader,data_loader_test)