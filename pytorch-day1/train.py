import torch
import torch.nn as nn
import torch.optim as optim
from datasets.cifar10 import get_data_loaders
from models.CNN import MyCNN


def train_model(model,data_loader,data_loader_test,device,epoch= 10,learning_rate = 0.01):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("start training")
    for e in range(epoch):
        model.train()
        acc = 0
        total =0
        for i, (images,labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            pred = torch.argmax(outputs,dim=1)
            acc+= torch.sum(pred==labels).item()
            total+= labels.size(0)
            l = loss(outputs,labels)
            l.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"epoch: {e}, step: {i}, loss: {l.item()}")
                print(f"accuracy: {acc / total}")
                acc = 0
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader, data_loader_test = get_data_loaders()
    model = MyCNN()
    model.to(device)
    print("model to device:", device)
    train_model(model,data_loader,data_loader_test,device=device,epoch=10,learning_rate=0.001)