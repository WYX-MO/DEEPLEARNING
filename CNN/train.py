import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets.cifar10 import get_data_loaders
from models.CNN import MyCNN
from models.ResNet import My_resnet
from models.ResidualBlock import MyResidualBlock
import torchvision.transforms as transforms

import random

seed = 42

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


def train_model(model,data_loader,data_loader_test,device,epoch= 10,learning_rate = 0.01):
    # loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("start training")

    max_acc_epoch =0
    for e in range(epoch):
        model.train()
        acc = 0
        total =0
        l_total = 0
        for i, (images,labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            pred = torch.argmax(outputs,dim=1)
            #conculate acc
            acc+= torch.sum(pred==labels).item()
            total+= labels.size(0)
            #backward
            l = loss(outputs,labels)
            l_total += l.item()
            l.backward()
            optimizer.step()
        #if i % 100 == 0:
        print(f"epoch: {e}, step: {i}, loss: {l_total }")
        print(f"train accuracy: {acc / total}") 
        model.eval()
        acc = 0
        total = 0

        # test accuracy
        with torch.no_grad():
            for i, (images,labels) in enumerate(data_loader_test):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                pred = torch.argmax(outputs,dim=1)
                acc+= torch.sum(pred==labels).item()
                total+= labels.size(0)
        print(f"test accuracy: {acc / total}")

        if acc / total > max_acc_epoch:
            max_acc_epoch = acc / total
            # torch.save({"model_state_dict": model.state_dict(),
            #     "optimizer_state_dict": optimizer.state_dict(),
            #     "epoch": e,
            #     "loss": l.item(),
            #     }, "checkpoints/augmentation.pth")
            # print(f"model saved at epoch {e} with best test accuracy: {max_acc_epoch}")

    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": l.item(),
                }, f"augmentation{epoch}.pth")
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader, data_loader_test = get_data_loaders()
    model1 = MyCNN()
    model1.to(device)
    model_resnet = My_resnet(MyResidualBlock, [2, 2, 2, 2])
    model_resnet.to(device)
    print("model to device:", device)
    print("exp:2")
    train_model(model1,data_loader,data_loader_test,device=device,epoch=10,learning_rate=0.001)