import torch
import torch.nn as nn
import torch.optim as optim
from datasets.cifar10 import get_data_loaders
from models.CNN import MyCNN
import torchvision.transforms as transforms

def image_augmentation(image):
    # horizontal flip
    transforms.RandomApply(transforms=[transforms.RandomHorizontalFlip(p=1)], p=0.5)(image)
    # random rotation
    transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=15)], p=0)(image)
    # random crop
    transforms.RandomApply(transforms=[transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0))], p=0)(image)
    # color jitter
    transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0)(image)
    return image

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
            # apply image augmentation
            images = image_augmentation(images)

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
        print(f"epoch: {e}, test accuracy: {acc / total}")

    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": l.item(),
                }, "baseline_cnn.pth")
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader, data_loader_test = get_data_loaders()
    model = MyCNN()
    model.to(device)
    print("model to device:", device)
    train_model(model,data_loader,data_loader_test,device=device,epoch=10,learning_rate=0.001)