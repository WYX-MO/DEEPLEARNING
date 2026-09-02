import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
from torchvision import datasets
import torch.nn as nn

from datasets.cifar10 import get_data_loaders

data_loader, data_loader_test = get_data_loaders()

# 验证 data_loader 可用：取一个 batch 看看形状
for images, labels in data_loader:
    print("batch shape:", images.shape, labels.shape)
    break

batch_size = 64


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        #conv
        self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        #batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(128*4*4, 32*4*4)
        self.linear2 = nn.Linear(32*4*4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.linear1(x.view(-1, 128 * 4 * 4))
        x = torch.relu(x)
        x = self.linear2(x)
        return x

model = MyCNN()
print(model(torch.randn(batch_size, 3, 32, 32)).shape)  # 测试模型输出形状  
