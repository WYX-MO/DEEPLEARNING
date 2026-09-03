#ResidualBlock.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
from torchvision import datasets
import torch.nn as nn

class MyResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(MyResidualBlock, self).__init__()
        #conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        #batch normalization
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride !=1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        '''
        conv5x5
        conv5x5
        '''
        residual =x
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        residual = self.downsample(residual) if self.downsample is not None else residual
        x += residual
        return torch.relu(x)

# model = MyResidualBlock(3, 32)
# print(model(torch.randn(batch_size, 3, 32, 32)).shape)  # 测试模型输出形状  
