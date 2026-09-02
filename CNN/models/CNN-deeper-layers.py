#CNN-deeper-layers.py
#this model's structure is deeper than the previous one, and it is the same as the resnet one,
#EXCEPT that its blocks have NO residual shortcut.
#goal: compare the two models and prove that the residual block is better than the normal one
import torch
import torch.nn as nn
from torch.nn.functional import avg_pool2d


class MyPlainBlock(nn.Module):
    """normal block: same two convs as MyResidualBlock, but WITHOUT the x += residual shortcut"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(MyPlainBlock, self).__init__()
        #conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, 5, stride=stride, padding=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 5, stride=1, padding=2)
        #batch normalization
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # no downsample / no shortcut here (stride is handled by conv1 itself)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return torch.relu(x)


class MyDepperCNN(nn.Module):
    # structure mirrors ResNet.py exactly (stem + 4 stages + global avg pool + linear),
    # but stacks MyPlainBlock instead of MyResidualBlock
    def __init__(self, block, num_blocks, num_classes=10):
        super(MyDepperCNN, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    model = MyDepperCNN(MyPlainBlock, [2, 2, 2, 2])
    print(model(torch.randn(2, 3, 32, 32)).shape)  # expect torch.Size([2, 10])
