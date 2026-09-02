# CIFAR-10 Image Classification from Scratch (PyTorch)

## Project Goal

Implement CNN and ResNet from scratch using PyTorch, and study the effect of Data Augmentation, BatchNorm, and Residual Connections.

## Model Evolution

CNN → Data Augmentation → BatchNorm → ResNet

## Experimental Results

| Model                                | Test Accuracy |
| ------------------------------------ | ------------- |
| CNN                                  | 73.98%        |
| CNN + Augmentation                   | 78.17%        |
| CNN + BatchNorm + Augmentation       | 80.33%        |
| ResNet-18 + BatchNorm + Augmentation | 85.93%        |

## Key Learnings

* Implemented CIFAR-10 Dataset pipeline.
* Implemented CNN manually.
* Implemented BatchNorm into CNN.
* Implemented ResidualBlock and ResNet-18 manually.
* Compared different training strategies through controlled experiments.
