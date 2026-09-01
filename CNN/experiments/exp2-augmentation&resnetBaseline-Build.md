# Baseline CNN

## 1. Experiment Goal

通过augmentation例如horizontal flip和rotation等等来提高泛化，防止过拟合
添加batchnormal层，查看准确率
搭建resnet的baseline，查看准确率差异

## 2. Environment

- Python:
- PyTorch:
- torchvision:
- Device: CUDA / CPU
- GPU:

## 3. Dataset

- Dataset: CIFAR-10
- Train samples: 50,000
- Test samples: 10,000
- Image size: 32 × 32
- Channels: 3
- Classes: 10

## 4. Data Preprocessing

```python
#train
transform_train = transforms.Compose([
        # image augmentation apply here
        transforms.RandomApply(transforms=[transforms.RandomHorizontalFlip(p=1)], p=0.5),
        transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=15)], p=0.5),    
        transforms.RandomApply(transforms=[transforms.RandomCrop(32, padding=4)], p=0.5),  
        transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),    

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])



#test：
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])
```
## 5. Model
CNN Architecture

Input: 3 × 32 × 32

- Conv2d: 3 → 32, kernel=5, padding=2
- ReLU
- MaxPool2d: 2 × 2
- Conv2d: 32 → 64, kernel=5, padding=2
- ReLU
- MaxPool2d: 2 × 2
- Conv2d: 64 → 128, kernel=5, padding=2
- ReLU
- MaxPool2d: 2 × 2
- Flatten
- Linear: 2048 → 512
- ReLU
- Linear: 512 → 10

## 6. Training Configuration
Batch size: 64
Epochs: 10
Optimizer: Adam
Learning rate: 0.001
Loss: CrossEntropyLoss

## 7. Results
Metric	Result
- Train Accuracy	~95%
- Test Accuracy	73.98%
- Final Loss	~0.35

## 8. Observation

Training accuracy is significantly higher than test accuracy,
indicating that the model has some degree of overfitting.

## 9. Next Experiment
Add data augmentation
Compare test accuracy
Keep the baseline unchanged for comparison


