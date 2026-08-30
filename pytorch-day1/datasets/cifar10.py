import os
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']


def get_data_loaders(batch_size=64):
    """加载 CIFAR-10 训练集和测试集，返回 (data_loader, data_loader_test)。"""
    # data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # load CIFAR10 dataset
    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True,
                                     transform=transform, download=False)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False,
                                    transform=transform, download=False)

    data_loader = DataLoader(dataset=train_dataset,
                             batch_size=batch_size, shuffle=True)
    data_loader_test = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size, shuffle=False)
    return data_loader, data_loader_test


if __name__ == "__main__":
    get_data_loaders()
    print("datasets done")
