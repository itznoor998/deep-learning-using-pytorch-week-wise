"""Small dataset utilities for the repo."""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_mnist_loaders(batch_size=64, val_split=0.2, data_root='data'):
    transform = transforms.ToTensor()
    full = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    val_size = int(len(full) * val_split)
    train_size = len(full) - val_size
    train_data, val_data = random_split(full, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_cifar10_loaders(batch_size=64, data_root='data'):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, None, test_loader
