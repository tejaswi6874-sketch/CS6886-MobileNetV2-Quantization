"""CIFAR-10 transforms and dataloaders used by the assignment."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_cifar10_loaders(root="./data", batch_size=128):
    """Return training and test loaders matching the notebook preprocessing."""
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform())
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def get_calibration_loader(root="./data", batch_size=128):
    """Deterministic calibration loader: no augmentation and no shuffling."""
    calibration_set = datasets.CIFAR10(
        root=root, train=True, download=True, transform=eval_transform()
    )
    return DataLoader(calibration_set, batch_size=batch_size, shuffle=False, num_workers=0)
