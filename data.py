# data loading & aug — ripped-together quick style
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=128, num_workers=4):
    # train transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),    # <-- resize CIFAR10 32×32 → 224×224
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),         # pytorch built-in
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    # test/val transforms
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),    # <-- same resize here
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    train_ds = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    val_ds = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    calib_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, calib_loader