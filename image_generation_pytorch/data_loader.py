from typing import Literal
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader


def get_loader(
    data_path: str,
    batch_size: int,
    mode: Literal["train", "test", "sample"],
    num_workers: int = 4,
) -> DataLoader:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=mode == "train", download=True, transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=mode != "train", num_workers=num_workers
    )
    return dataloader
