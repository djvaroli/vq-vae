import typing as t
from enum import Enum

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader


def _tanh_transform(n_channels: int):
    means = tuple(0.5 for _ in range(n_channels))
    std = tuple(0.5 for _ in range(n_channels))

    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(means, std)])


def _half_step_transform(n_channels: int):
    means = tuple(0.5 for _ in range(n_channels))
    std = tuple(1.0 for _ in range(n_channels))

    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(means, std)])


class ImageDatasetTransforms(Enum):
    TANH_RGB = _tanh_transform(3)
    TANH_BW = _tanh_transform(1)
    HALF_RGB = _half_step_transform(3)
    HALF_BW = _half_step_transform(1)


class CIFAR10DataLoader(DataLoader):
    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        n_workers: int = 0,
        pin_memory: bool = True,
        train: bool = True,
        data_root: str = "downloaded_datasets",
        transforms: t.Optional[ImageDatasetTransforms] = None,
        generator=None,
    ):

        dataset = datasets.CIFAR10(
            root=data_root, train=train, download=True, transform=transforms
        )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=n_workers,
            pin_memory=pin_memory,
            generator=generator,
        )

    def get_sample_images(
        self, n_samples: int = 1, seed: t.Optional[int] = None
    ) -> torch.Tensor:
        iterator = iter(self)
        if seed:
            torch.manual_seed(seed)
        next_batch = next(iterator)[0]
        return next_batch[:n_samples]
