import typing as t

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader



class CIFAR10DataLoader(DataLoader):
    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        n_workers: int = 0,
        pin_memory: bool = True,
        train: bool = True,
        data_root: str = "downloaded_datasets",
        generator=None,
    ):  

        dataset = datasets.CIFAR10(
            root=data_root, 
            train=train, 
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
            ])
        )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=n_workers,
            pin_memory=pin_memory,
            generator=generator
        )

    def get_sample_images(self, n_samples: int = 1, seed: t.Optional[int] = None) -> torch.Tensor:        
        rng = np.random.default_rng(seed=seed)
        n_total_samples = len(self.dataset.data)
        selected_indices = rng.choice(range(n_total_samples), size=n_samples, replace=False)
        sample_image_tensor = torch.from_numpy(self.dataset.data[selected_indices, :, :, :]).float()
        return sample_image_tensor.permute(0, 3, 1, 2)