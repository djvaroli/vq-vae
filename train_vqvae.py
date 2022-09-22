from dataclasses import dataclass
import typing as t
from argparse import ArgumentParser
from enum import Enum
import os

import torch
from torchvision.utils import make_grid
import neptune.new as neptune
from neptune.new.types import File
from tqdm import tqdm

from model import VQVAE, VQVAETrainer
from data import CIFAR10DataLoader


NEPTUNE_PROJECT_NAME = os.environ["NEPTUNE_PROJECT_NAME"]
NEPTUNE_API_TOKEN = os.environ["NEPTUNE_API_TOKEN"]


class DataloaderOptions(Enum):
    CIFAR10 = CIFAR10DataLoader


@dataclass
class TrainingParameters:
    dataset_name: str
    n_epochs: int
    device_id: str
    seed: t.Optional[int] = None


def train_vqvae(
    n_epochs: int,
    device_id: str,
    seed: t.Optional[int] = None
):
    device = torch.device(device_id)
    
    print(f'Using device: {device}.')

    run = neptune.init(
        project=NEPTUNE_PROJECT_NAME,
        api_token=NEPTUNE_API_TOKEN
    )

    run_parameters = TrainingParameters(
        dataset_name="CIFAR10",
        n_epochs=n_epochs,
        device_id=device_id,
        seed=seed
    )
    dataloader = DataloaderOptions[run_parameters.dataset_name].value()
    optimizer_factory = lambda params: torch.optim.Adam(params)

    run["parameters"] = run_parameters.__dict__

    model = VQVAE(3)
    trainer = VQVAETrainer(model, optimizer_factory, device=device)

    test_samples = dataloader.get_sample_images(8, seed=run_parameters.seed).to(device)

    test_reconstructions = model(test_samples).detach().cpu()
    test_reconstructions_grid = make_grid(test_reconstructions, nrow=4).permute(1, 2, 0).numpy()

    run["reconstruction-epoch-0"].log(File.as_image(test_reconstructions_grid))

    for epoch in range(1, n_epochs + 1):
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch} / {n_epochs}")
        trainer._train_single_epoch(dataloader, progress_bar, run)

        test_reconstructions = model(test_samples).detach().cpu()
        test_reconstructions_grid = make_grid(test_reconstructions, nrow=4).permute(1, 2, 0).numpy()

        run[f"reconstruction-epoch-{epoch}"].log(File.as_image(test_reconstructions_grid))
    
    run.stop()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--n_epochs", required=True, type=int, help="Number of epochs to train model for.")
    parser.add_argument("-d", "--device_id", required=True, type=str, help="Device to run model training on.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for stochastic operations.")
    args = parser.parse_args()

    train_vqvae(
        n_epochs=args.n_epochs,
        device_id=args.device_id,
        seed=args.seed
    )