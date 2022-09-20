from ast import parse
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


class DataloaderOptions(Enum):
    CIFAR10 = CIFAR10DataLoader


def train_vqvae(
    n_epochs: int
):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0")
    else:
        device = torch.device("cpu")
    
    print(f'Using device: {device}.')

    project_name = os.environ["NEPTUNE_PROJECT_NAME"]
    api_token = os.environ["NEPTUNE_API_TOKEN"]
    run = neptune.init(
        project=project_name,
        api_token=api_token
    )

    dataloader = DataloaderOptions.CIFAR10.value()
    optimizer_factory = lambda params: torch.optim.Adam(params)
    model = VQVAE(3)
    trainer = VQVAETrainer(model, optimizer_factory, device=device)

    test_samples = dataloader.get_sample_images(8, seed=1000).to(device)
    test_reconstructions = model(test_samples).detach().cpu()
    test_reconstructions_grid = make_grid(test_reconstructions, nrow=4).permute(1, 2, 0).numpy()

    run["reconstruction-epoch-0"].log(File.as_image(test_reconstructions_grid))

    for epoch in range(1, n_epochs + 1):
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch} / {n_epochs}")
        trainer._train_single_epoch(dataloader, progress_bar)

        test_reconstructions = model(test_samples).detach().cpu()
        test_reconstructions_grid = make_grid(test_reconstructions, nrow=4).permute(1, 2, 0).numpy()

        run[f"reconstruction-epoch-{epoch}"].log(File.as_image(test_reconstructions_grid))
    
    run.stop()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--n_epochs", required=True, type=int, help="Number of epochs to train model for.")
    args = parser.parse_args()

    train_vqvae(n_epochs=args.n_epochs)