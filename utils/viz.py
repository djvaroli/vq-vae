import typing as t

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()


def plot_images_in_grid(
    images: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    figure_size: t.Tuple[int, int] = (10, 12)
):  
    # starts out in (B, C, H0, W0), conver to (H1, W1, C)
    grid = make_grid(images, nrow, padding, normalize).permute(1, 2, 0)
    grid_np = grid.numpy()

    figure = plt.figure(figsize=figure_size)
    plt.grid(None)
    plt.imshow(grid_np)
    