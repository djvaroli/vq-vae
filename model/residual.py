from turtle import forward

import torch
from torch import nn


def default_residual_module(
    in_channels: int,
    residual_hidden_layer_channels: int,
) -> nn.Sequential:
    module = nn.Sequential(
        nn.ReLU(True),
        nn.Conv2d(
            in_channels,
            out_channels=residual_hidden_layer_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        nn.ReLU(True),
        nn.Conv2d(residual_hidden_layer_channels, in_channels, kernel_size=1, stride=1, bias=False),
    )
    return module


class Residual(nn.Module):
    def __init__(self, wrapped_module: nn.Module) -> None:
        super().__init__()
        self.wrapped_module = wrapped_module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.wrapped_module(inputs)


class ResidualStack(nn.Module):
    def __init__(
        self,
        n_layers: int,
        residual_module: Residual,
        output_activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.residual_stack = nn.ModuleList([residual_module for _ in range(n_layers)])
        self.output_activation = output_activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for residual_layer in self.residual_stack:
            output = residual_layer(output)
        return self.output_activation(output)
