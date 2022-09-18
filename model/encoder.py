from ctypes.wintypes import HDC
from turtle import forward

import torch
from torch import nn

from model.residual import Residual, ResidualStack, default_residual_module


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_layer_channels: int,
        residual_hidden_layer_channels: int,
        n_residual_layers: int,
        hidden_activation_fn: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_layer_channels // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_layer_channels // 2,
            out_channels=hidden_layer_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            in_channels=hidden_layer_channels,
            out_channels=hidden_layer_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.residual_module = Residual(
            wrapped_module=default_residual_module(
                in_channels=hidden_layer_channels,
                residual_hidden_layer_channels=residual_hidden_layer_channels
            )
        )
        self.residual_stack = ResidualStack(n_residual_layers, self.residual_module)
        self.hidden_activation_fn = hidden_activation_fn

    def forward(self, inputs: torch.Tensor):
        outputs = self.conv1(inputs)
        outputs = self.hidden_activation_fn(outputs)

        outputs = self.conv2(outputs)
        outputs = self.hidden_activation_fn(outputs)

        outputs = self.conv3(outputs)
        return self.residual_stack(outputs)
