import torch
from torch import nn

from model.residual import Residual, ResidualStack, default_residual_module


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_layer_channels: int,
        residual_hidden_layer_channels: int,
        n_residual_layers: int,
        hidden_activation_fn: nn.Module = nn.ReLU(),
        out_channels: int = 3,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
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

        self.conv_tr1 = nn.ConvTranspose2d(
            in_channels=hidden_layer_channels,
            out_channels=hidden_layer_channels // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.conv_tr2 = nn.ConvTranspose2d(
            in_channels=hidden_layer_channels // 2,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.hidden_activation_fn = hidden_activation_fn
        self.out_channels = out_channels

    def forward(self, inputs: torch.Tensor):
        outputs = self.conv1(inputs)

        outputs = self.residual_stack(outputs)
        
        outputs = self.conv_tr1(outputs)
        outputs = self.hidden_activation_fn(outputs)

        return self.conv_tr2(outputs)
