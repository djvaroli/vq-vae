from turtle import forward
import torch
from torch import nn

from .codebook import VectorQuantizer
from .encoder import Encoder
from .decoder import Decoder


class ConvVQVAE(nn.Module):
    def __init__(
        self,
        img_channels: int,
        n_codes: int = 512,
        code_vector_dim: int = 64,
        encoder_latent_dim: int = 128,
        encoder_residual_latent_dim: int = 32,
        encoder_n_residual_layers: int = 2,
        encoder_hidden_activation_fn: nn.Module = nn.ReLU(),
        decoder_latent_dim: int = 128,
        decoder_residual_latent_dim: int = 32,
        decoder_n_residual_layers: int = 2,
        decoder_hidden_activation_fn: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            img_channels,
            encoder_latent_dim,
            encoder_residual_latent_dim,
            encoder_n_residual_layers,
            encoder_hidden_activation_fn
        )
        self.pre_codebook_conv = nn.Conv2d(
            in_channels=encoder_latent_dim, 
            out_channels=code_vector_dim,
            kernel_size=1, 
            stride=1
        )
        self.quantizer = VectorQuantizer(
            n_codes, code_vector_dim
        )
        self.decoder = Decoder(
            code_vector_dim,
            decoder_latent_dim,
            decoder_residual_latent_dim,
            decoder_n_residual_layers,
            decoder_hidden_activation_fn,
            out_channels=img_channels
        )
    
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pre_codebook_conv(self.encoder(inputs))
    
    def quantize(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.quantizer(inputs)
    
    def decode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(inputs)
    
    def forward(self, inputs: torch.Tensor):
        encoded = self.encode(inputs)
        quantized = self.quantize(encoded)
        return self.decode(quantized)
    