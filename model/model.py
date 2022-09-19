import typing as t

import torch
from torch import nn

from .codebook import VectorQuantizer
from .encoder import Encoder
from .decoder import Decoder


class VQVAE(nn.Module):
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
            encoder_hidden_activation_fn,
        )
        self.pre_codebook_conv = nn.Conv2d(
            in_channels=encoder_latent_dim,
            out_channels=code_vector_dim,
            kernel_size=1,
            stride=1,
        )
        self.quantizer = VectorQuantizer(n_codes, code_vector_dim)
        self.decoder = Decoder(
            code_vector_dim,
            decoder_latent_dim,
            decoder_residual_latent_dim,
            decoder_n_residual_layers,
            decoder_hidden_activation_fn,
            out_channels=img_channels,
        )

        self._is_compiled = False
        self._commitment_loss_fn: nn.Module
        self._reconstruction_loss_fn: nn.Module

    @property
    def n_codes(self) -> int:
        return self.quantizer.n_codes

    @property
    def code_vector_dim(self) -> int:
        return self.quantizer.code_vector_dim

    @property
    def codebook_vectors(self) -> torch.Tensor:
        return self.quantizer.codebook_vectors

    def get_code_vectors(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(self.codebook_vectors.device)
        return torch.index_select(self.codebook_vectors, dim=0, index=indices)

    def get_random_code_vectors(
        self, n_code_vectors: int, seed: t.Optional[int] = None
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        code_vector_indices = torch.randint(0, self.n_codes, (n_code_vectors,)).int()
        code_vectors = self.get_code_vectors(code_vector_indices)

        return code_vectors

    def _index(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        flat_inputs = inputs.view((-1, self.code_vector_dim))
        code_vector_indices = self.quantizer.index_vectors(flat_inputs)
        return code_vector_indices.view((batch_size, -1))

    def index(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(inputs)
        return self._index(encoded)

    def encode(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        pre_quantized = self.pre_codebook_conv(self.encoder(inputs))
        return pre_quantized

    def encode_and_quantize(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.quantize(self.encode(inputs))

    def quantize(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.quantizer(inputs)

    def decode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(inputs)

    def forward(self, inputs: torch.Tensor):
        encoded = self.encode(inputs)
        quantized = self.quantize(encoded)
        return self.decode(quantized)
