import typing as t
from turtle import forward

import torch
from einops.layers.torch import Rearrange
from torch import nn

from .distance import EuclidianCodebookDistance


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_codes: int,
        code_vector_dim: int,
    ) -> None:
        super().__init__()
        self._n_codes = n_codes
        self._code_vector_dim = code_vector_dim

        self.codebook = nn.Embedding(n_codes, code_vector_dim)
        self.codebook.weight.data.uniform_(-1 / self.n_codes, 1 / self.n_codes)

        self._distance_f = EuclidianCodebookDistance()
        # TODO: where should the commitment cost term go

        self.rearrange_layer = Rearrange("b c h w -> b h w c")

    @property
    def n_codes(self) -> int:
        """Returns the number of code vectors."""
        return self._n_codes

    @property
    def code_vector_dim(self) -> int:
        """Returns the dimension of ecah code vector"""
        return self._code_vector_dim

    def quantize(self, vectors: torch.Tensor) -> torch.Tensor:
        # measure the distance between each input vector and each code vector
        distances = self._distance_f(vectors, self.codebook.weight)

        # gives us index of closest codevector to each input vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (n_x, 1)
        encodings_placeholder = torch.zeros(
            encoding_indices.shape[0], self.n_codes, device=vectors.device
        )
        encodings = encodings_placeholder.scatter(1, encoding_indices, 1)

        # perform the quantization step and reshape to original shape
        quantized_inputs = torch.matmul(encodings, self.codebook.weight)

        return quantized_inputs

    def forward(self, vectors: torch.Tensor):
        # inputs will have shape (B, C, H, W) convert to (B, H, W, C)
        inputs = self.rearrange_layer(vectors).contiguous()
        input_shape = inputs.shape

        # flatten inputs, essentially unrolling into a matrix of vectors
        unrolled_inputs = inputs.view((-1, self.code_vector_dim))

        # perform the quantization step and reshape to original shape
        quantized_inputs = self.quantize(unrolled_inputs).view(input_shape)

        return quantized_inputs
