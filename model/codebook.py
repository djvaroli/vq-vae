import typing as t

import torch
from torch import nn

from .distance import EuclidianCodebookDistance


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        distance_fn: nn.Module = EuclidianCodebookDistance()
    ):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings

        self.codebook = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(
            -1 / self.n_embeddings, 1 / self.n_embeddings
        )
        self._distance_fn = distance_fn

    def compute_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Compute a matrix of pair-wise distances between vectors in tensor t1 and tensor t2.

        Returns:
            torch.Tensor: matrix of pairwise distances of shape (n_vec_t1, n_vec_t2).
        """
        return self._distance_fn(t1, t2)

    def forward(self, inputs: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = self.compute_distance(flat_input, self.codebook.weight)

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.n_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)

        return quantized, encodings
