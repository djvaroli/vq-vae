import typing as t

import torch 
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .loss import QuantizationLoss
from .model import VQVAE


class VQVAETrainer:
    def __init__(
        self,
        model: VQVAE,
        optimizer_factory: t.Callable[[t.Iterator[nn.Parameter]], torch.optim.Optimizer],
        quantization_loss_fn: nn.Module = QuantizationLoss(0.25),
        reconstruction_loss_fn: nn.Module = torch.nn.MSELoss(),
        dataset_variance: t.Optional[float] = 1.,
        metric_update_steps: int = 25,
        device: t.Optional[torch.device] = None
    ) -> None:
        self.model = model
        self.optimizer: torch.optim.Optimizer = optimizer_factory(model.parameters())
        self.quantization_loss_fn = quantization_loss_fn
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.metric_update_steps = metric_update_steps
        self.device = device
        self.model.to(self.device)
    
    def set_training_device(self, device: torch.device):
        self.model.to(device)
        self.device = device
    
    def train(
        self, 
        dataloader: DataLoader,
        n_epochs: int = 1,
        scale_reconstruction_loss: bool = True
    ):
        for epoch in range(1, n_epochs + 1):
            progress_bar = tqdm(total=len(dataloader))
            progress_bar.set_description(f"Epoch {epoch} / {n_epochs}")
            for batch_idx, data_batch in enumerate(dataloader):
                if not isinstance(data_batch, torch.Tensor):
                    data_batch = data_batch[0]
                
                data_batch = data_batch.to(self.device)
                
                encoded = self.model.encode(data_batch)
                quantized = self.model.quantize(encoded)

                quantization_loss = self.quantization_loss_fn(encoded, quantized)

                # this trick allows us to pass decoder gradients to encoder through the quantization step
                quantized = encoded + (quantized - encoded).detach()
                
                reconsutruction = self.model.decode(quantized)

                reconstruction_loss = self.reconstruction_loss_fn(data_batch, reconsutruction)

                if scale_reconstruction_loss:
                    reconstruction_loss /= np.var(dataloader.dataset.data / 255.0)

                loss = reconstruction_loss + quantization_loss
                loss.backward()
                self.optimizer.step()
                
                progress_bar.update()

                if batch_idx % self.metric_update_steps == 0:
                    progress_bar.set_postfix(dict(
                        ReconstructionLoss=reconstruction_loss.detach().cpu().numpy(),
                        QuantizationLoss=quantization_loss.detach().cpu().numpy()
                    ))
            
            progress_bar.close()
