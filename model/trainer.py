import typing as t

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from neptune.new.metadata_containers.run import Run
from neptune.new.types import File

from .loss import QuantizationLoss
from .model import VQVAE


class VQVAETrainer:
    def __init__(
        self,
        model: VQVAE,
        optimizer:  torch.optim.Optimizer,
        quantization_loss_fn: nn.Module = QuantizationLoss(0.25),
        reconstruction_loss_fn: nn.Module = torch.nn.MSELoss(),
        metric_update_steps: int = 10,
        device: t.Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.quantization_loss_fn = quantization_loss_fn
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.metric_update_steps = metric_update_steps

        self.set_training_device(device)

        self.quantization_loss_history = list()
        self.reconstruction_loss_history = list()
        self.loss_history = list()
        self.perplexity_history = list()

    def set_training_device(self, device: torch.device):
        self.model.to(device)
        self.device = device

    def _train_single_epoch(
        self, dataloader: DataLoader, progress_bar: tqdm, run: t.Optional[Run] = None
    ):
        data_variance = np.var(dataloader.dataset.data / 255.0)

        for batch_idx, data_batch in enumerate(dataloader):
            if not isinstance(data_batch, torch.Tensor):
                data_batch = data_batch[0]

            data_batch = data_batch.to(self.device)

            self.optimizer.zero_grad()

            pre_quantized = self.model.encoder.forward(data_batch)
            pre_quantized = self.model.pre_quantizer_conv.forward(pre_quantized)

            pre_quantized = pre_quantized.permute(0, 2, 3, 1).contiguous()
            quantized, encodings = self.model.quantizer.forward(pre_quantized)

            quantization_loss: torch.Tensor = self.quantization_loss_fn(
                quantized, pre_quantized
            )

            # this trick allows us to pass decoder gradients to encoder through the quantization step
            quantized = pre_quantized + (quantized - pre_quantized).detach()

            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            quantized = quantized.permute(0, 3, 1, 2).contiguous()

            reconsutruction = self.model.decoder.forward(quantized)

            reconstruction_loss: torch.Tensor = (
                self.reconstruction_loss_fn(reconsutruction, data_batch) / data_variance
            )

            loss = reconstruction_loss + quantization_loss

            loss.backward()
            self.optimizer.step()

            progress_bar.update()

            reconstruction_loss_np = reconstruction_loss.detach().cpu().numpy()
            quantization_loss_np = quantization_loss.detach().cpu().numpy()
            loss_np = loss.detach().cpu().numpy()
            perplexity_np = perplexity.detach().cpu().numpy()

            self.quantization_loss_history.append(quantization_loss_np)
            self.reconstruction_loss_history.append(reconstruction_loss_np)
            self.loss_history.append(loss_np)
            self.perplexity_history.append(perplexity_np)

            if batch_idx % self.metric_update_steps == 0:
                progress_bar.set_postfix(
                    dict(
                        ReconstructionLoss=np.mean(
                            self.reconstruction_loss_history[-100:]
                        ),
                        QuantizationLoss=np.mean(self.quantization_loss_history[-100:]),
                        Perplexity=np.mean(self.perplexity_history[-100:]),
                    )
                )

            if run is not None:
                run["train/ReconstructionLoss"].log(float(reconstruction_loss_np))
                run["train/QuantizationLoss"].log(float(quantization_loss_np))
                run["train/Perplexity"].log(float(perplexity_np))

    def _remote_log_reconstruction(
        self, 
        run: Run, 
        epoch: int,
        samples: torch.Tensor
    ):
        reconstruction_id = f"reconstruction-epoch-{epoch}"
        with torch.no_grad():
            reconstructions = self.model(samples).cpu()
        samples_and_reconstructions = torch.concat((samples.cpu(), reconstructions), dim=0)

        grid = make_grid(samples_and_reconstructions, nrow=8, normalize=True).permute(1, 2, 0).numpy()
        run[reconstruction_id].log(File.as_image(grid))
    
    def train(
        self,
        dataloader: DataLoader,
        n_epochs: int = 1,
        neptune_run: t.Optional[Run] = None,
        test_samples: t.Optional[torch.Tensor] = None
    ):
        if test_samples is not None and neptune_run is not None:
                self._remote_log_reconstruction(neptune_run, 0, test_samples)
        
        for epoch in range(1, n_epochs + 1):
            progress_bar = tqdm(total=len(dataloader))
            progress_bar.set_description(f"Epoch {epoch} / {n_epochs}")
            self._train_single_epoch(
                dataloader, progress_bar, neptune_run
            )

            if test_samples is not None and neptune_run is not None:
                self._remote_log_reconstruction(neptune_run, epoch, test_samples)

            progress_bar.close()
