# Implementation according to https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder

import torch
from torch import nn
import lightning
from torch.utils.data import Dataset


class SparsityMSELoss:
    def __init__(self, sparsity_factor: float):
        self.sparsity_factor = sparsity_factor

    @staticmethod
    def mse_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """ Mean over all dimension (batch size, time, activations) """
        mse = nn.functional.mse_loss(x, x_hat, reduction="mean")
        return mse

    @staticmethod
    def sparsity_loss(z: torch.Tensor, activations_dimension: int) -> torch.Tensor:
        """ Sum over all activations, mean over batch and time dimension """
        summed_activations = torch.sum(torch.abs(z), dim=activations_dimension)
        sparsity_loss = torch.mean(summed_activations)
        return sparsity_loss

    def forward(
            self,
            x: torch.Tensor,
            z: torch.Tensor,
            x_hat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mse_loss = self.mse_loss(x, x_hat)
        sparsity_loss = self.sparsity_loss(z, activations_dimension=-1)
        total_loss = mse_loss + self.sparsity_factor * sparsity_loss
        return total_loss, mse_loss, sparsity_loss


class ActivationsDataset(Dataset):
    def __init__(self, activations: list[torch.Tensor]):
        self._activations = activations

    def __len__(self) -> int:
        return len(self._activations)

    def __getitem__(self, idx: int):
        act = self._activations[idx]
        fake_label = torch.zeros(act.shape[-1])  # maybe this should be a tensor?
        return act, fake_label


class Autoencoder(lightning.LightningModule):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            loss: SparsityMSELoss,
            learning_rate: float = 0.0005,
            unit_norm_loss_factor: float = 1.0
    ):
        # TODO: In transformer circuits talks about the decoder weights W_d having unit norm for its columns
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.loss = loss
        self.learning_rate = learning_rate
        self.unit_norm_loss_factor = unit_norm_loss_factor
        self.save_hyperparameters()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_bar = x - self.bias
        # b_e is already present in self.encoder
        f = nn.functional.relu(self.encoder.forward(x_bar))
        return f

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(x) + self.bias
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hidden = self.encode(x)
        x_reconstruct = self.decoder(x_hidden)
        return x_reconstruct

    def unit_norm_loss(self):
        column_norms_decoder = self.decoder.weight.norm(dim=1)
        diff_from_unit_norm = torch.abs(1.0 - column_norms_decoder)
        norm_loss = torch.sum(diff_from_unit_norm)
        return norm_loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss, mse_loss, sparsity_loss = self.loss.forward(x, z, x_hat)
        unit_norm_loss_tensor = self.unit_norm_loss()
        total_loss = loss + self.unit_norm_loss_factor * unit_norm_loss_tensor
        self.log("unit_norm_loss", unit_norm_loss_tensor, on_epoch=True, on_step=True, logger=True)
        self.log("mse_loss", mse_loss, on_epoch=True, on_step=True, logger=True)
        self.log("sparsity_loss", sparsity_loss, on_epoch=True, on_step=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True, on_step=False)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class AutoencoderWithModel:
    def __init__(self, model, autoencoder: Autoencoder):
        self.model = model
        self.autoencoder = autoencoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model_outputs: list[torch.Tensor] = []
        for key in self.model.readout_keys():
            out = self.model.forward(x, key)
            model_outputs.append(out)
        activations = torch.cat(model_outputs)

        hidden = self.autoencoder.encode(activations)
        return hidden
