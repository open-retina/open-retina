# Implementation according to https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder

import lightning
import torch
from torch import nn
from torch.utils.data import Dataset


class SparsityMSELoss:
    def __init__(self, sparsity_factor: float):
        self.sparsity_factor = sparsity_factor

    @staticmethod
    def mse_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Mean over all examples, sum over hidden neurons"""
        mse_full = nn.functional.mse_loss(x, x_hat, reduction="none")
        mse = mse_full.sum(dim=-1).mean()
        return mse

    @staticmethod
    def sparsity_loss(z: torch.Tensor) -> torch.Tensor:
        # The anthropic paper just sums over all neurons
        # Make sure the interpolation factor is small enough to not have a dominating sparsity loss
        return z.abs().sum()

    def forward(
        self, x: torch.Tensor, z: torch.Tensor, x_hat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mse_loss = self.mse_loss(x, x_hat)
        sparsity_loss = self.sparsity_loss(z)
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
        unit_norm_loss_factor: float = 1.0,
    ):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.loss = loss
        self.learning_rate = learning_rate
        self.unit_norm_loss_factor = unit_norm_loss_factor
        self.save_hyperparameters()
        self.maximum_activations_neurons: list[torch.Tensor] = []
        self.mean_activations_neurons: list[torch.Tensor] = []

    def hidden_dim(self) -> int:
        return self.encoder.out_features

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_bar = x - self.bias
        # b_e is already present in self.encoder
        z = nn.functional.relu(self.encoder.forward(x_bar))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z) + self.bias
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hidden = self.encode(x)
        x_reconstruct = self.decode(x_hidden)
        return x_reconstruct

    def decoder_norm_diff_from_unit_norm(self) -> torch.Tensor:
        column_norms_decoder = self.decoder.weight.norm(dim=1)
        diff_from_unit_norm = torch.abs(1.0 - column_norms_decoder)
        norm_loss = torch.sum(diff_from_unit_norm)
        return norm_loss

    def on_after_backward(self) -> None:
        # remove parallel information of gradient to decoder weight columns
        with torch.no_grad():
            weight_normed = self.decoder.weight / self.decoder.weight.norm(dim=-1, keepdim=True)
            weight_grad_proj = (self.decoder.weight.grad * weight_normed).sum(dim=-1, keepdim=True) * weight_normed
            self.decoder.weight.grad -= weight_grad_proj

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # make decoder weight unit norm
        self.decoder.weight.data[:] = self.decoder.weight / self.decoder.weight.norm(dim=-1, keepdim=True)

        x, _ = batch
        z = self.encode(x)
        x_hat = self.decode(z)

        # Statistics
        l0_norm = (z > 0).sum(dim=-1, dtype=torch.float).mean()
        z_hat_mean = torch.mean(z, dim=(0, 1))
        z_hat_max = torch.max(torch.max(z, dim=0).values, dim=0).values
        self.mean_activations_neurons.append(z_hat_mean.detach().cpu())
        self.maximum_activations_neurons.append(z_hat_max.detach().cpu())
        total_loss, mse_loss, sparsity_loss = self.loss.forward(x, z, x_hat)
        decoder_norm_diff_tensor = self.decoder_norm_diff_from_unit_norm()
        self.log("mse_loss", mse_loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        self.log("sparsity_loss", sparsity_loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        self.log(
            "decoder_norm_diff", decoder_norm_diff_tensor, prog_bar=False, on_epoch=True, on_step=False, logger=True
        )
        self.log("train_loss", total_loss, prog_bar=True, on_epoch=True, logger=True, on_step=False)
        self.log("active_neurons", l0_norm, prog_bar=True, on_epoch=True, logger=True, on_step=False)

        return total_loss

    def fraction_neurons_below_threshold(self, epoch_max: torch.Tensor, threshold: float) -> float:
        inactive_neurons = int((epoch_max <= threshold).sum())
        fraction_inactive_neurons = inactive_neurons / self.hidden_dim()
        return fraction_inactive_neurons

    def on_train_epoch_end(self) -> None:
        epoch_mean = torch.mean(torch.stack(self.mean_activations_neurons), dim=0)
        epoch_max = torch.max(torch.stack(self.maximum_activations_neurons), dim=0).values
        self.mean_activations_neurons = []
        self.maximum_activations_neurons = []
        max_hidden_activation = float(torch.max(epoch_max))
        mean_activation = float(torch.mean(epoch_mean))
        fraction_inactive_neurons = self.fraction_neurons_below_threshold(epoch_max, 0.0)
        fraction_barely_active_neurons = self.fraction_neurons_below_threshold(epoch_max, 0.1)
        self.log("mean_hidden_activations", mean_activation, on_epoch=True, logger=True)
        self.log("fraction_inactive_neurons", fraction_inactive_neurons, on_epoch=True, logger=True)
        self.log("fraction_barely_active_neurons", fraction_barely_active_neurons, on_epoch=True, logger=True)
        print(
            f"{fraction_inactive_neurons=:.1%} "
            f"{fraction_barely_active_neurons=:.1%} (under 0.1), "
            f"{mean_activation=:.3f} {max_hidden_activation=:.3f}"
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class AutoencoderWithModel(nn.Module):
    def __init__(self, model: torch.nn.Module, autoencoder: Autoencoder):
        super().__init__()
        self.model = model
        self.autoencoder = autoencoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model_outputs: list[torch.Tensor] = []
        for key in self.model.readout_keys():  # type: ignore
            out = self.model.forward(x, key)
            model_outputs.append(out)
        activations = torch.cat(model_outputs, dim=-1)

        hidden = self.autoencoder.encode(activations)
        return hidden
