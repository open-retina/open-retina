# Implementation according to https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder

import torch
from torch import nn
import lightning


class SparsityMSELoss:
    def __init__(self, sparsity_factor: float):
        self.sparsity_factor = sparsity_factor

    @staticmethod
    def mse_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """ Mean over all dimension (batch size, time, activations) """
        mse = nn.functional.mse_loss(x, x_hat, reduction="mean")
        return mse

    @staticmethod
    def sparsity_loss(x_hat: torch.Tensor, activations_dimension: int) -> torch.Tensor:
        """ Sum over all activations, mean over batch and time dimension """
        sparsity_loss = torch.mean(torch.sum(x_hat, dim=activations_dimension))
        return sparsity_loss

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse_loss(x, x_hat)
        sparsity_loss = self.sparsity_loss(x_hat, activations_dimension=-1)
        total_loss = mse_loss + self.sparsity_factor * sparsity_loss
        return total_loss


class Autoencoder(lightning.LightningModule):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            loss: SparsityMSELoss,
            learning_rate: float = 0.0005,
    ):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.loss = loss
        self.learning_rate = learning_rate

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

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss.forward(x, x_hat)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
