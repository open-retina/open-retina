import pytest
import numpy as np
import torch

from openretina.models.autoencoder import SparsityMSELoss


@pytest.mark.parametrize("x_hat, desired_loss", [
    (np.zeros((4, 2)), 0.0),
    (np.ones((2, 2)), 2.0),
    (np.ones((5, 3, 2)), 2.0),
    (np.array([[1.0, 0.0], [2.0, 2.0]]), 5.0/2.0),
])
def test_sparsity_loss(x_hat: np.array, desired_loss: float):
    x_hat_tensor = torch.Tensor(x_hat)
    loss = SparsityMSELoss.sparsity_loss(x_hat_tensor, activations_dimension=-1)
    loss_float = float(loss.float())
    assert loss_float == desired_loss
