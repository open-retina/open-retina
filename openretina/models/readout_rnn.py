import torch
from torch import nn


class ReadoutRnn(nn.Module):
    def __init__(
            self,
            input_dim: int = 1,
            hidden_dim: int = 64,
            output_dim: int = 1,
            n_layers: int = 1,
            drop_prob: float = 0.2,
    ):
        super(ReadoutRnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self._initial_hidden_vector = torch.nn.Parameter(
            data=torch.randn((self.n_layers, 1, hidden_dim)),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial_with_repeated_batch_dim = self._initial_hidden_vector.repeat((1, x.size(0), 1))
        out_gru, hidden_gru = self.gru(x, initial_with_repeated_batch_dim)
        out = self.fc(torch.nn.functional.relu(out_gru))
        return out
