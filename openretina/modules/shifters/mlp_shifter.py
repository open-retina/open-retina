"""MLP shifter networks (Sinz et al. 2018).

A shifter maps a behavioral variable -- here the pupil center (eye position) -- to a
2D shift that is added to each neuron's readout receptive-field center ``mu``,
correcting for eye movements. Ported from ``neuralpredictors.layers.shifters.mlp`` and
adapted to open-retina's per-session (``n_neurons_dict``-keyed) multi-session
convention, mirroring the multi-session readout.

Qiu et al. 2026 use an MLP with 3 fully-connected layers, 5 hidden features and
``tanh`` nonlinearities (hence the defaults below).
"""

from collections.abc import Iterable

import torch
from jaxtyping import Float
from torch import nn


class MLPShifter(nn.Module):
    """Single-session MLP mapping pupil position to a readout-grid shift.

    Maps ``pupil_center`` of shape ``(n, input_channels)`` to a shift of shape
    ``(n, 2)`` in normalized readout-grid units (``[-1, 1]``, matching the Gaussian
    readout grid) using ``n_layers`` fully-connected layers, each followed by a
    ``tanh`` nonlinearity. The final ``tanh`` bounds the shift to ``[-1, 1]``.
    """

    def __init__(
        self,
        input_channels: int = 2,
        hidden_channels: int = 5,
        n_layers: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")
        layers: list[nn.Module] = []
        prev_features = input_channels
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(prev_features, hidden_channels, bias=bias), nn.Tanh()])
            prev_features = hidden_channels
        layers.extend([nn.Linear(prev_features, 2, bias=bias), nn.Tanh()])
        self.mlp = nn.Sequential(*layers)
        self.initialize()

    def initialize(self) -> None:
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def regularizer(self) -> torch.Tensor | float:
        return 0.0

    def forward(self, pupil_center: Float[torch.Tensor, "n input_channels"]) -> Float[torch.Tensor, "n 2"]:
        in_features = self.mlp[0].in_features
        if pupil_center.shape[-1] != in_features:
            raise ValueError(
                f"Expected pupil_center with {in_features} features in the last dimension, "
                f"got shape {tuple(pupil_center.shape)}."
            )
        return self.mlp(pupil_center)


class MultiSessionMLPShifter(nn.ModuleDict):
    """Per-session collection of :class:`MLPShifter` modules.

    Holds one shifter per session (shared across all neurons within a session), keyed
    by session id exactly like the multi-session readout so that the same ``data_key``
    indexes both. Accepts ``n_neurons_dict`` (open-retina convention; only its keys are
    used) or an explicit ``data_keys`` iterable.
    """

    def __init__(
        self,
        n_neurons_dict: dict[str, int] | None = None,
        data_keys: Iterable[str] | None = None,
        input_channels: int = 2,
        hidden_channels: int = 5,
        n_layers: int = 3,
        gamma_shifter: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if n_neurons_dict is not None:
            keys: list[str] = list(n_neurons_dict.keys())
        elif data_keys is not None:
            keys = list(data_keys)
        else:
            raise ValueError("Provide either n_neurons_dict or data_keys to key the shifters.")
        self.gamma_shifter = gamma_shifter
        for key in keys:
            self.add_module(key, MLPShifter(input_channels, hidden_channels, n_layers, bias))

    def __getitem__(self, key: str) -> MLPShifter:
        """For type checking purposes."""
        res = self._modules[key]
        assert isinstance(res, MLPShifter)
        return res

    def initialize(self, **kwargs) -> None:
        for shifter in self.values():
            assert isinstance(shifter, MLPShifter)
            shifter.initialize()

    def regularizer(self, data_key: str) -> torch.Tensor | float:
        return self[data_key].regularizer() * self.gamma_shifter

    def forward(
        self, pupil_center: Float[torch.Tensor, "n input_channels"], data_key: str
    ) -> Float[torch.Tensor, "n 2"]:
        return self[data_key](pupil_center)
