"""Shared fixtures for the in-silico tests.

Everything here is built from scratch and stays tiny: these tests must run with no network,
no GPU, no checkpoint and no dataset.

The model built by :func:`tiny_qiu_model` mirrors the ``qiu_2026`` architecture -- three input
channels (1 video + 2 behavior), a multi-session Gaussian readout and a per-session MLP pupil
shifter -- at a size that forwards in milliseconds.
"""

import pytest
import torch
import torch.nn as nn
from jaxtyping import Float

from openretina.models.core_readout import BaseCoreReadout
from openretina.modules.core.base_core import SimpleCoreWrapper
from openretina.modules.readout.multi_readout import MultiSampledGaussianReadout
from openretina.modules.shifters.mlp_shifter import MultiSessionMLPShifter

# channels, time, height, width -- the qiu_2026 layout at toy size.
QIU_IN_SHAPE = (3, 20, 16, 18)
QIU_N_NEURONS_DICT = {"sess_a": 4, "sess_b": 3}
QIU_BEHAVIOR_CHANNELS = (1, 2)


def calibrate_batchnorm(core: nn.Module, in_shape: tuple[int, ...], n_batches: int = 30, batch_size: int = 4) -> None:
    """Populate the core's ``BatchNorm3d`` running statistics from random probe inputs.

    Without this a freshly-built core is useless in eval mode: ``BatchNorm3d`` normalizes with its
    *initial* ``(mean=0, var=1)`` running stats, which is a no-op, so the tiny randomly-initialized
    conv weights compound to a core output with a standard deviation around ``5e-6``. The readout's
    softplus then sits at ``log(2)`` for every input, gradients vanish, and any test that asks
    "does X change the response?" passes or fails for the wrong reason.

    Running a few forward passes in train mode fills the running stats with realistic values --
    exactly what training would do -- so eval mode is both deterministic *and* non-degenerate.
    """
    batchnorms = [m for m in core.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    was_training = core.training
    momenta = [m.momentum for m in batchnorms]
    for module in batchnorms:
        module.reset_running_stats()
        # momentum=None switches BatchNorm to a cumulative moving average. With the default 0.1 the
        # running variance decays as 0.9**n from its initial 1.0, which after any affordable number
        # of probe batches is still orders of magnitude above the true variance of this core's tiny
        # activations -- i.e. exactly the degeneracy this function is meant to remove.
        module.momentum = None
    core.train()
    with torch.no_grad():
        for _ in range(n_batches):
            core(torch.randn(batch_size, *in_shape))
    for module, momentum in zip(batchnorms, momenta, strict=True):
        module.momentum = momentum
    core.train(was_training)


def build_tiny_qiu_model(with_shifter: bool = True, seed: int = 0, calibrate: bool = True) -> BaseCoreReadout:
    """Build a fresh untrained qiu-shaped model. Deterministic given ``seed``.

    Args:
        with_shifter: attach a per-session MLP pupil shifter.
        seed: seeds weight initialization and the BatchNorm calibration probes.
        calibrate: run :func:`calibrate_batchnorm` on the core. Leave on unless a test specifically
            wants the degenerate untrained-BatchNorm behavior.
    """
    torch.manual_seed(seed)
    core = SimpleCoreWrapper(
        channels=(QIU_IN_SHAPE[0], 8, 8),
        temporal_kernel_sizes=(5, 5),
        spatial_kernel_sizes=(5, 5),
        gamma_input=0.0,
        gamma_temporal=0.0,
        gamma_in_sparse=0.0,
        gamma_hidden=0.0,
        input_padding=False,
        hidden_padding=True,
        cut_first_n_frames=0,
    )
    # `compute_readout_input_shape` does not touch `self`; call it unbound to avoid a chicken-and-egg
    # problem (the readout is a constructor argument of the model that owns the method).
    if calibrate:
        calibrate_batchnorm(core, QIU_IN_SHAPE)
    readout_in_shape = BaseCoreReadout.compute_readout_input_shape(
        BaseCoreReadout.__new__(BaseCoreReadout), QIU_IN_SHAPE, core
    )
    readout = MultiSampledGaussianReadout(
        in_shape=readout_in_shape,
        n_neurons_dict=QIU_N_NEURONS_DICT,
        bias=True,
        init_mu_range=0.1,
        init_sigma_range=0.3,
        gauss_type="full",
        grid_mean_predictor=None,
        gamma=1.0,
    )
    shifter = (
        MultiSessionMLPShifter(
            n_neurons_dict=QIU_N_NEURONS_DICT,
            input_channels=2,
            hidden_channels=5,
            n_layers=3,
            gamma_shifter=0.0,
        )
        if with_shifter
        else None
    )
    channels, height, width = QIU_IN_SHAPE[0], QIU_IN_SHAPE[2], QIU_IN_SHAPE[3]
    return BaseCoreReadout(
        core=core,
        readout=readout,
        learning_rate=1e-3,
        data_info={"input_shape": (channels, height, width)},
        shifter=shifter,
    )


@pytest.fixture
def tiny_qiu_model() -> BaseCoreReadout:
    """A qiu-shaped model with a shifter, in eval mode and with calibrated BatchNorm statistics.

    Eval mode matters twice over: it fixes the Gaussian readout to its mean grid (so repeated
    forwards are deterministic), and the response-grid helpers refuse to run on a training-mode
    model for exactly that reason.

    ``tests/models/test_core_readout_shifter.py`` documents why an untrained model has to be used
    in *train* mode to see any effect at all; :func:`calibrate_batchnorm` removes that constraint
    here, so this fixture is both deterministic and responsive to its input.
    """
    model = build_tiny_qiu_model()
    model.eval()
    return model


class LinearBehaviorStub(nn.Module):
    """A model whose response is an exactly known linear function of the behavior channels.

    Without a trained network there is no way to test that a behavior sweep reads out the *right*
    numbers rather than merely the right shapes. This stub closes that gap:

    ``response[b, t, n] = bias[n] + video_weight[n] * mean(video) + sum_k behavior_weight[n, k] * mean(channel_k)``

    where the means are taken over every non-channel dimension. Because the response is linear in
    the behavior values, finite-difference gradients over a sweep grid must reproduce
    ``behavior_weight`` to numerical precision.

    ``data_key`` selects the per-session weight block; the last received ``data_key``,
    ``pupil_center`` and assembled input are recorded for spy-style assertions.
    """

    def __init__(
        self,
        n_neurons_dict: dict[str, int] | None = None,
        num_channels: int = 3,
        behavior_channels: tuple[int, ...] = QIU_BEHAVIOR_CHANNELS,
        cut_frames: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        if n_neurons_dict is None:
            n_neurons_dict = dict(QIU_N_NEURONS_DICT)
        generator = torch.Generator().manual_seed(seed)
        self.n_neurons_dict = dict(n_neurons_dict)
        self.num_channels = num_channels
        self.behavior_channels = tuple(behavior_channels)
        self.cut_frames = cut_frames
        self.data_info = {"input_shape": (num_channels, QIU_IN_SHAPE[2], QIU_IN_SHAPE[3])}

        self.behavior_weight = nn.ParameterDict()
        self.video_weight = nn.ParameterDict()
        self.bias = nn.ParameterDict()
        for key, n_neurons in self.n_neurons_dict.items():
            self.behavior_weight[key] = nn.Parameter(
                torch.randn(n_neurons, len(self.behavior_channels), generator=generator)
            )
            self.video_weight[key] = nn.Parameter(torch.randn(n_neurons, generator=generator))
            self.bias[key] = nn.Parameter(torch.randn(n_neurons, generator=generator))

        self.last_data_key: str | None = None
        self.last_pupil_center: torch.Tensor | None = None
        self.last_input: torch.Tensor | None = None
        self.inputs: list[torch.Tensor] = []

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels t h w"],
        data_key: str | None = None,
        pupil_center: Float[torch.Tensor, "batch two t"] | None = None,
    ) -> Float[torch.Tensor, "batch t_out neurons"]:
        key = data_key if data_key is not None else next(iter(self.n_neurons_dict))
        self.last_data_key = data_key
        self.last_pupil_center = pupil_center
        self.last_input = x.detach().clone()
        self.inputs.append(self.last_input)

        video_channels = [c for c in range(self.num_channels) if c not in self.behavior_channels]
        video_mean = x[:, video_channels].mean(dim=(1, 2, 3, 4))  # (batch,)
        behavior_mean = x[:, list(self.behavior_channels)].mean(dim=(2, 3, 4))  # (batch, n_behavior)

        response = (
            self.bias[key] + video_mean[:, None] * self.video_weight[key] + behavior_mean @ self.behavior_weight[key].T
        )  # (batch, neurons)
        time_steps_out = x.shape[2] - self.cut_frames
        return response[:, None, :].expand(-1, time_steps_out, -1)

    def stimulus_shape(self, time_steps: int, num_batches: int = 1) -> tuple[int, int, int, int, int]:
        channels, height, width = self.data_info["input_shape"]
        return num_batches, channels, time_steps, height, width

    def readout_keys(self) -> list[str]:
        return list(self.n_neurons_dict)


@pytest.fixture
def linear_behavior_stub() -> LinearBehaviorStub:
    stub = LinearBehaviorStub()
    stub.eval()
    return stub


@pytest.fixture
def linear_behavior_stub_cls() -> type[LinearBehaviorStub]:
    """The stub *class*, for tests that need to construct it with non-default arguments.

    Handed out as a fixture because ``tests/`` has no ``__init__.py`` files, so a test in a
    sub-directory cannot import this module by name.
    """
    return LinearBehaviorStub


@pytest.fixture
def tiny_qiu_model_factory():
    """``build_tiny_qiu_model`` itself, for tests that need a shifter-free or re-seeded model."""
    return build_tiny_qiu_model
