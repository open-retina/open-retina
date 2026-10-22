"""Space-time separable LNP readout: a rank-1 spatiotemporal filter per neuron."""

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

from openretina.models.core_readout import UnifiedCoreReadout
from openretina.modules.readout.linear_nonlinear_poison import SeparableLNPReadout

IN_SHAPE = (1, 40, 12, 10)  # channels time height width, as the readout sees it
N_NEURONS = 3
TEMPORAL_KERNEL_SIZE = 5


def build_readout(**kwargs) -> SeparableLNPReadout:
    return SeparableLNPReadout(
        in_shape=IN_SHAPE,
        outdims=N_NEURONS,
        temporal_kernel_size=TEMPORAL_KERNEL_SIZE,
        nonlinearity="softplus",
        bias=True,
        **kwargs,
    )


def test_output_loses_the_frames_the_temporal_kernel_consumes() -> None:
    readout = build_readout()
    out = readout(torch.rand(2, *IN_SHAPE))

    assert out.shape == (2, IN_SHAPE[1] - (TEMPORAL_KERNEL_SIZE - 1), N_NEURONS)


def test_forward_equals_a_full_convolution_with_the_rank_one_filter() -> None:
    """The separable forward pass must be exactly the filter that gets exported as an STA."""
    torch.manual_seed(0)
    readout = build_readout()
    assert readout.temporal_kernel.bias is not None
    with torch.no_grad():
        readout.temporal_kernel.weight.normal_()
        readout.temporal_kernel.bias.normal_()
    x = torch.rand(2, *IN_SHAPE)

    with torch.no_grad():
        reference = F.conv3d(x, readout.spatiotemporal_filters(), bias=readout.temporal_kernel.bias)
        reference = rearrange(F.softplus(reference), "batch neurons t 1 1 -> batch t neurons")
        out = readout(x)

    assert torch.allclose(out, reference, atol=1e-5)


def test_starts_as_the_lag_zero_lnp() -> None:
    readout = build_readout()
    filters = readout.spatiotemporal_filters().detach()

    assert filters.shape == (N_NEURONS, 1, TEMPORAL_KERNEL_SIZE, *IN_SHAPE[2:])
    assert torch.all(filters[:, :, :-1] == 0)
    assert torch.equal(filters[:, :, -1], readout.spatial_kernel.weight[:, :, 0].detach())


def test_regularizer_is_finite_and_differentiable() -> None:
    readout = build_readout(smooth_weight=1.0, smooth_weight_temporal=1.0, sparse_weight=1.0)
    regularization = readout.regularizer()
    regularization.backward()

    assert torch.isfinite(regularization)
    assert readout.spatial_kernel.weight.grad is not None
    assert readout.temporal_kernel.weight.grad is not None


def build_greyscale_model() -> UnifiedCoreReadout:
    n_neurons_dict = {"session_a": N_NEURONS}
    core = OmegaConf.create(
        {
            "_target_": "openretina.modules.core.base_core.DummyCore",
            "_convert_": "object",
            "cut_first_n_frames": 21,
            "color_squashing_weights": [0.5, 0.5],
        }
    )
    readout = OmegaConf.create(
        {
            "_target_": "openretina.modules.readout.multi_readout.MultipleSeparableLNPReadout",
            "_convert_": "object",
            "in_shape": "???",  # derived by probing the core
            "nonlinearity": "softplus",
            "bias": True,
            "temporal_kernel_size": 10,
        }
    )
    return UnifiedCoreReadout(
        in_shape=(2, 150, 18, 16),
        n_neurons_dict=n_neurons_dict,
        core=core,
        readout=readout,
        data_info={"input_shape": (2, 18, 16), "n_neurons_dict": n_neurons_dict},
    )


def test_greyscale_model_cuts_thirty_frames_and_round_trips() -> None:
    model = build_greyscale_model()
    readout = model.readout["session_a"]
    assert isinstance(readout, SeparableLNPReadout)
    assert readout.in_channels == 1

    with torch.no_grad():
        out = model(torch.rand(2, 2, 150, 18, 16), "session_a")
    assert out.shape == (2, 150 - 30, N_NEURONS)

    build_greyscale_model().load_state_dict(model.state_dict(), strict=True)
