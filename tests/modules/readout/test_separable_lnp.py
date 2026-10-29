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


def test_uniform_init_weighs_every_frame_equally() -> None:
    readout = build_readout(temporal_init="uniform")
    filters = readout.spatiotemporal_filters().detach()
    spatial = readout.spatial_kernel.weight[:, :, 0].detach()

    # Every frame gets 1 / T of the spatial filter, so the total gain matches the current-frame init
    assert torch.allclose(filters, spatial[:, :, None].expand_as(filters) / TEMPORAL_KERNEL_SIZE)


def test_random_init_draws_every_tap_within_its_bound() -> None:
    torch.manual_seed(0)
    first = build_readout(temporal_init="random").temporal_kernels().detach()
    torch.manual_seed(0)
    second = build_readout(temporal_init="random").temporal_kernels().detach()

    assert torch.all(first.abs() <= (3.0 / TEMPORAL_KERNEL_SIZE) ** 0.5)
    assert torch.unique(first).numel() == first.numel()
    assert torch.equal(first, second)


COLOUR_IN_SHAPE = (2, *IN_SHAPE[1:])


def build_colour_readout(**kwargs) -> SeparableLNPReadout:
    return SeparableLNPReadout(
        in_shape=COLOUR_IN_SHAPE,
        outdims=N_NEURONS,
        temporal_kernel_size=TEMPORAL_KERNEL_SIZE,
        nonlinearity="softplus",
        bias=True,
        temporal_kernel_per_channel=True,
        **kwargs,
    )


def test_per_channel_forward_equals_a_full_convolution_with_its_filter() -> None:
    torch.manual_seed(0)
    readout = build_colour_readout(temporal_init="random")
    assert readout.output_bias is not None
    with torch.no_grad():
        readout.output_bias.normal_()
    x = torch.rand(2, *COLOUR_IN_SHAPE)

    filters = readout.spatiotemporal_filters().detach()
    # The channels really get different temporal kernels
    assert not torch.allclose(readout.temporal_kernels()[:, 0], readout.temporal_kernels()[:, 1])
    with torch.no_grad():
        reference = F.conv3d(x, filters, bias=readout.output_bias)
        reference = rearrange(F.softplus(reference), "batch neurons t 1 1 -> batch t neurons")
        out = readout(x)

    assert out.shape == (2, IN_SHAPE[1] - (TEMPORAL_KERNEL_SIZE - 1), N_NEURONS)
    assert torch.allclose(out, reference, atol=1e-5)


def test_per_channel_kernels_match_the_shared_kernel_for_one_channel() -> None:
    torch.manual_seed(0)
    shared = build_readout(temporal_init="random")
    per_channel = build_readout(temporal_kernel_per_channel=True)
    assert per_channel.output_bias is not None and shared.temporal_kernel.bias is not None
    with torch.no_grad():
        per_channel.spatial_kernel.weight.copy_(shared.spatial_kernel.weight)
        per_channel.temporal_kernel.weight.copy_(shared.temporal_kernel.weight)
        per_channel.output_bias.copy_(shared.temporal_kernel.bias)
    x = torch.rand(2, *IN_SHAPE)

    with torch.no_grad():
        assert torch.allclose(per_channel(x), shared(x), atol=1e-5)
    assert torch.equal(per_channel.spatiotemporal_filters(), shared.spatiotemporal_filters())


def test_per_channel_bias_starts_at_the_mean_activity() -> None:
    mean_activity = torch.arange(1.0, N_NEURONS + 1)
    readout = build_colour_readout(mean_activity=mean_activity)

    assert readout.temporal_kernel.bias is None
    assert readout.output_bias is not None
    assert torch.equal(readout.output_bias.detach(), mean_activity)


def test_per_channel_regularizer_is_finite_and_differentiable() -> None:
    readout = build_colour_readout(smooth_weight=1.0, smooth_weight_temporal=1.0, sparse_weight=1.0)
    regularization = readout.regularizer()
    regularization.backward()

    assert torch.isfinite(regularization)
    assert readout.spatial_kernel.weight.grad is not None
    assert readout.temporal_kernel.weight.grad is not None


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


def build_colour_model() -> UnifiedCoreReadout:
    n_neurons_dict = {"session_a": N_NEURONS}
    core = OmegaConf.create(
        {
            "_target_": "openretina.modules.core.base_core.DummyCore",
            "_convert_": "object",
            "cut_first_n_frames": 0,
        }
    )
    readout = OmegaConf.create(
        {
            "_target_": "openretina.modules.readout.multi_readout.MultipleSeparableLNPReadout",
            "_convert_": "object",
            "in_shape": "???",  # derived by probing the core
            "nonlinearity": "softplus",
            "bias": True,
            "temporal_kernel_size": 31,
            "temporal_init": "random",
            "temporal_kernel_per_channel": True,
        }
    )
    return UnifiedCoreReadout(
        in_shape=(2, 150, 18, 16),
        n_neurons_dict=n_neurons_dict,
        core=core,
        readout=readout,
        data_info={"input_shape": (2, 18, 16), "n_neurons_dict": n_neurons_dict},
    )


def test_colour_model_cuts_thirty_frames_and_round_trips() -> None:
    model = build_colour_model()
    readout = model.readout["session_a"]
    assert isinstance(readout, SeparableLNPReadout)
    assert readout.in_channels == 2
    assert readout.temporal_kernels().shape == (N_NEURONS, 2, 31)

    with torch.no_grad():
        out = model(torch.rand(2, 2, 150, 18, 16), "session_a")
    assert out.shape == (2, 150 - 30, N_NEURONS)

    build_colour_model().load_state_dict(model.state_dict(), strict=True)
