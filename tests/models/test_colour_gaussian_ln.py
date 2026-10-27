"""Colour LN model with one learnable Gaussian spatial filter and one temporal filter per colour channel."""

from pathlib import Path

import lightning
import numpy as np
import pytest
import torch

from openretina.models.colour_gaussian_ln import ColourGaussianLNCoreReadout
from openretina.modules.layers.regularizers import TimeLaplaceL23dnorm
from openretina.modules.losses import PoissonLoss3d
from openretina.modules.readout.colour_gaussian_ln_readout import ColourGaussianLNReadout
from openretina.modules.readout.spatial_contrast_model_readout import LearnableFilterSpatialContrastReadout
from openretina.utils.sta_processing import gaussian_2d

IN_SHAPE = (2, 40, 12, 10)  # channels time height width; the stimulus is green + UV
HEIGHT, WIDTH = IN_SHAPE[2:]
SESSION = "session_a"
N_NEURONS = 2
T_FILTER = 5
SIGMA_CONTOUR = 1.5
# Per neuron and channel: (center_x, center_y, sigma_x, sigma_y, theta)
GAUSSIANS = [
    [(3.3, 4.2, 1.4, 1.7, 0.3), (4.3, 5.6, 2.05, 1.45, -0.2)],
    [(6.5, 7.5, 1.6, 1.2, -0.4), (6.5, 7.0, 1.35, 1.9, 0.5)],
]


def distance_in_sigmas(center_x, center_y, sigma_x, sigma_y, theta) -> np.ndarray:
    x_grid, y_grid = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    x_diff, y_diff = x_grid - center_x, y_grid - center_y
    x_rot = np.cos(theta) * x_diff + np.sin(theta) * y_diff
    y_rot = -np.sin(theta) * x_diff + np.cos(theta) * y_diff
    return np.sqrt((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)


def test_no_pixel_sits_on_the_mask_contour() -> None:
    """Guards the other tests: a pixel on the 1.5 sigma contour may flip between float32 and float64 masks."""
    for neuron in GAUSSIANS:
        for gaussian in neuron:
            assert np.abs(distance_in_sigmas(*gaussian) - SIGMA_CONTOUR).min() > 0.05


def build_readout(**kwargs) -> ColourGaussianLNReadout:
    arguments: dict = dict(
        in_shape=IN_SHAPE,
        outdims=N_NEURONS,
        temporal_filter_length=T_FILTER,
        sigma_contour=SIGMA_CONTOUR,
        a_init=1.3,
        b_init=0.7,
        c_init=0.2,
    )
    arguments.update(kwargs)
    return ColourGaussianLNReadout(**arguments)


def set_gaussians(readout: ColourGaussianLNReadout, gaussians: list) -> None:
    """Set the readout's Gaussians to ``gaussians[neuron][channel]`` = (center_x, center_y, sigma_x, sigma_y, theta)."""
    params = torch.tensor(gaussians, dtype=readout.center_x.dtype)
    excess = params[..., 2:4] - readout.sigma_floor
    raw = excess + torch.log(-torch.expm1(-excess))  # inverse softplus
    with torch.no_grad():
        readout.center_x.copy_(params[..., 0])
        readout.center_y.copy_(params[..., 1])
        readout.sigma_x_raw.copy_(raw[..., 0])
        readout.sigma_y_raw.copy_(raw[..., 1])
        readout.theta.copy_(params[..., 4])


def test_output_is_the_sum_of_the_channels_filtered_signals() -> None:
    torch.manual_seed(0)
    readout = build_readout().double()
    set_gaussians(readout, GAUSSIANS)
    x = torch.rand(3, *IN_SHAPE, dtype=torch.float64)

    with torch.no_grad():
        output = readout(x).numpy()

    x_grid, y_grid = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    temporal = readout.temporal_weight.detach().numpy()
    temporal = temporal / np.linalg.norm(temporal.reshape(N_NEURONS, -1), axis=1)[:, None, None]
    a, b, c = (p.detach().numpy() for p in (readout.nl_a, readout.nl_b, readout.nl_c))
    t_out = IN_SHAPE[1] - T_FILTER + 1
    expected = np.zeros((3, t_out, N_NEURONS))
    for neuron in range(N_NEURONS):
        drive = np.zeros((3, t_out))
        for channel, params in enumerate(GAUSSIANS[neuron]):
            spatial = gaussian_2d((x_grid, y_grid), *params, 1.0).reshape(HEIGHT, WIDTH)
            spatial = spatial * (distance_in_sigmas(*params) <= SIGMA_CONTOUR)
            weighted_mean = np.einsum("bthw,hw->bt", x[:, channel].numpy(), spatial) / spatial.sum()
            for t in range(t_out):
                drive[:, t] += weighted_mean[:, t : t + T_FILTER] @ temporal[neuron, channel]
        expected[:, :, neuron] = a[neuron] * np.log1p(np.exp(b[neuron] * drive + c[neuron]))

    assert np.allclose(output, expected)


def test_a_channel_without_a_temporal_filter_is_ignored() -> None:
    torch.manual_seed(0)
    readout = build_readout()
    with torch.no_grad():
        readout.temporal_weight[:, 1] = 0.0
    x = torch.rand(3, *IN_SHAPE)
    x_other_uv = x.clone()
    x_other_uv[:, 1] = torch.rand_like(x[:, 1])

    with torch.no_grad():
        assert torch.allclose(readout(x), readout(x_other_uv))
        assert not torch.allclose(readout(x), readout(torch.rand_like(x)))


def test_equal_channels_reproduce_the_greyscale_ln_model() -> None:
    """With the same Gaussian and temporal filter in both channels, the colour model is the greyscale LN model on the
    channel mean, its gain scaled by sqrt(2) because the temporal filters are normalised over both channels."""
    torch.manual_seed(0)
    grey_gaussians = torch.tensor([neuron[0] for neuron in GAUSSIANS], dtype=torch.float64)
    temporal = torch.randn(N_NEURONS, T_FILTER, dtype=torch.float64)
    grey = LearnableFilterSpatialContrastReadout(
        in_shape=(1, *IN_SHAPE[1:]),
        outdims=N_NEURONS,
        spatial_filters=torch.zeros(N_NEURONS, HEIGHT, WIDTH, dtype=torch.float64),
        temporal_filters=temporal,
        gaussian_params=grey_gaussians,
        sigma_contour=SIGMA_CONTOUR,
        w_init=0.0,
        a_init=1.3,
        b_init=0.7,
        c_init=0.2,
    ).double()
    colour = build_readout().double()
    set_gaussians(colour, [[neuron[0], neuron[0]] for neuron in GAUSSIANS])
    with torch.no_grad():
        colour.temporal_weight.copy_(temporal[:, None, :].expand(-1, 2, -1))
        colour.nl_b.copy_(grey.nl_b / 2**0.5)
    x = torch.rand(3, *IN_SHAPE, dtype=torch.float64)

    with torch.no_grad():
        assert torch.allclose(colour(x), grey(x.mean(dim=1, keepdim=True)))


def test_each_channel_is_masked_at_its_own_contour() -> None:
    readout = build_readout()
    set_gaussians(readout, GAUSSIANS)
    spatial, temporal, sums = readout._filters()

    assert spatial.shape == (N_NEURONS, 2, HEIGHT, WIDTH)
    assert temporal.shape == (N_NEURONS, 2, T_FILTER)
    for neuron in range(N_NEURONS):
        assert torch.allclose(temporal[neuron].norm(), torch.tensor(1.0))
        for channel, params in enumerate(GAUSSIANS[neuron]):
            inside = torch.from_numpy(distance_in_sigmas(*params) <= SIGMA_CONTOUR)
            assert torch.all(spatial[neuron, channel][inside] > 0)
            assert torch.all(spatial[neuron, channel][~inside] == 0)
            assert torch.allclose(sums[neuron, channel], spatial[neuron, channel].sum())


def test_every_gaussian_starts_at_the_centre_with_the_initial_width() -> None:
    readout = build_readout(gaussian_init_sigma=3.0)
    sigma_x, sigma_y = readout.sigmas()

    assert torch.allclose(readout.center_x, torch.tensor((WIDTH - 1) / 2))
    assert torch.allclose(readout.center_y, torch.tensor((HEIGHT - 1) / 2))
    assert torch.allclose(sigma_x, torch.tensor(3.0)) and torch.allclose(sigma_y, torch.tensor(3.0))
    # The mask stays at sigma_contour sigmas of the wider Gaussian
    spatial, _, _ = readout._filters()
    distance = distance_in_sigmas((WIDTH - 1) / 2, (HEIGHT - 1) / 2, 3.0, 3.0, 0.0)
    assert torch.equal(spatial[0, 0] > 0, torch.from_numpy(distance <= SIGMA_CONTOUR))


def test_temporal_filters_are_drawn_from_the_seed() -> None:
    torch.manual_seed(0)
    first = build_readout().temporal_weight
    torch.manual_seed(0)
    second = build_readout().temporal_weight

    assert first.shape == (N_NEURONS, 2, T_FILTER)
    assert torch.all(first.abs() <= (3.0 / T_FILTER) ** 0.5)
    assert not torch.equal(first[0, 0], first[0, 1])
    assert torch.equal(first, second)


def test_offsets_start_at_each_neurons_mean_activity() -> None:
    mean_activity = torch.tensor([0.5, 2.0])
    readout = build_readout(a_init=1.0, mean_activity=mean_activity, init_offset_from_mean_activity=True)
    assert torch.allclose(readout.nl_a * torch.nn.functional.softplus(readout.nl_c), mean_activity)


def test_temporal_penalty_is_the_lnp_penalty_on_the_normalised_filters() -> None:
    torch.manual_seed(0)
    readout = build_readout(smooth_weight_temporal=10.0)
    _, temporal, _ = readout._filters()
    expected = 10.0 * TimeLaplaceL23dnorm()(temporal[..., None, None])

    with torch.no_grad():
        readout.temporal_weight[0] *= 5.0
    assert torch.allclose(readout.regularizer(), expected)
    assert torch.equal(build_readout().regularizer(), torch.tensor(0.0))


def data_info() -> dict:
    return {
        "input_shape": IN_SHAPE[:1] + IN_SHAPE[2:],
        "n_neurons_dict": {SESSION: N_NEURONS},
        "mean_activity_dict": {SESSION: torch.tensor([0.5, 2.0])},
    }


def build_model(**kwargs) -> ColourGaussianLNCoreReadout:
    arguments: dict = dict(
        in_shape=IN_SHAPE,
        n_neurons_dict={SESSION: N_NEURONS},
        temporal_filter_length=T_FILTER,
        target_model_cut_frames=30,
        sigma_contour=SIGMA_CONTOUR,
        smooth_weight_temporal=10.0,
        a_init=1.0,
        init_offset_from_mean_activity=True,
        scheduler_metric="poisson",
        filter_learning_rate=0.05,
        data_info=data_info(),
    )
    arguments.update(kwargs)
    return ColourGaussianLNCoreReadout(**arguments)


def take_optimizer_step(model: ColourGaussianLNCoreReadout) -> torch.Tensor:
    """One step of the model's own optimizer on its training objective; returns the objective before the step."""
    torch.manual_seed(0)
    optimizer = model.configure_optimizers()["optimizer"]
    x = torch.rand(3, *IN_SHAPE)
    targets = torch.rand(3, IN_SHAPE[1], N_NEURONS)
    output = model(x, SESSION)
    objective = model.loss(output, targets) + model.readout_regularization(SESSION, output)
    objective.backward()
    model.optimizer_step(0, 0, optimizer)
    return objective.detach()


def test_model_cuts_the_target_number_of_frames() -> None:
    model = build_model()
    output = model(torch.rand(3, *IN_SHAPE), SESSION)
    assert output.shape == (3, IN_SHAPE[1] - 30, N_NEURONS)


def test_filters_train_without_weight_decay() -> None:
    model = build_model()
    groups = model.configure_optimizers()["optimizer"].param_groups
    readout = model.readout[SESSION]
    assert isinstance(readout, ColourGaussianLNReadout)

    filter_group = next(group for group in groups if group["lr"] == 0.05)
    assert filter_group["weight_decay"] == 0.0
    assert {id(p) for p in filter_group["params"]} == {id(p) for p in readout.filter_parameters()}


def test_a_training_step_lowers_the_objective_and_keeps_the_constraints() -> None:
    model = build_model()
    before = take_optimizer_step(model)
    readout = model.readout[SESSION]
    assert isinstance(readout, ColourGaussianLNReadout)

    torch.manual_seed(0)
    x = torch.rand(3, *IN_SHAPE)
    targets = torch.rand(3, IN_SHAPE[1], N_NEURONS)
    with torch.no_grad():
        output = model(x, SESSION)
        after = model.loss(output, targets) + model.readout_regularization(SESSION, output)
    assert after < before
    assert torch.all(readout.nl_a >= 0)
    assert torch.all((readout.center_x >= 0) & (readout.center_x <= WIDTH - 1))


def test_temporal_penalty_has_the_strength_it_has_in_an_lnp() -> None:
    """Mean loss plus the scaled penalty is the LNP objective, a summed loss plus the penalty, over the elements."""
    torch.manual_seed(0)
    model = build_model()
    x = torch.rand(3, *IN_SHAPE)
    targets = torch.rand(3, IN_SHAPE[1], N_NEURONS)
    output = model(x, SESSION)

    mean_objective = model.loss(output, targets) + model.readout_regularization(SESSION, output)
    summed_objective = PoissonLoss3d()(output, targets) + model.readout.regularizer(SESSION)

    assert model.readout.regularizer(SESSION) > 0
    assert torch.allclose(mean_objective * output.numel(), summed_objective)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = build_model()
    take_optimizer_step(model)
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        checkpoint,
    )

    loaded = ColourGaussianLNCoreReadout.load_from_checkpoint(checkpoint, weights_only=False)

    assert loaded.state_dict().keys() == model.state_dict().keys()
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in loaded.state_dict().items())
    x = torch.rand(2, *IN_SHAPE)
    with torch.no_grad():
        assert torch.equal(loaded(x, SESSION), model(x, SESSION))


@pytest.mark.parametrize("kwargs", [dict(cut_first_n_frames=0), dict(temporal_filter_length=32)])
def test_bad_frame_cuts_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="cut"):
        build_model(**kwargs)
