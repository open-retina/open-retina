"""Spatial Contrast model on a 2-channel stimulus, with filters exported from greyscale LNPs.

A tiny greyscale LNP gets hand-set Gaussian filters (one ON, one OFF cell), is exported with
`export_lnp_filters_as_stas`, and the exported files then drive `SpatialContrastCoreReadout`.
"""

import json
import shutil
from pathlib import Path

import lightning
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from openretina.models.core_readout import UnifiedCoreReadout
from openretina.models.spatial_contrast import SingleCellSpatialContrast, SpatialContrastCoreReadout
from openretina.modules.layers.regularizers import TimeLaplaceL23dnorm
from openretina.modules.losses import PoissonLoss3d
from openretina.modules.readout.linear_nonlinear_poison import LNPReadout, SeparableLNPReadout
from openretina.modules.readout.spatial_contrast_model_readout import (
    LearnableFilterSpatialContrastReadout,
    SpatialContrastReadout,
)
from openretina.utils.sta_processing import export_lnp_filters_as_stas, gaussian_2d

IN_SHAPE = (2, 40, 12, 10)  # channels time height width; the stimulus is green + UV
HEIGHT, WIDTH = IN_SHAPE[2:]
SESSION = "session_a"
# Stored as floats, as in the hoefling data, to check that file names still use integers
ROI_IDS = np.array([3.0, 7.0])
SIGMA_CONTOUR = 1.5
TOTAL_CUT = 30
# (center_x, center_y, sigma_x, sigma_y, theta, amplitude): an ON and an OFF cell
GAUSSIANS = [(3.3, 4.2, 1.4, 1.7, 0.3, 1.0), (6.5, 7.5, 1.6, 1.2, -0.4, -1.0)]
TEMPORAL_KERNEL = [0.1, -0.3, 0.2, 1.0, 0.5]  # index -1 is the current frame


def gaussian_map(center_x, center_y, sigma_x, sigma_y, theta, amplitude) -> np.ndarray:
    x_grid, y_grid = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    return gaussian_2d((x_grid, y_grid), center_x, center_y, sigma_x, sigma_y, theta, amplitude).reshape(HEIGHT, WIDTH)


def test_no_pixel_sits_on_the_mask_contour() -> None:
    """Guards the other tests: a pixel on the 1.5 sigma contour may flip between float32 and float64 masks."""
    x_grid, y_grid = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    for center_x, center_y, sigma_x, sigma_y, theta, _ in GAUSSIANS:
        x_diff, y_diff = x_grid - center_x, y_grid - center_y
        x_rot = np.cos(theta) * x_diff + np.sin(theta) * y_diff
        y_rot = -np.sin(theta) * x_diff + np.cos(theta) * y_diff
        distance = np.sqrt((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)
        assert np.abs(distance - SIGMA_CONTOUR).min() > 0.05


def data_info() -> dict:
    return {
        "input_shape": IN_SHAPE[:1] + IN_SHAPE[2:],
        "n_neurons_dict": {SESSION: len(ROI_IDS)},
        "sessions_kwargs": {SESSION: {"roi_ids": ROI_IDS, "group_assignment": np.array([20, 3])}},
    }


def build_greyscale_lnp(temporal_kernel: list[float] | None) -> UnifiedCoreReadout:
    """A lag-0 LNP (temporal_kernel=None) or a separable one, with the hand-set filters."""
    readout_target = "MultipleLNPReadout" if temporal_kernel is None else "MultipleSeparableLNPReadout"
    readout_config: dict = {
        "_target_": f"openretina.modules.readout.multi_readout.{readout_target}",
        "_convert_": "object",
        "in_shape": "???",
        "nonlinearity": "softplus",
        "bias": True,
    }
    if temporal_kernel is not None:
        readout_config["temporal_kernel_size"] = len(temporal_kernel)
    model = UnifiedCoreReadout(
        in_shape=IN_SHAPE,
        n_neurons_dict={SESSION: len(ROI_IDS)},
        core=OmegaConf.create(
            {
                "_target_": "openretina.modules.core.base_core.DummyCore",
                "_convert_": "object",
                "cut_first_n_frames": 0,
                "color_squashing_weights": [0.5, 0.5],
            }
        ),
        readout=OmegaConf.create(readout_config),
        data_info=data_info(),
    )
    readout = model.readout[SESSION]
    with torch.no_grad():
        for neuron, params in enumerate(GAUSSIANS):
            spatial = torch.from_numpy(gaussian_map(*params)).float()
            if isinstance(readout, LNPReadout):
                readout.inner_product_kernel.weight[neuron, 0, 0] = spatial
            else:
                assert isinstance(readout, SeparableLNPReadout) and temporal_kernel is not None
                readout.spatial_kernel.weight[neuron, 0, 0] = spatial
                readout.temporal_kernel.weight[neuron, 0, :, 0, 0] = torch.tensor(temporal_kernel)
    return model


@pytest.fixture(params=[None, TEMPORAL_KERNEL], ids=["lag0", "separable"])
def temporal_kernel(request) -> list[float] | None:
    return request.param


@pytest.fixture
def sta_dir(tmp_path: Path, temporal_kernel: list[float] | None) -> Path:
    out = tmp_path / "stas"
    export_lnp_filters_as_stas(build_greyscale_lnp(temporal_kernel), out, source="test")
    return out


def n_frames(temporal_kernel: list[float] | None) -> int:
    return 1 if temporal_kernel is None else len(temporal_kernel)


def build_sc(sta_dir: Path | None, **kwargs) -> SpatialContrastCoreReadout:
    arguments = dict(
        in_shape=IN_SHAPE,
        n_neurons_dict={SESSION: len(ROI_IDS)},
        sta_dir=None if sta_dir is None else str(sta_dir),
        sta_file_pattern="{retina_index}_roi{cell_index}.npy",
        cell_id_key="roi_ids",
        temporal_crop_frames=None,
        sigma_contour=SIGMA_CONTOUR,
        target_model_cut_frames=TOTAL_CUT,
        color_squashing_weights=[0.5, 0.5],
        a_init=1.0,
        b_init=1.0,
        c_init=1.0,
        w_init=0.2,
        data_info=data_info(),
    )
    arguments.update(kwargs)
    return SpatialContrastCoreReadout(**arguments)  # type: ignore[arg-type]


def session_readout(model: SpatialContrastCoreReadout) -> SpatialContrastReadout:
    readout = model.readout[SESSION]
    assert isinstance(readout, SpatialContrastReadout)
    return readout


def test_export_writes_stas_manifest_and_info(sta_dir: Path, temporal_kernel: list[float] | None) -> None:
    for roi_id in (3, 7):
        sta = np.load(sta_dir / f"{SESSION}_roi{roi_id}.npy")
        assert sta.shape == (n_frames(temporal_kernel), HEIGHT, WIDTH)
        assert sta.dtype == np.float32

    manifest = json.loads((sta_dir / "export_info.json").read_text())
    assert manifest["color_squashing_weights"] == [0.5, 0.5]
    assert manifest["n_frames"] == n_frames(temporal_kernel)

    import pandas as pd

    rows = pd.read_csv(sta_dir / "manifest.csv")
    assert rows["roi_id"].tolist() == [3, 7]
    assert rows["polarity"].tolist() == [1.0, -1.0]
    assert rows["polarity_consistent"].all()
    assert rows["fit_success"].all()
    assert rows["group_assignment"].tolist() == [20, 3]


def test_export_refuses_colour_filters(tmp_path: Path) -> None:
    model = build_greyscale_lnp(None)
    readout = model.readout[SESSION]
    assert isinstance(readout, LNPReadout)
    readout.inner_product_kernel.weight = torch.nn.Parameter(torch.zeros(2, 2, 1, HEIGHT, WIDTH))
    with pytest.raises(ValueError, match="greyscale"):
        export_lnp_filters_as_stas(model, tmp_path)


def test_frozen_sc_output_and_polarity(sta_dir: Path, temporal_kernel: list[float] | None) -> None:
    model = build_sc(sta_dir)
    readout = session_readout(model)

    assert model.core._cut_first_n_frames == TOTAL_CUT - (n_frames(temporal_kernel) - 1)  # type: ignore[union-attr]
    on_filter, off_filter = readout.temporal_filters
    assert on_filter.shape == (n_frames(temporal_kernel),)
    # The spatial Gaussians are made positive, so the OFF cell's polarity lives in its temporal filter
    assert torch.allclose(off_filter, -on_filter, atol=1e-5)
    assert torch.all(readout.spatial_filters >= 0)

    with torch.no_grad():
        out = model(torch.rand(3, *IN_SHAPE), SESSION)
    assert out.shape == (3, IN_SHAPE[1] - TOTAL_CUT, len(ROI_IDS))


def test_frozen_sc_sees_the_channel_mean(sta_dir: Path) -> None:
    model = build_sc(sta_dir)
    green, uv = torch.rand(2, 1, *IN_SHAPE[1:]), torch.rand(2, 1, *IN_SHAPE[1:])
    mean = 0.5 * (green + uv)

    with torch.no_grad():
        from_colors = model(torch.cat([green, uv], dim=1), SESSION)
        from_mean = model(torch.cat([mean, mean], dim=1), SESSION)

    assert torch.allclose(from_colors, from_mean, atol=1e-6)


def take_optimizer_step(model: SpatialContrastCoreReadout) -> torch.optim.Optimizer:
    """One step of the model's own optimizer on a Poisson loss, followed by its parameter constraints."""
    torch.manual_seed(0)
    optimizer = model.configure_optimizers()["optimizer"]
    x = torch.rand(3, *IN_SHAPE)
    targets = torch.rand(3, IN_SHAPE[1], len(ROI_IDS))
    loss = model.loss(model(x, SESSION), targets)
    loss.backward()
    model.optimizer_step(0, 0, optimizer)
    return optimizer


def test_ln_baseline_keeps_w_at_zero(sta_dir: Path) -> None:
    model = build_sc(sta_dir, learn_contrast_weight=False, w_init=0.5)
    readout = session_readout(model)
    assert torch.all(readout.w == 0)

    take_optimizer_step(model)

    assert torch.all(readout.w == 0)
    assert not torch.all(readout.nl_c == 1.0)  # the rest of the model did train


def test_missing_filter_file_raises(sta_dir: Path) -> None:
    info = data_info()
    info["sessions_kwargs"][SESSION]["roi_ids"] = np.array([3.0, 8.0])
    with pytest.raises(FileNotFoundError):
        build_sc(sta_dir, data_info=info)


def test_colour_stimulus_without_squashing_raises(sta_dir: Path) -> None:
    with pytest.raises(ValueError, match="single channel"):
        build_sc(sta_dir, color_squashing_weights=None)


def test_filters_from_differently_squashed_stimulus_raise(sta_dir: Path) -> None:
    with pytest.raises(ValueError, match="color_squashing_weights"):
        build_sc(sta_dir, color_squashing_weights=[0.3, 0.7])


def test_cut_frames_and_target_cut_are_exclusive(sta_dir: Path) -> None:
    with pytest.raises(ValueError, match="either"):
        build_sc(sta_dir, cut_first_n_frames=0)


def test_learned_filters_start_as_the_frozen_model(sta_dir: Path) -> None:
    frozen = build_sc(sta_dir)
    learned = build_sc(sta_dir, learn_filters=True)
    assert isinstance(learned.readout[SESSION], LearnableFilterSpatialContrastReadout)

    x = torch.rand(3, *IN_SHAPE)
    with torch.no_grad():
        assert torch.allclose(learned(x, SESSION), frozen(x, SESSION), atol=1e-5)


@pytest.mark.parametrize("temporal_kernel", [TEMPORAL_KERNEL], ids=["separable"])
def test_learned_filters_train_without_weight_decay(sta_dir: Path) -> None:
    model = build_sc(sta_dir, learn_filters=True, filter_learning_rate=0.05)
    readout = model.readout[SESSION]
    assert isinstance(readout, LearnableFilterSpatialContrastReadout)
    before = {name: p.detach().clone() for name, p in readout.named_parameters()}

    optimizer = take_optimizer_step(model)

    filter_group = optimizer.param_groups[1]
    assert filter_group["weight_decay"] == 0.0
    assert filter_group["lr"] == 0.05
    for name in ("center_x", "center_y", "sigma_x_raw", "sigma_y_raw", "theta", "temporal_weight"):
        assert not torch.equal(getattr(readout, name), before[name]), name


@pytest.mark.parametrize("temporal_kernel", [None], ids=["lag0"])
def test_learned_filters_can_grow_longer_than_the_sta(sta_dir: Path) -> None:
    frozen = build_sc(sta_dir)
    learned = build_sc(sta_dir, learn_filters=True, temporal_filter_length=5)
    readout = learned.readout[SESSION]
    assert isinstance(readout, LearnableFilterSpatialContrastReadout)

    assert readout.temporal_weight.shape == (len(ROI_IDS), 5)
    assert learned.core._cut_first_n_frames == TOTAL_CUT - 4  # type: ignore[union-attr]
    x = torch.rand(3, *IN_SHAPE)
    with torch.no_grad():
        assert torch.allclose(learned(x, SESSION), frozen(x, SESSION), atol=1e-5)

    with pytest.raises(ValueError, match="shorter"):
        build_sc(sta_dir, learn_filters=True, temporal_filter_length=0)


@pytest.mark.parametrize("temporal_kernel", [TEMPORAL_KERNEL], ids=["separable"])
def test_neuron_without_a_filter_stays_switched_off(sta_dir: Path) -> None:
    """A NaN STA gives an all-zero temporal filter, which must not poison the loss or the other neurons."""
    np.save(sta_dir / f"{SESSION}_roi7.npy", np.full((len(TEMPORAL_KERNEL), HEIGHT, WIDTH), np.nan, np.float32))
    model = build_sc(sta_dir, learn_filters=True)
    readout = model.readout[SESSION]
    assert isinstance(readout, LearnableFilterSpatialContrastReadout)
    assert readout.active.tolist() == [1.0, 0.0]

    loss = model.loss(model(torch.rand(3, *IN_SHAPE), SESSION), torch.rand(3, IN_SHAPE[1], len(ROI_IDS)))
    loss.backward()

    assert torch.isfinite(loss)
    for parameter in readout.filter_parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))
        assert torch.all(parameter.grad[1] == 0)


def test_learned_receptive_fields_stay_on_the_stimulus(sta_dir: Path) -> None:
    model = build_sc(sta_dir, learn_filters=True)
    readout = model.readout[SESSION]
    assert isinstance(readout, LearnableFilterSpatialContrastReadout)
    with torch.no_grad():
        readout.center_x[0] = -5.0
        readout.center_y[0] = 100.0
        readout.nl_a[0] = -1.0

    model.optimizer_step(0, 0, torch.optim.SGD(model.parameters(), lr=0.0))

    assert readout.center_x[0] == 0.0
    assert readout.center_y[0] == HEIGHT - 1
    assert readout.nl_a[0] == 0.0


def test_learned_filter_plot_shows_the_current_filter(sta_dir: Path) -> None:
    model = build_sc(sta_dir, learn_filters=True)
    readout = model.readout[SESSION]
    assert isinstance(readout, LearnableFilterSpatialContrastReadout)
    with torch.no_grad():
        readout.center_x[0] += 2.0

    figure = readout.plot_weight_for_neuron(0)
    plotted = np.asarray(figure.axes[0].images[0].get_array())

    assert np.allclose(plotted, readout._filters()[0][0].detach().numpy())
    assert not np.allclose(plotted, readout.spatial_filters[0].numpy())


@pytest.mark.parametrize("temporal_kernel", [TEMPORAL_KERNEL], ids=["separable"])
def test_single_cell_model_clamps_the_gain_not_the_contrast_weight(sta_dir: Path) -> None:
    model = SingleCellSpatialContrast(
        in_shape=(1, *IN_SHAPE[1:]),
        sta_dir=str(sta_dir),
        sta_file_name=f"{SESSION}_roi3.npy",
        temporal_crop_frames=None,
    )
    with torch.no_grad():
        model.nl_a.fill_(-1.0)
        model.w.fill_(-1.0)

    model.optimizer_step(0, 0, torch.optim.SGD(model.parameters(), lr=0.0))

    assert model.nl_a == 0.0
    assert model.w == -1.0


def test_offsets_start_at_each_neurons_mean_activity(sta_dir: Path) -> None:
    info = data_info()
    mean_activity = torch.tensor([0.8, 2.0])
    info["mean_activity_dict"] = {SESSION: mean_activity}
    model = build_sc(sta_dir, data_info=info, a_init=1.5, init_offset_from_mean_activity=True)
    readout = session_readout(model)

    baseline = readout.nl_a * torch.nn.functional.softplus(readout.nl_c)
    assert torch.allclose(baseline, mean_activity, atol=1e-5)


@pytest.mark.parametrize("learn_filters", [False, True])
def test_scheduler_can_follow_the_poisson_loss(sta_dir: Path, learn_filters: bool) -> None:
    model = build_sc(sta_dir, learn_filters=learn_filters, scheduler_metric="poisson")
    configuration = model.configure_optimizers()

    assert configuration["lr_scheduler"]["monitor"] == "val_loss"
    assert configuration["lr_scheduler"]["scheduler"].mode == "min"
    assert len(configuration["optimizer"].param_groups) == (2 if learn_filters else 1)


def test_scheduler_waits_out_the_warm_up(sta_dir: Path) -> None:
    model = build_sc(sta_dir, learn_filters=True, scheduler_metric="poisson", scheduler_warmup_epochs=3)
    configuration = model.configure_optimizers()
    optimizer, scheduler = configuration["optimizer"], configuration["lr_scheduler"]["scheduler"]
    learning_rates = [group["lr"] for group in optimizer.param_groups]

    assert scheduler.warmup_epochs == 3
    for loss in range(20):  # a rising loss: past the warm-up and the patience, the learning rates must drop
        scheduler.step(float(loss))
        if loss < 3 + 5:
            assert [group["lr"] for group in optimizer.param_groups] == learning_rates
    assert all(group["lr"] < lr for group, lr in zip(optimizer.param_groups, learning_rates))


def test_unknown_scheduler_metric_raises(sta_dir: Path) -> None:
    with pytest.raises(ValueError, match="scheduler_metric"):
        build_sc(sta_dir, scheduler_metric="mse")


@pytest.mark.parametrize("learn_filters", [False, True])
def test_checkpoint_loads_without_its_stas(sta_dir: Path, tmp_path: Path, learn_filters: bool) -> None:
    """`load_from_checkpoint` restores the filters from the checkpoint, so the STAs need not exist any more."""
    model = build_sc(sta_dir, learn_filters=learn_filters)
    take_optimizer_step(model)  # move the parameters away from their STA-derived start
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        checkpoint,
    )
    shutil.rmtree(sta_dir)

    loaded = SpatialContrastCoreReadout.load_from_checkpoint(checkpoint, weights_only=False)

    assert loaded.state_dict().keys() == model.state_dict().keys()
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in loaded.state_dict().items())
    x = torch.rand(2, *IN_SHAPE)
    with torch.no_grad():
        assert torch.equal(loaded(x, SESSION), model(x, SESSION))


def build_random_sc(**kwargs) -> SpatialContrastCoreReadout:
    arguments: dict = dict(filter_init="random", learn_filters=True, temporal_filter_length=5)
    arguments.update(kwargs)
    return build_sc(None, **arguments)


def test_random_init_starts_every_neuron_at_the_centre() -> None:
    readout = session_readout(build_random_sc(gaussian_init_sigma=1.5))
    assert isinstance(readout, LearnableFilterSpatialContrastReadout)

    sigma_x, sigma_y = readout.sigmas()
    assert torch.allclose(readout.center_x, torch.full((len(ROI_IDS),), (WIDTH - 1) / 2))
    assert torch.allclose(readout.center_y, torch.full((len(ROI_IDS),), (HEIGHT - 1) / 2))
    assert torch.allclose(sigma_x, torch.tensor(1.5)) and torch.allclose(sigma_y, torch.tensor(1.5))
    # The frozen buffer holds the same Gaussian the learnable parameters describe
    spatial, _, _ = readout._filters()
    assert torch.allclose(spatial, readout.spatial_filters, atol=1e-6)


def test_random_init_draws_the_temporal_filters_from_the_seed() -> None:
    torch.manual_seed(0)
    first = session_readout(build_random_sc()).temporal_filters
    torch.manual_seed(0)
    second = session_readout(build_random_sc()).temporal_filters

    assert first.shape == (len(ROI_IDS), 5)
    assert torch.all(first.abs() <= (3.0 / 5) ** 0.5)
    assert not torch.equal(first[0], first[1])
    assert torch.equal(first, second)


@pytest.mark.parametrize(
    "kwargs", [dict(learn_filters=False), dict(temporal_filter_length=None)], ids=["frozen", "no_length"]
)
def test_random_init_needs_learned_filters_of_a_set_length(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="filter_init='random'"):
        build_random_sc(**kwargs)


def test_random_init_round_trips_through_a_checkpoint(tmp_path: Path) -> None:
    model = build_random_sc()
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

    loaded = SpatialContrastCoreReadout.load_from_checkpoint(checkpoint, weights_only=False)

    assert all(torch.equal(value, model.state_dict()[key]) for key, value in loaded.state_dict().items())
    x = torch.rand(2, *IN_SHAPE)
    with torch.no_grad():
        assert torch.equal(loaded(x, SESSION), model(x, SESSION))


def test_temporal_penalty_ignores_the_norm_of_the_stored_weights() -> None:
    torch.manual_seed(0)
    model = build_random_sc(smooth_weight_temporal=10.0)
    readout = session_readout(model)
    assert isinstance(readout, LearnableFilterSpatialContrastReadout)
    unit_filters = torch.nn.functional.normalize(readout.temporal_weight, dim=-1)
    expected = 10.0 * TimeLaplaceL23dnorm()(unit_filters[:, None, :, None, None])

    with torch.no_grad():
        readout.temporal_weight[0] *= 5.0
    assert torch.allclose(readout.regularizer(), expected)
    assert torch.equal(session_readout(build_random_sc()).regularizer(), torch.tensor(0.0))
    # The penalty adds nothing to the state dict, so older checkpoints still load
    assert set(model.state_dict()) == set(build_random_sc().state_dict())


def test_temporal_penalty_has_the_strength_it_has_in_an_lnp() -> None:
    """Mean loss plus the scaled penalty is the LNP objective, a summed loss plus the penalty, over the elements."""
    torch.manual_seed(0)
    model = build_random_sc(smooth_weight_temporal=10.0)
    x = torch.rand(3, *IN_SHAPE)
    targets = torch.rand(3, IN_SHAPE[1], len(ROI_IDS))
    output = model(x, SESSION)

    mean_objective = model.loss(output, targets) + model.readout_regularization(SESSION, output)
    summed_objective = PoissonLoss3d()(output, targets) + model.readout.regularizer(SESSION)

    assert model.readout.regularizer(SESSION) > 0
    assert torch.allclose(mean_objective * output.numel(), summed_objective)


def test_temporal_penalty_needs_learned_filters(sta_dir: Path) -> None:
    with pytest.raises(ValueError, match="smooth_weight_temporal"):
        build_sc(sta_dir, smooth_weight_temporal=1.0)
