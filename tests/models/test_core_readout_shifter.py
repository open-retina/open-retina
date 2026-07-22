import pytest
import torch

from openretina.models.core_readout import BaseCoreReadout
from openretina.modules.core.base_core import SimpleCoreWrapper
from openretina.modules.readout.multi_readout import MultiSampledGaussianReadout
from openretina.modules.shifters.mlp_shifter import MultiSessionMLPShifter

N_NEURONS_DICT = {"sess1": 5}
IN_SHAPE = (3, 20, 36, 64)  # channels, time, height, width
BATCH_SIZE = 2


def _build_core() -> SimpleCoreWrapper:
    return SimpleCoreWrapper(
        channels=[IN_SHAPE[0], 8, 8],
        temporal_kernel_sizes=[5, 5],
        spatial_kernel_sizes=[5, 5],
        gamma_input=0.0,
        gamma_temporal=0.0,
        gamma_in_sparse=0.0,
        gamma_hidden=0.0,
        input_padding=False,
        hidden_padding=True,
        cut_first_n_frames=0,
    )


def _build_readout(core: SimpleCoreWrapper) -> MultiSampledGaussianReadout:
    readout_in_shape = BaseCoreReadout.compute_readout_input_shape(
        BaseCoreReadout.__new__(BaseCoreReadout), IN_SHAPE, core
    )
    return MultiSampledGaussianReadout(
        in_shape=readout_in_shape,
        n_neurons_dict=N_NEURONS_DICT,
        bias=True,
        init_mu_range=0.1,
        init_sigma_range=0.3,
        gauss_type="full",
        grid_mean_predictor=None,
        gamma=1.0,
    )


def _build_shifter(gamma_shifter: float = 0.0) -> MultiSessionMLPShifter:
    return MultiSessionMLPShifter(
        n_neurons_dict=N_NEURONS_DICT,
        input_channels=2,
        hidden_channels=5,
        n_layers=3,
        gamma_shifter=gamma_shifter,
    )


@pytest.fixture
def core_and_readout() -> tuple[SimpleCoreWrapper, MultiSampledGaussianReadout]:
    core = _build_core()
    readout = _build_readout(core)
    return core, readout


def test_forward_without_shifter_is_unaffected_by_pupil_kwarg(core_and_readout) -> None:
    """Regression guard: a model with no shifter must ignore `pupil_center` entirely."""
    core, readout = core_and_readout
    model = BaseCoreReadout(core=core, readout=readout, learning_rate=1e-3)
    model.eval()  # fix the Gaussian readout grid to its mean so repeated forward calls are deterministic
    assert model.shifter is None

    x = torch.randn(BATCH_SIZE, *IN_SHAPE)
    pupil = torch.randn(BATCH_SIZE, 2, IN_SHAPE[1])

    out_no_pupil = model.forward(x, "sess1")
    out_with_pupil = model.forward(x, "sess1", pupil_center=pupil)

    assert torch.equal(out_no_pupil, out_with_pupil)


def test_forward_with_shifter_but_no_pupil_matches_no_shifter_model(core_and_readout) -> None:
    """A shifter-equipped model with `pupil_center=None` must behave exactly like one without a shifter."""
    core, readout = core_and_readout
    shifter = _build_shifter()
    model_with_shifter = BaseCoreReadout(core=core, readout=readout, learning_rate=1e-3, shifter=shifter)
    model_without_shifter = BaseCoreReadout(core=core, readout=readout, learning_rate=1e-3)
    model_with_shifter.eval()
    model_without_shifter.eval()

    x = torch.randn(BATCH_SIZE, *IN_SHAPE)

    out_with_shifter_module = model_with_shifter.forward(x, "sess1", pupil_center=None)
    out_without_shifter_module = model_without_shifter.forward(x, "sess1")

    assert torch.equal(out_with_shifter_module, out_without_shifter_module)


def test_forward_with_shifter_and_pupil_shifts_the_output(core_and_readout) -> None:
    """When both a shifter and `pupil_center` are supplied, the shift should actually change the output.

    The model stays in train mode: a freshly-initialized `SimpleCoreWrapper`'s conv weights are tiny by
    design, and in eval mode `BatchNorm3d` normalizes with its untrained (mean=0, var=1) running stats,
    passing that tiny signal through nearly unchanged and washing out any detectable shift effect. Train
    mode uses live batch statistics instead, giving a non-degenerate signal. The Gaussian readout's grid
    sampling is stochastic in train mode, so we fix the RNG seed identically before each call to isolate
    the shift as the only source of difference between the two forward passes.
    """
    core, readout = core_and_readout
    shifter = _build_shifter()
    model = BaseCoreReadout(core=core, readout=readout, learning_rate=1e-3, shifter=shifter)

    x = torch.randn(BATCH_SIZE, *IN_SHAPE)
    pupil = torch.randn(BATCH_SIZE, 2, IN_SHAPE[1])

    torch.manual_seed(0)
    out_unshifted = model.forward(x, "sess1", pupil_center=None)
    torch.manual_seed(0)
    out_shifted = model.forward(x, "sess1", pupil_center=pupil)

    assert out_shifted.shape == out_unshifted.shape
    assert not torch.allclose(out_shifted, out_unshifted)


def test_forward_output_time_dimension_matches_pupil_alignment(core_and_readout) -> None:
    """`T_out` from the core determines how many leading pupil frames get dropped; shapes must stay consistent."""
    core, readout = core_and_readout
    shifter = _build_shifter()
    model = BaseCoreReadout(core=core, readout=readout, learning_rate=1e-3, shifter=shifter)

    x = torch.randn(BATCH_SIZE, *IN_SHAPE)
    output_core = core(x)
    t_out = output_core.size(2)
    assert t_out < IN_SHAPE[1]  # core must reduce the time dimension for this test to be meaningful

    pupil = torch.randn(BATCH_SIZE, 2, IN_SHAPE[1])
    out = model.forward(x, "sess1", pupil_center=pupil)

    assert out.shape == (BATCH_SIZE, t_out, N_NEURONS_DICT["sess1"])


def test_shifter_regularizer_included_in_training_step_when_present(core_and_readout) -> None:
    core, readout = core_and_readout
    shifter = _build_shifter(gamma_shifter=0.5)
    model = BaseCoreReadout(core=core, readout=readout, learning_rate=1e-3, shifter=shifter)

    output_core = core(torch.zeros(1, *IN_SHAPE))
    t_out = output_core.size(2)

    from openretina.data_io.qiu_2026.dataloaders import QiuDataPoint

    data_point = QiuDataPoint(
        inputs=torch.randn(BATCH_SIZE, *IN_SHAPE),
        targets=torch.rand(BATCH_SIZE, t_out, N_NEURONS_DICT["sess1"]),
        pupil_center=torch.randn(BATCH_SIZE, 2, IN_SHAPE[1]),
    )

    total_loss = model.training_step(("sess1", data_point), 0)

    assert torch.isfinite(total_loss)
    # The shifter's own regularizer is 0.0 by construction (MLPShifter.regularizer), so gamma_shifter
    # only scales a constant zero here; this just guards that the code path runs end-to-end without error.


def test_training_step_still_works_with_plain_datapoint(core_and_readout) -> None:
    """Regression guard: an existing (non-qiu) 2-field DataPoint batch must still train unchanged."""
    from openretina.data_io.base_dataloader import DataPoint

    core, readout = core_and_readout
    model = BaseCoreReadout(core=core, readout=readout, learning_rate=1e-3)

    output_core = core(torch.zeros(1, *IN_SHAPE))
    t_out = output_core.size(2)

    data_point = DataPoint(
        inputs=torch.randn(BATCH_SIZE, *IN_SHAPE),
        targets=torch.rand(BATCH_SIZE, t_out, N_NEURONS_DICT["sess1"]),
    )

    total_loss = model.training_step(("sess1", data_point), 0)
    assert torch.isfinite(total_loss)
