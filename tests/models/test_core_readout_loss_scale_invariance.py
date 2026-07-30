import pytest
import torch

from openretina.models.core_readout import BaseCoreReadout
from openretina.modules.core.base_core import SimpleCoreWrapper
from openretina.modules.losses.poisson import PoissonLoss3d
from openretina.modules.readout.multi_readout import MultiGaussianMaskReadout

DATA_KEY = "session"
EXTRA_SESSION_NEURON_COUNTS = {"other_session_a": 4, "other_session_b": 11}
IN_CHANNELS = 2
HEIGHT = 8
WIDTH = 8

XFAIL_SCALING_DEFAULTS = pytest.mark.xfail(
    reason="PoissonLoss3d defaults to a summed reduction and the readout regularizer defaults to a "
    "summed-over-neurons reduction; both scale with batch size / time window / neuron count. "
    "Remove this marker once the defaults are changed to be scale-invariant.",
    strict=True,
)


@pytest.fixture(scope="module")
def core() -> SimpleCoreWrapper:
    """
    Shared across all tests in this file: the core's regularizer and forward pass don't depend on
    `n_neurons`, batch size, or time steps, so there's no need to rebuild it per test/model.
    """
    torch.manual_seed(0)
    return SimpleCoreWrapper(
        channels=(IN_CHANNELS, 4, 4),
        temporal_kernel_sizes=(3, 3),
        spatial_kernel_sizes=(3, 3),
        gamma_input=0.1,  # activates spatial_laplace
        gamma_hidden=0.1,  # activates group_sparsity
        gamma_temporal=0.1,  # activates temporal_smoothness
        gamma_in_sparse=0.1,  # activates group_sparsity_0
        cut_first_n_frames=0,
        input_padding=True,
        hidden_padding=True,
    )


def _build_model(core: SimpleCoreWrapper, n_neurons: int, with_multiple_sessions: bool = False) -> BaseCoreReadout:
    """
    Builds a CoreReadout model with every available core and readout regularization term active,
    using the library's default (summed) loss and regularizer reductions -- the same ones used by
    `BaseCoreReadout.training_step`. The readout is registered under `DATA_KEY`; if
    `with_multiple_sessions` is set, additional sessions (with fixed, unrelated neuron counts) are
    added alongside it, to check that their mere presence doesn't affect `DATA_KEY`'s loss/regularizer.
    """
    n_neurons_dict = {DATA_KEY: n_neurons}
    if with_multiple_sessions:
        n_neurons_dict.update(EXTRA_SESSION_NEURON_COUNTS)

    with torch.no_grad():
        dummy = torch.ones(1, IN_CHANNELS, 3, HEIGHT, WIDTH)
        readout_in_shape = tuple(core(dummy).shape[1:])

    readout = MultiGaussianMaskReadout(
        in_shape=readout_in_shape,
        n_neurons_dict=n_neurons_dict,
        scale=True,
        bias=True,
        gaussian_mean_scale=1.0,
        gaussian_var_scale=1.0,
        positive=False,
        mask_l1_reg=0.5,  # activates mask_l1
        feature_weights_l1_reg=0.3,  # activates feature_l1
    )
    # Overwrite the random init with fixed constants so the readout regularizer's value is exactly
    # comparable across models built with a different `n_neurons` (a differently-shaped random
    # tensor from the same seed is not an apples-to-apples comparison).
    with torch.no_grad():
        for session_key in n_neurons_dict:
            session_readout = readout[session_key]
            session_readout.features.fill_(0.5)
            session_readout.mask_log_var.fill_(0.0)  # type: ignore
            session_readout.mask_mean.fill_(0.0)  # type: ignore

    model = BaseCoreReadout(core=core, readout=readout, learning_rate=1e-3, loss=PoissonLoss3d())
    model.eval()
    return model


def _total_loss(model: BaseCoreReadout, batch_size: int, time_steps: int, with_regularization: bool) -> torch.Tensor:
    x = torch.ones(batch_size, IN_CHANNELS, time_steps, HEIGHT, WIDTH)
    with torch.no_grad():
        output = model.forward(x, data_key=DATA_KEY)
        target = torch.ones_like(output)
        loss = model.loss(output, target)
        if with_regularization:
            loss = loss + model.core.regularizer() + model.readout.regularizer(DATA_KEY)
    return loss


@XFAIL_SCALING_DEFAULTS
@pytest.mark.parametrize("with_regularization", [False, True], ids=["prediction_loss_only", "with_all_regularization"])
@pytest.mark.parametrize("with_multiple_sessions", [False, True], ids=["single_session", "multiple_sessions"])
def test_loss_invariant_to_batch_size(
    core: SimpleCoreWrapper, with_multiple_sessions: bool, with_regularization: bool
) -> None:
    model = _build_model(core, n_neurons=5, with_multiple_sessions=with_multiple_sessions)
    time_steps = 6

    reference_loss = _total_loss(model, batch_size=1, time_steps=time_steps, with_regularization=with_regularization)
    for batch_size in (2, 4, 8):
        loss = _total_loss(model, batch_size=batch_size, time_steps=time_steps, with_regularization=with_regularization)
        assert torch.allclose(loss, reference_loss, atol=1e-6), (
            f"Loss scales with batch size: {reference_loss=}, {loss=}, {batch_size=}"
        )


@XFAIL_SCALING_DEFAULTS
@pytest.mark.parametrize("with_regularization", [False, True], ids=["prediction_loss_only", "with_all_regularization"])
@pytest.mark.parametrize("with_multiple_sessions", [False, True], ids=["single_session", "multiple_sessions"])
def test_loss_invariant_to_time_steps(
    core: SimpleCoreWrapper, with_multiple_sessions: bool, with_regularization: bool
) -> None:
    model = _build_model(core, n_neurons=5, with_multiple_sessions=with_multiple_sessions)
    batch_size = 2

    reference_loss = _total_loss(model, batch_size=batch_size, time_steps=4, with_regularization=with_regularization)
    for time_steps in (8, 16, 32):
        loss = _total_loss(model, batch_size=batch_size, time_steps=time_steps, with_regularization=with_regularization)
        assert torch.allclose(loss, reference_loss, atol=1e-6), (
            f"Loss scales with time window: {reference_loss=}, {loss=}, {time_steps=}"
        )


@XFAIL_SCALING_DEFAULTS
@pytest.mark.parametrize("with_regularization", [False, True], ids=["prediction_loss_only", "with_all_regularization"])
@pytest.mark.parametrize("with_multiple_sessions", [False, True], ids=["single_session", "multiple_sessions"])
def test_loss_invariant_to_number_of_neurons(
    core: SimpleCoreWrapper, with_multiple_sessions: bool, with_regularization: bool
) -> None:
    batch_size = 2
    time_steps = 6

    reference_model = _build_model(core, n_neurons=3, with_multiple_sessions=with_multiple_sessions)
    reference_loss = _total_loss(
        reference_model, batch_size=batch_size, time_steps=time_steps, with_regularization=with_regularization
    )
    for n_neurons in (7, 20):
        model = _build_model(core, n_neurons=n_neurons, with_multiple_sessions=with_multiple_sessions)
        loss = _total_loss(model, batch_size=batch_size, time_steps=time_steps, with_regularization=with_regularization)
        assert torch.allclose(loss, reference_loss, atol=1e-6), (
            f"Loss scales with number of neurons: {reference_loss=}, {loss=}, {n_neurons=}"
        )
