import pytest
import torch

from openretina.models.core_readout import _MODEL_NAME_TO_REMOTE_LOCATION, load_core_readout_from_remote


@pytest.mark.parametrize("model_name", sorted(_MODEL_NAME_TO_REMOTE_LOCATION.keys()))
def test_load_core_readout_from_remote(model_name: str) -> None:
    num_batches = 1

    model = load_core_readout_from_remote(model_name, "cpu")
    stimulus = torch.rand(model.stimulus_shape(time_steps=50, num_batches=num_batches))
    responses = model.forward(stimulus)

    assert responses.shape[0] == num_batches
