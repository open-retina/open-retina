from typing import Optional

import pytest
import torch

from openretina.legacy.nnfabrik_model_loading import Center, load_ensemble_model_from_remote


@pytest.mark.parametrize(
    "session_id, center_readout",
    [
        ("2_ventral2_20201016", None),
        ("1_ventral1_20201021", None),
        ("2_ventral1_20201021", Center(target_mean=(0.0, 0.0))),
    ],
)
def test_loading_model_from_remote(session_id: str, center_readout: Optional[Center]) -> None:
    data_info, ensemble_model = load_ensemble_model_from_remote(device="cpu", center_readout=center_readout)
    assert session_id in data_info

    # run a forward pass with a zero tensor of batch size 1
    input_dim = data_info[session_id]["input_dimensions"]
    input_dim_batch_size_one = (1,) + input_dim[1:]
    inp_ = torch.zeros(input_dim_batch_size_one)
    out = ensemble_model(inp_, session_id)

    expected_output_dim = (1, input_dim[2] - 30, data_info[session_id]["output_dimension"])
    assert out.shape == expected_output_dim
