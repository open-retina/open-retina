import pytest
import torch

from openretina.models.linear_nonlinear import SingleCellSeparatedLNP


@pytest.mark.parametrize("smooth_weight_temp", (0.0, 1.0, 10.0))
def test_single_cell_separated_lnp(smooth_weight_temp: float) -> None:
    in_shape = (3, 10, 15, 15)  # "channel time height width"
    torch.manual_seed(0)

    model = SingleCellSeparatedLNP(
        in_shape,
        smooth_weight_temp=smooth_weight_temp,
    )

    x = torch.rand((1, *in_shape))
    x = model.crop_input(x)
    with torch.no_grad():
        y = model(x)
        regularization = model.regularizer()
    assert y.shape[0] == 1
    assert torch.isfinite(regularization)
    assert regularization >= 0
