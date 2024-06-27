import pytest

from openretina.optimization.optimization_stopper import EarlyStopper


@pytest.mark.parametrize("min_delta, patience, list_of_losses, list_of_results", [
    # one value
    (0.0, 1, [1.0], [False]),
    # Increasing sequence
    (0.0, 1, [1.0, 1.1, 1.2, 1.3], [False, True, True, True]),
    # Decreasing sequence
    (0.0, 1, [1.0, 0.9, 0.8, 0.7], [False, False, False, False]),
    # Patience 3
    # Increasing sequence
    (0.0, 3, [1.0, 1.1, 1.2, 1.3], [False, False, False, True]),
    # Decreasing sequence
    (0.0, 3, [1.0, 0.9, 0.8, 0.7], [False, False, False, False]),
    # Decreasing, then increasing
    (0.0, 3, [1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.0, 1.1], [False, False, False, False, False, False, True, True]),
    # min_delta > 0
    (0.1, 1, [1.0, 1.01, 1.09, 1.15, 0.9, 1.01], [False, False, False, True, False, True]),
])
def test_early_stopper_default(
        min_delta: float,
        patience: int,
        list_of_losses: list[float],
        list_of_results: list[bool],
):
    early_stopper = EarlyStopper(min_delta=min_delta, patience=patience)

    for loss, desired_result in zip(list_of_losses, list_of_results, strict=True):
        result = early_stopper.early_stop(loss)
        if result != desired_result:
            pass
        assert result == desired_result, f"{list_of_losses=}, {list_of_results=}"
