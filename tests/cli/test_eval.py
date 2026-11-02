"""Tests for ``openretina.cli.eval``."""

import math

from openretina.cli.eval import _mean_over_splits, _tagged_output_path


def test_tag_is_inserted_before_the_suffix() -> None:
    assert _tagged_output_path("eval/per_neuron.csv", "validation") == "eval/per_neuron_validation.csv"


def test_directory_and_suffix_are_preserved() -> None:
    tagged = _tagged_output_path("/abs/run/eval/summary.json", "test")
    assert tagged == "/abs/run/eval/summary_test.json"


def test_base64_split_names_cannot_escape_the_directory() -> None:
    """qiu_2026 keys its test splits by DataJoint ``condition_hash``, which is base64.

    A hash ending in ``/`` would otherwise turn the filename into a directory component, writing the
    results somewhere other than where the caller asked (or failing on a missing directory), and a
    ``+`` is merely awkward to type back. Both must come out as plain characters in one filename.
    """
    tagged = _tagged_output_path("eval/per_neuron.csv", "F5KOXJGuqZljKK9USGu/")
    assert tagged == "eval/per_neuron_F5KOXJGuqZljKK9USGu_.csv"

    tagged_plus = _tagged_output_path("eval/per_neuron.csv", "Oup5uAZxF2G7zEJkT+ui")
    assert tagged_plus == "eval/per_neuron_Oup5uAZxF2G7zEJkT_ui.csv"


def test_distinct_splits_stay_distinct() -> None:
    """Sanitising must not collide two of the six conditions onto one output file."""
    conditions = [
        "5zQTb77qI+ig8rigx1XU",
        "7UETOWO5Z8aWuHDBJ2GG",
        "GjCMo2GkJp6y5vricadg",
        "KXdTNAGMo1gCWz2Ge8zr",
        "Oup5uAZxF2G7zEJkT+ui",
        "ecUQJtcERZJGdqza1k7h",
    ]
    paths = {_tagged_output_path("eval/per_neuron.csv", c) for c in conditions}
    assert len(paths) == len(conditions)


def test_plus_and_slash_collide_and_so_must_be_rejected_not_sanitised_apart() -> None:
    """``+`` and ``/`` both sanitise to ``_``, so two such hashes CAN name one file.

    This pins the premise behind the collision check in ``evaluate_model``: the tag is deliberately
    not collision-free, so the caller is refused rather than silently losing one split's results.
    """
    assert _tagged_output_path("eval/per_neuron.csv", "AB+CD") == _tagged_output_path("eval/per_neuron.csv", "AB/CD")


def test_mean_over_splits_ignores_splits_that_scored_nan() -> None:
    """One split with no neurons above the cutoff must not turn the whole run's score nan."""
    assert _mean_over_splits([0.6, float("nan"), 0.8]) == 0.7


def test_mean_over_splits_is_nan_only_when_every_split_is() -> None:
    assert math.isnan(_mean_over_splits([float("nan"), float("nan")]))
