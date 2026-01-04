import os
import tempfile
from pathlib import Path

import pytest

from openretina.utils.file_utils import get_file_size, get_local_file_path, is_target_present


@pytest.mark.parametrize(
    "possibly_remote_file_path",
    [
        # single file
        "https://huggingface.co/datasets/open-retina/open-retina/"
        "blob/main/euler_lab/hoefling_2024/responses/"
        "rgc_natsim_subset_only_naturalspikes_2024-08-14.h5",
    ],
)
def test_get_local_file_path(possibly_remote_file_path: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        local_file_path = get_local_file_path(possibly_remote_file_path, cache_folder=temp_dir)
        assert is_target_present(Path(temp_dir), local_file_path)
        date = os.path.getmtime(local_file_path)
        # Call the function a second time to see if caching works
        local_file_path_two = get_local_file_path(possibly_remote_file_path, cache_folder=temp_dir)
        assert local_file_path == local_file_path_two
        date_two = os.path.getmtime(local_file_path)
        assert date == date_two


@pytest.mark.parametrize(
    "file_path, existing_file_paths, expect_match",
    [
        # file_name should not match grey scale model (even though they have same prefix)
        (
            "hoefling_2024_base_low_res.ckpt",
            ["hoefling_2024_base_low_res_grey_scale.ckpt"],
            False,
        ),
        # zip file that got unpacked
        (
            "baccus_lab/maheswaranathan_2023/neural_code_data.zip",
            ["baccus_lab/maheswaranathan_2023/neural_code_data"],
            True,
        ),
        # trivial case, two times the same file
        (
            "hoefling_2024_base_low_res.ckpt",
            ["hoefling_2024_base_low_res.ckpt"],
            True,
        ),
        # same file names in different folders should not match
        (
            "2025-01-01/hoefling_2024_base_low_res.ckpt",
            ["2025-12-12/hoefling_2024_base_low_res.ckpt"],
            False,
        ),
    ],
)
def test_is_target_present(file_path: str, existing_file_paths: list[str], expect_match: bool) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        for fp in existing_file_paths:
            cache_path = Path(temp_dir) / fp
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            cache_path.touch(exist_ok=True)

        local_path = is_target_present(Path(temp_dir), Path(file_path))
        local_path_is_none = local_path is not None
        assert local_path_is_none == expect_match


@pytest.mark.parametrize(
    "url",
    [
        # single file
        "https://huggingface.co/datasets/open-retina/open-retina/"
        "blob/main/euler_lab/hoefling_2024/responses/"
        "rgc_natsim_subset_only_naturalspikes_2024-08-14.h5",
        # zip file
        "https://huggingface.co/datasets/open-retina/open-retina/resolve"
        "/main/baccus_lab/maheswaranathan_2023/neural_code_data.zip",
    ],
)
def test_get_file_size(url: str) -> None:
    size = get_file_size(url)
    assert size > 0
