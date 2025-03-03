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
