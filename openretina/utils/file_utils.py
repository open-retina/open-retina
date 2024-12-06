import logging
import os
from pathlib import Path

import requests
import tenacity

LOGGER = logging.getLogger(__name__)
_DEFAULT_CACHE_DIRECTORY = "openretina_cache_folder"
GIN_BASE_URL = "https://gin.g-node.org/"


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(min=10))
def optionally_download_from_url(
    base_url: str,
    path: str,
    cache_folder: str = _DEFAULT_CACHE_DIRECTORY,
) -> Path:
    cache_path = Path(cache_folder) / Path(path)
    os.makedirs(Path(cache_path).parent, exist_ok=True)
    if not os.path.exists(cache_path):
        full_url = f"{base_url}/{path}"
        LOGGER.info(f"Downloading {full_url} ...")
        response = requests.get(full_url)

        if response.status_code != 200:
            raise FileNotFoundError(
                f"Received status code {response.status_code} " f"when trying to download from {full_url=}"
            )
        with open(cache_path, "wb") as f:
            f.write(response.content)
        LOGGER.info(f"Stored contents of {full_url} in {cache_path}")
    return cache_path


def get_local_file_path(file_path: str, cache_folder: str) -> Path:
    if file_path.startswith(GIN_BASE_URL):
        # gin node
        return optionally_download_from_url(
            GIN_BASE_URL,
            file_path[len(GIN_BASE_URL) :],
            os.path.expanduser(cache_folder),
        )
    elif file_path.startswith("http"):
        raise ValueError(f"Downloading from http {file_path=} is not supported yet.")
    else:
        # assuming this is a local file
        return Path(os.path.expanduser(file_path))
