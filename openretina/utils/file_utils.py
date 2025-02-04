import logging
import os
from pathlib import Path
from urllib.parse import urljoin

import requests
import tenacity
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)
HOME_DIR = str(Path.home())
GIN_BASE_URL = "https://gin.g-node.org/"


def get_cache_directory() -> str:
    home_dir = os.path.expanduser("~")
    return os.getenv("OPENRETINA_CACHE_DIRECTORY", os.path.join(home_dir, "openretina_cache"))


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(min=10), reraise=True)
def optionally_download_from_url(
    base_url: str,
    path: str,
    cache_folder: str | os.PathLike | None = None,
    max_num_directories_in_cache_folder: int = 2,
) -> Path:
    if cache_folder is None:
        cache_folder = get_cache_directory()
    download_file_name = Path(*Path(path).parts[-max_num_directories_in_cache_folder:])
    target_download_path = Path(cache_folder) / download_file_name
    os.makedirs(Path(target_download_path).parent, exist_ok=True)
    if not os.path.exists(target_download_path):
        full_url = urljoin(base_url, path)
        LOGGER.info(f"Downloading {full_url} to {target_download_path}...")

        try:
            with requests.get(full_url, stream=True) as response:
                if response.status_code != 200:
                    raise FileNotFoundError(
                        f"Received status code {response.status_code} when trying to download from {full_url=}"
                    )
                total_size = int(response.headers.get("content-length", 0))  # Total size in bytes
                chunk_size = 1024 * 1024  # 1 MB
                with (
                    open(target_download_path, "wb") as f,
                    tqdm(
                        total=total_size // chunk_size,
                        unit="MB",
                        unit_scale=True,
                        desc=f"Downloading {download_file_name} to {target_download_path}",
                    ) as progress_bar,
                ):
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        progress_bar.update(len(chunk) // chunk_size)

            LOGGER.info(f"Completed download of {full_url} to {target_download_path.resolve()}.")

        except KeyboardInterrupt:
            if target_download_path.exists():
                target_download_path.unlink()
                LOGGER.info(f"Partially downloaded file removed: {target_download_path}")
            raise

        except Exception as e:
            LOGGER.error(f"Download failed for {full_url}. Error: {e}")
            raise

    else:
        LOGGER.info(f"Target file for {base_url}/{path} already exists at {target_download_path.resolve()}.")
    return target_download_path


def get_local_file_path(file_path: str, cache_folder: str | os.PathLike | None = None) -> Path:
    if cache_folder is None:
        cache_folder = get_cache_directory()
    if file_path.startswith(GIN_BASE_URL):
        # gin node
        return optionally_download_from_url(
            GIN_BASE_URL,
            file_path[len(GIN_BASE_URL) :],
            os.path.expanduser(cache_folder),
        )
    elif file_path.startswith("https://"):
        raise NotImplementedError(f"Only {GIN_BASE_URL} data storage is supported for now.")
    else:
        local_path = Path(os.path.expanduser(file_path))
        if not local_path.exists():
            raise ValueError(f"Local file path {local_path} does not exists")
        return local_path
