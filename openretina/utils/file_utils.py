import logging
import os
import shutil
import zipfile
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


def unzip_and_cleanup(zip_path: Path) -> Path:
    """
    Unzips a file and removes the zip archive.

    - If the ZIP contains only one file, extracts directly to the same directory.
    - If it contains multiple files, extracts into a folder named after the ZIP.
    """
    extract_to = zip_path.with_suffix("")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_contents = zf.namelist()

        # If only one file, extract directly
        if len(zip_contents) == 1:
            extracted_file = zip_path.parent / zip_contents[0]
            zf.extract(zip_contents[0], zip_path.parent)
            LOGGER.info(f"Extracted single file to {extracted_file.resolve()}.")

            zip_path.unlink()
            return extracted_file  # Return extracted file path

        # Otherwise, extract into a folder
        shutil.unpack_archive(zip_path, extract_to)
        LOGGER.info(f"Extracted files to folder {extract_to.resolve()}.")

    zip_path.unlink()
    return extract_to  # Return extracted folder path


def download_file(url: str, target_path: Path):
    """Handles file downloading with progress tracking."""
    LOGGER.info(f"Downloading {url} to {target_path}...")
    try:
        with requests.get(url, stream=True) as response:
            if response.status_code != 200:
                raise FileNotFoundError(f"Failed to download {url} (status {response.status_code})")

            total_size = int(response.headers.get("content-length", 0))  # Total size in bytes
            chunk_size = 1024 * 1024  # 1 MB
            with (
                open(target_path, "wb") as f,
                tqdm(
                    total=total_size // chunk_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {target_path.name}",
                ) as progress_bar,
            ):
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        LOGGER.info(f"Download completed: {target_path.resolve()}.")

    except KeyboardInterrupt:
        if target_path.exists():
            target_path.unlink()
            LOGGER.info(f"Partially downloaded file removed due to KeyboardInterrupt: {target_path}")
        raise

    except Exception as e:
        LOGGER.exception(f"Download failed for {url}: {e}")
        raise


def is_target_present(cache_folder: Path, file_name: Path) -> Path | None:
    """Checks if the requested file, an extracted folder, or a single extracted file exists."""
    existing_matches = list(cache_folder.rglob(file_name.stem + "*"))
    if len(existing_matches) > 0:
        found_path = file_name.name if file_name.name in existing_matches else existing_matches[0]
        if found_path.is_dir():
            LOGGER.info(f"Found extracted folder at {found_path.resolve()}")
        elif found_path.suffix == ".zip":
            LOGGER.info(f"Found zip archive at {found_path.resolve()}")
        else:
            LOGGER.info(f"Found extracted file at {found_path.resolve()}")
        return found_path
    return None


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(min=10), reraise=True)
def optionally_download_from_url(
    base_url: str,
    path: str,
    cache_folder: str | os.PathLike | None = None,
    max_num_directories_in_cache_folder: int = 2,
) -> Path:
    if cache_folder is None:
        cache_folder = get_cache_directory()

    cache_folder = Path(cache_folder)
    file_name = Path(*Path(path).parts[-max_num_directories_in_cache_folder:])

    # Check if file or extracted folder already exists
    existing_path = is_target_present(cache_folder, file_name)
    if existing_path is not None:
        if existing_path.suffix == ".zip":
            return unzip_and_cleanup(existing_path)
        return existing_path

    # Download file
    target_path = cache_folder / file_name
    os.makedirs(target_path.parent, exist_ok=True)
    full_url = urljoin(base_url, path)

    download_file(full_url, target_path)

    # If downloaded file is a zip, extract it
    if target_path.suffix == ".zip":
        return unzip_and_cleanup(target_path)

    return target_path


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
