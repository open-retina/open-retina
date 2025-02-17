import logging
import os
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests
import tenacity
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)
HOME_DIR = str(Path.home())
GIN_BASE_URL = "https://gin.g-node.org/"
HUGGINGFACE_BASE_URL = "https://huggingface.co/"
HUGGINGFACE_REPO_ID = "open-retina/open-retina"


def get_cache_directory() -> str:
    home_dir = os.path.expanduser("~")
    return os.getenv("OPENRETINA_CACHE_DIRECTORY", os.path.join(home_dir, "openretina_cache"))


def unzip_and_cleanup(zip_path: Path) -> Path:
    """
    Unzips a file and removes the zip archive.

    - If the ZIP contains only one file, extracts directly to the same directory.
    - If the ZIP contains a single top-level folder named after the ZIP, extracts without duplicating the folder.
    - If it contains multiple files, extracts into a folder named after the ZIP.
    """
    extract_to = zip_path.with_suffix("")
    target_path = extract_to
    with zipfile.ZipFile(zip_path, "r") as zf:
        LOGGER.info(f"Extracting {zip_path}...")
        zip_contents = zf.namelist()

        # If only one file, extract directly
        if len(zip_contents) == 1:
            extracted_file = zip_path.parent / zip_contents[0]
            zf.extract(zip_contents[0], zip_path.parent)
            LOGGER.info(f"Extracted single file to {extracted_file.resolve()}.")

            zip_path.unlink()
            return extracted_file  # Return extracted file path

        # Check if zip contains a single folder with the same name as the zip
        top_level_items = {Path(item).parts[0] for item in zip_contents}  # Get unique top-level names
        if len(top_level_items) == 1 and next(iter(top_level_items)) == extract_to.name:
            extract_to = zip_path.parent  # Extract directly to parent folder. Target path to return will stay the same.

        shutil.unpack_archive(zip_path, extract_to)
        LOGGER.info(f"Extracted files to {target_path.resolve()}.")

    zip_path.unlink()
    return target_path  # Return extracted file path


def get_file_size(url: str, response: requests.Response | None = None) -> int:
    """Get file size from a URL.

    Attempts to retrieve the size of a file from a URL by making a HEAD request
    and checking the response headers. It handles redirects and supports both 'Content-Length'
    and 'X-Linked-Size' headers.

    """
    try:
        response = requests.head(url, allow_redirects=True) if response is None else response

        # If the server returns a redirect (e.g., 302), follow the location in the 'Location' header
        while response.status_code == 302:
            # Extract the new URL from the 'Location' header and follow the redirect
            url = response.headers["Location"]
            response = requests.head(url, allow_redirects=True)

        file_size = 0
        if "Content-Length" in response.headers:
            file_size = int(response.headers["Content-Length"])
        elif "X-Linked-Size" in response.headers:
            file_size = int(response.headers["X-Linked-Size"])

        return file_size

    except Exception as e:
        LOGGER.exception(f"Error occurred while retrieving file-size: {e}")
        return 0


def download_file(url: str, target_path: Path):
    """Handles file downloading with progress tracking."""
    LOGGER.info(f"Downloading {url} to {target_path}...")
    try:
        with requests.get(url, stream=True) as response:
            if response.status_code != 200:
                raise FileNotFoundError(f"Failed to download {url} (status {response.status_code})")

            total_size = get_file_size(url, response=response)  # Total size in bytes
            chunk_size = 1024 * 1024  # 1 MB
            with (
                open(target_path, "wb") as f,
                tqdm(
                    total=total_size,
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
        # Prioritize an exact file match
        for match in existing_matches:
            if match.name == file_name.name:
                LOGGER.info(f"Found target file at {match.resolve()}")
                return match

        # Otherwise, look for valid folders or zip files
        for match in existing_matches:
            if match.is_dir():
                LOGGER.info(f"Found extracted folder at {match.resolve()}")
                return match
            if match.suffix == ".zip":
                LOGGER.info(f"Found zip archive at {match.resolve()}")
                return match

    # No matches found, or temp files which are not folders or zip files
    return None


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(min=10), reraise=True)
def optionally_download_from_url(
    base_url: str,
    path: str,
    cache_folder: str | os.PathLike | None = None,
    max_num_directories_in_cache_folder: int | None = 2,
) -> Path:
    if cache_folder is None:
        cache_folder = get_cache_directory()

    cache_folder = Path(cache_folder)
    if max_num_directories_in_cache_folder is not None:
        file_name = Path(*Path(path).parts[-max_num_directories_in_cache_folder:])
    else:
        file_name = Path(path)

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


def download_file_from_huggingface(file_name: str, local_dir: str | Path | None = None, unzip: bool = True):
    """Download a single file from the Hugging Face Hub."""
    if local_dir is None:
        local_dir = get_cache_directory()
    # Check if file already exists, or its un-zipped version
    existing_path = is_target_present(Path(local_dir), Path(file_name))
    if existing_path is not None:
        if existing_path.suffix == ".zip":
            return unzip_and_cleanup(existing_path)
        return existing_path

    LOGGER.info(f"Downloading {file_name}...")
    downloaded_file = hf_hub_download(
        repo_id=HUGGINGFACE_REPO_ID,
        filename=file_name,
        local_dir=local_dir,
        repo_type="dataset",
    )

    if unzip and downloaded_file.endswith(".zip"):
        return unzip_and_cleanup(Path(downloaded_file))

    return downloaded_file


def huggingface_download(target_name: str, local_dir: str | Path | None = None, unzip: bool = True) -> Path:
    """Downloads a file or folder from HuggingFace dataset repository.

    This function downloads either a single file or all files in a folder from a HuggingFace
    dataset repository. If the target exists as a file, it downloads that specific file.
    If the target is a folder, it downloads all files within that folder.

    Args:
        target_name (str): Name of the file or folder to download from the repository. It is the relative path of
            the file or folder within the open-retina HuggingFace repository.
        local_dir (Union[str, Path, None], optional): Local directory where files should be downloaded.
            If None, uses the default cache directory. Defaults to None.
        unzip (bool, optional): Whether to automatically unzip downloaded files.
            Defaults to True.

    Returns:
        Path: Path to the downloaded file or folder. Returns None if no files
        are found for the given target_name.
    """
    if local_dir is None:
        local_dir = get_cache_directory()

    api = HfApi()

    # Get the list of all files in the repo
    LOGGER.info(f"Fetching file list for {HUGGINGFACE_REPO_ID}...")
    files = api.list_repo_files(HUGGINGFACE_REPO_ID, repo_type="dataset")

    if target_name in files:  # Direct match, download the file
        downloaded_file = download_file_from_huggingface(target_name, local_dir, unzip)
        return Path(downloaded_file)
    else:  # Otherwise, assume it's a folder
        folder_files = [file for file in files if file.startswith(target_name)]
        if not folder_files:
            LOGGER.info(f"No files found in the folder '{target_name}'.")
            raise EntryNotFoundError(f"No files found in the folder '{target_name}'.")

        for file in folder_files:
            download_file_from_huggingface(file, local_dir, unzip)

        return Path(os.path.join(local_dir, target_name))


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

    elif file_path.startswith(HUGGINGFACE_BASE_URL):
        target_relative_path = file_path.split("main/")[-1]
        return huggingface_download(target_relative_path, Path(cache_folder))

    elif file_path.startswith("https://"):
        raise NotImplementedError(f"Only {GIN_BASE_URL} data storage is supported for now.")

    else:
        local_path = Path(os.path.expanduser(file_path))
        if not local_path.exists():
            raise ValueError(f"Local file path {local_path} does not exists")
        return local_path
