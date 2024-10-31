import os
from pathlib import Path
import requests


MODEL_CACHE_DIRECTORY = "./model-cache"


def optionally_download(
    base_url: str,
    path: str,
) -> str:
    model_cache_path = f"{MODEL_CACHE_DIRECTORY}/{path}"
    os.makedirs(Path(model_cache_path).parent, exist_ok=True)
    if not os.path.exists(model_cache_path):
        full_url = f"{base_url}/{path}"
        response = requests.get(full_url)
        if response.status_code != 200:
            raise FileNotFoundError(
                f"Received status code {response.status_code} " f"when trying to download from {full_url=}"
            )
        with open(model_cache_path, "wb") as f:
            f.write(response.content)
    return model_cache_path
