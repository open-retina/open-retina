import os

from openretina.utils.file_utils import get_cache_directory

os.environ["OPENRETINA_CACHE_DIRECTORY"] = get_cache_directory()
