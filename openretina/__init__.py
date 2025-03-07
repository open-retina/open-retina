import os


def get_cache_directory() -> str:
    home_dir = os.path.expanduser("~")
    return os.getenv("OPENRETINA_CACHE_DIRECTORY", os.path.join(home_dir, "openretina_cache"))


os.environ["OPENRETINA_CACHE_DIRECTORY"] = get_cache_directory()
