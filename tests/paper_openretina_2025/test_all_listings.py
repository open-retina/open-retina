import pathlib
import runpy
import pytest

@pytest.mark.skip
@pytest.mark.parametrize('listing_path', list(pathlib.Path(__file__).parent.resolve().glob('listing_*.py')))
def test_listing_scripts(listing_path: pathlib.PosixPath):
    # TODO: rerun once uploaded newly trained models.
    print(f"Running script: {listing_path.resolve()}")
    runpy.run_path(str(listing_path))

