from setuptools import setup, find_packages

setup(
    name="openretina_dataloading",
    version="0.1.0",
    description="Numpy-based data loading functionality for OpenRetina",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "h5py", 
        "jaxtyping",
        "einops",
        "scipy",
        "tqdm",
    ],
    python_requires=">=3.8",
    package_data={
        "openretina_dataloading": ["py.typed"],
    },
)