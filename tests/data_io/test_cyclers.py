import pytest
from torch.utils.data import IterableDataset, DataLoader

from openretina.data_io.cyclers import LongCycler, ShortCycler


class Dataset(IterableDataset):
    def __init__(self, name: str, num_elements: int):
        self._name = name
        self._num_elements = num_elements

    def __iter__(self):
        for i in range(self._num_elements):
            yield f"{self._name}_{i}"

    def __len__(self) -> int:
        return self._num_elements


def get_dataloaders(num_elements_per_loader: list[int]) -> dict[str, DataLoader]:
    dataloaders = {}
    for i, num_elements in enumerate(num_elements_per_loader):
        name = f"dataset{i}"
        ds = Dataset(name, num_elements)
        dl = DataLoader(ds, drop_last=False)
        dataloaders[name] = dl
    return dataloaders


def get_all_elements_from_dataloaders(dataloaders) -> list:
    elements_list = [[x[0] for x in dl] for dl in dataloaders]
    elements = sum(elements_list, [])
    return elements


def get_all_elements_from_cycler_dataloader(dataloader) -> list:
    all_elements_lists = list(dataloader)
    all_elements = [x[1][0][0] for x in all_elements_lists]
    return all_elements


@pytest.mark.parametrize(
    "num_workers",
    [0, 1, 4],
)
def test_short_cycler(num_workers: int):
    dataloaders = get_dataloaders([1, 20, 5, 10])

    all_elements_dataloader = sorted(get_all_elements_from_dataloaders(dataloaders.values()))
    cycler = ShortCycler(dataloaders)
    dl_cycler = DataLoader(cycler, num_workers=num_workers)
    all_elements_cycler = sorted(get_all_elements_from_cycler_dataloader(dl_cycler))
    if num_workers <= 1:
        assert len(all_elements_cycler) == len(all_elements_dataloader)
    else:
        # for multiple processes we might duplicate data
        assert len(all_elements_cycler) >= len(all_elements_dataloader)
    assert set(all_elements_cycler) == set(all_elements_dataloader)


@pytest.mark.parametrize(
    "num_workers, shuffle",
    [
        (0, False),
        (0, True),
        (1, False),
        (1, True),
        (4, False),
        (4, True),
    ],
)
def test_long_cycler(num_workers: int, shuffle: bool):
    dataloaders = get_dataloaders([1, 20, 5, 1])

    all_elements_dataloader = get_all_elements_from_dataloaders(dataloaders.values())
    cycler = LongCycler(dataloaders, shuffle=shuffle)
    dl_cycler = DataLoader(cycler, num_workers=num_workers, drop_last=False)
    all_elements_cycler = get_all_elements_from_cycler_dataloader(dl_cycler)

    # cycler can produce more elements if the individual dataloaders have different sizes
    assert len(all_elements_dataloader) <= len(all_elements_cycler)
    # cycler should produce all elements in dataloaders
    assert set(all_elements_dataloader) == set(all_elements_cycler)
