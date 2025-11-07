# Data I/O

`openretina` supports any dataset where both the visual stimuli (inputs) and the retinal responses (outputs) can be expressed as time-aligned sequences. During training, the package consumes these data as tuples `(input_t, output_t)` emitted by PyTorch dataloaders.

Throughout the documentation we use *session* or *experimental session* to describe a single recording of retinal tissue under controlled conditions. Each session includes both the stimulus presentation and the corresponding neural responses, ensuring a shared temporal reference for inputs and outputs. Unless stated otherwise, we describe the general case of datasets that contain multiple sessions.

If you are still surveying which datasets are already available, take a look at the [dataset reference](../datasets.md). For an end-to-end overview of how data flows into the training pipeline, see the [training guide](../training/index.md).

## Integrating a New Dataset

You can bring a new dataset into `openretina` in two main ways:

1. **Write custom loaders** for your native data format and register the configuration with the package.
2. **Convert the data to the built-in HDF5 layout**, which is the format native to our utilities.

Both approaches are covered in this page. Worked examples are also available as an interactive tutorial notebook in the repository: [notebooks/new\_datasets\_guide.ipynb](https://github.com/open-retina/open-retina/blob/main/notebooks/new_datasets_guide.ipynb).

## Loading Stimuli

Movies are represented as 4D tensors with shape `(channels, time, height, width)`. In typical experiments, stimuli are delivered as separate training and testing sequences, with testing parts often repeated to estimate response reliability. The `MoviesTrainTestSplit` data class captures this structure.

### Quick start

```python
from openretina.data_io.base import MoviesTrainTestSplit
import numpy as np

example_stimulus = MoviesTrainTestSplit(
    train=np.random.rand(1, 200, 50, 50),  # channels, train_time, height, width
    test=np.random.rand(1, 100, 50, 50),   # channels, test_time, height, width
    stim_id=None,
    norm_mean=None,
    norm_std=None,
)
```

When dealing with multiple sessions, group the movies in a dictionary where keys are session identifiers and values are `MoviesTrainTestSplit` instances.

### Example: Maheswaranathan et al. (2023)

```python
from openretina.data_io.maheswaranathan_2023.stimuli import load_all_stimuli

# Primary download: https://doi.org/10.25740/rk663dm5577
# Mirror: OpenRetina HuggingFace hub

movies_dictionary = load_all_stimuli(
    base_data_path=...,          # Path to the "ganglion_cell_data" folder in the archive
    stim_type="naturalscene",   # "whitenoise" is also available
    normalize_stimuli=True,      # Recommended for stable training
)
```

The resulting dictionary maps session names to `MoviesTrainTestSplit` objects:

```python
{
    "15-10-07": MoviesTrainTestSplit(train=numpy.ndarray(shape=(1, 359802, 50, 50)),
                                      test=numpy.ndarray(shape=(1, 5996, 50, 50)),
                                      stim_id="naturalscene",
                                      random_sequences=None,
                                      norm_mean=51.4911,
                                      norm_std=53.6271),
    "15-11-21a": MoviesTrainTestSplit(...),
    "15-11-21b": MoviesTrainTestSplit(...),
}
```

## Loading Neural Responses

Neural activity for each session is stored as a 2D tensor `(neurons, time)`, aligned with the stimulus frames. The `ResponsesTrainTestSplit` class mirrors the structure of `MoviesTrainTestSplit` and optionally stores trial-resolved repetitions as well as session-level metadata.

### Quick start

```python
from openretina.data_io.base import ResponsesTrainTestSplit
import numpy as np

example_responses = ResponsesTrainTestSplit(
    train=np.random.rand(60, 200),      # neurons, train_time
    test=np.random.rand(60, 100),       # neurons, test_time
    test_by_trial=np.random.rand(10, 60, 100),  # trials, neurons, test_time
    stim_id=None,
    session_kwargs={"cell_types": np.random.randint(1, 42, 60)},
)
```

As with stimuli, organise multi-session recordings in a dictionary keyed by session names.

### Example: Maheswaranathan et al. (2023)

```python
from openretina.data_io.maheswaranathan_2023.responses import load_all_responses

responses_dictionary = load_all_responses(
    base_data_path=...,              # Same root used for stimuli
    response_type="firing_rate_20ms",  # 5 ms and 10 ms binning also available
    stim_type="naturalscene",
)
```

Typical output looks like:

```python
{
    "15-10-07": ResponsesTrainTestSplit(train=numpy.ndarray(shape=(9, 359802)),
                                          test=numpy.ndarray(shape=(9, 5997)),
                                          test_by_trial=None,
                                          stim_id="naturalscene",
                                          session_kwargs={}),
    "15-11-21a": ResponsesTrainTestSplit(...),
    "15-11-21b": ResponsesTrainTestSplit(...),
}
```

## Composing Complete Dataloaders

Once stimuli and responses are stored in matching dictionaries, the next step before training is to create PyTorch dataloaders that yield clipped stimulus-response pairs. This is handled by utilities in `openretina.data_io.base_dataloader`.

```python
from openretina.data_io.base_dataloader import multiple_movies_dataloaders

dataloaders = multiple_movies_dataloaders(
    neuron_data_dictionary=responses_dictionary,
    movies_dictionary=movies_dictionary,
    train_chunk_size=50,
    batch_size=64,
    seed=42,
    clip_length=90,
    num_val_clips=50,
    allow_over_boundaries=True,
)
```

The returned object is a dictionary with `train`, `validation`, and `test` keys, each of which contains session-specific `torch.utils.data.DataLoader` instances.

### Single-session helper

In rapid-prototyping scenarios, you may want a single dataloader for a single movie/response pair. Use `get_movie_dataloader` for that case:

```python
import numpy as np
from openretina.data_io.base_dataloader import get_movie_dataloader

random_stimulus = np.random.rand(1, 300, 50, 50)  # channels, time, height, width
random_responses = np.random.rand(300, 42)        # time, neurons

single_dataloader = get_movie_dataloader(
    movie=random_stimulus,
    responses=random_responses,
    split="train",
    scene_length=300,
    chunk_size=50,
    batch_size=1,
)
```

For guidance on how these dataloaders plug into the training loop, refer back to the [training overview](../training/index.md).

## HDF5-based Data Structure

If your lab's export format is still evolving, it can be simpler to adapt the preprocessing pipeline than to maintain custom Python loaders. `openretina` therefore ships with a reference HDF5 structure that works out-of-the-box with the utilities above.

Each session lives in its own `.hdf5` file, while the corresponding stimuli are stored as NumPy arrays under a `stimuli/` subdirectory.

You can generate synthetic data that matches the expected layout via the command-line interface (see the [CLI reference](../command_line.md)):

```bash
# Print help
openretina create-data --help

# Create data in ./test_data
openretina create-data ./test_data --num-colors 3 --num-stimuli 4 --num-sessions 2
```

Running this command produces two `.hdf5` files—one per session—in `test_data/`, plus stimulus files under `test_data/stimuli/`. Each HDF5 file contains a `session_info` attribute where you can store metadata such as cell types.

To keep stimuli and responses aligned, ensure they share the same sampling frequency and number of temporal bins. All stimuli should also have identical channel counts and spatial dimensions. Once exported, the data can be loaded with the helpers in `openretina.data_io.h5_dataset_reader` and used directly in existing configuration files (for example the `hoefling_2024` setup described in [data\_io/hoefling\_2024.md](./hoefling_2024.md)).

With these options, you should be ready to import new retinal datasets into `openretina` and move on to model training.
