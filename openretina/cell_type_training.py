import bisect
from collections import namedtuple
from typing import Dict, Iterable, List, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from jaxtyping import Float
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from .constants import CLIP_LENGTH, NUM_CLIPS, NUM_VAL_CLIPS, SCENE_LENGTH
from .dataloaders import filter_empty_videos
from .dev_models import (  # ! TODO: move the model to dev models once done developing.
    DEVICE,
    GRUEnabledCore,
    MultiGaussian2d,
    VideoEncoder,
    get_dims_for_loader_dict,
    get_module_output,
    itemgetter,
    set_seed,
)
from .hoefling_2022_data_io import (
    MoviesDict,
    gen_start_indices,
    get_all_movie_combinations,
)
from .hoefling_2022_models import ParametricFactorizedBatchConv3dCore
from .neuron_data_io import NeuronData

DataPoint = namedtuple("DataPoint", ("inputs", "targets"))


def reorganize_data_by_scan_seq(data: Dict[str, Dict[str, np.ndarray]]):
    reorganized_data = {}

    for session, session_data in data.items():
        scan_sequence_idx = session_data["scan_sequence_idx"]
        eye = session_data["eye"]
        group_assignment = session_data["group_assignment"]

        # Create a unique key by concatenating scan_sequence_idx and eye
        unique_key = f"{scan_sequence_idx}_{eye}"

        if unique_key not in reorganized_data:
            reorganized_data[unique_key] = {}

        for group_idx, group in enumerate(group_assignment):
            if group not in reorganized_data[unique_key]:
                reorganized_data[unique_key][group] = {
                    "scan_sequence_idx": scan_sequence_idx,
                    "eye": eye,
                    "session": session,
                }
            # Dynamically assign data entries based on the original dictionary contents
            for key, value in session_data.items():
                #! TODO: Right now we are only collecting the final responses, logic needs to be added to collect other data entries
                if isinstance(value, np.ndarray) and value.ndim > 1 and "final" in key:
                    if key not in reorganized_data[unique_key][group]:
                        reorganized_data[unique_key][group][key] = []
                    reorganized_data[unique_key][group][key].append(value[group_idx])

    # Cast the collected data back to their original data types
    for unique_key, group_data in reorganized_data.items():
        for group, data_entries in group_data.items():
            for key, value in data_entries.items():
                try:
                    reorganized_data[unique_key][group][key] = np.stack(value) if isinstance(value, list) else value
                except ValueError:
                    print(f"Could not stack {key} for {unique_key} and group {group}")
                    print(f"Shapes: {[v.shape for v in value]}")

    return reorganized_data


def append_matching_entries(base_dictionary, append_dictionary, concat_axis=0, fields=None):
    if fields is None:
        fields = append_dictionary.keys()
    if len(base_dictionary) == 0:
        return append_dictionary
    assert all(
        key in append_dictionary.keys() for key in base_dictionary.keys()
    ), f"Missing keys in the append dictionary. \n {base_dictionary.keys()} \n {append_dictionary.keys()}"
    return {
        key: np.concatenate([base_dictionary[key], append_dictionary[key]], axis=concat_axis)
        for key in base_dictionary
        if key in fields
    }


def flatten_session_data(data: Dict[str, Dict[str, np.ndarray]], keep_fields=None):
    if keep_fields is None:
        keep_fields = ["responses_final", "session_name", "group_assignment", "eye", "scan_sequence_idx"]

    flattened_data = {}
    for session, session_data in data.items():
        session_name = np.repeat(session, len(session_data["group_assignment"]))
        eye = np.repeat(session_data["eye"], len(session_data["group_assignment"]))
        scan_sequence_idx = np.repeat(session_data["scan_sequence_idx"], len(session_data["group_assignment"]))
        complete_session_data = session_data | {
            "session_name": session_name,
            "eye": eye,
            "scan_sequence_idx": scan_sequence_idx,
        }
        flattened_data = append_matching_entries(
            flattened_data,
            complete_session_data,
            fields=keep_fields,
        )
    return flattened_data


def reorganize_data_by_cell_type(data: Dict[str, Dict[str, np.ndarray]], cell_n_cutoff: int = 0):
    flattened_data = flatten_session_data(data)
    cell_types, cell_counts = np.unique(flattened_data["group_assignment"], return_counts=True)

    # Filter out cell types with less than cell_n_cutoff cells
    cell_types = cell_types[cell_counts >= cell_n_cutoff]

    # Create a dictionary to store the reorganized data
    reorganized_data = {cell_type: {} for cell_type in cell_types}
    for cell_type in cell_types:
        cell_type_idx = flattened_data["group_assignment"] == cell_type
        for key, value in flattened_data.items():
            reorganized_data[cell_type][key] = value[cell_type_idx]
    return reorganized_data


T_co = TypeVar("T_co", covariant=True)


class ConcatMovieDataset(Dataset[T_co]):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        self.roi_coords = None

        assert self.datasets, "datasets should not be an empty iterable"

        self.cumulative_sizes = self.cumsum(self.datasets)
        self.mean_response = torch.cat([d.mean_response for d in self.datasets])

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class MultipleMovieDataSet(Dataset):
    def __init__(
        self,
        movies: dict,
        responses: np.ndarray,
        roi_ids,
        roi_coords,
        group_assignment,
        eye,
        scan_sequence,
        split,
        chunk_size,
    ):

        # Will only be a dictionary for certain types of datasets, i.e. Hoefling 2022
        if split == "test" and isinstance(responses, dict):
            self.responses = responses[split]["avg"]
            self.roi_ids = roi_ids
            self.group_assignment = group_assignment
        else:
            self.responses = responses[split]
            self.roi_coords = roi_coords
        self.chunk_size = chunk_size
        # Calculate the mean response of the cell type (used for bias init in the model)
        self.mean_response = torch.mean(self.responses)
        self.movies = movies
        self.eye = eye
        self.split = split
        self.scan_sequence = scan_sequence

    def __getitem__(self, idxes):
        neuron_idx, clip_idx = idxes
        movie_eye = self.eye[neuron_idx]
        scan_sequence = self.scan_sequence[neuron_idx] if self.split == "train" else ...
        correct_movie = self.movies[movie_eye][self.split][scan_sequence]

        return DataPoint(
            *[
                correct_movie[:, clip_idx : clip_idx + self.chunk_size, ...],
                self.responses[clip_idx : clip_idx + self.chunk_size, neuron_idx : neuron_idx + 1],  # to keep dim
            ]
        )

    def __len__(self):
        # Returns the number of chunks of clips and responses used for training
        return self.responses.shape[0] // self.chunk_size

    def __str__(self):
        return f"MultipleMovieDataSet with {self.responses.shape[1]} neuron responses to a movie of shape {self.responses.shape[0]}."

    def __repr__(self):
        return str(self)


class MultipleMovieSampler(Sampler):
    def __init__(
        self, clip_start_indices: Float[np.ndarray, "n_indices"], n_neurons, split, chunk_size, scene_length=None
    ):
        self.start_indices = np.array(clip_start_indices)
        self.split = split
        self.n_neurons = n_neurons
        self.chunk_size = chunk_size
        self.scene_length = SCENE_LENGTH if scene_length is None else scene_length

    def __iter__(self):
        if self.split == "train" and (self.scene_length != self.chunk_size):
            # Always start the clip from a random point in the scene, within the chosen chunk size
            shifted_indices = gen_shifts(
                np.arange(0, max(self.start_indices), self.scene_length), self.start_indices, self.chunk_size
            )
            shuffle = True
        else:
            shifted_indices = self.start_indices
            shuffle = False

        neuron_indices = np.arange(self.n_neurons)

        all_indices = np.array(
            [(neuron_idx, start_idx) for neuron_idx in neuron_indices for start_idx in shifted_indices]
        )
        shuffle_indices = np.random.permutation(len(all_indices)) if shuffle else np.arange(len(all_indices))

        return iter(all_indices[shuffle_indices])

    def __len__(self):
        return len(self.start_indices) * self.n_neurons


def get_cell_types_dataloaders(
    neuron_data_dictionary,
    movies_dictionary: MoviesDict,
    train_chunk_size: int = 50,
    batch_size: int = 32,
    seed: int = 42,
    num_clips: int = NUM_CLIPS,
    clip_length: int = CLIP_LENGTH,
    num_val_clips: int = NUM_VAL_CLIPS,
    **kwargs,
):
    assert isinstance(
        neuron_data_dictionary, dict
    ), "neuron_data_dictionary should be a dictionary of sessions and their corresponding neuron data."
    assert (
        isinstance(movies_dictionary, dict) and "train" in movies_dictionary and "test" in movies_dictionary
    ), "movies_dictionary should be a dictionary with keys 'train' and 'test'."
    # assert all(
    #     field in next(iter(neuron_data_dictionary.values())) for field in ["responses_final", "stim_id"]
    # ), "Check the neuron data dictionary sub-dictionaries for the minimal required fields: 'responses_final' and 'stim_id'."

    # assert next(iter(neuron_data_dictionary.values()))["stim_id"] in [
    #     5,
    #     "salamander_natural",
    # ], "This function only supports natural movie stimuli."

    # Draw validation clips based on the random seed
    rnd = np.random.RandomState(seed)
    val_clip_idx = list(rnd.choice(num_clips, num_val_clips, replace=False))

    clip_chunk_sizes = {
        "train": train_chunk_size,
        "validation": clip_length,
        "test": movies_dictionary["test"].shape[1],
    }
    dataloaders = {"train": {}, "validation": {}, "test": {}}

    # Get the random sequences of movies presentatios for each session if available
    if "random_sequences" not in movies_dictionary or movies_dictionary["random_sequences"] is None:
        movie_length = movies_dictionary["train"].shape[1]
        random_sequences = np.arange(0, movie_length // clip_length)[:, np.newaxis]
    else:
        random_sequences = movies_dictionary["random_sequences"]

    movies = get_all_movie_combinations(
        movies_dictionary["train"],
        movies_dictionary["test"],
        random_sequences,
        val_clip_idx=val_clip_idx,
        clip_length=clip_length,
    )
    start_indices = gen_start_indices(
        random_sequences, val_clip_idx, clip_length, train_chunk_size, num_clips, unique_train=True
    )
    for cell_type, cell_type_data in tqdm(neuron_data_dictionary.items(), desc="Creating movie dataloaders"):
        neuron_data = CellTypeNeuronData(
            **cell_type_data,
            stim_id=5,
            random_sequences=random_sequences,  # Used together with the validation index to get the validation response in the corresponding dict
            val_clip_idx=val_clip_idx,
            num_clips=num_clips,
            clip_length=clip_length,
        )

        for fold in ["train", "validation", "test"]:
            dataloaders[fold][str(cell_type)] = get_movie_cell_types_dataloader(
                movies=movies,
                responses=neuron_data.response_dict,
                roi_ids=neuron_data.roi_ids,
                roi_coords=neuron_data.roi_coords,
                group_assignment=neuron_data.group_assignment,
                scan_sequence=neuron_data.scan_sequence_idx,
                split=fold,
                chunk_size=clip_chunk_sizes[fold],
                start_indices=start_indices[fold],
                eye=neuron_data.eye,
                batch_size=batch_size,
                scene_length=clip_length,
                **kwargs,
            )

    return dataloaders


class CellTypeNeuronData(NeuronData):
    def __init__(
        self,
        responses_final: Float[np.ndarray, "n_neurons n_timepoints"] | dict,  # noqa
        stim_id: Literal[5, 2, 1, "salamander_natural"],
        val_clip_idx: List[int],
        num_clips: int,
        clip_length: int,
        roi_coords: Optional[Float[np.ndarray, "n_neurons 2"]] = None,  # noqa
        roi_ids: Optional[Float[np.ndarray, "n_neurons"]] = None,  # noqa
        traces: Optional[Float[np.ndarray, "n_neurons n_timepoints"]] = None,  # noqa
        tracestimes: Optional[Float[np.ndarray, "n_timepoints"]] = None,  # noqa
        scan_sequence_idx: Float[
            np.ndarray, "n_neurons"
        ] = None,  #! Biggest difference with base class! Array here instead of scalar
        random_sequences: Optional[Float[np.ndarray, "n_clips n_sequences"]] = None,  # noqa
        eye: Optional[Float[np.ndarray, "n_neurons"]] = None,
        group_assignment: Optional[Float[np.ndarray, "n_neurons"]] = None,  # noqa
        key: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            responses_final=responses_final,
            stim_id=stim_id,
            val_clip_idx=val_clip_idx,
            num_clips=num_clips,
            clip_length=clip_length,
            roi_coords=roi_coords,
            roi_ids=roi_ids,
            traces=traces,
            tracestimes=tracestimes,
            scan_sequence_idx=scan_sequence_idx,
            random_sequences=random_sequences,
            eye=eye,
            group_assignment=group_assignment,
            key=key,
            **kwargs,
        )

    # Overwrite original method to account for multiple scan sequence indices
    def compute_validation_responses(self):

        # Initialise validation responses for all neurons
        self.responses_val = np.zeros([len(self.val_clip_idx) * self.clip_length, self.num_neurons])

        # Initialise validation mask for all neurons
        validation_mask = np.ones_like(self.responses_train_and_val, dtype=bool)

        for neuron_index, neuron_scan_seq in enumerate(self.scan_sequence_idx):
            movie_ordering = (
                np.arange(self.num_clips)
                if (len(self.random_sequences) == 0 or self.scan_sequence_idx is None)
                else self.random_sequences[:, neuron_scan_seq]
            )

            # Sort the movie ordering based on the scan sequence
            base_movie_sorting = np.argsort(movie_ordering)

            # Compute validation responses and remove sections from training responses for this neuron
            for i, ind1 in enumerate(self.val_clip_idx):
                grab_index = base_movie_sorting[ind1]
                self.responses_val[i * self.clip_length : (i + 1) * self.clip_length, neuron_index] = (
                    self.responses_train_and_val[
                        grab_index * self.clip_length : (grab_index + 1) * self.clip_length, neuron_index
                    ]
                )
                validation_mask[
                    (grab_index * self.clip_length) : (grab_index + 1) * self.clip_length,
                    neuron_index,
                ] = False

        self.responses_train = self.responses_train_and_val[validation_mask].reshape(-1, self.num_neurons)


def get_movie_cell_types_dataloader(
    movies: dict,
    responses: Float[np.ndarray, "n_neurons n_frames"],  # noqa
    roi_ids: Float[np.ndarray, "n_neurons"],  # noqa
    roi_coords: Float[np.ndarray, "n_neurons 2"],  # noqa
    group_assignment: Float[np.ndarray, "n_neurons"],  # noqa
    split: str,
    start_indices: Float[np.ndarray, "n_indices"],  # noqa
    eye: Float[np.ndarray, "n_neurons"],  # noqa
    scan_sequence: Float[np.ndarray, "n_neurons"],  # noqa
    chunk_size: int = 50,
    batch_size: int = 32,
    scene_length: Optional[int] = None,
    drop_last=True,
    **kwargs,
):

    dataset = MultipleMovieDataSet(
        movies, responses, roi_ids, roi_coords, group_assignment, eye, scan_sequence, split, chunk_size
    )
    sampler = MultipleMovieSampler(
        start_indices, n_neurons=len(group_assignment), split=split, chunk_size=chunk_size, scene_length=scene_length
    )

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=bool((split == "train" and drop_last)),
        collate_fn=filter_empty_videos,
        **kwargs,
    )


def gen_shifts(clip_bounds, start_indices, clip_chunk_size=50):

    def get_next_bound(value, bounds):
        insertion_index = bisect.bisect_right(bounds, value)
        return bounds[min(insertion_index, len(bounds) - 1)]

    shifted_indices = []
    shifts = np.random.randint(1, clip_chunk_size // 2, len(start_indices))

    for i, start_idx in enumerate(start_indices):
        if start_idx + shifts[i] + clip_chunk_size < (get_next_bound(start_idx, clip_bounds)):
            shifted_indices.append(start_idx + shifts[i])
        else:
            shifted_indices.append(start_idx)
    return shifted_indices


def naive_cell_types_model(
    dataloaders,
    seed,
    hidden_channels: Tuple[int] = (8,),  # core args
    temporal_kernel_size: Tuple[int] = (21,),
    spatial_kernel_size: Tuple[int] = (11,),
    layers: int = 1,
    gamma_hidden: float = 0,
    gamma_input: float = 0.1,
    gamma_temporal: float = 0.1,
    gamma_in_sparse=0.0,
    final_nonlinearity: bool = True,
    core_bias: bool = False,
    momentum: float = 0.1,
    input_padding: bool = False,
    hidden_padding: bool = True,
    batch_norm: bool = True,
    batch_norm_scale: bool = False,
    laplace_padding=None,
    batch_adaptation: bool = True,
    readout_scale: bool = False,
    readout_bias: bool = True,
    gamma_readout: float = 0.1,
    stack=None,
    use_avg_reg: bool = False,
    data_info: dict = None,
    nonlinearity: str = "ELU",
    conv_type: Literal["full", "separable", "custom_separable", "time_independent"] = "custom_separable",
    device=DEVICE,
    # use_gru: bool = False,
    # gru_kwargs: dict = {},
):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout
    Args:
        dataloaders: a dictionary of dataloaders, one loader per sessionin the format:
            {'train': {'session1': dataloader1, 'session2': dataloader2, ...},
             'validation': {'session1': dataloader1, 'session2': dataloader2, ...},
             'test': {'session1': dataloader1, 'session2': dataloader2, ...}}
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts
    Returns: An initialized model which consists of model.core and model.readout
    """

    # make sure trainloader is being used
    if data_info is not None:
        in_shapes_dict = {k: v["input_dimensions"] for k, v in data_info.items()}
        input_channels = [v["input_channels"] for k, v in data_info.items()]
        n_neurons_dict = {k: v["output_dimension"] for k, v in data_info.items()}
        roi_masks = {k: torch.tensor(v["roi_coords"]) for k, v in data_info.items()}
    else:
        dataloaders = dataloaders.get("train", dataloaders)

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name, *_ = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        print(session_shape_dict)
        n_neurons_dict = {k: 1 for k in session_shape_dict.keys()}  # ! Set to one for now, for naive model
        in_shapes_dict = {
            k: v[in_name] for k, v in session_shape_dict.items()
        }  # dictionary containing input shapes per session
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]  # gets the # of input channels
        roi_masks = {k: dataloaders[k].dataset.roi_coords for k in dataloaders.keys()}  # TODO: implement
    assert np.unique(input_channels).size == 1, "all input channels must be of equal size"

    set_seed(seed)

    # get a stacked factorized 3d core from below
    core = ParametricFactorizedBatchConv3dCore(
        n_neurons_dict=n_neurons_dict,
        input_channels=input_channels[0],
        num_scans=len(n_neurons_dict.keys()),
        hidden_channels=hidden_channels,
        temporal_kernel_size=temporal_kernel_size,
        spatial_kernel_size=spatial_kernel_size,
        layers=layers,
        gamma_hidden=gamma_hidden,
        gamma_input=gamma_input,
        gamma_in_sparse=gamma_in_sparse,
        gamma_temporal=gamma_temporal,
        final_nonlinearity=final_nonlinearity,
        bias=core_bias,
        momentum=momentum,
        input_padding=input_padding,
        hidden_padding=hidden_padding,
        batch_norm=batch_norm,
        batch_norm_scale=batch_norm_scale,
        laplace_padding=laplace_padding,
        stack=stack,
        batch_adaptation=batch_adaptation,
        use_avg_reg=use_avg_reg,
        nonlinearity=nonlinearity,
        conv_type=conv_type,
        device=device,
    )

    in_shapes_readout = {}
    subselect = itemgetter(0, 2, 3)
    for k in n_neurons_dict:  # iterate over sessions
        in_shapes_readout[k] = subselect(tuple(get_module_output(core, in_shapes_dict[k])[1:]))

    readout = MultiGaussian2d(
        in_shape_dict=in_shapes_readout,
        n_neurons_dict=n_neurons_dict,
        scale=readout_scale,
        bias=readout_bias,
        feature_reg_weights=gamma_readout,
    )

    # initializing readout bias to mean response
    if readout_bias is True:
        if data_info is None:
            for k in dataloaders:
                readout[k].bias.data = dataloaders[k].dataset.mean_response
        else:
            for k in data_info.keys():
                readout[k].bias.data = torch.from_numpy(data_info[k]["mean_response"])

    model = VideoEncoder(
        core,
        readout,
    )

    return model


## Old stuff

# from openretina.dataloaders import MovieDataSet, MovieSampler, filter_different_size


# def get_movie_dataset(
#     movies: Union[np.ndarray, Dict[int, np.ndarray]],
#     responses: Float[np.ndarray, "n_neurons n_frames"],  # noqa
#     roi_ids: Float[np.ndarray, "n_neurons"],  # noqa
#     roi_coords: Float[np.ndarray, "n_neurons 2"],  # noqa
#     group_assignment: Float[np.ndarray, "n_neurons"],  # noqa
#     split: str,
#     scan_sequence_idx: Optional[int] = None,
#     chunk_size: int = 50,
#     **kwargs,
# ):
#     if split == "train" and isinstance(movies, dict) and scan_sequence_idx is not None:
#         dataset = MovieDataSet(
#             movies[scan_sequence_idx],
#             responses,
#             roi_ids,
#             roi_coords,
#             group_assignment,
#             split,
#             chunk_size,
#         )
#     else:
#         dataset = MovieDataSet(
#             movies, responses, roi_ids, roi_coords, group_assignment, split, chunk_size
#         )

#     return dataset


# def get_cell_types_dataloader(
#     cell_type_datasets,
#     split: str,
#     start_indices: Union[List[int], Dict[int, List[int]]],
#     chunk_size: int = 50,
#     batch_size: int = 32,
#     scene_length: Optional[int] = None,
#     **kwargs,
# ):
#     dataset = ConcatMovieDataset(cell_type_datasets)

#     sampler = None  # MovieSampler(start_indices, split, chunk_size, scene_length=scene_length)

#     return DataLoader(
#         dataset,
#         sampler=sampler,
#         batch_size=batch_size,
#         drop_last=True if split == "train" else False,
#         collate_fn=filter_different_size,
#         **kwargs,
#     )

# class MoviesDict(TypedDict):
#     train: np.ndarray
#     test: np.ndarray
#     random_sequences: Optional[np.ndarray]


# def cell_type_dataloaders(
#     neuron_data_dictionary,
#     movies_dictionary: MoviesDict,
#     train_chunk_size: int = 50,
#     batch_size: int = 32,
#     seed: int = 42,
#     num_clips: int = NUM_CLIPS,
#     clip_length: int = CLIP_LENGTH,
#     num_val_clips: int = NUM_VAL_CLIPS,
# ):
#     assert isinstance(
#         neuron_data_dictionary, dict
#     ), "neuron_data_dictionary should be a dictionary of sessions and their corresponding neuron data."
#     assert (
#         isinstance(movies_dictionary, dict)
#         and "train" in movies_dictionary
#         and "test" in movies_dictionary
#     ), "movies_dictionary should be a dictionary with keys 'train' and 'test'."

#     # Draw validation clips based on the random seed
#     rnd = np.random.RandomState(seed)
#     val_clip_idx = list(rnd.choice(num_clips, num_val_clips, replace=False))

#     clip_chunk_sizes = {
#         "train": train_chunk_size,
#         "validation": clip_length,
#         "test": movies_dictionary["test"].shape[1],
#     }
#     dataloaders = {"train": {}, "validation": {}, "test": {}}
#     final_datasets = {"train": {}, "validation": {}, "test": {}}

#     # Get the random sequences of movies presentatios for each session if available
#     if (
#         "random_sequences" not in movies_dictionary
#         or movies_dictionary["random_sequences"] is None
#     ):
#         movie_length = movies_dictionary["train"].shape[1]
#         random_sequences = np.arange(0, movie_length // clip_length)[:, np.newaxis]
#     else:
#         random_sequences = movies_dictionary["random_sequences"]

#     movies = get_all_movie_combinations(
#         movies_dictionary["train"],
#         movies_dictionary["test"],
#         random_sequences,
#         val_clip_idx=val_clip_idx,
#         clip_length=clip_length,
#     )
#     start_indices = gen_start_indices(
#         random_sequences, val_clip_idx, clip_length, train_chunk_size, num_clips
#     )
#     # Loop through unique eye and scan_sequence_idx combinations
#     for unique_key, group_data in neuron_data_dictionary.items():
#         for cell_type, data_entries in group_data.items():
#             neuron_data = NeuronData(
#                 **data_entries,
#                 stim_id=5,
#                 random_sequences=random_sequences,
#                 val_clip_idx=val_clip_idx,
#                 num_clips=num_clips,
#                 clip_length=clip_length,
#             )

#             for fold in ["train", "validation", "test"]:
#                 existing_datasets = final_datasets[fold].get(cell_type, [])
#                 existing_datasets.append(
#                     get_movie_dataset(
#                         movies=movies[neuron_data.eye][fold],
#                         responses=neuron_data.response_dict[fold],
#                         roi_ids=neuron_data.roi_ids,
#                         roi_coords=neuron_data.roi_coords,
#                         group_assignment=neuron_data.group_assignment,
#                         split=fold,
#                         start_indices=start_indices[fold],
#                         chunk_size=clip_chunk_sizes[fold],
#                         scan_sequence_idx=neuron_data.scan_sequence_idx,
#                     )
#                 )
#                 final_datasets[fold][cell_type] = existing_datasets

#     for fold, fold_dataset in final_datasets.items():
#         for cell_type, cell_type_datasets in fold_dataset.items():
#             dataloaders[fold][str(cell_type)] = get_cell_types_dataloader(
#                 cell_type_datasets=cell_type_datasets,
#                 split=fold,
#                 start_indices=(
#                     start_indices[fold]
#                     if isinstance(start_indices[fold], list)
#                     else start_indices[fold][0]
#                 ),  # in the training case, but they are not the same TODO: fix this.
#                 batch_size=32,
#             )

#     return dataloaders
