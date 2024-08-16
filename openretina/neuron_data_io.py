import pickle
import warnings
from collections import defaultdict, namedtuple
from copy import deepcopy
from typing import Dict, List, Literal, Optional, no_type_check

import numpy as np
import torch
from jaxtyping import Float
from tqdm.auto import tqdm

from .hoefling_2024.constants import RGC_GROUP_NAMES_DICT, STIMULI_IDS

SingleNeuronInfoStruct = namedtuple(
    "SingleNeuronInfoStruct",
    [
        "neuron_id",  # identifier of the neuron
        "neuron_position",  # position in the tensor containing all responses of the session in the neural network
        "session_id",  # session_id passed to the neural network
        "roi_size",  # size of the neuron measured experimentally
        "celltype",  # Celltype according to an automatic classifier trained on labels following Baden (2016)
        "celltype_confidences",  # Confidences of the classifier for each celltype (celltype == confidence.argmax() + 1)
        "training_mean",  # Mean firing rate estimated on the training corpus
        "training_std",  # Standard deviation of firing rate estimated on the training corpus
    ],
    defaults=["None", -1, "None", -1, -1, np.zeros((46,)), 0.0, 1.0],
)


class NeuronGroupMembersStore:
    """Store all member of each neuron group in a dictionary and splits them in a train and test set"""

    def __init__(
        self,
        key,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
    ):
        self._all_neurons: List[SingleNeuronInfoStruct] = []
        self._group_to_neuron_dict: Dict[int, List[SingleNeuronInfoStruct]] = defaultdict(list)
        self._selection_keys = key
        self._train_ratio = train_ratio
        self._validation_ratio = validation_ratio

    def initialize_groups(self, neuron_list: List[SingleNeuronInfoStruct]):
        for neuron_struct in neuron_list:
            self._group_to_neuron_dict[neuron_struct.celltype].append(neuron_struct)
        self._all_neurons = neuron_list

    def reset(self):
        self._all_neurons = []
        self._group_to_neuron_dict = defaultdict(list)

    # Load and save functions were proposed by gpt, unsure if they work in all situations
    def save_to_file(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load_from_file(cls, file_path: str):
        with open(file_path, "rb") as f:
            obj_dict = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(obj_dict)
        return obj

    def estimate_mean_and_variance_on_training_data(
        self, dataloader_dict: Dict[str, torch.utils.data.dataloader.DataLoader]
    ) -> None:
        new_neuron_list: List[SingleNeuronInfoStruct] = []
        for session_id, dataloader in dataloader_dict.items():
            batches_targets = torch.concat([x.targets for x in dataloader], dim=0)
            targets = torch.concat(list(batches_targets), dim=0)
            means = targets.mean(dim=0)
            targets_centered = targets - means
            std = targets_centered.std(dim=0)

            neurons_in_session = {
                neuron_info.neuron_position: neuron_info
                for neuron_info in self._all_neurons
                if neuron_info.session_id == session_id
            }
            target_neurons = targets.shape[0]
            for pos in range(target_neurons):
                if pos in neurons_in_session:
                    old_neuron = neurons_in_session[pos]
                    neuron = SingleNeuronInfoStruct(
                        neuron_id=old_neuron.neuron_id,
                        neuron_position=old_neuron.neuron_position,
                        session_id=old_neuron.session_id,
                        roi_size=old_neuron.roi_size,
                        celltype=old_neuron.celltype,
                        training_mean=means[pos].item(),
                        training_std=std[pos].item(),
                        celltype_confidences=old_neuron.celltype_confidences,
                    )
                    new_neuron_list.append(neuron)

        diff_neurons = {n.neuron_id for n in self._all_neurons} - {n.neuron_id for n in new_neuron_list}
        if len(diff_neurons) > 0:
            raise ValueError(f"Could not find mean and std for {len(diff_neurons)} neurons")
        self.reset()
        self.initialize_groups(new_neuron_list)

    def get_training_samples_for_group(
        self,
        group_id: int,
        min_confidence: float = 0.0,
        min_neurons_per_group: int = 1,
    ) -> List[SingleNeuronInfoStruct]:
        neuron_list = self._group_to_neuron_dict[group_id]
        max_id = int(len(neuron_list) * self._train_ratio)
        neuron_list = neuron_list[:max_id]
        neuron_list_filtered = [n for n in neuron_list if n.celltype_confidences.max() > min_confidence]
        if len(neuron_list_filtered) < min_neurons_per_group:
            neuron_list_filtered = []
        return neuron_list_filtered

    def get_validation_samples_for_group(
        self, group_id: int, min_confidence: float = 0.0
    ) -> List[SingleNeuronInfoStruct]:
        neuron_struct_list = self._group_to_neuron_dict[group_id]
        min_id = int(len(neuron_struct_list) * self._train_ratio)
        max_id = int(len(neuron_struct_list) * (self._train_ratio + self._validation_ratio))
        neuron_struct_list = neuron_struct_list[min_id:max_id]
        neuron_struct_list = [n for n in neuron_struct_list if n.celltype_confidences.max() > min_confidence]
        return neuron_struct_list

    def get_test_samples_for_group(self, group_id: int, min_confidence: float = 0.0) -> List[SingleNeuronInfoStruct]:
        neuron_struct_list = self._group_to_neuron_dict[group_id]
        min_id = int(len(neuron_struct_list) * (self._train_ratio + self._validation_ratio))
        neuron_struct_list = neuron_struct_list[min_id:]
        neuron_struct_list = [n for n in neuron_struct_list if n.celltype_confidences.max() > min_confidence]
        return neuron_struct_list

    def get_all_training_samples(
        self,
        list_of_ids: List[int],
        min_confidence: float = 0.0,
        min_neurons_per_group: int = 1,
    ) -> List[SingleNeuronInfoStruct]:
        training_samples = sum(
            (
                self.get_training_samples_for_group(
                    group_id,
                    min_confidence=min_confidence,
                    min_neurons_per_group=min_neurons_per_group,
                )
                for group_id in list_of_ids
            ),
            [],
        )
        return training_samples

    def get_all_validation_samples(
        self, list_of_ids: List[int], min_confidence: float = 0.0
    ) -> List[SingleNeuronInfoStruct]:
        valid_samples = sum(
            (
                self.get_validation_samples_for_group(group_id, min_confidence=min_confidence)
                for group_id in list_of_ids
            ),
            [],
        )
        return valid_samples

    def get_all_test_samples(self, list_of_ids: List[int], min_confidence: float = 0.0) -> List[SingleNeuronInfoStruct]:
        test_samples = sum(
            (self.get_test_samples_for_group(group_id, min_confidence=min_confidence) for group_id in list_of_ids),
            [],
        )
        return test_samples

    def __str__(self):
        summary_ar = ["Train / Test number of neurons per group"]
        for group_id in range(1, 47):
            train_samples = self.get_training_samples_for_group(group_id)
            valid_samples = self.get_validation_samples_for_group(group_id)
            test_samples = self.get_test_samples_for_group(group_id)
            summary_ar.append(
                f"Group id {group_id} ({RGC_GROUP_NAMES_DICT[group_id]}): "
                f"{len(train_samples)} / {len(valid_samples)} / {len(test_samples)}"
            )
        return "\n".join(summary_ar)

    def __repr__(self):
        return self.__str__()


class NeuronData:
    def __init__(
        self,
        responses_final: Float[np.ndarray, "n_neurons n_timepoints"] | dict,  # noqa
        stim_id: Literal[5, 2, 1, "salamander_natural"],
        val_clip_idx: Optional[List[int]],
        num_clips: Optional[int],
        clip_length: Optional[int],
        roi_mask: Optional[Float[np.ndarray, "64 64"]] = None,  # noqa
        roi_ids: Optional[Float[np.ndarray, "n_neurons"]] = None,  # noqa
        traces: Optional[Float[np.ndarray, "n_neurons n_timepoints"]] = None,  # noqa
        tracestimes: Optional[Float[np.ndarray, "n_timepoints"]] = None,  # noqa
        scan_sequence_idx: Optional[int] = None,
        random_sequences: Optional[Float[np.ndarray, "n_clips n_sequences"]] = None,  # noqa
        eye: Optional[Literal["left", "right"]] = None,
        group_assignment: Optional[Float[np.ndarray, "n_neurons"]] = None,  # noqa
        key: Optional[dict] = None,
        use_base_sequence: Optional[bool] = False,
        **kwargs,
    ):
        """
        Initialize the NeuronData object.
        Boilerplate class to store neuron data. Added for backwards compatibility with Hoefling et al., 2022.

        Args:
            eye (str): The eye from which the neuron data is recorded.
            group_assignment (Float[np.ndarray, "n_neurons"]): The group assignment of neurons.
            key (dict): The key information for the neuron data,
                        includes date, exp_num, experimenter, field_id, stim_id.
            responses_final (Float[np.ndarray, "n_neurons n_timepoints"]) or
                dictionary with train and test responses of similar structure: The responses of neurons.
            roi_coords (Float[np.ndarray, "n_neurons 2"]): The coordinates of regions of interest (ROIs).
            roi_ids (Float[np.ndarray, "n_neurons"]): The IDs of regions of interest (ROIs).
            scan_sequence_idx (int): The index of the scan sequence.
            stim_id (int): The ID of the stimulus. 5 is mouse natural scenes.
            traces: The traces of the neuron data.
            tracestimes: The timestamps of the traces.
            random_sequences (Float[np.ndarray, "n_clips n_sequences"]): The random sequences of clips.
            val_clip_idx (List[int]): The indices of validation clips.
            num_clips (int): The number of clips.
            clip_length (int): The length of each clip.
            use_base_sequence (bool): Whether to re-order all training responses to use the same "base" sequence.
        """
        self.neural_responses = responses_final

        self.num_neurons = (
            self.neural_responses["train"].shape[0]
            if isinstance(self.neural_responses, dict)
            else self.neural_responses.shape[0]
        )

        self.eye = eye if eye is not None else "right"
        self.group_assignment = group_assignment
        self.key = key
        self.roi_ids = roi_ids
        self.roi_coords = (
            np.array(self.transform_roi_mask(roi_mask), dtype=np.float32) if roi_mask is not None else None
        )
        self.scan_sequence_idx = scan_sequence_idx
        self.stim_id = stim_id
        self.traces = traces
        self.tracestimes = tracestimes
        self.clip_length = clip_length
        self.num_clips = num_clips
        self.random_sequences = random_sequences if random_sequences is not None else np.array([])
        self.use_base_sequence = use_base_sequence

    # this has to become a regular method in the future!
    @property
    def response_dict(self):
        if self.stim_id == "salamander_natural":
            # Transpose the responses to have the shape (n_timepoints, n_neurons)
            self.responses_test = self.neural_responses["test"].T
            self.responses_train_and_val = self.neural_responses["train"].T
            self.test_responses_by_trial = []

        elif self.stim_id in [1, 2]:
            # Chirp and moving bar
            self.responses_test = np.nan
            self.test_responses_by_trial = np.nan

            self.responses_val = np.nan

            self.responses_train = self.neural_responses.T

        else:
            assert self.clip_length is not None, "Clip length must be provided for natural scenes"
            assert self.num_clips is not None, "Number of clips must be provided for natural scenes"
            assert self.val_clip_idx is not None, "Validation clip indices must be provided for natural scenes"

            self.responses_test = np.zeros((5 * self.clip_length, self.num_neurons))
            self.responses_train_and_val = np.zeros((self.num_clips * self.clip_length, self.num_neurons))

            self.test_responses_by_trial = []

            # Note: the hardcoded indices are the location of test clips in Hoefling 2024
            for roi in range(self.num_neurons):
                tmp = np.vstack(
                    (
                        self.neural_responses[roi, : 5 * self.clip_length],
                        self.neural_responses[roi, 59 * self.clip_length : 64 * self.clip_length],
                        self.neural_responses[roi, 118 * self.clip_length :],
                    )
                )
                self.test_responses_by_trial.append(tmp)
                self.responses_test[:, roi] = np.mean(tmp, 0)
                self.responses_train_and_val[:, roi] = np.concatenate(
                    (
                        self.neural_responses[roi, 5 * self.clip_length : 59 * self.clip_length],
                        self.neural_responses[roi, 64 * self.clip_length : 118 * self.clip_length],
                    )
                )
            self.test_responses_by_trial = np.asarray(self.test_responses_by_trial)

        if self.stim_id in [5, "salamander_natural"]:
            self.compute_validation_responses()

        return {
            "train": torch.tensor(self.responses_train).to(torch.float),
            "validation": torch.tensor(self.responses_val).to(torch.float),
            "test": {
                "avg": torch.tensor(self.responses_test).to(torch.float),
                "by_trial": torch.tensor(self.test_responses_by_trial),
            },
        }

    @no_type_check
    def compute_validation_responses(self) -> None:
        movie_ordering = (
            np.arange(self.num_clips)
            if (len(self.random_sequences) == 0 or self.scan_sequence_idx is None)
            else self.random_sequences[:, self.scan_sequence_idx]
        )

        # Initialise validation responses

        base_movie_sorting = np.argsort(movie_ordering)

        validation_mask = np.ones_like(self.responses_train_and_val, dtype=bool)
        self.responses_val = np.zeros([len(self.val_clip_idx) * self.clip_length, self.num_neurons])

        # Compute validation responses and remove sections from training responses

        for i, ind1 in enumerate(self.val_clip_idx):
            grab_index = base_movie_sorting[ind1]
            self.responses_val[i * self.clip_length : (i + 1) * self.clip_length, :] = self.responses_train_and_val[
                grab_index * self.clip_length : (grab_index + 1) * self.clip_length,
                :,
            ]
            validation_mask[
                (grab_index * self.clip_length) : (grab_index + 1) * self.clip_length,
                :,
            ] = False

        if self.use_base_sequence:
            # Reorder training responses to use the same "base" sequence, which follows the numbering of clips.
            # This way all training responses are wrt the same order of clips, which can be useful for some applications
            train_clip_idx = [i for i in range(self.num_clips) if i not in self.val_clip_idx]
            self.responses_train = np.zeros([len(train_clip_idx) * self.clip_length, self.num_neurons])
            for i, train_idx in enumerate(train_clip_idx):
                grab_index = base_movie_sorting[train_idx]
                self.responses_train[i * self.clip_length : (i + 1) * self.clip_length, :] = (
                    self.responses_train_and_val[
                        grab_index * self.clip_length : (grab_index + 1) * self.clip_length,
                        :,
                    ]
                )
        else:
            self.responses_train = self.responses_train_and_val[validation_mask].reshape(-1, self.num_neurons)

    def transform_roi_mask(self, roi_mask):
        roi_coords = np.zeros((len(self.roi_ids), 2))
        for i, roi_id in enumerate(self.roi_ids):
            single_roi_mask = np.zeros_like(roi_mask)
            single_roi_mask[roi_mask == -roi_id] = 1
            roi_coords[i] = self.roi2readout(single_roi_mask)
        return roi_coords

    def roi2readout(
        self,
        single_roi_mask,
        roi_mask_pixelsize=2,
        readout_mask_pixelsize=50,
        x_offset=2.75,
        y_offset=2.75,
    ):
        """
        Maps a roi mask of a single roi from recording coordinates to model
        readout coordinates
        :param single_roi_mask: 2d array with nonzero values indicating the pixels
                of the current roi
        :param roi_mask_pixelsize: size of a pixel in the roi mask in um
        :param readout_mask_pixelsize: size of a pixel in the readout mask in um
        :param x_offset: x offset indicating the start of the recording field in readout mask
        :param y_offset: y offset indicating the start of the recording field in readout mask
        :return:
        """
        pixel_factor = readout_mask_pixelsize / roi_mask_pixelsize
        y, x = np.nonzero(single_roi_mask)
        y_trans, x_trans = y / pixel_factor, x / pixel_factor
        y_trans += y_offset
        x_trans += x_offset
        x_trans = x_trans.mean()
        y_trans = y_trans.mean()
        coords = np.asarray(
            [
                self.map_to_range(max=8, val=y_trans),
                self.map_to_range(max=8, val=x_trans),
            ],
            dtype=np.float32,
        )
        return coords

    def map_to_range(self, max, val):
        val = val / max
        val = val - 0.5
        val = val * 2
        return val


def upsample_traces(
    triggertimes,
    traces,
    tracestimes,
    stim_id: int,
    target_fr: int = 30,
) -> np.ndarray:
    """
    Upsamples the traces based on the stimulus type.

    Args:
        triggertimes (list): List of trigger times.
        traces (list): List of traces.
        tracestimes (list): List of trace times.
        stim_id (int): Stimulus ID.
        stim_framerate (int, optional): Frame rate of the stimulus.
                                        Required for certain stimulus types like moving bar and chirp.
        target_fr (int, optional): Target frame rate for upsampling. Default is 30.

    Returns:
        numpy.ndarray: Upsampled responses.

    Raises:
        NotImplementedError: If the stimulus ID is not implemented.
    """
    if stim_id == 1:
        # Chirp: each chirp has two triggers, one at the start and one 5s later, after a 2s OFF and 3s full field ON.
        # We need only the first trigger of each chirp for the upsampling.
        # 32.98999999 is the total chirp duration in seconds. Should be 33 but there is a small discrepancy
        chirp_starts = triggertimes[::2]
        upsampled_triggertimes = _upsample_triggertimes(32.98999999, 33, chirp_starts, target_fr)
    elif stim_id == 2:
        # Moving bar: each bar has one trigger at the start of the bar stim. Bar duration is 4s.
        # It is a bit more because each trigger has a duration of 3 frames at 60Hz, so around 50 ms.
        upsampled_triggertimes = _upsample_triggertimes(4.054001, 4.1, triggertimes, target_fr)
    elif stim_id == 5:
        # Movie stimulus
        # 4.966666 is the time between triggers in the movie stimulus.
        # It is not exactly 5s because it is not a perfect world :)
        upsampled_triggertimes = _upsample_triggertimes(4.9666667, 5, triggertimes, target_fr)
    else:
        raise NotImplementedError(f"Stimulus ID {stim_id} not implemented")

    upsampled_responses = np.zeros((traces.shape[0], len(upsampled_triggertimes)))
    for i in range(traces.shape[0]):
        upsampled_responses[i] = np.interp(upsampled_triggertimes, tracestimes[i].ravel(), traces[i].ravel())

    upsampled_responses = upsampled_responses / np.std(
        upsampled_responses, axis=1, keepdims=True
    )  # normalize response std

    return upsampled_responses


def _upsample_triggertimes(stim_empirical_duration, stim_theoretical_duration, triggertimes, target_fr):
    # upsample triggertimes to get 1 trigger per frame, (instead of just 1 trigger at the start of the sequence)
    upsampled_triggertimes = [
        np.linspace(t, t + stim_empirical_duration, round(stim_theoretical_duration * target_fr)) for t in triggertimes
    ]
    upsampled_triggertimes = np.concatenate(upsampled_triggertimes)

    return upsampled_triggertimes


def _apply_mask_to_field(data_dict, field, mask):
    """
    Apply a mask to a specific field in a data dictionary.

    Args:
        data_dict (dict): A dictionary containing data fields.
        field (str): The field in the data dictionary to apply the mask to.
        mask (np.ndarray): The mask to apply to the field.

    Returns:
        None

    Raises:
        IndexError: If the mask index is out of bounds for the field data.

    Examples:
        _apply_mask_to_field(data_dict, 'field_name', mask)"""

    for key in data_dict[field].keys():
        if key in ["roi_mask", "roi_coords"]:
            continue
        if isinstance(data_dict[field][key], np.ndarray) and len(data_dict[field][key]) > 0:
            if len(data_dict[field][key].shape) == 1:
                data_dict[field][key] = data_dict[field][key][mask]
            elif len(data_dict[field][key].shape) == 2:
                if data_dict[field][key].shape[0] == mask.shape[0]:
                    data_dict[field][key] = data_dict[field][key][mask, :]
                else:
                    data_dict[field][key] = data_dict[field][key][:, mask]
            else:
                raise IndexError(f"Index out of bounds for field {field} and key {key}.")


def _apply_qi_mask(data_dict, qi_types: list[str], qi_thresholds: list[float], logic="or"):
    """
    Applies quality thresholds as a mask to the data dictionary.

    Args:
        data_dict (dict): The data dictionary.
        qi_types (list): List of quality index types.
        qi_threshold (list): List of quality index thresholds.
        logic (str): The logic to combine different qi_types. Can be 'and' or 'or'. Default is 'and'.

    Returns:
        dict: The updated data dictionary.
    """
    new_data_dict = deepcopy(data_dict)

    if logic not in ["and", "or"]:
        raise ValueError("logic must be either 'and' or 'or'")
    assert len(qi_types) == len(qi_thresholds), "qi_types and qi_thresholds must have the same length"

    for field in new_data_dict.keys():
        masks = [
            new_data_dict[field][f"{qi_type}_qi"] >= qi_threshold
            for qi_type, qi_threshold in zip(qi_types, qi_thresholds)
        ]

        if logic == "and":
            combined_mask = np.logical_and.reduce(masks)
        else:  # logic == 'or'
            combined_mask = np.logical_or.reduce(masks)

        _apply_mask_to_field(new_data_dict, field, combined_mask)

    return _clean_up_empty_fields(new_data_dict)


def _clean_up_empty_fields(data_dict, check_field="group_assignment"):
    """
    Remove empty fields from the data dictionary.

    Args:
        data_dict (dict): The data dictionary.
        check_field (str, optional): The field to check for emptiness. Defaults to "group_assignment".

    Returns:
        dict: The updated data dictionary.
    """
    return {k: v for k, v in data_dict.items() if len(v[check_field]) > 0}


def _mask_by_cell_type(data_dict, cell_types: List[int] | int):
    if not isinstance(cell_types, list):
        if isinstance(cell_types, int):
            cell_types = [cell_types]
        else:
            raise ValueError("cell_types must be a list of integers")
    new_data_dict = deepcopy(data_dict)
    for field in new_data_dict.keys():
        mask = np.isin(new_data_dict[field]["group_assignment"], cell_types)
        _apply_mask_to_field(new_data_dict, field, mask)

    return _clean_up_empty_fields(new_data_dict)


def _mask_by_classifier_confidence(data_dict, min_confidence: float):
    new_data_dict = deepcopy(data_dict)
    for field in new_data_dict.keys():
        mask = new_data_dict[field]["group_confidences"].max(axis=1) >= min_confidence
        _apply_mask_to_field(new_data_dict, field, mask)

    return _clean_up_empty_fields(new_data_dict)


def make_final_responses(
    data_dict: dict,
    response_type: Literal["natural", "chirp", "mb"] = "natural",
    trace_type: Literal["spikes", "raw", "preprocessed", "detrended"] = "spikes",
    d_qi: Optional[float] = None,
    chirp_qi: Optional[float] = None,
    qi_logic: Literal["and", "or"] = "or",
    scale_traces: float = 1.0,
):
    """
    Converts inferred spikes into final responses by upsampling the traces.

    Args:
        data_dict (dict): A dictionary containing the data.
        response_type (str, optional): The type of response. Defaults to "natural".

    Returns:
        dict: The updated data dictionary with final responses.
    Raises:
        NotImplementedError: If the conversion is not yet implemented for the given response type.
    """

    new_data_dict = deepcopy(data_dict)

    stim_id = STIMULI_IDS.get(response_type, None)
    if stim_id is None:
        raise NotImplementedError(f"Conversion not yet implemented for response type {response_type}")

    for field in tqdm(
        new_data_dict.keys(),
        desc=f"Upsampling {response_type} {trace_type} traces to get final responses.",
    ):
        if trace_type == "detrended":
            raw_traces = new_data_dict[field][f"{response_type}_raw_traces"]
            smoothed_traces = new_data_dict[field][f"{response_type}_smoothed_traces"]
            traces = raw_traces - smoothed_traces
        elif trace_type == "spikes":
            try:
                traces = new_data_dict[field][f"{response_type}_inferred_spikes"]
            except KeyError:
                # For new data format
                traces = new_data_dict[field][f"{response_type}_spikes"]
        else:
            traces = new_data_dict[field][f"{response_type}_{trace_type}_traces"]

        triggertimes = new_data_dict[field][f"{response_type}_trigger_times"][0]

        try:
            tracestimes = new_data_dict[field][f"{response_type}_traces_times"]
        except KeyError:
            # New djimaing exports have a different save format for trace_times
            traces_t0 = np.tile(
                np.atleast_1d(new_data_dict[field][f"{response_type}_traces_t0"].squeeze())[:, None],
                (1, traces.shape[1]),
            )
            traces_dt = np.tile(
                np.atleast_1d(new_data_dict[field][f"{response_type}_traces_dt"].squeeze())[:, None],
                (1, traces.shape[1]),
            )
            tracestimes = np.tile(np.arange(traces.shape[1]), reps=(traces.shape[0], 1)) * traces_dt + traces_t0

        upsampled_traces = (
            upsample_traces(
                triggertimes=triggertimes,
                traces=traces,
                tracestimes=tracestimes,
                stim_id=stim_id,
            )
            * scale_traces
        )

        new_data_dict[field][f"{response_type}_responses_final"] = upsampled_traces

        if "responses_final" in new_data_dict[field]:
            warnings.warn(
                f"You seem to already have computed `responses_final` for a "
                f"stim_id of {new_data_dict[field]['stim_id']}. "
                f"Overwriting with {stim_id} ({response_type})."
            )
        new_data_dict[field]["responses_final"] = upsampled_traces
        new_data_dict[field]["stim_id"] = stim_id

    d_qi = d_qi if d_qi is not None else 0.0
    chirp_qi = chirp_qi if chirp_qi is not None else 0.0
    new_data_dict = _apply_qi_mask(new_data_dict, ["d", "chirp"], [d_qi, chirp_qi], qi_logic)

    return new_data_dict


def filter_responses(
    all_responses: Dict[str, dict],
    filter_cell_types: bool = False,
    cell_types_list: Optional[List[int] | int] = None,
    chirp_qi: float = 0.35,
    d_qi: float = 0.6,
    qi_logic: Literal["and", "or"] = "or",
    filter_counts: bool = True,
    count_threshold: int = 10,
    classifier_confidence: float = 0.25,
    verbose: bool = False,
) -> Dict[str, dict]:
    """
    This function processes the input dictionary of neuron responses, applying various filters
    to exclude unwanted data based on the provided parameters. It can filter by cell types,
    quality indices, classifier confidence, and the number of responding cells, while also
    providing verbose output for tracking the filtering process.
    Note: default arguments are from the Hoefling et al., 2024 paper.

    Args:
        all_responses (Dict[str, dict]): A dictionary containing neuron response data.
        filter_cell_types (bool, optional): Whether to filter by specific cell types. Defaults to False.
        cell_types_list (Optional[List[int] | int], optional): List or single value of cell types to filter.
                                                                Defaults to None.
        chirp_qi (float, optional): Quality index threshold for chirp responses. Defaults to 0.35.
        d_qi (float, optional): Quality index threshold for d responses. Defaults to 0.6.
        qi_logic (Literal["and", "or"], optional): The logic to combine different quality indices. Defaults to "and".
        filter_counts (bool, optional): Whether to filter based on response counts. Defaults to True.
        count_threshold (int, optional): Minimum number of responding cells required. Defaults to 10.
        classifier_confidence (float, optional): Minimum confidence level for classifier responses. Defaults to 0.3.
        verbose (bool, optional): If True, prints detailed filtering information. Defaults to False.

    Returns:
        Dict[str, dict]: A filtered dictionary of neuron responses that meet the specified criteria.
    """

    def print_verbose(message):
        if verbose:
            print(message)

    def get_n_neurons(all_responses):
        return sum(len(field["group_assignment"]) for field in all_responses.values())

    original_neuron_count = get_n_neurons(all_responses)
    print_verbose(f"Original dataset contains {original_neuron_count} neurons over {len(all_responses)} fields")
    print_verbose(" ------------------------------------ ")

    # Filter by cell types
    if filter_cell_types and cell_types_list is not None:
        all_rgcs_responses_ct_filtered = _mask_by_cell_type(all_responses, cell_types_list)
        print_verbose(
            f"Dropped {len(all_responses) - len(all_rgcs_responses_ct_filtered)} fields that did not "
            f"contain the target cell types ({len(all_rgcs_responses_ct_filtered)} remaining)"
        )
        dropped_n_cell_types = original_neuron_count - get_n_neurons(all_rgcs_responses_ct_filtered)
        print_verbose(
            f"Overall, dropped {dropped_n_cell_types} neurons of non-target cell types "
            f"(-{(dropped_n_cell_types) / original_neuron_count :.2%})."
        )
        print_verbose(" ------------------------------------ ")
    else:
        all_rgcs_responses_ct_filtered = deepcopy(all_responses)

    count_before_checks = get_n_neurons(all_rgcs_responses_ct_filtered)

    # Apply quality checks
    d_qi = d_qi if d_qi is not None else 0.0
    all_rgcs_responses_ct_filtered = _apply_qi_mask(
        all_rgcs_responses_ct_filtered, ["d", "chirp"], [d_qi, chirp_qi], qi_logic
    )

    dropped_n_qi = count_before_checks - get_n_neurons(all_rgcs_responses_ct_filtered)

    print_verbose(
        f"Dropped {len(all_responses) - len(all_rgcs_responses_ct_filtered)} fields with quality indices "
        f"below threshold ({len(all_rgcs_responses_ct_filtered)} remaining)"
    )
    print_verbose(
        f"Overall, dropped {dropped_n_qi} neurons over quality checks (-{dropped_n_qi / count_before_checks:.2%})."
    )
    print_verbose(" ------------------------------------ ")

    # Filter by classifier confidence
    if classifier_confidence is not None and classifier_confidence > 0.0:
        all_rgcs_responses_confidence_filtered = _mask_by_classifier_confidence(
            all_rgcs_responses_ct_filtered, classifier_confidence
        )
        dropped_n_classifier = get_n_neurons(all_rgcs_responses_ct_filtered) - get_n_neurons(
            all_rgcs_responses_confidence_filtered
        )
        print_verbose(
            f"Dropped {len(all_rgcs_responses_ct_filtered) - len(all_rgcs_responses_confidence_filtered)} fields with "
            f"classifier confidences below {classifier_confidence}"
        )
        print_verbose(
            f"Overall, dropped {dropped_n_classifier} neurons with classifier confidences below {classifier_confidence}"
            f" (-{dropped_n_classifier / get_n_neurons(all_rgcs_responses_ct_filtered):.2%})."
        )
        print_verbose(" ------------------------------------ ")
    else:
        all_rgcs_responses_confidence_filtered = all_rgcs_responses_ct_filtered

    # Filter by low counts
    if filter_counts:
        all_rgcs_responses = {
            k: v
            for k, v in all_rgcs_responses_confidence_filtered.items()
            if len(v["group_assignment"]) > count_threshold
        }
        dropped_n_counts = get_n_neurons(all_rgcs_responses_confidence_filtered) - get_n_neurons(all_rgcs_responses)
        print_verbose(
            f"Dropped {len(all_rgcs_responses_confidence_filtered) - len(all_rgcs_responses)} fields with less than "
            f"{count_threshold} responding cells ({len(all_rgcs_responses)} remaining)"
        )
        print_verbose(
            f"Overall, dropped {dropped_n_counts} neurons in fields with less than {count_threshold} responding cells "
            f"(-{dropped_n_counts / get_n_neurons(all_rgcs_responses_confidence_filtered):.2%})."
        )
    else:
        all_rgcs_responses = all_rgcs_responses_confidence_filtered

    print_verbose(" ------------------------------------ ")
    print_verbose(
        f"Final dataset contains {get_n_neurons(all_rgcs_responses)} neurons over {len(all_rgcs_responses)} fields"
    )
    final_n_dropped = original_neuron_count - get_n_neurons(all_rgcs_responses)
    print(f"Total number of cells dropped: {final_n_dropped} " f"(-{(final_n_dropped) / original_neuron_count :.2%})")

    # Clean up RAM
    del all_rgcs_responses_ct_filtered
    del all_rgcs_responses_confidence_filtered

    return all_rgcs_responses
