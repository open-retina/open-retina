import pickle
import warnings
from collections import defaultdict, namedtuple
from copy import deepcopy
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from jaxtyping import Float
from tqdm.auto import tqdm

from .hoefling_2024.constants import RGC_GROUP_NAMES_DICT, STIMULI_IDS

relevant_rgc_ids = None  # Overwrite this!

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
            targets = torch.concat([x for x in batches_targets], dim=0)
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

        diff_neurons = set(n.neuron_id for n in self._all_neurons) - set(n.neuron_id for n in new_neuron_list)
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
        roi_coords: Optional[Float[np.ndarray, "n_neurons 2"]] = None,  # noqa
        roi_ids: Optional[Float[np.ndarray, "n_neurons"]] = None,  # noqa
        traces: Optional[Float[np.ndarray, "n_neurons n_timepoints"]] = None,  # noqa
        tracestimes: Optional[Float[np.ndarray, "n_timepoints"]] = None,  # noqa
        scan_sequence_idx: Optional[int] = None,
        random_sequences: Optional[Float[np.ndarray, "n_clips n_sequences"]] = None,  # noqa
        eye: Optional[Literal["left", "right"]] = None,
        group_assignment: Optional[Float[np.ndarray, "n_neurons"]] = None,  # noqa
        key: Optional[dict] = None,
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
        """
        assert relevant_rgc_ids is not None, "Overwrite relevant_rgc_ids!"
        mask = np.isin(group_assignment, relevant_rgc_ids)
        self.neural_responses = responses_final[mask]

        self.num_neurons = (
            self.neural_responses.shape[0]
            if not isinstance(self.neural_responses, dict)
            else self.neural_responses["train"].shape[0]
        )

        self.eye = eye if eye is not None else "right"
        self.group_assignment = group_assignment
        self.roi_ids = roi_ids
        self.key = key
        assert roi_coords.shape == (0, )
        self.roi_coords = roi_coords
        self.scan_sequence_idx = scan_sequence_idx
        self.stim_id = stim_id
        self.traces = traces
        self.tracestimes = tracestimes
        self.clip_length = clip_length
        self.num_clips = num_clips
        self.random_sequences = random_sequences if random_sequences is not None else np.array([])
        self.val_clip_idx = val_clip_idx

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

            self.responses_test = np.zeros((5 * self.clip_length, self.num_neurons))
            self.responses_train_and_val = np.zeros((self.num_clips * self.clip_length, self.num_neurons))

            self.test_responses_by_trial = []

            # Note: the hardcoded indices are the location of test clips in Hoefling 2022
            for roi in range(self.num_neurons):
                tmp = np.vstack(
                    (
                        self.neural_responses[roi, : 5 * self.clip_length],
                        self.neural_responses[roi, 59 * self.clip_length: 64 * self.clip_length],
                        self.neural_responses[roi, 118 * self.clip_length:],
                    )
                )
                self.test_responses_by_trial.append(tmp)
                self.responses_test[:, roi] = np.mean(tmp, 0)
                self.responses_train_and_val[:, roi] = np.concatenate(
                    (
                        self.neural_responses[roi, 5 * self.clip_length: 59 * self.clip_length],
                        self.neural_responses[roi, 64 * self.clip_length: 118 * self.clip_length],
                    )
                )
            self.test_responses_by_trial = np.asarray(self.test_responses_by_trial)

        if self.stim_id in [5, "salamander_natural"]:

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
                self.responses_val[i * self.clip_length: (i + 1) * self.clip_length, :] = self.responses_train_and_val[
                    grab_index * self.clip_length: (grab_index + 1) * self.clip_length,
                    :,
                ]
                validation_mask[
                    (grab_index * self.clip_length): (grab_index + 1) * self.clip_length,
                    :,
                ] = False
            self.responses_train = self.responses_train_and_val[validation_mask].reshape(-1, self.num_neurons)

        response_dict = {
            "train": torch.tensor(self.responses_train).to(torch.float),
            "validation": torch.tensor(self.responses_val).to(torch.float),
            "test": {
                "avg": torch.tensor(self.responses_test).to(torch.float),
                "by_trial": torch.tensor(self.test_responses_by_trial),
            },
        }

        return response_dict

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
    if stim_id == 5:
        # Movie stimulus
        # 4.966666 is the time between triggers in the movie stimulus.
        # It is not exactly 5s because it is not a perfect world :)
        upsampled_triggertimes = _upsample_triggertimes(4.9666667, 5,
                                                        triggertimes, target_fr)
    elif stim_id == 1:
        # Chirp: each chirp has two triggers, one at the start and one 5s later, after a 2s OFF and 3s full field ON.
        # We need only the first trigger of each chirp for the upsampling.
        # 32.98999999 is the total chirp duration in seconds. Should be 33 but there is a small discrepancy
        chirp_starts = triggertimes[::2]
        upsampled_triggertimes = _upsample_triggertimes(32.98999999, 33,
                                                        chirp_starts, target_fr)
    elif stim_id == 2:
        # Moving bar: each bar has one trigger at the start of the bar stim. Bar duration is 4s.
        # It is a bit more because each trigger has a duration of 3 frames at 60Hz, so around 50 ms.
        upsampled_triggertimes = _upsample_triggertimes(4.054001, 4.1,
                                                        triggertimes, target_fr)
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


def make_final_responses(data_dict: dict, response_type: Literal["natural", "chirp", "mb"] = "natural"):
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
        desc=f"Upsampling {response_type} traces to get final responses.",
    ):
        try:
            spikes = new_data_dict[field][f"{response_type}_inferred_spikes"]
        except KeyError:
            # For new data format
            spikes = new_data_dict[field][f"{response_type}_spikes"]
        triggertimes = new_data_dict[field][f"{response_type}_trigger_times"][0]
        tracestimes = new_data_dict[field][f"{response_type}_traces_times"]

        upsampled_traces = upsample_traces(
            triggertimes=triggertimes,
            traces=spikes,
            tracestimes=tracestimes,
            stim_id=stim_id,
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

    return new_data_dict
