import pickle
from collections import defaultdict, namedtuple
from typing import Dict, List

import numpy as np
import torch
from jaxtyping import Float

from .constants import RGC_GROUP_NAMES_DICT

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
        self, list_of_ids: List[int], min_confidence: float = 0.0, min_neurons_per_group: int = 1
    ) -> List[SingleNeuronInfoStruct]:
        training_samples = sum(
            (
                self.get_training_samples_for_group(
                    group_id, min_confidence=min_confidence, min_neurons_per_group=min_neurons_per_group
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
            (self.get_test_samples_for_group(group_id, min_confidence=min_confidence) for group_id in list_of_ids), []
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
        eye: str,
        group_assignment: Float[np.ndarray, "n_neurons"],  # noqa
        key: dict,
        responses_final: Float[np.ndarray, "n_neurons n_timepoints"],  # noqa
        roi_coords: Float[np.ndarray, "n_neurons 2"],  # noqa
        roi_ids: Float[np.ndarray, "n_neurons"],  # noqa
        scan_sequence_idx: int,
        stim_id: int,
        traces,
        tracestimes,
        random_sequences: Float[np.ndarray, "n_clips n_sequences"],  # noqa
        val_clip_idx: List[int],
        num_clips: int,
        clip_length: int,
    ):
        """
        Initialize the NeuronData object.
        Boilerplate class to store neuron data. Added for backwards compatibility with Hoefling et al., 2022.

        Args:
            eye (str): The eye from which the neuron data is recorded.
            group_assignment (Float[np.ndarray, "n_neurons"]): The group assignment of neurons.
            key (dict): The key information for the neuron data, includes date, exp_num, experimenter, field_id, stim_id.
            responses_final (Float[np.ndarray, "n_neurons n_timepoints"]): The final responses of neurons.
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
        self.eye = eye
        self.group_assignment = group_assignment
        self.key = key
        self.responses_final = responses_final
        self.roi_coords = roi_coords
        self.roi_ids = roi_ids
        self.scan_sequence_idx = scan_sequence_idx
        self.stim_id = stim_id
        self.traces = traces
        self.tracestimes = tracestimes
        self.clip_length = clip_length
        self.num_clips = num_clips
        self.random_sequences = random_sequences
        self.val_clip_idx = val_clip_idx

    #! this has to become a regular method in the future
    @property
    def response_dict(self):
        num_neurons = self.responses_final.shape[0]
        movie_ordering = (
            np.arange(self.num_clips)
            if len(self.random_sequences) == 0
            else self.random_sequences[:, self.scan_sequence_idx]
        )

        if self.stim_id == 0:
            self.responses_test = self.responses_final[:, : 10 * self.clip_length].T
            self.responses_train = self.responses_final[:, 10 * self.clip_length :].T
            self.test_responses_by_trial = None
        else:
            self.responses_test = np.zeros((5 * self.clip_length, num_neurons))
            self.responses_train = np.zeros((self.num_clips * self.clip_length, num_neurons))
            self.test_responses_by_trial = []
            for roi in range(num_neurons):
                tmp = np.vstack(
                    (
                        self.responses_final[roi, : 5 * self.clip_length],
                        self.responses_final[roi, 59 * self.clip_length : 64 * self.clip_length],
                        self.responses_final[roi, 118 * self.clip_length :],
                    )
                )
                self.test_responses_by_trial.append(tmp)
                self.responses_test[:, roi] = np.mean(tmp, 0)
                self.responses_train[:, roi] = np.concatenate(
                    (
                        self.responses_final[roi, 5 * self.clip_length : 59 * self.clip_length],
                        self.responses_final[roi, 64 * self.clip_length : 118 * self.clip_length],
                    )
                )
            self.test_responses_by_trial = np.asarray(self.test_responses_by_trial)

        if self.stim_id == 0:
            self.responses_val = np.zeros([len(self.val_clip_idx), self.clip_length, num_neurons])
            for i, ind in enumerate(self.val_clip_idx):
                self.responses_val[i] = self.responses_train[ind * self.clip_length : (ind + 1) * self.clip_length, :]
        else:
            self.responses_val = np.zeros([len(self.val_clip_idx) * self.clip_length, num_neurons])
            inv_order = np.argsort(movie_ordering)
            for i, ind1 in enumerate(self.val_clip_idx):
                ind2 = inv_order[ind1]
                self.responses_val[i * self.clip_length : (i + 1) * self.clip_length, :] = self.responses_train[
                    ind2 * self.clip_length : (ind2 + 1) * self.clip_length, :
                ]

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
