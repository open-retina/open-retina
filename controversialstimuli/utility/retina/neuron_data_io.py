import pickle
from collections import defaultdict, namedtuple
from typing import Dict, List, Optional

import numpy as np
import torch
from controversialstimuli.utility.retina.constants import RGC_GROUP_NAMES_DICT

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
