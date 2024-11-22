import warnings
from copy import deepcopy
from typing import Literal, Optional, no_type_check

import numpy as np
import torch
from jaxtyping import Float
from tqdm.auto import tqdm

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.data_io.hoefling_2024.constants import CLIP_LENGTH, NUM_CLIPS, STIMULI_IDS


class NeuronDataSplitHoefling:
    def __init__(
        self,
        neural_responses: ResponsesTrainTestSplit,
        val_clip_idx: Optional[list[int]],
        num_clips: Optional[int],
        clip_length: Optional[int],
        roi_mask: Optional[Float[np.ndarray, "64 64"]] = None,
        roi_ids: Optional[Float[np.ndarray, " n_neurons"]] = None,
        scan_sequence_idx: Optional[int] = None,
        random_sequences: Optional[Float[np.ndarray, "n_clips n_sequences"]] = None,
        eye: Optional[Literal["left", "right"]] = None,
        group_assignment: Optional[Float[np.ndarray, " n_neurons"]] = None,
        key: Optional[dict] = None,
        use_base_sequence: Optional[bool] = False,
        **kwargs,
    ):
        """
        Initialize the NeuronData object.
        Boilerplate class to store neuron data train/test/validation splits before feeding into a dataloader.
        Customized for compatibility with the data format in Hoefling et al., 2024.

        Args:
            eye (str): The eye from which the neuron data is recorded.
            group_assignment (Float[np.ndarray, "n_neurons"]): The group assignment of neurons.
            key (dict): The key information for the neuron data,
                        includes date, exp_num, experimenter, field_id, stim_id.
            responses_final (Float[np.ndarray, "n_neurons n_timepoints"]): The responses of neurons.
            roi_coords (Float[np.ndarray, "n_neurons 2"]): The coordinates of regions of interest (ROIs).
            roi_ids (Float[np.ndarray, "n_neurons"]): The IDs of regions of interest (ROIs).
            scan_sequence_idx (int): The index of the scan sequence.
            stim_id (int): The ID of the stimulus. 5 is mouse natural scenes.
            random_sequences (Float[np.ndarray, "n_clips n_sequences"]): The random sequences of clips.
            val_clip_idx (List[int]): The indices of validation clips.
            num_clips (int): The number of clips.
            clip_length (int): The length of each clip.
            use_base_sequence (bool): Whether to re-order all training responses to use the same "base" sequence.
        """
        self.neural_responses = neural_responses

        self.num_neurons = self.neural_responses.n_neurons

        self.eye = eye if eye is not None else "right"
        self.group_assignment = group_assignment
        self.key = key
        self.roi_ids = roi_ids
        self.roi_coords = (
            np.array(self.transform_roi_mask(roi_mask), dtype=np.float32) if roi_mask is not None else None
        )
        self.scan_sequence_idx = scan_sequence_idx
        self.stim_id = self.neural_responses.stim_id
        self.clip_length = clip_length
        self.num_clips = num_clips
        self.random_sequences = random_sequences if random_sequences is not None else np.array([])
        self.use_base_sequence = use_base_sequence
        self.val_clip_idx = val_clip_idx

        self.responses_train_and_val, self.responses_test, self.test_responses_by_trial = (
            self.neural_responses.train.T,
            self.neural_responses.test.T,
            self.neural_responses.test_by_trial.transpose(0, 2, 1)
            if self.neural_responses.test_by_trial is not None
            else None,
        )
        self.responses_train, self.responses_val = self.compute_validation_responses()

    @property
    def response_dict(self):
        return {
            "train": torch.tensor(self.responses_train).to(torch.float),
            "validation": torch.tensor(self.responses_val).to(torch.float),
            "test": {
                "avg": torch.tensor(self.responses_test).to(torch.float),
                "by_trial": torch.tensor(self.test_responses_by_trial),
            },
        }

    @no_type_check
    def compute_validation_responses(self) -> tuple[np.ndarray, np.ndarray]:
        if self.stim_id in ["mb", "chirp"]:
            # Chirp and moving bar
            responses_val = np.array(np.nan)
            responses_train = self.responses_train_and_val
        else:
            movie_ordering = (
                np.arange(self.num_clips)
                if (len(self.random_sequences) == 0 or self.scan_sequence_idx is None)
                else self.random_sequences[:, self.scan_sequence_idx]
            )

            # Initialise validation responses

            base_movie_sorting = np.argsort(movie_ordering)

            validation_mask = np.ones_like(self.responses_train_and_val, dtype=bool)
            responses_val = np.zeros([len(self.val_clip_idx) * self.clip_length, self.num_neurons])

            # Compute validation responses and remove sections from training responses

            for i, ind1 in enumerate(self.val_clip_idx):
                grab_index = base_movie_sorting[ind1]
                responses_val[i * self.clip_length : (i + 1) * self.clip_length, :] = self.responses_train_and_val[
                    grab_index * self.clip_length : (grab_index + 1) * self.clip_length,
                    :,
                ]
                validation_mask[
                    (grab_index * self.clip_length) : (grab_index + 1) * self.clip_length,
                    :,
                ] = False

            if self.use_base_sequence:
                # Reorder training responses to use the same "base" sequence, which follows the numbering of clips.
                # This way all training responses are wrt the same order of clips, which can be useful in analysis.
                train_clip_idx = [i for i in range(self.num_clips) if i not in self.val_clip_idx]
                responses_train = np.zeros([len(train_clip_idx) * self.clip_length, self.num_neurons])
                for i, train_idx in enumerate(train_clip_idx):
                    grab_index = base_movie_sorting[train_idx]
                    responses_train[i * self.clip_length : (i + 1) * self.clip_length, :] = (
                        self.responses_train_and_val[
                            grab_index * self.clip_length : (grab_index + 1) * self.clip_length,
                            :,
                        ]
                    )
            else:
                responses_train = self.responses_train_and_val[validation_mask].reshape(-1, self.num_neurons)

        return responses_train, responses_val

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


def compute_test_responses(
    neural_responses: Float[np.ndarray, "n_neurons time"],
    clip_length=CLIP_LENGTH,
    num_clips=NUM_CLIPS,
    response_type: Literal["natural", "chirp", "mb"] = "natural",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if response_type in ["chirp", "mb"]:
        # Chirp and moving bar
        responses_test = np.array(np.nan)
        test_responses_by_trial = [np.nan]
        responses_train_and_val = neural_responses.T

    else:
        assert clip_length is not None, "Clip length must be provided for natural scenes"
        assert num_clips is not None, "Number of clips must be provided for natural scenes"

        num_neurons = neural_responses.shape[0]

        responses_test = np.zeros((5 * clip_length, num_neurons))
        responses_train_and_val = np.zeros((num_clips * clip_length, num_neurons))

        test_responses_by_trial = []

        # Note: the hardcoded indices are the location of test clips in Hoefling 2024
        for roi in range(num_neurons):
            tmp = np.vstack(
                (
                    neural_responses[roi, : 5 * clip_length],
                    neural_responses[roi, 59 * clip_length : 64 * clip_length],
                    neural_responses[roi, 118 * clip_length :],
                )
            )
            test_responses_by_trial.append(tmp.T)  # type: ignore
            responses_test[:, roi] = np.mean(tmp, 0)
            responses_train_and_val[:, roi] = np.concatenate(
                (
                    neural_responses[roi, 5 * clip_length : 59 * clip_length],
                    neural_responses[roi, 64 * clip_length : 118 * clip_length],
                )
            )

    return responses_train_and_val.T, responses_test.T, np.asarray(test_responses_by_trial)


def upsample_traces(
    triggertimes,
    traces,
    tracestimes,
    stim_id: int,
    target_fr: int = 30,
    norm_by_std: bool = True,
) -> np.ndarray:
    """
    Upsamples the traces based on the stimulus type.

    Args:
        triggertimes (list): List of trigger times.
        traces (list): List of traces.
        tracestimes (list): List of trace times.
        stim_id (int): Stimulus ID.
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

    if norm_by_std:
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


def _apply_mask_to_field(data_dict: dict[str, dict], field: str, mask: Float[np.ndarray, " n_neurons"]) -> None:
    """
    Apply a mask to a specific field in a data dictionary.

    Args:
        data_dict (dict): A dictionary containing data fields.
        field (str): The field in the data dictionary to apply the mask to.
        mask (np.ndarray): The mask (assumed over neurons) to apply to each entry of the field.

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
                    # If neurons are in the first dimension of the target array
                    data_dict[field][key] = data_dict[field][key][mask, :]
                else:
                    # If neurons are in the second dimension of the target array
                    data_dict[field][key] = data_dict[field][key][:, mask]
            else:
                raise IndexError(f"Index out of bounds for field {field} and key {key}.")


def _apply_qi_mask(data_dict, qi_types: list[str], qi_thresholds: list[float], logic="or"):
    """
    Applies quality thresholds as a mask to the data dictionary.

    Args:
        data_dict (dict): The data dictionary.
        qi_types (list): List of quality index types. Supported types are 'd' and 'chirp', corresponding to the
                        quality indices for the direction selectivity and chirp responses, respectively.
        qi_thresholds (list): List of quality index thresholds.
        logic (str): The logic to combine different qi_types. Can be 'and' or 'or'. Default is 'and'.

    Returns:
        dict: The updated data dictionary.
    """
    new_data_dict = deepcopy(data_dict)

    if logic not in ["and", "or"]:
        raise ValueError("logic must be either 'and' or 'or'")

    for field in new_data_dict.keys():
        masks = [
            new_data_dict[field][f"{qi_type}_qi"] >= qi_threshold
            for qi_type, qi_threshold in zip(qi_types, qi_thresholds, strict=True)
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


def _mask_by_cell_type(data_dict, cell_types: list[int] | int):
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
    norm_by_std: bool = True,
):
    upsampled_data_dict = upsample_all_responses(
        data_dict, response_type, trace_type, d_qi, chirp_qi, qi_logic, scale_traces, norm_by_std
    )

    responses_dictionary = {}

    for field in upsampled_data_dict.keys():
        train_responses, test_responses, test_responses_by_trial = compute_test_responses(
            upsampled_data_dict[field]["responses_final"],
            response_type=response_type,
        )
        responses_dictionary[field] = ResponsesTrainTestSplit(
            train=train_responses,
            test=test_responses,
            test_by_trial=test_responses_by_trial,
            stim_id=response_type,
            session_kwargs={
                "eye": upsampled_data_dict[field]["eye"],
                "scan_sequence_idx": upsampled_data_dict[field]["scan_sequence_idx"],
            },
        )

    return responses_dictionary


def upsample_all_responses(
    data_dict: dict,
    response_type: Literal["natural", "chirp", "mb"] = "natural",
    trace_type: Literal["spikes", "raw", "preprocessed", "detrended"] = "spikes",
    d_qi: Optional[float] = None,
    chirp_qi: Optional[float] = None,
    qi_logic: Literal["and", "or"] = "or",
    scale_traces: float = 1.0,
    norm_by_std: bool = True,
):
    """
    Converts inferred spikes into final responses by upsampling the traces of all sessions of a given response_type.
    This is to match the framerate used in the stimulus presentation.

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
                norm_by_std=norm_by_std,
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
    # Apply quality indices mask only if requested
    if d_qi > 0.0 or chirp_qi > 0.0:
        new_data_dict = _apply_qi_mask(new_data_dict, ["d", "chirp"], [d_qi, chirp_qi], qi_logic)

    return new_data_dict


def filter_responses(
    all_responses: dict[str, dict],
    filter_cell_types: bool = False,
    cell_types_list: Optional[list[int] | int] = None,
    chirp_qi: float = 0.35,
    d_qi: float = 0.6,
    qi_logic: Literal["and", "or"] = "or",
    filter_counts: bool = True,
    count_threshold: int = 10,
    classifier_confidence: float = 0.25,
    verbose: bool = False,
) -> dict[str, dict]:
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

    def get_n_neurons(all_responses) -> int:
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
