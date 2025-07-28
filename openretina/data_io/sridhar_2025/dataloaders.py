import os
from pathlib import Path
import torch
import numpy as np
from collections import namedtuple

from torch.utils.data import DataLoader, Dataset

from openretina.data_io.sridhar_2025.constants import TEST_DATA_FRAMES
from openretina.data_io.sridhar_2025.dataloader_utils import get_trial_wise_validation_split
from openretina.data_io.sridhar_2025.responses import load_responses
from openretina.data_io.sridhar_2025.stimuli import process_fixations, load_frames

# from line_profiler_pycharm import profile
default_image_datapoint = namedtuple("DefaultDataPoint", ["images", "responses"])

def crop_based_on_fixation(
    img, x_center, y_center, img_h, img_w, flip=False, padding=200
):
    x_center += padding
    y_center += padding
    if padding != 0:
        img = img[
              x_center - int(img_w / 2): x_center + int(img_w / 2),
              y_center - int(img_h / 2): y_center + int(img_h / 2),
              ]
    if flip:
        img = torch.fliplr(img)
    assert img.shape == (img_w, img_h)

    return img


class MarmosetMovieDataset(Dataset):
    """
        A PyTorch‐style dataset that delivers *time chunks* of marmoset movie
        stimuli together with the corresponding neuronal responses recorded from
        visual cortex.

        The dataset is designed for **temporal models** (e.g., 3-D CNNs, RNNs,
        SSMs) that predict neural activity from short clips of video input.
        Compared with a vanilla ``torchvision.datasets.VisionDataset``, it handles:

        * Stimulus‐aligned **eye-fixation cropping** of movie frames
        * On-the-fly spatiotemporal subsampling and caching for efficient training
          on large movie corpora.

        Parameters
        ----------
        responses : dict
            Dictionary with at least two keys:
            * ``"train_responses"`` – ``(n_neurons, T_train, n_trials)`` array.
            * ``"test_responses"``  – ``(n_neurons, T_test)``   array.

        dir : str or pathlib.Path
            Root folder that contains the raw movie frames on disk.
        *data_keys : list[str]
            Order of modalities returned by ``__getitem__``.  Typical options are
            ``"images"`` and ``"responses"``.  The tuple is forwarded to
            ``self.data_point`` to build the sample object.
        indices : list[int]
            Trial indices to draw from train_responses.  Ignored for test sets.
        frames : np.ndarray
            All stimuus frames set loaded in memory,
            ``shape  == (N_frames, full_img_h, full_img_w)``.
        fixations : Sequence[Mapping[str, Any]]
            Per-frame metadata with gaze center and flip flag
            (expects keys ``"img_index"``, ``"center_x"``, ``"center_y"``,
            ``"flip"``).
        temporal_dilation : int, default 1
            Step (in frames) between successive **visible** inputs fed to the
            network.
        hidden_temporal_dilation : int | tuple[int], default 1
            Temporal dilation(s) applied between hidden layers.  If an ``int`` is
            provided it is broadcast to ``num_of_layers - 1`` layers.
        test_frames, train_frames : int, optional
            Offsets that separate the first *test_frames* from the rest
        img_dir_name : str, default ``"stimuli"``
            Subfolder inside ``dir`` that contains individual frame files.
        frame_file : str, default ``"_img_"``
            Substring common to every frame filename (used by :pyfunc:`glob`).
        test : bool, default ``False``
            If ``True`` the dataset serves the *held-out* test set and skips
            ``indices`` / shuffling logic.
        crop : int | tuple[int, int, int, int], default 0
            Pixels cropped from *(top, bottom, left, right)* **before**
            subsampling.  Passing an ``int`` applies the same pad on every side.
        subsample : int, default 1
            Spatial down-sample factor.  ``1`` keeps full resolution.
        num_of_frames : int, default 15
            Number of *visible* frames given to the first network layer.
        num_of_hidden_frames : int | tuple[int], optional
            Look-back window per hidden layer.  If ``None`` it defaults to
            ``num_of_frames``.
        num_of_layers : int, optional
            Total depth of the model (visible + hidden).  Required if
            ``hidden_temporal_dilation`` or ``num_of_hidden_frames`` are tuples.
        device : str, default ``"cpu"``
            Torch device where tensors are materialised.
        time_chunk_size : int, optional
            Total length (in frames) of each sample returned by ``__getitem__``.
            If provided, the iterator chunks the movie into
            ``time_chunk_size − frame_overhead`` non-overlapping segments.
        full_img_w, full_img_h : int, default 1000, 800
            Dimensions of the uncropped video frames.
        img_w, img_h : int, default 800, 600
            Target spatial resolution *after* cropping/subsampling.
        padding : int, default 200
            Extra pixels gathered around each frame"""
    def __init__(
            self,
            responses: dict,
            dir,
            *data_keys: list,
            indices: list,
            frames,
            fixations,
            temporal_dilation=1,
            hidden_temporal_dilation=1,
            test_frames: int = TEST_DATA_FRAMES,
            img_dir_name="stimuli",
            frame_file: str = "_img_",
            test: bool = False,
            crop: int or tuple = 0,
            subsample: int = 1,
            num_of_frames: int = 15,
            num_of_hidden_frames: int = 15,
            num_of_layers: int = None,
            device: str = "cpu",
            time_chunk_size: int = None,
            full_img_w=1000,
            full_img_h=800,
            img_w=800,
            img_h=600,
            padding=200,
    ):
        self.data_keys = data_keys
        if set(data_keys) == {"images", "responses"}:
            # this version IS serializable in pickle
            self.data_point = default_image_datapoint

        if isinstance(crop, int):
            crop = (crop, crop, crop, crop)
        self.crop = crop
        self.temporal_dilation = temporal_dilation
        self.num_of_layers = num_of_layers

        self.img_h = img_h
        self.img_w = img_w
        self.full_img_h = full_img_h
        self.full_img_w = full_img_w
        self.padding = padding

        self.test_frames = test_frames

        self.num_of_frames = num_of_frames
        if num_of_hidden_frames is None:
            self.num_of_hidden_frames = self.num_of_frames
        else:
            self.num_of_hidden_frames = num_of_hidden_frames

        if isinstance(hidden_temporal_dilation, str):
            hidden_temporal_dilation = int(hidden_temporal_dilation)

        if isinstance(hidden_temporal_dilation, int):
            hidden_temporal_dilation = (hidden_temporal_dilation,) * (
                    self.num_of_layers - 1
            )
        if isinstance(num_of_hidden_frames, int):
            num_of_hidden_frames = (num_of_hidden_frames,) * (self.num_of_layers - 1)

        if self.num_of_layers > 1:
            hidden_reach = sum(
            (f - 1) * d for f, d in zip(num_of_hidden_frames, hidden_temporal_dilation)
        )
        else:
            hidden_reach = 0

        self.frame_overhead = (
                                      num_of_frames - 1
                              ) * self.temporal_dilation + hidden_reach

        if time_chunk_size is not None:
            self.time_chunk_size = time_chunk_size + self.frame_overhead

        self.subsample = subsample
        self.device = device
        self.basepath = Path(dir).absolute()
        self.img_dir_name = img_dir_name
        self.frame_file = frame_file
        self.response_dict = responses
        if indices is not None:
            self.train_responses = torch.from_numpy(
                responses["train_responses"]
            ).float()
        self.test_responses = torch.from_numpy(responses["test_responses"]).float()

        self.fixations = fixations
        self.indices = indices
        self.frames = frames

        self.random_indices = np.random.permutation(indices)
        self.n_neurons = self.train_responses.shape[0]
        self.num_of_trials = self.train_responses.shape[2]
        self.num_of_imgs = int(self.train_responses.shape[1])

        if test:
            self.num_of_imgs = self.test_responses.shape[1]

        self.cache = []
        self.last_start_index = -1
        self.last_end_index = -1

        self._test = test
        if self._test:
            self._len = int(
                np.floor(
                    (self.num_of_imgs - self.frame_overhead)
                    / (self.time_chunk_size - self.frame_overhead)
                )
            )
        else:
            self._len = int(
                len(self.indices)
                * np.floor(
                    (self.num_of_imgs - self.frame_overhead)
                    / (self.time_chunk_size - self.frame_overhead)
                )
            )

    def transform(self, images):
        """
        applies transformations to the image: downsampling, cropping, rescaling, and dimension expansion.
        """
        if len(images.shape) == 3:
            h, w, num_of_imgs = images.shape
            images = images[
                     self.crop[0]: h - self.crop[1]: self.subsample,
                     self.crop[2]: w - self.crop[3]: self.subsample,
                     :,
                     ]
            return images

        elif len(images.shape) == 4:
            h, w, num_of_imgs = images.shape[:2]
            images = images[
                     self.crop[0][0]: h - self.crop[0][1]: self.subsample,
                     self.crop[1][0]: w - self.crop[1][1]: self.subsample,
                     :,
                     ]
            images = images.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"Image shape has to be three dimensional (time as channels) or four dimensional "
                f"(time, with w x h x c). got image shape {images.shape}"
            )
        return images

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        trial_index = int(
            np.floor(
                item
                / np.floor(
                    (self.num_of_imgs - self.frame_overhead)
                    / (self.time_chunk_size - self.frame_overhead)
                )
            )
        )
        actual_trial_index = self.indices[trial_index]
        trial_portion = int(
            item
            % np.floor(
                (self.num_of_imgs - self.frame_overhead)
                / (self.time_chunk_size - self.frame_overhead)
            )
        )
        starting_img_index = int(trial_portion * self.time_chunk_size)
        starting_img_index -= trial_portion * self.frame_overhead
        ending_img_index = int(starting_img_index + self.time_chunk_size)
        # print(f'item: {item}')
        # print(f'trial index: {trial_index}, actual trial index: {actual_trial_index}, trial portion: {trial_portion}')
        # print(f'starting img: {starting_img_index}, ending img: {ending_img_index}')
        # print(f'starting response: {starting_img_index + self.frame_overhead}, ending '
        #       f'response: {ending_img_index}')
        ret = []
        for data_key in self.data_keys:
            if data_key == "images":
                value = self.get_frames(
                    actual_trial_index, starting_img_index, ending_img_index
                )
                value = torch.unsqueeze(value, 0)
            else:
                if not self._test:

                    value = self.train_responses[
                            :,
                            starting_img_index + self.frame_overhead: ending_img_index,
                            actual_trial_index,
                            ]
                else:
                    value = self.test_responses[
                            :, starting_img_index + self.frame_overhead: ending_img_index
                            ]

            ret.append(value)
        x = self.data_point(*ret)

        return x

    def get_frames(self, trial_index, starting_img_index, ending_img_index):
        cache = []
        if self._test:
            starting_line = starting_img_index
            ending_line = ending_img_index
        else:
            starting_line = (
                                    starting_img_index + self.test_frames
                            ) + self.num_of_imgs * trial_index
            ending_line = starting_line + self.time_chunk_size

        fixations = self.fixations[int(starting_line): int(ending_line)]
        frames = torch.zeros((self.time_chunk_size, self.img_h, self.img_w))
        for i, (fixation, index) in enumerate(
                zip(fixations, range(starting_img_index, ending_img_index))
        ):
            if (index >= self.last_start_index) and (index < self.last_end_index):
                img = self.cache[index - self.last_start_index]
            else:
                img = torch.from_numpy(self.frames[fixation["img_index"]].astype(np.float32))
                img = crop_based_on_fixation(
                    img=img,
                    x_center=fixation["center_x"],
                    y_center=fixation["center_y"],
                    flip=fixation["flip"] == 0,
                    img_h=self.img_h,
                    img_w=self.img_w,
                    padding=self.padding,
                )
                img = torch.movedim(img, 0, 1)
            frames[i] = img
            cache.append(img)
        frames = torch.movedim(frames, 0, 2)
        frames = self.transform(frames)
        frames = torch.movedim(frames, 2, 0)
        self.last_start_index = starting_img_index
        self.last_end_index = ending_img_index
        self.cache = cache
        return frames

def get_dataloader(
    response_dict,
    path,
    indices,
    test,
    frames,
    batch_size,
    fixations,
    temporal_dilation=1,
    shuffle=True,
    num_of_frames=15,
    num_of_hidden_frames=None,
    device="cpu",
    crop=0,
    subsample=1,
    time_chunk_size=1,
    num_of_layers=1,
    hidden_temporal_dilation=1,
    padding=200,
    full_img_w=1000,
    full_img_h=800,
    img_h=600,
    img_w=800,
):
    data = MarmosetMovieDataset(
        response_dict,
        path,
        "images",
        "responses",
        frames=frames,
        fixations=fixations,
        indices=indices,
        test=test,
        crop=crop,
        temporal_dilation=temporal_dilation,
        hidden_temporal_dilation=hidden_temporal_dilation,
        subsample=subsample,
        device=device,
        time_chunk_size=time_chunk_size,
        num_of_layers=num_of_layers,
        num_of_frames=num_of_frames,
        num_of_hidden_frames=num_of_hidden_frames,
        padding=padding,
        full_img_h=full_img_h,
        full_img_w=full_img_w,
        img_h=img_h,
        img_w=img_w
    )

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def frame_movie_loader(
    files,
    fixation_files,
    big_crops,
    basepath,
    batch_size=16,
    seed=None,
    train_frac=0.8,
    subsample=1,
    crop=0,
    num_of_trials_to_use=None,
    start_using_trial=0,
    num_of_frames=None,
    num_of_hidden_frames=None,
    temporal_dilation=1,
    hidden_temporal_dilation=1,
    cell_index=None,
    retina_index=None,
    device="cpu",
    time_chunk_size=None,
    num_of_layers=None,
    excluded_cells=None,
    frame_file="_img_",
    img_dir_name="stimuli",
    full_img_w=1400,
    full_img_h=1200,
    img_h = None,
    img_w=None,
    final_training=False,
    padding=200,
    retina_specific_crops=True,
    stimulus_seed=0,
    hard_coded=None,
    flip_imgs=False,
    shuffle=None,

):
    """
        Build train/validation/test PyTorch dataloaders for *movie–response* experiments
        using fixation-aligned image crops and trial-wise splits.

        This function wires together the I/O helpers (``process_fixations``,
        ``load_responses``, ``load_frames``), performs a **trial-wise** split of the
        training set, and instantiates three dataloaders (train, validation, test)
        per retina. Internally, each dataloader wraps a temporal dataset
        (e.g. :class:`MarmosetMovieDataset`) configured with your spatiotemporal
        parameters (frame counts, dilations, chunk size, cropping, etc.).

        Parameters
        ----------
        files : Mapping[int, str] or similar
            Specification used by ``load_responses`` to locate neural response files
            for each retina. Keys are retina indices; values are paths or file
            descriptors understood by ``load_responses``. The loaded arrays are
            expected to have shape ``(n_neurons, frames_per_trial, n_trials)``.
        fixation_files : Mapping[int, str]
            Mapping from retina index to a fixation file path (relative to
            ``basepath``). TODO: only the first entry is actually read and used
            for all retinas in this function, assuming a shared fixation stream.
        big_crops : Mapping[int, int | tuple[int, int, int, int]]
            Retina-specific crop specifications *(top, bottom, left, right)* used
            if ``retina_specific_crops=True``.
        basepath : str or pathlib.Path
            Root directory that contains the response files and the stimulus frames.
        batch_size : int, default 16
            Batch size used for all splits.
        seed : int, optional
            Random seed forwarded to the trial-wise split helper.
        train_frac : float, default 0.8
            Fraction of training trials assigned to the train split by
            ``get_trial_wise_validation_split`` (unless ``hard_coded`` dictates
            otherwise).
        subsample : int, default 1
            Spatial down-sampling factor applied to frames inside the dataset.
        crop : int or tuple[int, int, int, int], default 0
            Global crop *(top, bottom, left, right)* applied if
            ``retina_specific_crops=False``. If an ``int`` is given, the same value
            is used on all sides.
        num_of_trials_to_use : int, optional
            Cap on the number of trials to use starting from
            ``start_using_trial``. By default, uses all available trials (train +
            validation) for the selected retina.
        start_using_trial : int, default 0
            Offset (0-based) for selecting a contiguous block of trials to use.
        num_of_frames : int, optional
            Number of **visible** frames per sample (passed to the underlying
            dataset). If ``None``, the dataset default is used.
        num_of_hidden_frames : int or tuple[int], optional
            Hidden-layer look-back window(s); broadcast/forwarded to the dataset.
        temporal_dilation : int, default 1
            First layer temporal dilation; forwarded to the dataset.
        hidden_temporal_dilation : int or tuple[int], default 1
            Hidden-layer temporal dilations; forwarded to the dataset.
        cell_index : int, optional
            Used to train single cell models, selects a single neuron, forwarded to ``load_responses``.
        retina_index : int, optional
            If given, build dataloaders for only this retina. Otherwise, build
            loaders for all keys in ``fixation_files``.
        device : {"cpu", "cuda", ...}, default "cpu"
            Torch device on which tensors will be created.
        time_chunk_size : int, optional
            Length (in frames) of each temporal chunk returned by the dataset.
        num_of_layers : int, optional
            Model depth (used to broadcast hidden-frame/dilation parameters in the
            dataset).
        excluded_cells : array-like of int, optional
            Indices of neurons to drop prior to training; forwarded to
            ``load_responses``.
        frame_file : str, default "_img_"
            Substring pattern used by ``load_frames`` to find frame files.
        img_dir_name : str, default "stimuli"
            Subdirectory of ``basepath`` that contains individual stimulus frames.
        full_img_w, full_img_h : int, default 1400, 1200, which are the dimensions of non-subsampled non-cropped images.
            Spatial dimensions of the *raw* full-frame stimuli on disk. For subsampled is 350, 300.
        img_h, img_w : int, optional
            Target spatial dimensions after fixation cropping. If either is
            ``None``, both default to ``full_img_* - 3 * padding``.
        final_training : bool, default False
            Passed through to the split helper; when ``True`` you may configure the
            helper to collapse validation into training (implementation-dependent).
        padding : int, default 200
            Extra margin captured around the gaze center when extracting crops.
        retina_specific_crops : bool, default True
            If ``True``, override ``crop`` with ``big_crops[retina_index]`` for each
            retina.
        stimulus_seed : int, default 0
            Random seed forwarded to ``load_responses`` for stimulus/order
            reproducibility.
        hard_coded : Mapping[str, Sequence[int]] or None
            Optional hard-coded split specification consumed by
            ``get_trial_wise_validation_split``.
        flip_imgs : bool, default False
            If ``True``, pass a flag to ``process_fixations`` to mirror fixations
            (and thus crops) across the vertical axis.
        shuffle : bool or None, optional
            If ``None``, training dataloader shuffles and validation/test do not.
            If a boolean is given, that value is used for **both** train and
            validation; test remains ``False``.

        Returns
        -------
        dataloaders : dict[str, dict[int, torch.utils.data.DataLoader]]
            A nested dictionary with three top-level keys: ``"train"``,
            ``"validation"``, and ``"test"``. Each maps retina indices to a
            dataloader configured for that split.

            * Train/validation loaders iterate over **training responses** for the
              selected trials.
            * The test loader iterates over the **held-out test responses**; its
              dataset ignores trial indices but uses the same temporal chunking.

        Notes
        -----
        - **Expected response shapes:** ``load_responses`` should return, for each
          retina, a dict with keys ``"train_responses"`` (``n_neurons × T_train ×
          n_trials``) and ``"test_responses"`` (``n_neurons × T_test``).
        - TODO: **Fixations.** Right now, only the first path from ``fixation_files`` is opened and
          parsed via ``process_fixations`` and then reused for all retinas.
        - **Trial selection window.** After the split, trials are restricted to the
          contiguous window
          ``[start_using_trial, start_using_trial + num_of_trials_to_use)`` (clipped
          to the number of available trials).
        - **Printing side effects.** The function prints the selected training and
          validation trial IDs to stdout."""

    dataloaders = {"train": {}, "validation": {}, "test": {}}
    if retina_index is None:
        retina_indices = list(fixation_files.keys())
    else:
        retina_indices = [retina_index]

    with open(
            f"{basepath}/{fixation_files[retina_indices[0]]}", "r"
    ) as file:
        fixation_file = file.readlines()
        fixations = process_fixations(fixation_file, flip_imgs=flip_imgs)

    responses = load_responses(basepath, files=files,
                               stimulus_seed=stimulus_seed,
                               excluded_cells=excluded_cells, cell_index=cell_index)
    frames = load_frames(
        img_dir_name=os.path.join(basepath,img_dir_name),
        frame_file=frame_file,
        full_img_h=full_img_h,
        full_img_w=full_img_w,
    )

    if img_h is None:
        img_h = full_img_h - 3 * padding
        img_w = full_img_w - 3 * padding

    for retina_index in retina_indices:
        train_responses, test_responses = responses[retina_index]['train_responses'], responses[retina_index]['test_responses']

        all_train_ids, all_validation_ids = get_trial_wise_validation_split(
            train_responses=train_responses,
            train_frac=train_frac,
            seed=seed,
            final_training=final_training,
            hard_coded=hard_coded
        )
        # print("train responses shape: ", train_responses.shape)
        if num_of_trials_to_use is None:
            num_of_trials_to_use = len(all_train_ids)+len(all_validation_ids)
        if hard_coded is None:
            train_ids = np.isin(
            all_train_ids,
            np.arange(start_using_trial, min(train_responses.shape[2], num_of_trials_to_use+start_using_trial)),
        )
            train_ids = np.asarray(all_train_ids)[train_ids]
            valid_ids = np.isin(
            all_validation_ids,
            np.arange(start_using_trial, min(train_responses.shape[2], num_of_trials_to_use + start_using_trial)),
        )
            valid_ids = all_validation_ids[valid_ids]
        else:
            num_trials = train_responses.shape[-1]
            train_ids = [int(x) for x in all_train_ids if int(x) < min(num_trials, num_of_trials_to_use+start_using_trial)]
            valid_ids = [int(x) for x in all_validation_ids if int(x) < min(num_trials, num_of_trials_to_use+start_using_trial)]

        print(f'Trial separation for {retina_index}')
        print("training trials: ", len(train_ids), train_ids)
        print("validation trials: ", len(valid_ids), valid_ids, '\n')

        if retina_specific_crops:
            crop = big_crops[retina_index]
        train_loader = get_dataloader(
            {"train_responses": train_responses, "test_responses": test_responses},
            fixations=fixations,
            path=basepath,
            indices=train_ids,
            test=False,
            batch_size=batch_size,
            num_of_frames=num_of_frames,
            device=device,
            crop=crop,
            shuffle=True if shuffle is None else shuffle,
            subsample=subsample,
            time_chunk_size=time_chunk_size,
            num_of_layers=num_of_layers,
            frames=frames,
            num_of_hidden_frames=num_of_hidden_frames,
            padding=padding,
            full_img_h=full_img_h,
            full_img_w=full_img_w,
            img_h=img_h,
            img_w=img_w,
            temporal_dilation=temporal_dilation,
            hidden_temporal_dilation=hidden_temporal_dilation
        )

        valid_loader = get_dataloader(
            {"train_responses": train_responses, "test_responses": test_responses},
            path=basepath,
            indices=valid_ids,
            test=False,
            batch_size=batch_size,
            fixations=fixations,
            num_of_frames=num_of_frames,
            device=device,
            crop=crop,
            shuffle=False if shuffle is None else shuffle,
            subsample=subsample,
            time_chunk_size=time_chunk_size,
            num_of_layers=num_of_layers,
            frames=frames,
            num_of_hidden_frames=num_of_hidden_frames,
            padding=padding,
            full_img_h=full_img_h,
            full_img_w=full_img_w,
            img_h=img_h,
            img_w=img_w,
            temporal_dilation=temporal_dilation,
            hidden_temporal_dilation=hidden_temporal_dilation
        )

        test_loader = get_dataloader(
            {"train_responses": train_responses, "test_responses": test_responses},
            fixations=fixations,
            path=basepath,
            indices=train_ids,
            test=True,
            batch_size=batch_size,
            num_of_frames=num_of_frames,
            device=device,
            crop=crop,
            shuffle=False,
            subsample=subsample,
            time_chunk_size=time_chunk_size,
            num_of_layers=num_of_layers,
            frames=frames,
            num_of_hidden_frames=num_of_hidden_frames,
            padding=padding,
            full_img_h=full_img_h,
            full_img_w=full_img_w,
            img_h=img_h,
            img_w=img_w,
            temporal_dilation=temporal_dilation,
            hidden_temporal_dilation=hidden_temporal_dilation
        )

        dataloaders["train"][retina_index] = train_loader
        dataloaders["validation"][retina_index] = valid_loader
        dataloaders["test"][retina_index] = test_loader

    return dataloaders
