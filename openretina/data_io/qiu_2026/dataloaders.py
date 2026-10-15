"""qiu_2026 dataloaders that carry ``pupil_center`` alongside inputs/targets.

open-retina's shared ``DataPoint`` is a 2-field ``(inputs, targets)`` namedtuple; extending it would break
every existing dataset (see the integration plan §0.4). Instead this module defines a qiu-local
``QiuDataPoint`` and a ``QiuMovieDataSet`` that slices a per-frame pupil trace with the same chunking as the
movie/responses, injected into :func:`get_movie_dataloader` via its ``dataset_cls``/``extra_arrays`` hook.

``qiu_2026_dataloaders`` mirrors :func:`multiple_movies_dataloaders` but (a) splits the pupil trace
identically to the movie via :func:`generate_movie_splits`, (b) honors each session's *curated* validation
split (the ``validation_clip_indices`` stashed in ``session_kwargs``) so validation trials are carved out of
training rather than chosen at random, (c) evaluates each test condition as one whole clip, and (d) by
default puts that condition's repeats in the batch dimension so the model runs once per repeat with that
repeat's own behavior, and the predictions -- not the inputs -- are averaged.
"""

import collections
from collections import namedtuple
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.data_io.base_dataloader import (
    MovieDataSet,
    NeuronDataSplit,
    _compute_test_batch_size,
    generate_movie_splits,
    get_movie_dataloader,
)

# qiu-local batch object: three real tensors, so PyTorch default_collate handles it unchanged.
QiuDataPoint = namedtuple("QiuDataPoint", ["inputs", "targets", "pupil_center"])

# Test-time variant whose batch dimension holds a condition's repeats rather than time chunks.
# ``reduce_over_batch`` marks it for `BaseCoreReadout.test_step`, which then averages the predictions
# over that dimension before scoring them. The marker rides on the batch rather than being a flag on
# the model because `cli/train.py` hands *every* dataloader -- train and validation included -- to
# `trainer.test`, and a model-level switch would reduce those too. `default_collate` preserves the
# namedtuple type, so the field survives batching.
QiuRepeatDataPoint = namedtuple("QiuRepeatDataPoint", ["inputs", "targets", "pupil_center", "reduce_over_batch"])


class QiuMovieDataSet(MovieDataSet):
    """``MovieDataSet`` that additionally carries a per-frame ``pupil_center`` ``(2, T)`` trace.

    ``pupil_center`` is sliced on its time axis (axis 1) with the exact chunk the movie (axis 1) and
    responses (axis 0) are sliced with, and returned as the third field of a :class:`QiuDataPoint`.
    """

    def __init__(
        self,
        movies,
        responses,
        roi_ids,
        roi_coords,
        group_assignment,
        split,
        chunk_size,
        pupil_center,
    ):
        super().__init__(movies, responses, roi_ids, roi_coords, group_assignment, split, chunk_size)
        self.pupil_center = torch.as_tensor(pupil_center, dtype=torch.float)  # (2, T)

    def __getitem__(self, idx) -> QiuDataPoint:  # type: ignore[override]
        if isinstance(idx, slice):
            return QiuDataPoint(self.samples[0][:, idx, ...], self.samples[1][idx, ...], self.pupil_center[:, idx])
        start = int(idx)
        chunk = slice(start, start + self.chunk_size)
        return QiuDataPoint(
            self.samples[0][:, chunk, ...],
            self.samples[1][chunk, ...],
            self.pupil_center[:, chunk],
        )


class QiuRepeatTestDataSet(MovieDataSet):
    """Test dataset whose index dimension is the *repeat*, not a time chunk.

    A test condition is one clip presented 15-20 times. The movie is identical on every repeat but the
    animal's behavior is not, so each item re-assembles the ``(C, T, H, W)`` input from one shared video
    channel plus that repeat's own behavior channels, and pairs it with that repeat's own pupil trace.
    The target is the trial-averaged response, the same for every item: the model sees R inputs and
    :meth:`BaseCoreReadout.test_step` averages the R predictions before scoring them, which is what the
    trial-averaged response actually is -- the average over repeats of what the retina did while the
    animal behaved as it did on that repeat.

    Storing R assembled input tensors instead would cost ~R times the test movie memory for a video that
    never changes (several GB across the full dataset), so the video is kept once and broadcast here.
    """

    def __init__(
        self,
        movies,
        responses,
        roi_ids,
        roi_coords,
        group_assignment,
        split,
        chunk_size,
        behavior_by_repeat,
        pupil_by_repeat,
    ):
        super().__init__(movies, responses, roi_ids, roi_coords, group_assignment, split, chunk_size)
        self.behavior_by_repeat = torch.as_tensor(behavior_by_repeat, dtype=torch.float)  # (R, n_beh, T)
        self.pupil_by_repeat = torch.as_tensor(pupil_by_repeat, dtype=torch.float)  # (R, 2, T)
        # Channel 0 of the repeat-averaged test movie is the video, identical on every repeat. A view.
        self.video = self.samples[0][:1]  # (1, T, H, W)

        n_repeats, _, n_time = self.behavior_by_repeat.shape
        if self.pupil_by_repeat.shape[0] != n_repeats:
            raise ValueError(
                f"Behavior has {n_repeats} repeats but pupil has {self.pupil_by_repeat.shape[0]}; "
                "both must come from the same test condition."
            )
        if n_time != self.video.shape[1] or self.pupil_by_repeat.shape[2] != self.video.shape[1]:
            raise ValueError(
                f"Per-repeat traces are not frame-aligned with the movie: behavior {n_time} frames, "
                f"pupil {self.pupil_by_repeat.shape[2]}, movie {self.video.shape[1]}."
            )

    def __len__(self) -> int:
        return int(self.behavior_by_repeat.shape[0])

    def __getitem__(self, idx) -> QiuRepeatDataPoint:  # type: ignore[override]
        repeat = int(idx)
        _, n_time, height, width = self.video.shape
        behavior = self.behavior_by_repeat[repeat][:, :, None, None].expand(-1, n_time, height, width)
        return QiuRepeatDataPoint(
            torch.cat([self.video, behavior], dim=0),
            self.samples[1],
            self.pupil_by_repeat[repeat],
            torch.tensor(True),
        )

    def __str__(self) -> str:
        return (
            f"QiuRepeatTestDataSet with {len(self)} repeats of a movie of shape "
            f"{list(self.samples[0].shape)} and {self.samples[1].shape[1]} neuron responses."
        )


def _split_pupil(
    pupil_train: np.ndarray,
    pupil_test_dict: dict[str, np.ndarray],
    val_clip_idx: list[int],
    num_clips: int,
    clip_length: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Split a ``(2, T)`` pupil trace exactly like the movie, by shaping it as a ``(2, T, 1, 1)`` fake movie."""
    fake_train = pupil_train[:, :, None, None]
    fake_test = {name: trace[:, :, None, None] for name, trace in pupil_test_dict.items()}
    train_subset, val, test = generate_movie_splits(
        fake_train, fake_test, val_clip_idc=val_clip_idx, num_clips=num_clips, clip_length=clip_length
    )
    return train_subset[:, :, 0, 0], val[:, :, 0, 0], {name: t[:, :, 0, 0] for name, t in test.items()}


def qiu_2026_dataloaders(
    neuron_data_dictionary: dict[str, ResponsesTrainTestSplit],
    movies_dictionary: dict[str, Any],
    pupil_dictionary: dict[str, dict[str, Any]],
    train_chunk_size: int = 50,
    batch_size: int = 32,
    seed: int = 42,
    clip_length: int = 300,
    num_val_clips: int = 10,
    val_clip_indices: list[int] | None = None,
    allow_over_boundaries: bool = False,
    release_movies: bool = False,
    per_repeat_test_behavior: bool = True,
) -> dict[str, dict[str, DataLoader]]:
    """Build ``dict[split][session] -> DataLoader`` yielding :class:`QiuDataPoint` batches.

    Validation-clip selection per session, in priority order: the explicit ``val_clip_indices`` argument, then
    the session's curated ``session_kwargs["validation_clip_indices"]``, then a random draw of
    ``num_val_clips``. The curated path is the intended one — it removes the dataset's designated validation
    trials from the training subset. Each test condition is evaluated as a single whole clip.

    Args:
        per_repeat_test_behavior: Evaluate each test condition by feeding every repeat's behavior and pupil
            trace through the model separately and averaging the R *predictions*, instead of averaging the
            R inputs and running the model once. The video is the same clip on every repeat, but the
            behavior channels and the pupil centre are not, and the model is nonlinear in both, so
            ``f(video, mean_r b_r) != mean_r f(video, b_r)``. Test dataloaders then yield
            :class:`QiuRepeatDataPoint` batches of R repeats; train and validation are unaffected either
            way. Set to ``False`` to reproduce the repeat-averaged-input numbers.

            Measured on the 10-session checkpoint, session ``dynamic28188-16-3``, over its 6 oracle
            conditions (``qiu_legacy_scripts/qiu_ab_repeat_estimator.py``): **median per-neuron
            correlation +0.012**, positive in all 6 conditions, 63-79% of reliable neurons improving.
            Real and consistent, but small. For context the same run found that removing the shifter
            entirely costs **-0.160** on averaged input -- so the shifter is *not* rendered inert by
            averaging its input, and the gain here comes from the nonlinearity, not from rescuing the
            shifter. Conditions with a single repeat are unaffected by construction, and this session
            has 13 of those out of 95.
        release_movies: **Destructive.** Drop each session's entry from ``movies_dictionary`` as soon as its
            splits have been taken, so already-consumed source movies are freed while the loop is still
            running. The full dataset is ~28 GB of movies and the splits duplicate all of it, so without
            this the caller holds both copies of every session at once; releasing as we go is what keeps
            peak RSS down (freeing afterwards would only help the steady state). The caller's dictionary is
            left EMPTY, so anything needing the raw movies — notably
            :func:`~openretina.data_io.base.compute_data_info`, which reads ``.train.shape`` — must run
            before this function, as ``openretina/cli/train.py`` and the qiu notebooks both do.

            Only frees anything if the caller passes its own dictionary directly. Going through
            ``hydra.utils.instantiate(cfg.dataloader, movies_dictionary=...)`` hands this function an
            OmegaConf-rebuilt copy of the container and of each wrapper, so dropping entries from it
            leaves the caller's references — and therefore the arrays — alive. Instantiate the builder
            with ``_partial_=True`` and call it normally instead; both CLI entry points do.
    """
    assert set(neuron_data_dictionary) == set(movies_dictionary) == set(pupil_dictionary), (
        "neuron_data_dictionary, movies_dictionary and pupil_dictionary must share the same session keys."
    )

    dataloaders: dict[str, Any] = collections.defaultdict(dict)
    for session_key, session_data in tqdm(neuron_data_dictionary.items(), desc="Creating qiu movie dataloaders"):
        num_clips = movies_dictionary[session_key].train.shape[1] // clip_length

        if val_clip_indices is not None:
            val_clip_idx = list(val_clip_indices)
        elif "validation_clip_indices" in session_data.session_kwargs:
            val_clip_idx = list(session_data.session_kwargs["validation_clip_indices"])
        else:
            rnd = np.random.RandomState(seed)
            val_clip_idx = list(rnd.choice(num_clips, num_val_clips, replace=False))

        movie_train_subset, movie_val, movie_test_dict = generate_movie_splits(
            movies_dictionary[session_key].train,
            movies_dictionary[session_key].test_dict,
            val_clip_idc=val_clip_idx,
            num_clips=num_clips,
            clip_length=clip_length,
        )
        # Read before the release_movies pop below, which empties the caller's dictionary.
        behavior_by_repeat_dict = getattr(movies_dictionary[session_key], "test_behavior_by_repeat_dict", {}) or {}
        # Last use of the raw movie for this session: everything below works off the splits above.
        # Safe to mutate mid-loop because we iterate `neuron_data_dictionary`, not this dictionary.
        if release_movies:
            movies_dictionary.pop(session_key, None)

        pupil_train_subset, pupil_val, pupil_test_dict = _split_pupil(
            pupil_dictionary[session_key]["train"],
            pupil_dictionary[session_key]["test_dict"],
            val_clip_idx,
            num_clips,
            clip_length,
        )
        neuron_data = NeuronDataSplit(
            responses=session_data, val_clip_idx=val_clip_idx, num_clips=num_clips, clip_length=clip_length
        )

        for fold, movie, pupil, chunk_size in [
            ("train", movie_train_subset, pupil_train_subset, train_chunk_size),
            ("validation", movie_val, pupil_val, clip_length),
        ]:
            dataloaders[fold][session_key] = get_movie_dataloader(
                movie=movie,
                responses=neuron_data.response_dict[fold],
                split=fold,
                chunk_size=chunk_size,
                batch_size=batch_size,
                scene_length=clip_length,
                allow_over_boundaries=allow_over_boundaries,
                dataset_cls=QiuMovieDataSet,
                extra_arrays={"pupil_center": pupil},
            )

        pupil_by_repeat_dict = pupil_dictionary[session_key].get("test_by_repeat_dict", {})
        if per_repeat_test_behavior:
            missing = [
                name
                for name in movie_test_dict
                if name not in behavior_by_repeat_dict or name not in pupil_by_repeat_dict
            ]
            if missing:
                raise ValueError(
                    f"Session {session_key}: per_repeat_test_behavior=True, but no per-repeat behavior/pupil "
                    f"was loaded for test conditions {missing}. Load stimuli and pupil with the qiu_2026 "
                    f"loaders (both populate them), or set dataloader.per_repeat_test_behavior=false to "
                    f"evaluate with repeat-averaged inputs."
                )

        # Each test condition is a single whole clip (300 or 450 frames): evaluate it in one chunk.
        for name, movie in movie_test_dict.items():
            test_chunk_size = movie.shape[1]
            if per_repeat_test_behavior:
                n_repeats = int(behavior_by_repeat_dict[name].shape[0])
                dataloaders[name][session_key] = get_movie_dataloader(
                    movie=movie,
                    responses=neuron_data.response_dict_test[name],
                    split="test",
                    chunk_size=test_chunk_size,
                    # One batch holding every repeat, so test_step can average the predictions over it.
                    batch_size=n_repeats,
                    # Explicit: the default start indices walk the time axis, but here the dataset is
                    # indexed by repeat, so leaving it to the default would yield only the first repeat.
                    start_indices=list(range(n_repeats)),
                    scene_length=clip_length,
                    allow_over_boundaries=True,
                    dataset_cls=QiuRepeatTestDataSet,
                    extra_arrays={
                        "behavior_by_repeat": behavior_by_repeat_dict[name],
                        "pupil_by_repeat": pupil_by_repeat_dict[name],
                    },
                )
            else:
                test_batch_size = _compute_test_batch_size(batch_size, train_chunk_size, test_chunk_size)
                dataloaders[name][session_key] = get_movie_dataloader(
                    movie=movie,
                    responses=neuron_data.response_dict_test[name],
                    split="test",
                    chunk_size=test_chunk_size,
                    batch_size=test_batch_size,
                    scene_length=clip_length,
                    allow_over_boundaries=True,
                    dataset_cls=QiuMovieDataSet,
                    extra_arrays={"pupil_center": pupil_test_dict[name]},
                )

    return dataloaders
