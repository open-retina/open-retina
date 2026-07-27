# Add the `qiu_2026` (Franke-lab) mouse dataset + MLP pupil shifter

## Summary

This PR adds native open-retina support for the **Franke-lab Qiu 2026** mouse dataset — a
multi-session two-photon recording of superior colliculus boutons responding to natural movies, with
simultaneous behavioral measurements (pupil size, locomotion, eye position). It contributes:

1. **Native data loaders** (`openretina/data_io/qiu_2026/`) that turn the shipped Sensorium-style
   FileTree sessions into open-retina's `MoviesTrainTestSplit` / `ResponsesTrainTestSplit` containers,
   with a 3-channel *video + behavior* stimulus, per-session normalization, quality masking, and a
   curated train/validation/test split.
2. **An MLP pupil shifter** (`openretina/modules/shifters/mlp_shifter.py`, Sinz et al. 2018) wired into
   `CoreReadout`, so the per-session Gaussian readout grid is shifted by the animal's eye position.
3. **Hydra configs** to train and evaluate the standard core+readout model on the dataset.

All changes to shared code are **additive and no-op / opt-in by default** — existing datasets and
models behave exactly as before.

**Result:** the full 10-session model trains end-to-end. A run over all 10 sessions (50 epochs,
`trainer=default_deterministic`, batch size 32) produces a checkpoint with held-out per-clip
correlations of **≈ 0.39–0.59** across the 98 test dataloaders, and predicted-vs-recorded inference
with the shifter active is demonstrated in `notebooks/qiu_2026_inspect_predictions.ipynb`.

---

## The dataset

- **10 sessions, 3 animals**, mouse. Session keys follow
  `dynamic{animal}-{scan}-{idx}-Fluorescence-7b721b-v4a` (`7b721b` = pipeline hash, `v4a` = version):

  | Animal | Sessions | # |
  |---|---|---|
  | `28188` | `16-3`, `16-5`, `17-2`, `18-4`, `19-9` | 5 |
  | `29163` | `2-7`, `4-4`, `5-8`, `6-5` | 4 |
  | `28712` | `3-8` | 1 |

- **Neurons (boutons):** per-session counts after the `neurons_fluor_good` quality mask range from
  **787 to 3175** (e.g. `16-3`: 7636 → 2710), ≈ **17.3k** boutons total. Modality is 2p axonal calcium
  at **30 Hz**.

- **Stimulus — `(C, T, H, W) = [3, 300, 36, 64]`.** Channel 0 is the natural-movie video; channels 1–2
  are behavior (**pupil size** and **locomotion speed**, `BEHAVIOR_CHANNELS=(0, 2)`; the Δpupil channel
  is dropped to match the reference). Behavior scalars are z-scored and broadcast across all pixels.
  The video is **normalized per session** using that session's own shipped `meta/statistics` mean/std.
  Raw trials arrive as `(H, W, 450)` NaN-padded arrays and are trimmed to the **300** valid frames of a
  clip.

- **Responses — `(N, T)`.** A per-neuron quality mask is applied, trailing whole-frame NaN padding is
  trimmed per trial, and raw calcium is mapped to a non-negative target via a `spike_inference` switch
  (`raw` / `subtract_min` / `cascade`). Default is **`subtract_min`** (per-neuron train-min subtraction);
  `cascade` currently raises `NotImplementedError` (see follow-ups). Non-negative targets pair with the
  readout's `softplus` output and `PoissonLoss3d`.

- **Pupil / eye position — `(2, T)`.** Z-scored and frame-aligned to the responses. This is **not a
  model target** — it is carried through the dataloader and fed to the shifter as `pupil_center`.

- **Splits & test tiers.** Train/validation come from the dataset's `tiers`, with a **curated
  validation carve-out** (`validation_clip_indices`) so val is reproducible rather than a random draw.
  The **test tier is bimodal**: ~6 repeated "oracle" clips (each shown 15–20 times *within* a session)
  plus singleton held-out clips; the loader represents each condition as one whole clip. Chirp /
  moving-bar `presentmoviearray` stimuli (test-tier only, with per-repeat frame jitter that used to
  crash `np.stack`) are excluded by a `stimulus_type="clip"` filter. Test **inputs** are averaged over a
  condition's repeats (a no-op for the video, a modeling choice for behavior/pupil — see follow-ups);
  per-repeat responses are retained in `test_by_trial_dict`.

---

## What was added

### (a) New modules & public API

`openretina/data_io/qiu_2026/`

| File | Contents |
|---|---|
| `constants.py` | `FRAME_RATE_MODEL=30`, `CLIP_LENGTH_FULL=450`, `CLIP_LENGTH_CUT=300`, `VIDEO_SHAPE=(1,36,64)`, `BEHAVIOR_CHANNELS=(0,2)`, `N_INPUT_CHANNELS=3`, quality-mask filename conventions. |
| `trials.py` | Shared discovery/split/trim helpers: `discover_sessions(base, sessions=None)` (applies the session filter at directory-listing time), `read_tiers`, `read_condition_hash`, `read_stimulus_type`, `load_trial_array`, `valid_length`/`trim_time`, `train_val_indices`, `validation_clip_indices`, `test_conditions`, `discover_quality_masks`. |
| `stimuli.py` | `load_all_stimuli(base, *, behavior_channels=…, stimulus_type="clip", sessions=None)` and `load_stimuli_for_session(...)` → `MoviesTrainTestSplit`; builds the `(C,T,H,W)` tensor and applies per-session normalization. |
| `responses.py` | `load_all_responses(base, *, apply_quality_mask=True, spike_inference="raw", sessions=None)` and `load_responses_for_session(...)` → `ResponsesTrainTestSplit`; the `spike_inference` switch lives here. Produces both `test_dict` (trial-averaged) and `test_by_trial_dict`. |
| `pupil.py` | `load_all_pupil(base, *, stimulus_type="clip", sessions=None)` and `load_pupil_for_session(...)` → `{"train": (2,T), "test_dict": {cond: (2,T)}}`. |
| `dataloaders.py` | `QiuDataPoint(inputs, targets, pupil_center)` namedtuple; `QiuMovieDataSet(MovieDataSet)` (carries the `(2,T)` pupil trace and slices it in lock-step with the movie); `qiu_2026_dataloaders(...)` which honours the curated val split and injects the pupil array via the shared hook. |

`openretina/modules/shifters/mlp_shifter.py` (exported from `modules/shifters/__init__.py`)

- `MLPShifter(input_channels=2, hidden_channels=5, n_layers=3)` — a small `Linear→Tanh` MLP mapping a
  `(·, 2)` pupil center to a `(·, 2)` readout-grid shift in normalized `[-1, 1]` units.
- `MultiSessionMLPShifter(n_neurons_dict=…, gamma_shifter=0.0, …)` — one `MLPShifter` per session key,
  shared across neurons within a session; provides `regularizer(data_key)`.

### (b) Surgical, backwards-compatible changes to shared code

Each is a no-op unless the qiu_2026 config/data path is used:

- **`data_io/base_dataloader.py`** — `get_movie_dataloader(...)` gains `dataset_cls=MovieDataSet` and
  `extra_arrays=None`. When unset, construction is byte-for-byte the original `MovieDataSet(...)`; the
  qiu loader passes `dataset_cls=QiuMovieDataSet, extra_arrays={"pupil_center": …}`. The shared
  `DataPoint` namedtuple is **unchanged** (a defaulted third field would break `default_collate`; hence
  the qiu-local `QiuDataPoint`).
- **`models/core_readout.py`** — `BaseCoreReadout` / `UnifiedCoreReadout` gain `shifter=None`;
  `forward(x, data_key=None, pupil_center=None)` applies a shift **only** when shifter, `pupil_center`,
  and `data_key` are all present. It aligns pupil to the core's temporal output
  (`model_cut_frames = x.time − core_out.time`, i.e. 18 frames) before the shifter. Train/val/test steps
  read `getattr(data_point, "pupil_center", None)`, so a plain `DataPoint` yields `None` and the model
  behaves exactly as before.
- **`data_io/base.py`** — `compute_data_info` now **warns instead of raising** on inconsistent
  per-session normalization stats (falling back to the first session's values), because qiu_2026
  normalizes each session with its own statistics and those aggregate stats are only used by the optional
  vector-field interpretability tool. The input-**shape** consistency check still raises.
- **`cli/train.py`** — passes a `pupil_dictionary` to the dataloader **only** when `cfg.data_io` defines
  a `pupil` block (absent for every other dataset).

### (c) Configs

- `configs/qiu_2026_core_readout.yaml` — top-level: `exp_name: core_readout_qiu_2026_mouse`,
  `model.in_shape: [3, 300, 36, 64]`, `trainer: default_deterministic`, `seed: 42`,
  `paths.data_dir` defaults to the HuggingFace URL.
- `configs/model/qiu_2026_core_readout.yaml` — `Core3d` (`temporal/spatial_kernel_sizes: [11,5,5]`,
  `input_padding: false`, `hidden_padding: false`) + `MultiSampledGaussianReadout`
  (`grid_mean_predictor: null` — decision D7, plain learned per-neuron means) + a
  `MultiSessionMLPShifter` block (`input_channels: 2`, `hidden_channels: 5`, `n_layers: 3`,
  `gamma_shifter: 0.0`).
- `configs/data_io/qiu_2026.yaml` — wires the three loaders; `responses` sets
  `apply_quality_mask: true`, `spike_inference: subtract_min`; `data_info` sets 30 Hz / mouse.
- `configs/dataloader/qiu_2026.yaml` — `qiu_2026_dataloaders`, `batch_size: 32`, `train_chunk_size: 50`,
  `clip_length: 300`.

---

## Using the dataset

### 1. Point open-retina at the data

`paths.data_dir` defaults to the HuggingFace dataset URL; anything downloaded lands under
`$OPENRETINA_CACHE_DIRECTORY` (default `~/openretina_cache`), i.e.
`…/openretina_cache/franke_lab/qiu_2026/<session>/`. To read from an existing local copy instead,
override `paths.data_dir=/path/to/franke_lab/qiu_2026`. (On the Bethge cluster the data is already on
the shared weka cache, so no override is needed.)

### 2. Train

```bash
# Full 10-session run (recommended; used for the DoD checkpoint)
sbatch run_qiu_full_train.sh
#   → h100-ferranti / 1×H100 / 16 CPU / 256 GB / 1 day,
#     caps OMP/MKL/OPENBLAS_NUM_THREADS=16, then:
#     uv run openretina train --config-name qiu_2026_core_readout trainer=default_deterministic

# Equivalent bare command
uv run openretina train --config-name qiu_2026_core_readout trainer=default_deterministic
```

Notes:
- Use **`trainer=default_deterministic`** (50 epochs, GPU) — **not** `trainer=debug`, which forces CPU +
  1 epoch + anomaly detection and will not produce a real checkpoint.
- If a run OOMs on GPU, drop `dataloader.batch_size` (e.g. 16 or 8).
- Checkpoints are written to
  `openretina_assets/runs/core_readout_qiu_2026_mouse/<timestamp>/checkpoints/*.ckpt`.
- Restrict to specific sessions by adding the filter to all three loader streams:
  ```bash
  '+data_io.stimuli.sessions=["dynamic28188-16-3-Fluorescence-7b721b-v4a"]' \
  '+data_io.responses.sessions=["dynamic28188-16-3-Fluorescence-7b721b-v4a"]' \
  '+data_io.pupil.sessions=["dynamic28188-16-3-Fluorescence-7b721b-v4a"]'
  ```
  (All three must be filtered identically or the stimulus↔response alignment check fails.)

### 3. Load a checkpoint and run inference with the shifter

```python
import torch
from openretina.models.core_readout import load_core_readout_model

SESSION = "dynamic28188-16-3-Fluorescence-7b721b-v4a"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_core_readout_model(str(ckpt_path), device).eval()

# inputs:       (B, 3, T, H, W)   1 video + 2 behavior channels, H×W = 36×64
# pupil_center: (B, 2, T)         eye position → shifter
out = model.forward(inputs, data_key=SESSION, pupil_center=pupil_center)  # → (B, T_out, N)
# T_out = T − 18 : the core's un-padded temporal convs (kernels 11,5,5) crop sum(k−1)=18 frames.
# Align the recorded trace by dropping its first (T − T_out) frames.
```

Passing `pupil_center=None` reproduces the pre-shifter path exactly — useful for quantifying the
shifter's contribution.

### 4. Call the loaders directly in Python

```python
import os
os.environ["OPENRETINA_CACHE_DIRECTORY"] = "/path/to/cache"
from openretina.data_io.qiu_2026.stimuli import load_all_stimuli
from openretina.data_io.qiu_2026.responses import load_all_responses
from openretina.data_io.qiu_2026.pupil import load_all_pupil

base = "/path/to/franke_lab/qiu_2026"     # or the HF URL to download into the cache
sessions = ["dynamic28188-16-3-Fluorescence-7b721b-v4a"]
movies    = load_all_stimuli(base, sessions=sessions)                       # dict[str, MoviesTrainTestSplit]
responses = load_all_responses(base, spike_inference="subtract_min", sessions=sessions)
pupil     = load_all_pupil(base, sessions=sessions)
```

---

## Results

Full run (Slurm job `420322`, `trainer=default_deterministic`, batch size 32, 50 epochs, all 10
sessions): saved checkpoint
`…/2026-07-23_14-52-28/checkpoints/epoch=49_val_evaluation_loss=0.402_final.ckpt`, complete test loop
over all 98 test dataloaders, per-clip **correlations ≈ 0.39–0.59**. (`PoissonLoss3d` reports large
negative values on the test tier — this is the loss's constant term, not a training failure; the
correlation metric is the meaningful readout here.)

---

## Tests

- `tests/models/test_core_readout_shifter.py` — 6 tests covering the shifter and the shifter-aware
  `forward` / train-val-test steps (including that `shifter=None` / no `pupil_center` reproduces the
  pre-shifter output).
- `tests/data_io/test_qiu_2026_train_wiring.py` — 2 offline, synthetic end-to-end wiring tests.
- No regressions in the existing model/module suite.

---

## Known limitations / follow-ups

- **CASCADE spike inference** is deferred — `spike_inference="cascade"` raises; `subtract_min` is the
  training default. The `raw`/`subtract_min`/`cascade` switch is in place so targets can be compared
  once a CASCADE recipe is chosen.
- **Test-input averaging over repeats** — inputs are averaged across a condition's repeats (a no-op for
  the video, a modeling choice for behavior/pupil). Flagged `# TODO(qiu_2026)` at both sites;
  `test_by_trial_dict` keeps the per-repeat responses.
- **A real-data data_io test suite** (exercising the actual loaders / FileTree, incl. a regression guard
  for the `presentmoviearray` fix) is not yet written — see `qiu_2026_integration_plan.md`, Step A.
- **Dead-but-harmless cortex-coordinate plumbing** remains in `responses.py`
  (`cell_motor_coordinates`) since `grid_mean_predictor: null` (D7); a candidate for later cleanup.

## Notebooks

- **`notebooks/qiu_2026_walkthrough.ipynb`** — a stage-by-stage re-implementation of what
  `uv run openretina train --config-name qiu_2026_core_readout` does under the hood: compose the config,
  call the three loaders, inspect a `QiuDataPoint` batch, build the model, and run a forward pass with
  and without the shifter. Educational / inspection notebook.
- **`notebooks/qiu_2026_inspect_predictions.ipynb`** — loads a trained checkpoint and compares predicted
  vs recorded responses on held-out clips for one session **with the shifter active**, plotting the
  per-neuron correlation distribution and example traces.
