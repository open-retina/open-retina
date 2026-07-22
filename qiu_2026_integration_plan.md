# Integration Plan: `qiu_2026` Dataset

Integrate the Franke-lab Qiu 2026 mouse dataset (10 sessions / 3 animals) so open-retina can
**train and evaluate** on it. Data-loading, model-side wiring (shifter) and Hydra configs are all
implemented and committed. **What's left is purely about running it on hardware with enough RAM** — see
`## Status as of 2026-07-22` immediately below for exactly where to pick up.

> Full history, verified-breakage analyses, and the intermediate design debates that led here live in
> **`qiu_2026_integration_plan_legacy.md`**. This document is the current, trimmed view: a summary of
> what exists and a detailed plan for what's left.

> **Recent decisions (2026-07-17), folded in below:**
> 1. **Ignore `source_grid` entirely.** The readout uses plain learned per-neuron means
>    (`grid_mean_predictor: null`); cortex coordinates are not consumed. This **removes the old
>    "Step 1"** (source_grid → `grid_mean_predictor`). Matches the reference's `v05aa`
>    ("no cortex coordinates") ablation.
> 2. **Movies are normalized per session** (each session's own shipped video stats) — already what the
>    loader does; this closes the cross-session-reconciliation question.

---

## Status as of 2026-07-22 — read this first if picking up the work

**Steps 1 and 2 are done and committed** on `qiu_2026-integration`:
- `8d03ffb` — shifter wiring in `core_readout.py` (Step 1) + all four Hydra configs and the `train.py`
  pupil-dictionary wiring (Step 2), with unit tests (`tests/models/test_core_readout_shifter.py`) and an
  offline synthetic end-to-end test (`tests/data_io/test_qiu_2026_train_wiring.py`) that drives the real
  configs through one training step without needing the real dataset.
- `923200c` — two **real-data-only bugs**, invisible to any single-session or synthetic test, found by
  actually running the real dataset through the pipeline for the first time (see below). Both fixed.

**As of this commit, these 2 commits are on the local `qiu_2026-integration` branch but not yet pushed**
(`git log origin/qiu_2026-integration..HEAD` — check before assuming the remote has them).

### The real dataset is downloaded and verified

All 10 sessions are confirmed to live at
`https://huggingface.co/datasets/open-retina/open-retina/tree/main/franke_lab/qiu_2026` (verified via
`HfApi.list_repo_files`) — the guessed path in this plan was correct. They are downloaded and extracted
locally on the machine this session ran on, at `/Users/lhoefling/data/franke_lab/qiu_2026`
(~50.6 GB compressed / ~131 GB extracted). If continuing on that same machine, point
`paths.data_dir` at that local folder directly (`get_local_file_path` returns a local path unchanged) to
skip re-downloading; on a different machine, `paths.data_dir` pointing at the HF URL above will trigger
the same download via `openretina.utils.file_utils.get_local_file_path`.

### Two bugs found and fixed by running on real data (both in commit `923200c`)

1. **Stimulus-type filtering was never actually implemented**, despite this plan's status table
   previously claiming it was done. Some sessions' `test` tier mixes in `presentmoviearray` trials
   (`chirp`/`moving_bar` functional-characterization stimuli) alongside real clips; these have
   per-repeat frame-count jitter (959 vs 960 frames) that crashes `np.stack` when building test-condition
   repeats. Fixed: `trials.read_stimulus_type()` (keys off `clip_movie_name` being populated) now filters
   `train_val_indices`/`validation_clip_indices`/`test_conditions`, threaded through all three loaders
   with `stimulus_type: str = "clip"` defaulting to the old (intended) behavior. Train/validation tiers
   were empirically confirmed clip-only in every session (0 contamination), so this only affects test-set
   construction.
2. **`compute_data_info` hard-crashed** on qiu_2026's intentional per-session video normalization (each
   session z-scored with its own stats, no shared global scalar — decision 2) because it asserted all
   sessions share one `norm_mean`/`norm_std`. That value is only consumed by the optional
   `insilico/vector_field_analysis` tool, not training, so the mismatch case now warns (matching an
   existing warn-not-crash precedent a few lines above in the same function) instead of raising.

Also found and fixed **within Step 2** (before the above): the model config's `core.hidden_padding` was
set to `true` per this plan's own (incorrect) Step 2 instructions below, but that contradicts the plan's
own worked example (`(64, 18, 46)` spatial output, `T_out = T_in − 18`) and the reference's actual
`padding=False`-everywhere spec. Verified empirically — `hidden_padding: false` is what reproduces
`(64, 18, 46)` for a `(3, 40, 36, 64)` input. **`configs/model/qiu_2026_core_readout.yaml` correctly has
`hidden_padding: false`; if any future edit changes it back to `true`, that's the bug, not a fix.**

### Hardware blocker — this is the actual remaining gap, not code

This machine has **34.4 GB RAM**. Real-data attempts, in order:
- All 10 sessions (`load_all_*` with no `sessions` filter, default `batch_size=32`): movie+response
  arrays alone estimate to ~35 GB resident simultaneously before any per-batch activation memory — SIGKILL
  (OOM) during Sanity Checking, the very first real batch.
- 5 largest sessions, default `batch_size=32`: same SIGKILL, same point (Sanity Checking, first batch) —
  ruled out as purely a "too many sessions" issue since 2-session runs hit the identical point.
- 2 smallest sessions (`dynamic28188-16-5`, `dynamic28188-16-3`, ~2.8 GB combined), default
  `batch_size=32`: still SIGKILL at the same point — confirms it's not about total resident data size but
  about **per-batch forward-pass memory** (a `(32, 3, 300, 36, 64)` validation batch through 3 un-padded
  conv layers).
- 2 smallest sessions, `dataloader.batch_size=4`: **no crash.** Got through model build, Sanity Checking
  (slowly — 8m48s for 2 validation batches) and 33/300 steps into training (epoch 0), at which point it
  was manually stopped (`SIGTERM`, not a crash) because system swap had filled to 17.99/18.4 GB and the
  whole machine was severely degraded (heavy compression + swap thrashing). It was still progressing when
  stopped, not stuck.

**Conclusion: the code is correct and now runs on real data; this specific machine just doesn't have
enough RAM to do it at a usable speed, even at 2 sessions / batch_size=4.** No checkpoint has been
produced anywhere yet.

**Next step for whoever picks this up:** run the literal DoD command (or the reduced-session variant
below) on a machine with substantially more RAM (a real GPU box, not just more CPU RAM, would also fix the
`accelerator=cpu` speed problem — `trainer=debug` forces CPU). The reduced command that got furthest here:

```bash
OPENRETINA_CACHE_DIRECTORY=/Users/lhoefling/data uv run openretina train \
  --config-name qiu_2026_core_readout trainer=debug \
  paths.data_dir=/Users/lhoefling/data/franke_lab/qiu_2026 \
  '+data_io.stimuli.sessions=["dynamic28188-16-5-Fluorescence-7b721b-v4a","dynamic28188-16-3-Fluorescence-7b721b-v4a"]' \
  '+data_io.responses.sessions=["dynamic28188-16-5-Fluorescence-7b721b-v4a","dynamic28188-16-3-Fluorescence-7b721b-v4a"]' \
  '+data_io.pupil.sessions=["dynamic28188-16-5-Fluorescence-7b721b-v4a","dynamic28188-16-3-Fluorescence-7b721b-v4a"]' \
  dataloader.batch_size=4
```

On a more capable machine, first try the literal DoD command with no overrides (all 10 sessions, default
batch size); fall back to the reduced command above only if memory is still a problem.

---

## Status at a glance

| Piece | Status | Location |
|---|---|---|
| Constants + shared split/trim/discovery helpers | ✅ Done | `data_io/qiu_2026/constants.py`, `trials.py` |
| 3-channel *video + behavior* stimulus loader, **per-session normalization** | ✅ Done | `data_io/qiu_2026/stimuli.py` |
| Response loader (quality mask, NaN-trim, spike-inference switch) | ✅ Done | `data_io/qiu_2026/responses.py` |
| Normalized, frame-aligned pupil loader | ✅ Done | `data_io/qiu_2026/pupil.py` |
| `QiuDataPoint` + `QiuMovieDataSet` + `qiu_2026_dataloaders` (carries `pupil_center`, honours curated val split) | ✅ Done | `data_io/qiu_2026/dataloaders.py` |
| Additive, no-op-by-default `dataset_cls`/`extra_arrays` hook | ✅ Done | `data_io/base_dataloader.py:270-341` |
| MLP shifter (single + multi-session), Sinz 2018 | ✅ Done | `modules/shifters/mlp_shifter.py` |
| Verification notebook (data + modules, end-to-end on cached session) | ✅ Done | `notebooks/qiu_2026_verification.ipynb` |
| ~~Per-session `source_grid` → readout `grid_mean_predictor`~~ | 🚫 **Dropped** | decision 1 — use `grid_mean_predictor: null` |
| **Shifter wired into `CoreReadout.forward` + train/val/test steps** | ✅ Done | `core_readout.py` (commit `8d03ffb`) |
| **Hydra configs** (data_io / dataloader / model / top-level) | ✅ Done | `configs/**/qiu_2026*` (commit `8d03ffb`) |
| Stimulus-type filter (**actually implemented now** — plan previously claimed done but wasn't) | ✅ Done | `trials.py` + all loaders (commit `923200c`) |
| `compute_data_info` per-session norm-stat crash | ✅ Fixed (warn, not raise) | `data_io/base.py` (commit `923200c`) |
| **Tests** — unit + offline synthetic done; dedicated real-session data_io suite still open | 🟡 Partial (Step 3) | `tests/models/test_core_readout_shifter.py`, `tests/data_io/test_qiu_2026_train_wiring.py` |
| **Run the DoD command to completion** | ❌ Blocked on hardware, not code | see "Status as of 2026-07-22" above |
| **CASCADE spike inference** (loaders currently default to `subtract_min`) | ❌ Open decision (Step 4) | `responses.py` |
| Now-dead cortex-coord plumbing in `responses.py` | 🧹 Optional cleanup | `responses.py:101-109` |

---

## Settled decisions (already implemented — do not revisit)

- **Native openretina loaders**, not the reference notebook's `sensorium.mouse_video_loader`
  (`sensorium`/`nnfabrik` are not dependencies). Loaders produce `MoviesTrainTestSplit` /
  `ResponsesTrainTestSplit` and feed the standard `LongCycler` stack.
- **Shared `DataPoint` is unchanged.** Behavior folds into the video as extra input channels
  (`C = 3`); `pupil_center` rides a qiu-local `QiuDataPoint(inputs, targets, pupil_center)` (3 real
  tensors → `default_collate` works). A defaulted-`None` 3rd field on the shared `DataPoint` would
  crash collation for every existing dataset — that path is closed.
- **Behavior channels `[0, 2]`** = pupil size + locomotion (Δpupil dropped) → `C = 3`.
- **Stimulus filter defaults to `"clip"`** (natural movies, 300 valid frames); `presentmoviearray` trials
  (`chirp`/`moving_bar` functional-characterization stimuli, test tier only, ~959-960 valid frames with
  per-repeat jitter — not `monet2`/`trippy` as earlier guessed, and not the same padded length as clips)
  are excluded via `trials.read_stimulus_type()`. The same `stimulus_type` threads through all three
  loaders so streams stay frame-aligned. **This is now actually implemented (commit `923200c`)** — see
  "Status as of 2026-07-22" at the top; it was previously listed as done here but wasn't.
- **Trimming:** trailing whole-frame NaN padding is stripped per trial before concatenation; train/val
  trials are a uniform 300 frames, test conditions internally uniform at their native length.
- **Responses → non-negative target.** `spike_inference` switch: `"raw"` (has ~25% negatives,
  Poisson-incompatible), `"subtract_min"` (current default, per-neuron train-min subtracted →
  non-negative interim target), `"cascade"` (raises `NotImplementedError` — see remaining Step 4).
- **Per-session movie normalization (decision 2).** Each session's video channel is z-scored with that
  session's own shipped `meta/statistics/videos/all/{mean,std}.npy` (a single scalar), and each
  behavior channel with its own `meta/statistics/behavior` stat — `stimuli.py:37-68`. This mirrors the
  reference (`dat.statistics.videos.all.mean/std` per session). **No cross-session reconciliation and
  no single shared global scalar** — the shared core sees per-session-standardized inputs by design.
- **Drop `source_grid` / cortex coordinates (decision 1).** The Gaussian readout uses plain learned
  per-neuron means (`grid_mean_predictor: null`). `responses.py` still loads/masks
  `cell_motor_coordinates` into `session_kwargs["cell_motor_coordinates"]` (`:101-109`), but **nothing
  consumes it** now; it is dead but harmless. `validation_clip_indices` in the same `session_kwargs`
  **stays** (it drives the curated val split). Optional cleanup: drop the `cell_motor_coordinates`
  entry and the `coordinate_dims` param from `responses.py`.

### Target model architecture (from the Qiu 2026 methods) — guides the remaining model/config work

Dynamic model = **3D factorized conv core** (Höfling 2024) + **Gaussian readout** (Lurz 2020b) +
**shifter network** (Sinz 2018), positive output. The paper's full model also adds a
**`grid_mean_predictor`** (Bashiri 2021) mapping cortex coordinates → readout mean, but per **decision 1
we deliberately omit it** and learn per-neuron means directly (the reference calls this the `v05aa`
"no cortex coordinates" variant; `v05aaa` is the with-coordinates variant we are not using).

Concrete hyperparameters read off the reference notebook
(`retina-axon-model/Analyses/Digital_twin_model/start-data-model.ipynb`):

- **Core:** three factorized `(spatial conv + temporal conv + BatchNorm + ELU)` blocks,
  `hidden_channels = [64, 64, 64]`, spatial kernels `(11,5,5)`, temporal kernels `(11,5,5)`,
  `stride=1`, `padding=False`, `batch_norm=True` with `momentum=0.7`, `independent_bn_bias=True`,
  `final_nonlin=True`, `input_regularizer='LaplaceL2norm'`, `gamma_input_spatial=10`,
  `gamma_input_temporal=0.01`. → openretina `SimpleCoreWrapper(channels=[3,64,64,64], ...)`.
  For `36×64` input the core emits `(64, 18, 46)` spatially; the temporal convs cut
  `sum(kernel-1) = 10+4+4 = 18` frames, so `T_out = T_in − 18`.
- **Readout:** Gaussian `N(μ, Σ)` per bouton, `gauss_type='full'`, learned per-neuron `μ`
  (**`grid_mean_predictor=null`**), reference inits `init_mu_range=0.1`, `init_sigma=0.3`,
  `feature_reg_weight (gamma_readout)=1.0`. openretina's `MultiSampledGaussianReadout` /
  `PointGaussianReadout` supports `grid_mean_predictor=None` directly.
- **Shifter:** 3-layer MLP, 5 hidden, `tanh`, maps `pupil_center (2) → shift (2)` added to `μ`,
  shared per session. Reference uses `gamma_shifter=0` (no shifter regularization). **Module built**;
  **model integration pending (Step 1).**
- **Output nonlinearity:** the readout's default `softplus` gives a strictly-positive Poisson rate
  (correct for non-negative targets + `PoissonLoss3d`). The reference wraps the encoder output in
  `ELU+1`; that is a faithfulness-only toggle, deferred.

### Data-format quick reference

Sensorium FileTree, per-trial `.npy` under `data/<stream>/{i}.npy`, positionally aligned to
`meta/trials/*.npy` rows, NaN-padded to 450 frames @ 30 fps. Streams: `videos (H=36,W=64,T)`,
`responses (N,T)` raw calcium, `behavior (3,T)`, `pupil_center (2,T)`. Quality masks are **index**
arrays under `data-quality/<prefix>_neurons_fluor_good.npy`. Per-session norm stats under
`meta/statistics/`. Cortex coords `meta/neurons/cell_motor_coordinates.npy` `(N,3)` (loaded but now
unused per decision 1). Full inventory in the legacy doc.

**Session-hash reading.** Session keys are `dynamic{animal}-{scan}-{idx}-Fluorescence-7b721b-v4a`;
`7b721b` is the pipeline hash (truncated to `7b7` in quality-mask filenames), `v4a` the version. The
leading `{animal}` groups the 10 sessions into 3 animals:

| Animal | Sessions | Note |
|---|---|---|
| `28188` (5) | `18-4`, `19-9`, `17-2`†, `16-5`†, `16-3`† | 3 of 5 have limited training data |
| `29163` (4) | `4-4`, `6-5`, `5-8`, `2-7`† | |
| `28712` (1) | `3-8` | |

† **Limited training data** (< 120 train trials): `28188-17-2`, `28188-16-5`, `28188-16-3`,
`29163-2-7`. Relevant to `LongCycler` balancing — these sessions contribute fewer clips.

**Per-session neuron counts after the `neurons_fluor_good` mask** (from the reference readout dims;
resolves the old "measure once all 10 downloaded" question; ~17.3k boutons total):

| Session | N | Session | N |
|---|---|---|---|
| `29163-4-4` | 3175 | `28188-19-9` | 787 |
| `28188-18-4` | 1593 | `28188-17-2` | 1714 |
| `28712-3-8` | 1728 | `29163-2-7` | 1327 |
| `29163-6-5` | 1079 | `28188-16-5` | 2046 |
| `29163-5-8` | 1113 | `28188-16-3` | 2710 (from 7636 pre-mask) |

**Test structure (natural `clip`).** There are **6 distinct test clips** of `stimulus_type="clip"`,
each repeated **15–20 times within a single session** (no cross-session pooling needed for repeats).
The reference identifies them by these 6 `condition_hash` values, shared across sessions:
`5zQTb77qI+ig8rigx1XU`, `7UETOWO5Z8aWuHDBJ2GG`, `GjCMo2GkJp6y5vricadg`, `KXdTNAGMo1gCWz2Ge8zr`,
`Oup5uAZxF2G7zEJkT+ui`, `ecUQJtcERZJGdqza1k7h`. So `test_conditions(..., stimulus_type="clip")` should
yield ~6 keys per session, each backed by 15–20 repeat trials.

---

## Remaining Steps

### Step 1 — Wire the shifter into `CoreReadout` (backward-compatible) — ✅ DONE (commit `8d03ffb`)

Implemented exactly as specified below; all 6 unit tests in `tests/models/test_core_readout_shifter.py`
pass, plus the full pre-existing `tests/models/` + `tests/modules/` suite (33 tests) with no regressions.
Kept below for reference on *how* it was done, not as a TODO.

The `MultiSessionMLPShifter` module exists but is never built or called. `BaseCoreReadout.forward`
(`core_readout.py:102`) takes only `(x, data_key)` and the train/val/test steps
(`:107/:124/:141`) never read a `pupil_center`.

**Do (all no-ops for existing datasets/models):**
1. **Build the shifter.** In `UnifiedCoreReadout.__init__`, if a `shifter` `DictConfig` is provided,
   `hydra.utils.instantiate(shifter, n_neurons_dict=n_neurons_dict)` so its keys match the readout's;
   store as `self.shifter`. Absent config → no `self.shifter`.
2. **`forward(self, x, data_key=None, pupil_center=None)`.** Run the core → `(B, C, T_out, H, W)`. If
   `self.shifter` and `pupil_center` are both present: **align pupil to the core-reduced time axis**
   (drop the leading `model_cut_frames = T_in − T_out` frames — the core is causal; for kernels
   `(11,5,5)` this is `18`, computed and stored in `data_info` at `core_readout.py:160-162`), rearrange
   `(B, T_out, 2) → (B·T_out, 2)`, `shift = self.shifter[data_key](pupil_aligned)`, and pass
   `shift=shift` into the readout. It flows `MultiSampledGaussianReadout.forward(**kwargs)` →
   `PointGaussianReadout.forward(..., shift=shift)` → `grid = grid + shift[:, None, None, :]`. If
   either is absent, behave exactly as today.
3. **Steps read pupil.** In `training/validation/test_step`,
   `pupil_center = getattr(data_point, "pupil_center", None)` then
   `forward(data_point.inputs, session_id, pupil_center=pupil_center)`. (Note `QiuDataPoint.pupil_center`
   is `(B, 2, T)` — transpose to `(B, T, 2)` before the time-axis alignment. The reference calls the
   model as `model(videos, data_key=..., pupil_center=(B, 2, T))`.)
4. **Regularizer.** Add `self.shifter.regularizer(session_id)` (scaled by `gamma_shifter`) to
   `total_loss` when the shifter exists. Reference uses `gamma_shifter=0`, so default it to `0`.

### Step 2 — Hydra configs — ✅ DONE (commit `8d03ffb`), with one correction

All four configs exist under `configs/`. **Correction to the spec below:** `core.hidden_padding` must be
`false`, not `true` — the `true` value below is wrong (contradicts this same plan's worked example,
`(64, 18, 46)` spatial output / `T_out = T_in − 18`, which only holds with no padding on any layer,
matching the reference's actual `padding=False`-everywhere spec). Verified empirically; see "Status as of
2026-07-22" above. The committed config correctly has `hidden_padding: false`.

**`configs/data_io/qiu_2026.yaml`** (mirror `karamanlis_2024`): `load_all_stimuli` + `load_all_responses`
targets with `base_data_path: ${paths.data_dir}`, `apply_quality_mask: true`,
`spike_inference: subtract_min` (until CASCADE lands), `data_info: {response_rate_hz: 30,
stimulus_rate_hz: 30, animal_species: mouse}`. **Must also expose the pupil dict** — the dataloader
takes a third `pupil_dictionary` arg, so add a `load_all_pupil` target and thread it through (see the
dataloader config below). `coordinate_dims` / cortex coords are no longer needed (decision 1).

**`configs/dataloader/qiu_2026.yaml`**:
```yaml
_target_: openretina.data_io.qiu_2026.dataloaders.qiu_2026_dataloaders
_convert_: object
batch_size: 32
train_chunk_size: 50          # strictly < clip_length keeps random-shift augmentation
clip_length: 300              # integral num_clips = train_time // clip_length
allow_over_boundaries: false  # confine chunks to trial boundaries
# num_val_clips only used as a fallback; the curated validation_clip_indices in session_kwargs win.
```
> **Resolved (commit `8d03ffb`):** `train.py` now builds a `dataloader_kwargs` dict and adds
> `pupil_dictionary = hydra.utils.call(cfg.data_io.pupil)` only `if "pupil" in cfg.data_io` — a no-op for
> every other dataset's `data_io` config, which has no `pupil` key.

**`configs/model/qiu_2026_core_readout.yaml`** (copy `configs/model/core_gaussian_readout.yaml`):
- `core`: `SimpleCoreWrapper`, `channels: [3, 64, 64, 64]`, `temporal_kernel_sizes: [11, 5, 5]`,
  `spatial_kernel_sizes: [11, 5, 5]`, `input_padding: false`, `hidden_padding: false` (**not** `true` —
  see the correction note above this section), `convolution_type: custom_separable`. Reference reg
  weights: `gamma_input_spatial=10`,
  `gamma_input_temporal=0.01` (map onto the template's `gamma_*` keys); BatchNorm `momentum=0.7`.
- `readout`: `MultiSampledGaussianReadout`, `gauss_type: full`, **`grid_mean_predictor: null`**
  (decision 1 — plain learned per-neuron μ, no cortex plumbing). Reference inits `init_mu_range=0.1`,
  `init_sigma=0.3`, `feature_reg_weight=1.0`. Keep the default `softplus` nonlinearity.
- `shifter` (**new key** — requires Step 1): `MultiSessionMLPShifter`, `input_channels: 2`,
  `hidden_channels: 5`, `n_layers: 3`; `n_neurons_dict` injected like the readout; `gamma_shifter: 0`.

**`configs/qiu_2026_core_readout.yaml`** (top-level — copy `karamanlis_2024_core_readout.yaml`):
compose the four configs above + `adamw` / `reduce_lr_on_plateau` / callbacks / loggers / trainer;
`exp_name: core_readout_qiu_2026_mouse`; `paths.data_dir` → the HF `franke_lab/qiu_2026` folder;
`model.in_shape: [3, 300, 36, 64]`. `n_neurons_dict` is injected by `train.py` from
`compute_data_info` — do **not** hardcode.

### Step 3 — Tests (`tests/data_io/test_qiu_2026.py`) — 🟡 partially done, this file itself not yet written

**Already covered elsewhere:** shifter/forward regression tests (Step 1's list item below) live in
`tests/models/test_core_readout_shifter.py` (6 tests, all passing). An offline synthetic end-to-end test
(`tests/data_io/test_qiu_2026_train_wiring.py`, 2 tests) drives the real dataloader + model configs
through one training step without real data — but it does not exercise the real loaders
(`load_*_for_session`) or the real FileTree data at all, so it would **not** have caught either of the
two bugs fixed in commit `923200c` (both only manifest with real multi-session data). The dedicated
suite below is still the gap; the real dataset is now cached locally (see "Status as of 2026-07-22"
above) so it can be written and run without a fresh download.

Use the cached `dynamic28188-16-3` via the `*_for_session` loaders (avoids the tens-of-GB full
download in CI). Cover:
- **Loaders:** `load_stimuli_for_session` → `MoviesTrainTestSplit`, `train.ndim == 4`, `C == 3`,
  per-session normalization applied (video channel roughly zero-mean/unit-std over train);
  `load_responses_for_session` → 2D `(N, T)`, NaN-free after trim+mask, `N == 2710` for `16-3`;
  `load_pupil_for_session` frame-aligned with responses (same `T_total`), normalized, NaN-free.
- **Container:** movie/response `test_dict` keys identical, `check_matching_stimulus` passes,
  `test_by_trial_dict` keys ⊆ `test_dict`, all train/val clips 300 frames. **For `stimulus_type="clip"`
  the test dict should have ~6 condition keys, each with 15–20 repeat trials** in `test_by_trial_dict`.
  **New (regression guard for the `923200c` fix):** assert none of the 6 condition hashes are the
  `presentmoviearray` ones (`chirp`/`moving_bar`) and that every repeat within a condition has identical
  shape — this is exactly what `np.stack` used to crash on before `read_stimulus_type` existed. Sessions
  `dynamic28188-19-9`, `28712-3-8`, `29163-4-4`, `29163-5-8`, `29163-6-5` are the ones that actually
  contain `presentmoviearray` trials (confirmed on real data); `16-3` itself has none, so a regression
  test for this specific bug needs one of those other sessions, not just `16-3`.
- **Dataloader:** a train batch is a `QiuDataPoint` with `inputs (b,3,t,h,w)`, `targets (b,t,n)`,
  `pupil_center (b,2,t)`; curated validation carve-out is disjoint from train.
- **Shifter/forward (after Step 1):** `CoreReadout.forward(x, data_key, pupil_center=...)` runs and
  shifts; with `pupil_center=None` still runs (regression guard for the unchanged contract). ✅ done,
  see `tests/models/test_core_readout_shifter.py`.
- **Regression:** a plain 2-field `DataPoint` batch from an existing loader still collates and trains.
  ✅ done, see `test_training_step_still_works_with_plain_datapoint` in the same file.
- **(Optional, slow/marked):** `load_all_*` against the folder returns 10 session keys.

### Step 4 — CASCADE spike inference (open decision)

`spike_inference="cascade"` currently raises. To resolve: which pretrained CASCADE model / retrain;
sampling-rate match (data 30 Hz); per-trial vs. concatenated-trace inference; NaN-boundary handling;
output smoothing. CASCADE rates are non-negative → pairs with the readout's `softplus` output +
`PoissonLoss3d`. Keep the `raw`/`subtract_min`/`cascade` switch so targets can be compared. Until this
lands, `subtract_min` is the training default.

---

## Success Criterion (Definition of Done)

**Not yet met — blocked on hardware, not code.** See "Status as of 2026-07-22" at the top for exactly
what's been tried and how far it got (33/300 training steps on 2 sessions before being stopped due to
this machine's RAM being insufficient). No checkpoint has been produced anywhere yet.

A full training epoch with non-NaN loss + a saved checkpoint:

```bash
uv run openretina train --config-name qiu_2026_core_readout trainer=debug
```

Then a checkpointed model predicts with the shifter active, for **all 10 session keys**:

```python
# stimulus_tensor: (batch, C=3, t, h, w); pupil_center: (batch, t, 2)
model.forward(stimulus_tensor, data_key="dynamic28188-16-3-Fluorescence-7b721b-v4a",
              pupil_center=pupil_center_tensor)
# → (batch, T_out, N_session); T_out = t − 18 (core temporal convs reduce time)
```

---

## Remaining Open Questions

1. **CASCADE recipe** — see Step 4.
2. **Hardware for a full run** — see "Status as of 2026-07-22" above. Needs a machine with substantially
   more than 34 GB RAM, ideally with a real GPU (the debug trainer forces `accelerator=cpu`, and actual
   training would want a GPU trainer config regardless of the memory question).
3. **`is_model_causal()` confirmation** — confirm the core passes the causality test. `model_cut_frames`
   is now known concretely (`18` for kernels `(11,5,5)`, confirmed empirically after the `hidden_padding`
   fix); Step 1's pupil alignment reads it from `data_info` dynamically regardless. Not yet actually run
   against the real qiu_2026 model — quick to check once a training run produces a model instance.
4. **Test-input averaging over repeats** — the loaders average the *input* over a condition's 15–20
   repeats: a no-op for the video but a real choice for the behavior channels (`stimuli.py:107-113`)
   and pupil (`pupil.py`), which differ per repeat. Averaging pairs them with trial-averaged responses;
   the alternative is per-repeat inputs with averaged predictions. `test_by_trial_dict` retains the
   per-repeat responses either way. Flagged `# TODO(qiu_2026)` at both sites.
5. **Push the branch** — `qiu_2026-integration` has 2 local commits (`8d03ffb`, `923200c`) not yet on
   `origin` as of this writing. Check `git log origin/qiu_2026-integration..HEAD` before assuming the
   remote is current.

### Resolved (kept for the record)

- ~~**`source_grid` normalization / routing.**~~ Dropped — decision 1 ignores cortex coordinates
  (`grid_mean_predictor: null`).
- ~~**Cross-session video-norm reconciliation.**~~ Resolved — decision 2 keeps per-session
  normalization; no shared global scalar.
- ~~**Per-session neuron counts after masking.**~~ Resolved — all 10 counts tabulated above.
- ~~**Pupil dict wiring into `train.py`.**~~ Resolved (commit `8d03ffb`) — conditional
  `"pupil" in cfg.data_io` check, no-op for every other dataset.
- ~~**Shifter wiring (Step 1) and Hydra configs (Step 2).**~~ Both done (commit `8d03ffb`).
- ~~**`hidden_padding` value in the model config.**~~ Was wrongly specified as `true` in this plan;
  corrected to `false`, verified empirically against the plan's own worked example. See "Status as of
  2026-07-22" above.
- ~~**Confirming the real qiu_2026 HF path.**~~ `https://huggingface.co/datasets/open-retina/open-retina/tree/main/franke_lab/qiu_2026`
  confirmed correct via `HfApi.list_repo_files`; all 10 sessions + quality masks present.
- ~~**Stimulus-type filter.**~~ Actually implemented now (commit `923200c`) — see "Status as of
  2026-07-22" above; previously claimed done in this plan but wasn't.
- ~~**Test-clip repeat count.**~~ Resolved — 6 clip conditions, 15–20 repeats **within** each session
  (earlier concern that a session tops out at 10 and needs cross-session pooling was wrong).
