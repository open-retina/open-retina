# Integration Plan: `qiu_2026` Dataset

> **⚠️ FROZEN / SUPERSEDED — archival only (snapshot as of 2026-07-08).**
> This is an old full-plan snapshot kept for its design-debate history. It **predates decisions D6
> (per-session normalization) and D7 (drop cortex coordinates)** and still describes the abandoned
> `grid_mean_predictor` / `source_grid` path, CASCADE as the default target, and ELU+1 output — all of
> which are wrong now. **Do not follow anything here as current guidance.** For the live view use:
> `qiu_2026_integration_plan.md` (status + next step), `qiu_2026_decisions.md` (settled decisions),
> `qiu_2026_reference.md` (architecture/data/session tables), `qiu_2026_history.md` (what happened).
> The still-valuable rationale from §0.1 (native loaders vs. sensorium) and §0.4 (why the shared
> `DataPoint` must not change) has been harvested into `qiu_2026_decisions.md` (D0, D1); the full
> per-stream data inventory below remains the most complete version and is why this file is kept.

## 0. Key architectural decisions (read first)

### 0.1 Native openretina loaders — **not** the notebook's `mouse_video_loader`

The reference notebook (`notebooks/data-loader-SC.ipynb`) loads this data with
`sensorium.datasets.mouse_video_loaders.mouse_video_loader`, pulled in via
`sys.path.append('/mnt/scratch09/yongrong/.../sensorium_2023' | 'neuralpredictors' | 'nnfabrik')`.
We will **not** reuse that loader. We write native openretina loaders that produce
`MoviesTrainTestSplit` / `ResponsesTrainTestSplit` and feed openretina's own dataloader stack.

| | openretina (what the model consumes) | notebook `mouse_video_loader` |
|---|---|---|
| Batch object | `DataPoint(inputs, targets)` namedtuple (`base_dataloader.py:17`) | 4-field NamedTuple `(videos, responses, behavior, pupil_center)` |
| Response axes | `targets` = `(batch, time, neurons)` | `(batch, neurons, time)` — transposed |
| Input channels | `(b, C, t, h, w)` | `(b, 3, t, h, w)` — behavior folded in as channels |
| Dict nesting | `dict[split][session]` | `dict[split][session]` (same) |
| Consumed by | `LongCycler` → `training_step` → `forward(x, data_key)` → `compute_data_info` | sensorium/nnfabrik `make_video_model` stack |

Reasons native wins:
- **Dependencies.** `sensorium` and `nnfabrik` are *not* openretina dependencies (only `neuralpredictors` is, and only as an optional `devmodels` extra). The notebook uses local scratch checkouts on one machine — not reproducible/distributable. openretina's `cyclers.py` is explicitly *"Adapted from sinzlab/neuralpredictors"* — the stack was deliberately re-implemented to avoid this dependency.
- **You would need an adapter anyway.** Even wrapping `mouse_video_loader` requires transposing responses `(b,n,t)→(b,t,n)`, dropping/rerouting behavior + pupil, re-boxing into `DataPoint`, and it bypasses `compute_data_info` (so `train.py`'s auto-injection of `n_neurons_dict`/norm-stats breaks).
- **The shifter buys nothing from the sensorium loader.** Its shifter integration lives in sensorium's `make_video_model`, which openretina does not use. openretina's model side must gain shifter support regardless of the data source (see §0.3), so the loader choice is orthogonal to the shifter.
- **The hard readout piece already exists in openretina** (see §0.3). Native loaders plug straight into `multiple_movies_dataloaders` → `LongCycler` → training → scoring → MEIs → HF caching → Hydra → tests.

### 0.2 Target model architecture (from the Qiu 2026 methods)

Dynamic model = **3D factorized convolutional core** (Höfling 2024) + **Gaussian readout** (Lurz 2020b)
with a **`grid_mean_predictor`** (Bashiri 2021) + **shifter network** (Sinz 2018), ELU+1 positive output.

- **Core:** three sequential (spatial conv 64ch + temporal conv 64ch + BatchNorm + ELU) blocks.
  First spatial kernel `1×11×11`, other two `1×5×5`. First temporal kernel `11×1×1`, other two `5×1×1`.
- **Readout:** generalized linear regression with learnable weight tensor `(c×w×h)` and **ELU+1** output
  (enforce positive response). Gaussian readout `N(μ, Σ)` per bouton; feature weights of size `c` per bouton.
- **`grid_mean_predictor`:** for the variant with recorded 2D cortex coordinates, an **MLP `2→30→2`
  shared per session** maps each bouton's cortex coordinate to its readout mean location `μ`.
- **Shifter:** to account for eye movements, an **MLP with 3 fully-connected layers, 5 hidden features,
  `tanh`** maps `pupil_center` → a shift added to the RF center `μ`. Shared per session.

### 0.3 What openretina already has vs. what must be built

| Piece | Status | Location |
|---|---|---|
| 3D factorized core | **Exists** (config-level) | `SimpleCoreWrapper` (`configs/model/*`) |
| Gaussian readout with `grid_mean_predictor` | **Exists** | `PointGaussianReadout` (`gaussian.py:15`, `init_grid_predictor` `:262`), multi-session `MultiSampledGaussianReadout` (`multi_readout.py:263`), template config `configs/model/core_gaussian_readout.yaml` |
| Readout `shift` argument | **Exists** | `PointGaussianReadout.forward(x, sample, shift, ...)` `gaussian.py:362`; `grid = grid + shift[:, None, None, :]` `:404-405`; `MultiSampledGaussianReadout.forward` passes `**kwargs` through `:309-326` |
| Positive output nonlinearity | **Exists (default)** | `MultiSampledGaussianReadout` default `softplus` (`multi_readout.py:286`) already enforces λ>0 for CASCADE (non-negative) targets + `PoissonLoss3d`. ELU+1 (paper) is a faithfulness-only toggle, deferred (see OQ4). |
| Behavior as extra input channels | **Loader work** | core takes arbitrary `in_shape[0]`; fold behavior in the loader |
| **Shifter network module** | **BUILT** ✅ | `openretina/modules/shifters/mlp_shifter.py`: `MLPShifter` (2→5→5→2, tanh, xavier init) + per-session `MultiSessionMLPShifter(nn.ModuleDict)` keyed by `n_neurons_dict`; unit-smoke-tested. |
| **pupil_center plumbing** to the readout | **MISSING — build** | `DataPoint` is 2-field; `forward` never passes `shift`; needs qiu-local datapoint + guarded forward |
| **Per-session `source_grid`** (cortex coords → readout) | **MISSING — build** | `MultiReadoutBase.__init__` passes the *same* `**kwargs` to all sessions (`multi_readout.py:71-83`); `core_readout.py:350-354` forwards only `n_neurons_dict` + `mean_activity_dict`. Cortex coords currently cannot reach the readout. |

### 0.4 Do **not** modify the shared `DataPoint` (verified breakage)

`DataPoint = namedtuple("DataPoint", ["inputs", "targets"])` (`base_dataloader.py:17`), constructed only in
`MovieDataSet.__getitem__` (`:84`, `:87`). Extending it to a 3rd field breaks existing datasets:

- **Collation crash (hard, immediate).** A defaulted `pupil=None` 3rd field makes every existing dataset
  emit a namedtuple with a `None` field; PyTorch `default_collate` raises
  `TypeError: default_collate: batch must contain tensors... found NoneType` on the **first batch**
  (verified empirically; 2-field and 3-field-with-tensor both collate fine — only `None` crashes).
  This would hit hoefling_2024, karamanlis_2024, maheswaranathan_2023, etc.
- **Positional-unpack sites.** `sparse_autoencoder.py:103` `x, _ = batch` (→ too many values) and
  `eval/metrics.py:42` `for *inputs, responses in loader` (→ appended field silently misread as responses).
- **Safe consumers.** All LightningModule steps (`core_readout`, `linear_nonlinear`, `spatial_contrast`) and
  `frame_fingerprints` use *attribute* access (`.inputs`/`.targets`), so they are indifferent to extra fields.

**Design (confirmed):** keep the shared contract unchanged.
- **Behavior needs no plumbing** — it folds into `inputs` channels (`C = 1 + n_behavior`) and rides the
  existing 2-field `DataPoint`.
- **pupil_center** is qiu-local: a `QiuMovieDataSet(MovieDataSet)` returning its own
  `QiuDataPoint(inputs, targets, pupil_center)` (3 real tensors → collates fine); a shifter-aware forward
  guarded by `getattr(data_point, "pupil_center", None)` (a genuine no-op for every existing dataset).

### 0.5 Behavior channel selection & response units (decided)

- **Behavior channels: `[0, 2]` only** = pupil size + locomotion (drop `Δpupil`, channel 1), matching the
  reference notebook → **`C = 3` input channels** (1 video + 2 behavior).
- **Responses → CASCADE inferred spike rates.** We will convert raw fluorescence to inferred spike rates
  with **CASCADE** (Rupprecht et al. 2021). This yields non-negative rates, consistent with the paper's
  ELU+1 positive output and a Poisson-style loss. *(Details TBD — to be fleshed out; see Open Questions.)*

---

## Scope

Integrate **all 10 sessions across 3 animals** (`dynamic28188-{16-3,16-5,17-2,18-4,19-9}`,
`dynamic28712-3-8`, `dynamic29163-{2-7,4-4,5-8,6-5}`, suffix `-Fluorescence-7b721b-v4a`) as a multi-session
dataset. For each session: strip NaN padding, apply the per-session `neurons_fluor_good` index mask,
fold behavior channels `[0,2]` into the video input, carry `pupil_center` for the shifter, run CASCADE on
responses, and contribute one entry to the session-keyed dicts. Point `paths.data_dir` at the
**`franke_lab/qiu_2026/` folder** (all 10 zips + `data-quality/`) — a tens-of-GB download.

**Trimming strategy is settled (verified against the data):**
- After removing trailing NaN padding, **all train and validation trials are uniformly 300 valid frames**
  → `clip_length = 300`, clean integral `num_clips = train_time // 300`.
- **Test conditions are internally uniform** (95 conds × 300 frames, 39 × 450); each `condition_hash` yields
  a clean dense `(repeats_i, N, T_cond)` at its native length — no padding/masking for the test split.
- **All four per-trial streams share the identical NaN-padding structure** (`videos`, `responses`,
  `behavior`, `pupil_center`), so the same trim applies uniformly.

## Success Criterion (Definition of Done)

Full training epoch with non-NaN loss + saved checkpoint:

```bash
uv run openretina train --config-name qiu_2026_core_readout trainer=debug
```

> `openretina` is the only registered console script; `train` is a subcommand, `--config-name` /
> `trainer=debug` forwarded to Hydra. There is **no** `openretina-train` executable.

Then a checkpointed model predicts without error, with the shifter active:

```python
# stimulus_tensor: (batch, channels=3, t, h, w); pupil_center: (batch, t, 2)
model.forward(stimulus_tensor, data_key="dynamic28188-16-3-Fluorescence-7b721b-v4a",
              pupil_center=pupil_center_tensor)
# returns (batch, T_out, N_session); T_out < t (core temporal convs reduce time)
```

Must pass for **all 10 session keys**, each with its own neuron count.

---

## Preliminary: Data Format Inventory

Raw data (`open-retina/open-retina` HuggingFace, `franke_lab/qiu_2026/`) is in **Sensorium FileTree trial
format**. Numbers below are for the cached session `dynamic28188-16-3`; per-trial shapes/conventions are
identical across sessions; counts (trials, neurons) vary.

| Property | Value |
|---|---|
| Sessions | 10 session zips across 3 animals under `franke_lab/qiu_2026/` (all `-Fluorescence-7b721b-v4a.zip`). |
| Quality masks | `franke_lab/qiu_2026/data-quality/<session>_neurons_fluor_good.npy` — an **INDEX array** (e.g. `(2710,)` int64), not boolean. Apply as fancy indexing. Filenames truncate the hash to `…-Fluorescence-7b7`. |
| Trials (this session) | 407 total: 100 train / 28 validation / 279 test (`meta/trials/tiers.npy`). |
| **`data/videos/{i}.npy`** | `(36, 64, 450)` → (H, W, T), single channel, float32 0–255. NaN-padded to 450 (valid 300 or 450). |
| **`data/responses/{i}.npy`** | `(7636, 450)` → (N, T) before masking (→ 2710 after `neurons_fluor_good`). Raw calcium fluorescence (~30% negative). NaN-padded. → **CASCADE** to spike rates. |
| **`data/behavior/{i}.npy`** | `(3, 450)` → (C_beh, T) = (pupil size, Δpupil, locomotion). **Use channels `[0,2]`.** Time-varying, spatially scalar. NaN-padded. |
| **`data/pupil_center/{i}.npy`** | `(2, 450)` → (x, y eye position, T). Raw pixel coords (~116–146). → **normalize, feed shifter.** NaN-padded. |
| Test structure | **134 unique conditions** (`condition_hash`), repeat counts 1–10. Ragged across conditions, uniform within each → per-condition dict. |
| Frame rate | `meta/trials/target_fps.npy` int64 all **30**. `FRAME_RATE_MODEL = 30`. |
| Cortex coordinates | `meta/neurons/cell_motor_coordinates.npy` `(7636, 3)` int64 (mask → `(2710,3)`); X∈[41,658], Y∈[-574,-306], Z∈[40,120]. **Used as `source_grid` for `grid_mean_predictor`** (2D → use X,Y). |
| Video norm stats | `meta/statistics/videos/all/{mean,std}.npy` `(36,64,450)` but constant (mean=80.939, std=62.797) — take `arr.flat[0]`. |
| Response norm stats | `meta/statistics/responses/all/{mean,std}.npy` `(7636,450)`, per-neuron broadcast. |
| **Behavior norm stats** | `meta/statistics/behavior/all/{mean,std}.npy` `(3,450)`, per-channel broadcast. |
| **Pupil norm stats** | `meta/statistics/pupil_center/all/{mean,std}.npy` `(2,450)`, per-channel broadcast — use to normalize pupil before the shifter. |

Key differences from existing datasets:
- Per-trial `.npy` files (not one continuous movie / H5).
- Videos `(H, W, T)` with no channel axis → reshape to `(C, T, H, W)`.
- NaN-padded trials (naive use → NaN loss).
- String-encoded metadata (`clip_cut_after`, timestamps, `monet2_fps`, …) as `"Decimal('10.000')"`, `'nan'`,
  `"Timestamp(...)"`, `'NaT'` → parse before numeric use.
- **Behavior + pupil_center streams** (absent from other integrations) → drive the extra input channels and
  the shifter.

---

## Step 1 — Create `openretina/data_io/qiu_2026/` module

Mirror **`karamanlis_2024`** multi-session structure (folder of per-session archives, a
`load_*_for_session()` worker + `load_all_*()` that lists/unzips the folder and returns session-keyed
dicts). Borrow `hoefling_2024`'s `session_kwargs` handling. **Unlike karamanlis, a real `dataloaders.py` is
needed** (to carry `pupil_center` — see §0.4).

**`constants.py`**
`FRAME_RATE_MODEL = 30`, `CLIP_LENGTH_FULL = 450`, `CLIP_LENGTH_CUT = 300`,
`VIDEO_SHAPE = (1, 36, 64)`, `BEHAVIOR_CHANNELS = (0, 2)` (pupil size + locomotion),
`N_INPUT_CHANNELS = 1 + len(BEHAVIOR_CHANNELS)` (= 3), `SESSION_ZIP_SUFFIX = "-Fluorescence-7b721b-v4a.zip"`.
Sessions discovered from folder listing (do not hard-code a single session).

**`stimuli.py`** — `load_all_stimuli(base_data_path, sessions=None, ...) -> dict[str, MoviesTrainTestSplit]`
+ `load_stimuli_for_session(session_path) -> MoviesTrainTestSplit`
- Mirror `karamanlis stimuli.py`: `get_local_file_path(str(base_data_path))` (downloads the whole
  `qiu_2026/` folder), `os.listdir`, keep entries ending in `SESSION_ZIP_SUFFIX` (or unzipped dirs),
  **skip `data-quality/`**. If path ends `.zip`, `unzip_and_cleanup`.
- Parse string-encoded metadata; read `tiers.npy` + `condition_hash.npy`.
- **Build the 3-channel input**: load video `data/videos/{i}.npy`, `rearrange("h w t -> 1 t h w")`; load
  `data/behavior/{i}.npy`, select channels `[0,2]`, normalize (per-channel behavior stats), **broadcast each
  behavior channel across H×W** and concatenate → `(C=3, T, H, W)`. **Trim to valid length** (300 train/val)
  before concatenating trials into the continuous train movie.
- Test movie `test_dict` keyed by `condition_hash` at native length (300/450, uniform within a condition).
  Keys must match the response `test_dict` exactly; all clips share `(C=3, H, W)`.
- **Cross-session normalization:** `compute_data_info` raises if per-session `norm_mean`/`norm_std` differ by
  `> atol=1`. Use a single **shared** video-normalization scalar across all sessions (e.g. the fixed
  `(80.939, 62.797)` or `/255`). Behavior channels are normalized *inside* the tensor (per-channel), so the
  `norm_mean`/`norm_std` passed to `MoviesTrainTestSplit` describe the shared video normalization.
- Return per session: `MoviesTrainTestSplit(train=..., test_dict=..., stim_id="clip", norm_mean=<shared>, norm_std=<shared>)`.

**`responses.py`** — `load_all_responses(base_data_path, apply_quality_mask=True, spike_inference="cascade", ...) -> dict[str, ResponsesTrainTestSplit]`
+ `load_responses_for_session(session_path, good_idx) -> ResponsesTrainTestSplit`
- Discover sessions like stimuli. For each, load `data-quality/<prefix>_neurons_fluor_good.npy` and pass to
  the worker (map session→mask by the truncated `…-Fluorescence-7b7` prefix).
- Load `data/responses/{i}.npy`; **apply `good_idx` first** (`r = r[good_idx]`, 7636→2710); the same
  `good_idx` indexes `cell_motor_coordinates`.
- **Trim to valid length** (300 train/val, native test). **Blocker:** any NaN left makes
  `get_movie_dataloader` return `None` (`base_dataloader.py:312-316`), later crashing `LongCycler`. Remove
  NaNs *before* building the container.
- **CASCADE spike inference (TBD):** convert raw fluorescence → inferred spike rates (non-negative). Exact
  model, sampling-rate matching (30 Hz), smoothing, and whether inference runs per-trial or on the
  concatenated trace are to be decided (see Open Questions). Placeholder: keep a `spike_inference` switch so
  raw/CASCADE can be compared.
- Concatenate trimmed train + validation into `(N_masked, T_total)`, aligned with the train movie `T_total`.
- Test: `test_by_trial_dict = {condition_hash: ndarray(repeats_i, N_masked, T_cond)}` — one array per
  condition. Do **not** use scalar `test_by_trial` (raises with multiple clips, `base.py:114-118`). Guard
  oracle/reliability against `repeats==1`.
- Load masked `cell_motor_coordinates` (cast float) into `session_kwargs` — **now consumed** by the readout
  `grid_mean_predictor` via new plumbing (§6).
- Return `ResponsesTrainTestSplit(train=..., test_dict=..., test_by_trial_dict=..., stim_id="clip", session_kwargs={"cell_motor_coordinates": ...})`.

**`pupil.py` (or fold into `responses.py`)** — load `data/pupil_center/{i}.npy` `(2, T)`, normalize with
`meta/statistics/pupil_center/all/{mean,std}`, trim to valid length, and split into train/val/test **using
the identical clip-splitting logic** applied to responses/movies so the pupil trace stays frame-aligned.
Returned as per-session arrays consumed by `dataloaders.py`.

**`dataloaders.py`** — a thin `qiu_2026_dataloaders(...)` that reuses the `multiple_movies_dataloaders`
machinery but injects `QiuMovieDataSet` (carrying `pupil_center`). Two implementation options:
  (a) add a `dataset_cls`/`extra_arrays` hook to `get_movie_dataloader` (`base_dataloader.py`) and reuse
      `multiple_movies_dataloaders`, or
  (b) a qiu-local copy of the split/dataloader loop that builds `QiuMovieDataSet`.
Prefer (a) (smallest surface, keeps a single split code-path). See §6.

**Container constraints** (`base.py`): movie/response `test_dict` keys identical; train times match; per-clip
test times match (`check_matching_stimulus`, `base.py:154-161`); matching `stim_id`; all clips share
`(C=3, H, W)` and a shared norm scalar.

---

## Step 2 — Hydra configs

**`configs/data_io/qiu_2026.yaml`** (mirror `karamanlis_2024`)
```yaml
stimuli:
  _target_: openretina.data_io.qiu_2026.stimuli.load_all_stimuli
  _convert_: object
  base_data_path: ${paths.data_dir}
responses:
  _target_: openretina.data_io.qiu_2026.responses.load_all_responses
  _convert_: object
  base_data_path: ${paths.data_dir}
  apply_quality_mask: true
  spike_inference: cascade   # TBD; raw | cascade
data_info:
  response_rate_hz: 30
  stimulus_rate_hz: 30
  animal_species: mouse
```

**`configs/dataloader/qiu_2026.yaml`** (copy `maheswaranathan_2023`, add pupil-carrying loader target)
```yaml
_target_: openretina.data_io.qiu_2026.dataloaders.qiu_2026_dataloaders  # carries pupil_center
_convert_: object
batch_size: 32
train_chunk_size: 50        # strictly < clip_length to retain random-shift augmentation
clip_length: 300            # integral num_clips = train_time // clip_length
num_val_clips: 10
allow_over_boundaries: false # confine chunks to trial boundaries
```
> `allow_over_boundaries: true` (default) lets chunks cross trial boundaries → must be `false`; then
> `train_chunk_size <= scene_length` is required and strictly-smaller keeps shift+shuffle active.

**`configs/model/qiu_2026_core_readout.yaml`** (copy `configs/model/core_gaussian_readout.yaml` — it already
uses `MultiSampledGaussianReadout` + `grid_mean_predictor`)
```yaml
in_shape: ???                       # injected top-level: [3, 450, 36, 64]
n_neurons_dict: ???                 # injected by train.py from compute_data_info — do not hardcode
core:
  _target_: openretina.modules.core.base_core.SimpleCoreWrapper
  _convert_: object
  # Paper: three 64-channel blocks; spatial kernels 11,5,5 ; temporal kernels 11,5,5.
  # VERIFIED: SimpleCoreWrapper takes `channels` (NOT `hidden_channels`); channels[0] is the
  # INPUT channel count, so [3, 64, 64, 64] == 3 input channels + three 64-ch blocks. The
  # kernel-size lists must each have len(channels)-1 == 3 entries (asserted in __init__).
  channels: [3, 64, 64, 64]
  temporal_kernel_sizes: [11, 5, 5]
  spatial_kernel_sizes: [11, 5, 5]
  input_padding: false              # first layer is valid-conv → reduces the time axis
  hidden_padding: true              # "same" → hidden layers preserve time
  convolution_type: custom_separable
  # gamma_input/gamma_temporal/gamma_in_sparse/gamma_hidden are REQUIRED (no defaults) —
  # copy the values from configs/model/core_gaussian_readout.yaml.
  gamma_input: ???
  gamma_temporal: ???
  gamma_in_sparse: ???
  gamma_hidden: ???
readout:
  _target_: openretina.modules.readout.multi_readout.MultiSampledGaussianReadout
  _convert_: object
  in_shape: ???
  bias: true
  gauss_type: full
  init_mu_range: 0.1
  init_sigma_range: 0.15
  grid_mean_predictor:             # cortex-coords → μ MLP (Bashiri 2021), paper: 2→30→2
    input_dimensions: 2
    hidden_layers: 1
    hidden_features: 30
    final_tanh: true
  # nonlinearity_function: keep the DEFAULT softplus. CASCADE targets are non-negative and the
  #   default loss is PoissonLoss3d (rate λ must be > 0), so softplus already enforces the
  #   required positivity. Paper uses ELU+1; that is a faithfulness-only toggle (see §5 / OQ4).
shifter:                            # NEW — BUILT in openretina/modules/shifters/ (see §5)
  _target_: openretina.modules.shifters.MultiSessionMLPShifter
  _convert_: object
  input_channels: 2
  hidden_channels: 5
  n_layers: 3
  # n_neurons_dict injected like the readout so shifter keys match session keys (see §7)
```

**`configs/qiu_2026_core_readout.yaml`** (top-level — copy `karamanlis_2024_core_readout.yaml`)
```yaml
defaults:
  - data_io: qiu_2026
  - dataloader: qiu_2026
  - model: qiu_2026_core_readout
  - optimizer: adamw
  - lr_scheduler: reduce_lr_on_plateau
  - training_callbacks: [early_stopping, lr_monitor, model_checkpoint]
  - logger: [tensorboard, csv]
  - trainer: default_deterministic
  - hydra: default
  - _self_

exp_name: core_readout_qiu_2026_mouse
seed: 42
check_stimuli_responses_match: true

paths:
  cache_dir: ${oc.env:OPENRETINA_CACHE_DIRECTORY}
  data_dir: "https://huggingface.co/datasets/open-retina/open-retina/tree/main/franke_lab/qiu_2026"
  log_dir: "."
  output_dir: ${hydra:runtime.output_dir}
  load_model_path: null

model:
  in_shape: [3, 450, 36, 64]   # (channels=3 = 1 video + 2 behavior, model time-window, H=36, W=64)
```
> `n_neurons_dict` is injected by `train.py:100` from `compute_data_info` — do not hardcode.
> `in_shape[0]` **must be 3** (video + 2 behavior channels).

---

## Step 3 — Fix the `IsADirectoryError` in the demo notebook

`movies_from_pickle()` (`hoefling_2024/stimuli.py`) called in `notebooks/openretina_demo_qiu.ipynb` fails
because the zip extracts to a Sensorium directory, not a pickle. The notebook downloads a single session zip
→ update the cell to call `qiu_2026.stimuli.load_stimuli_for_session(extracted_session_path)` (per-session
loader), not `load_all_stimuli` (which expects the parent folder).

---

## Step 4 — Readout mean location via `grid_mean_predictor` (cortex coordinates)

**Supersedes the earlier "learn μ from scratch" conclusion.** The paper maps recorded 2D cortex coordinates
→ readout mean `μ` with a per-session `2→30→2` MLP (Bashiri 2021). openretina's `PointGaussianReadout`
already supports this (`gaussian.py:262 init_grid_predictor(source_grid, hidden_features, hidden_layers,
final_tanh)`), exposed through `MultiSampledGaussianReadout(grid_mean_predictor=...)`.

**Required new plumbing (per-session `source_grid`):** `MultiReadoutBase.__init__` passes the *same*
`**kwargs` to every session (`multi_readout.py:71-83`), and `core_readout.py:350-354` forwards only
`n_neurons_dict` + `mean_activity_dict`. So cortex coordinates cannot currently reach the readout. Add:
- `MultiSampledGaussianReadout` (or `MultiReadoutBase`): accept a per-session `source_grids: dict[str, np.ndarray]`
  and pass `source_grid=source_grids[data_key]` into each per-session `PointGaussianReadout`.
- `core_readout.py`: extract `cell_motor_coordinates` from `data_info["sessions_kwargs"][session]` (X,Y only,
  normalized to the readout grid range) and pass as `source_grids` to the readout constructor.

**Fallback / staging:** `grid_mean_predictor: null` gives the plain learned-μ Gaussian readout (zero extra
plumbing) — usable as a first milestone before wiring per-session `source_grids`.

---

## Step 5 — Shifter network (BUILT ✅)

**Done** — `openretina/modules/shifters/mlp_shifter.py`. `neuralpredictors` is **not** an installed dependency
(only referenced conceptually), so the shifter was **ported** (reimplemented), not imported. Reference:
`neuralpredictors/layers/shifters/mlp.py` (Sinz 2018), adapted to open-retina's `n_neurons_dict`-keyed
multi-session convention (mirrors `MultiReadoutBase`).

**`openretina/modules/shifters/mlp_shifter.py`** (mypy/ruff clean, smoke-tested):
- `MLPShifter(input_channels=2, hidden_channels=5, n_layers=3, bias=True)`: MLP mapping `pupil_center (N, 2)` →
  `shift (N, 2)`, `n_layers` FC layers each + `tanh` (paper: 3 layers, 5 hidden). Final `tanh` bounds the
  shift to `[-1, 1]` (readout grid units). Xavier-normal init; `regularizer()` → 0.
- `MultiSessionMLPShifter(nn.ModuleDict)`: one `MLPShifter` per session key (shared across boutons within a
  session), keyed by `n_neurons_dict` like the readout; `forward(pupil_center, data_key)`,
  `regularizer(data_key)` scaled by `gamma_shifter`, typed `__getitem__` for mypy.

**Model integration (`CoreReadout`, backward-compatible):**
- `__init__`: if a `shifter` config is provided, build `self.shifter` keyed by `n_neurons_dict`.
- `forward(self, x, data_key=None, pupil_center=None)`: run core → `(B, C, T_out, H, W)`. If `self.shifter`
  and `pupil_center` are present, **align pupil to the core-reduced time axis** (drop the first
  `model_cut_frames = T_in - T_out` frames, causal core), rearrange `(B, T_out, 2) → (B·T_out, 2)`,
  `shift = self.shifter[data_key](pupil_aligned)`, and pass `shift=shift` into the readout (flows through
  `MultiSampledGaussianReadout.forward(**kwargs)` → `PointGaussianReadout.forward(..., shift=shift)` →
  `grid = grid + shift[:, None, None, :]`). If absent, behaves exactly as today (**no-op for all existing
  datasets/models**).
- `training/validation/test_step`: `pupil_center = getattr(data_point, "pupil_center", None)` then
  `forward(data_point.inputs, session_id, pupil_center=pupil_center)`.

---

## Step 6 — Model plumbing for `pupil_center` (NEW; no shared-contract change)

- **`QiuDataPoint = namedtuple("QiuDataPoint", ["inputs", "targets", "pupil_center"])`** (qiu-local; all
  fields real tensors → collates fine).
- **`QiuMovieDataSet(MovieDataSet)`**: override `__getitem__` to also slice `pupil_center` with the same
  chunking as `inputs`/`targets` and return `QiuDataPoint`. Time axis of `pupil_center` is the movie/response
  time axis (pre-core); alignment to core output happens in `forward` (§5).
- **`get_movie_dataloader` hook** (`base_dataloader.py`): add optional `dataset_cls=MovieDataSet` and an
  `extra_arrays: dict[str, np.ndarray] | None = None` (e.g. `{"pupil_center": ...}`) so
  `qiu_2026_dataloaders` can build `QiuMovieDataSet` while every other caller is untouched (defaults reproduce
  current behavior).
- **Verify collation**: PyTorch `default_collate` preserves namedtuple type and batches all three tensor
  fields (confirmed for tensor fields). No custom `collate_fn` required.

---

## Step 7 — Tests

Add `tests/data_io/test_qiu_2026.py` (use the cached `dynamic28188-16-3` via `load_*_for_session` to avoid
the tens-of-GB full download in CI):
- **Smoke:** `load_stimuli_for_session` returns `MoviesTrainTestSplit` with `train.ndim==4` and **C==3**
  (video + 2 behavior); `load_responses_for_session` returns 2D `(N, T)` with **no NaNs** after trimming and
  masking (7636→2710). Behavior channels are the `[0,2]` selection.
- **pupil:** per-session pupil arrays are frame-aligned with responses (same T_total), normalized, NaN-free.
- **Container:** movie/response `test_dict` keys identical, `check_matching_stimulus` passes;
  `test_by_trial_dict` keys ⊆ `test_dict`; train/val clips all 300 frames.
- **Dataloader:** `qiu_2026_dataloaders` on single-session dicts yields `dl[split][session]`; a train batch
  is a `QiuDataPoint` with `inputs (b,3,t,h,w)`, `targets (b,t,n)`, `pupil_center (b,t,2)`.
- **Shifter/forward:** `CoreReadout.forward(x, data_key, pupil_center=...)` runs and shifts; and with
  `pupil_center=None` (existing-dataset path) still runs (no-op) — regression guard for §0.4.
- **Regression:** a 2-field `DataPoint` batch from an existing loader still collates and trains (guards
  against accidental shared-contract changes).
- **(Optional, slow/marked)** `load_all_*` against the folder returns 10 session keys.

---

## Open Questions

1. **CASCADE spike inference (to flesh out):** which pretrained CASCADE model / retrain?; sampling-rate match
   (data 30 Hz); per-trial vs. concatenated-trace inference; handling of the NaN-padding boundaries; output
   smoothing; whether to keep a `raw` vs. `cascade` switch for comparison. Loss/nonlinearity pairing:
   CASCADE rates are non-negative → ELU+1 output + Poisson-style loss (paper).
2. **Core architecture match:** ~~how to map three 64-ch blocks onto `SimpleCoreWrapper`~~ **RESOLVED** —
   `channels: [3, 64, 64, 64]`, `temporal_kernel_sizes: [11, 5, 5]`, `spatial_kernel_sizes: [11, 5, 5]`,
   `input_padding: false`, `hidden_padding: true` (see Step 2). Still TODO: confirm `is_model_causal()` passes
   and read off the concrete `model_cut_frames` (the model computes & stores it in `data_info`,
   `core_readout.py:161`) so §5 pupil alignment uses it dynamically rather than hardcoding.
3. **`grid_mean_predictor` source grid:** normalize cortex X,Y to the readout grid range; decide whether Z is
   used (paper: 2D). Confirm the per-session `source_grids` plumbing (§4) and `final_tanh`/hidden sizes.
4. **ELU+1 nonlinearity:** **RESOLVED (deferred).** Not needed for a working model — the readout's default
   `softplus` already gives a strictly-positive Poisson rate, which is exactly what CASCADE (non-negative
   targets) + `PoissonLoss3d` require. ELU+1 only matters for bit-exact paper reproduction; if wanted later,
   add a named `ELU1(nn.Module)` to `openretina/modules/nonlinearities.py` (which already has
   `ParametrizedELU` — note its `device="cuda"` default breaks on CPU, so write a fresh device-agnostic one).
5. **Cross-session video-norm reconciliation:** confirm the 10 sessions' shipped video stats agree within
   `atol=1`; otherwise force a single shared scalar (recommended regardless).
6. **Per-session neuron counts after masking** (only `dynamic28188-16-3` = 7636→2710 measured) — validate
   once all 10 are downloaded.
7. **`train_chunk_size` / `num_val_clips` tuning** — functional (not correctness) choices; correctness
   constraints (`train_chunk_size < clip_length=300`, integral `num_clips`) are fixed.
8. **Test-set input averaging over repeats (TO RESOLVE):** for the test split the loaders average the
   *input* over a condition's repeats — a no-op for the video (identical across repeats) but a real choice
   for the **behavior** channels (`stimuli.py`) and the **pupil** trace (`pupil.py`), which genuinely differ
   per repeat. Averaging pairs them with the trial-averaged responses. Flagged with `# TODO(qiu_2026)` at
   both sites. Alternative: keep per-repeat inputs and average model predictions. Decide before final eval.
