# `qiu_2026` — Decision Log

Settled decisions and their rationale. **Do not revisit these without a new reason** — they are
recorded here precisely so the entry-point plan (`qiu_2026_integration_plan.md`) can stay short.
Append new decisions at the bottom with a date. Architecture/data facts referenced below are
tabulated in `qiu_2026_reference.md`.

---

## D0 — Native openretina loaders, not the reference notebook's `sensorium.mouse_video_loader`

We write native openretina loaders that produce `MoviesTrainTestSplit` / `ResponsesTrainTestSplit`
and feed the standard `LongCycler` stack — **not** the reference notebook's
`sensorium.datasets.mouse_video_loaders.mouse_video_loader`.

**Why:**
- **Dependencies.** `sensorium` and `nnfabrik` are *not* openretina dependencies (only
  `neuralpredictors` is, and only as an optional `devmodels` extra). The notebook uses local scratch
  checkouts on one machine — not reproducible/distributable. openretina's `cyclers.py` is explicitly
  *"Adapted from sinzlab/neuralpredictors"* — the stack was deliberately re-implemented to avoid this
  dependency.
- **You would need an adapter anyway.** Even wrapping `mouse_video_loader` requires transposing
  responses `(b,n,t)→(b,t,n)`, dropping/rerouting behavior + pupil, re-boxing into `DataPoint`, and it
  bypasses `compute_data_info` (so `train.py`'s auto-injection of `n_neurons_dict`/norm-stats breaks).
- **The shifter buys nothing from the sensorium loader.** Its shifter integration lives in sensorium's
  `make_video_model`, which openretina does not use. openretina's model side must gain shifter support
  regardless of the data source, so the loader choice is orthogonal to the shifter.
- **The hard readout piece already exists in openretina.** Native loaders plug straight into
  `multiple_movies_dataloaders` → `LongCycler` → training → scoring → MEIs → HF caching → Hydra → tests.

## D1 — Do **not** modify the shared `DataPoint`; `pupil_center` is qiu-local

`DataPoint = namedtuple("DataPoint", ["inputs", "targets"])` (`base_dataloader.py:17`) stays
unchanged. Extending it to a 3rd field breaks existing datasets (verified):

- **Collation crash (hard, immediate).** A defaulted `pupil=None` 3rd field makes every existing
  dataset emit a namedtuple with a `None` field; PyTorch `default_collate` raises
  `TypeError: default_collate: batch must contain tensors... found NoneType` on the **first batch**
  (verified empirically; 2-field and 3-field-with-tensor both collate fine — only `None` crashes).
  This would hit hoefling_2024, karamanlis_2024, maheswaranathan_2023, etc.
- **Positional-unpack sites.** `sparse_autoencoder.py:103` `x, _ = batch` (→ too many values) and
  `eval/metrics.py:42` `for *inputs, responses in loader` (→ appended field silently misread).
- **Safe consumers.** All LightningModule steps and `frame_fingerprints` use *attribute* access
  (`.inputs`/`.targets`), so they are indifferent to extra fields.

**Design (confirmed):**
- **Behavior needs no plumbing** — it folds into `inputs` channels (`C = 3`) and rides the existing
  2-field `DataPoint`.
- **`pupil_center`** rides a qiu-local `QiuDataPoint(inputs, targets, pupil_center)` (3 real tensors
  → `default_collate` works); a shifter-aware forward guarded by
  `getattr(data_point, "pupil_center", None)` is a genuine no-op for every existing dataset.

## D2 — Behavior channels `[0, 2]` = pupil size + locomotion → `C = 3`

Δpupil (channel 1) is dropped, matching the reference notebook. One video channel + two behavior
channels = 3 input channels; each behavior channel is broadcast across H×W and concatenated to the
video.

## D3 — Stimulus filter defaults to `"clip"`

Natural movies (300 valid frames) only. `presentmoviearray` trials (`chirp`/`moving_bar`
functional-characterization stimuli, test tier only, ~959–960 valid frames with per-repeat jitter —
not `monet2`/`trippy` as earlier guessed, and not the same padded length as clips) are excluded via
`trials.read_stimulus_type()`. The same `stimulus_type` threads through all three loaders so streams
stay frame-aligned. Train/validation tiers were empirically confirmed clip-only in every session (0
contamination), so this only affects test-set construction. *(Actually implemented in commit
`923200c` — see `qiu_2026_history.md`; a prior version of the plan wrongly claimed it was done.)*

## D4 — Trimming

Trailing whole-frame NaN padding is stripped per trial before concatenation; train/val trials are a
uniform 300 frames, test conditions internally uniform at their native length. All four per-trial
streams (`videos`, `responses`, `behavior`, `pupil_center`) share the identical NaN-padding
structure, so the same trim applies uniformly.

## D5 — Responses → non-negative target via a `spike_inference` switch

`spike_inference` values: `"raw"` (has ~25% negatives, Poisson-incompatible),
`"subtract_min"` (current default, per-neuron train-min subtracted → non-negative interim target),
`"cascade"` (raises `NotImplementedError` — still an open decision; see the plan's Step 4). Keeping
the switch lets the target choices be compared. Until CASCADE lands, `subtract_min` is the default.

## D6 — Per-session movie normalization (2026-07-17)

Each session's video channel is z-scored with that session's own shipped
`meta/statistics/videos/all/{mean,std}.npy` (a single scalar), and each behavior channel with its own
`meta/statistics/behavior` stat (`stimuli.py:37-68`). This mirrors the reference
(`dat.statistics.videos.all.mean/std` per session). **No cross-session reconciliation and no single
shared global scalar** — the shared core sees per-session-standardized inputs by design. This closed
the old cross-session-video-norm reconciliation question.

## D7 — Drop `source_grid` / cortex coordinates (2026-07-17)

The Gaussian readout uses plain learned per-neuron means (`grid_mean_predictor: null`); cortex
coordinates are **not** consumed. This matches the reference's `v05aa` ("no cortex coordinates")
ablation and **removed the old "Step 1"** (source_grid → `grid_mean_predictor`). `responses.py` still
loads/masks `cell_motor_coordinates` into `session_kwargs["cell_motor_coordinates"]` (`:101-109`), but
**nothing consumes it** now; it is dead but harmless. `validation_clip_indices` in the same
`session_kwargs` **stays** (it drives the curated val split). *Optional cleanup:* drop the
`cell_motor_coordinates` entry and the `coordinate_dims` param from `responses.py`.

## D8 — `core.hidden_padding` must be `false` (2026-07-22)

An intermediate spec wrongly said `hidden_padding: true`. That contradicts the reference's actual
`padding=False`-everywhere spec and the worked example (`(64, 18, 46)` spatial output, `T_out =
T_in − 18`), which only holds with no padding on any layer. Verified empirically: `hidden_padding:
false` is what reproduces `(64, 18, 46)` for a `(3, 40, 36, 64)` input. The committed
`configs/model/qiu_2026_core_readout.yaml` correctly has `false`. **If any future edit sets it back to
`true`, that is the bug, not a fix.**

---

## Resolved questions (kept for the record)

- **`source_grid` normalization / routing** — dropped (D7).
- **Cross-session video-norm reconciliation** — resolved by per-session normalization (D6); no shared
  global scalar.
- **Per-session neuron counts after masking** — all 10 counts tabulated in `qiu_2026_reference.md`.
- **Pupil dict wiring into `train.py`** — resolved (commit `8d03ffb`): a conditional
  `"pupil" in cfg.data_io` check, a no-op for every other dataset.
- **Shifter wiring and Hydra configs** — both done (commit `8d03ffb`); see `qiu_2026_history.md`.
- **`hidden_padding` value** — corrected to `false` (D8), verified empirically.
- **Confirming the real qiu_2026 HF path** —
  `https://huggingface.co/datasets/open-retina/open-retina/tree/main/franke_lab/qiu_2026` confirmed
  correct via `HfApi.list_repo_files`; all 10 sessions + quality masks present.
- **Stimulus-type filter** — actually implemented (commit `923200c`, D3); previously claimed done but
  wasn't.
- **Test-clip repeat count** — 6 clip conditions, 15–20 repeats **within** each session (the earlier
  concern that a session tops out at 10 and needs cross-session pooling was wrong).
