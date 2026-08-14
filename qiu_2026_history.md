# `qiu_2026` — Engineering History

Chronological journal of what was built and what broke, in order. Read this to understand *how* a
piece came to be or *why* a past attempt failed; read `qiu_2026_integration_plan.md` for current
status and the next step, and `qiu_2026_decisions.md` for the settled decisions.

---

## Steps 1 & 2 — shifter wiring + Hydra configs (commit `8d03ffb`)

Both committed on `qiu_2026-integration`, with unit tests (`tests/models/test_core_readout_shifter.py`,
6 tests) and an offline synthetic end-to-end test (`tests/data_io/test_qiu_2026_train_wiring.py`, 2
tests) that drives the real configs through one training step without needing the real dataset. All 6
shifter unit tests pass, plus the full pre-existing `tests/models/` + `tests/modules/` suite (33 tests)
with no regressions.

### How Step 1 was done — wire the shifter into `CoreReadout` (backward-compatible)

The `MultiSessionMLPShifter` module already existed (`modules/shifters/mlp_shifter.py`) but was never
built or called. `BaseCoreReadout.forward` took only `(x, data_key)` and the train/val/test steps
never read a `pupil_center`. Implemented (all no-ops for existing datasets/models):

1. **Build the shifter.** In `UnifiedCoreReadout.__init__`, if a `shifter` `DictConfig` is provided,
   `hydra.utils.instantiate(shifter, n_neurons_dict=n_neurons_dict)` so its keys match the readout's;
   stored as `self.shifter`. Absent config → no `self.shifter`.
2. **`forward(self, x, data_key=None, pupil_center=None)`.** Run the core → `(B, C, T_out, H, W)`. If
   `self.shifter` and `pupil_center` are both present: **align pupil to the core-reduced time axis**
   (drop the leading `model_cut_frames = T_in − T_out` frames — the core is causal; for kernels
   `(11,5,5)` this is `18`, computed and stored in `data_info` at `core_readout.py:160-162`), rearrange
   `(B, T_out, 2) → (B·T_out, 2)`, `shift = self.shifter[data_key](pupil_aligned)`, and pass
   `shift=shift` into the readout. It flows `MultiSampledGaussianReadout.forward(**kwargs)` →
   `PointGaussianReadout.forward(..., shift=shift)` → `grid = grid + shift[:, None, None, :]`. If
   either is absent, behaves exactly as before.
3. **Steps read pupil.** In `training/validation/test_step`,
   `pupil_center = getattr(data_point, "pupil_center", None)` then
   `forward(data_point.inputs, session_id, pupil_center=pupil_center)`. (`QiuDataPoint.pupil_center` is
   `(B, 2, T)` — transposed to `(B, T, 2)` before the time-axis alignment. The reference calls the
   model as `model(videos, data_key=..., pupil_center=(B, 2, T))`.)
4. **Regularizer.** `self.shifter.regularizer(session_id)` (scaled by `gamma_shifter`) is added to
   `total_loss` when the shifter exists. Reference uses `gamma_shifter=0`, so it defaults to `0`.

### How Step 2 was done — Hydra configs

All four configs exist under `configs/`:
- `configs/data_io/qiu_2026.yaml` — `load_all_stimuli` + `load_all_responses` targets with
  `base_data_path: ${paths.data_dir}`, `apply_quality_mask: true`, `spike_inference: subtract_min`,
  `data_info: {response_rate_hz: 30, stimulus_rate_hz: 30, animal_species: mouse}`, plus a
  `load_all_pupil` target for the third `pupil_dictionary` dataloader arg.
- `configs/dataloader/qiu_2026.yaml` — `qiu_2026_dataloaders`, `batch_size: 32`,
  `train_chunk_size: 50` (strictly `< clip_length` keeps random-shift augmentation), `clip_length:
  300`, `allow_over_boundaries: false`. `num_val_clips` is only a fallback; the curated
  `validation_clip_indices` in `session_kwargs` win.
- `configs/model/qiu_2026_core_readout.yaml` — `SimpleCoreWrapper`, `channels: [3, 64, 64, 64]`,
  `temporal_kernel_sizes: [11, 5, 5]`, `spatial_kernel_sizes: [11, 5, 5]`, `input_padding: false`,
  **`hidden_padding: false`** (see decision D8), `convolution_type: custom_separable`,
  `gamma_input_spatial=10`, `gamma_input_temporal=0.01`, BatchNorm `momentum=0.7`; readout
  `MultiSampledGaussianReadout`, `gauss_type: full`, **`grid_mean_predictor: null`**,
  `init_mu_range=0.1`, `init_sigma=0.3`, `feature_reg_weight=1.0`, default `softplus`; shifter
  `MultiSessionMLPShifter`, `input_channels: 2`, `hidden_channels: 5`, `n_layers: 3`, `gamma_shifter:
  0`.
- `configs/qiu_2026_core_readout.yaml` — top-level, composes the four + `adamw` /
  `reduce_lr_on_plateau` / callbacks / loggers / trainer; `exp_name: core_readout_qiu_2026_mouse`;
  `model.in_shape: [3, 300, 36, 64]`. `n_neurons_dict` is injected by `train.py` from
  `compute_data_info` — not hardcoded.

`train.py` now builds a `dataloader_kwargs` dict and adds
`pupil_dictionary = hydra.utils.call(cfg.data_io.pupil)` only `if "pupil" in cfg.data_io` — a no-op for
every other dataset's `data_io` config, which has no `pupil` key.

Found and fixed **within Step 2** (before running on real data): the model config's
`core.hidden_padding` was set to `true` per an intermediate (incorrect) spec, but that contradicts the
worked example and the reference's `padding=False`-everywhere spec. Corrected to `false` and verified
empirically. See decision D8.

---

## 2026-07-22 — two bugs found only by running on real data (commit `923200c`)

Both are **real-data-only bugs**, invisible to any single-session or synthetic test, found by running
the real multi-session dataset through the pipeline for the first time. Both fixed.

1. **Stimulus-type filtering was never actually implemented**, despite an earlier status table
   claiming it was done. Some sessions' `test` tier mixes in `presentmoviearray` trials
   (`chirp`/`moving_bar` functional-characterization stimuli) alongside real clips; these have
   per-repeat frame-count jitter (959 vs 960 frames) that crashes `np.stack` when building
   test-condition repeats. Fixed: `trials.read_stimulus_type()` (keys off `clip_movie_name` being
   populated) now filters `train_val_indices` / `validation_clip_indices` / `test_conditions`,
   threaded through all three loaders with `stimulus_type: str = "clip"` defaulting to the old
   (intended) behavior. Train/validation tiers were empirically confirmed clip-only in every session
   (0 contamination), so this only affects test-set construction.
2. **`compute_data_info` hard-crashed** on qiu_2026's intentional per-session video normalization
   (each session z-scored with its own stats, no shared global scalar — decision D6) because it
   asserted all sessions share one `norm_mean`/`norm_std`. That value is only consumed by the optional
   `insilico/vector_field_analysis` tool, not training, so the mismatch case now warns (matching an
   existing warn-not-crash precedent a few lines above in the same function) instead of raising.

---

## 2026-07-22 — hardware blocker: the 34 GB machine cannot run this

The local machine has **34.4 GB RAM**. Real-data attempts, in order:

- **All 10 sessions** (`load_all_*`, default `batch_size=32`): movie+response arrays alone estimate to
  ~35 GB resident simultaneously before any per-batch activation memory — SIGKILL (OOM) during Sanity
  Checking, the very first real batch.
- **5 largest sessions**, default `batch_size=32`: same SIGKILL, same point — ruled out as a "too many
  sessions" issue since 2-session runs hit the identical point.
- **2 smallest sessions** (`dynamic28188-16-5`, `dynamic28188-16-3`, ~2.8 GB combined), default
  `batch_size=32`: still SIGKILL at the same point — confirms it's not about total resident data size
  but about **per-batch forward-pass memory** (a `(32, 3, 300, 36, 64)` validation batch through 3
  un-padded conv layers).
- **2 smallest sessions, `dataloader.batch_size=4`**: **no crash.** Got through model build, Sanity
  Checking (slowly — 8m48s for 2 validation batches) and 33/300 steps into training (epoch 0), at which
  point it was manually stopped (`SIGTERM`, not a crash) because system swap had filled to 17.99/18.4
  GB and the whole machine was severely degraded (swap thrashing). It was still progressing when
  stopped, not stuck.

**Conclusion:** the code is correct and now runs on real data; this specific machine just doesn't have
enough RAM to do it at a usable speed, even at 2 sessions / batch_size=4. No checkpoint has been
produced anywhere yet.

## 2026-07-22 (later) — three node hangs on the reduced command

The reduced 2-session / `batch_size=4` command was launched **three more times** and each time the
**compute node went unresponsive within ~15 s of launch** (had to recover the node between attempts).
Hydra still captured per-run logs under
`openretina_assets/runs/core_readout_qiu_2026_mouse/<timestamp>/openretina_main.log` (+ `.hydra/` with
the exact `overrides.yaml`). The latest is `2026-07-22_17-32-47/` (overrides confirm: `trainer=debug`,
the two smallest sessions, `dataloader.batch_size=4`).

**What that log shows (220 lines):**
- `17:32:48` full config logged; `17:32:48–50` **first discovery pass** finds all 10 quality masks + 10
  extracted folders; `17:33:00–02` an **identical second discovery pass** (the responses loader
  re-scanning).
- Log then **stops dead at `17:33:02,996`** — no third (pupil) discovery pass, no model build, no
  dataloader construction, no "Sanity Checking", no training step, **and no error / traceback / OOM /
  SIGKILL line.**

**Interpretation.** The process cleared the cheap *discovery* phase (file listing, ~0 memory) and froze
right at the boundary where the loaders begin **reading the movie/response `.npy` arrays into RAM** —
the first memory-heavy op. The *absence* of any flushed error is the tell: a clean Python OOM would be
SIGKILL'd and the shell would report it, but here the logger simply can't flush anymore — the signature
of the **node itself seizing** (swap thrash / I/O-bound), matching the earlier 34 GB-machine behavior.
So all evidence still points at the **RAM wall during array load**, not a code fault or a new
environmental change.

**Side observation (needs a code check).** Even with only 2 sessions requested, `load_all_*`
**scans/iterates all 10 sessions twice** during discovery. Discovery is cheap so it is *not* the hang —
but it must be confirmed that the loaders actually filter to the 2 requested sessions **before**
loading arrays into memory. If `load_all_*` eagerly loads all 10 sessions' arrays regardless of the
`sessions=[...]` filter, that alone would blow RAM and would be a real code bug, not just the hardware
wall. This is the first concrete thing to check — carried into the plan's next-step checklist.

**Update (2026-07-22, resolved):** checked and cleared. `discover_sessions` (`trials.py:147-163`)
applies the `sessions=[...]` filter at directory-listing time (`trials.py:156`), *before* any array is
read; each `load_all_*` reads `.npy` movie/response/behavior/pupil arrays only inside
`load_*_for_session`, which is called exclusively for the filtered paths. The "10 scanned twice" is
just the directory listing plus `discover_quality_masks` loading 10 tiny index arrays (tens of KB) —
**not** an eager array load. Confirmed **not** a code bug; the RAM wall is genuine. (Secondary lever
noted for later: the per-session `np.concatenate([...])` over a full list of trials holds ~2× peak
during load — `stimuli.py:102`, `responses.py:90`, `pupil.py:62`.)
