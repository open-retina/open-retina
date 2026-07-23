# Integration Plan: `qiu_2026` Dataset

Integrate the Franke-lab Qiu 2026 mouse dataset (10 sessions / 3 animals) so open-retina can **train
and evaluate** on it.

**Status (2026-07-22):** Data-loading, model-side wiring (shifter), and Hydra configs are all
implemented and committed on `qiu_2026-integration` (commits `8d03ffb`, `923200c`). The code now runs
on the real multi-session dataset. **The only thing left is running a full training epoch on hardware
with enough RAM** — no checkpoint has been produced anywhere yet. The remaining code work (a real-data
test suite, and the CASCADE spike-inference decision) is optional/deferred, not blocking.

> ⚠️ The 2 commits above are on the **local** branch and may not be pushed —
> run `git log origin/qiu_2026-integration..HEAD` before assuming the remote has them.

## Where things live

| Doc | Contents |
|---|---|
| **`qiu_2026_integration_plan.md`** (this file) | Current status + the next step + remaining/open work |
| `qiu_2026_integration_plan_dataset_location.md` | **Run this first every session:** machine-identification checklist + how to set `paths.data_dir` in each case (cluster / laptop / naive machine) |
| `qiu_2026_reference.md` | Durable lookup: target architecture + hyperparameters, data format, session/animal/neuron/test tables, paper-methods quote |
| `qiu_2026_decisions.md` | Settled decisions (D0–D8) + rationale + resolved questions — **do not revisit** |
| `qiu_2026_history.md` | Chronological journal: how Steps 1 & 2 were built, the two real-data bugs, the hardware attempts, the node-hang forensics |
| `qiu_2026_integration_plan_legacy.md` | Frozen Jul-8 full-plan snapshot (superseded) — kept only for archival design-debate history |

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
| Shifter wired into `CoreReadout.forward` + train/val/test steps | ✅ Done | `core_readout.py` (commit `8d03ffb`) |
| Hydra configs (data_io / dataloader / model / top-level) | ✅ Done | `configs/**/qiu_2026*` (commit `8d03ffb`) |
| Stimulus-type filter | ✅ Done | `trials.py` + all loaders (commit `923200c`) |
| `compute_data_info` per-session norm-stat crash | ✅ Fixed (warn, not raise) | `data_io/base.py` (commit `923200c`) |
| Per-session `source_grid` → readout `grid_mean_predictor` | 🚫 Dropped | decision D7 — use `grid_mean_predictor: null` |
| **Run the DoD command to completion** | ❌ Blocked on hardware, not code | see **Next step** below |
| Real-session data_io test suite | 🟡 Open (Step A) | `tests/data_io/test_qiu_2026.py` (not yet written) |
| CASCADE spike inference (loaders default to `subtract_min`) | ❌ Open decision (Step B) | `responses.py` |
| Now-dead cortex-coord plumbing in `responses.py` | 🧹 Optional cleanup | `responses.py:101-109` |

---

## ▶ Next step — run a full training epoch on a bigger box

The code is correct and runs on real data; the 34 GB local machine simply cannot do a full run at
usable speed (details + the three node hangs in `qiu_2026_history.md`). The path forward:

1. **✅ Session filter verified correct (2026-07-22) — not a bug.** `discover_sessions` applies the
   `sessions=[...]` filter at directory-listing time (`trials.py:156`) *before* any array is read;
   each `load_all_*` calls `load_*_for_session` (the only place `.npy` movie/response/behavior/pupil
   arrays are read) exclusively for the filtered paths. The "all 10 sessions scanned twice" seen in the
   logs is just the cheap directory listing plus `discover_quality_masks` (`trials.py:166`) loading 10
   tiny `neurons_fluor_good` **index arrays** (tens of KB total) — **not** an eager load of all 10
   sessions' arrays. So this is genuinely the RAM / per-batch-forward-memory wall, not a filter bug.
2. **(Optional confirmation)** Compare the cutoff point of the four run logs under
   `openretina_assets/runs/core_readout_qiu_2026_mouse/` (`16-22-40`, `16-34-49`, `17-16-28`,
   `17-32-47`). If all die at the same post-discovery / pre-array-load boundary, that is strong
   confirmation it is purely the RAM wall.
3. **Get onto a bigger-RAM / real-GPU box and run the DoD command.** A real GPU also fixes the
   `accelerator=cpu` speed problem (`trainer=debug` forces CPU). First try the literal DoD command with
   no overrides (all 10 sessions, default batch size); fall back to the reduced command only if memory
   is still a problem.
4. **If a big box is unavailable**, reduce peak array-load memory. The per-session loaders build a
   Python list of every trimmed trial and *then* `np.concatenate` it (`stimuli.py:102`,
   `responses.py:90`, `pupil.py:62`), transiently holding both the list and the concatenated result
   (~2× peak). Preallocate-and-fill, or a lazy/memmap loader / per-trial streaming, would cut that. A
   real but secondary lever — the session filter (step 1) is already correct.

**Literal DoD command:**
```bash
uv run openretina train --config-name qiu_2026_core_readout trainer=debug
```

**Reduced 2-session / small-batch command (got furthest — 33/300 steps on the 34 GB box):**
```bash
OPENRETINA_CACHE_DIRECTORY=/Users/lhoefling/data uv run openretina train \
  --config-name qiu_2026_core_readout trainer=debug \
  paths.data_dir=/Users/lhoefling/data/franke_lab/qiu_2026 \
  '+data_io.stimuli.sessions=["dynamic28188-16-5-Fluorescence-7b721b-v4a","dynamic28188-16-3-Fluorescence-7b721b-v4a"]' \
  '+data_io.responses.sessions=["dynamic28188-16-5-Fluorescence-7b721b-v4a","dynamic28188-16-3-Fluorescence-7b721b-v4a"]' \
  '+data_io.pupil.sessions=["dynamic28188-16-5-Fluorescence-7b721b-v4a","dynamic28188-16-3-Fluorescence-7b721b-v4a"]' \
  dataloader.batch_size=4
```
> The `paths.data_dir` above is laptop-specific — **set it per the machine checklist in
> `qiu_2026_integration_plan_dataset_location.md`** before running anywhere else.

---

## Remaining work (non-blocking)

### Step A — Real-session data_io test suite (`tests/data_io/test_qiu_2026.py`)

Shifter/forward regression tests already live in `tests/models/test_core_readout_shifter.py` (6 tests),
and an offline synthetic end-to-end test in `tests/data_io/test_qiu_2026_train_wiring.py` (2 tests) —
but the latter never exercises the real loaders or FileTree data, so it would **not** have caught
either bug fixed in `923200c`. The dedicated real-data suite is still the gap. The real dataset is now
cached locally, so it can be written without a fresh download. Use the cached `dynamic28188-16-3` via
the `*_for_session` loaders. Cover:
- **Loaders:** `load_stimuli_for_session` → `MoviesTrainTestSplit`, `train.ndim == 4`, `C == 3`,
  per-session normalization applied; `load_responses_for_session` → 2D `(N, T)`, NaN-free after
  trim+mask, `N == 2710` for `16-3`; `load_pupil_for_session` frame-aligned with responses, normalized,
  NaN-free.
- **Container:** movie/response `test_dict` keys identical, `check_matching_stimulus` passes,
  `test_by_trial_dict` keys ⊆ `test_dict`, all train/val clips 300 frames. For `stimulus_type="clip"`
  the test dict should have ~6 condition keys, each with 15–20 repeat trials.
- **Regression guard for the `923200c` fix:** assert none of the 6 condition hashes are the
  `presentmoviearray` ones and that every repeat within a condition has identical shape (exactly what
  `np.stack` used to crash on). `16-3` has no `presentmoviearray` trials, so this specific regression
  needs one of `dynamic28188-19-9` / `28712-3-8` / `29163-4-4` / `29163-5-8` / `29163-6-5`.
- **Dataloader:** a train batch is a `QiuDataPoint` with `inputs (b,3,t,h,w)`, `targets (b,t,n)`,
  `pupil_center (b,2,t)`; curated validation carve-out disjoint from train.
- **(Optional, slow/marked):** `load_all_*` against the folder returns 10 session keys.

### Step B — CASCADE spike inference (open decision)

`spike_inference="cascade"` currently raises. To resolve: which pretrained CASCADE model / retrain;
sampling-rate match (data 30 Hz); per-trial vs. concatenated-trace inference; NaN-boundary handling;
output smoothing. CASCADE rates are non-negative → pairs with the readout's `softplus` output +
`PoissonLoss3d`. Keep the `raw`/`subtract_min`/`cascade` switch so targets can be compared. Until this
lands, `subtract_min` is the training default. (See decision D5.)

---

## Success Criterion (Definition of Done)

**Not yet met — blocked on hardware, not code.** A full training epoch with non-NaN loss + a saved
checkpoint:

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

## Open questions

1. **CASCADE recipe** — see Step B.
2. **Hardware for a full run** — see the Next step. Needs substantially more than 34 GB RAM, ideally a
   real GPU (the debug trainer forces `accelerator=cpu`, and real training would want a GPU trainer
   config regardless).
3. **`is_model_causal()` confirmation** — confirm the core passes the causality test. `model_cut_frames`
   is known (`18` for kernels `(11,5,5)`); Step 1's pupil alignment reads it from `data_info`
   dynamically regardless. Not yet run against the real qiu_2026 model — quick to check once a run
   produces a model instance.
4. **Test-input averaging over repeats** — the loaders average the *input* over a condition's 15–20
   repeats: a no-op for the video but a real choice for the behavior channels (`stimuli.py:107-113`) and
   pupil (`pupil.py`), which differ per repeat. Averaging pairs them with trial-averaged responses; the
   alternative is per-repeat inputs with averaged predictions. `test_by_trial_dict` retains the
   per-repeat responses either way. Flagged `# TODO(qiu_2026)` at both sites.
5. **Push the branch** — `qiu_2026-integration` has 2 local commits (`8d03ffb`, `923200c`) possibly not
   on `origin`. Check `git log origin/qiu_2026-integration..HEAD`.
