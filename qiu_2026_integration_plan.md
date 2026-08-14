# Integration Plan: `qiu_2026` Dataset

Integrate the Franke-lab Qiu 2026 mouse dataset (10 sessions / 3 animals) so open-retina can **train
and evaluate** on it.

**Status (2026-07-23): ✅ DONE — Definition of Done met.** Data-loading, model-side wiring (MLP pupil
shifter), and Hydra configs are all implemented, committed, and pushed on `qiu_2026-integration`. The
**full 10-session training run completed** (Slurm job `420322`, `trainer=default_deterministic`,
batch_size 32): all 10 sessions, 50 epochs, a saved checkpoint, and a full test loop over all 98 test
dataloaders with finite losses and per-clip correlations ≈ 0.39–0.59. The
`notebooks/qiu_2026_inspect_predictions.ipynb` notebook confirms predicted-vs-recorded inference runs
end-to-end **with the pupil shifter active**. What remains is a **PR** (see `qiu_2026_pr_report.md`) plus
two optional/deferred, non-blocking items: a real-data test suite (Step A) and the CASCADE
spike-inference decision (Step B).

**Trained checkpoint:**
`openretina_assets/runs/core_readout_qiu_2026_mouse/2026-07-23_14-52-28/checkpoints/epoch=49_val_evaluation_loss=0.402_final.ckpt`
(`val_evaluation_loss` is the correlation metric, `monitor=val_evaluation_loss, mode=max`).

> **Node-hang — resolved.** Earlier full-run attempts hung; the cause was running bare on a shared node
> with `trainer=debug` (which hardcodes `accelerator: cpu`), causing ~50–90× BLAS/OMP thread
> oversubscription on the 4-core slice. It was never a cluster RAM issue. Fix (now the standard recipe):
> a Slurm GPU allocation + `trainer=default_deterministic` (**not** `debug`) + capped thread env vars
> (`OMP/MKL/OPENBLAS_NUM_THREADS`) via `sbatch run_qiu_full_train.sh`. Full forensics live in
> `qiu_2026_history.md`.

## Where things live

| Doc | Contents |
|---|---|
| **`qiu_2026_pr_report.md`** | **PR body:** dataset summary + added code + usage instructions. Start here for the merge. |
| `qiu_2026_integration_plan.md` (this file) | Current status + remaining/open work |
| `qiu_2026_integration_plan_dataset_location.md` | Machine-identification checklist + how to set `paths.data_dir` per machine (cluster / laptop / naive) |
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
| Additive, no-op-by-default `dataset_cls`/`extra_arrays` hook | ✅ Done | `data_io/base_dataloader.py` |
| MLP shifter (single + multi-session), Sinz 2018 | ✅ Done | `modules/shifters/mlp_shifter.py` |
| Shifter wired into `CoreReadout.forward` + train/val/test steps | ✅ Done | `models/core_readout.py` (commit `8d03ffb`) |
| Hydra configs (data_io / dataloader / model / top-level) | ✅ Done | `configs/**/qiu_2026*` (commit `8d03ffb`) |
| Stimulus-type filter | ✅ Done | `trials.py` + all loaders (commit `923200c`) |
| `compute_data_info` per-session norm-stat crash | ✅ Fixed (warn, not raise) | `data_io/base.py` (commit `923200c`) |
| Per-session `source_grid` → readout `grid_mean_predictor` | 🚫 Dropped | decision D7 — use `grid_mean_predictor: null` |
| Verification / walkthrough notebook (data + modules, end-to-end) | ✅ Done | `notebooks/qiu_2026_walkthrough.ipynb` |
| Prediction-inspection notebook (predicted vs recorded, shifter active) | ✅ Done | `notebooks/qiu_2026_inspect_predictions.ipynb` |
| GPU smoke test (2 sessions, non-debug trainer) | ✅ Passed 2026-07-23 (job `420320`) | `scratch_qiu_smoketest.sh` |
| **Full DoD run (all 10 sessions → checkpoint)** | ✅ **Done** 2026-07-23 (job `420322`, 50 epochs, corr ≈0.39–0.59) | `run_qiu_full_train.sh` |
| Branch pushed to `origin/qiu_2026-integration` | ✅ Done | — |
| Real-session data_io test suite | 🟡 Open (Step A) | `tests/data_io/test_qiu_2026.py` (not yet written) |
| CASCADE spike inference (loaders default to `subtract_min`) | ❌ Open decision (Step B) | `responses.py` |
| Now-dead cortex-coord plumbing in `responses.py` | 🧹 Optional cleanup | `responses.py` |

---

## ▶ Next step — open the PR

The code is complete and the full run has produced a checkpoint. Open a PR from
`qiu_2026-integration` into `main` using **`qiu_2026_pr_report.md`** as the description. Both notebooks
(`qiu_2026_walkthrough.ipynb`, `qiu_2026_inspect_predictions.ipynb`) are part of the PR.

**Reproduce the full run** (already done; kept for reference):
```bash
sbatch run_qiu_full_train.sh          # all 10 sessions, batch_size=32, trainer=default_deterministic
tail -f openretina_assets/slurm/qiu_full_<jobid>.log
tensorboard --logdir openretina_assets/runs/core_readout_qiu_2026_mouse   # loss curves (TB + CSV loggers)
```
On the cluster the default cache resolves the data (10 extracted session folders under
`~/openretina_cache/franke_lab/qiu_2026`), so **no `paths.data_dir` override is needed**. If a run OOMs
on GPU (peak is a `(32,3,300,36,64)` batch through the un-padded conv stack), drop `dataloader.batch_size`
(e.g. 16 or 8).

---

## Remaining work (non-blocking)

### Step A — Real-session data_io test suite (`tests/data_io/test_qiu_2026.py`)

Shifter/forward regression tests already live in `tests/models/test_core_readout_shifter.py` (6 tests),
and an offline synthetic end-to-end test in `tests/data_io/test_qiu_2026_train_wiring.py` (2 tests) —
but the latter never exercises the real loaders or FileTree data, so it would **not** have caught
either bug fixed in `923200c`. The dedicated real-data suite is still the gap. The real dataset is
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

## Success Criterion (Definition of Done) — ✅ met

A full training epoch with non-NaN loss + a saved checkpoint, produced under the **non-debug** trainer
in a Slurm allocation. **Met 2026-07-23** (job `420322`): all 10 sessions, 50 epochs,
`trainer=default_deterministic`, checkpoint
`…/2026-07-23_14-52-28/checkpoints/epoch=49_val_evaluation_loss=0.402_final.ckpt`, full 98-dataloader
test loop, per-clip correlations ≈ 0.39–0.59.

The checkpointed model predicts with the shifter active, for **all 10 session keys** (verified in
`notebooks/qiu_2026_inspect_predictions.ipynb`):

```python
# stimulus_tensor: (batch, C=3, t, h, w); pupil_center: (batch, 2, t)
model.forward(stimulus_tensor, data_key="dynamic28188-16-3-Fluorescence-7b721b-v4a",
              pupil_center=pupil_center_tensor)
# → (batch, T_out, N_session); T_out = t − 18 (core temporal convs reduce time)
```

---

## Open questions

1. **CASCADE recipe** — open; see Step B.
2. **Hardware for a full run** — ✅ resolved. Runs on `h100-ferranti` (1×H100) under Slurm with
   `trainer=default_deterministic` + capped threads via `sbatch run_qiu_full_train.sh`. Never a cluster
   RAM issue; see the header note and `qiu_2026_history.md`.
3. **`is_model_causal()` confirmation** — the core uses un-padded causal temporal convs
   (`model_cut_frames = 18` for kernels `(11,5,5)`) and Step 1's pupil alignment reads it from
   `data_info` dynamically. Confirm the trained qiu_2026 model passes the causality test — quick to
   check on the saved checkpoint.
4. **Test-input averaging over repeats** — the loaders average the *input* over a condition's 15–20
   repeats: a no-op for the video but a real choice for the behavior channels (`stimuli.py`) and pupil
   (`pupil.py`), which differ per repeat. Averaging pairs them with trial-averaged responses; the
   alternative is per-repeat inputs with averaged predictions. `test_by_trial_dict` retains the
   per-repeat responses either way. Flagged `# TODO(qiu_2026)` at both sites.
5. **Push the branch** — ✅ resolved. `qiu_2026-integration` is on `origin`
   (`git log origin/qiu_2026-integration..HEAD` is empty).
