# MEI ensembling on the qiu_2026 models

## Context

Qiu et al. 2026 generate dynamic MEIs by gradient ascent on an **ensemble of three models trained
with different random seeds**. We have three trained checkpoints (seeds 42/43/44) and a fully
instrumented single-model MEI pipeline. That pipeline's MEIs are spatially rough (`autocorr_lag1`
median +0.11) and only weakly reproducible across model seeds (+0.23). The ensemble is the one
remaining methodological difference from the paper, so it is the next thing to try.

Two things came out of measuring the three checkpoints before writing any code, and both change what
gets run:

1. **The MEI session was never chosen.** `SESSION = None` falls through to `readout_sessions[0]` =
   `dynamic28188-16-3`. That session's top-24 boutons are the worst-fit top-24 of all ten sessions
   (0.858 vs 0.938–0.947 elsewhere), and it is the only session where the three members disagree.
2. **Seed 44's readout failed on that session specifically** — test correlation 0.252 vs 0.450/0.519,
   and its vertical RF positions barely correlate with the other two (0.18, spread 2.8× larger). On
   the other nine sessions all three members agree closely. Seed 44 is not a broken model.

So the plan moves the MEI work to **`dynamic29163-4-4`**, where all three members are valid, and runs
the faithful three-member ensemble there with a matched single-model control. No extra training.

**Correction to the record:** `val_evaluation_loss` is `CorrelationLoss3d(negate=False)` monitored
with `mode: "max"` (`configs/training_callbacks/model_checkpoint.yaml:9`) — higher is better. So
seed 43 (0.411) is the best member and seed 44 (0.385) the worst, the opposite of what was stated
earlier in this project. Per-neuron test medians agree: 0.519 / 0.450 / 0.252.

---

## Measured facts this plan rests on

All read statically from the three `*_final.ckpt` files — no GPU, no dataset.

**Poolability — clean.** Identical `state_dict` keys and shapes; identical `n_neurons_dict`
(10 sessions, 17 272 boutons), `input_shape` `(3, 36, 64)`, `model_cut_frames` 18,
`mean_activity_dict`. Readout scales within 3% (bias mean 339.1/337.9/339.4), so no member dominates
the mean by magnitude.

**Neuron-index alignment — confirmed.** Cross-member correlation of the per-neuron
`data_info["pretrained_performance"]` vectors: 0.890 / 0.799 / 0.659. Misalignment would give ~0.
Bouton index `i` is the same physical bouton in all three models.

**Effective RF displacement across members**, on the top-24 boutons, in core pixels. "Effective"
means `mu + shifter(PUPIL_CENTER)` — the shifter does **not** absorb the `mu` offset, it slightly
enlarges it (1.62 → 2.48 px for s42–s43), so the raw `mu` is the wrong thing to measure.

| session | worst-pair median | worst-pair max | boutons > 5 px | min support IoU |
|---|---|---|---|---|
| `dynamic29163-4-4` **(chosen)** | 2.75 px | 2.84 px | **0 / 24** | 0.75 |
| `dynamic28188-17-2` (fallback) | 2.55 px | 2.73 px | 0 / 24 | 0.76 |
| `dynamic28188-16-3` (current) | 3.82 px | 7.24 px | **10 / 24** | 0.47 |

Note the displacements are near-uniform within a session-pair (median ≈ max): they are **rigid
translations of the whole retinotopic map**, not per-bouton scatter. Consequence to report, not to
hide: averaging three RFs displaced by a common ~2.8 px on a 20 px support is a ~14% translation
blur, which will raise `autocorr_lag1` somewhat *by construction*. §5 has the control for it.

**Top-bouton sets are not stable across members** — top-24 overlap s42/s43 = 10/24, s42/s44 = 9/24.
This forces the comparison design in §7.

**Baseline runtime**, from the notebook's own `metadata.execution`: cell `2c3ee74b` 2064 s,
`mei_sweep` 2310 s, everything else ≈ 55 s; 74 min total on a V100.

---

## 1. The ensemble class

**New file: `openretina/models/ensemble.py`.** Subclass `EnsembleModel` from
`openretina/modules/layers/ensemble.py` — the mean-of-member-responses forward already does the
right thing and passes `data_key`/`pupil_center` through to every member. Add only metadata and
validation. Keep it a plain `nn.Module`, not a `LightningModule`: an ensemble is inference-only.

```python
class CoreReadoutEnsemble(EnsembleModel):
    """Mean-response ensemble of independently trained BaseCoreReadout members.

    Exposes the *metadata* surface the in-silico tooling needs and deliberately exposes no
    weight-bearing submodule: `core`, `readout` and `shifter` raise AttributeError, because each
    member has its own and returning member 0's would label member 0's result "the ensemble".
    """

    def __init__(self, *members: BaseCoreReadout):
        super().__init__(*members)              # -> self.members : nn.ModuleList
        self.data_info = self._validate(members)  # a fresh dict, not any member's
        self.member_data_info = [m.data_info for m in members]

    @classmethod
    def from_checkpoints(cls, checkpoint_paths, device) -> "CoreReadoutEnsemble":
        """Load each Lightning .ckpt with `load_core_readout_model`, `.eval()` it, and pool."""

    def stimulus_shape(self, time_steps, num_batches=1): ...   # same body as BaseCoreReadout
    def readout_keys(self) -> list[str]: ...                   # the validated common list
    def core_output_shape(self, time_steps, num_batches=1): ...  # (c, t, h, w); raises on disagreement

    @property
    def member_sources(self) -> tuple[str, ...]: ...           # checkpoint paths, for metadata

    @property
    def core(self): raise AttributeError(_member_specific("core"))
    @property
    def readout(self): raise AttributeError(_member_specific("readout"))
    @property
    def shifter(self): raise AttributeError(_member_specific("shifter"))
```

`AttributeError` is load-bearing, not decorative: `getattr(model, "shifter", None)` swallows it, so
every existing call site degrades to "this model has no shifter" — loud and wrong in the safe
direction — instead of silently reporting member 0's shift field under an "ensemble" heading. Also
not delegated: `hparams`, `update_model_data_info`, `save_weight_visualizations`, the Lightning
hooks, and `data_info["pretrained_performance*"]` (per-member; kept in `member_data_info`).

**`FixedChannelStimulusModel(ensemble)` then works unmodified** — `_infer_num_channels` reads
`data_info["input_shape"]`, `stimulus_shape` delegates via `getattr`, and `readout_keys()` is found
by its `hasattr(model, "readout_keys")` branch. No change to
`openretina/insilico/stimulus_optimization/fixed_channel_model.py`.

**Also add `core_output_shape(time_steps, num_batches=1)` to `BaseCoreReadout`**
(`openretina/models/core_readout.py`) so the notebook's cut-frames cell is unconditional rather than
branching on model type.

Export `CoreReadoutEnsemble` from `openretina/models/__init__.py`. Leave
`openretina/modules/layers/ensemble.py` and `openretina/utils/nnfabrik_model_loading.py` untouched.

### No change to `IncreaseObjective` is needed — the reason, to be pinned by a test

With `R_m(x)[t, n]` the member-`m` response, the ensemble is `R = (1/M) Σ_m R_m` and the objective is
`J(x) = SliceMean_t( mean_{n∈I} R(x)[t,n] )`. `SliceMeanReducer` is `narrow` + `mean`,
`responses[:, I].mean(-1)` is a mean, and `.squeeze(0)` is a reshape — all linear, all commuting with
the member average. So `J = (1/M) Σ_m J_m` **exactly**, and `∇J = (1/M) Σ_m ∇J_m`. This holds despite
each member's readout nonlinearity, because the average is taken after every member's nonlinearity.

Two consequences: gradient ascent on the ensemble needs no objective change, and the objective keeps
a single model's units and gradient scale, so the existing `(1.0, 0.3, 0.1)` lr ladder stays in
range. The identity does **not** hold for `ContrastiveNeuronObjective` (it has a ratio), so scope the
docstring claim to `IncreaseObjective`.

---

## 2. Constructor validation

**Raise `ValueError`** on: fewer than 2 members; differing `readout_keys()` (as ordered lists);
differing `n_neurons_dict`; differing `input_shape`; differing `model_cut_frames` where present in
≥2 members; any member with `training is True`; disagreeing `stimulus_shape(2)`. `core_output_shape`
checks lazily on first call.

The eval-mode check matters specifically: `MultiSampledGaussianReadout` samples its grid
stochastically in train mode, so a train-mode member makes the ensemble mean a noisy estimate and
every downstream number irreproducible — the same reason `_evaluate_grid` already refuses
(`behavior_modulation.py:168`). `from_checkpoints` calls `.eval()` for you. Document that
`ensemble.train()` can undo it afterwards; the downstream guard covers that.

**Warn** on: `mean_activity_dict` mismatch (init metadata, unused at inference); `stim_mean` /
`stim_std` / `sessions_kwargs` mismatch (evidence the members saw different data — pair the warning
with a pointer to the §5 alignment diagnostic); differing member `type()`; per-member
`pretrained_performance` medians differing by more than ~1.5× (the seed-44 situation — flag, do not
refuse; an unweighted mean is Qiu's recipe).

No `strict=` escape hatch. All three of our members pass every check.

---

## 3. Notebook wiring — `notebooks/qiu_2026_insilico.ipynb`

Smallest diff that leaves the single-model path as the default.

**Cell `c7333a88` (config)** — add:

```python
DEFAULT_RUN = "core_readout_qiu_2026_mouse"
MEI_MODEL_RUNS = tuple(r.strip() for r in os.environ.get("MEI_MODEL_RUNS", DEFAULT_RUN).split(",") if r.strip())
CKPT_PATHS = None
SESSION = os.environ.get("MEI_SESSION") or None      # was hard-coded None -> readout_sessions[0]
MEI_TOP_BOUTONS_FROM = os.environ.get("MEI_TOP_BOUTONS_FROM") or None
```

Keep `CKPT_PATH = CKPT_PATHS[0]` as an alias so the from-disk notebook and metadata keys keep working.

**Cell `b794a215` (locator)** — generalise to N runs, and glob `*_final.ckpt` explicitly. Today's
`*.ckpt` + newest-mtime picks the right file in all three run dirs, but only by luck: `last.ckpt`
and `epoch=NN.ckpt` sit beside it.

```python
def newest_final_checkpoint(run_name: str) -> Path:
    root = REPO / "openretina_assets/runs" / run_name
    ckpts = sorted(root.glob("*/checkpoints/*_final.ckpt"), key=lambda p: p.stat().st_mtime)
    assert ckpts, f"no *_final.ckpt under {root} -- is training still running?"
    return ckpts[-1]

if CKPT_PATHS is None:
    CKPT_PATHS = [newest_final_checkpoint(r) for r in MEI_MODEL_RUNS]
CKPT_PATHS = [Path(p) for p in CKPT_PATHS]
CKPT_PATH = CKPT_PATHS[0]
```

**Cell `19891303` (load)** — branch on member count, and stop reading through `.readout`:

```python
if len(CKPT_PATHS) == 1:
    model = load_core_readout_model(str(CKPT_PATHS[0]), device)
else:
    model = CoreReadoutEnsemble.from_checkpoints([str(p) for p in CKPT_PATHS], device)
model.eval()
MEMBERS = list(getattr(model, "members", [model]))

readout_sessions = model.readout_keys() if hasattr(model, "readout_keys") else list(model.readout.keys())
if SESSION is None:
    SESSION = readout_sessions[0]
assert model_shifters(model), "This notebook assumes shifter-equipped model(s)."
n_boutons = model.data_info["n_neurons_dict"][SESSION]
```

`readout_keys()` returns **sorted** keys while `list(model.readout.keys())` returns insertion order.
They coincide here, but pin `SESSION` explicitly via `MEI_SESSION` in the launch script so this
cannot drift.

**Cell `162ee4bf` (`RESULTS_DIR`)** — make it structurally impossible for a multi-model run to
overwrite the single-model baseline, without adding a path segment (that would break the from-disk
notebook's variant discovery):

```python
if len(CKPT_PATHS) > 1 and MEI_VARIANT is None:
    MEI_VARIANT = f"ens{len(CKPT_PATHS)}-" + hashlib.sha1(
        "|".join(str(p) for p in CKPT_PATHS).encode()).hexdigest()[:8]
assert len(CKPT_PATHS) == 1 or MEI_VARIANT is not None
```

Extend `record_metadata` with `checkpoints`, `model_runs`, `n_members`, `member_val_corr_median`.
Keep the existing `checkpoint` key — the from-disk notebook reads it.

**Cell `2c3ee74b`** — replace `model.core(probe).shape[2]` with
`model.core_output_shape(MEI_TIME_STEPS)[1]`.

**Two latent bugs to fix in the same diff**, both of which the ensemble run would hit:

- Cell `8ec54e89` hard-codes `re_dir = Path("openretina_assets/insilico/qiu_2026") / SESSION`,
  ignoring `MEI_VARIANT`, then asserts the reloaded arrays match. Any variant run dies there.
  `run_qiu_insilico_variant.sh` has never actually been executed, which is why it has never fired.
  Fix: `re_dir = RESULTS_DIR`.
- `shifter.npz` gains a member axis (§4). Keep `shifts`/`align_corners` at member 0's shape so the
  from-disk notebook keeps working, and add `shifts_members` / `align_corners_members` /
  `shifter_member_runs` alongside. Same shape for `boutons.npz`: keep `val_corr` as the ensemble's,
  add `val_corr_members` `(M, n)`.

---

## 4. Which sections use the ensemble

| cells | model | why |
|---|---|---|
| `19891303`, `162ee4bf` | ensemble (metadata only) | — |
| `bf269b66` dataloaders | n/a | model-free |
| `7b3c6cf5` bouton ranking | **ensemble + per-member** | the per-member `val_corr` vectors *are* the §5 alignment diagnostic; ~4 s each |
| all MEI cells (`2c3ee74b`, `490e262c`, `mei_convergence`, `mei_sweep`, `286bb741`, `49ec289f`) | **ensemble** | the point of the exercise |
| behavior grids (`54dfda80`, `fcadcd07`, `6711c161`, `b12a8e6d`) | **ensemble** | `behavior_response_grid` needs only `forward`, `next(model.parameters())`, `model.training` — all fine. "How does behavior modulate the response" is a legitimate ensemble question. 2.5 s → 8 s. |
| `2e4e8704`, `b391d0bb` shifter shift grid | **per member, looped** | there is no ensemble shifter; each member applies its own inside its own forward, so an average of three MLP outputs is a field applied nowhere. Report the cross-member spread of `max \|shift\|` as an extra number. Uses `member.core(...)` and `member.readout[s].align_corners`, both available on a real member. |
| `8a2ecf31` `pupil_center_response_grid` | **ensemble** | needs a shifter only to *exist*; the shifting happens inside each member's forward with its own shifter, which is correct |

To make the last row work, one small library change in
`openretina/insilico/tuning_analyses/behavior_modulation.py`:

```python
def model_shifters(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Every shifter `model` will actually apply, in member order. [] if it applies none."""
    members = getattr(model, "members", None)
    if members is not None:
        return [s for s in (getattr(m, "shifter", None) for m in members) if s is not None]
    shifter = getattr(model, "shifter", None)
    return [] if shifter is None else [shifter]
```

- `pupil_center_response_grid` (line 323): `if not model_shifters(model): raise ValueError(...)` —
  accepts ensembles, still rejects shifter-free models.
- `shifter_shift_grid` (line 367): raise on `len(shifters) != 1`, with a message pointing at
  `model.members[i]`. The function that *reports* a shift field refuses to guess; the function that
  merely needs one to exist works.

Export `model_shifters` from `openretina/insilico/__init__.py`.

---

## 5. Gates and diagnostics

### Gate A — linearized-RF check, before the full run (minutes, one GPU)

`gradient_support` already backprops the objective at a zero stimulus and keeps only the sign
pattern of `probe.grad`. That gradient **is** the linearized RF at grey. Per the §1 linearity
identity, the ensemble's gradient RF is the mean of the members', so this needs no new code beyond
the ensemble class. For each member and the ensemble, on the chosen session's top-24 boutons,
measure `mei_metrics`' `autocorr_lag1` of the gradient RF, plus `support_px` and pairwise IoU.

| outcome | reading |
|---|---|
| members' gradient RFs already smooth while their MEIs are rough | roughness is optimizer/projection-induced, **not** model-induced. The ensemble will not fix it — do step-size decay / Adam first (`mei_optimization_reference.md` §6). |
| ensemble gradient RF smoother than the per-member mean **and** median `support_px` ≤ 500 | genuine cancellation of idiosyncratic high-frequency structure. Proceed. |
| smoother but median `support_px` > 600 or median IoU < 0.6 | blur from displaced RFs, not a result. Stop and re-plan. |

The static measurements predict ≈ 456 px (1.14×) on `dynamic29163-4-4`, so this should pass — but it
is the cheapest possible way to be wrong early.

### Gate B — translation-blur control

The three members' maps are rigidly offset by ~2.8 px, so *any* averaging raises `autocorr_lag1`
somewhat. The control: on **8 boutons**, compute each member's own single-model MEI and measure the
`autocorr_lag1` of their post-hoc mean, then compare to the jointly-optimized ensemble MEI. If joint
optimization does not beat the post-hoc average, the "win" is averaging, not ensembling. 24 extra
MEIs, cheap.

### In-run diagnostics — new cell `ensemble_checks`, saved to `ensemble_diagnostics.npz`

- per bouton, per pair: **effective** RF displacement in core px (`mu + shifter(PUPIL_CENTER)`, not
  raw `mu` — the shifter does not absorb the offset)
- per bouton, per member: `gradient_support` mask and `support_px`; per pair: IoU
- per bouton: ensemble `support_px` and its ratio to the single-model 400 px
- per bouton, per member and ensemble: gradient-RF `autocorr_lag1`
- per-member `val_corr` vectors and all pairwise correlations. **Abort if any pair < 0.3** (measured
  today: 0.89 / 0.80 / 0.66; misalignment gives ~0, so 0.3 is a wide moat)
- per-member mean response magnitude on the top boutons — confirms no member dominates the mean

Trust thresholds: median ensemble `support_px` ≤ 500 and median pairwise IoU ≥ 0.6 and median
displacement ≤ 3 px. Flag any individual bouton with `support_px > 600` and exclude it from the
headline metric, reported separately.

**Displacement-stratified reporting.** Split the top-24 into aligned (< 2 px worst-pair) and
displaced (≥ 2 px) and report `autocorr_lag1` separately. If the improvement appears only in the
displaced bucket, the win is blur and must not be reported as a win. Zero extra compute.

**Cross-member MEI transfer.** Evaluate the single-model MEIs under each member (48 × 3 forward
passes, ~1 s). Weak transfer quantifies non-transferability directly and predicts how much the
ensemble MEI must compromise.

---

## 6. Tests

**New: `tests/models/test_core_readout_ensemble.py`**, built entirely from
`build_tiny_qiu_model(seed=...)` in `tests/insilico/conftest.py` — no network, no GPU, no checkpoint.
Mirror `tests/insilico/stimulus_optimization/test_fixed_channel_model.py`'s style (spy member for
kwarg relay; an end-to-end `optimize_stimulus` test).

1. `forward` is the exact arithmetic mean of the members' forwards.
2. `data_key` and `pupil_center` reach **every** member unchanged (one spy per member).
3. `IncreaseObjective(ensemble).forward(x) == mean_i IncreaseObjective(member_i).forward(x)`, with
   multi-neuron `neuron_indices` and a `SliceMeanReducer(start=k, length=l)`. **This is the test that
   pins the "no objective change needed" argument.**
4. The gradient version of (3): `∇_x` of the ensemble objective equals the mean of the members'.
5. `FixedChannelStimulusModel(ensemble, {1: 0.0, 2: 1.5})` works unmodified — inferred
   `num_channels`, `stimulus_shape` drops constant channels, `readout_keys()` matches.
6. `.core` / `.readout` / `.shifter` raise `AttributeError`, **and**
   `getattr(ensemble, "shifter", None) is None` — the whole safety argument depends on the property
   being `getattr`-default-friendly.
7. `core_output_shape` matches `member.core(probe).shape[1:]`, and raises when members disagree.
8. Each validation failure raises `ValueError` naming the offending key; a train-mode member raises;
   a single member raises. Needs a small conftest addition: an `n_neurons_dict` parameter on
   `build_tiny_qiu_model` so a genuine mismatch can be built rather than monkeypatched.
9. `.eval()` and `.to()` propagate to members.
10. End-to-end `optimize_stimulus` on a 2-member tiny ensemble through `FixedChannelStimulusModel`
    with the norm projection: objective increases, constant channels bit-exact.

**Extend `tests/insilico/tuning_analyses/test_behavior_modulation.py`:** `model_shifters` returns 1 /
M / `[]`; `behavior_response_grid` and `pupil_center_response_grid` run on a 2-member ensemble and
return the mean of the per-member grids; `shifter_shift_grid(ensemble)` raises pointing at
`members[i]` while `shifter_shift_grid(ensemble.members[0])` works; the train-mode guard still fires
after `ensemble.train()`.

Do not test `from_checkpoints` — it needs real files, and it stays a thin loop over
`load_core_readout_model` with nothing in it to test.

---

## 7. Comparing ensemble against single model

Because the session is changing, the control has to be a **new single-model run on the same
session** — the existing `dynamic28188-16-3` results are not a valid comparison. Run seed 42 alone on
`dynamic29163-4-4` (~1.2 GPU-h) in parallel with the ensemble job.

**Both runs must use the same bouton list.** Top-24 sets differ across members by 10/24, so ranking
the ensemble run by the ensemble would silently change the subject. Implement
`MEI_TOP_BOUTONS_FROM=<results dir>`: when set, cell `7b3c6cf5` loads `boutons.npz["top_boutons"]`
from that directory instead of ranking, but still computes and saves its own `val_corr` and
`val_corr_members`, plus `top_boutons_source` in the metadata. Run the single-model control first so
the ensemble run can point at it.

Comparison cell in `notebooks/qiu_2026_insilico_from_disk.ipynb`, reading two results directories,
no GPU:

1. **Paired `autocorr_lag1`** over the 48 `(bouton, state)` pairs: median paired difference plus
   Wilcoxon signed-rank; also stratified by the §5 alignment bucket. The headline number.
2. **`support_px` ratio** ensemble/single, per bouton — the confound. Always reported beside (1),
   never separately.
3. **Single-vs-ensemble MEI correlation** per bouton, inside the **intersection** of the two supports
   (not the union: pixels one of them cannot reach are exactly 0 there and would inflate it).
4. **Cross-initialization-seed MEI correlation** from the `seed` ladder of `mei_knob_sweep.npz`,
   ensemble vs single. The direct successor to the +0.23. Name it carefully in any write-up: the
   sweep's "seed" is the MEI *initialization* seed, not the *model* seed.
5. **Prediction-side sanity, reported separately:** median ensemble `val_corr` vs each member's. It
   should exceed all three; if it does not, the ensemble is not even a better predictor and the MEI
   question is moot.
6. `rank1_frac`, `rms_inside`, `clipped_frac`, orbit width, rung usage — paired, to confirm the
   optimizer behaved comparably rather than dropping to lr=0.1 everywhere.

---

## 8. Slurm and runtime

MEI cells are 4375 s of the baseline's 4430 s. At 3× forward+backward ≈ 3.6 h, plus Gate B's 24 extra
MEIs. Worst case, if the ensemble landscape rejects more lr rungs (the baseline averaged 2.0 of 3),
≈ 5.5 h. **Raise `run_qiu_insilico_variant.sh` to `#SBATCH --time=0-12:00:00`** — the current 8 h is
too thin a margin for a 5-hour job, and `run_qiu_train_seed.sh` already uses `2-12:00:00` on
`a100-galvani`. Keep `--partition=a100-galvani,v100-galvani` and `--gres=gpu:1`. Memory is unaffected:
three members are 18 MB of weights each at batch 1.

The script already forwards arbitrary `MEI_*=VALUE` arguments, so `MEI_MODEL_RUNS`, `MEI_SESSION` and
`MEI_TOP_BOUTONS_FROM` need no script change. One thing does: its `*_final.ckpt` guard hard-codes the
unsuffixed run name — loop it over the comma-separated `MEI_MODEL_RUNS`, defaulting to the current
behaviour.

`mei_sweep` **stays in the ensemble run, unchanged.** Its `seed` ladder produces the
cross-initialization-seed correlation the ensemble is hypothesised to move, and its `smoothness`
ladder is where the baseline pointed. Do not restructure it to drop the contrast/window ladders: the
`.npz` key layout is consumed by the from-disk notebook, and changing it costs more than the 40
minutes it saves.

Launch commands:

```bash
# control: single model, new session
sbatch run_qiu_insilico_variant.sh 0.0 single-29163-4-4 \
  MEI_SESSION=dynamic29163-4-4-Fluorescence-7b721b-v4a

# headline: three-member ensemble, same session and same boutons
sbatch run_qiu_insilico_variant.sh 0.0 ens3-29163-4-4 \
  MEI_MODEL_RUNS=core_readout_qiu_2026_mouse,core_readout_qiu_2026_mouse_seed43,core_readout_qiu_2026_mouse_seed44 \
  MEI_SESSION=dynamic29163-4-4-Fluorescence-7b721b-v4a \
  MEI_TOP_BOUTONS_FROM=openretina_assets/insilico/qiu_2026/dynamic29163-4-4-Fluorescence-7b721b-v4a/single-29163-4-4
```

---

## Ordered steps

| # | step | depends on |
|---|---|---|
| 0 | ~~Commit the pending `notebooks/qiu_2026_inspect_predictions.ipynb` edit, and correct the val-loss direction in its `RUN_NAME` comment (seed 43 is the best member, not the worst)~~ **done** | — |
| 1 | Commit the static pre-flight measurements as a script under `scripts/` or a notebook cell, so the §"Measured facts" table is reproducible | — |
| 2 | `openretina/models/ensemble.py`: `CoreReadoutEnsemble`, `from_checkpoints`, validation. Export it. Add `core_output_shape` to `BaseCoreReadout` | 1 |
| 3 | `behavior_modulation.py`: `model_shifters`, rewire the two `getattr(model, "shifter")` sites, export | 2 |
| 4 | Tests (§6) incl. the conftest `n_neurons_dict` parameter. `make test-all` green before any GPU time | 2, 3 |
| 5 | **Gate A** on `dynamic29163-4-4`: gradient-RF `autocorr_lag1`, `support_px`, IoU for the members and the ensemble. **Stop and re-plan if it fails** | 2 |
| 6 | Notebook wiring (§3): config, locator, load, `RESULTS_DIR` guard, `core_output_shape`, `MEI_TOP_BOUTONS_FROM`, per-member val pass, `ensemble_checks` cell, shifter cells per member, `re_dir = RESULTS_DIR` fix, backward-compatible `.npz` keys | 2, 3, 5 |
| 7 | Smoke run on an interactive GPU with `N_BOUTONS=2` and `MAX_ITERATIONS=20` — catches shape/attribute errors in minutes, not hours | 6 |
| 8 | Script changes (§8); submit the single-model control on the new session | 7 |
| 9 | Submit the three-member ensemble pointing at the control's `top_boutons` | 8 |
| 10 | Gate B translation-blur control (8 boutons, per-member MEIs + post-hoc mean) | 9 |
| 11 | Comparison cell (§7) in the from-disk notebook | 9, 10 |
| 12 | Add an "Ensembling" section to `mei_optimization_reference.md` with the displacement table, the session-choice finding, and the go/no-go rule | 11 |

---

## Verification

- `make test-unittests` — the new ensemble tests plus the existing guards
  (`test_fixed_channel_model.py`, `test_regularizer.py`, `test_behavior_modulation.py`,
  `test_qiu_2026_stimulus_constants.py`).
- `make test-types` and `make test-codestyle` — the new module is type-checked.
- `make test-notebooks` for the notebook edits (nbmake), which exercises the single-model default path.
- Step 7's 2-bouton smoke run is the real integration test: it must produce non-zero MEIs, zero
  `norm_outside_frac`, and an `ensemble_diagnostics.npz`.
- The single-model control run on the new session must reproduce the baseline's structural invariants:
  `support_px == 400`, `norm_outside_frac == 0`, every MEI at its trace maximum.
- The ensemble run must clear every §5 threshold before its `autocorr_lag1` is quoted anywhere.

## Explicitly out of scope

- Changing `IncreaseObjective` or `optimize_stimulus` (provably unnecessary — §1; and the user has
  said not to touch the objective).
- Weighting members by validation performance — deviates from Qiu's recipe and turns the diagnostic
  into a knob.
- Step-size decay / Adam for the projection orbit (`mei_optimization_reference.md` §6). Still the
  right next experiment, but it is a separate change and Gate A may promote it to first priority.
- Training additional seeds. The session switch removes the need.
