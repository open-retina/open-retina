# MEI optimization — reference

Why this doc exists: the shipped MEI recipe (`openretina/cli/visualize_model_neurons.py`,
`notebooks/mei_example.ipynb`) is calibrated for `hoefling_2024` and **silently produces garbage on
`qiu_2026`** — not an error, just plausible-looking noise. Four of its steps are inert or wrong at
this dataset's geometry. This documents the mechanics, the instrumentation added to catch them, and
what the first properly instrumented run found.

Read this before changing anything under `openretina/insilico/stimulus_optimization/` or
`notebooks/qiu_2026_insilico.ipynb`.

---

## 1. The architecture determines almost everything

From `configs/model/qiu_2026_core_readout.yaml`: 3 layers, `temporal_kernel_sizes: [11, 5, 5]`,
`spatial_kernel_sizes: [11, 5, 5]`, `input_padding: false`, `hidden_padding: false`.

| Axis | Reach | Consequence |
|---|---|---|
| time | `10+4+4 = 18` frames | `T_out = T_in − 18`; minimum usable input 19 frames |
| space | `10+4+4 = 18` px | **19×19 px footprint** per feature-map location; 36×64 → 18×46 |

The `MultiSampledGaussianReadout` samples **one bilinear point** of the feature map in eval mode, so
a single bouton is driven by a **20×20 px window of the 36×64 frame — 17% of it**. Measured on a real
run: exactly 400 px for all 48 MEIs. Everything outside has *identically zero gradient*, forever:
the readout `mu` and the shifter output are frozen during stimulus optimization, and ELU has no hard
zeros, so the support is architectural and constant. Measure it with one backward pass rather than
assuming it (`gradient_support` in the notebook).

Verify the temporal number against `model.data_info["model_cut_frames"]` (18 in the shipped
checkpoint) instead of hard-coding it.

---

## 2. Five things that are inert or wrong, and were shipped that way

### 2.1 `RangeRegularizationLoss` contributes exactly zero

`optimize_stimulus` applies postprocessors *after* each step, so every forward pass sees a stimulus
that `ChangeNormJointlyClipRangeSeparately` has already clamped to range and renormalized below the
norm target. Both of the loss's hinge terms are therefore `relu(negative) = 0`, on every iteration
including the first. `factor=0.1` weights nothing.

Consequence: the CLI and every notebook that copies it run with **no regularization at all**. Pinned
by `tests/insilico/stimulus_optimization/test_regularizer.py::test_range_loss_is_inert_under_the_norm_projection`.

### 2.2 The `* 0.1` in the canonical init is inert

```python
stimulus = torch.randn(stimulus_shape, device=device)
stimulus.data = stimulus_postprocessor.process(stimulus.data * 0.1)   # the 0.1 does nothing
```

`process` renormalizes to the target norm, so the init lands at *full* target contrast regardless.
This works for `hoefling_2024` by accident — norm 30 over 28800 elements is RMS 0.177, so the
projection *shrinks* a std-0.1 init. For `qiu_2026` the target RMS is 0.996, so it *grows* it.

Combined with §1, that is how ~90% of a 50-frame `qiu_2026` MEI tensor ended up as frozen
full-contrast white noise holding **51% of the squared norm** and dominating every figure.

### 2.3 A whole-frame norm does not constrain a 20×20 receptive field

`ChangeNormJointlyClipRangeSeparately` computes one L2 norm over channels × time × H × W. Spending a
36×64-derived norm on a 20×20 support puts the MEI at ~2.4× the training video's contrast without
saying so. Derive the target over the support instead:

```python
norm = rms_factor * STIMULUS_RANGE_CONSTRAINTS["rms_video"] * (T * int(support.sum())) ** 0.5
```

Relative-contrast comparison across the codebase, since this is easy to get wrong:

| Call site | norm | elements | RMS | × data RMS |
|---|---|---|---|---|
| `mei_example.ipynb` | 5.0 | 28 800 | 0.029 | ~3% |
| `cli/visualize_model_neurons.py` (hoefling) | 30.0 | 28 800 | 0.177 | ~18% |
| `qiu_2026` `video_range_and_norm` | derived | — | 0.996 | **100%** |

### 2.4 Masking the init is not enough once any regularizer is active

The objective's gradient is confined to the support; a smoothness or range loss is computed over the
whole tensor, so *its* gradient pushes the surround back off zero. Measured leakage: 5% of the
squared norm at `factor_spatial=1e3`, 9% at `1e4`. Re-apply the mask every step, **before** the norm
processor so the norm lands on the support:

```python
stimulus_postprocessor=[ZeroOutsideMaskProcessor(support),
                        ChangeNormJointlyClipRangeSeparately(min_max_values=ranges, norm=norm)]
```

Also: starting from a small init is *not* a substitute for masking. Whether the frozen remainder ends
up negligible depends on accumulated gradient outweighing the init, which varies per bouton and per
step size — one bouton at `lr=0.3` still had 52% of its norm outside the support.

### 2.5 Fixed-step SGD cannot converge under a norm projection

The projection re-places the stimulus on a fixed-norm sphere after every step, so plain SGD with a
constant `lr` **orbits** the optimum in a band whose width scales with the step size. Measured orbit
width `(peak − final)/peak`: median 4.1%, max 18.9%.

Two consequences:

- The final iterate is arbitrary within the band. **Never assert on it.** An earlier
  "final within 1% of peak" check rejected all three step sizes on a run that was healthy at 98.7% of
  peak, and cost a 40-minute GPU job. Test the median of the trace's final tenth instead
  (`trajectory_is_stable`).
- Keep the **best** iterate, not the last. `optimize_stimulus` returns `None` and records no history
  (there is no `optimize_stimulus_with_history` anywhere in the repo); a thin proxy around the
  objective captures the trace and the argmax for free with no library change.

`OptimizationStopper.early_stop` always returns `False` — `max_iterations` is the only budget.
`EarlyStopper` exists but no shipped caller uses it.

---

## 3. Behavior channels: z-scored does not mean symmetric

`qiu_2026` folds pupil size and locomotion in as spatially-constant input channels 1 and 2. Both are
z-scored, which invites reading `0` as typical and `±1` as low/high. For locomotion that is wrong:
the channel is one-sided with a hard floor at `−mean/std` (≈ −0.19, session-dependent over
−0.11 … −0.30), and **69% of frames sit exactly on the floor**.

Occupancy of round-number states, session `dynamic28188-16-3`:

| State | Frames |
|---|---|
| `(-1, -1)` | **0.00%** — below anything the animal can produce |
| `(0, 0)` | 0.07% |
| `(+1, +1)` | 4.2% |

The joint distribution is bimodal. Use its modes, computed per session from the behavior trace:
**stationary** (−0.22, −0.19) at 94.5%, **running** (+3.35, +3.73) at 5.5%. Note the running mode sits
at +3.7, **outside** the ±2 box `BEHAVIOR_SWEEP_RANGE` covers — read the sweep figures accordingly.

Naming collision to watch: `BEHAVIOR_CHANNELS = (0, 2)` in
`openretina/data_io/qiu_2026/constants.py` indexes the *raw behavior array*; `BEHAVIOR_CHANNELS =
(1, 2)` in the notebooks indexes *model input channels* (0 = video). Same name, different meaning.

Freeze the channels with `FixedChannelStimulusModel`, never by resetting them after each step — the
joint norm makes the naive version decay the video geometrically to zero.

---

## 4. Instrumentation

Saved per MEI in `meis.npz`; reproducible with no GPU from
`notebooks/qiu_2026_insilico_from_disk.ipynb`.

| Name | Measures | Read as |
|---|---|---|
| `support_px` | reachable pixels | 400 = 20×20, architectural |
| `norm_outside_frac` | squared norm outside the support | 0 by construction; a regression guard, not a measurement |
| `rms_inside` | contrast within the support | compare to `rms_video = 0.996` |
| `clipped_frac` | support pixels on a range boundary | <1% not binding · 5–10% shaping · >20% near-binary |
| `autocorr_lag1` | lag-1 spatial autocorrelation in the support | →1 smooth · 0 noise · <0 checkerboard. **The smoothness number.** Tables abbreviate it `autocorr` |
| `rank1_frac` | space-time separable share of variance | →1 separable · ~0.2 a rank-1 spatial plot shows a fifth of what is there |
| orbit width | `(peak − final)/peak` | a few % is the projection; tens of % means step down |
| tail gain | gain in the final tenth of the budget | ~0% budget is comfortable · double digits still climbing |
| cross-seed correlation | pairwise MEI correlation across seeds, inside the support | →1 identifiable · →0 each run reports an arbitrary point |

`plot_stimulus_composition` shows the **rank-1** spatial component, so it is a fair summary only to
the extent `rank1_frac` is high. Pass `scale_bar_label=None` for any non-hoefling dataset — the bar's
width is hard-coded for the retina geometry.

`SmoothnessRegularizationLoss` is normalized by the mean square, so each term equals `2·(1−r)` with
`r` the lag-1 autocorrelation. Two implications: the weight means the same thing at every contrast
rung, and it acts directly on the number you measure on the result. **Scale**: `optimize_stimulus`
minimizes `−objective + Σ regularizers`, and the terms here are O(1) while `qiu_2026` responses are
O(10³) — so the useful weight range is O(10²–10³), not the repo's habitual `0.1`.

---

## 5. What the first instrumented run found

24 boutons × 2 behavioral states, 28 frames, 1000 iterations, `smoothness = 0`. Slurm 2784289,
V100, 1:14:06.

| | |
|---|---|
| cross-seed MEI correlation | **median +0.23** (−0.10 … +0.61, 12 pairs) |
| objective spread across seeds | 6374 / 5398 / 6193 (~15%) |
| `autocorr_lag1` | median **+0.11** (−0.31 … +0.35) |
| `rank1_frac` | median 0.25 |
| step sizes chosen | `{0.3: 34, 1.0: 7, 0.1: 7}` |

**The §2 fixes worked and the MEIs are still not smooth.** `norm_outside_frac` is 0.000 throughout,
support is exactly 400 px — and `autocorr_lag1` is +0.11. The budget problem was real, is fixed, and
was never the cause of the roughness.

Knob sweeps (4 boutons, reference state, one factor at a time):

- **Contrast** — objective rises monotonically with `rms_factor` (2097 → 6458 over 0.1 → 1.0), but
  `autocorr` goes the *other* way: **−0.32 at `rms_factor=0.1`**. Lower contrast makes the MEI more
  high-frequency, not less. In the near-linear regime the norm-constrained optimum *is* the linearized
  RF, so this says the model's learned spatial filters for these boutons are high-frequency. A finding
  about the fit, not a knob.
- **Spatial smoothness** — the only lever that moves `autocorr`. At `factor_spatial=1e3`,
  `autocorr = +0.62` while the objective holds at **84%** of unconstrained. At `1e4` it degenerates:
  objective −66% and `autocorr` back to −0.04.
- **Window length** — `{1, 3, 10}` output frames changes the objective as expected and leaves
  `autocorr` flat. Not the lever.

---

## 6. What is NOT established

**The non-identifiability result is confounded with optimizer non-convergence.** Cross-seed
correlation of +0.23 was measured with a *constant* step size, so each seed stops at an arbitrary
point inside its own 4%-median orbit (§2.5). Best-iterate selection reduces this but does not remove
it — the best iterate of an orbiting walk is still a sample from the orbit. A flat landscape and a
non-converging optimizer predict the same low correlation.

Resolve before building on it: add step-size decay (or Adam) so runs actually settle, then re-measure
cross-seed correlation on the same boutons and seeds. If it rises substantially, the landscape is not
flat and the +0.23 was measurement noise.

Second open question, cheap to answer and prior to any MEI work: **is the model's receptive field
itself smooth?** Backprop the response at a grey stimulus — that gradient *is* the linearized RF —
and measure its `autocorr_lag1`; inspect the first-layer 11×11 spatial filters; compare the MEI's
radial spatial power spectrum against the training video's. If the gradient RF is checkerboard, a
smooth MEI is an imposition rather than a discovery, and the question moves upstream to
`gamma_input: 10` (the layer-1 spatial Laplace weight) and whether the model fits pixel-scale noise
at 0.56 validation correlation.

---

## 7. Do not

- **Assert on the final iterate** of a projected optimization (§2.5).
- **Tune `factor_spatial` until the figures look right.** At `1e4` the penalty produces degenerate
  MEIs, not smooth ones, and per §5 smoothing may be overriding what the model encodes.
- **Report a single MEI per neuron** without a cross-seed correlation beside it. One seed gives
  reproducibility, which is not identifiability.
- **Read `norm_outside_frac = 0` as a result.** It is true by construction.
- **Trust `rms_factor` comparisons under an unnormalized smoothness penalty.** The normalization in
  `SmoothnessRegularizationLoss` exists precisely so the weight survives a contrast change.

---

## 8. File map

| Path | Role |
|---|---|
| `openretina/insilico/stimulus_optimization/optimizer.py` | the only driver; no history, no LR schedule, no gradient clipping |
| `.../objective.py` | `IncreaseObjective`, `SliceMeanReducer` (indexes **output** frames), `pupil_center` handling |
| `.../regularizer.py` | `ChangeNormJointlyClipRangeSeparately`, `SmoothnessRegularizationLoss`, `ZeroOutsideMaskProcessor`, `TemporalGaussianLowPassFilterProcessor` |
| `.../fixed_channel_model.py` | `FixedChannelStimulusModel`; module docstring has the geometric-decay arithmetic |
| `openretina/data_io/qiu_2026/constants.py` | `STIMULUS_RANGE_CONSTRAINTS`, `video_range_and_norm(time_steps, ..., rms_factor)` |
| `notebooks/qiu_2026_insilico.ipynb` | producer; MEI knobs are env-overridable, results go to `RESULTS_ROOT/<session>/<MEI_VARIANT>` |
| `notebooks/qiu_2026_insilico_from_disk.ipynb` | every figure from the `.npz` files, no GPU; `VARIANT` selects the run |
| `run_qiu_insilico_notebook.sh` | baseline run; writes executed outputs back over the tracked notebook |
| `run_qiu_insilico_variant.sh` | variant run; keeps the executed copy beside its results instead |
| `tests/insilico/stimulus_optimization/test_regularizer.py` | pins §2.1 and §2.4 |

Slurm: a100/v100 only. `bethge` and `2080-galvani` are both entirely rtx2080ti, and a 2080 Ti is too
slow now that the MEI cells run ~1e5 forward/backward passes.
