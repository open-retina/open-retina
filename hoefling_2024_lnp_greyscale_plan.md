# Greyscale LNP on hoefling_2024 — plan and handoff

Status as of 2026-08-25: **code complete and verified offline, training not yet run.**
Everything below is committed on branch `feature/greyscale-lnp`. The remaining work is to
submit `run_hoefling_lnp_grey.sh` on a machine with a GPU allocation and interpret the result.

## Why

The hoefling_2024 stimulus is 2-channel (green, UV). The LNP model is `DummyCore` (a
pass-through that only cuts the first 30 frames) plus `MultipleLNPReadout`, whose per-neuron
kernel is a full-field `nn.Conv3d(kernel_size=(1, 18, 16))`. Colour therefore costs
`2 * 18 * 16 = 576` weights per neuron, and those are the *only* trainable weights in the whole
model — the core has none.

We want a greyscale variant, i.e. collapse the two colour channels to one before the readout,
which halves the readout to 288 weights per neuron. That is a meaningful capacity cut for a model
that overfits within ~13 epochs unless it is heavily smoothness-regularized.

## The blocker that made this more than a config change

`DummyCore` had no colour-squashing support — only `SimpleCoreWrapper` did (added in #146). Worse,
`DummyCore.__init__` ended in `**kwargs`, so adding `color_squashing_weights` to the yaml would
have been **silently swallowed**: no error, no squashing, and a run labelled "grey" that actually
trained on both colour channels.

That failure mode is not hypothetical in this config family. `data_io.retina_pixel_size_um: 50`
sits at the top level of the `data_io` block in four top-level configs, while
`configs/data_io/hoefling_2024.yaml` defines the key at `data_io.data_info.retina_pixel_size_um`.
Nothing reads it by name, so it is inert either way — but it is the same class of silently-dead
config key. Left as-is; out of scope.

## How greyscale actually gets wired

The readout's channel count is **derived, not declared**. `configs/model/linear_nonlinear_poisson.yaml`
leaves `readout.in_shape: ???`; OmegaConf reports a MISSING value as absent, so
`UnifiedCoreReadout.__init__` (`openretina/models/core_readout.py:343-345`) takes this branch:

```python
if "in_shape" not in readout:
    in_shape_readout = self.compute_readout_input_shape(in_shape, core_module)
    readout["in_shape"] = (in_shape_readout[0],) + tuple(in_shape_readout[1:])
```

`compute_readout_input_shape` (`core_readout.py:235-249`) runs a real probe forward through the
core with a zeros tensor. So squashing inside `DummyCore` propagates automatically:

```
model.in_shape [2,150,18,16] -> probe (1,2,150,18,16) -> DummyCore squash+cut
    -> (1,1,120,18,16) -> readout.in_shape (1,120,18,16) -> Conv3d(in_channels=1) -> 288 weights
```

### `model.in_shape` must stay `[2, 150, 18, 16]`

This is the one thing that is easy to get wrong. `in_shape` describes the **stimulus**, which is
still 2-channel; the squash is internal to the core. Note this is the *opposite* convention to
`SimpleCoreWrapper`, which validates `channels[0] == 1` (`base_core.py:95-99`) and therefore needs
`in_shape[0] = 1`. Setting `[1, ...]` here is wrong.

Nothing else disagrees about the channel count: `data_info["input_shape"]` is read off the actual
movie (`openretina/data_io/base.py:325-329`) and stays `(2, 18, 16)`, which is what
`BaseCoreReadout.stimulus_shape` should report — a stimulus fed to a grey model is still
2-channel. MEI/insilico stimulus generation therefore keeps working unchanged.

## What changed (six commits on `feature/greyscale-lnp`)

| # | Commit | Files |
|---|---|---|
| 1 | `fix(configs): set optimizer and lr_scheduler for the LNP config` | `configs/hoefling_2024_core_readout_low_res_lnp.yaml` |
| 2 | `fix(reducers): reject channel-weight counts that do not match the input` | `openretina/modules/layers/reducers.py`, `tests/modules/layers/test_reducers.py` |
| 3 | `feat(core): optional greyscale colour squashing in DummyCore` | `openretina/modules/core/base_core.py`, `tests/modules/core/test_base_core.py` |
| 4 | `feat(models): log the readout input shape derived from the core` | `openretina/models/core_readout.py` |
| 5 | `feat(configs): greyscale LNP training config for hoefling_2024` | `configs/model/linear_nonlinear_poisson.yaml`, `configs/hoefling_2024_core_readout_low_res_lnp_grey.yaml`, `tests/models/test_lnp_greyscale.py` |
| 6 | `docs: greyscale LNP plan and portable launch script` | this file, `run_hoefling_lnp_grey.sh` |

Notes on the non-obvious ones:

- **Commit 1** is an independent pre-existing bug: the LNP top-level config was missing the
  `optimizer` and `lr_scheduler` config groups, so `${optimizer}` in the model config raised
  `InterpolationKeyError`. Four other configs still have this gap and were left alone:
  `vystrcilova_2024_{wn,nm}_ln.yaml`, `sridhar_2025_{wn,nm}_sc.yaml`.
- **Commit 3** deliberately **drops `**kwargs`** from `DummyCore.__init__`, replacing it with an
  explicit `n_neurons_dict=None  # for compatibility` (the pattern `SimpleCoreWrapper` already
  uses at `base_core.py:80`, since hydra injects that kwarg into every core). A typo such as
  `color_squashing_weight` now raises `TypeError` instead of quietly training a colour model.
  Blast radius checked: the only other caller is `openretina/models/spatial_contrast.py:425`,
  which passes `cut_first_n_frames` only.
- **Commit 3** reuses the existing `WeightedChannelSumLayer`
  (`openretina/modules/layers/reducers.py`), which was already exported and tested. No new layer.
- **Commit 5** adds `color_squashing_weights: null` to the *shared* model config purely for
  discoverability — it is a no-op for the other consumers (`maheswaranathan_2023_LNP.yaml` is
  single-channel) and it means CLI overrides work without hydra's `+` prefix.

The grey config differs from the colour one in exactly two ways: `exp_name` (load-bearing —
`configs/hydra/default.yaml` interpolates it into `hydra.run.dir`, so sharing it would interleave
grey and colour output trees) and the new `model.core` block.

## Verification already done (offline, no GPU, no data)

10 tests pass across the three test files; `tests/modules/` + `tests/models/` (excluding the
network-dependent `test_core_readout.py`) gives 20 passed / 12 xfailed, no failures.

| check | result |
|---|---|
| colour config still builds 2 channels | **576** weights/neuron |
| grey config builds 1 channel | **288** weights/neuron |
| squash is numerically a mean | `f([g, uv]) == f([mean, mean])` |
| existing colour checkpoint loads after the change | `color_squashing_layer is None`, 2 channels, forward OK |
| grey → grey `state_dict` | strict load, all keys matched |
| colour ← grey `state_dict` | rejected with `RuntimeError`, as it should be |
| typo'd kwarg | raises `TypeError` |
| hparams carry the squash weights | so `load_from_checkpoint` rebuilds the layer |

Checkpoint compatibility is safe in both directions because `save_hyperparameters(logger=False)`
(`core_readout.py:89`) stores the `core` DictConfig: an old checkpoint has no
`color_squashing_weights` key, so the new signature defaults it to `None`, no layer is built, and
no `state_dict` key is expected. Verified against the real artifact
`openretina_assets/runs/lnp_reg/441210/smooth3e4/checkpoints/epoch=83_val_evaluation_loss=0.098_final.ckpt`.

Caveat: old code cannot load a *grey* checkpoint, and `load_core_readout_model`'s bare `except:`
(`core_readout.py:537-539`) would mask the real "unexpected key" error behind a confusing
`ExampleCoreReadout` fallback. Not fixed; just know it.

### Reproducing the verification on the new machine

`pytest` is a declared `dev` optional dependency but was **not** installed in the `.venv` on the
original machine (nor were `ruff` and `mypy`, so **this branch has not been lint- or
type-checked**). Install and run:

```bash
uv pip install --python .venv/bin/python pytest    # or: pip install -e '.[dev]'
.venv/bin/python -m pytest tests/modules/core/test_base_core.py \
    tests/modules/layers/test_reducers.py tests/models/test_lnp_greyscale.py -q
```

Config resolution check (seconds, no data download — needs the cache dir only for the
`${oc.env:...}` interpolation to resolve):

```bash
OPENRETINA_CACHE_DIRECTORY=<cache> .venv/bin/openretina train \
  --config-name hoefling_2024_core_readout_low_res_lnp_grey --cfg job --resolve
```

Expect `exp_name: ..._lnp_grey`, `model.in_shape: [2, 150, 18, 16]`,
`model.core.color_squashing_weights: [0.5, 0.5]`, and `model.readout.in_shape: ???`
(missing on purpose — it is filled in at model construction).

## The run

```bash
export OPENRETINA_CACHE_DIRECTORY=<cache>          # required, no default
sbatch --partition=<gpu-partition> --gres=gpu:1 run_hoefling_lnp_grey.sh
```

The script is site-agnostic on purpose: it derives the repo root from its own location, requires
`OPENRETINA_CACHE_DIRECTORY`, and carries **no partition or `--gres` directive** — pass those on
the command line. `OPENRETINA_BIN`, `PYTHON_BIN` and `MAX_EPOCHS` are overridable. It also runs on
CPU (drop `--gres`), just slower.

Two arms in one sequential job:

| arm | `model.readout.smooth_weight` | rationale |
|---|---|---|
| `grey_base` | `1` (config default) | numerically inert (~2e-4 of the Poisson term), so this is the like-for-like counterpart of the colour baseline |
| `grey_smooth3e4` | `3e4` | best colour arm from the earlier sweep |

Resources: `--mem=16G`, `--time=0-00:30:00`, `--cpus-per-task=6`. The colour reference runs (job
441210) each finished in **under 3 minutes** on one H100 with a measured peak RSS of 4.5 GB, so
this is generous. It is one job rather than a 2-task array because a small, short, single request
backfills far better on a saturated partition.

After each arm the script prints a **greyscale receipt** parsed from `csv/hparams.yaml`:
`readout in_shape=[1, 120, 18, 16] -> GREYSCALE`. If it prints `[2, ...] -> *** NOT GREYSCALE ***`
the squash did not happen and the result is meaningless — that check exists precisely because the
channel count is derived rather than declared.

The final aggregation calls `scratch_lnp_reg_summarize.py`, which is **untracked local tooling**
from the earlier colour sweep and is therefore not on this branch. The script guards for its
absence and tells you where the raw `csv/metrics.csv` files are. To summarize by hand: peak
validation correlation is `max(val_evaluation_loss)`, and the test score is the
`CorrelationLoss3d/dataloader_idx_2` row written by `trainer.test`.

## Numbers to compare against

From the earlier colour sweep on the same data, seed, and schedule:

| model | best val corr | test corr | peak epoch |
|---|---|---|---|
| colour, `smooth_weight=1` (baseline) | 0.0860 | 0.2503 | 3, early-stopped at 13 |
| colour, `smooth_weight=3e4` (best) | 0.0983 | 0.2594 | 83 |

Within-arm seed noise on the colour model was ~0.004 val, so treat anything smaller than that as
indistinguishable. `smooth_weight` values of 1e2 and 1e3 were *worse* than the inert default, and
1e5 overshot (val 0.0818) — the useful range is narrow.

### Two things to keep in mind when interpreting

1. **`smooth_weight` should transfer, `sparse_weight` should not.** `LaplaceL2norm` is the ratio
   `sum(laplace(w)^2) / sum(w^2)` (`openretina/modules/layers/regularizers.py:209-215`), so it is
   scale- and channel-count-free; measured at init it is 19.85 colour vs 19.43 grey. But
   `weights_l1` averages over `in_channels * 288` weights
   (`openretina/modules/readout/linear_nonlinear_poison.py:74-77`), so at a fixed `sparse_weight`
   the grey model feels ~2x the per-weight L1 pressure, and more once the kernel grows to
   compensate for the mean-not-sum input scaling. The earlier sweep found `sparse_weight` a near
   no-op between 1 and 1e5 for the colour model, so this is probably second order — but it is the
   one term that does not transfer cleanly. If the grey arms underperform unexpectedly, a third
   arm with `model.readout.sparse_weight=0.5` is the first thing to try.
2. **Init streams are not comparable.** `seed_everything(42)` now draws `288 * N` xavier samples
   instead of `576 * N`, so the grey run's initialization is not a matched pair with the colour
   run's. Inherent, not fixable.

## Open questions the run should settle

1. Does greyscale cost accuracy? A drop of ≤0.004 val would mean colour information is not being
   used by this model class, which would be a real finding — the LNP has no temporal filter at
   all (`kernel_size=(1, 18, 16)`, `# Not using time`), so it may not be exploiting chromatic
   structure either.
2. Does `smooth_weight=3e4` still beat `1` at half the parameter count, or does the capacity cut
   substitute for the regularizer?
3. If greyscale holds up, should `[0.5, 0.5]` become trainable? `WeightedChannelSumLayer` already
   takes a `trainable` flag, though `DummyCore` does not currently expose it — that would be a
   learned 1-D chromatic filter rather than a greyscale model, so it is a different experiment.

## Decisions already made (do not silently revisit)

- Extend `DummyCore` rather than adding a new core class or squashing in the dataloader.
- Weights `[0.5, 0.5]` — the mean, not the sum. Keeps the stimulus in the same range as each
  individual channel, so learning rate and regularizer scales carry over from the colour runs.
- Not trainable.
- Two arms, `smooth_weight` 1 and 3e4, rather than assuming the colour optimum transfers.
- `smooth_weight` is **not** written into the shared `configs/model/linear_nonlinear_poisson.yaml`;
  that file is also used by `maheswaranathan_2023_LNP.yaml` and the sum-reduced Poisson term
  scales with batch x time x neurons, so the right value is dataset-specific.
