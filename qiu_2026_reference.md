# `qiu_2026` — Reference

Durable lookup material for the qiu_2026 integration: target model architecture and
hyperparameters, data format, and the session / neuron / test-structure tables. None of this
changes as the work progresses — it is read, not edited. Current status and next steps live in
`qiu_2026_integration_plan.md`; the rationale for the design choices lives in
`qiu_2026_decisions.md`.

---

## Target model architecture (from the Qiu 2026 methods)

Dynamic model = **3D factorized conv core** (Höfling 2024) + **Gaussian readout** (Lurz 2020b) +
**shifter network** (Sinz 2018), positive output. The paper's full model also adds a
**`grid_mean_predictor`** (Bashiri 2021) mapping cortex coordinates → readout mean, but per
**decision 1 we deliberately omit it** and learn per-neuron means directly (the reference calls this
the `v05aa` "no cortex coordinates" variant; `v05aaa` is the with-coordinates variant we are not
using).

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
  shared per session. Reference uses `gamma_shifter=0` (no shifter regularization).
- **Output nonlinearity:** the readout's default `softplus` gives a strictly-positive Poisson rate
  (correct for non-negative targets + `PoissonLoss3d`). The reference wraps the encoder output in
  `ELU+1`; that is a faithfulness-only toggle, deferred.

## Data-format quick reference

Sensorium FileTree, per-trial `.npy` under `data/<stream>/{i}.npy`, positionally aligned to
`meta/trials/*.npy` rows, NaN-padded to 450 frames @ 30 fps. Streams: `videos (H=36,W=64,T)`,
`responses (N,T)` raw calcium, `behavior (3,T)`, `pupil_center (2,T)`. Quality masks are **index**
arrays under `data-quality/<prefix>_neurons_fluor_good.npy`. Per-session norm stats under
`meta/statistics/`. Cortex coords `meta/neurons/cell_motor_coordinates.npy` `(N,3)` (loaded but now
unused per decision 1). The full historical per-stream inventory (shapes, dtypes, norm-stat paths,
string-encoded-metadata quirks) is in the archived `qiu_2026_integration_plan_legacy.md`.

## Session-hash reading

Session keys are `dynamic{animal}-{scan}-{idx}-Fluorescence-7b721b-v4a`; `7b721b` is the pipeline
hash (truncated to `7b7` in quality-mask filenames), `v4a` the version. The leading `{animal}`
groups the 10 sessions into 3 animals:

| Animal | Sessions | Note |
|---|---|---|
| `28188` (5) | `18-4`, `19-9`, `17-2`†, `16-5`†, `16-3`† | 3 of 5 have limited training data |
| `29163` (4) | `4-4`, `6-5`, `5-8`, `2-7`† | |
| `28712` (1) | `3-8` | |

† **Limited training data** (< 120 train trials): `28188-17-2`, `28188-16-5`, `28188-16-3`,
`29163-2-7`. Relevant to `LongCycler` balancing — these sessions contribute fewer clips.

## Per-session neuron counts (after the `neurons_fluor_good` mask)

From the reference readout dims; ~17.3k boutons total.

| Session | N | Session | N |
|---|---|---|---|
| `29163-4-4` | 3175 | `28188-19-9` | 787 |
| `28188-18-4` | 1593 | `28188-17-2` | 1714 |
| `28712-3-8` | 1728 | `29163-2-7` | 1327 |
| `29163-6-5` | 1079 | `28188-16-5` | 2046 |
| `29163-5-8` | 1113 | `28188-16-3` | 2710 (from 7636 pre-mask) |

## Test structure (natural `clip`)

There are **6 distinct test clips** of `stimulus_type="clip"`, each repeated **15–20 times within a
single session** (no cross-session pooling needed for repeats). The reference identifies them by
these 6 `condition_hash` values, shared across sessions:
`5zQTb77qI+ig8rigx1XU`, `7UETOWO5Z8aWuHDBJ2GG`, `GjCMo2GkJp6y5vricadg`, `KXdTNAGMo1gCWz2Ge8zr`,
`Oup5uAZxF2G7zEJkT+ui`, `ecUQJtcERZJGdqza1k7h`. So `test_conditions(..., stimulus_type="clip")` should
yield ~6 keys per session, each backed by 15–20 repeat trials.

The `presentmoviearray` trials (`chirp`/`moving_bar` functional-characterization stimuli, test tier
only, ~959–960 valid frames with per-repeat jitter) are **not** clips and are excluded by the
stimulus-type filter. Sessions `dynamic28188-19-9`, `28712-3-8`, `29163-4-4`, `29163-5-8`,
`29163-6-5` are the ones that actually contain `presentmoviearray` trials (confirmed on real data);
`16-3` has none.

---

## Appendix — paper methods, verbatim

> Model architectures. We used a dynamic network consisting of a 3D factorized convolutional core
> (Höfling et al., 2024), Gaussian readouts (Lurz et al., 2020b), and shifter modules (Sinz et al.,
> 2018) to model stimulus-response functions. The core included three sequential structures with a
> spatial convolutional layer (64 output channels) and a temporal convolutional layer (64 output
> channels) followed by a batch normalization layer and an exponential linear unit (ELU) function.
> The first spatial kernel was 1 × 11 × 11 and the other two were 1 × 5 × 5. Similarly, the first
> temporal kernel was 11 × 1 × 1 and the other two were 5 × 1 × 1. To estimate neuronal activity for
> each axonal bouton from the core output tensor (w × h × c, width x height x channels), we performed
> generalized linear regression with a learnable weight tensor (c × w × h) and an ELU function offset
> by one (ELU + 1) to enforce positive response. To reduce the learnable parameters in this
> regression problem, we adopted Gaussian readout computing a 2D Gaussian distribution N (μ, Σ) (μ,
> mean location of width and height in the core output tensor; Σ, uncertainty of the location) for
> each bouton's receptive field position (Lurz et al., 2020b). Parallel to the position learning, the
> model learned the feature weights of size c per bouton. For the model with access to the recorded
> 2D cortex coordinates of each bouton, we used a multilayer perceptron (MLP) with a size of 2−30−2,
> shared by all boutons in one session, to map the coordinates to the mean location μ (Bashiri et
> al., 2021). To account for the mouse eye movements during stimulus presentation, we employed a
> shifter network using the pupil center to shift the model neuron's receptive field center μ (Sinz
> et al., 2018). The shifter is an MLP consisting of three fully connected layers with n = 5 hidden
> features and a tanh nonlinearity.
