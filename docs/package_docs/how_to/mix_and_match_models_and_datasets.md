# Mix and match models and datasets

One strength of OpenRetina is that model definitions and dataset loaders are decoupled through shared interfaces. In practice, this means you can often reuse a model family across datasets by changing configs rather than rewriting training code.

## Concrete example: base core-readout across datasets

The `base_core_readout` model is used in multiple dataset-specific top-level configs, including:

- [`configs/hoefling_2024_core_readout_low_res.yaml`](https://github.com/open-retina/open-retina/blob/main/configs/hoefling_2024_core_readout_low_res.yaml)
- [`configs/karamanlis_2024_core_readout.yaml`](https://github.com/open-retina/open-retina/blob/main/configs/karamanlis_2024_core_readout.yaml)

Both compose:

- dataset-specific `data_io/*`
- dataset-specific `dataloader/*`
- shared `model: base_core_readout`

This is the core "mix-and-match" pattern: keep the model family, change dataset wiring.

## Practical recipe

1. Pick a top-level config close to your target data format.
2. Keep `model: base_core_readout` (or another compatible model group).
3. Swap `data_io` and `dataloader` groups to match the dataset.
4. Verify input shape and required metadata in the top-level `model:` overrides.

## Minimum config requirements (what you always need)

When you mix and match, these fields should always be explicitly defined in your top-level config:

- `exp_name`: run identifier used in logs and output directories.
- `paths.cache_dir`: where downloaded files are cached.
- `paths.data_dir`: local or remote data location consumed by `data_io` loaders. If remote, the download target will be in `paths.cache_dir`.
- `model.in_shape`: expected stimulus shape as `(channels, time, height, width)`.
- dataset/model defaults in `defaults`: `data_io`, `dataloader`, `model`, `trainer`, `logger`, and callbacks.

In practice, you should also confirm:

- `check_stimuli_responses_match`: set to `true` unless you are debugging loader mismatches.
- `trainer.max_epochs` and precision/device settings are realistic for your hardware.

Important: `n_neurons_dict` is required by the model, but it is injected automatically at train time from dataset metadata (`compute_data_info`). You do not need to set it manually in top-level configs.

## Compatibility caveats

Not every model can run on every dataset without adaptation.
For example, the package conatins single-cell models that rely on white-noise-derived receptive field parameters, which may require metadata that natural-movie-only datasets do not provide.

## How to validate compatibility quickly

- Run one short training/evaluation job with reduced epochs/batch size.
- Check that dataloaders expose expected keys (`train`, `validation`, `test`, or custom test names).
- Confirm model input/output shapes and neuron counts are resolved correctly.

See also:

- [Config templates](./config_templates.md)
- [Unified Training Script](../training/unified_training.md)
- [`docs/data_io_flow.md`](../../data_io_flow.md)
