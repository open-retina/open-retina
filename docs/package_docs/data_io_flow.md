# Data IO flow and multi-test support

This document captures the assumptions shared between the data containers, dataloaders, and CLI entry points. Refer to it when wiring a new dataset or troubleshooting evaluation.

## Containers

- `MoviesTrainTestSplit` stores the continuous training movie plus _named_ test stimuli in
  `test_dict`. Provide either a single test array (it will automatically become
  `{"test": array}`) or a dictionary such as `{"frozen_a": array_a, "frozen_b": array_b}`.
- `ResponsesTrainTestSplit` mirrors that structure via `test_dict` for the averaged test
  responses and `test_by_trial_dict` for per-trial traces (shape
  `trials × neurons × time`). When only one test stimulus exists you may pass
  `test_by_trial` instead of the dictionary, but multi-test datasets **must** populate
  `test_by_trial_dict` with matching keys.
- `check_matching_stimulus` asserts that the movie and response dictionaries share the
  exact same keys and time bins, making downstream iteration safe.

## Splitting into dataloaders

1. `multiple_movies_dataloaders` slices the training stimulus/response pairs into training
   and validation clips (optionally reusing user-provided validation indices).
2. It wraps responses with `NeuronDataSplit`, which:
   - Transposes train/validation responses into `[time, neurons]`.
   - Exposes `response_dict_test[name] = {"avg": tensor, "by_trial": tensor}` for every
     key defined in `ResponsesTrainTestSplit.test_dict`.
3. For every `name, movie` in `MoviesTrainTestSplit.test_dict.items()` we create a
   dedicated PyTorch `DataLoader` whose dataset contains both the averaged responses and
   the per-trial tensor (if available). As a consequence, `dataloaders` contains entries
   such as `"train"`, `"validation"`, `"test"` (default name), and `"frozen_b"` if a
   second test sequence exists.

`MovieDataSet` automatically stores `dataset.test_responses_by_trial` whenever its
`responses` argument is a dictionary with the keys `"avg"` and `"by_trial"`. The
per-trial tensor is optional; if missing we propagate an empty tensor so evaluation code
can fall back gracefully.

## Training / evaluation scripts

- `openretina/cli/train.py` consumes the dictionary returned by
  `multiple_movies_dataloaders`. Lightning sees only the `"train"` and `"validation"`
  loaders during fitting, while **all** entries (train/val/test/custom) are passed to
  `trainer.test` after training so you get metrics for every split.
- `openretina/cli/eval.py` instantiates the same dataloader config but evaluates exactly
  one split specified via `cfg.evaluation.data_split` (default: `"test"`). This lets
  you reuse the script to benchmark on `"train"`, `"validation"`, or any additional
  frozen stimulus without changing code. The output dataframe includes a `data_split`
  column so you can aggregate within or across stimuli later.
- If `cfg.evaluation.compute_dataset_statistics` is enabled (default), evaluation also iterates through all dataloader splits to compute frame/transition fingerprints. This is useful for diagnostics but can add runtime for large datasets.

## Adding a new dataset

1. Implement `load_all_stimuli()` and `load_all_responses()` that return dictionaries of
   the two split classes defined above. The dictionary keys represent session names and
   **must** match between stimuli and responses.
2. Populate `MoviesTrainTestSplit.test_dict` with one entry per frozen test stimulus.
   Use descriptive names (e.g. `"flicker_repeat_0"`).
3. Populate `ResponsesTrainTestSplit.test_dict` with matching keys. If per-trial repeats
   are available for a stimulus, add them to `test_by_trial_dict[key]`.
4. Point a Hydra config at those two functions so that `openretina/cli/train.py` and
   `openretina/cli/eval.py` can instantiate them. No further wiring is required to gain
   multi-test handling in training, testing, and evaluation.

Following these conventions keeps the entire stack (data containers, dataloaders, Lightning training, and evaluation) synchronized and capable of handling multiple test movies plus their trial repetitions.
