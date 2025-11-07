# Training and Evaluation

Here we show how to launch a training loop manually, and how to evaluate a trained model. The examples assume you already prepared matching dataloaders as described in the [Data I/O guide](../data_io/index.md) and instantiated a model such as `ExampleCoreReadout` from the [model overview](./core_readout.md).

## Training Loop

The dataloaders returned by `openretina.data_io.base_dataloader.multiple_movies_dataloaders` are organised by session. Utilities in `openretina.data_io.cyclers` let you iterate across sessions to present data to PyTorch Lightning as a single stream.

```python
from lightning import Trainer
from openretina.data_io.cyclers import LongCycler, ShortCycler

train_loader = LongCycler(dataloaders["train"], shuffle=True)
val_loader = ShortCycler(dataloaders["validation"])

trainer = Trainer(max_epochs=50, gradient_clip_val=0.5)
trainer.fit(model, train_loader, val_loader)
```

Lightning handles the boilerplate for checkpoints, mixed precision, logging, and distributed execution. Loss functions, optimisers, and regularisers live inside the model (see the “Extending or Customising Models” section in the [core + readout page](./core_readout.md)), so the training loop only needs the model and the loaders.

Tips:

* Use `LongCycler` for splits where you want balanced sampling across sessions; use `ShortCycler` when one pass over each session is sufficient (e.g. validation or test).
* Pass callbacks (early stopping, learning-rate schedulers) to `Trainer` if you require certain functionalities.
* To monitor metrics, enable Lightning loggers such as TensorBoard or CSV by providing `logger=...` to `Trainer`.

Refer to the [Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for the full list of trainer arguments.

## Evaluating a Model

After training, switch to the test split and let Lightning handle evaluation:

```python
from openretina.data_io.cyclers import ShortCycler

test_loader = ShortCycler(dataloaders["test"])
trainer.test(model, test_loader)
```

During `trainer.test`, the model runs in evaluation mode (`model.eval()`) and metrics defined in the module are logged. By default `BaseCoreReadout` uses the Poisson loss for optimisation and reports the time-averaged Pearson correlation as an evaluation metric. You can customise these via the model constructor.

## Additional Metrics and Oracles

Beyond the losses embedded in each model, `openretina` provides richer evaluation utilities under `openretina.eval`:

* **Leave-One-Out (Jackknife) Oracle**: estimates an upper bound on achievable correlation by averaging all but one repeat of the same stimulus before comparison \[Lurz et al., 2022].
* **Fraction of Explainable Variance Explained (FEVE)**: measures the proportion of stimulus-driven variance captured by the model \[Cadena et al., 2019].

These helpers accept the same stimulus/response dictionaries used for training and can be run after `trainer.test` to generate publication-ready scores.

## Putting It Together

Whether you call Lightning manually or rely on the [`openretina train`](./unified_training.md) command, the workflow always follows the same pattern:

1. Prepare data dictionaries and dataloaders.
2. Instantiate a model with the correct input shape and neuron counts.
3. Train with `Trainer.fit(...)` using cyclers to merge session-wise loaders.
4. Evaluate with `Trainer.test(...)` and optional metrics from `openretina.eval`.

Choose this manual approach when you need fine-grained control or are prototyping new trainer settings; otherwise the unified script provides the same functionality with additional convenience features.
