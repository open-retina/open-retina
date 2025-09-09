import inspect
import os
from contextlib import contextmanager

import torch
from lightning.pytorch.callbacks import Callback


class ConditionalWeightVisualizationCallback(Callback):
    """
    PyTorch Lightning callback that calls the model's save_weight_visualizations
    function conditionally based on epoch settings.
    """

    def __init__(
        self,
        every_n_epochs: int = 1,
        start_epoch: int = 0,
        call_on_train_epoch_end: bool = True,
        call_on_validation_epoch_end: bool = False,
        save_plots: bool = False,
        folder_path: str | None = None,
    ):
        """
        Args:
            every_n_epochs (int): Visualize every N epochs
            start_epoch (int): Start visualizing from this epoch
            call_on_train_epoch_end (bool): Whether to call after training epoch
            call_on_validation_epoch_end (bool): Whether to call after validation epoch
            save_plots (bool): Whether to save plots
            folder_path (str): Directory to save plots (if save_plots=True)
        """
        super().__init__()  # Explicit parent call
        self.every_n_epochs = every_n_epochs
        self.start_epoch = start_epoch
        self.call_on_train_epoch_end = call_on_train_epoch_end
        self.call_on_validation_epoch_end = call_on_validation_epoch_end
        self.save_plots = save_plots
        self.folder_path = folder_path

    def _should_visualize(self, current_epoch: int) -> bool:
        """Check if we should visualize at the current epoch."""
        return current_epoch >= self.start_epoch and current_epoch % self.every_n_epochs == 0

    def _call_visualization(self, trainer, pl_module, stage=""):
        """Helper method to call the visualization function."""
        if not hasattr(pl_module, "save_weight_visualizations"):
            print("Warning: Model does not have 'save_weight_visualizations' method")
            return

        try:
            kwargs = {}

            # Handle folder_path if requested
            if self.save_plots and self.folder_path:
                os.makedirs(self.folder_path, exist_ok=True)

                # Check if the function accepts folder_path parameter
                sig = inspect.signature(pl_module.save_weight_visualizations)
                if "folder_path" in sig.parameters:
                    kwargs["folder_path"] = self.folder_path
                if "state_suffix" in sig.parameters:
                    kwargs["state_suffix"] = f"{stage}_epoch_{trainer.current_epoch}"
                else:
                    print(
                        "Weights visualizations will only save the last epoch. "
                        "For epoch by epoch visualization, implement state_suffix in the viz function"
                    )
            pl_module.save_weight_visualizations(**kwargs)

        except Exception as e:
            print(f"Warning: Failed to plot weight visualization at epoch {trainer.current_epoch}: {e}")

    def on_train_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            # print("Running weight visualization on train start")
            self._call_visualization(trainer, pl_module, "init")

    def on_train_epoch_end(self, trainer, pl_module):
        if self.call_on_train_epoch_end and self._should_visualize(trainer.current_epoch):
            self._call_visualization(trainer, pl_module, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.call_on_validation_epoch_end and self._should_visualize(trainer.current_epoch):
            self._call_visualization(trainer, pl_module, "val")
