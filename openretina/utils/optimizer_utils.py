"""
Utility functions for configuring optimizers and learning rate schedulers with Hydra.

This module provides helper functions to instantiate PyTorch optimizers and schedulers
from Hydra DictConfig objects, with special handling for PyTorch Lightning integration
and schedulers with unique requirements (e.g., OneCycleLR).

These utilities are used by UnifiedCoreReadout to enable configurable optimization
strategies through Hydra configs.
"""

import logging
from typing import Any

import hydra.utils
import torch
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


def instantiate_optimizer(
    optimizer_config: DictConfig | None,
    model_parameters: Any,
    default_learning_rate: float,
) -> torch.optim.Optimizer:
    """
    Instantiate an optimizer from a Hydra config or use default AdamW.

    Args:
        optimizer_config: Hydra DictConfig for optimizer instantiation. If None, defaults to AdamW.
        model_parameters: Model parameters to optimize (typically from model.parameters()).
        default_learning_rate: Learning rate to use if no config is provided.

    Returns:
        Instantiated PyTorch optimizer.

    Example:
        >>> optimizer_config = OmegaConf.create({
        ...     "_target_": "torch.optim.SGD",
        ...     "_partial_": True,
        ...     "lr": 0.01,
        ...     "momentum": 0.9,
        ... })
        >>> optimizer = instantiate_optimizer(optimizer_config, model.parameters(), 0.01)
    """
    if optimizer_config is not None:
        optimizer = hydra.utils.instantiate(optimizer_config, params=model_parameters)
    else:
        optimizer = torch.optim.AdamW(model_parameters, lr=default_learning_rate)

    return optimizer


def instantiate_scheduler(
    scheduler_config: DictConfig | None,
    optimizer: torch.optim.Optimizer,
    default_learning_rate: float,
    trainer: Any | None = None,
) -> dict[str, Any]:
    """
    Instantiate a learning rate scheduler from a Hydra config or use default ReduceLROnPlateau.

    This function handles:
    - Default ReduceLROnPlateau scheduler if no config provided
    - Extraction of PyTorch Lightning-specific parameters (monitor, interval, frequency)
    - Special handling for OneCycleLR which requires total_steps
    - Computing min_lr for ReduceLROnPlateau if not explicitly set

    Args:
        scheduler_config: Hydra DictConfig for scheduler instantiation. If None, defaults to ReduceLROnPlateau.
        optimizer: The optimizer instance to attach the scheduler to.
        default_learning_rate: Learning rate used for computing default min_lr.
        trainer: Optional PyTorch Lightning trainer instance (used for OneCycleLR total_steps).

    Returns:
        Dictionary with "scheduler" and PyTorch Lightning configuration, or None if no scheduler.
        Format: {
            "scheduler": scheduler_instance,
            "monitor": metric_name,
            "interval": "epoch" or "step",
            "frequency": int,
        }

    Example:
        >>> scheduler_config = OmegaConf.create({
        ...     "_target_": "torch.optim.lr_scheduler.StepLR",
        ...     "_partial_": True,
        ...     "step_size": 10,
        ...     "gamma": 0.1,
        ...     "monitor": None,
        ...     "interval": "epoch",
        ...     "frequency": 1,
        ... })
        >>> scheduler_dict = instantiate_scheduler(scheduler_config, optimizer, 0.01)
    """
    # If no scheduler config provided, use default ReduceLROnPlateau
    if scheduler_config is None:
        return _create_default_scheduler(optimizer, default_learning_rate)

    # Use configured scheduler
    scheduler_config_dict: dict[str, Any] = dict(scheduler_config)  # type: ignore[assignment]

    # Extract PyTorch Lightning-specific configuration
    monitor = scheduler_config_dict.pop("monitor", "val_correlation")
    interval = scheduler_config_dict.pop("interval", "epoch")
    frequency = scheduler_config_dict.pop("frequency", 1)

    # Handle special cases for schedulers with unique requirements
    _handle_special_scheduler_cases(scheduler_config_dict, trainer, default_learning_rate)

    # Instantiate the scheduler
    scheduler = hydra.utils.instantiate(scheduler_config_dict, optimizer=optimizer)

    return {
        "scheduler": scheduler,
        "monitor": monitor,
        "interval": interval,
        "frequency": frequency,
    }


def _create_default_scheduler(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> dict[str, Any]:
    """
    Create default ReduceLROnPlateau scheduler with standard settings.

    Args:
        optimizer: The optimizer instance.
        learning_rate: Base learning rate for computing min_lr.

    Returns:
        Dictionary with scheduler and Lightning configuration.
    """
    lr_decay_factor = 0.3
    patience = 5
    tolerance = 0.0005
    min_lr = learning_rate * (lr_decay_factor**3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        threshold_mode="abs",
        min_lr=min_lr,
    )

    return {
        "scheduler": scheduler,
        "monitor": "val_correlation",
        "frequency": 1,
    }


def _handle_special_scheduler_cases(
    scheduler_config: dict[str, Any],
    trainer: Any | None,
    learning_rate: float,
) -> None:
    """
    Handle special cases for schedulers with unique requirements.

    Currently handles:
    - OneCycleLR: Requires total_steps or steps_per_epoch + epochs
    - ReduceLROnPlateau: Computes min_lr if not set

    Args:
        scheduler_config: Mutable scheduler config dictionary.
        trainer: Optional PyTorch Lightning trainer instance.
        learning_rate: Learning rate for computing min_lr.

    Note:
        This function modifies scheduler_config in-place.
    """
    if "_target_" not in scheduler_config:
        return

    target = scheduler_config["_target_"]

    # OneCycleLR needs total_steps or steps_per_epoch + epochs
    if "OneCycleLR" in target:
        _handle_one_cycle_lr(scheduler_config, trainer)

    # ReduceLROnPlateau: compute min_lr if not explicitly set
    elif "ReduceLROnPlateau" in target:
        _handle_reduce_lr_on_plateau(scheduler_config, learning_rate)


def _handle_one_cycle_lr(
    scheduler_config: dict[str, Any],
    trainer: Any | None,
) -> None:
    """
    Handle OneCycleLR scheduler which requires total_steps.

    Args:
        scheduler_config: Mutable scheduler config dictionary.
        trainer: Optional PyTorch Lightning trainer instance.

    Note:
        This function modifies scheduler_config in-place.
    """
    # Check if total_steps or steps_per_epoch is already provided
    if "total_steps" in scheduler_config or "steps_per_epoch" in scheduler_config:
        return

    # Try to get total_steps from trainer (configure_optimizers() is called after trainer is initialized usually)
    if trainer is not None and hasattr(trainer, "estimated_stepping_batches"):
        scheduler_config["total_steps"] = trainer.estimated_stepping_batches
        LOGGER.info(f"OneCycleLR: Automatically set total_steps={scheduler_config['total_steps']} from trainer")
    else:
        LOGGER.warning(
            "OneCycleLR requires 'total_steps' or 'steps_per_epoch' + 'epochs'. "
            "These were not found in config and could not be determined from trainer. "
            "The scheduler may fail to instantiate. Consider setting these explicitly in your config."
        )


def _handle_reduce_lr_on_plateau(
    scheduler_config: dict[str, Any],
    learning_rate: float,
) -> None:
    """
    Handle ReduceLROnPlateau scheduler - compute min_lr if not explicitly set.

    The default behavior is: min_lr = learning_rate * (factor ** 3)
    This matches the original hardcoded behavior.

    Args:
        scheduler_config: Mutable scheduler config dictionary.
        learning_rate: Learning rate for computing min_lr.

    Note:
        This function modifies scheduler_config in-place.
    """
    # Only compute min_lr if not already set
    if "min_lr" in scheduler_config:
        return

    # Get the factor from config, default to 0.3
    factor = scheduler_config.get("factor", 0.3)

    # Compute min_lr: learning_rate * (factor ** 3)
    min_lr = learning_rate * (factor**3)
    scheduler_config["min_lr"] = min_lr

    LOGGER.debug(f"ReduceLROnPlateau: Computed min_lr={min_lr:.2e} from lr={learning_rate} and factor={factor}")
