import argparse
import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from openretina.cli import create_data, visualize_model_neurons
from openretina.utils.file_utils import get_cache_directory

log = logging.getLogger(__name__)
logging.captureWarnings(True)


def get_config_path(config_path: str | None) -> str:
    """Get the absolute path to the configs directory"""
    if config_path is None:
        path = str(Path(__file__).parent.parent.parent / "configs")
        if not os.path.isdir(path):
            raise ValueError(
                f"Config directory not found at {path}. Please specify a config directory using '--config-path'"
            )
    else:
        path = str(Path(config_path).absolute())
    return path


class HydraRunner:
    """Wrapper class to run Hydra commands"""

    @staticmethod
    def train(config_path: str, args: list[str]) -> None:
        # Modify sys.argv to work with Hydra
        sys.argv = [sys.argv[0]] + args

        @hydra.main(
            version_base="1.3",
            config_path=get_config_path(config_path),
            config_name="hoefling_2024_core_readout_high_res",
        )
        def _train(cfg: DictConfig) -> float | None:
            from openretina.cli.train import train_model  # Import actual training function

            # Check if cache_dir is set or left as none, in which case we set it to the cache directory
            if cfg.paths.cache_dir is None:
                cfg.paths.cache_dir = get_cache_directory()

            try:
                score = train_model(cfg)
                return score
            except Exception as e:
                # If we are running hyperparameter tuning, we return the worst score on exceptions,
                # otherwise we re-raise the exception
                objective_direction: str = str(cfg.get("objective_direction")).lower()

                if objective_direction == "maximize":
                    score = float("-inf")
                elif objective_direction == "minimize":
                    score = float("inf")
                else:
                    raise e

                log.exception(f"Error during training, {objective_direction=}, {score=}")

        _train()

    @staticmethod
    def eval(config_path: str, args: list[str]) -> None:
        # Modify sys.argv to work with Hydra
        sys.argv = [sys.argv[0]] + args

        @hydra.main(
            version_base="1.3",
            config_path=get_config_path(config_path),
            config_name="hoefling_2024_core_readout_high_res",
        )
        def _eval(cfg: DictConfig) -> float | None:
            from openretina.cli.eval import evaluate_model  # Import actual training function

            # Check if cache_dir is set or left as none, in which case we set it to the cache directory
            if cfg.paths.cache_dir is None:
                cfg.paths.cache_dir = get_cache_directory()

            return evaluate_model(cfg)

        _eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenRetina CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config-path", default=None, help="Path to config directory")

    # Test command
    train_parser = subparsers.add_parser("eval", help="Evaluate a model")
    train_parser.add_argument("--config-path", default=None, help="Path to config directory")

    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize a model")
    visualize_model_neurons.add_parser_arguments(visualize_parser)

    # Create data command
    create_data_parser = subparsers.add_parser("create-data", help="Create artificial data")
    create_data.add_parser_arguments(create_data_parser)

    args, unknown_args = parser.parse_known_args()  # this is an 'internal' method
    command = args.__dict__.pop("command")

    if command == "train":
        return HydraRunner.train(args.config_path, unknown_args)
    if command == "eval":
        return HydraRunner.eval(args.config_path, unknown_args)
    elif command == "visualize":
        if len(unknown_args) > 0:
            print(f"Warn: found the following unknown args: {unknown_args}")
        visualize_model_neurons.visualize_model_neurons(**vars(args))
    elif command == "create-data":
        create_data.write_data_to_directory(**vars(args))
    elif command is None:
        parser.print_help()
    else:
        raise ValueError(f"Unknown command: {command}")
