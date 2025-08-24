import argparse
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from openretina.cli import create_data, visualize_model_neurons
from openretina.utils.file_utils import get_cache_directory


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

            return train_model(cfg)

        _train()

    @staticmethod
    def distributed_train(config_path: str, args: list[str]) -> None:
        # Set an environment variable to indicate we're in distributed training mode
        # This helps child processes bypass CLI argument parsing
        os.environ["OPENRETINA_DISTRIBUTED_MODE"] = "1"
        os.environ["OPENRETINA_CONFIG_PATH"] = get_config_path(config_path)
        os.environ["OPENRETINA_HYDRA_ARGS"] = " ".join(args)

        # Modify sys.argv to work with Hydra (only script name + hydra args)
        sys.argv = [sys.argv[0]] + args

        @hydra.main(
            version_base="1.3",
            config_path=get_config_path(config_path),
            config_name="hoefling_2024_distributed_example",
        )
        def _distributed_train(cfg: DictConfig) -> float | None:
            from openretina.cli.distributed_train import train_model  # Import actual distributed training function

            # Check if cache_dir is set or left as none, in which case we set it to the cache directory
            if cfg.paths.cache_dir is None:
                cfg.paths.cache_dir = get_cache_directory()

            return train_model(cfg)

        _distributed_train()


def main() -> None:
    # Check if we're in a distributed training child process
    if os.environ.get("OPENRETINA_DISTRIBUTED_MODE") == "1":
        # We're in a child process of distributed training
        # Use the stored configuration and args instead of parsing CLI
        config_path = os.environ.get("OPENRETINA_CONFIG_PATH")
        hydra_args = os.environ.get("OPENRETINA_HYDRA_ARGS", "").split()

        # Set sys.argv for Hydra
        sys.argv = [sys.argv[0]] + hydra_args

        @hydra.main(
            version_base="1.3",
            config_path=config_path,
            config_name="hoefling_2024_distributed_example",
        )
        def _child_distributed_train(cfg: DictConfig) -> float | None:
            from openretina.cli.distributed_train import train_model

            if cfg.paths.cache_dir is None:
                cfg.paths.cache_dir = get_cache_directory()

            return train_model(cfg)

        return _child_distributed_train()

    parser = argparse.ArgumentParser(description="OpenRetina CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config-path", default=None, help="Path to config directory")

    # Distributed train command
    distributed_train_parser = subparsers.add_parser(
        "distributed-train", help="Train a model with distributed training"
    )
    distributed_train_parser.add_argument("--config-path", default=None, help="Path to config directory")

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
    elif command == "distributed-train":
        return HydraRunner.distributed_train(args.config_path, unknown_args)
    elif command == "visualize":
        if len(unknown_args) > 0:
            print(f"Warn: found the following unknown args: {unknown_args}")
        visualize_model_neurons.visualize_model_neurons(**vars(args))
    elif command == "create-data":
        create_data.write_data_to_directory(**vars(args))
    elif command is None:
        parser.print_help()
    else:
        raise ValueError(f"Unknown command: {args.command}")
