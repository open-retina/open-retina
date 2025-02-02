import argparse
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from openretina.cli import visualize_model_neurons
from openretina.utils.file_utils import OPENRETINA_CACHE_DIRECTORY


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
        # check if data.root if in argument, otherwise set it to openretina cache dir
        if not any("data.root_dir" in x for x in args):
            args.append(f"data.root_dir='{OPENRETINA_CACHE_DIRECTORY}'")
        # Modify sys.argv to work with Hydra
        sys.argv = [sys.argv[0]] + args

        @hydra.main(
            version_base="1.3",
            config_path=get_config_path(config_path),
            config_name="hoefling_2024_core_readout_high_res",
        )
        def _train(cfg: DictConfig) -> None:
            from openretina.cli.train import train_model  # Import actual training function

            train_model(cfg)

        _train()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenRetina CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config-path", default=None, help="Path to config directory")

    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize a model")
    visualize_model_neurons.add_parser_arguments(visualize_parser)

    args, unknown_args = parser.parse_known_args()  # this is an 'internal' method
    command = args.__dict__.pop("command")

    if command == "train":
        HydraRunner.train(args.config_path, unknown_args)
    elif command == "visualize":
        if len(unknown_args) > 0:
            print(f"Warn: found the following unknown args: {unknown_args}")
        visualize_model_neurons.visualize_model_neurons(**vars(args))
    elif command is None:
        parser.print_help()
    else:
        raise ValueError(f"Unknown command: {args.command}")
