import argparse
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from openretina.cli import visualize_model_neurons


def get_config_path():
    """Get the absolute path to the configs directory"""
    import openretina

    return os.path.join(Path(openretina.__file__).parent.parent, "configs")


class HydraRunner:
    """Wrapper class to run Hydra commands"""

    @staticmethod
    def train(args):
        # Modify sys.argv to work with Hydra
        sys.argv = [sys.argv[0]] + args

        @hydra.main(
            version_base="1.3", config_path=get_config_path(), config_name="hoefling_2024_core_readout_high_res"
        )
        def _train(cfg: DictConfig) -> None:
            from openretina.cli.train import train_model  # Import actual training function

            train_model(cfg)

        _train()


def main():
    parser = argparse.ArgumentParser(description="OpenRetina CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("hydra_args", nargs="*", help="Hydra arguments")

    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize a model")
    visualize_model_neurons.add_parser_arguments(visualize_parser)

    args = parser.parse_args()
    command = args.__dict__.pop("command")

    if command == "train":
        HydraRunner.train(args.hydra_args)
    elif command == "visualize":
        visualize_model_neurons.visualize_model_neurons(**vars(args))
    elif command is None:
        parser.print_help()
    else:
        raise ValueError(f"Unknown command: {args.command}")
