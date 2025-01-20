import argparse
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


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
            from openretina_scripts.train import train_model  # Import actual training function

            train_model(cfg)

        _train()


def main():
    parser = argparse.ArgumentParser(description="OpenRetina CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("hydra_args", nargs="*", help="Hydra arguments")

    args = parser.parse_args()

    if args.command == "train":
        HydraRunner.train(args.hydra_args)
    elif args.command is None:
        parser.print_help()
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
