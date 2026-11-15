"""Training entry point."""

import argparse

from .utils import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a predictive maintenance model")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    raise NotImplementedError(
        "Connect data loading and the training loop after selecting a dataset format."
    )


if __name__ == "__main__":
    main()
