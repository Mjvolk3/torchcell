# torchcell/config
# [[torchcell.config]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/config
# Test file: tests/torchcell/test_config.py

"""Read a wandb sweep YAML and expose its project name."""

import argparse
import logging

import yaml

log = logging.getLogger(__name__)


def main(sweep_file: str) -> str:
    """Load the sweep YAML file and return its ``project`` value."""
    with open(sweep_file) as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)
    project: str = sweep_config["project"]
    return project


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_name", help="The name of the sweep file to open. Include .yaml extension"
    )
    args = parser.parse_args()
    print(main(args.file_name))
