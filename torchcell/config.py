# torchcell/config
# [[torchcell.config]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/config
# Test file: tests/torchcell/test_config.py

import argparse
import logging
import os.path as osp

import yaml

log = logging.getLogger(__name__)


def main(sweep_file: str) -> str:
    with open(sweep_file, "r") as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)
    return sweep_config["project"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_name", help="The name of the sweep file to open. Include .yaml extension"
    )
    args = parser.parse_args()
    print(main(args.file_name))
