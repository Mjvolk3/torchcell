"""Cliff command for building the torchcell database image via shell scripts."""

import argparse
import os
import os.path as osp
import subprocess
from typing import IO, TYPE_CHECKING, cast

from cliff.command import Command

if TYPE_CHECKING:
    from cliff._argparse import ArgumentParser as CliffArgumentParser


class BuildCommand(Command):
    """A simple command that builds something."""

    def get_parser(self, prog_name: str) -> "CliffArgumentParser":
        """Build the argument parser, adding the build mode option."""
        parser = super().get_parser(prog_name)
        parser.add_argument(
            "-m",
            "--mode",
            choices=["regular", "fresh"],
            default="regular",
            help="Build mode (default: %(default)s)",
        )
        return parser

    def take_action(self, parsed_args: argparse.Namespace) -> int | None:
        """Run the selected build script and stream its output."""
        script_path = osp.join(os.getcwd(), "database", "build")
        script = (
            "build_linux-arm.sh"
            if parsed_args.mode == "regular"
            else "build_image_fresh_linux-arm.sh"
        )
        full_script_path = osp.join(script_path, script)

        print(
            f"{parsed_args.mode.capitalize()} mode: Executing script {full_script_path}"
        )

        # Ensure the script is executable
        os.chmod(full_script_path, 0o755)

        # Start the process
        process = subprocess.Popen(
            [full_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
        )

        # Stream the output
        stdout = cast(IO[str], process.stdout)
        while True:
            output = stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())
        rc = process.poll()
        return rc
