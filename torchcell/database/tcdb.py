"""Cliff-based command-line entry point for the torchcell database."""

# db_cli.py
from typing import cast

from cliff.app import App
from cliff.commandmanager import CommandManager

from torchcell.database import BuildCommand  # Import the command


class TCDB(App):
    """Cliff application exposing the torchcell database build commands."""

    def __init__(self) -> None:
        """Configure the CLI app and register the ``build`` command."""
        super().__init__(
            description="Database CLI",
            version="0.1",
            command_manager=CommandManager("db.cli"),
            deferred_help=True,
        )
        self.command_manager.add_command("build", BuildCommand)


def main(argv: list[str] | None = None) -> int:
    """Run the database CLI application and return its exit code."""
    myapp = TCDB()
    return myapp.run(cast("list[str]", argv))


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
