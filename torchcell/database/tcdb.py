# db_cli.py
from cliff.app import App
from cliff.commandmanager import CommandManager
from torchcell.database import BuildCommand  # Import the command


class TCDB(App):

    def __init__(self):
        super(TCDB, self).__init__(
            description="Database CLI",
            version="0.1",
            command_manager=CommandManager("db.cli"),
            deferred_help=True,
        )
        self.command_manager.add_command("build", BuildCommand)


def main(argv=None):
    myapp = TCDB()
    return myapp.run(argv)


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
