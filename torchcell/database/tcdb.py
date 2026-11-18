"""The ``tcdb`` command line for the torchcell database, on ``argparse``.

Subcommands ``build``, ``help`` and ``complete`` and the global options (``--version``,
``-v``/``-q``, ``--log-file``, ``--debug``) are the ones the earlier cliff application
exposed, with the same exit codes: the build script's return code, 2 for a usage error,
1 for an error raised by a command or ``help`` of an unknown command. Running ``tcdb``
with no subcommand prints the help (cliff opened an interactive shell instead).
"""

import argparse
import logging
import re
import sys

from torchcell.database import BuildCommand

VERSION = "0.1"
CONSOLE_MESSAGE_FORMAT = "%(message)s"
LOG_FILE_MESSAGE_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s %(message)s"
HELP_DESCRIPTION = "print detailed help for another command"
COMPLETE_DESCRIPTION = "print bash completion command"

log = logging.getLogger(__name__)

_BASH_TRAILER = """
  dash=-
  underscore=_
  cmd=""
  words[0]=""
  completed="${cmds}"
  for var in "${words[@]:1}"
  do
    if [[ ${var} == -* ]] ; then
      break
    fi
    if [ -z "${cmd}" ] ; then
      proposed="${var}"
    else
      proposed="${cmd}_${var}"
    fi
    local i="cmds_${proposed}"
    i=${i//$dash/$underscore}
    local comp="${!i}"
    if [ -z "${comp}" ] ; then
      break
    fi
    if [[ ${comp} == -* ]] ; then
      if [[ ${cur} != -* ]] ; then
        completed=""
        break
      fi
    fi
    cmd="${proposed}"
    completed="${comp}"
  done

  if [ -z "${completed}" ] ; then
    COMPREPLY=( $( compgen -f -- "$cur" ) $( compgen -d -- "$cur" ) )
  else
    COMPREPLY=( $(compgen -W "${completed}" -- ${cur}) )
  fi
  return 0
}
"""


def build_parser() -> tuple[
    argparse.ArgumentParser, dict[str, argparse.ArgumentParser]
]:
    """The ``tcdb`` parser and its subcommand parsers keyed by name."""
    parser = argparse.ArgumentParser(prog="tcdb", description="Database CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="count",
        dest="verbose_level",
        default=1,
        help="Increase verbosity of output. Can be repeated.",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        dest="verbose_level",
        const=0,
        help="Suppress output except warnings and errors.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Specify a file to log output. Disabled by default.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show tracebacks on errors."
    )
    commands = parser.add_subparsers(dest="command", title="Commands")
    build = commands.add_parser(
        "build",
        help="A simple command that builds something",
        description=BuildCommand.description,
    )
    BuildCommand.add_arguments(build)
    complete = commands.add_parser(
        "complete", help=COMPLETE_DESCRIPTION, description=COMPLETE_DESCRIPTION
    )
    complete.add_argument(
        "--name",
        default=None,
        metavar="<command_name>",
        help="Command name to support with command completion",
    )
    complete.add_argument(
        "--shell",
        default="bash",
        metavar="<shell>",
        choices=["bash", "none"],
        help="Shell being used. Use none for data only (default: bash)",
    )
    help_parser = commands.add_parser(
        "help", help=HELP_DESCRIPTION, description=HELP_DESCRIPTION
    )
    help_parser.add_argument("cmd", nargs="*", help="name of the command")
    return parser, {"build": build, "complete": complete, "help": help_parser}


def configure_logging(verbose_level: int, log_file: str | None) -> None:
    """Root logger at DEBUG, a stderr handler by verbosity, and an optional file."""
    root = logging.getLogger("")
    root.setLevel(logging.DEBUG)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(logging.Formatter(LOG_FILE_MESSAGE_FORMAT))
        root.addHandler(file_handler)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(
        {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
            verbose_level, logging.DEBUG
        )
    )
    console.setFormatter(logging.Formatter(CONSOLE_MESSAGE_FORMAT))
    root.addHandler(console)


def completion_data(subparsers: dict[str, argparse.ArgumentParser]) -> list[str]:
    """The ``cmds`` lines of the bash completion: command names, then each one's options."""
    lines = [f"  cmds='{' '.join(sorted(subparsers))}'"]
    for name in sorted(subparsers):
        options = [
            option
            for action in subparsers[name]._actions
            for option in action.option_strings
        ]
        lines.append(f"  cmds_{name}='{' '.join(options)}'")
    return lines


def bash_completion(name: str, subparsers: dict[str, argparse.ArgumentParser]) -> str:
    """A bash completion function for ``name`` (the script cliff's ``complete`` printed)."""
    escaped = re.sub(r"[^A-Za-z0-9_]", "_", name)
    header = (
        f"_{escaped}()\n{{\n  local cur prev words\n  COMPREPLY=()\n"
        "  _get_comp_words_by_ref -n : cur prev words\n\n  # Command data:\n"
    )
    data = "\n".join(completion_data(subparsers)) + "\n"
    return f"{header}{data}{_BASH_TRAILER}complete -F _{escaped} {name}\n"


def main(argv: list[str] | None = None) -> int:
    """Run the database CLI and return its exit code."""
    parser, subparsers = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    configure_logging(args.verbose_level, args.log_file)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "help":
        if not args.cmd:
            parser.print_help()
            return 0
        if args.cmd[0] not in subparsers:
            log.error(f"Unknown command {args.cmd}")
            return 1
        subparsers[args.cmd[0]].print_help()
        return 0
    if args.command == "complete":
        if args.shell == "none":
            print("\n".join(completion_data(subparsers)))
        else:
            print(bash_completion(args.name or "tcdb", subparsers), end="")
        return 0
    # The error boundary cliff provided: a failing command logs its message and exits 1;
    # ``--debug`` adds the traceback.
    try:
        return BuildCommand().take_action(args) or 0
    except Exception as err:
        if args.debug:
            log.exception(err)
        else:
            log.error(err)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
