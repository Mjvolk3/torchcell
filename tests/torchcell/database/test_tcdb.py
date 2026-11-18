# tests/torchcell/database/test_tcdb.py
# [[tests.torchcell.database.test_tcdb]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/database/test_tcdb.py
"""The ``tcdb`` argparse command line (the ``tcdb`` console script in pyproject).

Fixture: the working directory is ``tmp_path`` holding an empty
``database/build/build_linux-arm.sh``, and ``subprocess.Popen`` returns a recorder with
empty output and a scripted return code. The root logger's handlers and level are
restored after each test because ``main`` configures logging. No script runs.

The bash completion text is compared with the script cliff's ``complete`` printed for
the same commands (captured 2026-11-17 before the move to argparse): the data lines are
asserted verbatim and the function header and footer by their exact lines.
"""

from __future__ import annotations

import io
import logging
import stat
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from torchcell.database.tcdb import build_parser, main


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def build_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    build = tmp_path / "database" / "build"
    build.mkdir(parents=True)
    (build / "build_linux-arm.sh").write_text("#!/bin/sh\n")
    monkeypatch.chdir(tmp_path)
    return build


def _fake_popen(monkeypatch: pytest.MonkeyPatch, code: int) -> list[list[str]]:
    launched: list[list[str]] = []

    class _Proc:
        def __init__(self, args: list[str], **kwargs: Any) -> None:
            launched.append(args)
            self.stdout = io.StringIO("")

        def poll(self) -> int:
            return code

    monkeypatch.setattr(subprocess, "Popen", _Proc)
    return launched


def test_parser_has_build_complete_help_and_the_global_options() -> None:
    parser, subparsers = build_parser()
    assert sorted(subparsers) == ["build", "complete", "help"]
    assert parser.description == "Database CLI"
    assert parser.prog == "tcdb"
    args = parser.parse_args(["-v", "-v", "--log-file", "x.log", "--debug", "build"])
    assert (args.verbose_level, args.log_file, args.debug, args.command, args.mode) == (
        3,
        "x.log",
        True,
        "build",
        "regular",
    )
    assert parser.parse_args(["-q", "build", "-m", "fresh"]).verbose_level == 0
    assert parser.parse_args(["build"]).verbose_level == 1


def test_main_runs_build_and_returns_its_code(
    monkeypatch: pytest.MonkeyPatch, build_dir: Path
) -> None:
    launched = _fake_popen(monkeypatch, 7)
    assert main(["build", "--mode", "regular"]) == 7
    script = f"{build_dir}/build_linux-arm.sh"
    assert launched == [[script]]
    assert stat.S_IMODE((build_dir / "build_linux-arm.sh").stat().st_mode) == 0o755


def test_a_zero_build_code_returns_zero(
    monkeypatch: pytest.MonkeyPatch, build_dir: Path
) -> None:
    _fake_popen(monkeypatch, 0)
    assert main(["build"]) == 0


def test_a_failing_command_logs_its_message_and_returns_one(
    monkeypatch: pytest.MonkeyPatch, build_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fresh mode names a script that does not exist: the message only, exit 1."""
    missing = build_dir / "build_image_fresh_linux-arm.sh"
    assert main(["build", "--mode", "fresh"]) == 1
    err = capsys.readouterr().err
    assert err == f"[Errno 2] No such file or directory: '{missing}'\n"


def test_debug_adds_the_traceback(
    monkeypatch: pytest.MonkeyPatch, build_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--debug", "build", "--mode", "fresh"]) == 1
    err = capsys.readouterr().err.splitlines()
    assert err[1] == "Traceback (most recent call last):"
    assert err[-1].startswith("FileNotFoundError: [Errno 2] No such file or directory")


def test_usage_errors_exit_two(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as unknown:
        main(["nosuch"])
    with pytest.raises(SystemExit) as bad_mode:
        main(["build", "--mode", "warm"])
    assert (unknown.value.code, bad_mode.value.code) == (2, 2)
    err = capsys.readouterr().err
    assert "tcdb: error: argument command: invalid choice: 'nosuch'" in err
    assert (
        "tcdb build: error: argument -m/--mode: invalid choice: 'warm' "
        "(choose from 'regular', 'fresh')"
    ) in err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == "tcdb 0.1\n"


def test_help_prints_the_top_level_or_one_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["help"]) == 0
    assert capsys.readouterr().out.startswith("usage: tcdb [-h] [--version]")
    assert main(["help", "build"]) == 0
    assert capsys.readouterr().out == (
        "usage: tcdb build [-h] [-m {regular,fresh}]\n\n"
        "A simple command that builds something.\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  -m, --mode {regular,fresh}\n"
        "                        Build mode (default: regular)\n"
    )
    assert main([]) == 0
    assert capsys.readouterr().out.startswith("usage: tcdb [-h] [--version]")


def test_help_of_an_unknown_command_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["help", "nosuch"]) == 1
    assert capsys.readouterr().err == "Unknown command ['nosuch']\n"


_DATA = [
    "  cmds='build complete help'",
    "  cmds_build='-h --help -m --mode'",
    "  cmds_complete='-h --help --name --shell'",
    "  cmds_help='-h --help'",
]


def test_complete_none_prints_the_command_data_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["complete", "--shell", "none"]) == 0
    assert capsys.readouterr().out == "\n".join(_DATA) + "\n"


def test_complete_bash_wraps_the_data_in_the_completion_function(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["complete"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:7] == [
        "_tcdb()",
        "{",
        "  local cur prev words",
        "  COMPREPLY=()",
        "  _get_comp_words_by_ref -n : cur prev words",
        "",
        "  # Command data:",
    ]
    assert lines[7:11] == _DATA
    assert lines[-3:] == ["  return 0", "}", "complete -F _tcdb tcdb"]
    assert len(lines) == 51  # the line count of cliff's script
    assert main(["complete", "--name", "tc-db"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "complete -F _tc_db tc-db"
