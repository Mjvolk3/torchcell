# tests/torchcell/database/test_tcdb.py
# [[tests.torchcell.database.test_tcdb]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/database/test_tcdb.py
"""The ``tcdb`` cliff application (the ``tcdb`` console script in pyproject).

Fixture: the working directory is ``tmp_path`` holding an empty
``database/build/build_linux-arm.sh``, and ``subprocess.Popen`` returns a recorder with
empty output and a scripted return code, and the root logger's handlers and level are restored after
each test because ``App.run`` configures logging. No script runs.
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

from torchcell.database import BuildCommand
from torchcell.database.tcdb import TCDB, main


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_tcdb_registers_build_help_and_complete() -> None:
    app = TCDB()
    assert app.command_manager.find_command(["build"]) == (BuildCommand, "build", [])
    assert sorted(name for name, _ in app.command_manager) == [
        "build",
        "complete",
        "help",
    ]
    assert app.parser.description == "Database CLI"


def test_main_runs_build_and_returns_its_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launched: list[list[str]] = []
    build = tmp_path / "database" / "build"
    build.mkdir(parents=True)
    (build / "build_linux-arm.sh").write_text("#!/bin/sh\n")

    class _Proc:
        def __init__(self, args: list[str], **kwargs: Any) -> None:
            launched.append(args)
            self.stdout = io.StringIO("")

        def poll(self) -> int:
            return 7

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", _Proc)
    assert main(["build", "--mode", "regular"]) == 7
    script = f"{tmp_path}/database/build/build_linux-arm.sh"
    assert launched == [[script]]
    assert stat.S_IMODE((build / "build_linux-arm.sh").stat().st_mode) == 0o755
