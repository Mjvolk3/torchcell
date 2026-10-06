# tests/torchcell/database/test_build_command.py
# [[tests.torchcell.database.test_build_command]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/database/test_build_command.py
"""The cliff ``build`` command with the build script stubbed.

Fixture: the working directory is ``tmp_path`` (``monkeypatch.chdir``), which
holds ``database/build/`` with empty files named exactly as the repo's
``database/build/`` directory names them (read from the checkout, nothing written
there). ``subprocess.Popen`` is replaced at its import site by a recorder whose stdout
is a scripted ``StringIO`` and whose ``poll`` returns the scripted code once stdout is
drained. No script runs.

Finding pinned: ``--mode fresh`` names ``build_image_fresh_linux-arm.sh`` but the repo
file is ``build-image-fresh_linux-arm.sh``, so fresh mode fails before running anything.
"""

from __future__ import annotations

import io
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from torchcell.database.build_command import BuildCommand

REPO = Path(__file__).resolve().parents[3]


class _FakePopen:
    instances: list[_FakePopen] = []

    def __init__(self, args: list[str], **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.stdout = io.StringIO(self.script_output)
        self.polls = 0
        _FakePopen.instances.append(self)

    script_output = ""
    return_code = 0

    def poll(self) -> int | None:
        self.polls += 1
        if self.stdout.tell() < len(self.stdout.getvalue()):
            return None
        return self.return_code


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    build = tmp_path / "database" / "build"
    build.mkdir(parents=True)
    for name in sorted(os.listdir(REPO / "database" / "build")):
        (build / name).write_text("#!/bin/sh\n")
        (build / name).chmod(0o644)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    _FakePopen.instances = []
    return tmp_path


def test_parser_mode_default_and_choices(capsys: pytest.CaptureFixture[str]) -> None:
    parser = BuildCommand(None, None).get_parser("tcdb build")  # type: ignore[arg-type, unused-ignore]
    assert parser.parse_args([]).mode == "regular"
    assert parser.parse_args(["-m", "fresh"]).mode == "fresh"
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "warm"])
    assert capsys.readouterr().err == (
        "usage: tcdb build [-h] [-m {regular,fresh}]\n"
        "tcdb build: error: argument -m/--mode: invalid choice: 'warm' "
        "(choose from 'regular', 'fresh')\n"
    )


def test_regular_mode_runs_the_arm_script_and_streams_output(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The script is made 0755, launched once with shell=True and stderr merged into
    stdout, every line printed stripped (a blank line read as a lone newline is truthy, so it prints
    as an empty line), and the poll code returned.
    """
    monkeypatch.setattr(_FakePopen, "script_output", "  step 1  \n\nstep 2\n")
    monkeypatch.setattr(_FakePopen, "return_code", 3)
    script = workspace / "database" / "build" / "build_linux-arm.sh"
    cmd = BuildCommand(None, None)  # type: ignore[arg-type, unused-ignore]
    rc = cmd.take_action(cmd.get_parser("b").parse_args([]))
    assert rc == 3
    assert stat.S_IMODE(script.stat().st_mode) == 0o755
    (proc,) = _FakePopen.instances
    assert proc.args == [str(script)]
    assert proc.kwargs == {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "shell": True,
        "text": True,
    }
    assert capsys.readouterr().out == (
        f"Regular mode: Executing script {script}\nstep 1\n\nstep 2\n"
    )


def test_fresh_mode_names_a_script_the_repo_does_not_have(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: fresh mode builds ``build_image_fresh_linux-arm.sh`` (underscores);
    the repo's script is ``build-image-fresh_linux-arm.sh`` (hyphens), so ``os.chmod``
    raises before any process starts and ``tcdb build --mode fresh`` cannot run.
    Pinned until the name matches the file (build_command.py:36).
    """
    names = set(os.listdir(REPO / "database" / "build"))
    assert "build-image-fresh_linux-arm.sh" in names
    assert "build_image_fresh_linux-arm.sh" not in names
    missing = workspace / "database" / "build" / "build_image_fresh_linux-arm.sh"
    cmd = BuildCommand(None, None)  # type: ignore[arg-type, unused-ignore]
    with pytest.raises(
        FileNotFoundError, match=re.escape(f"No such file or directory: '{missing}'")
    ):
        cmd.take_action(cmd.get_parser("b").parse_args(["--mode", "fresh"]))
    assert _FakePopen.instances == []
    assert capsys.readouterr().out == f"Fresh mode: Executing script {missing}\n"
