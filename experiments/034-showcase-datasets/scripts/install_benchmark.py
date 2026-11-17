# experiments/034-showcase-datasets/scripts/install_benchmark.py
# [[experiments.034-showcase-datasets.scripts.install_benchmark]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/034-showcase-datasets/scripts/install_benchmark
"""Time a first install of torchcell from PyPI with pip and with uv, in fresh venvs.

Standard library only and Python 3.9 compatible, because it runs on a machine that has
nothing installed yet (the Radiant VM ships Python 3.9 and no pip). Everything it writes
stays under one work directory: the uv binary, the managed Python 3.13, both package
caches, the temporary directory and one venv per arm. Nothing touches ``~/.cache``,
``~/.local`` or ``/tmp``, so a small root disk is not filled and both tools do their I/O
on the same filesystem.

Arms, run in this order so the two tools alternate:

- ``pip_cold``: ``pip install <requirement>`` with an empty pip cache.
- ``uv_cold``: ``uv pip install <requirement>`` with an empty uv cache.
- ``pip_warm``: the same pip install into a second venv, the cache now filled.
- ``uv_warm``: the same uv install into a second venv, the cache now filled.
- ``uv_cpu_cold``: ``uv pip install --torch-backend cpu <requirement>`` with its own
  empty cache; the CPU build of torch, for a machine with no GPU.
- ``pip_cpu_cold``: ``pip install --extra-index-url <pytorch cpu index> <requirement>``
  with its own empty cache; pip's way to the same CPU build.

For a uv arm the result also holds uv's own split of the install (packages and time
to resolve, to prepare, which is download and unpack, and to install), read from the
closing lines of its log.

Every arm gets a venv made by ``uv venv --python 3.13`` (``--seed`` for the pip arms, so
the venv holds pip), and the time to make it is recorded apart from the install. An arm
that fails is recorded with its exit code and its log, and the run goes on to the next
arm; nothing is retried. The work directory must not already hold a venv or a cache,
because a leftover cache would make a cold arm warm.

Usage::

    python3 install_benchmark.py --work <empty dir> [--requirement torchcell]

Writes ``<work>/install_benchmark.json`` and one log per arm under ``<work>/logs/``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UV_URL = (
    "https://github.com/astral-sh/uv/releases/latest/download/"
    "uv-x86_64-unknown-linux-gnu.tar.gz"
)
PYTHON_VERSION = "3.13"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# uv's own closing lines: "Resolved 258 packages in 12.91s", "Prepared ...", "Installed ...".
UV_PHASE = re.compile(r"^(Resolved|Prepared|Installed) (\d+) packages? in (.+)$")
# (arm, tool, cache family, extra install arguments)
ARMS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("pip_cold", "pip", "default", ()),
    ("uv_cold", "uv", "default", ()),
    ("pip_warm", "pip", "default", ()),
    ("uv_warm", "uv", "default", ()),
    ("uv_cpu_cold", "uv", "cpu", ("--torch-backend", "cpu")),
    ("pip_cpu_cold", "pip", "cpu", ("--extra-index-url", TORCH_CPU_INDEX)),
)
PROBE = """
import importlib.metadata as md, json, time
started = time.perf_counter()
import torchcell, torch, numpy, pandas
import torchcell.datasets.ecoli.campos2018
seconds = time.perf_counter() - started
print(json.dumps({
    "torchcell": torchcell.__version__, "torch": torch.__version__,
    "numpy": numpy.__version__, "pandas": pandas.__version__,
    "cuda_build": torch.version.cuda,
    "n_distributions": len({d.metadata["Name"] for d in md.distributions()}),
    "import_seconds": round(seconds, 2),
}))
"""


def run(command: list[str], env: dict[str, str], log: Path) -> tuple[int, float]:
    """Run ``command`` with its output appended to ``log``; return (exit code, seconds)."""
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        started = time.perf_counter()
        code = subprocess.call(
            command, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        seconds = time.perf_counter() - started
        handle.write("[exit %d in %.1f s]\n" % (code, seconds))
    return code, seconds


def output(command: list[str], env: dict[str, str]) -> str:
    """Standard output of ``command``, stripped; raises when it exits non-zero."""
    return subprocess.check_output(command, env=env, text=True).strip()


def fetch_uv(bin_dir: Path) -> Path:
    """Download the uv release tarball and extract the ``uv`` binary into ``bin_dir``."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    archive = bin_dir / "uv.tar.gz"
    urllib.request.urlretrieve(UV_URL, str(archive))
    with tarfile.open(str(archive), "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and os.path.basename(member.name) == "uv":
                member.name = "uv"
                tar.extract(member, str(bin_dir), filter="data")
    archive.unlink()
    uv = bin_dir / "uv"
    uv.chmod(0o755)
    return uv


def uv_phases(log: Path) -> dict[str, dict[str, Any]]:
    """Uv's split of one install into resolve, prepare and install, read from its log."""
    phases: dict[str, dict[str, Any]] = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        match = UV_PHASE.match(line.strip())
        if match:
            phases[match.group(1).lower()] = {
                "packages": int(match.group(2)),
                "time": match.group(3),
            }
    return phases


def tree_bytes(path: Path, env: dict[str, str]) -> int:
    """Apparent size of a directory tree in bytes (``du -sb``)."""
    return int(output(["du", "-sb", str(path)], env).split()[0])


def main() -> int:
    """Run every arm and write the result file; exit 1 when the work directory is used."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--requirement", default="torchcell")
    parser.add_argument(
        "--arms",
        default=",".join(arm[0] for arm in ARMS),
        help="comma-separated subset",
    )
    args = parser.parse_args()
    work: Path = args.work.resolve()
    wanted = args.arms.split(",")
    unknown = sorted(set(wanted) - {arm[0] for arm in ARMS})
    if unknown:
        print("unknown arms: %s" % ", ".join(unknown), file=sys.stderr)
        return 1
    stale = [p.name for p in work.glob("venv-*")] + [
        p.name for p in work.glob("*-cache-*") if any(p.iterdir())
    ]
    if stale:
        print(
            "refusing a used work directory: %s" % ", ".join(sorted(stale)),
            file=sys.stderr,
        )
        return 1
    logs = work / "logs"
    for sub in (logs, work / "tmp", work / "python"):
        sub.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(
        {
            "TMPDIR": str(work / "tmp"),
            "UV_PYTHON_INSTALL_DIR": str(work / "python"),
            "UV_PYTHON_PREFERENCE": "only-managed",
            "UV_NO_PROGRESS": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_PROGRESS_BAR": "off",
        }
    )
    for name in ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH", "PIP_INDEX_URL"):
        env.pop(name, None)

    started = time.perf_counter()
    uv = str(fetch_uv(work / "bin"))
    fetch_uv_seconds = time.perf_counter() - started
    # --no-bin: without it uv links python3.13 into ~/.local/bin, outside the work directory.
    code, python_seconds = run(
        [uv, "python", "install", "--no-bin", PYTHON_VERSION],
        env,
        logs / "python_install.log",
    )
    if code != 0:
        print(
            "uv python install failed; see %s" % (logs / "python_install.log"),
            file=sys.stderr,
        )
        return 1

    result: dict[str, Any] = {
        "host": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "work": str(work),
        # findmnt, not `stat -f`: stat names every ext filesystem "ext2/ext3".
        "work_filesystem": output(["findmnt", "-no", "FSTYPE", "-T", str(work)], env),
        "requirement": args.requirement,
        "uv_version": output([uv, "--version"], env),
        "fetch_uv_seconds": round(fetch_uv_seconds, 2),
        "python_install_seconds": round(python_seconds, 2),
        "arms": [],
    }
    out_path = work / "install_benchmark.json"

    for arm, tool, family, extra in ARMS:
        if arm not in wanted:
            continue
        venv = work / ("venv-" + arm)
        log = logs / (arm + ".log")
        arm_env = dict(env)
        arm_env["UV_CACHE_DIR"] = str(work / ("uv-cache-" + family))
        arm_env["PIP_CACHE_DIR"] = str(work / ("pip-cache-" + family))
        seed = ["--seed"] if tool == "pip" else []
        venv_code, venv_seconds = run(
            [uv, "venv", "--python", PYTHON_VERSION, *seed, str(venv)], arm_env, log
        )
        python = str(venv / "bin" / "python")
        if tool == "pip":
            command = [python, "-m", "pip", "install", *extra, args.requirement]
            shown = ["pip", "install", *extra, args.requirement]
        else:
            command = [
                uv,
                "pip",
                "install",
                "--python",
                python,
                *extra,
                args.requirement,
            ]
            shown = ["uv", "pip", "install", *extra, args.requirement]
        row: dict[str, Any] = {
            "arm": arm,
            "tool": tool,
            "cache": "warm" if arm.endswith("_warm") else "empty",
            "command": " ".join(shown),
            "venv_seconds": round(venv_seconds, 2),
        }
        if venv_code != 0:
            row.update({"exit_code": venv_code, "failed_at": "venv"})
        else:
            code, seconds = run(command, arm_env, log)
            row.update({"exit_code": code, "install_seconds": round(seconds, 1)})
            if tool == "uv":
                row["uv_phases"] = uv_phases(log)
            if code == 0:
                if tool == "pip":
                    row["pip_version"] = output(
                        [python, "-m", "pip", "--version"], arm_env
                    ).split()[1]
                probe = subprocess.run(
                    [python, "-c", PROBE], env=arm_env, capture_output=True, text=True
                )
                row["probe_exit_code"] = probe.returncode
                if probe.returncode == 0:
                    row.update(json.loads(probe.stdout.strip().splitlines()[-1]))
                else:
                    row["probe_error"] = probe.stderr.strip().splitlines()[-1:]
                row["venv_bytes"] = tree_bytes(venv, arm_env)
        result["arms"].append(row)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
