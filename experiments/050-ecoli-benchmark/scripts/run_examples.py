# experiments/050-ecoli-benchmark/scripts/run_examples.py
# [[experiments.050-ecoli-benchmark.scripts.run_examples]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/050-ecoli-benchmark/scripts/run_examples
"""Run the four benchmark example scripts in an empty directory and keep what they print.

Each file of ``examples/benchmark`` is run one after the other with ``--python``, in
``--work`` (which must be empty), with ``TC_DATA_URL`` and ``DATA_ROOT`` removed from the
environment so the script's own defaults apply. Standard output goes to
``experiments/050-ecoli-benchmark/results/examples/<name>.out`` and the interpreter's
torchcell version to ``versions.json``.

    python experiments/050-ecoli-benchmark/scripts/run_examples.py \
        --python <python with torchcell> --work <empty dir> [--pythonpath <checkout>]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EXAMPLES = Path("examples/benchmark")
OUT = Path("experiments/050-ecoli-benchmark/results/examples")


def main() -> int:
    """Run every example and write its standard output."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", required=True)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--pythonpath", default=None)
    args = parser.parse_args()
    work: Path = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        print(f"refusing a non-empty work directory: {work}", file=sys.stderr)
        return 1
    env = dict(os.environ)
    for name in ("TC_DATA_URL", "DATA_ROOT", "PYTHONPATH"):
        env.pop(name, None)
    if args.pythonpath:
        env["PYTHONPATH"] = str(Path(args.pythonpath).resolve())
    OUT.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output(
        [args.python, "-c", "import torchcell; print(torchcell.__version__)"],
        env=env,
        text=True,
        cwd=str(work),
    ).strip()
    for script in sorted(EXAMPLES.glob("*.py")):
        done = subprocess.run(
            [args.python, str(script.resolve())],
            env=env,
            cwd=str(work),
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            print(done.stderr[-2000:], file=sys.stderr)
            return 1
        (OUT / f"{script.stem}.out").write_text(done.stdout, encoding="utf-8")
        print(f"{script.name}: ok")
    (OUT / "versions.json").write_text(
        json.dumps({"torchcell": version}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
