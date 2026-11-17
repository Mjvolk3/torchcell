# experiments/034-showcase-datasets/scripts/ecoli_download_tutorial_page.py
# [[experiments.034-showcase-datasets.scripts.ecoli_download_tutorial_page]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/034-showcase-datasets/scripts/ecoli_download_tutorial_page
"""Run the E. coli download tutorial on a host, and write the page fragments from the runs.

Two subcommands, both standard library only, because ``run`` is also executed on a
machine that holds nothing but a fresh venv with torchcell in it:

``run``
    Executes ``ecoli_download_tutorial.py`` with a given interpreter in a fresh, empty
    work directory, with ``TC_DATA_URL``, ``DATA_ROOT`` and ``PYTHONPATH`` removed from
    the environment (``--pythonpath`` puts one checkout back, for a run from the
    repository). Records the standard output and its sha256, the seconds each loader
    took (the tutorial's own timings file), the bytes each dataset occupies on disk, the
    four index rows of the endpoint, and the versions of the interpreter and of the
    packages the output depends on. Writes one JSON file per host.

``page``
    Reads every ``results/ecoli_download_tutorial.<host>.json`` and every
    ``results/install_benchmark.<host>.json`` and writes the MyST fragments the page
    includes into ``docs/source/datasets/ecoli/_generated/download-tutorial/``. The
    printed output on the page is the reference host's; ``page`` exits 1 when any other
    host printed different bytes, so the page cannot claim an agreement that did not
    hold.

Usage, from the repository root::

    python experiments/034-showcase-datasets/scripts/ecoli_download_tutorial_page.py run \
        --python ~/miniconda3/envs/torchcell/bin/python --pythonpath "$PWD" \
        --installed-from "repository checkout" --work <empty dir> \
        --out experiments/034-showcase-datasets/results/ecoli_download_tutorial.gilahyper.json
    python experiments/034-showcase-datasets/scripts/ecoli_download_tutorial_page.py page
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT = "experiments/034-showcase-datasets/scripts/ecoli_download_tutorial_page.py"
TUTORIAL = "experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py"
RESULTS_DIR = Path("experiments/034-showcase-datasets/results")
FRAGMENT_DIR = Path("docs/source/datasets/ecoli/_generated/download-tutorial")
TIMINGS_FILE = "data/torchcell/ecoli_download_tutorial_timings.json"
BANNER = re.compile(r"^={12} (.+) ={12}$")

# Dataset rows of the page, in page order: slug, label on the page, loader module, what
# one record's label is. The record and feature counts are read from the runs.
DATASETS: tuple[tuple[str, str, str, str], ...] = (
    (
        "ecoli_growth_rate_campos2018",
        "Campos 2018, growth rate",
        "torchcell.datasets.ecoli.campos2018",
        "maximal growth rate relative to the parent strain",
    ),
    (
        "ecoli_morphology_campos2018",
        "Campos 2018, imaging",
        "torchcell.datasets.ecoli.campos2018",
        "cell-shape and cell-cycle features",
    ),
    (
        "gene_interaction_babu2014",
        "Babu 2014, genetic interaction",
        "torchcell.datasets.ecoli.babu2014",
        "S-score of a double mutant",
    ),
    (
        "metabolome_fuhrer2017",
        "Fuhrer 2017, metabolome",
        "torchcell.datasets.ecoli.fuhrer2017",
        "modified z-score per detected ion",
    ),
)
# Values per record. One for the two scalar labels; for the other two the tutorial
# prints the shape of the table it builds, and the number is read from that line.
WIDTH_PATTERNS: dict[str, tuple[str, str]] = {
    "ecoli_morphology_campos2018": ("morphology", r"^table shape\s+\(\d+, (\d+)\)$"),
    "metabolome_fuhrer2017": ("metabolome", r"^matrix shape \(\d+, (\d+)\) "),
}
PROBE = """
import json, sys, numpy, pandas, torch, torchcell
print(json.dumps({
    "torchcell": torchcell.__version__, "torchcell_file": torchcell.__file__,
    "python": sys.version.split()[0], "numpy": numpy.__version__,
    "pandas": pandas.__version__, "torch": torch.__version__,
}))
"""
ARM_LABELS = {
    "pip_cold": "pip, empty cache",
    "uv_cold": "uv, empty cache",
    "pip_warm": "pip, filled cache",
    "uv_warm": "uv, filled cache",
    "uv_cpu_cold": "uv, CPU build of torch, empty cache",
    "pip_cpu_cold": "pip, CPU build of torch, empty cache",
}


# ------------------------------------------------------------------------------- run
def tree_bytes(root: Path) -> int:
    """Total size of the regular files under ``root``."""
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def run_tutorial(args: argparse.Namespace) -> int:
    """Run the tutorial once in an empty directory and write the host's result file."""
    work: Path = args.work.resolve()
    if work.exists() and any(work.iterdir()):
        print(f"refusing a non-empty work directory: {work}", file=sys.stderr)
        return 1
    work.mkdir(parents=True, exist_ok=True)
    tutorial = Path(args.tutorial).resolve()
    env = dict(os.environ)
    for name in ("TC_DATA_URL", "DATA_ROOT", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(name, None)
    if args.pythonpath:
        env["PYTHONPATH"] = str(Path(args.pythonpath).resolve())

    probe = json.loads(
        subprocess.check_output(
            [args.python, "-c", PROBE], env=env, text=True, cwd=str(work)
        )
        .strip()
        .splitlines()[-1]
    )
    started = time.perf_counter()
    done = subprocess.run(
        [args.python, str(tutorial)],
        env=env,
        cwd=str(work),
        capture_output=True,
        text=True,
    )
    seconds = time.perf_counter() - started
    (work / "stdout.txt").write_text(done.stdout, encoding="utf-8")
    (work / "stderr.txt").write_text(done.stderr, encoding="utf-8")
    if done.returncode != 0:
        print(done.stderr[-3000:], file=sys.stderr)
        print(f"the tutorial exited {done.returncode}", file=sys.stderr)
        return 1

    endpoint = next(
        line.split(" ", 1)[1]
        for line in done.stdout.splitlines()
        if line.startswith("endpoint ")
    )
    with urllib.request.urlopen(endpoint + "/datasets", timeout=60) as response:
        index = json.loads(response.read().decode("utf-8"))
    slugs = [row[0] for row in DATASETS]
    rows = {}
    for slug in slugs:
        archive = next(
            line.split()[1]
            for line in done.stdout.splitlines()
            if line.startswith(f"  archive  {slug}-")
        )
        row = next(a for a in index["artifacts"] if a["archive"] == archive)
        rows[slug] = {
            key: row[key]
            for key in (
                "dataset_class",
                "archive",
                "archive_bytes",
                "archive_sha256",
                "content_sha256",
                "n_experiments",
                "torchcell_version",
                "torchcell_commit",
                "built_at",
                "packaged_at",
                "kg_release",
                "kg_version",
            )
        }
    data = work / "data" / "torchcell"
    result: dict[str, Any] = {
        "host": args.host or socket.gethostname(),
        "ran_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "installed_from": args.installed_from,
        "interpreter": probe,
        "endpoint": endpoint,
        "tutorial_sha256": hashlib.sha256(tutorial.read_bytes()).hexdigest(),
        "whole_script_seconds": round(seconds, 1),
        "load_seconds": json.loads((work / TIMINGS_FILE).read_text(encoding="utf-8")),
        "disk_bytes": {
            slug: {
                sub: tree_bytes(data / slug / sub)
                for sub in ("artifacts", "processed", "preprocess")
            }
            for slug in slugs
        },
        "index_rows": rows,
        "stdout_sha256": hashlib.sha256(done.stdout.encode("utf-8")).hexdigest(),
        "stdout": done.stdout,
    }
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "wrote {} (stdout sha256 {}, {:.1f} s)".format(
            out, result["stdout_sha256"], seconds
        )
    )
    return 0


# ------------------------------------------------------------------------------ page
def sections(stdout: str) -> dict[str, str]:
    """Standard output split on the tutorial's section banners, keyed by section title."""
    out: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in stdout.splitlines():
        match = BANNER.match(line)
        if match:
            current = out.setdefault(match.group(1), [])
        elif current is None:
            raise ValueError(f"output before the first section banner: {line!r}")
        else:
            current.append(line)
    return {title: "\n".join(lines) for title, lines in out.items()}


def header(stamp: str) -> str:
    """The HTML comment every generated fragment opens with."""
    return f"<!-- Generated by {SCRIPT} (run {stamp}). Do not edit by hand. -->\n\n"


def megabytes(n: int) -> str:
    """Bytes as megabytes (10^6 bytes), two decimals below 10 MB and one above."""
    value = n / 1e6
    return (f"{value:.2f}") if value < 10 else (f"{value:.1f}")


def width_of(slug: str, by_section: dict[str, str]) -> int:
    """Values per record: 1 for a scalar label, else read from the tutorial's shape line."""
    if slug not in WIDTH_PATTERNS:
        return 1
    title, pattern = WIDTH_PATTERNS[slug]
    for line in by_section[title].splitlines():
        match = re.match(pattern, line)
        if match:
            return int(match.group(1))
    raise ValueError(f"no shape line for {slug} in section {title!r}")


def table(head: list[str], rows: list[list[str]], right: set[int] | None = None) -> str:
    """A pipe table; columns whose index is in ``right`` are right-aligned."""
    right = right or set()
    rule = ["---:" if i in right else "---" for i in range(len(head))]
    lines = ["| " + " | ".join(cells) + " |" for cells in [head, rule, *rows]]
    return "\n".join(lines) + "\n"


def loaders_import(arms: list[dict[str, Any]]) -> str:
    """Whether the import probe passed in every installed venv, else its last error line."""
    installed = [a for a in arms if a.get("exit_code") == 0]
    failed = [a for a in installed if a.get("probe_exit_code") != 0]
    if not installed:
        return "no arm installed"
    if not failed:
        return "yes"
    errors = sorted({" ".join(a.get("probe_error", [])) for a in failed})
    return "no, in {} of {} venvs: `{}`".format(
        len(failed), len(installed), "; ".join(errors)
    )


def write_page(args: argparse.Namespace) -> int:
    """Write every fragment of the page from the committed result files."""
    runs = {
        path.name.split(".")[1]: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RESULTS_DIR.glob("ecoli_download_tutorial.*.json"))
    }
    if args.reference not in runs:
        print(
            f"no result file for the reference host {args.reference!r}", file=sys.stderr
        )
        return 1
    reference = runs[args.reference]
    others = [name for name in runs if name != args.reference]
    hosts = [args.reference, *others]
    differing = [
        n for n in others if runs[n]["stdout_sha256"] != reference["stdout_sha256"]
    ]
    if differing:
        print(
            "standard output differs from {} on: {}".format(
                args.reference, ", ".join(differing)
            ),
            file=sys.stderr,
        )
        return 1
    scripts = {runs[n]["tutorial_sha256"] for n in hosts}
    if len(scripts) != 1:
        print("the hosts did not run the same tutorial file", file=sys.stderr)
        return 1

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    FRAGMENT_DIR.mkdir(parents=True, exist_ok=True)
    by_section = sections(reference["stdout"])
    for title, text in by_section.items():
        name = "output_{}.md".format(title.replace(" ", "_"))
        (FRAGMENT_DIR / name).write_text(
            header(stamp) + "```text\n" + text + "\n```\n", encoding="utf-8"
        )

    rows = reference["index_rows"]
    dataset_rows = [
        [
            label,
            "`{}.{}`".format(module, rows[slug]["dataset_class"]),
            what,
            "{:,}".format(rows[slug]["n_experiments"]),
            f"{width_of(slug, by_section):,}",
            f"`{slug}`",
        ]
        for slug, label, module, what in DATASETS
    ]
    (FRAGMENT_DIR / "datasets_table.md").write_text(
        header(stamp)
        + table(
            [
                "Dataset",
                "Loader",
                "Label of one record",
                "Records",
                "Values per record",
                "Store slug",
            ],
            dataset_rows,
            right={3, 4},
        ),
        encoding="utf-8",
    )

    timing_head = ["Dataset", "Archive (MB)", "On disk after unpacking (MB)"] + [
        f"Seconds on {name}" for name in hosts
    ]
    timing_rows = []
    for slug, label, _module, _what in DATASETS:
        disk = reference["disk_bytes"][slug]
        timing_rows.append(
            [
                label,
                megabytes(rows[slug]["archive_bytes"]),
                megabytes(disk["processed"] + disk["preprocess"]),
                *["{:.2f}".format(runs[name]["load_seconds"][slug]) for name in hosts],
            ]
        )
    totals = [
        "**All four**",
        megabytes(sum(rows[s]["archive_bytes"] for s, *_ in DATASETS)),
        megabytes(
            sum(
                reference["disk_bytes"][s]["processed"]
                + reference["disk_bytes"][s]["preprocess"]
                for s, *_ in DATASETS
            )
        ),
        *["{:.2f}".format(sum(runs[name]["load_seconds"].values())) for name in hosts],
    ]
    (FRAGMENT_DIR / "timings.md").write_text(
        header(stamp)
        + table(
            timing_head, [*timing_rows, totals], right=set(range(1, len(timing_head)))
        ),
        encoding="utf-8",
    )

    archive_rows = [
        [
            "`{}`".format(rows[slug]["archive"]),
            "`{}`".format(rows[slug]["archive_sha256"][:16]),
            "`{}`".format(rows[slug]["content_sha256"][:16]),
            "`{}`".format(rows[slug]["torchcell_commit"][:8]),
            rows[slug]["built_at"][:10],
            rows[slug]["torchcell_version"],
            "{} ({})".format(rows[slug]["kg_release"], rows[slug]["kg_version"]),
        ]
        for slug, *_ in DATASETS
    ]
    (FRAGMENT_DIR / "archives.md").write_text(
        header(stamp)
        + table(
            [
                "Archive",
                "sha256, first 16",
                "Content hash, first 16",
                "Store built at commit",
                "Store built on",
                "Packaged with torchcell",
                "Knowledge-graph release (version)",
            ],
            archive_rows,
        ),
        encoding="utf-8",
    )

    run_rows = [
        [
            name,
            runs[name]["installed_from"],
            runs[name]["interpreter"]["torchcell"],
            runs[name]["interpreter"]["python"],
            runs[name]["interpreter"]["pandas"],
            runs[name]["interpreter"]["numpy"],
            "{:.0f}".format(runs[name]["whole_script_seconds"]),
            "`{}`".format(runs[name]["stdout_sha256"][:16]),
        ]
        for name in hosts
    ]
    (FRAGMENT_DIR / "runs.md").write_text(
        header(stamp)
        + table(
            [
                "Host",
                "torchcell installed from",
                "torchcell",
                "Python",
                "pandas",
                "numpy",
                "Whole script (s)",
                "sha256 of the output, first 16",
            ],
            run_rows,
            right={6},
        ),
        encoding="utf-8",
    )

    benches = {
        path.name.split(".")[1]: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RESULTS_DIR.glob("install_benchmark.*.json"))
    }
    if benches:
        bench_hosts = [h for h in hosts if h in benches] + [
            h for h in benches if h not in hosts
        ]
        arms: list[str] = []
        for name in bench_hosts:
            arms += [a["arm"] for a in benches[name]["arms"] if a["arm"] not in arms]
        by_arm = {
            name: {a["arm"]: a for a in benches[name]["arms"]} for name in bench_hosts
        }

        def cell(name: str, arm: str, key: str, fmt: str, scale: float = 1.0) -> str:
            row = by_arm[name].get(arm)
            if row is None:
                return "not run"
            if row.get("exit_code") != 0:
                return "failed (exit {})".format(row.get("exit_code"))
            return fmt % (row[key] / scale)

        head = ["Install", "Command"]
        for name in bench_hosts:
            head += [f"Seconds on {name}", f"venv on {name} (GB)"]
        bench_rows = []
        for arm in arms:
            command = next(
                by_arm[n][arm]["command"] for n in bench_hosts if arm in by_arm[n]
            )
            line = [ARM_LABELS[arm], f"`{command}`"]
            for name in bench_hosts:
                line += [
                    cell(name, arm, "install_seconds", "%.0f"),
                    cell(name, arm, "venv_bytes", "%.1f", 1e9),
                ]
            bench_rows.append(line)
        setup_rows = [
            [
                name,
                benches[name]["requirement"],
                benches[name]["uv_version"].split()[1],
                next(
                    (
                        a["pip_version"]
                        for a in benches[name]["arms"]
                        if "pip_version" in a
                    ),
                    "not recorded",
                ),
                benches[name]["work_filesystem"],
                str(benches[name]["cpu_count"]),
                "{:.0f}".format(benches[name]["python_install_seconds"]),
                loaders_import(benches[name]["arms"]),
                benches[name]["started_at"][:10],
            ]
            for name in bench_hosts
        ]
        cold = [
            "`uv pip install` took {} s and `pip install` took {} s on {}".format(
                cell(name, "uv_cold", "install_seconds", "%.0f"),
                cell(name, "pip_cold", "install_seconds", "%.0f"),
                name,
            )
            for name in bench_hosts
            if {"uv_cold", "pip_cold"} <= set(by_arm[name])
        ]
        (FRAGMENT_DIR / "install_benchmark.md").write_text(
            header(stamp)
            + ("With an empty cache, " + ", and ".join(cold) + ".\n\n" if cold else "")
            + table(head, bench_rows, right=set(range(2, len(head))))
            + "\n"
            + table(
                [
                    "Host",
                    "Requirement",
                    "uv",
                    "pip",
                    "Filesystem of the venvs and caches",
                    "CPUs",
                    "Seconds for uv to fetch Python 3.13",
                    "Loaders import after the install",
                    "Run on",
                ],
                setup_rows,
                right={5, 6},
            ),
            encoding="utf-8",
        )
    print(f"wrote {len(list(FRAGMENT_DIR.glob('*.md')))} fragments to {FRAGMENT_DIR}")
    return 0


def main() -> int:
    """Dispatch ``run`` or ``page``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="run the tutorial here and write this host's result"
    )
    run.add_argument("--python", required=True, help="interpreter that has torchcell")
    run.add_argument("--tutorial", default=TUTORIAL)
    run.add_argument(
        "--pythonpath", default=None, help="a checkout to import torchcell from"
    )
    run.add_argument(
        "--installed-from", required=True, help="how this torchcell was installed"
    )
    run.add_argument(
        "--host", default=None, help="name for the host (default: hostname)"
    )
    run.add_argument(
        "--work", required=True, type=Path, help="empty directory to run in"
    )
    run.add_argument("--out", required=True, type=Path)
    page = sub.add_parser("page", help="write the page fragments from the result files")
    page.add_argument("--reference", default="gilahyper")
    args = parser.parse_args()
    return run_tutorial(args) if args.command == "run" else write_page(args)


if __name__ == "__main__":
    raise SystemExit(main())
