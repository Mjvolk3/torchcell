# scripts/coverage_gaps.py
# [[scripts.coverage_gaps]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/coverage_gaps.py
"""Per-module coverage table for the test campaign, generated, never hand-typed.

Reads ``coverage json`` output (``coverage json -o file.json``) and prints a markdown
table with one row per ``torchcell`` module: whether it is live-critical (reached from a
production root through imports, per ``scripts/legacy_partition.py``), the statement
count, and one coverage column per JSON file given. Rows sort live-critical first, then
ascending by the ``--after`` column, so effort visibly goes to modules that matter and
are least covered. A TOTAL row reports coverage.py's combined line+branch percent and
the line-only percent from the same run.

Columns and their runs (Decisions 14, 17 and 20 of
[[plan.test-suite-buildout.2026.09.25]]):

* ``--before`` / ``--after``: the behavioral run,
  ``coverage run -m pytest tests/torchcell --deselect tests/torchcell/test_import_all.py``,
  before and after a campaign phase; a delta column follows when both are given.
* ``--import-only``: ``coverage run -m pytest tests/torchcell/test_import_all.py``, so a
  reader sees how much of a module's number is mere def/class execution.
* ``--local``: the same behavioral run on a machine with the real ``DATA_ROOT`` and
  ``--slow --data``, a different statistic labeled as such.

Usage::

    make cov-gaps
    python scripts/coverage_gaps.py --after cov.json --before prev.json --import-only imp.json
    python scripts/coverage_gaps.py --after cov.json --all      # every module, not just >= 50 statements

Paste the output verbatim into the campaign note (``notes/test-campaign.*.md``).
Stdlib only; imports ``legacy_partition`` from this directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from legacy_partition import (  # type: ignore[import-not-found]  # noqa: E402
    REPO,
    ImporterGraph,
)

MIN_STATEMENTS = 50


def load(path: Path) -> dict[str, dict[str, float]]:
    """``{repo-relative path: summary}`` from a ``coverage json`` file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    files: dict[str, dict[str, float]] = {}
    for name, entry in data["files"].items():
        rel = name
        if Path(name).is_absolute():
            rel = Path(name).resolve().relative_to(REPO).as_posix()
        files[rel] = entry["summary"]
    files["TOTAL"] = data["totals"]
    return files


def _pct(summary: dict[str, float] | None) -> str:
    if summary is None:
        return "n/a"
    return f"{summary['percent_covered']:.1f}%"


def _delta(before: dict[str, float] | None, after: dict[str, float] | None) -> str:
    if before is None or after is None:
        return ""
    return f"{after['percent_covered'] - before['percent_covered']:+.1f}"


def live_critical_modules() -> set[str]:
    """Repo-relative paths of modules reached from a non-test root."""
    graph = ImporterGraph()
    reached, _ = graph.reachable(graph.production_roots())
    return {graph.modules[m].relative_to(REPO).as_posix() for m in reached}


def render(
    columns: dict[str, dict[str, dict[str, float]]],
    order: list[str],
    critical: set[str],
    min_statements: int,
    commit: str,
) -> str:
    """The markdown table; ``columns`` maps a header to a loaded coverage file."""
    after = columns.get("after") or next(iter(columns.values()))
    paths = sorted(p for p in after if p != "TOTAL")
    rows = []
    for path in paths:
        statements = int(after[path]["num_statements"])
        if statements < min_statements:
            continue
        rows.append((path, statements))
    rows.sort(
        key=lambda r: (r[0] not in critical, after[r[0]]["percent_covered"], r[0])
    )

    headers = ["Module", "Live-critical", "Statements"] + [
        f"{name} @ {commit}" for name in order
    ]
    if "before" in columns and "after" in columns:
        headers.append("Delta")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for path, statements in rows:
        cells = [
            f"`{path}`",
            "yes" if path in critical else "",
            str(statements),
            *(_pct(columns[name].get(path)) for name in order),
        ]
        if "before" in columns and "after" in columns:
            cells.append(
                _delta(columns["before"].get(path), columns["after"].get(path))
            )
        lines.append("| " + " | ".join(cells) + " |")
    total_cells = [
        "TOTAL (line+branch)",
        "",
        str(int(after["TOTAL"]["num_statements"])),
    ]
    total_cells += [_pct(columns[name].get("TOTAL")) for name in order]
    if "before" in columns and "after" in columns:
        total_cells.append(
            _delta(columns["before"]["TOTAL"], columns["after"]["TOTAL"])
        )
    lines.append("| " + " | ".join(total_cells) + " |")
    line_only = [
        f"{columns[name]['TOTAL']['percent_statements_covered']:.1f}%"
        if "TOTAL" in columns[name]
        else "n/a"
        for name in order
    ]
    lines.append(
        "| TOTAL (line only) | | "
        + str(int(after["TOTAL"]["num_statements"]))
        + " | "
        + " | ".join(line_only)
        + (" | |" if "Delta" in headers else " |")
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--after", type=Path, required=True, help="behavioral run, now")
    parser.add_argument("--before", type=Path, help="behavioral run, previous phase")
    parser.add_argument("--import-only", type=Path, help="import-all run")
    parser.add_argument("--local", type=Path, help="behavioral run with --slow --data")
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"every module, not only >= {MIN_STATEMENTS} statements",
    )
    args = parser.parse_args(argv)

    columns: dict[str, dict[str, dict[str, float]]] = {}
    order: list[str] = []
    for name in ("before", "after", "import_only", "local"):
        path = getattr(args, name)
        if path is not None:
            label = name.replace("_", "-")
            columns[label] = load(path)
            order.append(label)
    commit = (
        subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )

    print(f"Generated by: python scripts/coverage_gaps.py {' '.join(sys.argv[1:])}")
    print(f"Live-critical: importer graph (scripts/legacy_partition.py) @ {commit}")
    print()
    print(
        render(
            columns,
            order,
            live_critical_modules(),
            0 if args.all else MIN_STATEMENTS,
            commit,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
