# scripts/check_paired_tests.py
# [[scripts.check_paired_tests]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/check_paired_tests.py
"""Paired-test gate: every new ``torchcell`` module ships with its mirrored test file.

For each Python file added under ``torchcell/`` since the merge base with ``--base``
(default ``origin/main``), the test ``tests/torchcell/<same relative dir>/test_<name>.py``
must exist. ``--strict`` widens the check to every added or modified module (hard
mode, switched on in the legacy-move PR). ``__init__.py`` and ``__main__.py`` are
exempt.

Exceptions live in ONE place, ``[tool.torchcell.test_exceptions]`` in
``pyproject.toml``: ``paths`` (a directory prefix ending in ``/``, a ``fnmatch`` glob
when it contains ``*``, or an exact path) and ``pairs`` (source path -> the test file
that covers it when the mirrored basename would collide, see Gotcha 4 of
[[plan.test-suite-buildout.2026.09.25]]).

The diff is index-vs-merge-base (``git diff --cached``), so the pre-commit hook sees
staged additions and CI, where nothing is staged, sees the branch's commits.

Usage::

    python scripts/check_paired_tests.py                 # added modules, vs origin/main
    python scripts/check_paired_tests.py --strict        # added or modified modules
    python scripts/check_paired_tests.py --base main

Exit 1 names every module without a paired test. Stdlib only.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXEMPT_BASENAMES = {"__init__.py", "__main__.py"}


def load_exceptions(pyproject: Path) -> tuple[list[str], dict[str, str]]:
    """``paths`` and ``pairs`` from ``[tool.torchcell.test_exceptions]``."""
    table = (
        tomllib.loads(pyproject.read_text(encoding="utf-8"))
        .get("tool", {})
        .get("torchcell", {})
        .get("test_exceptions", {})
    )
    return list(table.get("paths", [])), dict(table.get("pairs", {}))


def is_excepted(path: str, patterns: list[str]) -> bool:
    """Prefix (``dir/``), glob (``*``), or exact match against the repo-relative path."""
    name = Path(path).name
    for pattern in patterns:
        if pattern.endswith("/"):
            if path.startswith(pattern):
                return True
        elif "*" in pattern:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(name, pattern):
                return True
        elif path == pattern:
            return True
    return False


def expected_test(path: str) -> str:
    """``torchcell/a/b.py`` -> ``tests/torchcell/a/test_b.py``."""
    source = Path(path)
    return (Path("tests") / source.parent / f"test_{source.name}").as_posix()


def changed_modules(base: str, strict: bool, repo: Path) -> list[str]:
    """Python files under ``torchcell/`` added (or, strict, modified) since the merge base."""
    merge_base = subprocess.run(
        ["git", "-C", str(repo), "merge-base", base, "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    diff_filter = "AM" if strict else "A"
    out = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--cached",
            "--name-only",
            f"--diff-filter={diff_filter}",
            merge_base,
            "--",
            "torchcell/",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(
        line
        for line in out.splitlines()
        if line.endswith(".py") and Path(line).name not in EXEMPT_BASENAMES
    )


def missing_pairs(
    modules: list[str], patterns: list[str], pairs: dict[str, str], repo: Path
) -> list[tuple[str, str]]:
    """(module, expected test) for every module whose test file does not exist."""
    missing: list[tuple[str, str]] = []
    for module in modules:
        if is_excepted(module, patterns):
            continue
        test = pairs.get(module, expected_test(module))
        if not (repo / test).is_file():
            missing.append((module, test))
    return missing


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="origin/main", help="branch to diff against")
    parser.add_argument(
        "--strict", action="store_true", help="check modified modules too (hard mode)"
    )
    parser.add_argument("--repo", type=Path, default=REPO, help="repository root")
    args = parser.parse_args(argv)

    patterns, pairs = load_exceptions(args.repo / "pyproject.toml")
    modules = changed_modules(args.base, args.strict, args.repo)
    missing = missing_pairs(modules, patterns, pairs, args.repo)
    mode = "added or modified" if args.strict else "added"
    for module, test in missing:
        print(f"paired-tests: {module} has no test file {test}")
    if missing:
        print(
            f"paired-tests: {len(missing)} of {len(modules)} {mode} module(s) lack a "
            "paired test; add the test or an entry under "
            "[tool.torchcell.test_exceptions] in pyproject.toml"
        )
        return 1
    print(f"paired-tests: {len(modules)} {mode} module(s) vs {args.base}, all paired")
    return 0


if __name__ == "__main__":
    sys.exit(main())
