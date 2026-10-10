# scripts/check_candidate_verdicts.py
# [[scripts.check_candidate_verdicts]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/check_candidate_verdicts.py
"""Candidate-verdict tripwire: a NEW ``@register_dataset`` ships its candidate verdict.

For each Python file under ``torchcell/datasets/`` added or modified since the merge base
with ``--base`` (default ``origin/main``), the classes decorated with ``register_dataset``
in the INDEX version are compared with those in the merge-base version. Every class that
is new must sit in a module declaring a module-level ``CITATION_KEY`` string, and
``database/candidates/<CITATION_KEY>.json`` must be in the same index. Whether that
verdict passes is the enforcement test's job
(``tests/torchcell/datasets/test_candidate_verdicts.py``); this hook only stops a
registration from reaching a commit without one.

The diff is index-vs-merge-base, the shape of ``scripts/check_paired_tests.py``: the
pre-commit hook sees the staged addition, and CI, where nothing is staged, sees the
branch's commits.

Usage::

    python scripts/check_candidate_verdicts.py
    python scripts/check_candidate_verdicts.py --base main

Exit 1 names every new class without a key or a verdict. Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = "torchcell/datasets/"
STORE = "database/candidates"
DECORATOR = "register_dataset"


def _git(
    repo: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def registered_classes(source: str) -> set[str]:
    """Names of top-level classes decorated with ``register_dataset`` (bare or dotted)."""
    out: set[str] = set()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            name = (
                decorator.id
                if isinstance(decorator, ast.Name)
                else decorator.attr
                if isinstance(decorator, ast.Attribute)
                else None
            )
            if name == DECORATOR:
                out.add(node.name)
    return out


def citation_key(source: str) -> str | None:
    """The module-level ``CITATION_KEY`` string constant, if declared."""
    for node in ast.parse(source).body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if (
            isinstance(target, ast.Name)
            and target.id == "CITATION_KEY"
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            return value.value
    return None


def changed_dataset_modules(base: str, repo: Path) -> tuple[str, list[str]]:
    """(merge base, dataset modules added or modified in the index since it)."""
    merge_base = _git(repo, "merge-base", base, "HEAD").stdout.strip()
    out = _git(
        repo,
        "diff",
        "--cached",
        "--name-only",
        "--diff-filter=AM",
        merge_base,
        "--",
        DATASETS,
    ).stdout
    return merge_base, sorted(line for line in out.splitlines() if line.endswith(".py"))


def new_registrations(base: str, repo: Path) -> list[tuple[str, str, str | None]]:
    """(module, class, CITATION_KEY) for every class registered in the index, not the base."""
    merge_base, modules = changed_dataset_modules(base, repo)
    found: list[tuple[str, str, str | None]] = []
    for module in modules:
        staged = _git(repo, "show", f":{module}").stdout
        before = _git(repo, "show", f"{merge_base}:{module}", check=False)
        old = registered_classes(before.stdout) if before.returncode == 0 else set()
        key = citation_key(staged)
        for name in sorted(registered_classes(staged) - old):
            found.append((module, name, key))
    return found


def in_index(path: str, repo: Path) -> bool:
    """Whether ``path`` is staged (or committed) in the index."""
    return _git(repo, "cat-file", "-e", f":{path}", check=False).returncode == 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="origin/main", help="branch to diff against")
    parser.add_argument("--repo", type=Path, default=REPO, help="repository root")
    args = parser.parse_args(argv)

    found = new_registrations(args.base, args.repo)
    problems: list[str] = []
    for module, name, key in found:
        if key is None:
            problems.append(
                f"{module}: {name} is new and the module has no CITATION_KEY"
            )
        elif not in_index(f"{STORE}/{key}.json", args.repo):
            problems.append(
                f"{module}: {name} is new and {STORE}/{key}.json is not in the index "
                f"(python -m torchcell.candidates gate --citation-key {key} --write)"
            )
    for problem in problems:
        print(f"candidate-verdicts: {problem}")
    if problems:
        print(
            f"candidate-verdicts: {len(problems)} of {len(found)} new registration(s) "
            "lack a candidate verdict"
        )
        return 1
    print(
        f"candidate-verdicts: {len(found)} new registration(s) vs {args.base}, all carry a verdict"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
