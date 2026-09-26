# scripts/legacy_partition.py
# [[scripts.legacy_partition]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/legacy_partition.py
"""Partition ``torchcell/`` into live and legacy modules from the importer graph.

A module is *live* when a root reaches it through imports; *legacy* when no root
does. Roots are the places code is launched from today: ``tests/``, ``scripts/``,
``database/``, experiments numbered 016 and later (a leading letter is ignored, so
``W019-*`` counts and ``W006-*`` does not), the ``Makefile``,
``.pre-commit-config.yaml``, the GitHub workflows, and the ``[project.scripts]``
entry points in ``pyproject.toml``. Frozen experiments 015 and earlier are not roots:
they rerun from the ``legacy-pre-move-*`` tag, never from the live tree.

Edges come from three sources. Python files are parsed with ``ast`` and every
``import`` / ``from ... import`` that resolves to a ``torchcell`` module is an edge,
including the implicit parent packages an import executes. A ``from pkg import Name``
whose ``Name`` is re-exported by ``pkg/__init__.py`` also edges to the submodule that
defines it, so a package used through its ``__init__`` keeps that submodule live.
Root files of every kind (Python, shell, slurm, YAML, Makefile, workflows) are scanned
for dotted ``torchcell.a.b`` and path ``torchcell/a/b.py`` references, so a hydra
``_target_``, an ``importlib`` string, or a ``python torchcell/x.py`` launcher line
keeps its module live. Package files are scanned only in their non-docstring string
constants (a ``-m torchcell.x`` subprocess target, a sibling ``runner.py`` path);
frontmatter comments and docstrings name modules without importing them.

Categories in the report:

* ``live``: reached from a root through imports.
* ``init-only``: executed only because a package ``__init__.py`` imports it
  wholesale; nothing outside that ``__init__`` asks for a name it defines. Moving it
  means dropping that import from the ``__init__``.
* ``legacy``: unreachable from every root.
* ``carve-out``: ``torchcell/scratch/`` and ``torchcell/experiments/``, excluded from
  every gate already and not classified.

An ``__init__.py`` follows its package: it is ``live`` while any module under it is
live, init-only or carve-out, and ``legacy`` only when the whole package is.

``--check`` exits 1 when a module outside ``torchcell/legacy/`` and the carve-outs is
``legacy`` or ``init-only``, or when any root imports ``torchcell.legacy``. Used by
``make legacy-check`` and the CI test workflow after the move (plan
[[plan.test-suite-buildout.2026.09.25]], Decision 22).

Usage::

    python scripts/legacy_partition.py --table
    python scripts/legacy_partition.py --output notes/assets/test-campaign/legacy_partition.json
    python scripts/legacy_partition.py --check

Stdlib only, so it runs in CI before the package is installed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

REPO = Path(__file__).resolve().parents[1]
PACKAGE = "torchcell"

CARVE_OUT_PREFIXES = ("torchcell/scratch/", "torchcell/experiments/")
LEGACY_PREFIX = "torchcell/legacy/"

ROOT_DIRS = ("tests", "scripts", "database")
# pyproject.toml is NOT string-scanned: its mypy/ruff carve-out lists and the
# test-exception table name modules precisely because they are dead. Only its
# [project.scripts] entry points and the setuptools version attr count (below).
ROOT_FILES = ("Makefile", ".pre-commit-config.yaml")
ROOT_GLOBS = (".github/workflows/*.yaml", ".github/workflows/*.yml")
FIRST_LIVE_EXPERIMENT = 16
TEXT_SUFFIXES = {".py", ".sh", ".slurm", ".yaml", ".yml", ".toml", ".cfg", ".txt", ""}

_DOTTED = re.compile(r"\btorchcell(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_PATHREF = re.compile(r"\btorchcell/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py\b")
_SIBLING = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.py$")
_EXPERIMENT_DIR = re.compile(r"^[A-Za-z]?(\d{3})-")


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def module_name(path: Path) -> str:
    """Dotted module name of a ``torchcell/`` source file (``__init__`` -> package)."""
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def package_modules() -> dict[str, Path]:
    """Every ``torchcell/**/*.py`` keyed by dotted name.

    ``torchcell/cell.py`` and ``torchcell/cell/__init__.py`` both spell
    ``torchcell.cell``; the directory wins at import time, so the file is keyed
    ``torchcell.cell#file`` and reported on its own row.
    """
    modules: dict[str, Path] = {}
    for path in sorted((REPO / PACKAGE).rglob("*.py")):
        name = module_name(path)
        if name in modules:
            other = modules[name]
            keep, drop = (path, other) if other.name != "__init__.py" else (other, path)
            print(
                f"warning: {name} is spelled by {_rel(keep)} and {_rel(drop)}; "
                f"reporting {_rel(drop)} as {name}#file",
                file=sys.stderr,
            )
            modules[name + "#file"] = drop
            modules[name] = keep
            continue
        modules[name] = path
    return modules


def _experiment_roots() -> Iterator[Path]:
    for entry in sorted((REPO / "experiments").iterdir()):
        match = _EXPERIMENT_DIR.match(entry.name)
        if entry.is_dir() and match and int(match.group(1)) >= FIRST_LIVE_EXPERIMENT:
            yield entry


def root_files() -> list[Path]:
    """Files whose references define what is live, in a stable order."""
    files: list[Path] = []
    dirs = [REPO / d for d in ROOT_DIRS] + list(_experiment_roots())
    for directory in dirs:
        for path in sorted(directory.rglob("*")):
            if (
                path.is_file()
                and path.suffix in TEXT_SUFFIXES
                and ".git" not in path.parts
            ):
                files.append(path)
    files.extend(REPO / f for f in ROOT_FILES if (REPO / f).is_file())
    for pattern in ROOT_GLOBS:
        files.extend(sorted(REPO.glob(pattern)))
    return files


def _parents(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        print(f"warning: skipping unparsable {_rel(path)}: {exc}", file=sys.stderr)
        return None


def _from_base(node: ast.ImportFrom, current: str, is_package: bool) -> str:
    if node.level == 0:
        return node.module or ""
    anchor = current.split(".")
    if not is_package:
        anchor = anchor[:-1]
    anchor = anchor[: len(anchor) - (node.level - 1)]
    return ".".join(anchor + ([node.module] if node.module else []))


class _Imports:
    """Import statements of one file, kept as (base, imported names) pairs."""

    def __init__(self, tree: ast.Module, current: str, is_package: bool) -> None:
        self.plain: set[str] = set()
        self.froms: list[tuple[str, list[str]]] = []
        self.strings: set[str] = set()
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.plain.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = _from_base(node, current, is_package)
                if base:
                    self.froms.append((base, [alias.name for alias in node.names]))
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                self.strings.add(node.value)


def _docstring_nodes(tree: ast.Module) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
            ):
                ids.add(id(body[0].value))
    return ids


def string_references(text: str) -> set[str]:
    """Dotted and path-style ``torchcell`` references in arbitrary text."""
    names = set(_DOTTED.findall(text))
    for ref in _PATHREF.findall(text):
        names.add(ref[: -len(".py")].replace("/", "."))
    return names


def _entry_point_modules() -> set[str]:
    """``[project.scripts]`` targets plus the setuptools dynamic version attr."""
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject.get("project", {}).get("scripts", {})
    names = {target.split(":")[0] for target in scripts.values()}
    version = pyproject.get("tool", {}).get("setuptools", {}).get("dynamic", {})
    attr = version.get("version", {}).get("attr", "")
    if attr:
        names.add(attr.rsplit(".", 1)[0])
    return names


class ImporterGraph:
    """Import edges among ``torchcell`` modules plus the root files that import them."""

    def __init__(self) -> None:
        """Parse every package module and every root file once."""
        self.modules = package_modules()
        self.edges: dict[str, set[str]] = {}
        self.root_edges: dict[str, set[str]] = {}
        parsed: dict[str, _Imports] = {}
        for name, path in self.modules.items():
            tree = _parse(path)
            if tree is not None:
                parsed[name] = _Imports(
                    tree, name.split("#")[0], path.name == "__init__.py"
                )
        # name -> defining submodule, per package __init__ (its from-imports of submodules)
        self.reexports: dict[str, dict[str, str]] = {}
        for name, imports in parsed.items():
            if self.modules[name].name != "__init__.py":
                continue
            table: dict[str, str] = {}
            for base, names in imports.froms:
                if base in self.modules and base != name:
                    for imported in names:
                        table[imported] = base
            self.reexports[name] = table
        for name, path in self.modules.items():
            file_imports = parsed.get(name)
            if file_imports is None:
                self.edges[name] = set()
                continue
            targets = self._resolve(file_imports)
            for value in file_imports.strings:
                if _SIBLING.match(value):
                    sibling = module_name(path.parent / value)
                    if sibling in self.modules:
                        targets.add(sibling)
                targets |= self._known(string_references(value))
            self.edges[name] = targets - {name}
        for path in root_files():
            rel = _rel(path)
            text = path.read_text(encoding="utf-8", errors="replace")
            targets = self._known(string_references(text))
            if path.suffix == ".py":
                tree = _parse(path)
                if tree is not None:
                    targets |= self._resolve(
                        _Imports(tree, rel[:-3].replace("/", "."), False)
                    )
            self.root_edges[rel] = targets
        self.root_edges["pyproject.toml#project.scripts"] = self._known(
            _entry_point_modules()
        )

    def _known(self, names: Iterable[str]) -> set[str]:
        """Filter candidate dotted names to package modules, trying shorter prefixes."""
        known: set[str] = set()
        for name in names:
            if not name.startswith(PACKAGE):
                continue
            for candidate in reversed(_parents(name)):
                if candidate in self.modules:
                    known.update(p for p in _parents(candidate) if p in self.modules)
                    break
        return known

    def _resolve(self, imports: _Imports) -> set[str]:
        targets = self._known(imports.plain)
        for base, names in imports.froms:
            targets |= self._known([base])
            for imported in names:
                dotted = f"{base}.{imported}"
                if dotted in self.modules:
                    targets.add(dotted)
                elif imported == "*" and base in self.reexports:
                    targets.update(self.reexports[base].values())
                elif base in self.reexports and imported in self.reexports[base]:
                    targets.add(self.reexports[base][imported])
        return targets

    def reachable(
        self, roots: Iterable[str], skip_init_edges: bool = False
    ) -> tuple[dict[str, str], dict[str, str]]:
        """BFS from the given roots: (module -> origin root, module -> predecessor)."""
        origin: dict[str, str] = {}
        pred: dict[str, str] = {}
        queue: deque[str] = deque()
        for root in roots:
            for target in sorted(self.root_edges.get(root, ())):
                if target not in origin:
                    origin[target] = root
                    pred[target] = root
                    queue.append(target)
        while queue:
            module = queue.popleft()
            if skip_init_edges and self.modules[module].name == "__init__.py":
                continue
            for target in sorted(self.edges[module]):
                if target not in origin:
                    origin[target] = origin[module]
                    pred[target] = module
                    queue.append(target)
        return origin, pred

    def production_roots(self) -> list[str]:
        """Roots other than tests: what live-critical is measured against."""
        return [r for r in self.root_edges if not r.startswith("tests/")]


def _git_last_commit(path: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO), "log", "-1", "--format=%cs", "--", _rel(path)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return out or "untracked"


def _line_count(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def _nearest_init(graph: ImporterGraph, pred: dict[str, str], module: str) -> str:
    """Walk the BFS chain back to the ``__init__`` that pulled the module in."""
    node = module
    while node in pred and pred[node] in graph.modules:
        node = pred[node]
        if graph.modules[node].name == "__init__.py":
            return node
    return pred.get(module, "")


def partition(graph: ImporterGraph, with_git: bool = True) -> list[dict[str, object]]:
    """One row per package module with its category and reachability proof."""
    all_roots = list(graph.root_edges)
    full, pred_full = graph.reachable(all_roots)
    no_init, _ = graph.reachable(all_roots, skip_init_edges=True)
    critical, _ = graph.reachable(graph.production_roots())
    rows: dict[str, dict[str, object]] = {}
    for name in sorted(graph.modules):
        path = graph.modules[name]
        rel = _rel(path)
        if rel.startswith(CARVE_OUT_PREFIXES):
            category, via = "carve-out", ""
        elif name in no_init:
            category, via = "live", no_init[name]
        elif name in full:
            category, via = "init-only", _nearest_init(graph, pred_full, name)
        else:
            category, via = "legacy", "unreachable from every root"
        rows[name] = {
            "module": name,
            "path": rel,
            "lines": _line_count(path),
            "last_commit": _git_last_commit(path) if with_git else "",
            "category": category,
            "via": via,
            "live_critical": name in critical,
            "already_legacy": rel.startswith(LEGACY_PREFIX),
        }
    # An __init__ follows its package.
    for name, row in rows.items():
        if graph.modules[name].name != "__init__.py" or row["category"] == "carve-out":
            continue
        members = [
            r
            for other, r in rows.items()
            if other != name and other.startswith(name + ".")
        ]
        if any(r["category"] != "legacy" for r in members):
            row["category"], row["via"] = "live", "package has live members"
        elif members:
            row["category"], row["via"] = "legacy", "every package member is legacy"
    return list(rows.values())


def render_table(rows: list[dict[str, object]], categories: set[str]) -> str:
    """Markdown table of the rows in the given categories."""
    lines = [
        "| Module | Lines | Last commit | Category | Reached via |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        if row["category"] in categories:
            lines.append(
                f"| `{row['path']}` | {row['lines']} | {row['last_commit']} | "
                f"{row['category']} | {row['via']} |"
            )
    return "\n".join(lines)


def check(graph: ImporterGraph, rows: list[dict[str, object]]) -> int:
    """Hard-mode invariant: the live tree holds no legacy or init-only module."""
    failures = [
        f"{r['path']}: {r['category']} ({r['via']})"
        for r in rows
        if r["category"] in {"legacy", "init-only"} and not r["already_legacy"]
    ]
    for root, targets in sorted(graph.root_edges.items()):
        hits = sorted(t for t in targets if t.startswith(f"{PACKAGE}.legacy"))
        if hits:
            failures.append(f"{root}: imports {', '.join(hits)}")
    for line in failures:
        print(f"legacy-check: {line}")
    if failures:
        print(f"legacy-check: {len(failures)} violation(s)")
        return 1
    print("legacy-check: the live tree is closed under imports; no root touches legacy")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", action="store_true", help="print a markdown table")
    parser.add_argument(
        "--categories",
        default="legacy,init-only",
        help="comma-separated categories for --table (default: legacy,init-only)",
    )
    parser.add_argument("--output", type=Path, help="write every row as JSON here")
    parser.add_argument("--check", action="store_true", help="hard-mode invariant")
    parser.add_argument(
        "--no-git", action="store_true", help="skip the per-file git log lookup"
    )
    args = parser.parse_args(argv)

    graph = ImporterGraph()
    rows = partition(graph, with_git=not args.no_git)
    categories = ("live", "init-only", "legacy", "carve-out")
    counts = {c: sum(1 for r in rows if r["category"] == c) for c in categories}
    lines = {
        c: sum(cast(int, r["lines"]) for r in rows if r["category"] == c)
        for c in categories
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"counts": counts, "lines": lines, "rows": rows}, indent=2)
            + "\n"
        )
        print(f"wrote {args.output}")
    if args.table:
        print(render_table(rows, set(args.categories.split(","))))
        print()
    print(
        "modules: "
        + ", ".join(f"{c} {counts[c]} ({lines[c]} lines)" for c in categories)
    )
    if args.check:
        return check(graph, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
