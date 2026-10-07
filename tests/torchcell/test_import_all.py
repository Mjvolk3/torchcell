# tests/torchcell/test_import_all.py
# [[tests.torchcell.test_import_all]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/test_import_all.py
"""Smoke test: every ``torchcell`` module imports, one parametrized case per module.

The module list comes from the filesystem, not ``pkgutil.walk_packages``, so nothing is
imported at collection time. Two committed lists shape the sweep (Decision 9 of
[[plan.test-suite-buildout.2026.09.25]]):

* ``NEVER_IMPORT``: modules whose import has a side effect no test should trigger (a
  HuggingFace download, a dataset build, ``plt.show()``, a cwd log file, SSL env
  mutation). They are excluded from the parameter set outright; an xfail would still run
  the side effect.
* ``KNOWN_BROKEN``: modules that fail to import today for a recorded reason. They are
  strict xfails, so a fix turns into an XPASS failure and the list decays instead of
  rotting.

Each case runs under ``monkeypatch.chdir(tmp_path)`` because biocypher creates
``biocypher-log/`` in the working directory for every adapter and several
``knowledge_graphs`` modules ``basicConfig`` a log file there. ``pytest-timeout``
(``timeout = 300`` in pyproject) bounds a hanging import.

This file is deselected from the behavioral coverage run (``make cov``, the CI
behavioral step) and measured separately as the import-only column of
``scripts/coverage_gaps.py`` (Decision 17): executing def/class/import lines is not
testing behavior.

A second test scans the package source for machine-specific absolute paths; the known
hits are allowlisted with their line so a new one fails.
"""

import importlib
import re
import sys
from pathlib import Path

import pytest

import torchcell

PACKAGE_DIR = Path(torchcell.__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent

NEVER_IMPORT: tuple[str, ...] = (
    # carve-outs: one-off scratch scripts and experiment code
    r"^torchcell\.(scratch|experiments)(\.|$)",
)

# module -> reason; strict xfail, so a fix must remove the entry. Recorded from the
# first sweep (2026.09.26); every entry was in the legacy partition of
# scripts/legacy_partition.py and left with the 2026.10.06 legacy retirement, which
# emptied the list (and the side-effecting NEVER_IMPORT entries with it).
KNOWN_BROKEN: dict[str, str] = {}

# Machine-specific absolute paths in package source, with the line that carries each.
# A new hit fails test_no_hard_coded_machine_paths; a fixed one must be removed here.
HARD_CODED_PATH_ALLOWLIST: dict[str, int] = {
    "torchcell/datasets/scerevisiae/spell.py": 21
}
_HARD_CODED_PATH = re.compile(
    r"/Users/michaelvolk|/home/michaelvolk|~/Documents/projects"
    r"|\"Documents\", \"projects\""
)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO_DIR).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def all_modules() -> list[str]:
    """Every importable ``torchcell`` module name, sorted, minus ``NEVER_IMPORT``."""
    names = {_module_name(p) for p in PACKAGE_DIR.rglob("*.py")}
    skip = [re.compile(pattern) for pattern in NEVER_IMPORT]
    return sorted(n for n in names if not any(p.search(n) for p in skip))


def _case(name: str) -> object:
    if name in KNOWN_BROKEN:
        return pytest.param(
            name, marks=pytest.mark.xfail(strict=True, reason=KNOWN_BROKEN[name])
        )
    return name


@pytest.mark.parametrize("name", [_case(n) for n in all_modules()])
def test_module_imports(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The module imports and registers under its own name."""
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module(name)
    assert module.__name__ == name
    assert sys.modules[name] is module


def test_known_broken_entries_are_real_modules() -> None:
    """Every KNOWN_BROKEN key names a module the sweep collects (no stale entries)."""
    collected = set(all_modules())
    assert set(KNOWN_BROKEN) <= collected, sorted(set(KNOWN_BROKEN) - collected)


def test_no_hard_coded_machine_paths() -> None:
    """Package source names no developer machine path beyond the allowlisted lines."""
    hits: dict[str, int] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        rel = path.relative_to(REPO_DIR).as_posix()
        if rel.startswith(("torchcell/scratch/", "torchcell/experiments/")):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _HARD_CODED_PATH.search(line):
                hits[rel] = lineno
                break
    assert hits == HARD_CODED_PATH_ALLOWLIST
