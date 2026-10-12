# tests/torchcell/provenance/_schema_change_repo.py
# [[tests.torchcell.provenance._schema_change_repo]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/provenance/_schema_change_repo.py
"""A throwaway git repository for the schema-change tests.

``make_repo`` commits, on ``main``, a two-file schema surface (``pydant.py`` with
``ModelStrict``; ``schema.py`` with the ``Unit`` enum, ``Media``, ``FitnessPhenotype`` and
``Experiment``), two loaders and one release snapshot, then checks out ``feature``.
``torchcell/datasets/served.py`` imports ``Experiment`` and declares ``ServedDataset``,
the one dataset the snapshot ``2026.10.06-fixture`` serves, so its closure is
``Experiment, FitnessPhenotype, Media, ModelStrict, Unit``. ``torchcell/datasets/dev.py``
imports ``Media`` and declares the unserved ``DevOnlyDataset`` (closure ``Media,
ModelStrict``). Git is hermetic: ``hermetic_git`` points ``HOME`` and the global config
into ``tmp_path``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from torchcell.knowledge_graphs.release_snapshot import (
    KgReleaseSnapshot,
    SnapshotDataset,
)

PYDANT = """\
from pydantic import BaseModel


class ModelStrict(BaseModel):
    pass
"""

SCHEMA = """\
from enum import Enum

from pydantic import Field

from torchcell.datamodels.pydant import ModelStrict


class Unit(str, Enum):
    a = "a"
    b = "b"


class Media(ModelStrict):
    name: str
    state: str


class FitnessPhenotype(ModelStrict):
    fitness: float
    fitness_std: float | None = Field(default=None)
    unit: Unit


class Experiment(ModelStrict):
    media: Media
    phenotype: FitnessPhenotype
"""

SERVED_LOADER = """\
from torchcell.datamodels.schema import Experiment


class ServedDataset:
    pass
"""

DEV_LOADER = """\
from torchcell.datamodels.schema import Media


class DevOnlyDataset:
    pass
"""

RELEASE = "2026.10.06-fixture"


def snapshot_json(release: str, built_at: str, datasets: list[str]) -> str:
    """A minimal valid release snapshot serving ``datasets``."""
    return KgReleaseSnapshot(
        release=release,
        version="3.0",
        torchcell_commit="0" * 40,
        torchcell_version=None,
        torchcell_tag=None,
        built_at=built_at,
        neo4j_version="5",
        biocypher_version="0.15.2",
        store_host="fixture",
        n_nodes=None,
        datasets={
            name: SnapshotDataset(
                dataset_class=name,
                n_experiments=1,
                content_sha256="0" * 64,
                import_mode="full",
                admitted_at=built_at,
            )
            for name in datasets
        },
        graph_schema={},
        events=[],
        composite_sha256="0" * 64,
    ).model_dump_json(indent=2)


def git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd``; fail the test on a nonzero exit."""
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True
    )
    assert result.returncode == 0, f"git {' '.join(args)}: {result.stderr}"
    return result.stdout.strip()


def write(repo: Path, rel: str, text: str) -> None:
    """Write ``text`` to ``repo/rel``, creating parents."""
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def hermetic_git(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """Point git's identity and config at ``home``; clear inherited repo variables."""
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "t")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "t@t")


def make_repo(tmp_path: Path) -> Path:
    """The fixture repository, on branch ``feature`` one commit after ``main``'s root."""
    repo = tmp_path / "repo"
    git(tmp_path, "init", "-q", "-b", "main", str(repo))
    write(repo, "torchcell/datamodels/pydant.py", PYDANT)
    write(repo, "torchcell/datamodels/schema.py", SCHEMA)
    write(repo, "torchcell/datasets/served.py", SERVED_LOADER)
    write(repo, "torchcell/datasets/dev.py", DEV_LOADER)
    write(
        repo,
        f"database/releases/{RELEASE}.json",
        snapshot_json(RELEASE, "2026-10-06T04:42:48-05:00", ["ServedDataset"]),
    )
    write(repo, "database/schema_changes/.keep", "")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    git(repo, "checkout", "-q", "-b", "feature")
    return repo


def edit_schema(repo: Path, old: str, new: str, *, stage: bool = True) -> None:
    """Replace ``old`` with ``new`` in the fixture schema (exactly once), then stage it."""
    path = repo / "torchcell/datamodels/schema.py"
    text = path.read_text()
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new))
    if stage:
        git(repo, "add", "torchcell/datamodels/schema.py")
