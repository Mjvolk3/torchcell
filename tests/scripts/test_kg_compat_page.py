# tests/scripts/test_kg_compat_page.py
# [[tests.scripts.test_kg_compat_page]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/scripts/test_kg_compat_page.py
"""The compatibility page from fake snapshots against a real, throwaway git repo.

The repo under ``tmp_path`` carries the two surface modules with one class each
(``Experiment`` in ``schema.py``, ``ModelStrict`` in ``pydant.py``). Tag ``v1.1.0`` is
made before the surface modules exist, ``v1.2.0`` with the first surface, ``v1.2.1``
after ``Experiment`` gains a field. Two snapshots are committed under
``database/releases/``: ``R1`` (datasets ``DsA`` and ``DsB``, closures taken at the first
surface) and ``R2`` (``DsA`` alone, closure taken at the second surface). So the matrix
is: ``v1.2.0`` compatible with R1 and partial with R2; ``v1.2.1`` partial with R1 (both
datasets drifted) and compatible with R2; ``v1.1.0`` is not listed because the page
starts at ``v1.2.0``, and a tag before the surface modules reads ``unknown``.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from torchcell.knowledge_graphs.kg_manifest import GraphSchemaEntry
from torchcell.knowledge_graphs.release_snapshot import (
    KgReleaseSnapshot,
    SnapshotDataset,
    SnapshotEvent,
    composite_sha256,
    write_snapshot,
)
from torchcell.provenance.schema_deps import load_surface_from_sources

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "kg_compat_page.py"

PYDANT = "class ModelStrict:\n    pass\n"
SCHEMA_1 = (
    "from pydant import ModelStrict\n\nclass Experiment(ModelStrict):\n    a: int\n"
)
SCHEMA_2 = (
    "from pydant import ModelStrict\n\nclass Experiment(ModelStrict):\n    a: int\n"
    "    b: str\n"
)


@pytest.fixture(scope="module")
def page() -> ModuleType:
    spec = importlib.util.spec_from_file_location("kg_compat_page", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _fingerprints(schema: str) -> dict[str, str]:
    surface = load_surface_from_sources(
        {
            "torchcell/datamodels/schema.py": schema,
            "torchcell/datamodels/pydant.py": PYDANT,
        }
    )
    return dict(surface.fingerprints)


def _snapshot(
    release: str, built_at: str, datasets: dict[str, str]
) -> KgReleaseSnapshot:
    entries = {
        name: SnapshotDataset(
            dataset_class=name,
            n_experiments=1,
            content_sha256=digest,
            import_mode="full",
            admitted_at=built_at,
        )
        for name, digest in datasets.items()
    }
    return KgReleaseSnapshot(
        release=release,
        version="1.0",
        torchcell_commit="7715ee35d95c535620b9",
        torchcell_version="1.2.0",
        torchcell_tag=None,
        built_at=built_at,
        neo4j_version="5.26.28",
        biocypher_version="0.15.2",
        store_host="gilahyper",
        n_nodes=None,
        datasets=entries,
        graph_schema={"experiment": GraphSchemaEntry(kind="node")},
        events=[
            SnapshotEvent(
                kind="bootstrap", at=built_at, torchcell_commit="7715ee35", datasets=[]
            )
        ],
        composite_sha256=composite_sha256(entries),
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """The tagged throwaway repo with two committed snapshots (see the module docstring)."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "README.md").write_text("x\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "REL: 1.1.0")
    _git(root, "tag", "v1.1.0")
    modules = root / "torchcell" / "datamodels"
    modules.mkdir(parents=True)
    (modules / "pydant.py").write_text(PYDANT, encoding="utf-8")
    (modules / "schema.py").write_text(SCHEMA_1, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "REL: 1.2.0")
    _git(root, "tag", "v1.2.0")
    (modules / "schema.py").write_text(SCHEMA_2, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "REL: 1.2.1")
    _git(root, "tag", "v1.2.1")
    first, second = _fingerprints(SCHEMA_1), _fingerprints(SCHEMA_2)
    r1 = _snapshot(
        "2026.09.17-7715ee35",
        "2026-09-17T20:36:32-05:00",
        {"DsA": "a" * 64, "DsB": "b" * 64},
    )
    r2 = _snapshot(
        "2026.09.30-abcdef01", "2026-09-30T00:00:00+00:00", {"DsA": "c" * 64}
    )
    write_snapshot(r1, {"DsA": first, "DsB": first}, root)
    write_snapshot(r2, {"DsA": second}, root)
    return root


EXPECTED_PAGE = """<!-- Generated by scripts/kg_compat_page.py. Do not edit by hand. -->

# Database releases and package compatibility

This page is generated by `scripts/kg_compat_page.py` from the release snapshots committed under `database/releases/` and must not be edited by hand; regenerate it with `python scripts/kg_compat_page.py` after a release is stamped, and `--check` fails when it is stale.

## Releases

| KG release | KG version | built | commit | package version at build | tag |
| --- | --- | --- | --- | --- | --- |
| 2026.09.17-7715ee35 | 1.0 | 2026-09-17T20:36:32-05:00 | 7715ee35 | 1.2.0 | (untagged) |
| 2026.09.30-abcdef01 | 1.0 | 2026-09-30T00:00:00+00:00 | 7715ee35 | 1.2.0 | (untagged) |

## Compatibility

| package tag | 2026.09.17-7715ee35 | 2026.09.30-abcdef01 |
| --- | --- | --- |
| v1.2.0 | compatible | partial (all 1 datasets drifted) |
| v1.2.1 | partial (all 2 datasets drifted) | compatible |

## How compatibility is decided

"""


def test_package_tags_start_at_v1_2_0_in_version_order(
    page: ModuleType, repo: Path
) -> None:
    assert page.package_tags(repo) == ["v1.2.0", "v1.2.1"]
    assert page.package_tags(repo, first="v1.1.0") == ["v1.1.0", "v1.2.0", "v1.2.1"]
    with pytest.raises(ValueError, match="tag v9.0.0 is not in"):
        page.package_tags(repo, first="v9.0.0")


def test_surface_at_tag_reads_git_show_and_is_none_before_the_modules(
    page: ModuleType, repo: Path
) -> None:
    assert page.surface_at_tag(repo, "v1.1.0") is None
    assert page.surface_at_tag(repo, "v1.2.0").fingerprints == _fingerprints(SCHEMA_1)
    assert page.surface_at_tag(repo, "v1.2.1").fingerprints == _fingerprints(SCHEMA_2)


def test_verdict_names_the_drifted_datasets_or_all_of_them(page: ModuleType) -> None:
    """DsA drifts (fingerprint from the other surface), DsB matches: partial names DsA;
    every dataset drifted collapses to the count; no surface is unknown.
    """
    first, second = _fingerprints(SCHEMA_1), _fingerprints(SCHEMA_2)
    surface = load_surface_from_sources(
        {
            "torchcell/datamodels/schema.py": SCHEMA_1,
            "torchcell/datamodels/pydant.py": PYDANT,
        }
    )
    snapshot = _snapshot("r", "t", {"DsA": "a" * 64, "DsB": "b" * 64})
    assert page.verdict(snapshot, {"DsA": second, "DsB": first}, surface) == (
        "partial (1 datasets drifted: DsA)"
    )
    assert page.verdict(snapshot, {"DsA": second, "DsB": second}, surface) == (
        "partial (all 2 datasets drifted)"
    )
    assert page.verdict(snapshot, {"DsA": first, "DsB": first}, surface) == "compatible"
    assert page.verdict(snapshot, {"DsA": first, "DsB": first}, None) == "unknown"


def test_main_writes_the_page_and_check_reports_staleness(
    page: ModuleType, repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = repo / "docs" / "source" / "database" / "compatibility.md"
    assert page.main(["--repo-root", str(repo), "--check"]) == 1
    assert capsys.readouterr().out == (
        f"{out}: stale; run python scripts/kg_compat_page.py\n"
    )
    assert page.main(["--repo-root", str(repo)]) == 0
    assert capsys.readouterr().out == f"wrote {out}\n"
    text = out.read_text(encoding="utf-8")
    assert text == EXPECTED_PAGE + page.HOW_DECIDED + "\n"
    assert page.main(["--repo-root", str(repo), "--check"]) == 0
    assert capsys.readouterr().out == f"{out}: current\n"
    out.write_text(text + "edited\n", encoding="utf-8")
    assert page.main(["--repo-root", str(repo), "--check"]) == 1
    assert out.read_text(encoding="utf-8").endswith("edited\n")
    other = repo / "elsewhere.md"
    assert page.main(["--repo-root", str(repo), "--output", str(other)]) == 0
    assert other.read_text(encoding="utf-8") == text


def test_how_decided_is_three_sentences(page: ModuleType) -> None:
    assert page.HOW_DECIDED.count(". ") + 1 == 3
    assert page.HOW_DECIDED.endswith("the rest read unchanged.")


def test_main_refuses_a_repo_without_snapshots(
    page: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="no release snapshots under"):
        page.main(["--repo-root", str(tmp_path)])
