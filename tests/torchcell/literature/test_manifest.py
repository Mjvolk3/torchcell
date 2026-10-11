# tests/torchcell/literature/test_manifest.py
"""Mirror-wide manifest validation: ``scan_mirror`` and the ``validate`` command (#923)."""

import os
from pathlib import Path

import pytest

from torchcell.literature.manifest import (
    MIRROR_TREES,
    Manifest,
    ManifestFailure,
    build_manifest,
    main,
    scan_mirror,
    write_manifest,
)

# The #923 shape: a hand-written retrieval with free fields and no retriever/sha256.
LEGACY = (
    '{"citation_key": "zzLegacy2020", "files": [{"path": "data/a.csv", '
    '"role": "raw_data", "bytes": 1, "sha256": "ab", "retrieval": {"method": '
    '"direct_url", "retrieved_at": "2026-09-14", "retrieval_command": "gh api x", '
    '"pinned_commit": "f95f"}}]}'
)


def _key(root: Path, key: str, doi: str | None = None) -> None:
    (root / key / "data").mkdir(parents=True)
    (root / key / "data" / "a.csv").write_bytes(b"x")
    write_manifest(
        root / key,
        build_manifest(root / key, citation_key=key, doi=doi, created_at="t"),
    )


def _mirror(tmp_path: Path) -> Path:
    root = tmp_path / "torchcell-raw"
    _key(root, "aaGood2019", doi="10.1000/good")
    (root / "zzLegacy2020").mkdir()
    (root / "zzLegacy2020" / "manifest.json").write_text(LEGACY)
    (root / "mmNotJson2021").mkdir()
    (root / "mmNotJson2021" / "manifest.json").write_text("{not json")
    (root / "_bib").mkdir()
    (root / "_bib" / "manifest.json").write_text('{"version": 1, "bibs": []}')
    (root / "noManifest2022").mkdir()
    return root


def test_scan_mirror_splits_valid_from_failing_and_skips_service_stores(
    tmp_path: Path,
) -> None:
    """Valid keys are returned validated; every failing key is named with its error."""
    root = _mirror(tmp_path)
    scan = scan_mirror(root)
    assert scan.root == str(root)
    assert list(scan.manifests) == ["aaGood2019"]
    good = scan.manifests["aaGood2019"]
    assert isinstance(good, Manifest)
    assert good.doi == "10.1000/good"
    assert [(f.path, f.role, f.bytes) for f in good.files] == [
        ("data/a.csv", "raw_data", 1)
    ]
    assert [(f.citation_key, f.n_errors) for f in scan.failures] == [
        ("mmNotJson2021", 1),
        ("zzLegacy2020", 4),
    ]
    legacy = scan.failures[1]
    assert isinstance(legacy, ManifestFailure)
    assert legacy.path == str(root / "zzLegacy2020" / "manifest.json")
    for field in ("retriever", "sha256", "retrieval_command", "pinned_commit"):
        assert f"files.0.retrieval.{field}" in legacy.error
    assert "Invalid JSON" in scan.failures[0].error


def test_validate_command_fails_naming_each_bad_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 1 with one FAIL line per bad key; the counts line per tree."""
    _mirror(tmp_path)
    assert (
        main(["validate", "--data-root", str(tmp_path), "--tree", "torchcell-raw"]) == 1
    )
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "torchcell-raw: 1 valid, 2 failing"
    fails = [line.split(":")[0] for line in out if line.startswith("FAIL ")]
    assert fails == [
        "FAIL torchcell-raw/mmNotJson2021",
        "FAIL torchcell-raw/zzLegacy2020",
    ]


def test_validate_command_passes_a_clean_mirror(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both trees clean: exit 0, one counts line each."""
    for tree in MIRROR_TREES:
        _key(tmp_path / tree, "aaGood2019")
    assert main(["validate", "--data-root", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "torchcell-library: 1 valid, 0 failing",
        "torchcell-raw: 1 valid, 0 failing",
    ]


def test_validate_command_refuses_a_missing_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tree that does not exist is exit 2, not an empty pass."""
    assert main(["validate", "--data-root", str(tmp_path)]) == 2
    assert capsys.readouterr().err.strip() == (
        f"torchcell-library: no such directory {tmp_path / 'torchcell-library'}"
    )


@pytest.mark.data
@pytest.mark.parametrize("tree", MIRROR_TREES)
def test_every_real_mirror_manifest_validates(tree: str) -> None:
    """Every paper-key manifest in the real mirror is a valid ``Manifest`` (#923)."""
    scan = scan_mirror(Path(os.environ["DATA_ROOT"]) / tree)
    assert scan.manifests, f"{tree}: no manifests read"
    bad = [f"{f.citation_key} ({f.n_errors} errors)" for f in scan.failures]
    assert bad == [], f"{tree}: manifests failing validation: {bad}"
