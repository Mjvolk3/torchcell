# tests/torchcell/provenance/test_build_manifest.py
"""Tests for build_manifest: manifest round-trip, drift detection, and fleet staleness scan."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from torchcell.provenance import build_manifest as bm
from torchcell.provenance import schema_deps as sd

SCHEMA = """
from pydantic import BaseModel, Field


class ModelStrict(BaseModel):
    class Config:
        extra = "forbid"


class Media(ModelStrict):
    name: str
    state: str


class Environment(ModelStrict):
    media: Media
    temperature: float
"""


def _surface(src: str) -> sd.SchemaSurface:
    return sd.load_surface_from_sources({"schema": src})


def _loader(tmp_path: Path, imports: str = "Environment") -> Path:
    path = tmp_path / "loader.py"
    path.write_text(
        f"from torchcell.datamodels.schema import {imports}\nclass MyDataset:\n    pass\n"
    )
    return path


def _manifest(
    tmp_path: Path, surface: sd.SchemaSurface, name: str = "slug"
) -> bm.BuildManifest:
    return bm.compute_manifest(
        dataset_name=name,
        loader_module="pkg.loader",
        loader_class="MyDataset",
        loader_path=_loader(tmp_path),
        surface=surface,
        built_at="2026-07-15T00:00:00+00:00",
        hostname="testhost",
        torchcell_commit=None,
        torchcell_dirty=None,
    )


def test_manifest_captures_closure_and_round_trips(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, _surface(SCHEMA))
    # Environment -> Media in closure, so both appear (plus the ModelStrict base)
    assert {"Environment", "Media"} <= set(manifest.closure)
    restored = bm.BuildManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest


def test_check_manifest_fresh_against_same_surface(tmp_path: Path) -> None:
    surface = _surface(SCHEMA)
    result = bm.check_manifest(_manifest(tmp_path, surface), surface, str(tmp_path))
    assert result.is_stale is False
    assert result.drift == []


def test_check_manifest_detects_contract_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, _surface(SCHEMA))
    changed = SCHEMA.replace("    name: str\n", "    name: str\n    ph: float\n")
    result = bm.check_manifest(manifest, _surface(changed), str(tmp_path))
    assert result.is_stale is True
    assert any(drift.symbol == "Media" for drift in result.drift)


def test_check_manifest_detects_removed_symbol(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, _surface(SCHEMA))
    removed = SCHEMA.replace(
        "class Media(ModelStrict):\n    name: str\n    state: str\n", ""
    )
    result = bm.check_manifest(manifest, _surface(removed), str(tmp_path))
    assert result.is_stale is True
    media_drift = [drift for drift in result.drift if drift.symbol == "Media"]
    assert media_drift and media_drift[0].current_fingerprint is None


def _build_dataset_dir(root: Path, slug: str) -> Path:
    slug_dir = root / "data" / "torchcell" / slug
    (slug_dir / "processed" / "lmdb").mkdir(parents=True)
    (slug_dir / "preprocess").mkdir(parents=True)
    return slug_dir


def test_check_all_reports_fresh_stale_unmanifested(tmp_path: Path) -> None:
    surface = _surface(SCHEMA)

    fresh_dir = _build_dataset_dir(tmp_path, "ds_fresh")
    fresh = _manifest(tmp_path, surface, name="ds_fresh")
    (fresh_dir / "preprocess" / bm.MANIFEST_FILENAME).write_text(
        fresh.model_dump_json()
    )

    stale_dir = _build_dataset_dir(tmp_path, "ds_stale")
    stale = fresh.model_copy(
        update={"dataset_name": "ds_stale", "closure": {"Media": "deadbeef"}}
    )
    (stale_dir / "preprocess" / bm.MANIFEST_FILENAME).write_text(
        stale.model_dump_json()
    )

    _build_dataset_dir(tmp_path, "ds_bare")  # built LMDB but no manifest

    status = {
        check.dataset_name: check.status for check in bm.check_all(tmp_path, surface)
    }
    assert status == {
        "ds_fresh": "fresh",
        "ds_stale": "stale",
        "ds_bare": "unmanifested",
    }


def test_write_build_manifest_end_to_end(tmp_path: Path) -> None:
    # a real on-disk loader module importing a REAL schema symbol, exercising the same path the
    # post_process build hook takes (module resolution + real-schema closure + manifest write).
    module_file = tmp_path / "fake_loader_mod.py"
    module_file.write_text(
        "from torchcell.datamodels.schema import Environment\n\n"
        "class FakeDataset:\n"
        "    def __init__(self, root: str) -> None:\n"
        "        self.root = root\n"
        "        self.preprocess_dir = root + '/preprocess'\n"
    )
    spec = importlib.util.spec_from_file_location("fake_loader_mod", module_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["fake_loader_mod"] = module
    try:
        spec.loader.exec_module(module)
        root = tmp_path / "data" / "torchcell" / "fake_ds"
        (root / "preprocess").mkdir(parents=True)
        dataset = module.FakeDataset(str(root))

        out = bm.write_build_manifest(dataset)
        assert out == root / "preprocess" / bm.MANIFEST_FILENAME
        manifest = bm.BuildManifest.model_validate_json(out.read_text())
        assert manifest.dataset_name == "fake_ds"
        assert manifest.loader_class == "FakeDataset"
        assert manifest.hostname
        # closure resolved against the REAL schema: Environment pulls in Media
        assert {"Environment", "Media"} <= set(manifest.closure)
    finally:
        sys.modules.pop("fake_loader_mod", None)
