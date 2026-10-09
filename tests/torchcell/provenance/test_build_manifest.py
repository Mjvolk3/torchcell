# tests/torchcell/provenance/test_build_manifest.py
# [[tests.torchcell.provenance.test_build_manifest]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/provenance/test_build_manifest.py
"""Tests for build_manifest: manifest round-trip, drift detection, and fleet staleness scan.

2026.09.30 (Phase 12). On the three-class synthetic schema (``ModelStrict`` base,
``Media``, ``Environment`` holding a ``Media``), a loader importing ``Environment`` has
the closure {Environment, Media, ModelStrict}, each mapped to the surface's own
fingerprint, and ``surface_modules`` is ["schema"] (the source key). Fingerprints are
per symbol: adding ``ph`` to ``Media`` drifts ``Media`` alone, so a loader importing only
``Media`` goes stale while an ``Environment`` edit leaves it fresh. Added: every
manifest field, the exact drift records, the scan's directory rule (only
``data/torchcell/<slug>/processed/lmdb`` counts) and its use of the manifest's own
``dataset_name``, the refusal of a malformed manifest, ``_git_info`` on a throwaway repo
(commit, clean, dirty) and outside any repo, and the CLI: the exact report lines and
exit codes 0 (fresh), 1 (stale), 0 with only unmanifested stores (Finding), the
``$DATA_ROOT`` default, and the ``KeyError`` when neither is given.

2026.10.09 (#734). A second synthetic surface, ``VOCAB_SCHEMA``, shaped like the live
chain the issue was filed on: a module-level ``Literal`` alias annotated on a field, and a
pattern map reached through a validator rather than named in the class body. The closure
of a loader importing that class holds both, so narrowing the ``Literal``, widening it, or
changing one namespace's regex stales the store, with the drift reported on the BINDING
while the class fingerprint stays equal. A vocabulary the closure does not reach leaves it
fresh.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pydantic
import pytest

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


ENV_EDIT = SCHEMA.replace("    temperature: float\n", "    temperature: int\n")
MEDIA_EDIT = SCHEMA.replace("    name: str\n", "    name: str\n    ph: float\n")


def test_manifest_records_every_field(tmp_path: Path) -> None:
    """Closure {Environment, Media, ModelStrict} at the surface's fingerprints, sorted."""
    surface = _surface(SCHEMA)
    manifest = _manifest(tmp_path, surface)
    assert manifest.model_dump() == {
        "manifest_schema_version": 1,
        "dataset_name": "slug",
        "loader_class": "MyDataset",
        "loader_module": "pkg.loader",
        "surface_modules": ["schema"],
        "closure": {
            name: surface.fingerprints[name]
            for name in ("Environment", "Media", "ModelStrict")
        },
        "built_at": "2026-07-15T00:00:00+00:00",
        "hostname": "testhost",
        "torchcell_commit": None,
        "torchcell_dirty": None,
    }
    assert list(manifest.closure) == ["Environment", "Media", "ModelStrict"]


def test_drift_lists_exactly_the_changed_symbol(tmp_path: Path) -> None:
    """A field added to Media drifts Media only; Environment's fingerprint is unchanged."""
    manifest = _manifest(tmp_path, _surface(SCHEMA))
    changed = _surface(MEDIA_EDIT)
    result = bm.check_manifest(manifest, changed, "/pre")
    assert result == bm.StaleResult(
        dataset_name="slug",
        preprocess_dir="/pre",
        is_stale=True,
        drift=[
            bm.SymbolDrift(
                symbol="Media",
                stored_fingerprint=manifest.closure["Media"],
                current_fingerprint=changed.fingerprints["Media"],
            )
        ],
    )


def test_change_outside_the_closure_leaves_a_loader_fresh(tmp_path: Path) -> None:
    """A loader importing only Media: closure {Media, ModelStrict}; an Environment edit
    is outside it (fresh), a Media edit is inside it (stale).
    """
    surface = _surface(SCHEMA)
    manifest = bm.compute_manifest(
        dataset_name="media_only",
        loader_module="pkg.loader",
        loader_class="MyDataset",
        loader_path=_loader(tmp_path, imports="Media"),
        surface=surface,
        built_at="2026-07-15T00:00:00+00:00",
        hostname="testhost",
        torchcell_commit="abc",
        torchcell_dirty=True,
    )
    assert list(manifest.closure) == ["Media", "ModelStrict"]
    env_edit = _surface(ENV_EDIT)
    assert env_edit.fingerprints["Environment"] != surface.fingerprints["Environment"]
    assert (manifest.torchcell_commit, manifest.torchcell_dirty) == ("abc", True)
    assert not bm.check_manifest(manifest, env_edit, "p").is_stale
    stale = bm.check_manifest(manifest, _surface(MEDIA_EDIT), "p")
    assert [d.symbol for d in stale.drift] == ["Media"]


def _write(slug_dir: Path, manifest: bm.BuildManifest) -> None:
    (slug_dir / "preprocess" / bm.MANIFEST_FILENAME).write_text(
        manifest.model_dump_json()
    )


def test_scan_counts_only_built_lmdbs_and_reports_the_manifest_name(
    tmp_path: Path,
) -> None:
    """A slug without ``processed/lmdb`` is ignored even with a manifest; a manifest in
    ``ds_dir`` naming itself ``renamed`` is reported as ``renamed`` (the name comes from
    the file, not the directory).
    """
    surface = _surface(SCHEMA)
    unbuilt = tmp_path / "data" / "torchcell" / "ds_unbuilt" / "preprocess"
    unbuilt.mkdir(parents=True)
    (unbuilt / bm.MANIFEST_FILENAME).write_text(
        _manifest(tmp_path, surface).model_dump_json()
    )
    _write(
        _build_dataset_dir(tmp_path, "ds_dir"),
        _manifest(tmp_path, surface, name="renamed"),
    )
    assert bm.check_all(tmp_path, surface) == [
        bm.DatasetCheck(dataset_name="renamed", status="fresh", drift=[])
    ]


def test_malformed_manifest_is_refused(tmp_path: Path) -> None:
    """A manifest missing required fields raises instead of being counted."""
    slug_dir = _build_dataset_dir(tmp_path, "ds_bad")
    (slug_dir / "preprocess" / bm.MANIFEST_FILENAME).write_text('{"dataset_name": "x"}')
    with pytest.raises(pydantic.ValidationError, match="loader_class"):
        bm.check_all(tmp_path, _surface(SCHEMA))


def _fleet(tmp_path: Path, stale: bool, bare: bool) -> sd.SchemaSurface:
    """``ds_fresh`` always; ``ds_stale`` (Media drifted) and ``ds_bare`` on request."""
    surface = _surface(SCHEMA)
    fresh = _manifest(tmp_path, surface, name="ds_fresh")
    _write(_build_dataset_dir(tmp_path, "ds_fresh"), fresh)
    if stale:
        closure = {**fresh.closure, "Media": "deadbeef", "ModelStrict": "cafe"}
        _write(
            _build_dataset_dir(tmp_path, "ds_stale"),
            fresh.model_copy(update={"dataset_name": "ds_stale", "closure": closure}),
        )
    if bare:
        _build_dataset_dir(tmp_path, "ds_bare")
    return surface


def _cli(
    monkeypatch: pytest.MonkeyPatch, surface: sd.SchemaSurface, argv: list[str]
) -> int:
    monkeypatch.setattr(bm, "load_default_surface", lambda: surface)
    monkeypatch.setattr(bm, "load_dotenv", lambda: False)
    return bm.main(argv)


def test_cli_all_fresh_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One fresh store: the count line, a blank line, the all-fresh line; exit 0."""
    surface = _fleet(tmp_path, stale=False, bare=False)
    assert _cli(monkeypatch, surface, ["--data-root", str(tmp_path)]) == 0
    assert capsys.readouterr().out == (
        "Built datasets: 1  (fresh 1, stale 0, unmanifested 0)\n"
        "\n"
        "  All built datasets are fresh against the local schema.\n"
    )


def test_cli_stale_exits_one_and_names_the_changed_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stale lines first (symbols sorted), then unmanifested; exit 1."""
    surface = _fleet(tmp_path, stale=True, bare=True)
    assert _cli(monkeypatch, surface, ["--data-root", str(tmp_path)]) == 1
    assert capsys.readouterr().out == (
        "Built datasets: 3  (fresh 1, stale 1, unmanifested 1)\n"
        "\n"
        "  [STALE] ds_stale  -> rebuild; changed: Media, ModelStrict\n"
        "  [no manifest] ds_bare  -> written on next rebuild\n"
    )


def test_cli_unmanifested_only_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: a store with no manifest is listed but the exit code is 0
    (``build_manifest.py:281`` counts only ``stale``), so a script gating a full rebuild on
    this exit code (CLAUDE.md: every mapped dev store must read ``fresh``) passes a store
    whose freshness is unknown. Pinned until unmanifested stores fail the gate.
    """
    surface = _fleet(tmp_path, stale=False, bare=True)
    assert _cli(monkeypatch, surface, ["--data-root", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Built datasets: 2  (fresh 1, stale 0, unmanifested 1)",
        "",
        "  [no manifest] ds_bare  -> written on next rebuild",
    ]


def test_cli_defaults_to_the_data_root_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No ``--data-root``: ``$DATA_ROOT`` is scanned (here a stale fleet, exit 1)."""
    surface = _fleet(tmp_path, stale=True, bare=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert _cli(monkeypatch, surface, []) == 1
    assert capsys.readouterr().out.splitlines()[0] == (
        "Built datasets: 2  (fresh 1, stale 1, unmanifested 0)"
    )


def test_cli_without_any_data_root_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither the flag nor ``$DATA_ROOT`` (and no ``.env``): ``KeyError('DATA_ROOT')``."""
    monkeypatch.delenv("DATA_ROOT", raising=False)
    with pytest.raises(KeyError, match="^'DATA_ROOT'$"):
        _cli(monkeypatch, _surface(SCHEMA), [])


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
            "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()  # fmt: skip


def test_git_info_on_a_throwaway_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean checkout gives (HEAD, False); an untracked file makes it (HEAD, True)."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("a\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "one")
    head = _git(repo, "rev-parse", "HEAD")
    assert len(head) == 40
    assert bm._git_info(repo) == (head, False)
    (repo / "b.txt").write_text("b\n")
    assert bm._git_info(repo) == (head, True)


def test_git_info_outside_a_repo_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not a checkout (ceiling stops the upward search): ``(None, None)``, no raise."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    assert bm._git_info(plain) == (None, None)


# --- A module-level vocabulary is part of the contract a built store recorded (#734) ---
# Shaped like the live chain the issue was filed on:
# ``TransposonInsertionPerturbation -> _validate_bacterial_locus_tag ->
# BACTERIAL_LOCUS_TAG_PATTERN -> BACTERIAL_LOCUS_TAG_PATTERNS``, where the pattern map is
# reached through a validator rather than named in the class body.

VOCAB_SCHEMA = r"""
from typing import Literal

from pydantic import BaseModel, field_validator


class ModelStrict(BaseModel):
    class Config:
        extra = "forbid"


GeneNamespace = Literal["mg1655_locus_tag", "rel606_locus_tag"]

LOCUS_TAG_PATTERNS = {
    "mg1655_locus_tag": r"^b\d{4}$",
    "rel606_locus_tag": r"^ECB_\d{5}$",
}


def validate_locus_tag(namespace: str, locus_tag: str) -> str:
    assert re.match(LOCUS_TAG_PATTERNS[namespace], locus_tag)
    return locus_tag


class Perturbation(ModelStrict):
    # The leaf a loader writes: its namespace comes from the Literal above.
    namespace: GeneNamespace
    locus_tag: str

    @field_validator("locus_tag")
    @classmethod
    def _check(cls, value: str, info: object) -> str:
        return validate_locus_tag("mg1655_locus_tag", value)
"""


def _vocab_manifest(tmp_path: Path, surface: sd.SchemaSurface) -> bm.BuildManifest:
    path = tmp_path / "vocab_loader.py"
    path.write_text(
        "from torchcell.datamodels.schema import Perturbation\n"
        "class MyDataset:\n    pass\n"
    )
    return bm.compute_manifest(
        dataset_name="vocab_slug",
        loader_module="pkg.vocab_loader",
        loader_class="MyDataset",
        loader_path=path,
        surface=surface,
        built_at="2026-10-09T00:00:00+00:00",
        hostname="testhost",
        torchcell_commit=None,
        torchcell_dirty=None,
    )


def test_a_built_store_records_the_vocabularies_its_fields_are_annotated_with(
    tmp_path: Path,
) -> None:
    """The closure holds the Literal alias, the validator and the pattern map it reads."""
    manifest = _vocab_manifest(tmp_path, _surface(VOCAB_SCHEMA))
    assert set(manifest.closure) == {
        "Perturbation",
        "ModelStrict",
        "GeneNamespace",
        "validate_locus_tag",
        "LOCUS_TAG_PATTERNS",
    }


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('"mg1655_locus_tag", "rel606_locus_tag"', '"mg1655_locus_tag"'),
        (
            '"mg1655_locus_tag", "rel606_locus_tag"',
            '"mg1655_locus_tag", "rel606_locus_tag", "w3110_locus_tag"',
        ),
    ],
    ids=["narrowed", "widened"],
)
def test_a_changed_literal_vocabulary_stales_the_store_that_carries_the_field(
    tmp_path: Path, old: str, new: str
) -> None:
    """Issue #734: this edit moved NO fingerprint before the vocabularies were nodes.

    Both directions are reported. A narrowing makes stored records unrepresentable and a
    widening does not, but the gate's job is to say the contract changed; which direction
    is breaking is ``schema_impact``'s verdict, not this one's.

    The drift is on the BINDING, not on ``Perturbation``: a class fingerprint is its own
    declaration, so the class reads unchanged and the vocabulary is named as the symbol
    that moved. That split is what keeps the historical compatibility pairings readable
    while still staling the store.
    """
    manifest = _vocab_manifest(tmp_path, _surface(VOCAB_SCHEMA))
    changed = _surface(VOCAB_SCHEMA.replace(old, new))
    result = bm.check_manifest(manifest, changed, str(tmp_path))
    assert result.is_stale is True
    assert [drift.symbol for drift in result.drift] == ["GeneNamespace"]
    assert result.drift[0].stored_fingerprint == manifest.closure["GeneNamespace"]
    assert result.drift[0].current_fingerprint == changed.fingerprints["GeneNamespace"]
    assert changed.fingerprints["Perturbation"] == manifest.closure["Perturbation"]


def test_a_changed_locus_tag_pattern_stales_the_store(tmp_path: Path) -> None:
    """A pattern a stored namespace already uses changes what its records may say."""
    manifest = _vocab_manifest(tmp_path, _surface(VOCAB_SCHEMA))
    changed = _surface(VOCAB_SCHEMA.replace(r"^b\d{4}$", r"^b\d{4}[a-z]?$"))
    result = bm.check_manifest(manifest, changed, str(tmp_path))
    assert result.is_stale is True
    assert [drift.symbol for drift in result.drift] == ["LOCUS_TAG_PATTERNS"]


def test_a_vocabulary_outside_the_closure_leaves_the_store_fresh(
    tmp_path: Path,
) -> None:
    """Only the vocabularies the loader's closure REACHES gate it."""
    manifest = _vocab_manifest(tmp_path, _surface(VOCAB_SCHEMA))
    unrelated = _surface(VOCAB_SCHEMA + '\nPLASMID_MARKERS = ["kanR", "ampR"]\n')
    result = bm.check_manifest(manifest, unrelated, str(tmp_path))
    assert result.is_stale is False
    assert result.drift == []
