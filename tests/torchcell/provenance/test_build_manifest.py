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

2026.10.09 (#833). Every synthetic store now holds a real one-record LMDB, since
``check_store`` ends in a bounded read of the first record and ``fresh`` therefore means
readable as well as fingerprint-equal. Added: the read resolves an interned ``$ref``
reference; a record (and an interned sub-object) pickled under a class a throwaway module
carried and no longer does reads ``unreadable: ModuleNotFoundError: ...`` while every
fingerprint still matches; a record carrying a field its class forbids reads
``unreadable`` with ``extra_forbidden``; an empty LMDB reads ``unreadable``; the states
are decided in order and a stale store is never read (pinned with a read that raises);
and the fleet CLI prints the ``[UNREADABLE]`` line and exits 1.
"""

from __future__ import annotations

import importlib.util
import pickle
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import lmdb
import pydantic
import pytest

from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    Publication,
    ReferenceGenome,
)
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


_ENVIRONMENT = Environment(media=Media(name="YPD", state="solid", is_synthetic=False))
_PUBLICATION = Publication(pubmed_id="1", pubmed_url="u", doi="d", doi_url="du")


def _fitness_record(gene: str = "YAL001C") -> dict[str, Any]:
    """One stored record, exactly as a loader serializes it: three dumped dicts."""
    experiment = FitnessExperiment(
        dataset_name="ds",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=gene
                )
            ]
        ),
        environment=_ENVIRONMENT,
        phenotype=FitnessPhenotype(fitness=0.5),
    )
    reference = FitnessExperimentReference(
        dataset_name="ds",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=FitnessPhenotype(fitness=1.0),
    )
    return {
        "experiment": experiment.model_dump(),
        "reference": reference.model_dump(),
        "publication": _PUBLICATION.model_dump(),
    }


def _write_store(
    slug_dir: Path, records: list[Any], interned: dict[str, Any] | None = None
) -> None:
    """Write ``records`` into ``processed/lmdb`` (and ``interned`` into its sibling env).

    A record given as ``bytes`` is stored as those bytes, so a test can plant a pickle
    this process can write and a later read cannot resolve.
    """
    env = lmdb.open(str(slug_dir / "processed" / "lmdb"), map_size=10**7)
    with env.begin(write=True) as txn:
        for i, record in enumerate(records):
            txn.put(
                f"{i}".encode(),
                record if isinstance(record, bytes) else pickle.dumps(record),
            )
    env.close()
    if interned is None:
        return
    ienv = lmdb.open(str(slug_dir / "processed" / "interned"), map_size=10**7)
    with ienv.begin(write=True) as txn:
        for ref, value in interned.items():
            txn.put(
                ref.encode(), value if isinstance(value, bytes) else pickle.dumps(value)
            )
    ienv.close()


def _build_dataset_dir(root: Path, slug: str) -> Path:
    """A built store: one readable fitness record, so ``fresh`` means readable too."""
    slug_dir = root / "data" / "torchcell" / slug
    (slug_dir / "processed" / "lmdb").mkdir(parents=True)
    (slug_dir / "preprocess").mkdir(parents=True)
    _write_store(slug_dir, [_fitness_record()])
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
        "ds_bare": "no_manifest",
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
        "Built datasets: 1  (fresh 1, stale 0, unreadable 0, unmanifested 0)\n"
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
        "Built datasets: 3  (fresh 1, stale 1, unreadable 0, unmanifested 1)\n"
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
        "Built datasets: 2  (fresh 1, stale 0, unreadable 0, unmanifested 1)",
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
        "Built datasets: 2  (fresh 1, stale 1, unreadable 0, unmanifested 0)"
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


# --------------------------------------------------------------------------- #
# The bounded first-record read: the unreadable state (#833)
# --------------------------------------------------------------------------- #
LOST_MODULE = "tc_lost_schema_module"


@pytest.fixture
def lost_class(monkeypatch: pytest.MonkeyPatch) -> type:
    """A class that pickles now and cannot be unpickled once the fixture is torn down.

    The shape of the five bacterial stores of #833: records were written under a schema
    that declared ``Censoring`` / ``BacterialVariantType`` / ``FoldChangeScale``, and
    the code that reads them does not declare it. Standing in a throwaway module makes
    that hermetic: the pickle names ``tc_lost_schema_module.Censoring``, and the module
    is gone by the time anything reads it.
    """
    module = types.ModuleType(LOST_MODULE)

    class Censoring:
        pass

    Censoring.__module__ = LOST_MODULE
    Censoring.__qualname__ = "Censoring"  # else pickle names a local object
    module.Censoring = Censoring  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, LOST_MODULE, module)
    return Censoring


def _fresh_manifest(tmp_path: Path, slug_dir: Path, surface: sd.SchemaSurface) -> None:
    _write(slug_dir, _manifest(tmp_path, surface, name=slug_dir.name))


def test_read_first_record_resolves_the_interned_reference(tmp_path: Path) -> None:
    """The probe is the store's own read path: the ``$ref`` is spliced back in.

    A loader interns the reference object (one per dataset) and leaves a pointer in
    every record, so a reader that skipped the interned env would hand pydantic
    ``{"$ref": ...}`` and fail a readable store.
    """
    slug_dir = tmp_path / "data" / "torchcell" / "ds_interned"
    (slug_dir / "processed" / "lmdb").mkdir(parents=True)
    (slug_dir / "preprocess").mkdir(parents=True)
    record = _fitness_record()
    reference = record["reference"]
    _write_store(
        slug_dir,
        [{**record, "reference": {"$ref": "abc123", "name": "ds"}}],
        interned={"abc123": reference},
    )
    assert bm.read_first_record(slug_dir) == record


def test_a_record_pickled_under_a_class_the_schema_lost_reads_unreadable(
    tmp_path: Path, lost_class: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #833 store: the manifest compares equal and only the read sees the problem.

    The fingerprints match (the lost class is in no closure, so nothing drifted), so
    ``stale`` cannot name it; the first-record read raises ``ModuleNotFoundError`` and
    the verdict is ``unreadable`` with that exception on it.
    """
    surface = _surface(SCHEMA)
    slug_dir = tmp_path / "data" / "torchcell" / "ds_lost"
    (slug_dir / "processed" / "lmdb").mkdir(parents=True)
    (slug_dir / "preprocess").mkdir(parents=True)
    record = _fitness_record()
    record["experiment"]["phenotype"]["censoring"] = lost_class()
    _write_store(slug_dir, [record])
    _fresh_manifest(tmp_path, slug_dir, surface)
    monkeypatch.delitem(sys.modules, LOST_MODULE)

    freshness = bm.check_store(slug_dir, surface)
    assert freshness.state == "unreadable"
    assert freshness.drift == []
    assert freshness.reason == (f"ModuleNotFoundError: No module named '{LOST_MODULE}'")
    assert freshness.needs_rebuild is True
    assert freshness.describe() == f"unreadable: {freshness.reason}"
    with pytest.raises(ModuleNotFoundError):
        bm.read_first_record(slug_dir)


def test_an_interned_sub_object_of_a_lost_class_reads_unreadable(
    tmp_path: Path, lost_class: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interned env is read first, so a lost class there is caught as well.

    The reference and publication objects live in ``processed/interned``, which is
    where a per-dataset constant built under an extended schema sits.
    """
    surface = _surface(SCHEMA)
    slug_dir = tmp_path / "data" / "torchcell" / "ds_lost_interned"
    (slug_dir / "processed" / "lmdb").mkdir(parents=True)
    (slug_dir / "preprocess").mkdir(parents=True)
    record = _fitness_record()
    _write_store(
        slug_dir,
        [{**record, "reference": {"$ref": "abc123", "name": "ds"}}],
        interned={"abc123": {**record["reference"], "censoring": lost_class()}},
    )
    _fresh_manifest(tmp_path, slug_dir, surface)
    monkeypatch.delitem(sys.modules, LOST_MODULE)
    assert bm.check_store(slug_dir, surface).reason == (
        f"ModuleNotFoundError: No module named '{LOST_MODULE}'"
    )


def test_a_record_carrying_a_field_its_class_forbids_reads_unreadable(
    tmp_path: Path,
) -> None:
    """The second way a store goes unreadable: ``extra_forbidden`` on a stored field.

    A field removed from the schema leaves records that unpickle (dicts always do) and
    fail validation, which is why the probe types the record through the class its own
    ``experiment_type`` names rather than stopping at ``pickle.loads``.
    """
    surface = _surface(SCHEMA)
    slug_dir = tmp_path / "data" / "torchcell" / "ds_extra"
    (slug_dir / "processed" / "lmdb").mkdir(parents=True)
    (slug_dir / "preprocess").mkdir(parents=True)
    record = _fitness_record()
    record["experiment"]["retired_field"] = 1
    _write_store(slug_dir, [record])
    _fresh_manifest(tmp_path, slug_dir, surface)

    freshness = bm.check_store(slug_dir, surface)
    assert freshness.state == "unreadable"
    assert freshness.reason is not None
    assert freshness.reason.startswith("ValidationError: ")
    assert "extra_forbidden" in freshness.reason
    assert "retired_field" in freshness.reason


def test_a_store_with_no_records_reads_unreadable(tmp_path: Path) -> None:
    """An LMDB that holds nothing is not a store the graph can be built from."""
    surface = _surface(SCHEMA)
    slug_dir = tmp_path / "data" / "torchcell" / "ds_empty"
    (slug_dir / "processed" / "lmdb").mkdir(parents=True)
    (slug_dir / "preprocess").mkdir(parents=True)
    _write_store(slug_dir, [])
    _fresh_manifest(tmp_path, slug_dir, surface)
    freshness = bm.check_store(slug_dir, surface)
    assert freshness.state == "unreadable"
    assert freshness.reason == (
        f"ValueError: {slug_dir / 'processed' / 'lmdb'} holds no records"
    )


def test_check_store_states_in_order_and_reads_only_a_fingerprint_fresh_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existence, then the manifest, then the fingerprints, then the read.

    The read runs last and only when every fingerprint matches: a store already
    reported ``stale`` is rebuilt either way, so the probe never pays for it. Pinned by
    making the read raise -- a stale store still reports ``stale``.
    """
    surface = _surface(SCHEMA)
    unbuilt = tmp_path / "data" / "torchcell" / "ds_unbuilt"
    unbuilt.mkdir(parents=True)
    assert bm.check_store(unbuilt, surface).state == "no_lmdb"

    bare = _build_dataset_dir(tmp_path, "ds_bare")
    assert bm.check_store(bare, surface).state == "no_manifest"

    stale_dir = _build_dataset_dir(tmp_path, "ds_stale")
    _write(
        stale_dir,
        _manifest(tmp_path, surface, name="ds_stale").model_copy(
            update={"closure": {"Media": "deadbeef"}}
        ),
    )

    def refuse(_root: str | Path) -> dict[str, Any]:
        raise AssertionError("the read must not run on a stale store")

    monkeypatch.setattr(bm, "read_first_record", refuse)
    stale = bm.check_store(stale_dir, surface)
    assert (stale.state, stale.drift) == ("stale", ["Media"])


def test_the_fleet_scan_and_cli_report_an_unreadable_store_and_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    lost_class: type,
) -> None:
    """``python -m torchcell.provenance.build_manifest`` names it and exits 1.

    The CLI is what the live rebuild's runbook has the owner run before a full build,
    so an unreadable store has to fail it: before this it exited 0 on all five.
    """
    surface = _fleet(tmp_path, stale=False, bare=False)
    slug_dir = tmp_path / "data" / "torchcell" / "ds_lost"
    (slug_dir / "processed" / "lmdb").mkdir(parents=True)
    (slug_dir / "preprocess").mkdir(parents=True)
    record = _fitness_record()
    record["experiment"]["phenotype"]["censoring"] = lost_class()
    _write_store(slug_dir, [record])
    _write(slug_dir, _manifest(tmp_path, surface, name="ds_lost"))
    monkeypatch.delitem(sys.modules, LOST_MODULE)

    assert _cli(monkeypatch, surface, ["--data-root", str(tmp_path)]) == 1
    assert capsys.readouterr().out.splitlines() == [
        "Built datasets: 2  (fresh 1, stale 0, unreadable 1, unmanifested 0)",
        "",
        "  [UNREADABLE] ds_lost  -> rebuild; unreadable: ModuleNotFoundError: "
        f"No module named '{LOST_MODULE}'",
    ]


def test_describe_words_every_state_for_a_preflight_refusal(tmp_path: Path) -> None:
    """The line the live rebuild's preflight refuses with, one per state.

    The preflight prints these and exits; the wording is what tells the owner whether
    to rebuild a store, build it for the first time, or look at what the records carry.
    """
    assert bm.StoreFreshness(root="/r", state="fresh").describe() == "fresh"
    assert (
        bm.StoreFreshness(root="/r", state="stale", drift=["Media"]).describe()
        == "stale on ['Media']"
    )
    assert bm.StoreFreshness(root="/r", state="no_lmdb").describe() == "no LMDB at /r"
    assert (
        bm.StoreFreshness(root="/r", state="no_manifest").describe()
        == "no build manifest"
    )
    assert (
        bm.StoreFreshness(
            root="/r", state="unreadable", reason="KeyError: 'x'"
        ).describe()
        == "unreadable: KeyError: 'x'"
    )
