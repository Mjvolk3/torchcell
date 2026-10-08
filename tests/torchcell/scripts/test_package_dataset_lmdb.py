# tests/torchcell/scripts/test_package_dataset_lmdb.py
# [[tests.torchcell.scripts.test_package_dataset_lmdb]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/scripts/test_package_dataset_lmdb.py
"""``scripts/package_dataset_lmdb.py`` on a three-record fake dataset under ``tmp_path``."""

import hashlib
import json
import pickle
import sys
import tarfile
from pathlib import Path

import lmdb
import pytest

from torchcell import __version__
from torchcell.datasets.artifact import ArtifactIndex, archive_name, content_sha256
from torchcell.provenance.build_manifest import MANIFEST_FILENAME, BuildManifest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import package_dataset_lmdb as pkg  # type: ignore[import-not-found]  # noqa: E402

BUILT_AT = "2026-09-14T07:25:14+00:00"
ENV = {"media": {"name": "YPD"}, "temperature": {"value": 30}}
ENV_REF = hashlib.sha256(json.dumps(ENV, sort_keys=True).encode()).hexdigest()


def _manifest(slug: str, closure: dict[str, str] | None = None) -> BuildManifest:
    return BuildManifest(
        dataset_name=slug,
        loader_class="SmfFakeDataset",
        loader_module="tests.fake",
        surface_modules=["schema.py"],
        closure=closure or {},
        built_at=BUILT_AT,
        hostname="gilahyper",
        torchcell_commit="901d5ec31",
        torchcell_dirty=False,
    )


def experiments(n: int = 3) -> list[dict[str, object]]:
    """The resolved ``experiment`` dicts the fake LMDB holds (environment inlined)."""
    return [
        {
            "genotype": {"perturbations": [{"systematic_gene_name": f"YAL00{i}C"}]},
            "environment": ENV,
            "phenotype": {"fitness": 0.5 + i / 10},
        }
        for i in range(n)
    ]


def build_fake_dataset(
    root: Path, slug: str = "smf_fake", manifest: BuildManifest | None = None
) -> Path:
    """``root/data/torchcell/<slug>`` with raw/, preprocess/ (manifest) and processed/.

    Records intern the environment through a ``$ref`` into the sibling ``interned``
    LMDB, the production layout, so the content hash exercises the resolve step.
    """
    dataset_dir = root / "data" / "torchcell" / slug
    (dataset_dir / "raw").mkdir(parents=True)
    (dataset_dir / "raw" / "raw.txt").write_text("publisher file; never packaged")
    (dataset_dir / "preprocess").mkdir()
    (dataset_dir / "preprocess" / "gene_set.json").write_text('["YAL000C"]')
    if manifest is not None:
        (dataset_dir / "preprocess" / MANIFEST_FILENAME).write_text(
            manifest.model_dump_json(indent=2)
        )
    (dataset_dir / "processed" / "lmdb").mkdir(parents=True)
    (dataset_dir / "processed" / "interned").mkdir()
    env = lmdb.open(str(dataset_dir / "processed" / "lmdb"), map_size=int(1e8))
    interned = lmdb.open(str(dataset_dir / "processed" / "interned"), map_size=int(1e8))
    with env.begin(write=True) as txn, interned.begin(write=True) as itxn:
        itxn.put(ENV_REF.encode(), pickle.dumps(ENV))
        for i, experiment in enumerate(experiments()):
            stored = {**experiment, "environment": {"$ref": ENV_REF, "name": "YPD"}}
            record = {"experiment": stored, "reference": {}, "publication": {}}
            txn.put(f"{i}".encode(), pickle.dumps(record))
    env.close()
    interned.close()
    return dataset_dir


def expected_content_sha256() -> str:
    return content_sha256(
        hashlib.sha256(json.dumps(e).encode("utf-8")).hexdigest() for e in experiments()
    )


def test_experiment_ids_resolve_interned_refs(tmp_path: Path) -> None:
    dataset_dir = build_fake_dataset(tmp_path, manifest=_manifest("smf_fake"))
    ids = pkg.experiment_ids(dataset_dir / "processed")
    assert ids == [
        hashlib.sha256(json.dumps(e).encode("utf-8")).hexdigest() for e in experiments()
    ]


def test_package_writes_archive_index_and_sums(tmp_path: Path) -> None:
    dataset_dir = build_fake_dataset(tmp_path, manifest=_manifest("smf_fake"))
    store = tmp_path / "store"
    artifact = pkg.package_dataset(
        dataset_dir,
        store,
        kg_release="2026.09.21-ab6d8c5d",
        kg_version="1.2",
        packaged_at="2026-09-29T10:00:00+00:00",
    )
    archive = store / artifact.rel_path
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert artifact.archive_sha256 == digest
    assert artifact.archive == archive_name("smf_fake", __version__, digest)
    assert artifact.archive_bytes == archive.stat().st_size
    assert (artifact.slug, artifact.dataset_class, artifact.torchcell_version) == (
        "smf_fake",
        "SmfFakeDataset",
        __version__,
    )
    assert (artifact.kg_release, artifact.kg_version) == ("2026.09.21-ab6d8c5d", "1.2")
    assert (artifact.n_experiments, artifact.content_sha256) == (
        3,
        expected_content_sha256(),
    )
    assert (artifact.built_at, artifact.torchcell_commit, artifact.status) == (
        BUILT_AT,
        "901d5ec31",
        "supported",
    )
    assert artifact.build_manifest == _manifest("smf_fake")
    with tarfile.open(archive, "r:xz") as tar:
        members = tar.getmembers()
    assert [m.name for m in members] == [
        "preprocess/build_manifest.json",
        "preprocess/gene_set.json",
        "processed/interned/data.mdb",
        "processed/lmdb/data.mdb",
    ]
    assert {(m.uid, m.gid, m.mode, m.mtime) for m in members} == {
        (0, 0, 0o644, 1789370714)
    }
    index = ArtifactIndex.load(store / "index.json")
    assert index.artifacts == [artifact]
    assert index.generated_at == "2026-09-29T10:00:00+00:00"
    assert (store / "SHA256SUMS").read_text() == f"{digest}  {artifact.rel_path}\n"
    assert not (store / "smf_fake" / ".smf_fake.partial.tar.xz").exists()


def test_packaging_twice_is_byte_identical_and_replaces_the_row(tmp_path: Path) -> None:
    dataset_dir = build_fake_dataset(tmp_path, manifest=_manifest("smf_fake"))
    store = tmp_path / "store"
    first = pkg.package_dataset(dataset_dir, store, packaged_at="t1")
    second = pkg.package_dataset(dataset_dir, store, packaged_at="t2")
    assert second.archive_sha256 == first.archive_sha256
    index = ArtifactIndex.load(store / "index.json")
    assert [a.packaged_at for a in index.artifacts] == ["t2"]
    assert sorted(p.name for p in (store / "smf_fake").iterdir()) == [first.archive]


def test_no_content_hash_leaves_the_field_null(tmp_path: Path) -> None:
    dataset_dir = build_fake_dataset(tmp_path, manifest=_manifest("smf_fake"))
    artifact = pkg.package_dataset(
        dataset_dir, tmp_path / "store", compute_content_hash=False
    )
    assert artifact.content_sha256 is None
    assert artifact.n_experiments == 3


def test_refuses_without_a_build_manifest(tmp_path: Path) -> None:
    dataset_dir = build_fake_dataset(tmp_path, manifest=None)
    with pytest.raises(pkg.PackagingRefused, match="no build_manifest.json under"):
        pkg.package_dataset(dataset_dir, tmp_path / "store")
    assert not (tmp_path / "store").exists()


def test_refuses_a_manifest_named_for_another_dataset(tmp_path: Path) -> None:
    dataset_dir = build_fake_dataset(tmp_path, manifest=_manifest("other_slug"))
    with pytest.raises(
        pkg.PackagingRefused, match="dataset_name 'other_slug' != directory name"
    ):
        pkg.package_dataset(dataset_dir, tmp_path / "store")


def test_refuses_a_stale_manifest(tmp_path: Path) -> None:
    stale = _manifest("smf_fake", closure={"Environment": "deadbeef"})
    dataset_dir = build_fake_dataset(tmp_path, manifest=stale)
    with pytest.raises(pkg.PackagingRefused, match="STALE.*changed: Environment"):
        pkg.package_dataset(dataset_dir, tmp_path / "store")


def test_refuses_without_a_processed_lmdb(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "torchcell" / "smf_fake"
    (dataset_dir / "preprocess").mkdir(parents=True)
    (dataset_dir / "preprocess" / MANIFEST_FILENAME).write_text(
        _manifest("smf_fake").model_dump_json()
    )
    with pytest.raises(pkg.PackagingRefused, match="no processed/lmdb under"):
        pkg.package_dataset(dataset_dir, tmp_path / "store")


def test_main_exits_zero_and_reports_or_one_on_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_dir = build_fake_dataset(tmp_path, manifest=_manifest("smf_fake"))
    store = tmp_path / "store"
    code = pkg.main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--store",
            str(store),
            "--status",
            "deprecated",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("packaged smf_fake (3 records) -> ")
    assert ArtifactIndex.load(store / "index.json").artifacts[0].status == "deprecated"
    bare = build_fake_dataset(tmp_path / "bare", manifest=None)
    assert pkg.main(["--dataset-dir", str(bare), "--store", str(store)]) == 1
    assert capsys.readouterr().err.startswith("refused: no build_manifest.json")


def test_refuses_a_private_loader_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dataset whose loader class is private is never published in a release.

    The gate reads ``visibility`` off the CLASS named by the build manifest, so no
    spelling of the command publishes in-house data. ``loader_class`` is registered here
    under its own name, which is how a real private loader reaches the registry.
    """
    from torchcell.data.experiment_dataset import Visibility
    from torchcell.datasets.dataset_registry import dataset_registry

    class PrivateFakeDataset:
        visibility = Visibility.private

    monkeypatch.setitem(dataset_registry, "SmfFakeDataset", PrivateFakeDataset)
    dataset_dir = build_fake_dataset(tmp_path, manifest=_manifest("smf_fake"))
    with pytest.raises(
        pkg.PackagingRefused, match="SmfFakeDataset is PRIVATE .visibility=private."
    ):
        pkg.package_dataset(dataset_dir, tmp_path / "store")
    assert not (tmp_path / "store").exists()


def test_refuses_the_real_private_bioscreen_dataset(tmp_path: Path) -> None:
    """The in-house 2021 Bioscreen loader, registered by importing its package, is
    refused by name: an --include-private graph build does not open the release path.
    """
    import torchcell.datasets.private_torchcell  # noqa: F401  # registers the loader

    manifest = _manifest("inhibitor_bioscreen_volk2021").model_copy(
        update={"loader_class": "InhibitorBioscreenVolk2021Dataset"}
    )
    dataset_dir = build_fake_dataset(
        tmp_path, slug="inhibitor_bioscreen_volk2021", manifest=manifest
    )
    with pytest.raises(
        pkg.PackagingRefused,
        match="InhibitorBioscreenVolk2021Dataset is PRIVATE .visibility=private.",
    ):
        pkg.package_dataset(dataset_dir, tmp_path / "store")
    assert not (tmp_path / "store").exists()


def test_a_public_or_unregistered_loader_class_packages_normally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``public`` passes, and so does a class the registry does not hold at all.

    An unregistered ``loader_class`` is not private, it is unregistered, which is a
    different and pre-existing condition; the visibility gate must not start refusing it.
    """
    from torchcell.data.experiment_dataset import Visibility
    from torchcell.datasets.dataset_registry import dataset_registry

    class PublicFakeDataset:
        visibility = Visibility.public

    monkeypatch.setitem(dataset_registry, "SmfFakeDataset", PublicFakeDataset)
    public_dir = build_fake_dataset(tmp_path / "public", manifest=_manifest("smf_fake"))
    assert pkg.package_dataset(public_dir, tmp_path / "store").slug == "smf_fake"
    monkeypatch.delitem(dataset_registry, "SmfFakeDataset", raising=False)
    bare_dir = build_fake_dataset(tmp_path / "bare", manifest=_manifest("smf_fake"))
    assert pkg.package_dataset(bare_dir, tmp_path / "store2").slug == "smf_fake"
