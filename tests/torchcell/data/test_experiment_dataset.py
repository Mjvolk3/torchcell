# tests/torchcell/data/test_experiment_dataset.py
# [[tests.torchcell.data.test_experiment_dataset]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_experiment_dataset.py
"""``ExperimentDataset`` end to end on a three-record toy loader under ``tmp_path``.

2026.09.30 (Phase 15). ``ToyDataset`` is the smallest concrete subclass whose
``process`` (decorated with ``post_process``, as every real loader is) writes three
fitness records through the base class's own ``_open_write_lmdb`` and
``_intern_record``, so the build, the interning, the gene set, the reference index and
the build manifest all run through the production code:

- record 0: deletion of YAL001C, fitness 0.5, reference A (reference fitness 1.0);
- record 1: deletions of YAL002W and YAL001C, fitness 0.25, reference A;
- record 2: deletion of YBR001C, fitness 0.75, reference B (reference fitness 0.9).

Every experiment and both references carry the SGA selection environment, whose
canonical JSON is 9,544 bytes (over ``INTERN_MIN_BYTES`` = 512), so the environment is
interned once; each reference is 10,031 bytes and is interned once per distinct
reference; the publication is 66 bytes and stays inline. The interned env therefore
holds exactly 3 rows (one environment, two references), and a stored record carries
``{"$ref": <sha256>, "name": <hint>}`` where the hint is the media name for the
environment and the ``dataset_name`` for the reference. Reading resolves the pointers
back, so ``dataset[i]`` equals the plain ``model_dump()`` of the three objects.

Derived values: the gene set is the sorted union ``["YAL001C", "YAL002W", "YBR001C"]``;
the reference index groups by first sighting, ``A -> [0, 1]`` then ``B -> [2]``; a stale
index covering only record 0 leaves records 1 and 2 uncovered and fails the build's
coverage assertion. The tc-data download path is exercised with a fake
``DatasetClient`` (no socket) whose archives are built here with ``tarfile``.

2026.09.30 (issues #518, #524, #528, #537): the raw-file sha256 helpers every pinned
loader shares. ``b"abc"`` hashes to ``ba7816bf...15ad`` (the FIPS 180-2 test vector);
each refusal is a ``RawSha256MismatchError`` whose message names the file (or URL), the
expected and the observed digest, and leaves the destination exactly as it was.
"""

import io
import json
import pickle
import re
import socket
import tarfile
from pathlib import Path
from typing import Any, ClassVar

import lmdb
import numpy as np
import pandas as pd
import pytest

import torchcell.provenance.build_manifest as build_manifest
from torchcell import __version__
from torchcell.data import compute_sha256_hash
from torchcell.data.experiment_dataset import (
    ExperimentDataset,
    RawSha256MismatchError,
    _compute_reference_hash_parallel,
    canonical_json,
    copy_verified,
    file_sha256,
    link_verified,
    post_process,
    process_reference_batch,
    serialize_for_hashing,
    verify_raw_files,
    verify_sha256,
    write_verified,
)
from torchcell.datamodels.media import SGA_DM_SELECTION
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.artifact import DatasetArtifact
from torchcell.datasets.client import DatasetClient
from torchcell.provenance.build_manifest import MANIFEST_FILENAME
from torchcell.sequence import GeneSet

ENVIRONMENT = Environment(media=SGA_DM_SELECTION, temperature=Temperature(value=26))
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
PUBLICATION = Publication(pubmed_id="1", pubmed_url="u", doi="d", doi_url="du")


def _reference(fitness: float) -> FitnessExperimentReference:
    return FitnessExperimentReference(
        dataset_name="Toy",
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT,
        phenotype_reference=FitnessPhenotype(fitness=fitness),
    )


def _experiment(genes: list[str], fitness: float) -> FitnessExperiment:
    return FitnessExperiment(
        dataset_name="Toy",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(systematic_gene_name=g, perturbed_gene_name=g)
                for g in genes
            ]
        ),
        environment=ENVIRONMENT,
        phenotype=FitnessPhenotype(fitness=fitness, fitness_std=0.1),
    )


REF_A = _reference(1.0)
REF_B = _reference(0.9)
RECORDS: list[tuple[FitnessExperiment, FitnessExperimentReference]] = [
    (_experiment(["YAL001C"], 0.5), REF_A),
    (_experiment(["YAL002W", "YAL001C"], 0.25), REF_A),
    (_experiment(["YBR001C"], 0.75), REF_B),
]


def _dumped(i: int) -> dict[str, Any]:
    experiment, reference = RECORDS[i]
    return {
        "experiment": experiment.model_dump(),
        "reference": reference.model_dump(),
        "publication": PUBLICATION.model_dump(),
    }


class ToyDataset(ExperimentDataset):
    """Three fitness records written through the base class's interning writer."""

    calls: ClassVar[list[str]] = []

    @property
    def experiment_class(self) -> type[FitnessExperiment]:
        """Fitness records."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[FitnessExperimentReference]:
        """Fitness references."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """One marker file written by ``download``."""
        return ["raw.txt"]

    def download(self) -> None:
        """The publisher path: record the call and write the marker file."""
        ToyDataset.calls.append("download")
        Path(self.raw_dir, "raw.txt").write_text("toy\n")

    @post_process
    def process(self) -> None:
        """Write the three records, interned txn committed before the records txn."""
        ToyDataset.calls.append("process")
        Path(self.preprocess_dir).mkdir(parents=True, exist_ok=True)
        env, interned_env = self._open_write_lmdb(f"{self.processed_dir}/lmdb")
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for i, (experiment, reference) in enumerate(RECORDS):
                txn.put(
                    f"{i}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
        env.close()
        interned_env.close()

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Unused: the toy has no raw table."""
        raise NotImplementedError

    def create_experiment(self) -> None:
        """Unused: records come from ``RECORDS``."""
        raise NotImplementedError


@pytest.fixture
def no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest's commit lookup shells out to git; pin it instead."""
    monkeypatch.setattr(build_manifest, "_git_info", lambda _: ("c0ffee", False))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    ToyDataset.calls.clear()


def _build(tmp_path: Path, io_workers: int = 0) -> ToyDataset:
    return ToyDataset(root=str(tmp_path / "toy_slug"), io_workers=io_workers)


def test_build_runs_download_then_process_once_and_reads_back_the_records(
    tmp_path: Path, no_git: None
) -> None:
    """The publisher download runs (TC_DATA_URL unset), then ``process``; each record
    reads back as the plain dump of its three objects, ``len`` is 3 and ``repr`` is
    ``ToyDataset(3)``. A second construction on the same root finds the LMDB and runs
    neither.
    """
    dataset = _build(tmp_path)
    assert ToyDataset.calls == ["download", "process"]
    assert len(dataset) == 3
    assert repr(dataset) == "ToyDataset(3)"
    assert [dataset[i] for i in range(3)] == [_dumped(i) for i in range(3)]
    dataset.close_lmdb()  # the CI py-lmdb refuses a second open of one path per process
    again = _build(tmp_path)
    assert ToyDataset.calls == ["download", "process"]
    assert again[2] == _dumped(2)


def test_heavy_sub_objects_are_interned_once_and_small_ones_stay_inline(
    tmp_path: Path, no_git: None
) -> None:
    """The stored bytes: the environment and each reference are ``$ref`` pointers named
    by the media name and the dataset name, the 66-byte publication is inline, and the
    sibling ``interned`` env holds 3 rows (one environment, references A and B).
    """
    dataset = _build(tmp_path)
    env_digest = compute_sha256_hash(canonical_json(ENVIRONMENT))
    ref_digests = [compute_sha256_hash(canonical_json(r)) for r in (REF_A, REF_B)]
    env = lmdb.open(f"{dataset.processed_dir}/lmdb", readonly=True, lock=False)
    with env.begin() as txn:
        raw_bytes = [txn.get(f"{i}".encode()) for i in range(3)]
    env.close()
    stored = []
    for value in raw_bytes:
        assert value is not None
        stored.append(pickle.loads(value))
    assert [s["experiment"]["environment"] for s in stored] == [
        {"$ref": env_digest, "name": SGA_DM_SELECTION.name}
    ] * 3
    assert [s["reference"] for s in stored] == [
        {"$ref": ref_digests[0], "name": "Toy"},
        {"$ref": ref_digests[0], "name": "Toy"},
        {"$ref": ref_digests[1], "name": "Toy"},
    ]
    assert stored[0]["publication"] == PUBLICATION.model_dump()
    ienv = lmdb.open(f"{dataset.processed_dir}/interned", readonly=True, lock=False)
    with ienv.begin() as itxn:
        interned_keys = sorted(key.decode() for key, _ in itxn.cursor())
    ienv.close()
    assert interned_keys == sorted([env_digest, *ref_digests])


def test_get_accepts_an_index_list_a_bool_mask_and_returns_none_past_the_end(
    tmp_path: Path, no_git: None
) -> None:
    """``get([0, 2])`` and ``get(mask [T, F, T])`` both return records 0 and 2; a key the
    LMDB does not hold (99) is None, not an error.
    """
    dataset = _build(tmp_path)
    expected = [_dumped(0), _dumped(2)]
    assert dataset.get([0, 2]) == expected
    assert dataset.get(np.array([True, False, True])) == expected
    assert dataset.get(99) is None


def test_post_process_writes_the_sorted_gene_set_and_the_reference_index(
    tmp_path: Path, no_git: None
) -> None:
    """``gene_set.json`` is the sorted union of the systematic names; the reference index
    groups by first sighting (A: [0, 1], B: [2]) and is written in member-index form.
    Once loaded it is cached: deleting the file afterwards returns the cached index and
    does not recompute or rewrite it.
    """
    dataset = _build(tmp_path)
    preprocess = Path(dataset.preprocess_dir)
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YAL002W",
        "YBR001C",
    ]
    assert dataset.gene_set == GeneSet(["YAL001C", "YAL002W", "YBR001C"])
    stored_index = json.loads(
        (preprocess / "experiment_reference_index.json").read_text()
    )
    assert [row["member_indices"] for row in stored_index] == [[0, 1], [2]]
    index = dataset.experiment_reference_index
    assert index is not None
    assert [(eri.reference, eri.member_indices) for eri in index] == [
        (REF_A, [0, 1]),
        (REF_B, [2]),
    ]
    (preprocess / "experiment_reference_index.json").unlink()
    assert dataset.experiment_reference_index == index
    assert not (preprocess / "experiment_reference_index.json").exists()


def test_post_process_writes_the_build_manifest_for_the_loader(
    tmp_path: Path, no_git: None
) -> None:
    """The manifest names the root's basename, this loader's class and module, the pinned
    commit and the host.
    """
    dataset = _build(tmp_path)
    manifest = json.loads(
        (Path(dataset.preprocess_dir) / MANIFEST_FILENAME).read_text()
    )
    assert (
        manifest["dataset_name"],
        manifest["loader_class"],
        manifest["loader_module"],
        manifest["torchcell_commit"],
        manifest["torchcell_dirty"],
        manifest["hostname"],
    ) == ("toy_slug", "ToyDataset", __name__, "c0ffee", False, socket.gethostname())


def test_a_stale_reference_index_fails_the_coverage_check(
    tmp_path: Path, no_git: None
) -> None:
    """An index left in ``preprocess/`` that covers only record 0 is loaded instead of
    recomputed, leaves records 1 and 2 uncovered, and the build refuses.
    """
    preprocess = tmp_path / "toy_slug" / "preprocess"
    preprocess.mkdir(parents=True)
    (preprocess / "experiment_reference_index.json").write_text(
        json.dumps([{"reference": REF_A.model_dump(), "member_indices": [0]}])
    )
    with pytest.raises(
        AssertionError,
        match="Each item in the dataset must be covered by exactly one reference.",
    ):
        _build(tmp_path)


def test_parallel_workers_build_the_same_gene_set_and_index(
    tmp_path: Path, no_git: None
) -> None:
    """``io_workers=2`` takes the multiprocessing gene-set loader and the parallel index
    entry point; both must equal the sequential build on the same records.
    """
    parallel = _build(tmp_path / "parallel", io_workers=2)
    sequential = _build(tmp_path / "sequential")
    assert parallel.compute_gene_set() == sequential.compute_gene_set()
    assert parallel.compute_gene_set() == GeneSet(["YAL001C", "YAL002W", "YBR001C"])
    parallel_index = parallel.experiment_reference_index
    sequential_index = sequential.experiment_reference_index
    assert parallel_index is not None and sequential_index is not None
    assert [e.model_dump() for e in parallel_index] == [
        e.model_dump() for e in sequential_index
    ]


def test_gene_set_file_is_authoritative_and_an_empty_set_is_refused(
    tmp_path: Path, no_git: None
) -> None:
    """``gene_set`` re-reads ``gene_set.json`` on every access (an edit on disk shows up);
    assigning an empty set raises before anything is written; with the file gone the
    getter recomputes the three genes from the LMDB and does not write the file back.
    """
    dataset = _build(tmp_path)
    path = Path(dataset.preprocess_dir) / "gene_set.json"
    path.write_text(json.dumps(["YAL001C"]))
    assert dataset.gene_set == GeneSet(["YAL001C"])
    with pytest.raises(ValueError, match="Cannot set an empty gene_set"):
        dataset.gene_set = GeneSet()
    assert json.loads(path.read_text()) == ["YAL001C"]
    path.unlink()
    assert dataset.gene_set == GeneSet(["YAL001C", "YAL002W", "YBR001C"])
    assert not path.exists()


def test_a_loader_without_gene_perturbations_may_declare_an_empty_gene_set(
    tmp_path: Path, no_git: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``has_gene_perturbations = False`` is the one way an empty gene set is accepted;
    ``None`` is refused either way, and the declaration is a class-level fact.
    """
    dataset = _build(tmp_path)
    assert ToyDataset.has_gene_perturbations is True
    monkeypatch.setattr(ToyDataset, "has_gene_perturbations", False)
    dataset.gene_set = GeneSet()
    path = Path(dataset.preprocess_dir) / "gene_set.json"
    assert json.loads(path.read_text()) == []
    assert dataset.gene_set == GeneSet()
    with pytest.raises(ValueError, match="Cannot set None"):
        dataset.gene_set = None  # type: ignore[assignment]


def test_df_is_none_until_preprocess_holds_data_csv(
    tmp_path: Path, no_git: None
) -> None:
    """``df`` reads ``preprocess/data.csv`` when it exists and is None before."""
    dataset = _build(tmp_path)
    assert dataset.df is None
    frame = pd.DataFrame({"gene": ["YAL001C", "YBR001C"], "fitness": [0.5, 0.75]})
    frame.to_csv(Path(dataset.preprocess_dir) / "data.csv", index=False)
    loaded = dataset.df
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, frame)


def test_transform_item_round_trips_a_stored_record_to_typed_objects(
    tmp_path: Path, no_git: None
) -> None:
    """Record 1 comes back as the exact ``FitnessExperiment``, reference and publication
    it was built from.
    """
    dataset = _build(tmp_path)
    item = dataset.transform_item(dataset[1])
    assert item == {
        "experiment": RECORDS[1][0],
        "reference": REF_A,
        "publication": PUBLICATION,
    }
    assert type(item["experiment"]) is FitnessExperiment
    assert type(item["reference"]) is FitnessExperimentReference


def test_transform_item_builds_the_reference_once(
    tmp_path: Path, no_git: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``transform_item`` validates each item's reference exactly once (issue #532 removed
    a repeated construction whose result was discarded) and returns it equal to the
    stored reference. The count is taken by wrapping the real class's ``__init__`` for
    this test only (a counting subclass would enter the schema's subclass discovery and
    break the ontology tree tests).
    """
    dataset = _build(tmp_path)
    original_init = FitnessExperimentReference.__init__
    constructions: list[int] = []

    def counting_init(self: FitnessExperimentReference, **data: Any) -> None:
        constructions.append(1)
        original_init(self, **data)

    monkeypatch.setattr(FitnessExperimentReference, "__init__", counting_init)
    item = dataset.transform_item(dataset[0])
    assert len(constructions) == 1
    assert item["reference"] == REF_A


def test_serialize_for_hashing_sorts_a_model_at_every_depth_like_its_dump() -> None:
    """A reference MODEL serializes as ``json.dumps(model_dump(), sort_keys=True)``, the
    same string as its dump, so the two hash identically (issue #532; the model branch
    used to sort only the top-level keys). The dict path the reference index takes is
    unchanged: the three fixture records hash to the literal ids the pre-fix code
    produced (records 0 and 1 share ``REF_A``), derived before the change. 2026.10.02
    (issue #602): the ids were re-derived once ``FitnessPhenotype`` gained
    ``screen_id``, which every reference dump now carries as ``null``. 2026.10.09
    (issue #753): re-derived again once ``Environment`` gained
    ``dilution_rate_per_hour`` and ``ProvenanceGap`` gained ``keys``, both of which a
    reference dump now carries. A content address is supposed to move when the content's
    shape does, which is the BREAKING verdict ``scripts/schema_impact_check.py`` reports.
    """
    as_model = serialize_for_hashing(REF_A)
    as_dict = serialize_for_hashing(REF_A.model_dump())
    assert as_dict == json.dumps(REF_A.model_dump(), sort_keys=True)
    assert as_model == as_dict
    assert compute_sha256_hash(as_model) == compute_sha256_hash(as_dict)
    assert process_reference_batch([_dumped(i) for i in range(3)]) == [
        "59bbf8bc9d6e83acd45578bc3a2bd1a584b2be11386a1547b8062296a9f566d7",
        "59bbf8bc9d6e83acd45578bc3a2bd1a584b2be11386a1547b8062296a9f566d7",
        "211b373fad48862c5ad09323ff5fcf8cf03bff837a825bde856d15ce253b2045",
    ]


def test_reference_ids_ignore_key_order_at_every_depth() -> None:
    """A reference id depends on content only: the same reference written with its
    keys in another order, at the top level and inside a nested dictionary, gets the
    same id from both helpers, equal references share an id, and a different reference
    gets another. This is what ``sort_keys=True`` buys and what a plain ``json.dumps``
    would break.
    """
    shuffled = [
        {"reference": {"b": 1, "a": {"y": 2, "x": 3}}},
        {"reference": {"a": {"x": 3, "y": 2}, "b": 1}},
        {"reference": {"a": {"x": 3, "y": 2}, "b": 2}},
    ]
    batch = process_reference_batch(shuffled)
    assert batch[0] == batch[1] != batch[2]
    assert [_compute_reference_hash_parallel(i) for i in shuffled] == batch
    items = [_dumped(0), _dumped(1), _dumped(2)]
    ids = process_reference_batch(items)
    assert ids[0] == ids[1] != ids[2]


# --- the tc-data download path with a fake client --------------------------------- #
def _tar_xz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _packaged_members(source: ToyDataset) -> dict[str, bytes]:
    """processed/ and preprocess/ of a built toy, as an archive would carry them."""
    root = Path(source.root)
    members: dict[str, bytes] = {}
    for sub in ("processed", "preprocess"):
        for path in sorted((root / sub).rglob("*")):
            if path.is_file() and not path.name.endswith(".pt"):
                members[str(path.relative_to(root))] = path.read_bytes()
    return members


class _FakeClient:
    """Stands in for ``DatasetClient``: answers ``select`` and writes ``download``."""

    url = "http://tc-data.test"

    def __init__(self, artifact: DatasetArtifact | None, archive: bytes) -> None:
        self.artifact = artifact
        self.archive = archive
        self.selected: list[str] = []
        self.downloaded: list[Path] = []

    def select(self, slug: str) -> DatasetArtifact | None:
        self.selected.append(slug)
        return self.artifact

    def download(self, artifact: DatasetArtifact, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.archive)
        self.downloaded.append(dest)
        return dest


def _artifact(dataset_class: str = "ToyDataset") -> DatasetArtifact:
    return DatasetArtifact(
        slug="toy_slug",
        dataset_class=dataset_class,
        torchcell_version=__version__,
        n_experiments=3,
        archive="toy_slug-archive.tar.xz",
        archive_sha256="a" * 64,
        archive_bytes=1,
        built_at="2026-09-30T00:00:00+00:00",
        packaged_at="2026-09-30T00:00:00+00:00",
    )


def _use_fake(monkeypatch: pytest.MonkeyPatch, fake: _FakeClient) -> None:
    monkeypatch.setenv("TC_DATA_URL", fake.url)
    monkeypatch.setattr(
        DatasetClient, "from_env", classmethod(lambda cls, http=None: fake)
    )


def test_endpoint_artifact_is_unpacked_in_place_of_download_and_process(
    tmp_path: Path, no_git: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With TC_DATA_URL set, the root's basename is the slug asked for, the archive lands
    at ``<root>/artifacts/<archive>``, and the unpacked LMDB serves the three records;
    neither ``download`` nor ``process`` runs.
    """
    source = _build(tmp_path / "source")
    fake = _FakeClient(_artifact(), _tar_xz(_packaged_members(source)))
    ToyDataset.calls.clear()
    _use_fake(monkeypatch, fake)
    root = tmp_path / "client" / "toy_slug"
    dataset = ToyDataset(root=str(root))
    assert ToyDataset.calls == []
    assert fake.selected == ["toy_slug"]
    assert fake.downloaded == [root / "artifacts" / "toy_slug-archive.tar.xz"]
    assert [dataset[i] for i in range(3)] == [_dumped(i) for i in range(3)]


def test_endpoint_with_no_compatible_artifact_raises_naming_the_slug(
    tmp_path: Path, no_git: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``select`` returning None is an error naming the endpoint, slug and version, and
    the publisher download is never tried.
    """
    fake = _FakeClient(None, b"")
    _use_fake(monkeypatch, fake)
    message = (
        f"http://tc-data.test has no supported artifact for 'toy_slug' compatible "
        f"with torchcell {__version__}; unset TC_DATA_URL to build from the "
        "publisher files instead"
    )
    with pytest.raises(RuntimeError) as excinfo:
        ToyDataset(root=str(tmp_path / "toy_slug"))
    assert str(excinfo.value) == message
    assert ToyDataset.calls == []
    assert fake.downloaded == []


def test_archive_without_a_build_manifest_is_refused(
    tmp_path: Path, no_git: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An archive that unpacks an LMDB but no ``preprocess/build_manifest.json`` raises
    ``FileNotFoundError`` naming the artifact, and ``process`` does not run to cover it.
    """
    source = _build(tmp_path / "source")
    members = {
        k: v
        for k, v in _packaged_members(source).items()
        if k != f"preprocess/{MANIFEST_FILENAME}"
    }
    fake = _FakeClient(_artifact(), _tar_xz(members))
    ToyDataset.calls.clear()
    _use_fake(monkeypatch, fake)
    with pytest.raises(FileNotFoundError) as excinfo:
        ToyDataset(root=str(tmp_path / "client" / "toy_slug"))
    assert str(excinfo.value) == (
        f"toy_slug/toy_slug-archive.tar.xz unpacked without preprocess/{MANIFEST_FILENAME}"
    )
    assert ToyDataset.calls == []


_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
_ZEROS = "0" * 64


def test_file_sha256_reads_in_chunks_and_follows_a_symlink(tmp_path: Path) -> None:
    """The digest is the same whatever the chunk size, and a symlink hashes its target."""
    path = tmp_path / "abc"
    path.write_bytes(b"abc")
    link = tmp_path / "link"
    link.symlink_to(path)
    assert file_sha256(path) == _ABC
    assert file_sha256(path, chunk_size=1) == _ABC
    assert file_sha256(link) == _ABC


def test_verify_sha256_and_verify_raw_files_name_the_file_and_both_digests(
    tmp_path: Path,
) -> None:
    """A matching pin returns None; the first mismatching pin in mapping order raises
    with the exact message and the three attributes.
    """
    (tmp_path / "a.txt").write_bytes(b"abc")
    (tmp_path / "b.txt").write_bytes(b"abc")
    verify_sha256(tmp_path / "a.txt", _ABC)
    verify_raw_files(str(tmp_path), {"a.txt": _ABC, "b.txt": _ABC})
    with pytest.raises(RawSha256MismatchError) as err:
        verify_raw_files(str(tmp_path), {"a.txt": _ABC, "b.txt": _ZEROS})
    assert str(err.value) == (
        f"sha256 mismatch for {tmp_path / 'b.txt'}: expected {_ZEROS}, observed {_ABC}"
    )
    assert (err.value.path, err.value.expected, err.value.observed) == (
        str(tmp_path / "b.txt"),
        _ZEROS,
        _ABC,
    )
    assert isinstance(err.value, RuntimeError)


def test_write_verified_writes_only_a_matching_payload(tmp_path: Path) -> None:
    """A payload off the pin raises naming the source and writes nothing (no
    ``.partial`` either); a matching one lands byte for byte.
    """
    dest = tmp_path / "out.txt"
    with pytest.raises(RawSha256MismatchError) as err:
        write_verified(b"abc", dest, _ZEROS, "https://example.org/out.txt")
    assert str(err.value) == (
        f"sha256 mismatch for https://example.org/out.txt: expected {_ZEROS}, "
        f"observed {_ABC}"
    )
    assert list(tmp_path.iterdir()) == []
    write_verified(b"abc", dest, _ABC, "https://example.org/out.txt")
    assert dest.read_bytes() == b"abc"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.txt"]


def test_copy_verified_hashes_the_source_before_writing(tmp_path: Path) -> None:
    """A source off the pin raises naming the SOURCE and leaves an existing destination
    untouched; a matching source replaces the destination with no ``.partial`` left.
    """
    src = tmp_path / "src.txt"
    src.write_bytes(b"abc")
    dest_dir = tmp_path / "raw"
    dest_dir.mkdir()
    dest = dest_dir / "dest.txt"
    dest.write_bytes(b"old")
    with pytest.raises(RawSha256MismatchError) as err:
        copy_verified(src, dest, _ZEROS)
    assert str(err.value) == (
        f"sha256 mismatch for {src}: expected {_ZEROS}, observed {_ABC}"
    )
    assert dest.read_bytes() == b"old"
    copy_verified(src, dest, _ABC)
    assert dest.read_bytes() == b"abc"
    assert [p.name for p in dest_dir.iterdir()] == ["dest.txt"]


def test_link_verified_links_only_a_matching_source_and_keeps_an_existing_link(
    tmp_path: Path,
) -> None:
    """A source off the pin raises and creates no link; a matching source is linked;
    a second call (even for another verified source) keeps the first link.
    """
    src = tmp_path / "src.txt"
    src.write_bytes(b"abc")
    other = tmp_path / "other.txt"
    other.write_bytes(b"abc")
    dest = tmp_path / "raw.txt"
    with pytest.raises(RawSha256MismatchError) as err:
        link_verified(src, dest, _ZEROS)
    assert str(err.value) == (
        f"sha256 mismatch for {src}: expected {_ZEROS}, observed {_ABC}"
    )
    assert not dest.is_symlink() and not dest.exists()
    link_verified(src, dest, _ABC)
    assert dest.is_symlink() and dest.readlink() == src
    link_verified(other, dest, _ABC)
    assert dest.readlink() == src


# ---- 2026.10.06 (Phase 21): abstract bodies, the interned cache, splice/harvest ---- #
def test_the_base_class_cannot_be_instantiated_and_its_abstract_bodies() -> None:
    """``ExperimentDataset`` itself refuses instantiation, naming all seven abstract
    members. Called through the base class on a concrete instance, ``download`` and
    ``process`` raise ``NotImplementedError`` (the base ``process`` carries no
    ``post_process`` decorator, so nothing else runs), while the three
    abstract properties and ``preprocess_raw`` / ``create_experiment`` have ``...``
    bodies and return ``None``.
    """
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Can't instantiate abstract class ExperimentDataset without an "
            "implementation for abstract methods 'create_experiment', 'download', "
            "'experiment_class', 'preprocess_raw', 'process', 'raw_file_names', "
            "'reference_class'"
        ),
    ):
        ExperimentDataset(root="unused")
    toy = ToyDataset.__new__(ToyDataset)
    with pytest.raises(NotImplementedError):
        ExperimentDataset.download(toy)
    with pytest.raises(NotImplementedError):
        ExperimentDataset.process(toy)
    base = ExperimentDataset.__dict__
    assert base["experiment_class"].fget(toy) is None
    assert base["reference_class"].fget(toy) is None
    assert base["raw_file_names"].fget(toy) is None
    assert ExperimentDataset.preprocess_raw(toy, pd.DataFrame()) is None
    assert ExperimentDataset.create_experiment(toy) is None


def test_a_second_dataset_on_one_root_shares_the_interned_tables(
    tmp_path: Path, no_git: None
) -> None:
    """``_load_interned`` caches each ``interned`` dir per process: a second dataset on
    the same root re-attaches the SAME raw table and validated-instance dict (identity,
    not equality) instead of reading the env again; ``get_single_item`` called with
    the env closed reopens it.
    """
    first = _build(tmp_path)
    assert first[0] == _dumped(0)
    first.close_lmdb()
    second = _build(tmp_path)
    assert second.env is None
    assert second.get_single_item(1) == _dumped(1)
    assert second._interned is first._interned
    assert second._validated_interned is first._validated_interned
    loaded = second._interned
    second._load_interned()
    assert second._interned is loaded


def test_a_store_rebuilt_in_place_in_one_process_reads_the_stale_interned_table(
    tmp_path: Path, no_git: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: ``_INTERNED_BY_DIR`` (experiment_dataset.py line 236, read at line
    505 and filled at line 525) is keyed by directory and never invalidated, so when a store is deleted
    and rebuilt at the same root in the same process with a different constant (here
    the environment at 30 C instead of 26 C), reading the new records looks up the
    new environment's content hash in the OLD table and the rebuild fails with a
    ``KeyError`` on that digest. Pinned until the cache is keyed by something that
    changes with the store (or cleared when ``process`` writes the interned env).
    """
    import shutil
    import sys

    first = _build(tmp_path)
    assert first[0] == _dumped(0)
    first.close_lmdb()
    shutil.rmtree(tmp_path / "toy_slug")
    warm = Environment(media=SGA_DM_SELECTION, temperature=Temperature(value=30))
    rebuilt = [(e.model_copy(update={"environment": warm}), r) for e, r in RECORDS]
    monkeypatch.setattr(sys.modules[__name__], "RECORDS", rebuilt)
    digest = compute_sha256_hash(canonical_json(warm))
    with pytest.raises(KeyError, match=re.escape(digest)):
        _build(tmp_path)


def test_copy_and_pickle_drop_the_per_directory_tables(
    tmp_path: Path, no_git: None
) -> None:
    """``__getstate__`` nulls ``_interned`` and ``_experiment_reference_index`` and
    empties ``_validated_interned`` in the copy (they are re-attached from the
    per-process cache or recomputed), and leaves the original untouched; the
    unpickled copy still reads record 2. ``__getstate__`` does NOT drop the open LMDB
    handle, so pickling a dataset whose env is open raises ``TypeError: cannot pickle
    'Environment' object``; the test pins that, then closes the env (as ``len`` and the
    DataLoader path do) before copying.
    """
    import copy

    dataset = _build(tmp_path)
    assert dataset[0] == _dumped(0)
    dataset.transform_item(dataset[0])
    assert dataset.env is not None
    with pytest.raises(TypeError, match=r"^cannot pickle 'Environment' object$"):
        pickle.dumps(dataset)
    dataset.close_lmdb()
    assert dataset._interned is not None and len(dataset._interned) == 3
    assert dataset._validated_interned != {}
    state = dataset.__getstate__()
    assert (
        state["_interned"],
        state["_validated_interned"],
        state["_experiment_reference_index"],
    ) == (None, {}, None)
    shallow = copy.copy(dataset)
    assert shallow._interned is None and shallow._validated_interned == {}
    assert dataset._interned is not None and len(dataset._interned) == 3
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._interned is None
    assert restored[2] == _dumped(2)


def test_splice_is_copy_on_write_through_dicts_and_lists() -> None:
    """With ref ``r1`` cached as ``sentinel`` and ``r2`` not cached: ``{"a": [r1, 5],
    "b": r2}`` returns a NEW dict whose list is a NEW list ``[sentinel, 5]``, ``b`` is
    the same uncached ``InternedDict``, and the pending flag is True (r2); the input is
    not mutated. A structure with nothing to replace comes back as the very same
    objects with pending False.
    """
    from torchcell.data.experiment_dataset import InternedDict

    toy = ToyDataset.__new__(ToyDataset)
    sentinel = object()
    toy._validated_interned = {"r1": sentinel}
    r1 = InternedDict({"x": 1}, "r1")
    r2 = InternedDict({"y": 2}, "r2")
    inner = [r1, 5]
    obj: dict[str, Any] = {"a": inner, "b": r2}
    spliced, pending = toy._splice_validated(obj)
    assert pending is True
    assert spliced is not obj and spliced["a"] is not inner
    assert spliced["a"][0] is sentinel and spliced["a"][1] == 5
    assert spliced["b"] is r2
    assert obj["a"] is inner and inner[0] is r1
    plain: dict[str, Any] = {"a": [1, {"b": 2}]}
    same, plain_pending = toy._splice_validated(plain)
    assert same is plain and plain_pending is False
    only_cached = [r1]
    replaced, list_pending = toy._splice_validated(only_cached)
    assert replaced == [sentinel] and replaced is not only_cached
    assert list_pending is False
    out, out_pending = toy._splice_validated([r2])
    assert out[0] is r2 and out_pending is True


def test_harvest_caches_list_members_once_and_build_returns_a_cached_constant() -> None:
    """``_harvest_validated`` walks dicts and lists in step with the built model and
    caches the instance behind each ``InternedDict`` the FIRST time only; ``_build`` on
    an interned constant that is already cached returns that instance without calling
    the model class.
    """
    from torchcell.data.experiment_dataset import InternedDict

    toy = ToyDataset.__new__(ToyDataset)
    toy._validated_interned = {}
    raw = {"items": [InternedDict({"x": 1}, "r3"), 7], "tag": "t"}
    first = {"items": ["model-r3", 7], "tag": "t"}
    toy._harvest_validated(first, raw)
    assert toy._validated_interned == {"r3": "model-r3"}
    toy._harvest_validated({"items": ["other", 7], "tag": "t"}, raw)
    assert toy._validated_interned == {"r3": "model-r3"}

    toy._validated_interned = {"ref-a": REF_A}

    def refuse(**_: Any) -> None:
        raise AssertionError("the model class must not be called")

    assert toy._build(refuse, InternedDict(REF_A.model_dump(), "ref-a")) is REF_A


def test_check_manifest_pin_names_the_path_and_both_digests() -> None:
    """Equal digests pass silently; a mismatch raises ``ManifestPinMismatchError``
    carrying the path and both digests as attributes and in the message.
    """
    from torchcell.data.experiment_dataset import (
        ManifestPinMismatchError,
        check_manifest_pin,
    )

    check_manifest_pin("data/a.csv", "aa", "aa")
    with pytest.raises(ManifestPinMismatchError) as excinfo:
        check_manifest_pin("data/a.csv", "aa", "bb")
    assert str(excinfo.value) == (
        "raw-mirror manifest records sha256 aa for data/a.csv, but the loader pins bb"
    )
    assert (excinfo.value.relpath, excinfo.value.recorded, excinfo.value.pin) == (
        "data/a.csv",
        "aa",
        "bb",
    )
