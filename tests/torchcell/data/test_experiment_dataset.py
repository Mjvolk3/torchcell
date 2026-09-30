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
"""

import io
import json
import pickle
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
    _compute_reference_hash_parallel,
    canonical_json,
    post_process,
    process_reference_batch,
    serialize_for_hashing,
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
    with pytest.raises(ValueError, match="Cannot set an empty or None value"):
        dataset.gene_set = GeneSet()
    assert json.loads(path.read_text()) == ["YAL001C"]
    path.unlink()
    assert dataset.gene_set == GeneSet(["YAL001C", "YAL002W", "YBR001C"])
    assert not path.exists()


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


def test_transform_item_builds_the_reference_twice(
    tmp_path: Path, no_git: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: ``transform_item`` repeats ``reference = self.reference_class(...)``
    (experiment_dataset.py:639-640), so every item validates its reference twice and
    discards the first; the returned value is unaffected. The count is taken by wrapping
    the real class's ``__init__`` for this test only (a counting subclass would enter
    the schema's subclass discovery and break the ontology tree tests). Pinned until the
    duplicate line is removed, when the count becomes 1.
    """
    dataset = _build(tmp_path)
    original_init = FitnessExperimentReference.__init__
    constructions: list[int] = []

    def counting_init(self: FitnessExperimentReference, **data: Any) -> None:
        constructions.append(1)
        original_init(self, **data)

    monkeypatch.setattr(FitnessExperimentReference, "__init__", counting_init)
    item = dataset.transform_item(dataset[0])
    assert len(constructions) == 2
    assert item["reference"] == REF_A


def test_serialize_for_hashing_sorts_only_top_level_keys_of_a_model() -> None:
    """Finding: a reference MODEL is serialized as ``json.dumps(dict(sorted(dump)))``,
    which sorts the top-level keys only, while a plain dict is dumped with
    ``sort_keys=True`` at every depth, so the same reference hashes differently as a
    model and as its dump (nested ``environment_reference`` keys are in declaration
    order in the first). The reference index always hashes the stored dict, so the
    split is invisible there. Pinned until the model branch sorts recursively.
    """
    as_model = serialize_for_hashing(REF_A)
    as_dict = serialize_for_hashing(REF_A.model_dump())
    assert as_dict == json.dumps(REF_A.model_dump(), sort_keys=True)
    assert as_model == json.dumps(dict(sorted(REF_A.model_dump().items())))
    assert as_model != as_dict
    assert json.loads(as_model) == json.loads(as_dict)


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
