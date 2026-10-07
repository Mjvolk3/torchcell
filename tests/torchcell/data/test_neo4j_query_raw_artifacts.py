# tests/torchcell/data/test_neo4j_query_raw_artifacts.py
# [[tests.torchcell.data.test_neo4j_query_raw_artifacts]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_neo4j_query_raw_artifacts.py
"""The raw stage's resolvability gate and ``Neo4jQueryRaw.materialize``. No Neo4j.

No schema class on main carries an ``ArtifactRef`` yet (phase 3 adds them), so the
records use a test experiment family, ``"artifact fitness"``, whose phenotype carries an
optional ref and a list of refs, registered in the module's ``EXPERIMENT_TYPE_MAP`` for
the test. Their references are plain ``FitnessExperimentReference``s, because the stored
reference index validates references against the fixed schema union, which a test
subclass is not in; the walk of a REFERENCE is pinned on rendered rows instead. Five
records, ``PROCESS_BATCH`` 2 (batches [0, 1], [2, 3], [4]):

* 0: phenotype ref A.
* 1: a plain ``FitnessExperiment`` with no ref.
* 2: phenotype ref A2 (A's file, another member) and refs [B].
* 3: refs [B, A].
* 4: refs [C].

Distinct ``(tier, key, path, sha256)``: A, B, C, met in records 0, 2, 4, so the resolver
is called exactly [A, B, C] across the three batches.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lmdb
import pytest

from tests.torchcell.artifacts._fakes import (
    FakeRemote,
    literature_manifest,
    ref_for,
    write_tier,
)
from torchcell.artifacts import (
    ArtifactIntegrityError,
    ArtifactRef,
    ArtifactUnresolvableError,
    RemoteSource,
    distinct_refs,
    ref_key,
)
from torchcell.data import neo4j_query_raw as nqr
from torchcell.data.neo4j_query_raw import Neo4jQueryRaw, UnresolvableArtifactError
from torchcell.datamodels import schema as s

A = ref_for("objects", "set-a", "a.npy", b"alpha")
A2 = ref_for("objects", "set-a", "a.npy", b"alpha", member="row-7")
B = ref_for("objects", "set-a", "b.npy", b"beta")
C = ref_for("objects", "set-c", "c.npy", b"gamma")


class RefPhenotype(s.FitnessPhenotype):
    """A fitness phenotype that also points at off-graph bytes.

    ``Phenotype.validate_label_fields`` checks the label fields against the concrete
    class's own annotations, so the two are redeclared here.
    """

    fitness: float
    fitness_se: float | None = None
    ref: ArtifactRef | None = None
    refs: list[ArtifactRef] = []


class RefFitnessExperiment(s.FitnessExperiment):
    """The test family: a fitness experiment whose phenotype may carry refs."""

    experiment_type: str = "artifact fitness"
    phenotype: RefPhenotype


class RefFitnessReference(s.FitnessExperimentReference):
    """The test family's reference, whose phenotype may carry refs too."""

    experiment_reference_type: str = "artifact fitness"
    phenotype_reference: RefPhenotype


ENVIRONMENT = s.Environment(
    media=s.Media(name="YPD", state="solid", is_synthetic=False)
)
GENOME = s.ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")


def _genotype(gene: str) -> s.Genotype:
    return s.Genotype(
        perturbations=[
            s.KanMxDeletionPerturbation(
                systematic_gene_name=gene, perturbed_gene_name=gene
            )
        ]
    )


def _ref_record(
    gene: str, ref: ArtifactRef | None = None, refs: list[ArtifactRef] | None = None
) -> dict[str, Any]:
    return {
        "experiment": RefFitnessExperiment(
            dataset_name="toy_art",
            genotype=_genotype(gene),
            environment=ENVIRONMENT,
            phenotype=RefPhenotype(fitness=0.8, ref=ref, refs=refs or []),
        ),
        "experiment_reference": s.FitnessExperimentReference(
            dataset_name="toy_art",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=s.FitnessPhenotype(fitness=1.0),
        ),
    }


def _plain_record(gene: str) -> dict[str, Any]:
    return {
        "experiment": s.FitnessExperiment(
            dataset_name="toy_plain",
            genotype=_genotype(gene),
            environment=ENVIRONMENT,
            phenotype=s.FitnessPhenotype(fitness=0.5),
        ),
        "experiment_reference": s.FitnessExperimentReference(
            dataset_name="toy_plain",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=s.FitnessPhenotype(fitness=1.0),
        ),
    }


RECORDS = [
    _ref_record("YAL001C", ref=A),
    _plain_record("YAL002W"),
    _ref_record("YAL003W", ref=A2, refs=[B]),
    _ref_record("YAL004W", refs=[B, A]),
    _ref_record("YAL005W", refs=[C]),
]


def _serialize(record: dict[str, Any]) -> str:
    return json.dumps(record, default=lambda o: o.model_dump())


class _MemoryQueryRaw(Neo4jQueryRaw):
    """``Neo4jQueryRaw`` whose query yields ``RECORDS`` in property shape, no bolt."""

    def fetch_data(self) -> Iterator[Any]:
        for record in RECORDS:
            yield {
                "e_serialized": json.dumps(record["experiment"].model_dump()),
                "ref_serialized": json.dumps(
                    record["experiment_reference"].model_dump()
                ),
            }


PARTITIONED_QUERY = (
    "MATCH (e:Experiment) WHERE true{partition} ORDER BY e.id "
    "RETURN e.serialized_data AS e_serialized"
)


class _PartitionedQueryRaw(_MemoryQueryRaw):
    """Serves every record from the ``e.id`` prefix-``0`` partition, none elsewhere."""

    def fetch_query(self, query: str) -> Iterator[Any]:
        if "STARTS WITH '0'" in query:
            yield from self.fetch_data()


class _FakeResolver:
    """An ``ArtifactResolver`` that records each call and misses on chosen keys."""

    def __init__(self, missing: tuple[ArtifactRef, ...] = ()) -> None:
        self.calls: list[ArtifactRef] = []
        self.kwargs: list[dict[str, Any]] = []
        self.missing = {ref_key(r) for r in missing}

    def __call__(
        self,
        ref: ArtifactRef,
        *,
        data_root: str | Path | None,
        client: RemoteSource | None,
    ) -> object:
        self.calls.append(ref)
        self.kwargs.append({"data_root": data_root, "client": client})
        if ref_key(ref) in self.missing:
            raise ArtifactUnresolvableError(f"{ref} did not resolve; sources tried: x")
        return True


@pytest.fixture(autouse=True)
def _artifact_family(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the test family and render in batches of 2."""
    monkeypatch.setattr(
        nqr,
        "EXPERIMENT_TYPE_MAP",
        {**s.EXPERIMENT_TYPE_MAP, "artifact fitness": RefFitnessExperiment},
    )
    monkeypatch.setattr(
        nqr,
        "EXPERIMENT_REFERENCE_TYPE_MAP",
        {**s.EXPERIMENT_REFERENCE_TYPE_MAP, "artifact fitness": RefFitnessReference},
    )
    monkeypatch.setattr(nqr, "PROCESS_BATCH", 2)


def _build(root: Path, **kwargs: Any) -> _MemoryQueryRaw:
    raw = _MemoryQueryRaw(
        uri="bolt://none",
        username="",
        password="",
        root_dir=str(root),
        query="Q",
        **kwargs,
    )
    raw.close_lmdb()
    return raw


def _stored(raw: Neo4jQueryRaw) -> list[str]:
    env = lmdb.open(raw.lmdb_dir, readonly=True, lock=False)
    with env.begin() as txn:
        values = [txn.get(f"data_{i}".encode()).decode() for i in range(len(RECORDS))]
    env.close()
    return values


def _assert_no_store(root: Path) -> None:
    assert not (root / "raw" / "lmdb" / "data.mdb").exists()
    assert not (root / "raw" / "lmdb.partial").exists()


def test_gate_resolves_each_distinct_ref_once_across_batches(tmp_path: Path) -> None:
    """Resolver calls are exactly [A, B, C] (A2 and the repeats are not re-checked),
    with the view's data root and client; every record is stored byte-equal to its
    serialization and reads back equal, refs included.
    """
    resolver = _FakeResolver()
    client = FakeRemote({}, {})
    raw = _build(
        tmp_path,
        resolver=resolver,
        artifact_data_root="/tier/root",
        artifact_client=client,
    )
    assert resolver.calls == [A, B, C]
    assert resolver.kwargs == [{"data_root": "/tier/root", "client": client}] * 3
    assert raw._checked_artifacts == {ref_key(A), ref_key(B), ref_key(C)}
    assert _stored(raw) == [_serialize(r) for r in RECORDS]
    assert raw[3] == RECORDS[3]
    assert raw[4]["experiment"].phenotype.refs == [C]
    raw.close_lmdb()


def test_partitioned_build_gates_in_the_writing_process(tmp_path: Path) -> None:
    """``fetch_workers=2``: forked workers render, the parent checks. The parent's
    resolver sees [A, B, C] once each and the store equals the single-session one.
    """
    resolver = _FakeResolver()
    raw = _PartitionedQueryRaw(
        uri="bolt://none",
        username="",
        password="",
        root_dir=str(tmp_path),
        query=PARTITIONED_QUERY,
        fetch_workers=2,
        resolver=resolver,
    )
    raw.close_lmdb()
    assert resolver.calls == [A, B, C]
    assert _stored(raw) == [_serialize(r) for r in RECORDS]


def test_partitioned_build_with_a_dangling_ref_leaves_no_store(tmp_path: Path) -> None:
    """The same gate failure on the partitioned path: C misses, no store."""
    with pytest.raises(UnresolvableArtifactError, match="record data_4 carries"):
        _PartitionedQueryRaw(
            uri="bolt://none",
            username="",
            password="",
            root_dir=str(tmp_path),
            query=PARTITIONED_QUERY,
            fetch_workers=2,
            resolver=_FakeResolver(missing=(C,)),
        )
    _assert_no_store(tmp_path)


def test_rows_carry_the_distinct_refs_of_their_record(tmp_path: Path) -> None:
    """Rendered row refs: (A,), (), (A2, B), (B, A), (C,): distinct per record, the
    first per key kept, the plain record empty. A sixth record whose REFERENCE carries
    B (experiment ref A) renders (A, B): the reference is walked after the experiment.
    """
    raw = _build(tmp_path, artifact_check=False)
    with_reference_ref = {
        "experiment": RECORDS[0]["experiment"],
        "experiment_reference": RefFitnessReference(
            dataset_name="toy_art",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=RefPhenotype(fitness=1.0, ref=B),
        ),
    }
    batch = [
        (
            i,
            json.loads(json.dumps(r["experiment"].model_dump())),
            json.dumps(r["experiment_reference"].model_dump()),
        )
        for i, r in enumerate([*RECORDS, with_reference_ref])
    ]
    rows = raw._render_batch(batch, {}, "record")
    assert [row[4] for row in rows] == [(A,), (), (A2, B), (B, A), (C,), (A, B)]


def test_records_without_refs_are_never_walked_or_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only records whose JSON holds ``ARTIFACT_REF_MARKER`` are walked: 4 of 5 here
    (the plain record 1 is not); the resolver sees only ref-bearing records' refs.
    """
    walked: list[int] = []
    original = distinct_refs

    def counting(models: Any) -> Any:
        walked.append(1)
        return original(models)

    monkeypatch.setattr(nqr, "distinct_refs", counting)
    resolver = _FakeResolver()
    _build(tmp_path, resolver=resolver).close_lmdb()
    assert len(walked) == 4
    assert nqr.ARTIFACT_REF_MARKER not in _serialize(RECORDS[1])
    assert all(nqr.ARTIFACT_REF_MARKER in _serialize(RECORDS[i]) for i in (0, 2, 3, 4))


def test_an_unresolvable_ref_aborts_the_build_and_leaves_no_store(
    tmp_path: Path,
) -> None:
    """B misses: the error names record data_2 (where B is first met), B's tc:// string
    and sha256, and the resolver's message; nothing is left at raw/lmdb.
    """
    resolver = _FakeResolver(missing=(B,))
    with pytest.raises(UnresolvableArtifactError) as caught:
        _build(tmp_path, resolver=resolver)
    message = str(caught.value)
    assert message.startswith(f"record data_2 carries {B} (sha256 {B.sha256})")
    assert "did not resolve; sources tried: x" in message
    assert isinstance(caught.value.__cause__, ArtifactUnresolvableError)
    assert resolver.calls == [A, B]
    _assert_no_store(tmp_path)


def test_an_integrity_error_propagates_unchanged(tmp_path: Path) -> None:
    """A sha256 disagreement is not a miss: ``ArtifactIntegrityError`` itself, no store."""
    calls: list[ArtifactRef] = []
    error = ArtifactIntegrityError(f"{A}: manifest lists another sha256")

    def disagreeing(ref: ArtifactRef, **_: Any) -> object:
        calls.append(ref)
        raise error

    with pytest.raises(ArtifactIntegrityError) as caught:
        _build(tmp_path, resolver=disagreeing)
    assert calls == [A]
    assert caught.value is error
    assert error.__cause__ is None
    _assert_no_store(tmp_path)


def test_artifact_check_false_skips_the_gate(tmp_path: Path) -> None:
    """With the gate off a resolver that misses everything is never called."""
    resolver = _FakeResolver(missing=(A, B, C))
    raw = _build(tmp_path, resolver=resolver, artifact_check=False)
    assert resolver.calls == []
    assert _stored(raw) == [_serialize(r) for r in RECORDS]


def test_a_second_process_run_checks_again(tmp_path: Path) -> None:
    """The checked set is per ``process`` run: rebuilding into a fresh root on the
    same instance resolves A, B, C again.
    """
    resolver = _FakeResolver()
    raw = _build(tmp_path / "one", resolver=resolver)
    raw.root_dir = str(tmp_path / "two")
    raw.raw_dir = osp.join(raw.root_dir, "raw")
    raw.lmdb_dir = osp.join(raw.raw_dir, "lmdb")
    os.makedirs(raw.raw_dir)
    raw.process()
    assert resolver.calls == [A, B, C, A, B, C]


def test_the_default_resolver_checks_manifests_without_downloading(
    tmp_path: Path,
) -> None:
    """The real resolver: A and B listed in a local objects tier, C only in the fake
    remote's manifest. The build passes, nothing is downloaded, no cache is written.
    """
    data_root = tmp_path / "data_root"
    write_tier(data_root, "objects", "set-a", {"a.npy": b"alpha", "b.npy": b"beta"})
    remote_files = {"c.npy": b"gamma"}
    client = FakeRemote(
        {("objects", "set-c"): literature_manifest("set-c", remote_files)},
        {("objects", "set-c", "c.npy"): b"gamma"},
    )
    raw = _build(
        tmp_path / "build", artifact_data_root=str(data_root), artifact_client=client
    )
    assert client.manifest_calls == [("objects", "set-c")]
    assert client.downloads == []
    assert not (data_root / "artifact-cache").exists()
    assert len(raw) == len(RECORDS)


def test_the_default_resolver_fails_a_dangling_ref_with_its_sources(
    tmp_path: Path,
) -> None:
    """C held nowhere: the message carries the resolver's numbered source list."""
    data_root = tmp_path / "data_root"
    write_tier(data_root, "objects", "set-a", {"a.npy": b"alpha", "b.npy": b"beta"})
    with pytest.raises(UnresolvableArtifactError) as caught:
        _build(
            tmp_path / "build",
            artifact_data_root=str(data_root),
            artifact_client=FakeRemote({}, {}),
        )
    message = str(caught.value)
    assert message.startswith(f"record data_4 carries {C}")
    assert "(1) local objects manifest" in message
    assert "(2) fake-remote manifest: no key objects/set-c" in message
    _assert_no_store(tmp_path / "build")


def test_materialize_returns_the_resolvers_verified_path(tmp_path: Path) -> None:
    """``materialize(A)`` is the local tier file; ``materialize(C)`` downloads C from the
    remote into ``artifact-cache/<sha256>/c.npy`` with its bytes.
    """
    data_root = tmp_path / "data_root"
    write_tier(data_root, "objects", "set-a", {"a.npy": b"alpha", "b.npy": b"beta"})
    client = FakeRemote(
        {("objects", "set-c"): literature_manifest("set-c", {"c.npy": b"gamma"})},
        {("objects", "set-c", "c.npy"): b"gamma"},
    )
    raw = _build(
        tmp_path / "build", artifact_data_root=str(data_root), artifact_client=client
    )
    assert raw.materialize(A) == data_root / "torchcell-objects" / "set-a" / "a.npy"
    cached = raw.materialize(C)
    assert cached == data_root / "artifact-cache" / C.sha256 / "c.npy"
    assert cached.read_bytes() == b"gamma"
