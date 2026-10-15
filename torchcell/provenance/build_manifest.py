# torchcell/provenance/build_manifest.py
# [[torchcell.provenance.build_manifest]]
"""Per-dataset build manifests: the authoritative "which built LMDB is now stale" record.

When a dataset's ``process()`` finishes, :func:`write_build_manifest` drops a
``build_manifest.json`` into that dataset's ``preprocess/`` directory (beside
``verification_report.json``) recording the contract fingerprint of every schema symbol the
loader transitively depends on. A later :func:`check_all` recomputes those fingerprints from
the *local* schema and flags any dataset whose stored fingerprints no longer match.

Multi-machine design:
  - Fingerprints are content-addressed (see ``schema_deps``), so a manifest is meaningful on
    any machine; staleness is judged against each machine's LOCAL schema, no network needed.
  - Manifests live under ``DATA_ROOT`` (per-machine, next to the LMDB), NOT in git -- so each
    machine tracks its own built artifacts with zero cross-machine git contention.
  - ``torchcell_commit`` is recorded for provenance only and is NEVER used for the staleness
    decision (a schema commit that doesn't touch a loader's closure leaves it fresh). It is
    ``None`` when the built code is not a git checkout (e.g. an installed wheel).

Fingerprints alone cannot see every stale store (issue #833). A record pickled under a
class the local schema no longer DEFINES is in no closure, so its manifest compares equal
and the store reads fresh while nothing can deserialize it: 5 bacterial dev stores were in
that state on 2026-10-09 and ``--list-stale`` named 1. :func:`check_store` therefore ends
with a bounded read of the store's FIRST record (:func:`read_first_record`) and reports
``unreadable`` with the exception when it fails. It is the ONE freshness function: the
``build_manifest`` CLI (:func:`check_all`), ``build_dataset_lmdb --list-stale`` and the
live rebuild's preflight all call it, so the three cannot drift apart.

Since manifest schema version 2 (the ontology unification guards, 2026-10-15) a manifest
also records ``perturbation_leaves``: the sorted leaf class names of every genotype and
environment perturbation the store holds, measured by :func:`walk_perturbation_leaves`
over EVERY record at build time. The rebuild preflight checks those names against the
lanes the dataset's adapter conf enables (``torchcell.knowledge_graphs.perturbation_lanes``)
without opening the store, so a leaf no enabled lane emits (the Carruthers
``GeneAdditionPerturbation`` under a bacterial-only conf) is refused before generation
instead of becoming a dangling ``perturbation member of`` row the import drops. A
manifest without the field is from before the walk existed and reads ``stale``: the
sweep that follows this change rewrites every manifest with the field.

Since manifest schema version 3 (lane V, 2026-10-15) the same pass also records
``content_id_collisions``: how many of the store's records share an ``Experiment``
content id with another record of the same store. The adapter's ``Experiment`` node id
is the sha256 of the record's experiment dump and ``FastCsvSink`` deduplicates node ids,
so two records with byte-identical dumps become ONE node and the second measurement is
in the LMDB and absent from the graph, with no error anywhere. It had happened to four
served datasets and 65 records before anyone asked (lane Q's scan of 176 stores,
``experiments/database/scripts/content_id_collision_scan.py``), which is the argument
for the number living in the manifest rather than in a script somebody remembers to run.
The manifest RECORDS it and never refuses to be written -- the count is a measured
property of a store that already exists, and a build that ends with no manifest reads
``no_manifest`` and gets rebuilt into the same state -- and the rebuild preflight
(``torchcell.database.build_dataset_lmdb.mapped_store_collisions``) is what REFUSES a
store whose collisions the dataset has not declared with its released evidence.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import socket
import subprocess
import sys
from array import array
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.provenance.schema_deps import (
    SchemaSurface,
    load_default_surface,
    loader_closure,
)

__all__ = [
    "BuildManifest",
    "ContentIdCollisions",
    "PerturbationLeaves",
    "StoreWalk",
    "SymbolDrift",
    "StaleResult",
    "DatasetCheck",
    "StoreFreshness",
    "StoreState",
    "compute_manifest",
    "walk_store",
    "walk_perturbation_leaves",
    "leaf_class_by_perturbation_type",
    "write_build_manifest",
    "check_manifest",
    "check_store",
    "read_first_record",
    "check_all",
    "MANIFEST_FILENAME",
]

MANIFEST_FILENAME = "build_manifest.json"
MANIFEST_SCHEMA_VERSION = 3
"""Version 2 added ``perturbation_leaves`` and version 3 ``content_id_collisions``
(both 2026-10-15); a manifest missing either field reads stale."""
PERTURBATION_LEAVES_FIELD = "perturbation_leaves"
CONTENT_ID_COLLISIONS_FIELD = "content_id_collisions"

StoreState = Literal["fresh", "stale", "no_lmdb", "no_manifest", "unreadable"]
"""What one built store is, against the local schema. Only ``fresh`` needs no rebuild."""


class PerturbationLeaves(BaseModel):
    """The perturbation leaf classes a built store holds, by the list they sit in.

    Sorted class names (``GeneAdditionPerturbation``, ``PhagePerturbation``), one entry
    per leaf present in at least one record, measured over every record of the store by
    :func:`walk_perturbation_leaves`. ``genotype`` is drawn from ``genotype.perturbations``
    (absent on a ``SegregantGenotype``, which carries no perturbation list) and
    ``environment`` from ``environment.perturbations``.
    """

    model_config = ConfigDict(frozen=True)

    genotype: list[str] = Field(default_factory=list)
    environment: list[str] = Field(default_factory=list)


class ContentIdCollisions(BaseModel):
    """How many of a store's records share one ``Experiment`` content id (lane V).

    ``n_records_lost`` is the number that will be ABSENT from the graph: the sink keeps
    one node per id, so a group of ``k`` records becomes one node and ``k - 1``
    measurements are in the LMDB and nowhere else. ``example_groups`` holds the LMDB
    keys of up to five groups so the rows can be read without rerunning the scan.
    ``n_records_lost == 0`` is the only state a dataset gets for free; anything else is
    declared with its released evidence or refuses the rebuild
    (``torchcell.database.build_dataset_lmdb.EXPECTED_CONTENT_ID_COLLISIONS``).
    """

    model_config = ConfigDict(frozen=True)

    n_records: int
    n_distinct: int
    n_groups: int
    n_records_lost: int
    example_groups: list[list[str]] = Field(default_factory=list)


class StoreWalk(BaseModel):
    """What one streaming pass over a built store measures (:func:`walk_store`)."""

    model_config = ConfigDict(frozen=True)

    perturbation_leaves: PerturbationLeaves
    content_id_collisions: ContentIdCollisions


class BuildManifest(BaseModel):
    """Records the schema contract a built LMDB was produced against."""

    manifest_schema_version: int = MANIFEST_SCHEMA_VERSION
    dataset_name: str  # slug, e.g. "env_chemgen_hoepfner2014"
    loader_class: str  # e.g. "EnvChemgenHoepfner2014Dataset"
    loader_module: str  # e.g. "torchcell.datasets.scerevisiae.hoepfner2014"
    surface_modules: list[
        str
    ]  # basenames of the schema-surface files, e.g. ["schema.py", ...]
    closure: dict[
        str, str
    ]  # symbol -> contract fingerprint (THE staleness key), sorted
    built_at: str  # ISO-8601 UTC
    hostname: str
    torchcell_commit: str | None = None  # provenance only; NOT used for staleness
    torchcell_dirty: bool | None = None
    # The leaves the store holds (schema version 2). ``None`` only on a manifest written
    # before the field existed, which :func:`check_manifest` reports as stale.
    perturbation_leaves: PerturbationLeaves | None = None
    # The store's own Experiment content-id collisions (schema version 3). ``None`` only
    # on a manifest written before the count existed, which reads stale for the same
    # reason: the preflight reads the number off this field and cannot check a store
    # that never measured it.
    content_id_collisions: ContentIdCollisions | None = None


class SymbolDrift(BaseModel):
    """One schema symbol whose contract fingerprint no longer matches the manifest."""

    symbol: str
    stored_fingerprint: str
    current_fingerprint: str | None  # None == symbol removed from the schema entirely


class StaleResult(BaseModel):
    """Result of checking one manifest against the current schema surface."""

    dataset_name: str
    preprocess_dir: str
    is_stale: bool
    drift: list[SymbolDrift]
    # Manifest fields the current schema version requires and this manifest lacks
    # (``perturbation_leaves`` on a version 1 manifest). Each one stales the store.
    missing_fields: list[str] = Field(default_factory=list)


class StoreFreshness(BaseModel):
    """One built store's state: the answer :func:`check_store` gives about one directory."""

    model_config = ConfigDict(frozen=True)

    root: str
    state: StoreState
    drift: list[str] = Field(default_factory=list)
    reason: str | None = None  # set only for ``unreadable``: "<ExcClass>: <message>"

    @property
    def needs_rebuild(self) -> bool:
        """Everything but ``fresh`` is rebuilt before a full knowledge-graph build."""
        return self.state != "fresh"

    def describe(self) -> str:
        """One line naming what is wrong, for a preflight's refusal list."""
        if self.state == "fresh":
            return "fresh"
        if self.state == "stale":
            return f"stale on {self.drift}"
        if self.state == "no_lmdb":
            return f"no LMDB at {self.root}"
        if self.state == "no_manifest":
            return "no build manifest"
        return f"unreadable: {self.reason}"


class DatasetCheck(BaseModel):
    """Per-dataset outcome of a fleet-wide staleness scan."""

    dataset_name: str
    status: StoreState
    drift: list[SymbolDrift] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    reason: str | None = None


def _git_info(repo_dir: Path) -> tuple[str | None, bool | None]:
    """Best-effort (commit, dirty) for provenance; ``(None, None)`` if not a git checkout.

    Uses explicit return-code checks (never ``check=True``), so a non-checkout is modeled as
    absent metadata rather than a masked error -- consistent with the no-fallback rule since
    the commit is provenance-only and never gates staleness.
    """
    head = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0:
        return None, None
    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return head.stdout.strip(), dirty


def compute_manifest(
    *,
    dataset_name: str,
    loader_module: str,
    loader_class: str,
    loader_path: Path,
    surface: SchemaSurface,
    built_at: str,
    hostname: str,
    torchcell_commit: str | None,
    torchcell_dirty: bool | None,
    perturbation_leaves: PerturbationLeaves,
    content_id_collisions: ContentIdCollisions,
) -> BuildManifest:
    """Build a :class:`BuildManifest` from a loader's transitive schema closure."""
    closure = loader_closure(loader_path, surface)
    surface_modules = sorted(
        {osp.basename(module) for module in surface.module_of.values()}
    )
    return BuildManifest(
        dataset_name=dataset_name,
        loader_class=loader_class,
        loader_module=loader_module,
        surface_modules=surface_modules,
        closure={name: surface.fingerprints[name] for name in sorted(closure)},
        built_at=built_at,
        hostname=hostname,
        torchcell_commit=torchcell_commit,
        torchcell_dirty=torchcell_dirty,
        perturbation_leaves=perturbation_leaves,
        content_id_collisions=content_id_collisions,
    )


def leaf_class_by_perturbation_type() -> dict[str, str]:
    """``perturbation_type`` literal -> leaf class name, over both perturbation unions.

    The literal is unique per leaf (43 genotype and 5 environment members, measured
    2026-10-15), so the stored string names the class without validating the record. A
    stored literal outside this map is a record no union member explains, and the walk
    raises on it rather than guessing.
    """
    from typing import get_args

    from torchcell.datamodels.schema import (
        EnvironmentPerturbationType,
        GenePerturbationType,
    )

    by_type: dict[str, str] = {}
    for union in (GenePerturbationType, EnvironmentPerturbationType):
        for cls in get_args(union):
            literal = cls.model_fields["perturbation_type"].default
            if literal in by_type:
                raise ValueError(
                    f"perturbation_type {literal!r} is shared by {by_type[literal]} and "
                    f"{cls.__name__}; the manifest walk needs it unique"
                )
            by_type[literal] = cls.__name__
    return by_type


def _collisions_of(
    digests: bytearray, key_blob: bytearray, key_ends: array[int], max_examples: int = 5
) -> ContentIdCollisions:
    """Group the walked records by their 32-byte digest and count the collisions.

    ``digests`` is one contiguous buffer of 32 bytes per record and ``key_blob`` holds
    the LMDB keys end to end, with ``key_ends[i]`` the end offset of record ``i``'s key,
    so nothing is kept per record as a Python object. That is what keeps the pass
    affordable on the largest store: for Hoepfner's 30,308,574 records the buffers are
    about 970 MB of digests, 240 MB of key bytes and 240 MB of offsets, against roughly
    4.5 GB for a ``dict[bytes, int]`` of the same digests and 2 GB for a list of key
    strings.

    Exact, with no truncation: the digests are whole, and numpy's fixed-width ``S32``
    comparison strips trailing NUL bytes from both sides, which cannot make two
    different 32-byte values compare equal. The sort is stable, so a group's keys come
    out in store-iteration order.
    """
    import numpy as np

    n = len(key_ends)
    if n == 0:
        return ContentIdCollisions(
            n_records=0, n_distinct=0, n_groups=0, n_records_lost=0
        )
    array = np.frombuffer(memoryview(digests), dtype="S32")
    order = np.argsort(array, kind="stable")
    sorted_digests = array[order]
    same_as_previous = np.empty(n, dtype=bool)
    same_as_previous[0] = False
    if n > 1:
        same_as_previous[1:] = sorted_digests[1:] == sorted_digests[:-1]
    n_distinct = int(n - same_as_previous.sum())
    starts = np.flatnonzero(~same_as_previous)
    ends = np.append(starts[1:], n)
    groups = np.flatnonzero(ends - starts > 1)

    def key_at(index: int) -> str:
        start = key_ends[index - 1] if index else 0
        return key_blob[start : key_ends[index]].decode()

    examples = [
        [key_at(int(i)) for i in order[starts[g] : ends[g]]]
        for g in groups[:max_examples]
    ]
    return ContentIdCollisions(
        n_records=n,
        n_distinct=n_distinct,
        n_groups=int(groups.size),
        n_records_lost=int(n - n_distinct),
        example_groups=examples,
    )


def walk_store(root: str | Path) -> StoreWalk:
    """One streaming pass over ``<root>/processed/lmdb``: its leaves and its collisions.

    The leaves are the perturbation leaf classes present in EVERY record; the
    collisions are the records whose ``Experiment`` content id another record of the
    same store shares. Both come from one read, the same read path as the store's own
    (``pickle.loads`` + ``resolve_interned``; a loader may intern the genotype or the
    environment, and ``_maybe_intern`` always interns a large environment). The leaf
    walk reads the ``perturbation_type`` literal of each perturbation dict and names its
    class through :func:`leaf_class_by_perturbation_type`, so no record is validated and
    the pass costs what ``compute_gene_set_sequential`` costs. The collision digest is
    ``sha256(json.dumps(experiment))`` over the fully resolved experiment dict, which is
    the id ``CellAdapter._experiment_node`` computes from the revalidated model's
    ``model_dump()``: the stored dict IS that dump (a loader pickles
    ``experiment.model_dump()``), so the two agree without validating anything here.
    Raises on an empty store and on a literal no union member carries.
    """
    import hashlib
    import json
    import pickle

    import lmdb

    from torchcell.data.experiment_dataset import resolve_interned

    by_type = leaf_class_by_perturbation_type()
    processed = Path(root) / "processed"
    interned: dict[str, Any] = {}
    if (processed / "interned").is_dir():
        ienv = lmdb.open(str(processed / "interned"), readonly=True, lock=False)
        try:
            with ienv.begin() as itxn:
                for key, value in itxn.cursor():
                    interned[key.decode()] = pickle.loads(value)  # noqa: S301
        finally:
            ienv.close()
    genotype_types: set[str] = set()
    environment_types: set[str] = set()
    digests = bytearray()
    key_blob = bytearray()
    key_ends: array[int] = array("q")
    env = lmdb.open(str(processed / "lmdb"), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            for key, value in txn.cursor():
                experiment = resolve_interned(
                    pickle.loads(value)["experiment"],
                    interned,  # noqa: S301
                )
                key_blob += key
                key_ends.append(len(key_blob))
                digests += hashlib.sha256(
                    json.dumps(experiment).encode("utf-8")
                ).digest()
                genotype = experiment["genotype"]
                # A SegregantGenotype carries haplotype blocks and no perturbation list.
                if "perturbations" in genotype:
                    for perturbation in genotype["perturbations"]:
                        genotype_types.add(perturbation["perturbation_type"])
                for perturbation in experiment["environment"]["perturbations"]:
                    environment_types.add(perturbation["perturbation_type"])
    finally:
        env.close()
    if not key_ends:
        raise ValueError(f"{processed / 'lmdb'} holds no records")
    unknown = sorted((genotype_types | environment_types) - set(by_type))
    if unknown:
        raise KeyError(
            f"{processed / 'lmdb'} stores perturbation_type literals no union member "
            f"carries: {unknown}"
        )
    return StoreWalk(
        perturbation_leaves=PerturbationLeaves(
            genotype=sorted(by_type[t] for t in genotype_types),
            environment=sorted(by_type[t] for t in environment_types),
        ),
        content_id_collisions=_collisions_of(digests, key_blob, key_ends),
    )


def walk_perturbation_leaves(root: str | Path) -> PerturbationLeaves:
    """The leaf classes present in EVERY record of ``<root>/processed/lmdb``.

    The leaf half of :func:`walk_store`, kept as its own name because the rebuild
    preflight and the lane arithmetic are written in terms of leaves.
    """
    return walk_store(root).perturbation_leaves


def write_build_manifest(dataset: Any) -> Path:
    """Compute and write ``build_manifest.json`` for a freshly built dataset instance.

    Called from the ``post_process`` build hook (``torchcell/data/experiment_dataset.py``),
    so every dataset build records its schema contract automatically. ``dataset`` is typed
    ``Any`` to avoid importing ``ExperimentDataset`` here (which would create an import cycle,
    since that module imports this one).
    """
    cls = type(dataset)
    module = sys.modules[cls.__module__]
    module_file = module.__file__
    assert module_file is not None, f"loader module {cls.__module__} has no __file__"
    loader_path = Path(module_file)
    surface = load_default_surface()
    commit, dirty = _git_info(loader_path.parent)
    walk = walk_store(dataset.root)
    manifest = compute_manifest(
        dataset_name=osp.basename(osp.normpath(dataset.root)),
        loader_module=cls.__module__,
        loader_class=cls.__name__,
        loader_path=loader_path,
        surface=surface,
        built_at=datetime.now(UTC).isoformat(),
        hostname=socket.gethostname(),
        torchcell_commit=commit,
        torchcell_dirty=dirty,
        perturbation_leaves=walk.perturbation_leaves,
        content_id_collisions=walk.content_id_collisions,
    )
    out = Path(dataset.preprocess_dir) / MANIFEST_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return out


def check_manifest(
    manifest: BuildManifest, surface: SchemaSurface, preprocess_dir: str
) -> StaleResult:
    """Compare a manifest's stored fingerprints against the current local schema surface.

    A manifest without ``perturbation_leaves`` (written under manifest schema version
    1) is stale too: the preflight reads the store's leaves off that field, and a store
    with no measured leaves cannot be checked against its conf's lanes. The same holds
    for ``content_id_collisions`` (version 2 and earlier): the preflight reads the
    store's collision count off that field, and a store that never measured it could
    carry the silent record loss of lane Q's four datasets unseen.
    """
    drift = [
        SymbolDrift(
            symbol=symbol,
            stored_fingerprint=stored,
            current_fingerprint=surface.fingerprints.get(symbol),
        )
        for symbol, stored in manifest.closure.items()
        if surface.fingerprints.get(symbol) != stored
    ]
    missing_fields = [
        field
        for field, value in (
            (PERTURBATION_LEAVES_FIELD, manifest.perturbation_leaves),
            (CONTENT_ID_COLLISIONS_FIELD, manifest.content_id_collisions),
        )
        if value is None
    ]
    return StaleResult(
        dataset_name=manifest.dataset_name,
        preprocess_dir=preprocess_dir,
        is_stale=bool(drift) or bool(missing_fields),
        drift=drift,
        missing_fields=missing_fields,
    )


def read_first_record(root: str | Path) -> dict[str, Any]:
    """Deserialize the FIRST record of ``<root>/processed/lmdb`` and type it; return it.

    The bounded read behind the ``unreadable`` state. It is the store's own read path:
    the interned sibling env, ``pickle.loads``, ``resolve_interned`` (what
    ``ExperimentDataset.get_single_item`` does), then the three stored dicts validated
    into the classes their own ``experiment_type`` /
    ``experiment_reference_type`` name. Both ways a store goes unreadable raise here:
    a record (or an interned sub-object) pickled under a class the local schema no
    longer defines fails in ``pickle.loads``, and a record carrying a field the class
    no longer accepts fails pydantic with ``extra_forbidden``.

    ONE record, because unpickling is per-record and the classes a store uses are fixed
    at build time: reading record 0 exercises every class in its records' closure, and
    the interned table it loads first carries the reference and publication objects of
    the whole store. Raises for the caller to report; never returns a verdict.
    """
    import pickle

    import lmdb

    from torchcell.data.experiment_dataset import resolve_interned
    from torchcell.datamodels.schema import (
        EXPERIMENT_REFERENCE_TYPE_MAP,
        EXPERIMENT_TYPE_MAP,
        Publication,
    )

    processed = Path(root) / "processed"
    interned: dict[str, Any] = {}
    if (processed / "interned").is_dir():
        ienv = lmdb.open(str(processed / "interned"), readonly=True, lock=False)
        try:
            with ienv.begin() as itxn:
                for key, value in itxn.cursor():
                    interned[key.decode()] = pickle.loads(value)  # noqa: S301
        finally:
            ienv.close()
    env = lmdb.open(str(processed / "lmdb"), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            if not cursor.first():
                raise ValueError(f"{processed / 'lmdb'} holds no records")
            raw = cursor.value()
    finally:
        env.close()
    record: dict[str, Any] = resolve_interned(pickle.loads(raw), interned)  # noqa: S301
    experiment = record["experiment"]
    EXPERIMENT_TYPE_MAP[experiment["experiment_type"]](**experiment)
    reference = record["reference"]
    EXPERIMENT_REFERENCE_TYPE_MAP[reference["experiment_reference_type"]](**reference)
    Publication(**record["publication"])
    return record


def check_store(root: str | Path, surface: SchemaSurface) -> StoreFreshness:
    """The ONE freshness verdict on one built dataset directory.

    In order: the LMDB must exist, its build manifest must exist, every fingerprint in
    the manifest's closure must equal the current surface's, and the store's first
    record must deserialize. The read runs LAST and only on a store whose fingerprints
    all match, since a store already reported ``stale`` is rebuilt either way and the
    read costs an LMDB open.

    ``except Exception`` around the read is the probe's product, not a swallowed error:
    the state and the exception text are reported as the verdict, and nothing continues
    as though the store were readable.
    """
    lmdb_dir = Path(root) / "processed" / "lmdb"
    preprocess = Path(root) / "preprocess"
    if not lmdb_dir.is_dir():
        return StoreFreshness(root=str(root), state="no_lmdb")
    manifest_path = preprocess / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return StoreFreshness(root=str(root), state="no_manifest")
    manifest = BuildManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    result = check_manifest(manifest, surface, str(preprocess))
    if result.is_stale:
        return StoreFreshness(
            root=str(root),
            state="stale",
            drift=sorted({d.symbol for d in result.drift} | set(result.missing_fields)),
        )
    try:
        read_first_record(root)
    except Exception as error:  # the probe's verdict; reported, never ignored
        return StoreFreshness(
            root=str(root),
            state="unreadable",
            reason=f"{type(error).__name__}: {error}",
        )
    return StoreFreshness(root=str(root), state="fresh")


def _built_dataset_dirs(data_root: Path) -> list[Path]:
    """Directories under DATA_ROOT that hold a built LMDB (``<slug>/processed/lmdb``)."""
    return sorted(
        {
            lmdb.parent.parent
            for lmdb in data_root.glob("data/torchcell/*/processed/lmdb")
        }
    )


def check_all(data_root: Path, surface: SchemaSurface) -> list[DatasetCheck]:
    """Check every built dataset under DATA_ROOT for schema staleness.

    Per directory this is :func:`check_store`, so the scan sees an unreadable store the
    same way ``build_dataset_lmdb --list-stale`` and the live rebuild's preflight do.
    The reported name is the manifest's own ``dataset_name`` where there is a manifest
    to read it from, and the directory name otherwise.
    """
    results: list[DatasetCheck] = []
    for slug_dir in _built_dataset_dirs(data_root):
        freshness = check_store(slug_dir, surface)
        manifest_path = slug_dir / "preprocess" / MANIFEST_FILENAME
        name = slug_dir.name
        drift: list[SymbolDrift] = []
        missing_fields: list[str] = []
        if manifest_path.is_file():
            manifest = BuildManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            name = manifest.dataset_name
            result = check_manifest(manifest, surface, str(slug_dir / "preprocess"))
            drift = result.drift
            missing_fields = result.missing_fields
        results.append(
            DatasetCheck(
                dataset_name=name,
                status=freshness.state,
                drift=drift,
                missing_fields=missing_fields,
                reason=freshness.reason,
            )
        )
    return results


def _format_report(checks: list[DatasetCheck]) -> str:
    stale = [c for c in checks if c.status == "stale"]
    unreadable = [c for c in checks if c.status == "unreadable"]
    unmanifested = [c for c in checks if c.status in ("no_manifest", "no_lmdb")]
    fresh = [c for c in checks if c.status == "fresh"]
    lines: list[str] = [
        f"Built datasets: {len(checks)}  "
        f"(fresh {len(fresh)}, stale {len(stale)}, unreadable {len(unreadable)}, "
        f"unmanifested {len(unmanifested)})",
        "",
    ]
    for check in stale:
        what: list[str] = []
        if check.drift:
            what.append("changed: " + ", ".join(sorted(d.symbol for d in check.drift)))
        if check.missing_fields:
            what.append("manifest lacks: " + ", ".join(check.missing_fields))
        lines.append(f"  [STALE] {check.dataset_name}  -> rebuild; {'; '.join(what)}")
    for check in unreadable:
        lines.append(
            f"  [UNREADABLE] {check.dataset_name}  -> rebuild; unreadable: "
            f"{check.reason}"
        )
    for check in unmanifested:
        lines.append(
            f"  [no manifest] {check.dataset_name}  -> written on next rebuild"
        )
    if not stale and not unreadable and not unmanifested:
        lines.append("  All built datasets are fresh against the local schema.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.provenance.build_manifest",
        description="Check built datasets under DATA_ROOT for schema-contract staleness.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="root holding data/torchcell/<slug>/ (default: $DATA_ROOT from the environment)",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = Path(args.data_root or os.environ["DATA_ROOT"])
    checks = check_all(data_root, load_default_surface())
    print(_format_report(checks))
    return 1 if any(c.status in ("stale", "unreadable") for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
