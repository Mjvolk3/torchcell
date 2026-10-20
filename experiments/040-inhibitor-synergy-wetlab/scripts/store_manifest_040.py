# experiments/040-inhibitor-synergy-wetlab/scripts/store_manifest_040.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.store_manifest_040]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/store_manifest_040
"""The 040 store's manifest: what it was built from, what it holds, and how it is verified.

``build_040_store.py`` writes ``processed/store_manifest.json`` (:class:`StoreManifest`)
beside the processed LMDB; :func:`verify_store` reads it back and checks the bytes of
``processed/lmdb/data.mdb`` against the recorded size and sha256 before a trainer reads a
record. This module imports nothing from torchcell so the trainer's data layer and the
launchers' preflight can load it without the package: the build imports torchcell at the
release's own commit, the trainer at the experiment branch, and the manifest is the one
object both agree on.

Why a sha256 and not a file list. Delta's Lustre target 106 stopped answering on
2026-10-16 and left about one file in a hundred unreadable with no error on stat; a job
that touched one hung to its 12 h limit. A manifest that names the bytes makes that a
preflight failure.
"""

from __future__ import annotations

import hashlib
import os.path as osp

from pydantic import BaseModel

MANIFEST_NAME = "store_manifest.json"


class DatasetCount(BaseModel):
    """Records returned by the query for one dataset against the count the release serves.

    ``served`` is the release's count; ``admitted`` is what the block's clause admits,
    counted directly in the served graph (``generate_040_cql.count_query``); ``returned``
    is what the raw stage wrote. The build asserts ``returned == admitted``, so
    ``dropped`` is the clause's exclusion and nothing else: a perturbed gene outside the
    gene set (``outside_gene_set``, served records per such gene; in a smoke build every
    gene but the smoke list), or a genotype of no perturbations (``empty_genotype``, a
    released wild-type screen; the private wells' block has no genotype clause and keeps
    its 977).
    """

    dataset: str
    served: int
    admitted: int
    returned: int
    #: ``served - returned``
    dropped: int
    #: served records per perturbed gene outside the gene set, most records first
    outside_gene_set: dict[str, int]
    #: served records whose genotype has no perturbation
    empty_genotype: int


class StoreManifest(BaseModel):
    """What the store was built from and what it holds; verified before every run."""

    experiment: str
    store_tag: str
    dataset_root: str
    built_at: str
    host: str
    slurm_job_id: str | None
    build_seconds: float
    #: the served graph
    neo4j_uri: str
    kg_release: str
    kg_version: str
    kg_torchcell_version: str
    #: the package the build imported
    torchcell_src: str
    torchcell_commit: str | None
    torchcell_version: str
    #: the query
    query_file: str
    query_sha256: str
    gene_set_size: int
    smoke_genes: list[str] | None
    #: what the store holds
    counts: list[DatasetCount]
    n_raw_records: int
    n_processed_entries: int
    data_mdb_bytes: int
    data_mdb_sha256: str

    def returned(self, dataset: str) -> int:
        """Records the query returned for ``dataset``."""
        return next(c.returned for c in self.counts if c.dataset == dataset)


def sha256_file(path: str) -> str:
    """sha256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(store: str) -> str:
    """Where the manifest sits inside a store root."""
    return osp.join(store, "processed", MANIFEST_NAME)


def read_manifest(store: str) -> StoreManifest:
    """The manifest of the store at ``store``; a store without one is not a 040 store."""
    path = manifest_path(store)
    assert osp.isfile(path), f"no store manifest at {path}"
    with open(path) as f:
        return StoreManifest.model_validate_json(f.read())


def verify_store(store: str) -> StoreManifest:
    """The manifest, after the processed LMDB's bytes match it (size, then sha256).

    A full-size file with a different sha256 is the Lustre case; a shorter file is a
    partial copy. Both stop the run here rather than inside a training step.
    """
    manifest = read_manifest(store)
    data_mdb = osp.join(store, "processed", "lmdb", "data.mdb")
    assert osp.isfile(data_mdb), f"no processed LMDB at {data_mdb}"
    size = osp.getsize(data_mdb)
    assert size == manifest.data_mdb_bytes, (
        f"{data_mdb} is {size:,} bytes; the manifest records {manifest.data_mdb_bytes:,}"
    )
    digest = sha256_file(data_mdb)
    assert digest == manifest.data_mdb_sha256, (
        f"{data_mdb} sha256 {digest} does not match the manifest's "
        f"{manifest.data_mdb_sha256}: the copy is damaged, re-copy the store"
    )
    return manifest
