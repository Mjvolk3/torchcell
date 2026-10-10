# tests/torchcell/candidates/_builders.py
# [[tests.torchcell.candidates._builders]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/candidates/_builders.py
"""Shared builders for the candidate-gate tests: evidence, gates, verdicts, fixture trees.

Everything here writes into a caller's ``tmp_path``; nothing reads the real mirrors, the
genomes tier or a dev store.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
from pathlib import Path
from typing import Any

import lmdb
from openpyxl import Workbook

from torchcell.candidates.verdict import (
    AggregationRecord,
    CandidateVerdict,
    G2Host,
    G2Record,
    G4KeyRecord,
    G5Record,
    GateResult,
    InventoryRecord,
)
from torchcell.literature.manifest import (
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.registry import GenomeManifest
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def sha(data: bytes) -> str:
    """sha256 hex of bytes."""
    return hashlib.sha256(data).hexdigest()


def evidence(
    quote: str = "a verbatim quote",
    *,
    key: str = "keyPaper2020",
    uri: str = "paper.md",
    digest: str = "a" * 64,
) -> SourcedValue:
    """A SourcedValue bound to ``<key>/<uri>`` with the given sha256."""
    return SourcedValue(
        value=1,
        provenance=Provenance(source_uri=uri, citation_key=key, sha256=digest),
        quote=quote,
    )


def aggregation(n: int = 3) -> AggregationRecord:
    """A re-measured aggregation of ``n`` studies, nothing else measured."""
    return AggregationRecord(
        n_source_studies=n,
        attribution_field="Publication",
        value_origin="re_measured",
        n_sources_mirrored=None,
        net_new_vs_served=None,
        evidence=(evidence(),),
    )


def gate(name: str, outcome: str = "pass", **extra: Any) -> GateResult:
    """One gate with a fixed reason."""
    return GateResult(gate=name, outcome=outcome, reason=f"{name} {outcome}", **extra)  # type: ignore[arg-type]


def unmeasured_tail(names: tuple[str, ...]) -> list[GateResult]:
    """``unmeasured`` gates for ``names``."""
    return [gate(n, "unmeasured") for n in names]


def verdict(**overrides: Any) -> CandidateVerdict:
    """A refused-at-G1 transcription verdict, overridable field by field."""
    fields: dict[str, Any] = {
        "citation_key": "keyPaper2020",
        "table": "bacteria",
        "row_name": "Key 2020",
        "gates": (gate("G1", "fail"), *unmeasured_tail(("G2", "G3", "G4", "G5"))),
        "zotero_item": "present",
        "source_kind": "transcription",
        "aggregation": None,
        "g2": None,
        "g3": None,
        "g4_key": None,
        "g4_value": None,
        "g5": None,
        "decided_at": "2026-10-10",
        "torchcell_commit": COMMIT,
    }
    fields.update(overrides)
    return CandidateVerdict(**fields)


def admissible_verdict(
    *, gap_issue: int | None = None, **overrides: Any
) -> CandidateVerdict:
    """Every gate measured and passing (G5 a filed gap when ``gap_issue`` is given)."""
    g5 = gate("G5", "gap", issue=gap_issue) if gap_issue is not None else gate("G5")
    fields: dict[str, Any] = {
        "gates": (gate("G1"), gate("G2"), gate("G3"), gate("G4"), g5),
        "source_kind": "primary",
        "g2": G2Record(
            branch="assembly_set",
            outcome="resolvable_readable",
            hosts=(
                G2Host(name="MG1655", assembly_set="s", resolvable=True, readable=True),
            ),
        ),
        "g3": InventoryRecord(
            citation_key="keyPaper2020", trees_read=("raw",), files=()
        ),
        "g4_key": G4KeyRecord(
            doi=None, pmid=None, accessions=(), hits=(), pmid_checked=False
        ),
        "g5": G5Record(
            phenotype_class="FitnessPhenotype", class_exists=gap_issue is None
        ),
    }
    fields.update(overrides)
    return verdict(**fields)


def xlsx_bytes(sheets: dict[str, list[list[Any]]]) -> bytes:
    """An xlsx workbook with the given sheets and rows."""
    workbook = Workbook()
    first = True
    for title, rows in sheets.items():
        ws = workbook.active if first else workbook.create_sheet()
        assert ws is not None
        ws.title = title
        for row in rows:
            ws.append(row)
        first = False
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def write_mirror(
    data_root: Path,
    tree: str,
    key: str,
    files: dict[str, bytes | None],
    *,
    doi: str | None = None,
    zotero_item_key: str | None = None,
    manual: tuple[str, ...] = (),
    pinned: dict[str, str] | None = None,
) -> Path:
    """A ``<tree>/<key>/`` deposit with a manifest; ``None`` bytes = listed, not on disk.

    ``manual`` paths carry a ``manual_browser`` retrieval; ``pinned`` overrides the
    manifest sha256 of a path (to fake drift).
    """
    root = data_root / tree / key
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for rel, data in files.items():
        payload = data if data is not None else b"absent"
        if data is not None:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(data)
        digest = (pinned or {}).get(rel, sha(payload))
        method = (
            RetrievalMethod.manual_browser
            if rel in manual
            else RetrievalMethod.direct_url
        )
        records.append(
            ArtifactRecord(
                path=rel,
                role="raw_data" if rel.startswith("data/") else "si_data",
                bytes=len(payload),
                sha256=digest,
                retrieval=RetrievalRecord(
                    method=method,
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"recipe": f"fetch {rel}"},
                    sha256=digest,
                    retrieved_at="2026-10-10",
                ),
            )
        )
    manifest = Manifest(
        citation_key=key, doi=doi, zotero_item_key=zotero_item_key, files=records
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def write_assembly_set(data_root: Path, assembly_set: str) -> None:
    """A one-file assembly set in ``torchcell-genomes`` with a valid manifest."""
    directory = data_root / "torchcell-genomes" / assembly_set
    directory.mkdir(parents=True)
    data = f">{assembly_set}\nACGT\n".encode()
    (directory / "genome.fna").write_bytes(data)
    manifest = GenomeManifest(
        assembly_set=assembly_set,
        organism="fixture",
        strain_or_population="fixture",
        source="fixture",
        release="1",
        files=[
            ArtifactRecord(
                path="genome.fna", role="sequence", bytes=len(data), sha256=sha(data)
            )
        ],
        provenance_complete=True,
        created_at="2026-10-10",
    )
    (directory / "manifest.json").write_text(manifest.model_dump_json())


def write_store(
    store_dir: Path,
    records: list[dict[str, Any]],
    *,
    dirty: bool = False,
    interned: dict[str, Any] | None = None,
) -> None:
    """A dev store: ``preprocess/build_manifest.json`` and a pickled records LMDB."""
    (store_dir / "preprocess").mkdir(parents=True)
    (store_dir / "preprocess" / "build_manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "fixture",
                "loader_class": "FixtureDataset",
                "loader_module": "torchcell.datasets.fixture",
                "surface_modules": [],
                "closure": {},
                "built_at": "2026-10-10T00:00:00+00:00",
                "hostname": "fixture",
                "torchcell_commit": COMMIT,
                "torchcell_dirty": dirty,
            }
        )
    )
    (store_dir / "processed").mkdir()
    env = lmdb.open(str(store_dir / "processed" / "lmdb"), map_size=1 << 24)
    with env.begin(write=True) as txn:
        for i, record in enumerate(records):
            txn.put(str(i).encode(), pickle.dumps(record))
    env.close()
    if interned is not None:
        ienv = lmdb.open(str(store_dir / "processed" / "interned"), map_size=1 << 24)
        with ienv.begin(write=True) as txn:
            for ref, obj in interned.items():
                txn.put(ref.encode(), pickle.dumps(obj))
        ienv.close()


def fitness_record(sample: str, locus: str, value: float) -> dict[str, Any]:
    """A record shaped like the Borchert 2024 store's (sample, locus, fitness)."""
    return {
        "experiment": {
            "phenotype": {"screen_id": sample, "environment_response": value},
            "genotype": {"perturbations": [{"systematic_gene_name": locus}]},
        }
    }
