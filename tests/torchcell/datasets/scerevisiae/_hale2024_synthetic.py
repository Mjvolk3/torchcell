# tests/torchcell/datasets/scerevisiae/_hale2024_synthetic.py
# [[tests.torchcell.datasets.scerevisiae._hale2024_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/_hale2024_synthetic.py
"""A hand-derived synthetic Hale 2024 release, shared by the loader and verifier tests.

Everything ``process()`` and the verifier read is written under ``tmp_path`` as
``$DATA_ROOT``: the raw mirror (seven released files and a ``manifest.json``), the two
genomes-tier sets the parents pin (a two-chromosome-plus-mitochondrion S288C FASTA and a
four-row 3S variant table), and the literature-mirror paths are never touched (the
verifier's provenance audit is skipped on a synthetic release). ``patch(monkeypatch)``
repoints the module's release-shape constants at this release.

Segregant table (Supplemental Data 1): barcodes GBY -> x_BY_BYx, GA1 and GA2 -> segA
(a segregant with two barcodes), GB1 -> segB, GC1 -> segC (no lineage in any assay).

Genotype matrix (Supplemental Data 12), markers chr1 100/200/300, chr2 50/60, chr17 10:

    x_BY_BYx  0 0 0 | 0 0  | 0  -> chrI 100-300 p1 (3), chrII 50-60 p1 (2), chrmt 10 p1 (1)
    segA      0 1 1 | 1 NA | 1  -> chrI 100 p1 (1), chrI 200-300 p2 (2), chrII 50 p2 (1),
                                    chrmt 10 p2 (1)
    segB      1 NA 0 | 0 0 | 1  -> chrI 100 p2 (1), chrI 300 p1 (1), chrII 50-60 p1 (2),
                                    chrmt 10 p2 (1)
    segC      1 1 1 | 1 1  | 1

Supplemental Data 2 is the subset chr1 100/300 and chr17 10 of the first three columns.

Guides: gA (YAL001C, standard name AAA1), gB (YBL001C, no standard name), and the
control gRNAs c1, c2, c3 (ORF NA). Lineages (geno_bc, guide_bc, guide) and raw fitness
per assay, ATC1 / ATC2 / CON:

    GA1 b1 gA   0.40  0.41  0.50       GBY b5 gB   0.20  0.18  0.30
    GA2 b2 gA   0.38  0.39  0.52       GB1 b6 c1   0.10  0.12  0.10
    GA1 b3 gB   0.30  ---   0.36       GA1 b7 c2   0.30  0.32  0.20
    GB1 b4 gB  -0.10 -0.08  0.05       GBY b8 c3   0.05  0.02 -0.01

``b3`` is absent from ATC2. c3's CON mean is negative, so it is dropped: 2 of 3 control
gRNAs retained. Medians of the retained control means: ATC1 (0.10+0.30)/2 = 0.20, ATC2
(0.12+0.32)/2 = 0.22, CON (0.10+0.20)/2 = 0.15. Shifted CON values sum to
0.50+0.52+0.36+0.05+0.30+0.10+0.20-0.01 - 8 * 0.15 = 0.82, mean 0.1025; offsets
ATC1 0.3025, ATC2 0.3225, CON 0.2525, and every released ``fitness`` is ``raw -
offset`` computed in float as the R code computes it. Records: 5 gene-targeting rows in
ATC1, 4 in ATC2, 5 in CON = 14; control rows ledgered: 3 + 3 + 3 = 9.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from torchcell.datasets.scerevisiae import hale2024 as h
from torchcell.literature.manifest import ROLE_RAW_DATA, ArtifactRecord, Manifest
from torchcell.sequence.genome.registry import (
    HALE2024_3S_VARIANTS,
    SGD_S288C_R64,
    GenomeManifest,
)

SEGREGANTS = [
    ("x_BY_BYx_#1", "GBY", "x_BY_BYx", "TRUE"),
    ("segA_#1", "GA1", "segA", "TRUE"),
    ("segA_#2", "GA2", "segA", "TRUE"),
    ("segB_#1", "GB1", "segB", "FALSE"),
    ("segC_#1", "GC1", "segC", "TRUE"),
]
MARKERS = [(1, 100), (1, 200), (1, 300), (2, 50), (2, 60), (17, 10)]
CALLS: dict[str, list[float]] = {
    "x_BY_BYx": [0, 0, 0, 0, 0, 0],
    "segA": [0, 1, 1, 1, np.nan, 1],
    "segB": [1, np.nan, 0, 0, 0, 1],
    "segC": [1, 1, 1, 1, 1, 1],
}
GUIDES = {"gA": "YAL001C", "gB": "YBL001C", "c1": None, "c2": None, "c3": None}
GENE_NAMES = {"gA": "AAA1", "gB": None}
#: (geno_bc, guide_bc, guide, raw ATC1, raw ATC2 or None, raw CON)
LINEAGES: list[tuple[str, str, str, float, float | None, float]] = [
    ("GA1", "b1", "gA", 0.40, 0.41, 0.50),
    ("GA2", "b2", "gA", 0.38, 0.39, 0.52),
    ("GA1", "b3", "gB", 0.30, None, 0.36),
    ("GB1", "b4", "gB", -0.10, -0.08, 0.05),
    ("GBY", "b5", "gB", 0.20, 0.18, 0.30),
    ("GB1", "b6", "c1", 0.10, 0.12, 0.10),
    ("GA1", "b7", "c2", 0.30, 0.32, 0.20),
    ("GBY", "b8", "c3", 0.05, 0.02, -0.01),
]
GENOTYPE_OF = {bc: g for _, bc, g, _ in SEGREGANTS}
SEG_NAME_OF = {bc: n for n, bc, _, _ in SEGREGANTS}
#: The fasta: chrI 400 bp, chrII 100 bp, mito 20 bp. Variant rows (code, pos, ref, alt):
#: chrI:100 SNV, chrI:250 deletion, chrII:60 SNV, chrII:70 insertion.
FASTA = {"I": "ACGT" * 100, "II": "TTGCA" * 20, "mt": "A" * 20}
VARIANTS = [("1_S288C", 100, "T", "C"), ("1_S288C", 250, "C", "-"),
            ("2_S288C", 60, "A", "G"), ("2_S288C", 70, "-", "T")]  # fmt: skip


def offsets() -> dict[str, float]:
    """The per-assay offsets the R code computes for this release (hand-derived above)."""
    medians = {
        "ATC1": (0.10 + 0.30) / 2,
        "ATC2": (0.12 + 0.32) / 2,
        "CON": (0.10 + 0.20) / 2,
    }
    con = np.array([row[5] for row in LINEAGES]) - medians["CON"]
    mean = float(con.mean())
    return {k: v + mean for k, v in medians.items()}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lineage_csv(screen: str) -> bytes:
    column = {"ATC1": 3, "ATC2": 4, "CON": 5}[screen]
    offset = offsets()[screen]
    rows: list[dict[str, Any]] = []
    for row in LINEAGES:
        raw = row[column]
        if raw is None:
            continue
        geno_bc, guide_bc, guide = row[0], row[1], row[2]
        orf = GUIDES[guide]
        name = GENE_NAMES.get(guide)
        rows.append(
            {
                "geno_bc": geno_bc, "guide_bc": guide_bc, "T0": 10.0, "T1": 9.0,
                "T2": 8.0, "T3": 7.0, "T5": 6.0, "raw_fitness": raw,
                "error": 0.05 + 0.001 * len(rows), "log_likelihood": 20.0,
                "seg_name": SEG_NAME_OF[geno_bc], "genotype": GENOTYPE_OF[geno_bc],
                "guide_seq": guide, "gene_orf": "NA" if orf is None else orf,
                "gene_name": "NA" if name is None else name, "guide_category": "E",
                "fitness": float(raw) - offset,
            }
        )  # fmt: skip
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return buffer.getvalue().encode()


class SyntheticRelease(SimpleNamespace):
    """Paths and pins of one written synthetic release."""

    data_root: Path
    raw_dir: Path
    sha: dict[str, str]
    s288c: Path
    variants: Path


def write_release(root: Path) -> SyntheticRelease:
    """Write the raw mirror, its manifest and both genomes-tier sets under ``root``."""
    mirror = root / h.RAW_DIR_REL
    data = mirror / "data"
    data.mkdir(parents=True)
    pd.DataFrame(
        [(n, bc, bc[::-1], g, flag) for n, bc, g, flag in SEGREGANTS],
        columns=["seg_name", "geno_bc", "rev_geno_bc", "genotype", "genotype_data"],
    ).to_csv(data / h.SD1, index=False)
    matrix = pd.DataFrame(MARKERS, columns=["c", "p"])
    matrix.insert(2, "gp", matrix["p"] + (matrix["c"] - 1) * 1000)
    for name, calls in CALLS.items():
        matrix[name] = calls
    matrix.to_csv(data / h.SD12, sep="\t", index=False, na_rep="NA")
    subset = matrix.iloc[[0, 2, 5]][["c", "p", "gp", "x_BY_BYx", "segA", "segB"]]
    subset.to_csv(data / h.SD2, sep="\t", index=False, na_rep="NA")
    pd.DataFrame(
        [
            (g, "essential_collection", "NA" if o is None else o)
            for g, o in GUIDES.items()
        ],
        columns=["guide", "category", "ORF"],
    ).to_csv(data / h.SD3, index=False)
    for spec in h.ASSAYS:
        with zipfile.ZipFile(data / spec.released_file, "w") as archive:
            archive.writestr(spec.zip_member, _lineage_csv(spec.screen_id))
    sha = {name: _sha(data / name) for name in h.RELEASED_FILES}
    manifest = Manifest(
        citation_key=h.CITATION_KEY,
        doi=h.PAPER_DOI,
        title=h.TITLE,
        files=[
            ArtifactRecord(
                path=f"data/{name}",
                role=ROLE_RAW_DATA,
                bytes=(data / name).stat().st_size,
                sha256=sha[name],
            )
            for name in h.RELEASED_FILES
        ],
        provenance_complete=True,
        created_at="2026-10-11T00:00:00+00:00",
    )
    (mirror / "manifest.json").write_text(manifest.model_dump_json())

    genomes = root / "torchcell-genomes"
    s288c_dir = genomes / SGD_S288C_R64
    s288c_dir.mkdir(parents=True)
    s288c = s288c_dir / h.S288C_ASSEMBLY_FSA
    s288c.write_text(
        f">ref|NC_001133| [chromosome=I]\n{FASTA['I']}\n"
        f">ref|NC_001134| [chromosome=II]\n{FASTA['II']}\n"
        f">ref|NC_001224| [location=mitochondrion]\n{FASTA['mt']}\n"
    )
    variants_dir = genomes / HALE2024_3S_VARIANTS
    variants_dir.mkdir(parents=True)
    variants = variants_dir / h.SD13
    variants.write_text("".join(f"{c}\t{p}\t{r}\t{a}\n" for c, p, r, a in VARIANTS))
    for directory, path in ((s288c_dir, s288c), (variants_dir, variants)):
        record = ArtifactRecord(
            path=path.name,
            role="sequence",
            bytes=path.stat().st_size,
            sha256=_sha(path),
        )
        genome_manifest = GenomeManifest(
            assembly_set=directory.name,
            organism="Saccharomyces cerevisiae",
            strain_or_population="synthetic",
            source="test",
            release="test",
            files=[record],
            provenance_complete=True,
            created_at="2026-10-11T00:00:00+00:00",
        )
        (directory / "manifest.json").write_text(genome_manifest.model_dump_json())
    return SyntheticRelease(
        data_root=root, raw_dir=data, sha=sha, s288c=s288c, variants=variants
    )


def patch(monkeypatch: pytest.MonkeyPatch, release: SyntheticRelease) -> None:
    """Repoint the release-shape constants at the synthetic release."""
    monkeypatch.setenv("DATA_ROOT", str(release.data_root))
    monkeypatch.setattr(
        h,
        "RELEASED_FILES",
        {
            name: spec.model_copy(update={"sha256": release.sha[name]})
            for name, spec in h.RELEASED_FILES.items()
        },
    )
    released = {"ATC1": 8, "ATC2": 7, "CON": 8}
    gene = {"ATC1": 5, "ATC2": 4, "CON": 5}
    monkeypatch.setattr(
        h,
        "ASSAYS",
        [
            a.model_copy(
                update={
                    "released_rows": released[a.screen_id],
                    "gene_rows": gene[a.screen_id],
                }
            )
            for a in h.ASSAYS
        ],
    )
    monkeypatch.setattr(h, "EXPECTED_RECORDS", 14)
    monkeypatch.setattr(h, "N_BACKGROUNDS", 3)
    monkeypatch.setattr(h, "COMMIT_EVERY", 4)
    monkeypatch.setattr(
        h, "CONTROL_GUIDES", h.CONTROL_GUIDES.model_copy(update={"value": 3})
    )
    monkeypatch.setattr(
        h,
        "CONTROL_GUIDES_REMOVED",
        h.CONTROL_GUIDES_REMOVED.model_copy(update={"value": 1}),
    )


class _Resolution(SimpleNamespace):
    is_current_gene: bool
    systematic_name: str | None
    status: str


class StubGenome:
    """``gene_set``, ``resolve_gene_name`` and the standard-name index, nothing else."""

    gene_set = ["YAL001C", "YBL001C", "YCL001W"]
    feature_index: dict[str, Any] = {"standard_to_ids": {"AAA1": ["YAL001C"]}}

    def resolve_gene_name(self, name: str) -> _Resolution:
        """Systematic names resolve to themselves (current), AAA1 to YAL001C, else nothing."""
        target = {"AAA1": "YAL001C"}.get(name, name)
        current = target in self.gene_set
        return _Resolution(
            is_current_gene=current,
            systematic_name=target if current else None,
            status="current" if current else "unresolved",
        )


def read_json(path: Path) -> Any:
    """Read a JSON file the build wrote."""
    return json.loads(path.read_text())
