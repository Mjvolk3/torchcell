# torchcell/datasets/pputida/royet2025
# [[torchcell.datasets.pputida.royet2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/royet2025
# Test file: tests/torchcell/datasets/pputida/test_royet2025.py
r"""Royet 2025: the P. putida KT2440 mariner Tn-seq screen in four metals.

Royet, Kergoat, Lutz, Oriol, Parisot, Schori, Ahrens, Rodrigue and Gueguen 2025
(Environmental Microbiology 27:e70095, doi:10.1111/1462-2920.70095, PMC12041740, citation
key ``royetHighThroughputTnSeqScreens2025``) grew one saturated, NON-barcoded mariner
transposon library of KT2440 for 12 generations in LB and in LB with a sub-inhibitory
dose of cobalt, copper, zinc or cadmium chloride, sequenced the insertion junctions of
each pool, and compared every gene's insertion reads in each metal against LB with
TRANSIT's RESAMPLING permutation test. Rank 53 of the bacterial candidate table.

RECORD = one (gene x metal) ``BacterialEnvironmentResponseExperiment`` from Table S5
(``EMI-27-e70095-s006.xlsx``, one sheet per metal): 5,729 genes x 4 metals = 22,916
released cells, every cell filled, 21,583 stored (DROPPED below).

GENOTYPE, AND THE ROW'S SCHEMA NEED. The candidate table recorded the need for "a
fitness record whose genotype is a gene rather than a strain, because a non-barcoded
insertion pool is never resolvable to a clone". The schema already carries exactly that:
``TransposonInsertionPerturbation`` with ``insertion_position``, ``insertion_strand`` and
``barcode`` left ``None`` is the gene-level aggregate over every insertion mutant of a
gene, the form Girgis 2009 (microarray footprinting, also non-barcoded) and Borchert
2024 (gene fitness averaged over strains) already store. So no schema change is made;
the three unreleased per-mutant fields are typed absences in
``preprocess/perturbation_field_gaps.json``.

PHENOTYPE. ``EnvironmentResponsePhenotype``, ``measurement_type=log2_ratio``,
``assay_type=other``: TRANSIT's ``log2FC`` of the gene's mean normalized insertion reads
in LB plus the metal over LB. It is signed (``FitnessPhenotype`` would clamp every
depletion to 0) and it is not a barcode count, so ``pooled_competitive_growth_barcode``
would be a false type; the readout is spelled out in ``units``, the Girgis 2009
precedent. ``n_samples = 2`` biological replicates per arm (Table S3 releases pools
``#1`` and ``#2`` for LB and for every metal; the build re-reads them). The release
carries a permutation p and a Benjamini-Hochberg q and no dispersion, so the uncertainty
and the standard error are typed gaps; the p and q are written to
``preprocess/not_stored.json`` with the other unstored columns.

REFERENCE. The unperturbed parent in the same metal: ``log2FC = 0``, a gene whose
insertion mutants are no more or less abundant after the metal than after LB. Measured
over the stored records, the per-metal median is -0.01 (Co, Cu) and -0.00 (Zn, Cd).

DROPPED: 1,333 cells, under ONE rule, ``no_insertion_reads_in_either_arm``. Where both
``Mean A`` (LB) and ``Mean B`` (LB + metal) are released as ``0.0``, no insertion mutant
of the gene was read in either arm, and TRANSIT releases ``log2FC 0.00`` with ``q 1``
for an empty gene. That zero is the absence of a measurement, not a measured null: 989
of the 1,333 are genes the LB HMM calls essential (ES), and 52 are the 13 genes with no
TA site at all, where no insertion can exist. Storing them would assert "no metal
effect" for genes the screen could not see. Per metal: Co 335, Cu 328, Zn 331, Cd 339.
Every other cell is stored, including the 717 measured exact zeros, and every one of the
thirty-four q <= 0.05 hits the Results count (9, 14, 3, 8) is among the stored records.

HOST. The library was built in the laboratory's own KT2440 isolate, which the authors
resequenced (CP036494) at 100 % average nucleotide identity to AE015451.2 and call
KT2440 throughout. Records are pinned to the genome tier's KT2440 assembly
(GCA_000007565.2, replicon AE015451.2); every one of the 5,729 released ``#Orf`` tags is
a current locus tag of it (measured, ``preprocess/locus_tag_reconciliation.json``).

ENVIRONMENT. LB at 28 C, shaken (aerobic), 12 generations, plus one metal chloride at
the dose the Methods print. The LB formulation is UNSTATED (no amounts are printed), so
the shared ``MEDIA_LIBRARY["LB"]`` object is used, the Cui 2018 precedent.

DUPLICATION AGAINST BORCHERT 2024 (measured,
``experiments/036-dataset-fixes-before-kg-build/results/
royetHighThroughputTnSeqScreens2025_release_inventory.json``): independent. The
compendium's 332 samples are all on MOPS, M9 or RCH2 minimal media, none on LB and none
naming a metal, so no (gene, condition) pair of this release is in it; its libraries are
the barcoded ``Putida_ML5`` and ``Putida_ML5_JBEI``, not this pool; and Royet 2025 is
none of its four source studies. All 4,732 compendium genes are among these 5,729, which
is the shared gene universe and not a shared measurement.

NOT STORED. Table S4 (the HMM essentiality calls, four states, on LB agar and after LB
outgrowth) is another phenotype family; it is not loaded here (see the dataset note).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import os.path as osp
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

import openpyxl
import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    write_verified,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import LB
from torchcell.datamodels.schema import (
    AssayType,
    AssemblyReferenceGenome,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    TransposonInsertionPerturbation,
)
from torchcell.datasets.bacteria_common import (
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.provenance import run_retriever
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "royetHighThroughputTnSeqScreens2025"
PAPER_DOI = "10.1111/1462-2920.70095"
PMCID = "PMC12041740"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"
PAPER_MD = "paper.md"
#: sha256 of the mirrored OCR every quote below is a literal substring of.
PAPER_MD_SHA256 = "0a614be0b7b2f834f7bee191df047136eb59a4b10415dc6d7c5578e51ca53f8c"
#: When the two consumed files were re-retrieved from the PMC bucket for the raw mirror;
#: the bytes equal the literature mirror's 2026-10-07 capture.
DATA_RETRIEVED_AT = "2026-10-10T06:24:35.374102+00:00"

TABLE_S5 = "EMI-27-e70095-s006.xlsx"
TABLE_S3 = "EMI-27-e70095-s005.xlsx"
#: The literature mirror's names for the same bytes (its ``si/`` directory).
LIBRARY_SI_NAME: Final[dict[str, str]] = {TABLE_S5: "si9.xlsx", TABLE_S3: "si7.xlsx"}


class RawFile(BaseModel):
    """One file the loader consumes: its pinned bytes and how they were retrieved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sha256: str
    bytes: int
    description: str
    retrieval: RetrievalRecord

    @property
    def mirror_relpath(self) -> str:
        """Path inside the raw mirror (``data/<name>``)."""
        return f"data/{self.name}"


def _pmc_file(name: str, sha256: str, size: int, description: str) -> RawFile:
    """A publisher SI file of the PMC Article Datasets bucket (``pmc_cloud``)."""
    key = f"{PMCID}.1/{name}"
    return RawFile(
        name=name,
        sha256=sha256,
        bytes=size,
        description=description,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=f"https://pmc-oa-opendata.s3.amazonaws.com/{key}",
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": key},
            sha256=sha256,
            retrieved_at=DATA_RETRIEVED_AT,
        ),
    )


RAW_FILES: tuple[RawFile, ...] = (
    _pmc_file(
        TABLE_S5,
        "7c32029a8039353c53992742e19182b6da07fd29f38538078670098037ec40eb",
        2568406,
        "Table S5: TRANSIT HMM (LB) and RESAMPLING (LB vs metal) output for every gene, "
        "one sheet per metal plus a log2FC/q summary sheet; the values this loader stores",
    ),
    _pmc_file(
        TABLE_S3,
        "283c919e627634f50647e52b637817dfb180cf67b4c007a15b582e8095857418",
        11249,
        "Table S3: the sequenced pools; read to prove the two replicate pools per arm",
    ),
)
DATA_SHA256: Final[dict[str, str]] = {raw.name: raw.sha256 for raw in RAW_FILES}
#: The released SI files this loader does not consume, and why.
NOT_MIRRORED: tuple[str, ...] = (
    "Table S4 (EMI-27-e70095-s002.xlsx): TRANSIT HMM four-state essentiality calls on LB "
    "agar and after LB outgrowth; a gene-essentiality phenotype family, not loaded here",
    "Tables S1 and S2 (s010.xlsx, s004.xlsx): strains, plasmids and oligonucleotides",
    "Supporting Information S1 (s003.docx) and Figures S1-S4 (s001, s007, s009, s008)",
)

_PAPER = Provenance(
    source_uri=PAPER_MD,
    citation_key=CITATION_KEY,
    sha256=PAPER_MD_SHA256,
    method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
)


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the pinned ``paper.md``."""
    return SourcedValue(value=value, provenance=_PAPER, quote=quote, note=note)


# --------------------------------------------------------------------------- #
# Verbatim quotes (literal substrings of paper.md at PAPER_MD_SHA256)
# --------------------------------------------------------------------------- #
_Q_HOST = (
    "The genome is registered under Genbank accession CP036494. The Average Nucleotide "
    "Identity between this strain and P. putida KT2440 (Genbank AE015451.2) is "
    "$1 0 0 \\%$ (http://enve-omics.ce.gatech. edu/ani/). The PP1 is thus referred to "
    "as the KT2440 strain in the article."
)
_Q_TEMPERATURE = (
    "The culture was then incubated at $2 8 ^ { \\circ } \\mathrm { C }$ with shaking "
    "at $1 8 0 \\mathrm { r p m }$ ."
)
_Q_DOSES = (
    "metals were added independently at the following subinhibitory concentrations: "
    "cobalt $1 0 \\mu \\mathrm { M }$ , copper $2 . 5 \\mathrm { m M }$ ,zinc $1 2 5 \\mu "
    "\\mathrm { M }$ , and cadmium $1 2 . 5 \\mu \\mathrm { M }$"
)
_Q_SALTS = (
    "During Tn-seq experiments, metals were used at a subinhibitory concentration: "
    "$\\mathrm { C o C l } _ { 2 }$ $1 0 \\mu \\mathrm { M }$ , $\\mathrm { Z n C l } _ { _ "
    "2 } 1 2 5 \\mu \\mathrm { M }$ , $\\operatorname { C u C l } _ { 2 } 2 . 5 "
    "\\operatorname { m M }$ $\\operatorname { C d C l } _ { 2 } 1 2 . 5 \\mu \\mathrm { "
    "M }$ ."
)
_Q_GENERATIONS = "This procedure was carried out for 12 generations."
_Q_LB = (
    "We used LB rich medium in our screens instead of minimal medium to prevent the loss "
    "of auxotrophic mutants or biosynthesis pathways that could be important for metal "
    "tolerance during growth."
)
_Q_REPLICATES = (
    "For the Tn-seq screening, biological replicates were performed to ensure the "
    "reproducibility of the method."
)
_Q_RESAMPLING = (
    "we conducted a RESAMPLING (permutation test) analysis using the TRANSIT software. "
    "We compared the results obtained from culture in LB to those obtained from culture "
    "in LB with an excess of metal (Table S5)."
)
_Q_TABLE_S5 = "Raw data of all datasets analysed by TRANSIT are presented in Table S5."
_Q_TRANSPOSON = (
    "The Mariner transposon can specifically insert itself into the genome at TA sites. "
    "In P. putida KT2440, 129,002 TA sites can be targeted by this transposon."
)
_Q_HITS = (
    "we identified 9 genes involved in cobalt tolerance, 14 in copper tolerance, 3 in "
    "zinc tolerance, and 8 in cadmium tolerance."
)
_Q_NORMALIZATION = (
    "Read counts per insertion were normalised using the LOESS method as described in "
    "Zomer et al. (Zomer et al. 2012)."
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "host": _paper(
        "KT2440",
        _Q_HOST,
        note="the library parent is the laboratory isolate PP1, resequenced as CP036494; "
        "the paper equates it with KT2440, so records pin the genome tier's KT2440 "
        "assembly (AE015451.2) with no background",
    ),
    "temperature_c": _paper(28.0, _Q_TEMPERATURE),
    "aerobicity": _paper(
        "aerobic", _Q_TEMPERATURE, note="a shaken flask culture at 180 rpm"
    ),
    "duration_generations": _paper(12.0, _Q_GENERATIONS),
    "medium": _paper(
        "LB",
        _Q_LB,
        note="no amounts are printed, so the formulation (Miller or Lennox) is "
        "unstated; the shared MEDIA_LIBRARY['LB'] object is used, as Cui 2018 does",
    ),
    "doses": _paper(
        {"cobalt": "10 uM", "copper": "2.5 mM", "zinc": "125 uM", "cadmium": "12.5 uM"},
        _Q_DOSES,
    ),
    "salts": _paper(
        {"cobalt": "CoCl2", "zinc": "ZnCl2", "copper": "CuCl2", "cadmium": "CdCl2"},
        _Q_SALTS,
        note="CuCl2 is resolved through the compound table's 'copper(II) chloride' row; "
        "the table carries no 'CuCl2' synonym",
    ),
    "n_samples": _paper(
        2,
        _Q_REPLICATES,
        note="the count is Table S3's: pools '#1' and '#2' for LB and for each metal, "
        "re-read at build time by read_replicate_pools",
    ),
    "statistic": _paper("TRANSIT RESAMPLING log2FC, LB + metal over LB", _Q_RESAMPLING),
    "release": _paper("Table S5", _Q_TABLE_S5),
    "transposon": _paper("mariner (Himar1)", _Q_TRANSPOSON),
    "hit_counts": _paper(
        {"Co": 9, "Cu": 14, "Zn": 3, "Cd": 8},
        _Q_HITS,
        note="the q <= 0.05 count of each metal's sheet, re-derived at build time",
    ),
    "normalization": _paper("LOESS", _Q_NORMALIZATION),
}

PUBLICATION = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")


# --------------------------------------------------------------------------- #
# The four metals of Table S5
# --------------------------------------------------------------------------- #
class MetalSpec(BaseModel):
    """One metal arm: its Table S5 sheet, the compound label and its dose."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    sheet: str
    group_header: str
    compound_label: str
    dose: float
    unit: ConcentrationUnit
    paper_hits: int


METALS: tuple[MetalSpec, ...] = (
    MetalSpec(
        code="Co",
        sheet="LB-Co",
        group_header="RESAMPLING LB vs Cobalt",
        compound_label="CoCl2",
        dose=10.0,
        unit=ConcentrationUnit.micromolar,
        paper_hits=9,
    ),
    MetalSpec(
        code="Cu",
        sheet="LB-Cu",
        group_header="RESAMPLING LB vs Copper",
        compound_label="copper(II) chloride",
        dose=2.5,
        unit=ConcentrationUnit.millimolar,
        paper_hits=14,
    ),
    MetalSpec(
        code="Zn",
        sheet="LB-Zn",
        group_header="RESAMPLING LB vs Zinc",
        compound_label="ZnCl2",
        dose=125.0,
        unit=ConcentrationUnit.micromolar,
        paper_hits=3,
    ),
    MetalSpec(
        code="Cd",
        sheet="LB-Cd",
        group_header="RESAMPLING LB vs Cadmium",
        compound_label="CdCl2",
        dose=12.5,
        unit=ConcentrationUnit.micromolar,
        paper_hits=8,
    ),
)

#: Table S5's per-metal header, verbatim (row 3 of each sheet).
S5_HEADER: tuple[str, ...] = (
    "#Orf",
    "Name",
    "Description",
    "Sites",
    "Number of sites labeled E",
    "Number of sites labeled GD ",
    "Number of sites labeled NE",
    "Number of sites labeled GA",
    "Mean insertion rate within the gene",
    "Mean read count within the gene",
    "State",
    "Mean A",
    "Mean B",
    "Delta sum",
    "log2FC",
    "p-value",
    "q-value",
)
S5_HEADER_ROW = 3
S5_FIRST_DATA_ROW = 5
COL_ORF, COL_SITES, COL_STATE = 0, 3, 10
COL_MEAN_A, COL_MEAN_B, COL_DELTA, COL_LOG2FC, COL_P, COL_Q = 11, 12, 13, 14, 15, 16
Q_THRESHOLD = 0.05

#: Build oracles, every one measured on the pinned bytes (the release inventory).
N_GENES = 5729
N_SOURCE_CELLS = N_GENES * len(METALS)
EXPECTED_DROPPED: Final[dict[str, int]] = {"Co": 335, "Cu": 328, "Zn": 331, "Cd": 339}
EXPECTED_RECORDS = N_SOURCE_CELLS - sum(EXPECTED_DROPPED.values())
N_REPLICATES = 2
#: The arms Table S3 must release two pools for.
REPLICATE_ARMS: tuple[str, ...] = ("LB", *(m.code for m in METALS))
MIN_RESOLVED_FRACTION = 1.0

DROP_EMPTY_GENE = "no_insertion_reads_in_either_arm"
DROP_RULE_DESCRIPTIONS: dict[str, str] = {
    DROP_EMPTY_GENE: "Mean A (LB) and Mean B (LB + metal) are both released as 0.0: no "
    "insertion mutant of the gene was read in either arm, and TRANSIT releases log2FC "
    "0.00 with q 1 for such a gene. The zero is the absence of a measurement, not a "
    "measured null, so it is not stored"
}


class S5Row(BaseModel):
    """One gene's row of one Table S5 metal sheet, typed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    orf: str
    sites: int
    state: str
    mean_a: float
    mean_b: float
    delta_sum: float
    log2fc: float
    p_value: float
    q_value: float

    @property
    def is_empty(self) -> bool:
        """No insertion read in either arm (both released means are 0.0)."""
        return self.mean_a == 0.0 and self.mean_b == 0.0


class SheetFormatError(ValueError):
    """A released sheet whose header, ids or a cell is not what the loader reads."""


# --------------------------------------------------------------------------- #
# Mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/royetHighThroughputTnSeqScreens2025``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/royetHighThroughputTnSeqScreens2025``."""
    return Path(data_root or _data_root()) / LIBRARY_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve_raw_files(dest_dir: str | Path) -> dict[str, Path]:
    """Run each file's recorded retriever and write the verified bytes to ``dest_dir``."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for raw in RAW_FILES:
        path = dest / raw.name
        write_verified(
            run_retriever(raw.retrieval),
            path,
            raw.sha256,
            raw.retrieval.source_url or raw.name,
        )
        out[raw.name] = path
    return out


def deposit_raw_mirror(
    *, sources: Mapping[str, str | Path], data_root: str | None = None
) -> Path:
    """Write the raw mirror from already-retrieved files plus its ``manifest.json``.

    Idempotent by sha256: a mirror file with the pinned hash is left alone, and one with
    any other hash raises rather than being overwritten.
    """
    missing = sorted(set(DATA_SHA256) - set(sources))
    if missing:
        raise KeyError(f"no source given for {missing}")
    root = raw_mirror_dir(data_root)
    records: list[ArtifactRecord] = []
    for raw in RAW_FILES:
        src = Path(sources[raw.name])
        got = _sha256(src)
        if got != raw.sha256:
            raise RuntimeError(
                f"{src} sha256 mismatch: got {got}, expected {raw.sha256}"
            )
        dest = root / raw.mirror_relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != raw.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        records.append(
            ArtifactRecord(
                path=raw.mirror_relpath,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=raw.sha256,
                source=raw.retrieval.source_url,
                retrieval=raw.retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title="High-Throughput Tn-Seq Screens Identify Both Known and Novel "
        "Pseudomonas putida KT2440 Genes Involved in Metal Tolerance",
        files=records,
        si_data_sources=[r.retrieval.source_url or r.name for r in RAW_FILES],
        si_expected=list(NOT_MIRRORED),
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Reading the released sheets
# --------------------------------------------------------------------------- #
def _sheet_rows(path: str | Path, sheet: str) -> list[tuple[Any, ...]]:
    """Every row of one sheet, values only."""
    book = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    try:
        return list(book[sheet].iter_rows(values_only=True))
    finally:
        book.close()


def read_metal_sheet(path: str | Path, spec: MetalSpec) -> list[S5Row]:
    """One metal's Table S5 sheet, typed, with its header and group header asserted.

    Every number is released as a TEXT cell (``'-0.32'``), so each is parsed with
    ``float`` and a cell that does not parse stops the build. ``-0.00`` becomes 0.0.
    """
    rows = _sheet_rows(path, spec.sheet)
    header = tuple(rows[S5_HEADER_ROW - 1])
    if header != S5_HEADER:
        raise SheetFormatError(f"{spec.sheet} is headed {header}, expected {S5_HEADER}")
    if spec.group_header not in rows[S5_HEADER_ROW - 2]:
        raise SheetFormatError(
            f"{spec.sheet} does not carry the group header {spec.group_header!r}"
        )
    out: list[S5Row] = []
    for cells in rows[S5_FIRST_DATA_ROW - 1 :]:
        if cells[COL_ORF] is None:
            raise SheetFormatError(f"{spec.sheet} has a row with no #Orf")
        out.append(
            S5Row(
                orf=str(cells[COL_ORF]),
                sites=int(cells[COL_SITES]),
                state=str(cells[COL_STATE]),
                mean_a=float(cells[COL_MEAN_A]),
                mean_b=float(cells[COL_MEAN_B]),
                delta_sum=float(cells[COL_DELTA]),
                log2fc=float(cells[COL_LOG2FC]) + 0.0,
                p_value=float(cells[COL_P]),
                q_value=float(cells[COL_Q]),
            )
        )
    if len(out) != N_GENES or len({r.orf for r in out}) != N_GENES:
        raise SheetFormatError(
            f"{spec.sheet} carries {len(out)} rows / {len({r.orf for r in out})} unique "
            f"#Orf, expected {N_GENES}"
        )
    hits = sum(1 for r in out if r.q_value <= Q_THRESHOLD)
    if hits != spec.paper_hits:
        raise SheetFormatError(
            f"{spec.sheet} has {hits} genes at q <= {Q_THRESHOLD}, the Results state "
            f"{spec.paper_hits}"
        )
    return out


def read_release(path: str | Path) -> dict[str, list[S5Row]]:
    """All four metal sheets, checked to name the same genes in the same order."""
    sheets = {spec.code: read_metal_sheet(path, spec) for spec in METALS}
    orders = {code: [r.orf for r in rows] for code, rows in sheets.items()}
    first = orders[METALS[0].code]
    for code, order in orders.items():
        if order != first:
            raise SheetFormatError(f"sheet {code} lists the genes in another order")
    return sheets


def read_replicate_pools(path: str | Path) -> dict[str, int]:
    """Table S3's pools per arm (``'Co #1'`` -> arm ``Co``), every arm required at 2."""
    rows = _sheet_rows(path, "Feuil1")
    counts: Counter[str] = Counter()
    for cells in rows:
        label = cells[0]
        if not isinstance(label, str):
            continue
        arm, sep, rep = label.partition(" #")
        if sep and rep.isdigit():
            counts[arm] += 1
    pools = {arm: counts[arm] for arm in REPLICATE_ARMS}
    if any(n != N_REPLICATES for n in pools.values()):
        raise SheetFormatError(
            f"Table S3 releases {pools} pools per arm, expected {N_REPLICATES} each"
        )
    return pools


# --------------------------------------------------------------------------- #
# Environment, genotype, phenotype (pure, no files)
# --------------------------------------------------------------------------- #
def metal(spec: MetalSpec) -> SmallMoleculePerturbation:
    """One metal chloride at its printed dose."""
    return SmallMoleculePerturbation(
        compound=resolved_compound(spec.compound_label),
        concentration=Concentration(value=spec.dose, unit=spec.unit),
    )


def environment(spec: MetalSpec) -> Environment:
    """The selection culture of one metal: LB at 28 C for 12 generations plus the metal."""
    return Environment(
        media=LB,
        temperature=Temperature(value=float(SOURCED_VALUES["temperature_c"].value)),
        perturbations=[metal(spec)],
        aerobicity=str(SOURCED_VALUES["aerobicity"].value),
        duration_generations=float(SOURCED_VALUES["duration_generations"].value),
    )


TRANSPOSON: Final[str] = str(SOURCED_VALUES["transposon"].value)
GENE_NAMESPACE: Final = "pputida_kt2440_locus_tag"


def canonical_symbol(genome: PPutidaKT2440Genome, tag: str) -> str:
    """The annotation's gene symbol of ``tag`` when it resolves back to ``tag``, else
    the tag, so the stored pair always round-trips through the resolver.
    """
    symbol = genome.genbank.loci[tag].symbol
    if symbol is None:
        return tag
    resolution = genome.resolve_gene_name(symbol)
    return symbol if resolution.systematic_name == tag else tag


def insertion_genotype(locus_tag: str, symbol: str) -> Genotype:
    """The gene-level mariner disruption of one KT2440 locus (no clone, no site)."""
    return Genotype(
        perturbations=[
            TransposonInsertionPerturbation(
                systematic_gene_name=locus_tag,
                perturbed_gene_name=symbol,
                gene_namespace=GENE_NAMESPACE,
                transposon=TRANSPOSON,
            )
        ]
    )


UNITS = (
    "TRANSIT RESAMPLING log2FC of the gene's mean LOESS-normalized insertion reads per "
    "TA site in LB plus the metal over LB, non-barcoded mariner Tn-seq, two biological "
    "replicate pools per arm, 12 generations; negative = insertion mutants of the gene "
    "were depleted by the metal"
)
UNITS_REFERENCE = (
    "the unperturbed parent in the same metal: log2FC 0, insertion mutants no more or "
    "less abundant after the metal than after LB"
)

_PAPER_LOOKED_IN = Provenance(
    source_uri=PAPER_MD,
    citation_key=CITATION_KEY,
    sha256=PAPER_MD_SHA256,
    method="full Experimental Procedures and Results read, plus every SI workbook",
)
PHENOTYPE_GAPS: tuple[ProvenanceGap, ...] = (
    ProvenanceGap(
        field="environment_response_uncertainty",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_PAPER_LOOKED_IN,
        note="Table S5 releases a permutation p-value and a Benjamini-Hochberg q per "
        "gene, which test the log2FC, and no dispersion of it",
    ),
    ProvenanceGap(
        field="environment_response_se",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_PAPER_LOOKED_IN,
        note="no dispersion is released, so no standard error can be derived",
    ),
)
_PERTURBATION_GAP_NOTES: dict[str, str] = {
    "barcode": "non-barcoded mariner Tn-seq: a mutant is read by its insertion junction, "
    "so the library carries no molecular barcode",
    "insertion_position": "a record is the gene-level aggregate over every insertion in "
    "the gene's TA sites; per-site counts are not released",
    "insertion_strand": "no per-insertion orientation is released",
}
#: Typed absences of the transposon leaf's own fields (the leaf has no gap slot).
PERTURBATION_FIELD_GAPS: tuple[ProvenanceGap, ...] = tuple(
    ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_PAPER_LOOKED_IN,
        note=note,
    )
    for field, note in _PERTURBATION_GAP_NOTES.items()
)


def phenotype(value: float) -> EnvironmentResponsePhenotype:
    """One released log2FC of one gene in one metal."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.other,
        environment_response=value,
        n_samples=N_REPLICATES,
        sample_unit=SampleUnit.biological_replicate,
        units=UNITS,
        provenance_gaps=list(PHENOTYPE_GAPS),
    )


def reference_phenotype() -> EnvironmentResponsePhenotype:
    """The parent in the same metal: log2FC 0."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.other,
        environment_response=0.0,
        n_samples=N_REPLICATES,
        sample_unit=SampleUnit.biological_replicate,
        units=UNITS_REFERENCE,
    )


def reference_genome(data_root: str | None = None) -> AssemblyReferenceGenome:
    """KT2440 pinned to its GenBank assembly."""
    return assembly_reference("KT2440", data_root=data_root)


def build_reference(
    dataset_name: str, genome: AssemblyReferenceGenome, env: Environment
) -> BacterialEnvironmentResponseExperimentReference:
    """The unperturbed parent of one metal condition, scoring 0."""
    return BacterialEnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome,
        environment_reference=env.model_copy(),
        phenotype_reference=reference_phenotype(),
    )


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #
class MetalLedger(BaseModel):
    """What one metal sheet released and what of it was stored."""

    model_config = ConfigDict(extra="forbid")

    metal: str
    sheet: str
    source_cells: int
    kept_records: int
    dropped_empty: int
    dropped_empty_states: dict[str, int]
    dropped_empty_zero_sites: int
    kept_exact_zero: int
    kept_q_le_threshold: int


class DropLog(BaseModel):
    """The release-to-records accounting, per metal and in total."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    source_genes: int
    source_cells: int
    kept_records: int
    dropped_records: int
    rules: dict[str, str]
    per_metal: list[MetalLedger]


def build_drop_log(dataset_name: str, sheets: Mapping[str, Sequence[S5Row]]) -> DropLog:
    """Count every released cell into kept or the one drop rule."""
    ledgers: list[MetalLedger] = []
    for spec in METALS:
        rows = sheets[spec.code]
        empty = [r for r in rows if r.is_empty]
        kept = [r for r in rows if not r.is_empty]
        ledgers.append(
            MetalLedger(
                metal=spec.code,
                sheet=spec.sheet,
                source_cells=len(rows),
                kept_records=len(kept),
                dropped_empty=len(empty),
                dropped_empty_states=dict(
                    sorted(Counter(r.state for r in empty).items())
                ),
                dropped_empty_zero_sites=sum(1 for r in empty if r.sites == 0),
                kept_exact_zero=sum(1 for r in kept if r.log2fc == 0.0),
                kept_q_le_threshold=sum(1 for r in kept if r.q_value <= Q_THRESHOLD),
            )
        )
    kept_total = sum(m.kept_records for m in ledgers)
    source_total = sum(m.source_cells for m in ledgers)
    return DropLog(
        dataset=dataset_name,
        source_genes=N_GENES,
        source_cells=source_total,
        kept_records=kept_total,
        dropped_records=source_total - kept_total,
        rules=DROP_RULE_DESCRIPTIONS,
        per_metal=ledgers,
    )


def stored_cells(
    sheets: Mapping[str, Sequence[S5Row]], symbols: Mapping[str, str]
) -> Iterator[tuple[str, str, MetalSpec, float]]:
    """``(locus tag, symbol, metal, log2FC)`` for every kept cell, metal-major."""
    for spec in METALS:
        for row in sheets[spec.code]:
            if not row.is_empty:
                yield row.orf, symbols[row.orf], spec, row.log2fc


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
DATASET_ROOT_REL = "data/torchcell/pputida_env_metal_tnseq_royet2025"


@register_dataset
class EnvMetalTnseqRoyet2025Dataset(ExperimentDataset):
    """KT2440 mariner Tn-seq log2FC of every gene in cobalt, copper, zinc and cadmium."""

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        pputida_genome: PPutidaKT2440Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``pputida_genome`` is injected by the build entry points."""
        self.pputida_genome = pputida_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialEnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialEnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Every consumed file, linked from the raw mirror."""
        return [raw.name for raw in RAW_FILES]

    def download(self) -> None:
        """Link each mirror file into ``raw/`` after checking the manifest and sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for raw in RAW_FILES:
            check_manifest_pin(
                raw.mirror_relpath,
                manifest_sha256(manifest, raw.mirror_relpath),
                raw.sha256,
            )
            src = raw_mirror_dir(data_root) / raw.mirror_relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, raw.name), raw.sha256)
        log.info("Royet 2025 raw files linked into %s (sha256 verified)", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected genome, or the KT2440 default cache (a direct run)."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", "KT2440")
        return self.pputida_genome

    @post_process
    def process(self) -> None:
        """Parse Table S5 into per-gene, per-metal records + LMDB, checking it first."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        pools = read_replicate_pools(osp.join(self.raw_dir, TABLE_S3))
        sheets = read_release(osp.join(self.raw_dir, TABLE_S5))
        genome = self._genome()
        orfs = [row.orf for row in sheets[METALS[0].code]]
        _, reconciliation = reconcile_locus_tags(
            genome, _series(orfs), label=f"{self.name} Table S5 #Orf"
        )
        reconciliation.require_resolved(MIN_RESOLVED_FRACTION)
        if reconciliation.remapped:
            raise SheetFormatError(
                f"{reconciliation.remapped} #Orf tags are not current locus tags"
            )
        symbols = {tag: canonical_symbol(genome, tag) for tag in orfs}
        drop_log = build_drop_log(self.name, sheets)
        if (
            drop_log.kept_records != EXPECTED_RECORDS
            or {m.metal: m.dropped_empty for m in drop_log.per_metal}
            != EXPECTED_DROPPED
        ):
            raise RuntimeError(
                f"{drop_log.kept_records} records kept, "
                f"{[(m.metal, m.dropped_empty) for m in drop_log.per_metal]} dropped; "
                f"the module declares {EXPECTED_RECORDS} and {EXPECTED_DROPPED}"
            )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_ledgers(drop_log, reconciliation, pools, sheets)

        environments = {spec.code: environment(spec) for spec in METALS}
        genome_reference = reference_genome()
        references = {
            spec.code: build_reference(
                self.name, genome_reference, environments[spec.code]
            )
            for spec in METALS
        }
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        index = 0
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for locus_tag, symbol, spec, value in tqdm(
                stored_cells(sheets, symbols),
                total=drop_log.kept_records,
                desc="royet2025",
            ):
                experiment = BacterialEnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=insertion_genotype(locus_tag, symbol),
                    environment=environments[spec.code],
                    phenotype=phenotype(value),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(
                        experiment, references[spec.code], PUBLICATION, itxn
                    ),
                )
                index += 1
        env_out.close()
        interned_env.close()
        if index != drop_log.kept_records:
            raise RuntimeError(
                f"wrote {index} records, the ledger counted {drop_log.kept_records}"
            )
        log.info("Wrote %d Royet 2025 environment-response experiments to LMDB", index)

    def _write_ledgers(
        self,
        drop_log: DropLog,
        reconciliation: LocusTagReconciliation,
        pools: Mapping[str, int],
        sheets: Mapping[str, Sequence[S5Row]],
    ) -> None:
        """The drop log, identifiers, replicate pools, unstored columns and field gaps."""
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(drop_log.model_dump_json(indent=2))
        (out / "locus_tag_reconciliation.json").write_text(
            reconciliation.model_dump_json(indent=2)
        )
        (out / "replicate_structure.json").write_text(
            json.dumps({"table_s3_pools_per_arm": dict(pools)}, indent=2)
        )
        (out / "perturbation_field_gaps.json").write_text(
            json.dumps(
                [gap.model_dump(mode="json") for gap in PERTURBATION_FIELD_GAPS],
                indent=2,
            )
        )
        (out / "not_stored.json").write_text(
            json.dumps(
                {
                    "why_not_stored": "EnvironmentResponsePhenotype carries no test "
                    "statistic and no per-arm read count; the values are kept here "
                    "verbatim (as parsed) rather than coerced onto a field",
                    "columns": ["Mean A", "Mean B", "Delta sum", "p-value", "q-value"],
                    "values": {
                        spec.code: {
                            row.orf: [
                                row.mean_a,
                                row.mean_b,
                                row.delta_sum,
                                row.p_value,
                                row.q_value,
                            ]
                            for row in sheets[spec.code]
                        }
                        for spec in METALS
                    },
                }
            )
        )
        (out / "sourced_values.json").write_text(
            json.dumps(
                {k: v.model_dump(mode="json") for k, v in SOURCED_VALUES.items()},
                indent=2,
            )
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process()."""
        raise NotImplementedError(
            "EnvMetalTnseqRoyet2025Dataset builds records in process()"
        )


def _series(values: Sequence[str]) -> pd.Series:
    """A pandas Series of the released tags (``reconcile_locus_tags`` takes one)."""
    return pd.Series(list(values), dtype=object)


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built tree
# --------------------------------------------------------------------------- #
def verify_build(
    dataset_root: str,
    *,
    genome: PPutidaKT2440Genome | None = None,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the environment-response L0-L4 gate on a built tree and write its report.

    Every record is checked against the KT2440 genome its reference pins (the resolver
    of the canonical-name rule, and every GenBank locus of the assembly as the L4
    universe), and every ``SourcedValue`` is audited against the pinned ``paper.md``
    when the literature mirror is mounted. The report is written to
    ``preprocess/verification_report.json``.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records
    from torchcell.verification.sourced import library_available

    base = data_root or _data_root()
    if genome is None:
        genome = bacterial_genome("pputida", "KT2440", base)
    report = verify_environment_response_dataset_streaming(
        stream_records(dataset_root),
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/data/{TABLE_S5}",
            citation_key=CITATION_KEY,
            sha256=DATA_SHA256[TABLE_S5],
            method="Table S5, TRANSIT RESAMPLING LB vs metal; one "
            "BacterialEnvironmentResponseExperiment per (gene, metal), gene-level "
            "TransposonInsertionPerturbation against KT2440, reference = the parent in "
            "the same metal at log2FC 0",
            page="Table S5 (EMI-27-e70095-s006.xlsx), sheets LB-Co, LB-Cu, LB-Zn, LB-Cd",
            retrieved=DATA_RETRIEVED_AT,
        ),
        expected_count=expected_count,
        sgd_genes=set(genome.genbank.loci),
        resolve_gene_name=genome.resolve_gene_name,
    )
    library_root = Path(base) / "torchcell-library"
    if library_available(library_root):
        for value in SOURCED_VALUES.values():
            report.add(audit_sourced_value(value, library_root))
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` the dev LMDB, or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.pputida.royet2025"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser("deposit", help="deposit the raw mirror")
    deposit.add_argument(
        "--retrieve-into",
        default=None,
        help="re-run every recorded PMC retrieval into this directory and deposit those "
        "bytes; without it the literature mirror's captured SI files are used",
    )
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "deposit":
        sources: dict[str, str | Path]
        if args.retrieve_into is not None:
            sources = dict(retrieve_raw_files(args.retrieve_into))
        else:
            sources = {
                name: library_dir(data_root) / "si" / LIBRARY_SI_NAME[name]
                for name in DATA_SHA256
            }
        print(deposit_raw_mirror(sources=sources, data_root=data_root))
        return 0
    root = osp.join(data_root, DATASET_ROOT_REL)
    if args.command == "build":
        dataset = EnvMetalTnseqRoyet2025Dataset(root=root)
        print(f"len = {len(dataset)}")
        return 0
    report = verify_build(root, data_root=data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
