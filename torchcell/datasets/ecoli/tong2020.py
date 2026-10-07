# torchcell/datasets/ecoli/tong2020
# [[torchcell.datasets.ecoli.tong2020]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/tong2020
# Test file: tests/torchcell/datasets/ecoli/test_tong2020.py
"""Tong 2020: E. coli single-deletion colony growth on thirty carbon sources.

Tong et al. 2020 (mBio 11:e02259-20, doi:10.1128/mBio.02259-20; PMID 32994326;
PMC7527729) pinned the Keio collection and a library of small RNA and small protein
deletions at 1,536 colonies per plate onto solid MOPS minimal agar carrying one carbon
source, and scanned each plate every 20 min for 24 h at 37 C. Table S1A ("Endpoint
Biomass") releases the normalized 24 h end-point growth of 3,796 strains on 30 carbon
sources, 113,880 cells, one value per (strain, carbon source).

RECORD = one (deletion strain x carbon source) ``BacterialFitnessExperiment``:

- GENOTYPE: one ``BacterialDeletionPerturbation``. Two collections, two backgrounds, so
  two ``AssemblyReferenceGenome`` pins. A Keio strain is written against BW25113
  (``ecoli_k12_bw25113_locus_tag``, collection "Keio collection", cassette "kanamycin
  cassette"); a strain of the sRNA and small protein library against MG1655
  (``ecoli_k12_mg1655_bnumber``).
- ENVIRONMENT: ``TONG2020_MOPS_MINIMAL_AGAR``, the solid form of the shared
  ``MOPS_MINIMAL`` (``base_medium="MOPS_MINIMAL"``), at 37 C for 24 h, plus ONE
  ``EnvironmentPhysicalPerturbation(factor=carbon_source)`` whose ``agent`` is the
  carbon source as the column header names it. The carbon-source concentration is not in
  the mirror (the paper puts it on its Carbon Phenotype Explorer web app), so
  ``magnitude`` is a typed ``ProvenanceGap``, never a guess.
- PHENOTYPE: ``FitnessPhenotype``; the released value is the end-point colony integrated
  density minus the first time point, divided by the interquartile mean of its plate row
  and then of its column, so a typical (wild-type-like) colony is 1.0. The reference
  carries 1.0.

WHY ``FitnessPhenotype`` AND NOT ``EnvironmentResponsePhenotype``. The plan's section 3c
table lists this row among the four chemical-genomics rows it serves with
``EnvironmentResponsePhenotype``, and the same table lists "Keio growth" under
``FitnessPhenotype``. The readout decides it: each carbon source was normalized on its own
plates ("We normalized each carbon source treatment individually"), so the number is a
strain's growth relative to a typical colony IN THE SAME ENVIRONMENT, a ko/wt ratio with
baseline 1, not a response relative to a control environment. ``MeasurementType`` has no
member for a normalized end-point colony size (``colony_size`` is defined as
unnormalized, ``growth_rate`` is a rate), and the environment-response verifier's numeric
L3 rule requires a reference value of 0. ``FitnessPhenotype`` is what the yeast SGA
colony-size fitness datasets use for the same kind of number.

THE TWO BACKGROUNDS. "The sRNA and small protein deletion library were generated in E.
coli strain MG1655 ..., while the Keio collection was in E. coli strain BW25113". Table
S1A does not label a row's collection. Table S1B ("Comparison to Keio Data") lists the
3,727 rows (3,725 distinct b-numbers) whose Table S1A glucose value the paper compares
with the Keio collection's own MOPS measurements (Baba 2006); every one of them is in
Table S1A and its glucose value is identical. The 71 Table S1A b-numbers absent from S1B
are, measured on the MG1655 GenBank annotation, 36 ncRNA genes (rybB, micC, oxyS, sgrS,
...), 32 protein-coding genes of 60 to 237 nt (at most 78 codons: tisB, ldrA, mokC, ...),
the tmRNA ssrA (b2621), the 87 nt pseudogene yeeH (b4639) and b4590 (absent from MG1655);
the 3,725 S1B b-numbers include no ncRNA gene. That is the content of the second library.
So a row in S1B is a Keio strain (BW25113) and a row outside it a strain of the sRNA and
small protein library (MG1655). This assignment is DERIVED from the two tables plus the
gene class; the paper never states it per row.

IDENTIFIERS. Table S1A reports MG1655 b-numbers for both collections. The Keio rows are
written against BW25113, whose GenBank locus tags (``BW25113_NNNN``) are not derivable from
a b-number by string surgery (plan D5), so each Keio b-number is first reconciled against
the MG1655 annotation (``reconcile_locus_tags``) and then carried to BW25113 through the
one-to-one ECK synonym join (``bacteria_common.eck_crosswalk``), the only published 1:1
key between the two annotations (plan section 4). The record has no slot that names the
derivation, so the per-strain map (reported b-number, MG1655 tag, ECK id, BW25113 tag)
is written to ``preprocess/identifier_reconciliation.json`` beside the three
reconciliation histograms.

RECORDS DROPPED (rule + counts in ``preprocess/dropped_records.json``), each a whole strain
across all 30 carbon sources:

1. ``b_number_is_a_fragment_of_a_merged_mg1655_locus``: two or more reported b-numbers
   resolve to ONE current MG1655 locus (the Keio collection deleted the fragments of a
   pseudogene that the current annotation merges, e.g. yedS_1/yedS_2/yedS_3 as b1964,
   b1965, b1966 against b4496). They are distinct strains, so mapping them onto one locus
   would merge them. The member that IS the locus tag (b2681 beside its merged fragment
   b2680) keeps it; the members that reach it only through a synonym are dropped, the
   precedent Smith 2006 sets for an alias colliding with a directly present ORF. A group
   with no direct member is dropped whole.
2. ``b_number_is_not_in_the_mg1655_annotation``: the reported b-number is no locus,
   symbol or synonym of GCA_000005845.2 (``RETIRED``).
3. ``b_number_is_ambiguous_in_mg1655``: the b-number matches more than one MG1655 locus.
4. ``mg1655_locus_has_no_one_to_one_eck_partner_in_bw25113`` (Keio rows only): the
   current MG1655 locus shares no ECK synonym one-to-one with a BW25113 locus, so no
   BW25113 locus tag can be written for the strain.
5. ``carbon_source_cell_is_blank``: a strain has no value in a carbon-source column (0 in
   the released table).

SOURCED VALUES (module-level ``SourcedValue``s anchored to the sha256 of
``tongGeneDispensabilityEscherichia2020/paper.md``, or typed ``ProvenanceGap``s):

- ``n_samples = 2``, ``sample_unit=technical_replicate`` for the Keio records: "The assay
  plates were tested in two technical replicates, both at 1,536-colony-density." The
  paragraph describes the Keio plates; the sRNA and small protein library's plate layout
  is not described, so those records carry a ``not_reported_by_primary`` gap on
  ``n_samples`` and ``sample_unit``.
- Uncertainty: Table S1A releases one value per cell and no dispersion, so
  ``fitness_uncertainty`` is a ``not_reported_by_primary`` gap. The only replicate
  statistic in the paper is the whole-data-set correlation between the two technical
  replicates in Fig. S3a (an image; R = 0.911 in its panel).
- ``Temperature(37.0)``, ``duration_hours = 24.0`` and the solid MOPS medium from the
  screening paragraph; ``aerobic`` from plates incubated in air.

DATA SOURCE: Table S1, ``mBio.02259-20-st001.xlsx``, from the PMC Article Datasets bucket
(``pmc_cloud``), deposited in ``$DATA_ROOT/torchcell-raw/tongGeneDispensabilityEscherichia2020/``
with a ``manifest.json``. The published bytes hold a complete xlsx archive (the first
1,803,783 bytes, whose own end-of-central-directory record is consistent) followed by a
partial second copy of the same archive whose central directory offsets point into the
first; Python's ``zipfile`` therefore refuses the file as published. Every member of the
leading archive carries the CRC32 and sizes the trailing central directory lists, and the
six trailing members that are complete are byte-identical to the leading ones, so the
loader reads the leading archive and pins its sha256 (``LEADING_ARCHIVE_SHA256``) as well
as the file's.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import os.path as osp
import shutil
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import MOPS_MINIMAL
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialDeletionPerturbation,
    BacterialFitnessExperiment,
    BacterialFitnessExperimentReference,
    BacterialGeneNamespace,
    Environment,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    FitnessPhenotype,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    PhysicalFactor,
    Publication,
    SampleUnit,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    STRAIN_GENE_NAMESPACES,
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    eck_crosswalk,
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
from torchcell.literature.retrieve import pmc_cloud_url
from torchcell.sequence.genome.bacterial import BacterialGenome
from torchcell.sequence.genome.ecoli.k12 import (
    EcoliK12BW25113Genome,
    EcoliK12Genome,
    EcoliK12MG1655Genome,
    EcoliK12StrainName,
)
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

DOI = "10.1128/mBio.02259-20"
PMID = "32994326"
PMCID = "PMC7527729"

CITATION_KEY = "tongGeneDispensabilityEscherichia2020"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

XLSX_FILENAME = "mBio.02259-20-st001.xlsx"
XLSX_REL = f"data/{XLSX_FILENAME}"
XLSX_SHA256 = "d291160d41f65a1a3bd4fc8fe7d58e42bbdd92b4be2ef8c74bb2f284db8e8c84"
XLSX_RETRIEVED_AT = "2026-10-07"
#: The PMC Article Datasets bucket key of Table S1 (article version 1).
PMC_CLOUD_KEY = f"{PMCID}.1/{XLSX_FILENAME}"
XLSX_URL = pmc_cloud_url(PMC_CLOUD_KEY)

#: sha256 of the complete xlsx archive at the start of the published bytes (module
#: docstring, DATA SOURCE): the bytes up to and including its first end-of-central-
#: directory record.
LEADING_ARCHIVE_SHA256 = (
    "1c3f7b8aea226f90e5bcb237ecf483f62dcac84c63569a57b13307fba54aa166"
)

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "daea2b924f553b75c3bfc626b2dbd10147e4a23c4272dd4b03b703242ac1623a"

#: The Carbon Phenotype Explorer, where the paper lists each carbon source's
#: concentration. A Shiny app: a scripted GET returns only its "Please Wait" loader
#: page (measured 2026-10-07), so the concentrations are a manual-browser retrieval.
CARPE_URL = "https://edbrownlab.shinyapps.io/CarPE/"

ENDPOINT_SHEET = "S1A - Endpoint Biomass"
KEIO_COMPARISON_SHEET = "S1B - Comparison to Keio Data"
ID_COL = "B-numbers"
GENE_COL = "Gene"
#: The b-number column of Table S1B; the trailing space is in the released header.
KEIO_ID_COL = "B-number "

#: The 30 carbon-source columns of Table S1A, verbatim and in sheet order. Each header is
#: the source label handed to ``resolved_compound``.
CARBON_SOURCE_COLUMNS: tuple[str, ...] = (
    "Galactose",
    "L-alanine",
    "D-alanine",
    "Mannose",
    "Glucosamine",
    "Thymidine",
    "Adenosine",
    "Saccharate",
    "Acetate",
    "α-ketoglutarate",
    "Malate",
    "Succinate",
    "Fumarate",
    "Ribose",
    "Fucose",
    "Glycerol",
    "Lactate",
    "Oxaloacetate",
    "Pyruvate",
    "Galacturonate",
    "Maltose",
    "Fructose",
    "Trehalose",
    "Mannitol",
    "Sorbitol",
    "Glucuronate",
    "Gluconate",
    "Xylose",
    "Glucose",
    "N-acetyl Glucosamine",
)

#: Fraction of a collection's distinct b-numbers that must resolve to one MG1655 locus
#: (checklist item 4). Below it the build stops and reports instead of dropping strains.
#: Measured on the release: 3,720 of 3,725 Keio b-numbers and 70 of 71 library
#: b-numbers resolve.
MIN_RESOLVED_FRACTION = 0.95

KEIO_COLLECTION = "Keio collection"
SRNA_COLLECTION = "sRNA and small protein deletion library"
KEIO_CASSETTE = "kanamycin cassette"

Collection = Literal["keio", "srna"]

#: The reference strain each collection's deletions are edits against.
COLLECTION_STRAIN: dict[Collection, EcoliK12StrainName] = {
    "keio": "BW25113",
    "srna": "MG1655",
}


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned paper OCR mirror."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page=page,
        ),
    )


_STRAINS = "Materials and Methods, 'Bacterial strains and growth conditions'"
_SCREEN = "Materials and Methods, 'Screening conditions'"
_ANALYSIS = "Materials and Methods, 'Data analysis'"
_CHEMICALS = "Materials and Methods, 'Chemicals'"
_CARPE = "Materials and Methods, 'Carbon Phenotype Explorer'"

KEIO_STRAIN = _paper(
    "BW25113", "while the Keio collection was in E. coli strain BW25113", page=_STRAINS
)
SRNA_STRAIN = _paper(
    "MG1655",
    "The sRNA and small protein deletion library were generated in E. coli strain "
    "MG1655",
    page=_STRAINS,
)
SCREENED_COLLECTIONS = _paper(
    (KEIO_COLLECTION, SRNA_COLLECTION),
    "In this study, we screened the Keio collection of E. coli K-12 nonessential gene "
    "deletions (11) and 100 small RNA (sRNA) and small protein deletions (18) in 30 "
    "different carbon sources.",
    page=_STRAINS,
)
N_STRAINS = _paper(
    3796,
    "In our final data set, we have gathered information on 3,796 strains of E. coli .",
    page=_STRAINS,
    note="equals the 3,796 rows of Table S1A",
)
N_REPLICATES = _paper(
    2,
    "The assay plates were tested in two technical replicates, both at "
    "1,536-colony-density.",
    page=_SCREEN,
    note="the two technical replicate assay plates of the Keio screen. How the two "
    "plates combine into the single Table S1A value is not stated for the end point; "
    "the kinetic curves are averaged ('We then took the average of our two replicates "
    "at each time point.'). The count is of plate measurements behind each value",
)
_INCUBATION_QUOTE = (
    "Assay plates were incubated for $2 4 ~ \\mathrm { h }$ at "
    "$3 7 ^ { \\circ } \\mathsf { C }$"
)
TEMPERATURE_C = _paper(37.0, _INCUBATION_QUOTE, page=_SCREEN)
DURATION_HOURS = _paper(
    24.0,
    _INCUBATION_QUOTE,
    page=_SCREEN,
    note="the assay plate's incubation, over which the released end point is read. The "
    "plates that inoculated it had already spent 24 h on the same carbon source to "
    "deplete carried-over nutrients, which this field does not count",
)
AEROBICITY = _paper(
    "aerobic",
    _INCUBATION_QUOTE,
    page=_SCREEN,
    note="agar plates incubated in a scanner incubator; no anaerobic chamber or gas "
    "control is described",
)
SOLID_MEDIUM = _paper(
    "solid",
    "Keio plates were pinned from LB agar plates onto solid MOPS minimal media "
    "containing a carbon source in 1,536-colony-density.",
    page=_SCREEN,
)
CARBON_IS_THE_VARIABLE = _paper(
    "carbon source varied on a fixed MOPS base",
    "Using a chemically defined minimal medium (morpholinepropanesulfonic acid [MOPS]) "
    "and changing only the carbon source (34)",
    page="Introduction",
)
MOPS_VENDOR = _paper(
    "Teknova MOPS minimal medium",
    "MOPS minimal media (Teknova) was used for all work in minimal media.",
    page=_CHEMICALS,
)
CARBON_CONCENTRATION_LOCATION = _paper(
    CARPE_URL,
    "A full list of carbon sources and the concentrations can be found on the carbon "
    "conditions tab in the Carbon Phenotype Explorer "
    "(https://edbrownlab.shinyapps.io/CarPE/).",
    page=_CHEMICALS,
)
EQUAL_CARBON = _paper(
    "equal carbon added per condition",
    "Concentrations were picked so that all carbon sources resulted in the same amount "
    "of carbon added.",
    page=_CHEMICALS,
    note="a design rule, not a concentration; it does not fix any one condition's "
    "molarity without the per-condition list",
)
ENDPOINT_DEFINITION = _paper(
    "last minus first time point",
    "For the endpoint values, the first time point was subtracted from the last time "
    "point to remove any background noise caused by a large initial inoculum.",
    page=_ANALYSIS,
)
NORMALIZATION = _paper(
    "row then column interquartile mean",
    "The raw integrated density values of each colony were divided by the "
    "interquartile mean of the row and then by the column in which it belonged.",
    page=_ANALYSIS,
)
WILD_TYPE_BASELINE = _paper(
    1.0,
    "Given that most of our data normalize to the same growth value of one, we can make "
    "the assumption that most gene mutations do not affect the growth of E. coli.",
    page=_CARPE,
    note="the reference fitness: a typical colony on the plate normalizes to 1",
)
KEIO_CASSETTE_QUOTE = _paper(
    KEIO_CASSETTE,
    "Since the mutants in the Keio collection contain a kanamycin cassette",
    page="Results, 'Carbon Phenotype Explorer'",
)
RELEASED_TABLE = _paper(
    "Table S1A",
    "These endpoint biomass values formed a highly replicating final data set of growth "
    "amplitudes for 3,796 genes tested across 30 different carbon sources (see Fig. S3 "
    "and Table S1A in the supplemental material).",
    page="Results, 'A genome-wide screen of E. coli in different carbon sources'",
)

#: Every module-level sourced value, for the mirror audit.
SOURCED_VALUES: tuple[SourcedValue, ...] = (
    KEIO_STRAIN,
    SRNA_STRAIN,
    SCREENED_COLLECTIONS,
    N_STRAINS,
    N_REPLICATES,
    TEMPERATURE_C,
    DURATION_HOURS,
    AEROBICITY,
    SOLID_MEDIUM,
    CARBON_IS_THE_VARIABLE,
    MOPS_VENDOR,
    CARBON_CONCENTRATION_LOCATION,
    EQUAL_CARBON,
    ENDPOINT_DEFINITION,
    NORMALIZATION,
    WILD_TYPE_BASELINE,
    KEIO_CASSETTE_QUOTE,
    RELEASED_TABLE,
)

#: The solid MOPS minimal medium of the screen. ``MOPS_MINIMAL`` is the shared recipe
#: (Neidhardt 1974 as tabulated by Price 2018, which the media library adopts for Tong
#: 2020's Teknova medium); the screen plates are its agar form, and the paper gives no
#: agar amount. The carbon source is the varied factor, so it is not a component.
TONG2020_MOPS_MINIMAL_AGAR = Media(
    name="MOPS minimal agar, carbon source varied (Tong 2020)",
    state="solid",
    is_synthetic=True,
    base_medium="MOPS_MINIMAL",
    components=[
        *MOPS_MINIMAL.components,
        MediaComponent(
            compound=resolved_compound("agar"),
            role=MediaComponentRole.gelling_agent,
            provenance=[SOLID_MEDIUM],
            note="the paper states solid MOPS minimal medium and gives no agar amount",
        ),
    ],
    provenance=[SOLID_MEDIUM, CARBON_IS_THE_VARIABLE, MOPS_VENDOR],
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/tongGeneDispensabilityEscherichia2020``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_s1_retrieval(retrieved_at: str = XLSX_RETRIEVED_AT) -> RetrievalRecord:
    """The recorded retrieval of Table S1: one PMC Article Datasets object.

    ``run_retriever`` on this record re-fetches the bytes; the pinned sha256 is the
    anchor a rebuild verifies against.
    """
    return RetrievalRecord(
        method=RetrievalMethod.pmc_cloud,
        source_url=XLSX_URL,
        retriever="torchcell.literature.retrieve.pmc_cloud_object",
        params={"key": PMC_CLOUD_KEY},
        sha256=XLSX_SHA256,
        retrieved_at=retrieved_at,
    )


def deposit_raw_mirror(
    *,
    xlsx_path: str | Path,
    retrieved_at: str = XLSX_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from the already-retrieved Table S1 plus its ``manifest.json``.

    Idempotent by sha256: an existing mirror file with the recorded hash is left alone and
    a differing one raises rather than being overwritten. The bytes are verified against
    ``XLSX_SHA256`` before anything is written.
    """
    got = _sha256(xlsx_path)
    if got != XLSX_SHA256:
        raise RuntimeError(
            f"{xlsx_path} sha256 mismatch: got {got}, expected {XLSX_SHA256}"
        )
    root = raw_mirror_dir(data_root)
    dest = root / XLSX_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != XLSX_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(xlsx_path, dest)
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=(
            "Gene Dispensability in Escherichia coli Grown in Thirty Different Carbon "
            "Environments"
        ),
        files=[
            ArtifactRecord(
                path=XLSX_REL,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=XLSX_SHA256,
                source=XLSX_URL,
                retrieval=table_s1_retrieval(retrieved_at),
            )
        ],
        si_data_sources=[XLSX_URL],
        si_expected=[
            "Table S1 (mBio.02259-20-st001.xlsx): sheet 'S1A - Endpoint Biomass' (the "
            "per-strain, per-carbon-source normalized end-point growth this dataset is "
            "built from) and sheet 'S1B - Comparison to Keio Data' (the Keio membership "
            "the background assignment reads)",
            "Carbon Phenotype Explorer 'carbon conditions' tab (carbon-source "
            f"concentrations, {CARPE_URL}): a Shiny app, not scriptable, NOT mirrored",
        ],
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
# Reading Table S1
# --------------------------------------------------------------------------- #
_EOCD_SIGNATURE = b"PK\x05\x06"
_EOCD_FIXED_BYTES = 22


def leading_archive(data: bytes) -> bytes:
    """The bytes up to and including the FIRST end-of-central-directory record.

    For the published Table S1 that is the complete xlsx archive the file starts with
    (module docstring, DATA SOURCE). The caller verifies the result against
    ``LEADING_ARCHIVE_SHA256``, so a different file cannot be read by accident.
    """
    eocd = data.find(_EOCD_SIGNATURE)
    if eocd < 0:
        raise ValueError("no zip end-of-central-directory record in the bytes")
    comment_length = int.from_bytes(data[eocd + 20 : eocd + 22], "little")
    return data[: eocd + _EOCD_FIXED_BYTES + comment_length]


class TableS1(BaseModel):
    """The two sheets of Table S1 the loader reads."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    endpoint: pd.DataFrame
    keio_ids: frozenset[str]
    keio_rows: int


def read_table_s1(path: str | Path) -> TableS1:
    """Read sheets S1A and S1B from the published Table S1 bytes at ``path``.

    Refuses a leading archive whose sha256 is not ``LEADING_ARCHIVE_SHA256``, an S1A whose
    header is not ``(B-numbers, Gene, *CARBON_SOURCE_COLUMNS)``, a repeated S1A b-number,
    and an S1B b-number absent from S1A.
    """
    archive = leading_archive(Path(path).read_bytes())
    got = hashlib.sha256(archive).hexdigest()
    if got != LEADING_ARCHIVE_SHA256:
        raise RuntimeError(
            f"{path}: leading xlsx archive sha256 {got}, expected "
            f"{LEADING_ARCHIVE_SHA256}"
        )
    endpoint = pd.read_excel(
        io.BytesIO(archive), sheet_name=ENDPOINT_SHEET, engine="openpyxl"
    )
    expected = [ID_COL, GENE_COL, *CARBON_SOURCE_COLUMNS]
    if list(endpoint.columns) != expected:
        raise ValueError(
            f"{ENDPOINT_SHEET} header {list(endpoint.columns)} is not {expected}"
        )
    endpoint[ID_COL] = endpoint[ID_COL].astype(str)
    repeated = endpoint[ID_COL][endpoint[ID_COL].duplicated()].tolist()
    if repeated:
        raise ValueError(f"{ENDPOINT_SHEET} repeats b-numbers {repeated}")
    comparison = pd.read_excel(
        io.BytesIO(archive), sheet_name=KEIO_COMPARISON_SHEET, engine="openpyxl"
    )
    keio_ids = frozenset(comparison[KEIO_ID_COL].astype(str))
    absent = sorted(keio_ids - set(endpoint[ID_COL]))
    if absent:
        raise ValueError(f"{KEIO_COMPARISON_SHEET} b-numbers absent from S1A: {absent}")
    return TableS1(endpoint=endpoint, keio_ids=keio_ids, keio_rows=len(comparison))


# --------------------------------------------------------------------------- #
# Identifiers: the two collections, the MG1655 reconciliation, the ECK crosswalk
# --------------------------------------------------------------------------- #
class StrainRecord(BaseModel):
    """One Table S1A row the build keeps, with the identity it is stored under."""

    model_config = ConfigDict(frozen=True)

    row: int
    reported: str
    collection: Collection
    systematic_gene_name: str
    perturbed_gene_name: str
    gene_namespace: BacterialGeneNamespace


class CrosswalkUse(BaseModel):
    """How one Keio b-number reached its BW25113 locus tag."""

    model_config = ConfigDict(frozen=True)

    reported: str
    mg1655: str
    eck: str
    bw25113: str
    numerics_agree: bool


class ResolvedStrains(BaseModel):
    """The strains a build keeps and the per-rule drop lists, with the three reports."""

    kept: list[StrainRecord]
    dropped_fragment: list[str]
    dropped_not_in_mg1655: list[str]
    dropped_ambiguous: list[str]
    dropped_no_eck_partner: list[str]
    keio_vs_mg1655: LocusTagReconciliation
    srna_vs_mg1655: LocusTagReconciliation
    keio_vs_bw25113: LocusTagReconciliation
    crosswalk_pairs: int
    crosswalk: list[CrosswalkUse]


def canonical_symbol(genome: BacterialGenome[Any], tag: str) -> str:
    """The genome's own gene symbol for ``tag`` when it resolves back to ``tag``.

    One spelling per locus is what keeps a perturbation from splitting into two graph
    nodes, so the symbol comes from the pinned annotation, not from the release's Gene
    column. A locus with no symbol, or whose symbol resolves elsewhere, is named by its
    tag.
    """
    symbol = genome.genbank.loci[tag].symbol
    if symbol is None:
        return tag
    resolved = genome.resolve_gene_name(symbol).systematic_name
    return symbol if resolved == tag else tag


def _drop_reason(
    name: str, report: LocusTagReconciliation, loci: Mapping[str, object]
) -> str | None:
    """Which MG1655-side drop rule a reported b-number falls under, if any.

    In a collision group (two or more reported b-numbers resolving to one MG1655 locus)
    the member that IS that locus tag is kept under it and the members that reached it
    through a synonym are dropped, the precedent the yeast loaders set for an alias that
    collides with a directly present ORF (Smith 2006). A group with no direct member is
    dropped whole.
    """
    if name in report.kept_on_collision and name not in loci:
        return "fragment"
    if name in report.retired_kept:
        return "not_in_mg1655"
    if name in report.ambiguous_kept:
        return "ambiguous"
    return None


def resolve_strains(
    endpoint: pd.DataFrame,
    keio_ids: frozenset[str],
    mg1655: EcoliK12MG1655Genome,
    bw25113: EcoliK12BW25113Genome,
    *,
    label: str,
) -> ResolvedStrains:
    """Assign each Table S1A row its collection and the locus tag it is stored under.

    A b-number in S1B is a Keio strain, any other an sRNA and small protein library
    strain (module docstring). Each collection's b-numbers are reconciled against MG1655
    separately (a Keio strain and a library strain of one gene are different strains in
    different backgrounds, not a collision). A library strain keeps its MG1655 b-number.
    A Keio strain's MG1655 locus is carried to BW25113 through the one-to-one ECK pairs;
    the BW25113 tags are then reconciled against BW25113, where every one must be a locus
    tag of the annotation and no two may coincide.
    """
    if not endpoint.index.equals(pd.RangeIndex(len(endpoint))):
        raise ValueError("the endpoint table must carry its positional RangeIndex")
    ids = endpoint[ID_COL]
    is_keio = ids.isin(keio_ids)
    keio_stored, keio_report = reconcile_locus_tags(
        mg1655, ids[is_keio], label=f"{label} Keio b-numbers vs MG1655"
    )
    keio_report.require_resolved(MIN_RESOLVED_FRACTION)
    srna_stored, srna_report = reconcile_locus_tags(
        mg1655, ids[~is_keio], label=f"{label} sRNA/small protein b-numbers vs MG1655"
    )
    srna_report.require_resolved(MIN_RESOLVED_FRACTION)

    crosswalk = eck_crosswalk(mg1655, bw25113)
    pair_of = {pair.mg1655: pair for pair in crosswalk.pairs}
    drops: dict[str, list[str]] = {
        "fragment": [],
        "not_in_mg1655": [],
        "ambiguous": [],
        "no_eck_partner": [],
    }
    staged: list[tuple[int, str, Collection, str]] = []
    uses: list[CrosswalkUse] = []
    for row, reported in ids.items():
        name = str(reported)
        row_index = int(cast(int, row))
        if bool(is_keio[row_index]):
            reason = _drop_reason(name, keio_report, mg1655.genbank.loci)
            if reason is not None:
                drops[reason].append(name)
                continue
            mg_tag = str(keio_stored[row_index])
            pair = pair_of.get(mg_tag)
            if pair is None:
                drops["no_eck_partner"].append(name)
                continue
            uses.append(
                CrosswalkUse(
                    reported=name,
                    mg1655=mg_tag,
                    eck=pair.eck,
                    bw25113=pair.bw25113,
                    numerics_agree=pair.numerics_agree,
                )
            )
            staged.append((row_index, name, "keio", pair.bw25113))
        else:
            reason = _drop_reason(name, srna_report, mg1655.genbank.loci)
            if reason is not None:
                drops[reason].append(name)
                continue
            staged.append((row_index, name, "srna", str(srna_stored[row_index])))

    bw_tags = pd.Series([tag for _, _, c, tag in staged if c == "keio"], dtype=str)
    bw_stored, bw_report = reconcile_locus_tags(
        bw25113, bw_tags, label=f"{label} ECK-crosswalked Keio tags vs BW25113"
    )
    bw_report.require_resolved(1.0)
    if bw_report.remapped or bw_report.kept_on_collision or bw_report.outside_namespace:
        raise RuntimeError(
            f"{label}: the ECK crosswalk produced BW25113 tags that are not distinct "
            f"locus tags of the annotation (remapped {bw_report.remapped}, collisions "
            f"{bw_report.kept_on_collision}, outside {bw_report.outside_namespace})"
        )
    if not bw_stored.equals(bw_tags):
        raise RuntimeError(f"{label}: BW25113 reconciliation changed a crosswalked tag")

    genomes: dict[Collection, BacterialGenome[Any]] = {"keio": bw25113, "srna": mg1655}
    kept = [
        StrainRecord(
            row=row_index,
            reported=name,
            collection=collection,
            systematic_gene_name=tag,
            perturbed_gene_name=canonical_symbol(genomes[collection], tag),
            gene_namespace=STRAIN_GENE_NAMESPACES[COLLECTION_STRAIN[collection]],
        )
        for row_index, name, collection, tag in staged
    ]
    return ResolvedStrains(
        kept=kept,
        dropped_fragment=sorted(drops["fragment"]),
        dropped_not_in_mg1655=sorted(drops["not_in_mg1655"]),
        dropped_ambiguous=sorted(drops["ambiguous"]),
        dropped_no_eck_partner=sorted(drops["no_eck_partner"]),
        keio_vs_mg1655=keio_report,
        srna_vs_mg1655=srna_report,
        keio_vs_bw25113=bw_report,
        crosswalk_pairs=len(crosswalk.pairs),
        crosswalk=uses,
    )


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the items it removed them for."""

    rule: str
    scope: Literal["strain", "cell"]
    description: str
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]


class IdentifierReport(BaseModel):
    """What the identifier step did, written to ``preprocess/``."""

    dataset: str
    keio_comparison_rows: int
    keio_comparison_b_numbers: int
    collections: dict[str, int]
    keio_vs_mg1655: LocusTagReconciliation
    srna_vs_mg1655: LocusTagReconciliation
    keio_vs_bw25113: LocusTagReconciliation
    crosswalk_pairs: int
    crosswalk_numeric_disagreements_used: list[CrosswalkUse]
    crosswalk: list[CrosswalkUse]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def carbon_source_gap() -> ProvenanceGap:
    """The typed absence of a carbon source's concentration (it lives on CarPE)."""
    return ProvenanceGap(
        field="magnitude",
        reason=ProvenanceGapReason.deferred_pending_source_review,
        resolve_with=Provenance(
            source_uri=CARPE_URL,
            citation_key=CITATION_KEY,
            method="manual browser: a Shiny app; a scripted GET returns only its "
            "'Please Wait' loader page (measured 2026-10-07)",
            page="carbon conditions tab",
        ),
        note="the paper lists every carbon source's concentration only on its Carbon "
        "Phenotype Explorer web app; the concentrations were chosen to add the same "
        "amount of carbon per condition",
    )


def environment(carbon_source: str) -> Environment:
    """The screen plate: solid MOPS minimal medium plus its carbon source."""
    return Environment(
        media=TONG2020_MOPS_MINIMAL_AGAR,
        temperature=Temperature(value=TEMPERATURE_C.value),
        perturbations=[
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.carbon_source,
                agent=resolved_compound(carbon_source),
                provenance_gaps=[carbon_source_gap()],
            )
        ],
        aerobicity=AEROBICITY.value,
        duration_hours=DURATION_HOURS.value,
    )


def _undescribed_replicates() -> list[ProvenanceGap]:
    """The library strains' replicate design: the paper describes only the Keio plates."""
    note = (
        "the screening paragraph describes the Keio plates ('The assay plates were "
        "tested in two technical replicates'); the sRNA and small protein library's "
        "plate layout and replicate count are not described"
    )
    return [
        ProvenanceGap(
            field=field, reason=ProvenanceGapReason.not_reported_by_primary, note=note
        )
        for field in ("n_samples", "sample_unit")
    ]


def _uncertainty_gap() -> ProvenanceGap:
    """Table S1A releases no per-cell dispersion."""
    return ProvenanceGap(
        field="fitness_uncertainty",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="Table S1A releases one value per strain and carbon source and no "
        "dispersion; the paper's only replicate statistic is the whole-data-set "
        "correlation of the two technical replicates in Fig. S3a",
    )


def phenotype(value: float, collection: Collection) -> FitnessPhenotype:
    """One released Table S1A cell as a fitness (a typical colony is 1)."""
    if collection == "keio":
        return FitnessPhenotype(
            fitness=value,
            n_samples=N_REPLICATES.value,
            sample_unit=SampleUnit.technical_replicate,
            provenance_gaps=[_uncertainty_gap()],
        )
    return FitnessPhenotype(
        fitness=value, provenance_gaps=[_uncertainty_gap(), *_undescribed_replicates()]
    )


def reference_phenotype() -> FitnessPhenotype:
    """The wild-type baseline: the per-plate normalization puts a typical colony at 1."""
    return FitnessPhenotype(
        fitness=WILD_TYPE_BASELINE.value,
        provenance_gaps=[
            ProvenanceGap(
                field="n_samples",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the 1.0 baseline is the per-plate normalization (row and column "
                "interquartile means), not a measured set of wild-type replicates",
            )
        ],
    )


def genotype(strain: StrainRecord) -> Genotype:
    """One deletion strain of its collection."""
    keio = strain.collection == "keio"
    return Genotype(
        perturbations=[
            BacterialDeletionPerturbation(
                systematic_gene_name=strain.systematic_gene_name,
                perturbed_gene_name=strain.perturbed_gene_name,
                gene_namespace=strain.gene_namespace,
                collection=KEIO_COLLECTION if keio else SRNA_COLLECTION,
                cassette=KEIO_CASSETTE if keio else None,
            )
        ]
    )


@register_dataset
class CarbonSourceTong2020Dataset(ExperimentDataset):
    """Tong 2020 Keio and sRNA/small protein deletion growth on 30 carbon sources."""

    #: The strain whose genome the build entry points inject as ``ecoli_genome``. The
    #: MG1655 genome the library strains and the crosswalk also need is opened in
    #: ``process`` from its default cache root.
    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = "data/torchcell/ecoli_carbon_source_tong2020",
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; the BW25113 genome is injected or opened in process."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialFitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialFitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The mirrored Table S1 workbook."""
        return [XLSX_FILENAME]

    def download(self) -> None:
        """Link the mirror file into ``raw/`` after verifying it against ``XLSX_SHA256``.

        The mirror plus the pin is canonical; the PMC bucket URL is retrieval metadata
        ``deposit_raw_mirror`` records, never a live build dependency.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        check_manifest_pin(XLSX_REL, manifest_sha256(manifest, XLSX_REL), XLSX_SHA256)
        src = raw_mirror_dir(data_root) / XLSX_REL
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(src, osp.join(self.raw_dir, XLSX_FILENAME), XLSX_SHA256)
        log.info("Tong 2020 Table S1 linked into %s (sha256 verified)", self.raw_dir)

    def _genomes(self) -> tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome]:
        """The MG1655 genome (opened here) and the BW25113 genome (injected or opened)."""
        if self.ecoli_genome is None:  # a direct run; the build entry points inject it
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        bw25113 = self.ecoli_genome
        if not isinstance(bw25113, EcoliK12BW25113Genome):
            raise TypeError(
                f"{self.name} needs the BW25113 genome, got {type(bw25113).__name__}"
            )
        mg1655 = bacterial_genome("ecoli", "MG1655")
        if not isinstance(mg1655, EcoliK12MG1655Genome):
            raise TypeError(f"expected the MG1655 genome, got {type(mg1655).__name__}")
        return mg1655, bw25113

    @post_process
    def process(self) -> None:
        """Parse Table S1A into per-(strain, carbon source) records; write the LMDB."""
        verify_raw_files(self.raw_dir, {XLSX_FILENAME: XLSX_SHA256})
        table = read_table_s1(osp.join(self.raw_dir, XLSX_FILENAME))
        mg1655, bw25113 = self._genomes()
        strains = resolve_strains(
            table.endpoint, table.keio_ids, mg1655, bw25113, label=self.name
        )
        references: dict[Collection, AssemblyReferenceGenome] = {
            collection: assembly_reference(strain)
            for collection, strain in COLLECTION_STRAIN.items()
        }
        environments = {carbon: environment(carbon) for carbon in CARBON_SOURCE_COLUMNS}
        experiment_references = {
            (collection, carbon): BacterialFitnessExperimentReference(
                dataset_name=self.name,
                genome_reference=references[collection],
                environment_reference=environments[carbon],
                phenotype_reference=reference_phenotype(),
            )
            for collection in COLLECTION_STRAIN
            for carbon in CARBON_SOURCE_COLUMNS
        }
        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )
        n_conditions = len(CARBON_SOURCE_COLUMNS)
        source_records = len(table.endpoint) * n_conditions

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_blank_cells = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for strain in tqdm(strains.kept, desc="tong2020"):
                row = table.endpoint.iloc[strain.row]
                strain_genotype = genotype(strain)
                for carbon in CARBON_SOURCE_COLUMNS:
                    value = row[carbon]
                    if pd.isna(value):
                        n_blank_cells += 1
                        continue
                    experiment = BacterialFitnessExperiment(
                        dataset_name=self.name,
                        genotype=strain_genotype,
                        environment=environments[carbon],
                        phenotype=phenotype(float(value), strain.collection),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(
                            experiment,
                            experiment_references[(strain.collection, carbon)],
                            publication,
                            itxn,
                        ),
                    )
                    idx += 1
        env.close()
        interned_env.close()

        rules = [
            DropRule(
                rule="b_number_is_a_fragment_of_a_merged_mg1655_locus",
                scope="strain",
                description=(
                    "two or more reported b-numbers of one collection resolve to ONE "
                    "current MG1655 locus (the collection deleted fragments of a "
                    "pseudogene the current annotation merges, e.g. yedS_1/2/3 as "
                    "b1964/b1965/b1966 against b4496); they are distinct strains, so "
                    "storing them under one locus would merge them, and the record has "
                    "no slot for the deleted fragment. The member that IS the locus tag "
                    "keeps it; the members reaching it through a synonym are dropped"
                ),
                n_records=len(strains.dropped_fragment) * n_conditions,
                items=strains.dropped_fragment,
            ),
            DropRule(
                rule="b_number_is_not_in_the_mg1655_annotation",
                scope="strain",
                description=(
                    "the reported b-number is no locus tag, symbol or synonym of "
                    "GCA_000005845.2, so no locus of either pinned assembly can be "
                    "written for the strain"
                ),
                n_records=len(strains.dropped_not_in_mg1655) * n_conditions,
                items=strains.dropped_not_in_mg1655,
            ),
            DropRule(
                rule="b_number_is_ambiguous_in_mg1655",
                scope="strain",
                description="the reported b-number matches more than one MG1655 locus",
                n_records=len(strains.dropped_ambiguous) * n_conditions,
                items=strains.dropped_ambiguous,
            ),
            DropRule(
                rule="mg1655_locus_has_no_one_to_one_eck_partner_in_bw25113",
                scope="strain",
                description=(
                    "a Keio strain whose current MG1655 locus shares no ECK synonym "
                    "one-to-one with a BW25113 locus; the ECK join is the only "
                    "published one-to-one key between the two annotations, so no "
                    "BW25113 locus tag can be written for the strain"
                ),
                n_records=len(strains.dropped_no_eck_partner) * n_conditions,
                items=strains.dropped_no_eck_partner,
            ),
            DropRule(
                rule="carbon_source_cell_is_blank",
                scope="cell",
                description=(
                    "the strain has no value in this carbon source's column (the "
                    "released table has none, so this is 0)"
                ),
                n_records=n_blank_cells,
                items=[],
            ),
        ]
        drop_log = DropLog(
            dataset=self.name,
            source_records=source_records,
            kept_records=idx,
            dropped_records=source_records - idx,
            rules=rules,
        )
        accounted = sum(rule.n_records for rule in rules)
        if accounted != drop_log.dropped_records:
            raise RuntimeError(
                f"drop accounting mismatch: rules total {accounted}, "
                f"{drop_log.dropped_records} records missing from the build"
            )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        collections = pd.Series([s.collection for s in strains.kept]).value_counts()
        identifier_report = IdentifierReport(
            dataset=self.name,
            keio_comparison_rows=table.keio_rows,
            keio_comparison_b_numbers=len(table.keio_ids),
            collections={str(k): int(v) for k, v in collections.items()},
            keio_vs_mg1655=strains.keio_vs_mg1655,
            srna_vs_mg1655=strains.srna_vs_mg1655,
            keio_vs_bw25113=strains.keio_vs_bw25113,
            crosswalk_pairs=strains.crosswalk_pairs,
            crosswalk_numeric_disagreements_used=[
                use for use in strains.crosswalk if not use.numerics_agree
            ],
            crosswalk=strains.crosswalk,
        )
        with open(
            osp.join(self.preprocess_dir, "identifier_reconciliation.json"), "w"
        ) as handle:
            handle.write(identifier_report.model_dump_json(indent=2))
        log.info(
            "Tong2020: wrote %d records from %d strains (%s); dropped %d fragment, %d "
            "not-in-MG1655, %d ambiguous, %d no-ECK-partner strains",
            idx,
            len(strains.kept),
            identifier_report.collections,
            len(strains.dropped_fragment),
            len(strains.dropped_not_in_mg1655),
            len(strains.dropped_ambiguous),
            len(strains.dropped_no_eck_partner),
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built tree, per background
# --------------------------------------------------------------------------- #
#: The frozen record-count oracle per background, from the drop accounting of the
#: release: 3,644 Keio strains and 70 library strains, 30 carbon sources each.
EXPECTED_RECORDS: dict[EcoliK12StrainName, int] = {"BW25113": 109320, "MG1655": 2100}


def verify_build(
    dataset_root: str, data_root: str | None = None
) -> dict[str, VerificationReport]:
    """Run the fitness verifier on a built tree, once per pinned background.

    Each background is checked against its own genome: the resolver and the L4 gene
    universe (every GenBank locus, pseudogenes included) of the strain its records pin.
    One resolver cannot serve both, because a gene symbol resolves in both annotations.
    Each report is written to ``preprocess/verification_report_<strain>.json``.
    """
    from torchcell.verification.fitness import verify_fitness_dataset
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    by_strain: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        strain = record["reference"]["genome_reference"]["strain"]
        by_strain.setdefault(str(strain), []).append(record)
    reports: dict[str, VerificationReport] = {}
    for strain_name, expected in EXPECTED_RECORDS.items():
        genome = bacterial_genome("ecoli", strain_name, data_root)
        report = verify_fitness_dataset(
            by_strain.get(strain_name, []),
            dataset_name=f"CarbonSourceTong2020Dataset[{strain_name}]",
            provenance=Provenance(
                source_uri=XLSX_URL,
                citation_key=CITATION_KEY,
                sha256=XLSX_SHA256,
                method=(
                    "Table S1A normalized 24 h end-point colony integrated density "
                    "(last minus first time point, divided by the row then column "
                    "interquartile mean); a typical colony is 1.0"
                ),
                page="Table S1, sheet 'S1A - Endpoint Biomass'",
                retrieved=XLSX_RETRIEVED_AT,
            ),
            expected_count=expected,
            resolve_gene_name=genome.resolve_gene_name,
            sgd_genes=set(genome.genbank.loci),
        )
        out = osp.join(
            dataset_root, "preprocess", f"verification_report_{strain_name}.json"
        )
        with open(out, "w") as handle:
            handle.write(report.model_dump_json(indent=2))
        reports[strain_name] = report
    return reports


def main() -> None:
    """Build the dataset and verify it, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    root = osp.join(data_root, "data/torchcell/ecoli_carbon_source_tong2020")
    dataset = CarbonSourceTong2020Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])
    print(
        json.dumps(
            json.loads(Path(root, "preprocess", "dropped_records.json").read_text())[
                "rules"
            ],
            indent=2,
        )[:3000]
    )
    for strain, report in verify_build(root, data_root).items():
        print(strain, report.summary())


if __name__ == "__main__":
    main()
