# torchcell/datasets/ecoli/wang2015
# [[torchcell.datasets.ecoli.wang2015]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/wang2015
# Test file: tests/torchcell/datasets/ecoli/test_wang2015.py
"""Wang 2015 isoprenol tolerance of E. coli Keio multidrug-transporter deletions.

Wang, Yang, Shah, Choi and Kim 2015 (Sci Rep 5:16505, doi:10.1038/srep16505) grew the
parent BW25113 and 46 of its Keio single-gene deletions (45 multidrug-transporter genes
plus the outer-membrane channel ``tolC``) in 2YT with and without 0.5% (v/v) isoprenol
(3-methyl-3-buten-1-ol) at 30 C and read the OD600 after 12 h. Supplementary Table S3
releases, per strain, the mean and SD of that OD600 in each condition, over two
biological replicates. It is the only per-strain table of the tolerance screen.

RECORD = one Keio deletion strain under 0.5% (v/v) isoprenol, a
``BacterialEnvironmentResponseExperiment``:

- GENOTYPE: one ``BacterialDeletionPerturbation`` on the BW25113 locus-tag namespace,
  collection "Keio collection" (Wang's wording), cassette from Baba 2006.
- ENVIRONMENT: ``YT_2X`` at 30 C, shaken, 12 h, plus two edits: the medium's stated pH
  7.0 (``PhysicalFactor.ph``; the adjusting acid or base is a typed gap) and isoprenol at
  0.5 ``percent_v/v`` (``SmallMoleculePerturbation``).
- PHENOTYPE: ``EnvironmentResponsePhenotype``, ``log2_ratio`` by ``liquid_od_growth``: the
  log2 of the paper's own "relative tolerance capacity" of the strain to BW25113,
  ``log2[(OD_iso / OD_0)_strain / (OD_iso / OD_0)_BW25113]`` from the Table S3 means,
  where ``1 - OD_iso / OD_0`` is the paper's growth inhibition. Negative = more
  susceptible than the parent. The reference is BW25113 in the same environment, whose
  value is log2(1) = 0. ``n_samples = 2`` biological replicates.

WHY ``EnvironmentResponsePhenotype`` AND NOT ``FitnessPhenotype``. The readout is each
strain's growth under isoprenol normalized by its OWN isoprenol-free growth, then compared
with the parent's: a response to an environmental edit relative to a control, which is
what this class models and what the plan's section 3c assigns to the tolerance rows. The
number is signed (25 of the 46 stored values are below 0), and ``FitnessPhenotype`` is a
growth ratio within ONE environment that clamps non-positive values.

DATA. The SI PDF (``srep16505-s1.pdf``, PMC Open Access bucket, ``pmc_cloud``; the
literature mirror's ``si/si1.pdf``) is the one consumed file. Tables S2 (strain, Keio
number, MDT family) and S3 are read from its born-digital text layer with
``pdftotext -layout -enc UTF-8`` (poppler; the version is recorded in the raw mirror's
manifest). Extraction is checked at build time: every ``BWΔ`` label of each table parses
to a row, Table S3 holds exactly one BW25113 row and 46 deletion rows, its 46 names equal
Table S2's 46 Keio entries, the parsed values hash to ``TABLE_S3_SHA256``, and the table
reproduces the paper's own counts (17 strains above 52.5% growth inhibition, 7 above
57.5%, 11 multidrug-transporter strains below 47.5%; 45 non-OMP Keio entries in S2).

STRAIN (``SOURCED_VALUES["background_strain"]``). The paper names BW25113 and says its
deletion mutants came from the Keio collection; Baba 2006 (mirrored) is the Keio
construction paper. Records pin ``assembly_reference("BW25113")`` and the
``ecoli_k12_bw25113_locus_tag`` namespace.

IDENTIFIERS. Table S3 labels a strain only by gene name (``BWΔacrA``); Table S2 adds a
Keio number. On the BW25113 GenBank annotation, which carries the Keio JW numbers as
``gene_synonym``, 41 of the 46 Table S2 numbers name the locus their strain's gene name
names and 5 do not: ``acrA``/``acrB`` and ``emrK``/``emrY`` are swapped pairs (acrB's is
printed ``JJW0452``, not a Keio number at all; its well-formed reading JW0452 is acrA's),
and ``mdtD``'s ``JW2077`` is the number of ``gatB`` (it repeats the digits of mdtD's
Blattner number b2077). The paper's own Table S4
measured an ``acrB`` transcript in the strain it calls BWΔacrA and an ``acrA`` transcript
in BWΔacrB (``SOURCED_VALUES["identity_evidence_*"]``), so for those two the strain is
not the deletion its Table S2 number names. The record's locus is therefore the one the
strain NAME resolves to, through ``reconcile_locus_tags`` on BW25113; the Table S2 number
is stored as ``construction.strain_accession`` only when it names that same locus, and
every disagreement is written to ``preprocess/identifier_reconciliation.json`` with its
evidence.

RECORDS DROPPED (rule + items in ``preprocess/dropped_records.json``): a strain name that
does not resolve to exactly one BW25113 locus (``strain_name_not_on_a_bw25113_locus``;
none on the pinned table). The BW25113 row is the reference, not a record.

UNCERTAINTY. Table S3 gives a sample SD for each of the four OD600 means a record is
computed from, not for the ratio itself, and the per-replicate ODs are not released; the
ratio's SD therefore cannot be derived without assuming the four cultures independent.
``environment_response_uncertainty`` and ``environment_response_se`` are typed gaps.

COMPOUND. ``isoprenol`` has no row in ``compound_identity_table.json`` yet, so
``resolved_compound`` returns the name with an ``inchikey`` gap (the same object the
Carruthers 2025 loader stores). Its identity, ``ISOPRENOL_INCHIKEY``, is recorded here and
checked against any row the table gains; adding that row is a separate change.

NOT LOADED, because the paper releases them only as figures or as another readout:
the Fig. 2 time courses; the transporter overexpression strains on pTrc99A (Fig. 3C,
Fig. S3); the BWΔacrAB double and BWΔABC triple knockouts and the 0.75% isoprenol arm
(Fig. 4); the other alcohols (Fig. 5); pT-tolC complementation (Fig. S5); and Table S4, a
tabulated RT-qPCR panel of 9 transporter transcripts for which no phenotype class exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import os.path as osp
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

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
from torchcell.datamodels.media import YT_2X
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    AssayType,
    AssemblyReferenceGenome,
    BacterialDeletionPerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Compound,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    PhysicalFactor,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainConstruction,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    LOCUS_TAG_PATTERNS,
    STRAIN_GENE_NAMESPACES,
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
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.provenance import run_retriever
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "wangDynamicInterplayMultidrug2015"
PAPER_DOI = "10.1038/srep16505"
PAPER_TITLE = (
    "Dynamic interplay of multidrug transporters with TolC for isoprenol tolerance in "
    "Escherichia coli"
)
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"
DATASET_ROOT_REL = "data/torchcell/env_chemgen_wang2015"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "9cd90f00599871689fa8e62849b4220fa0801a502160b575bca4d04067fc3cd2"
#: MinerU OCR of the SI PDF; the anchor of every SI quote (the tables are parsed from
#: the PDF's text layer instead, which is born-digital and lossless).
SI1_MD = "si/si1.md"
SI1_MD_SHA256 = "d1c4aeeb6edcaf88a715838cc1cca10e995e061af1801cc8e74eba3d0055272a"
#: Baba 2006, the Keio construction paper (Wang's ref 27).
BABA_KEY = "babaConstructionEscherichiaColi2006"
BABA_SHA256 = "ca71475baf25d070562c70296d5aad8ec17a59d5b843f264b1e226362b6daf0d"

SI_PDF = "si1.pdf"
SI_PDF_SHA256 = "ed2a029aef1d7d45aa9aa853162ba7960132b7ebc674a9c92fbc618aae605952"
SI_PDF_BYTES = 1098793
#: The PMC Open Access bucket key the literature mirror captured the SI from; its
#: ``RetrievalRecord`` below is copied from that key's ``manifest.json``.
_SI_PDF_KEY = "PMC4643228.1/srep16505-s1.pdf"
_SI_PDF_RETRIEVED_AT = "2026-10-07T11:43:31.857547+00:00"
SI_SOURCE_URL = f"https://pmc-oa-opendata.s3.amazonaws.com/{_SI_PDF_KEY}"
#: ``pdftotext`` arguments of the extraction (``-`` writes the text to stdout).
PDFTOTEXT_ARGS: tuple[str, ...] = ("-layout", "-enc", "UTF-8")


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


RAW_FILES: tuple[RawFile, ...] = (
    RawFile(
        name=SI_PDF,
        sha256=SI_PDF_SHA256,
        bytes=SI_PDF_BYTES,
        description="Supplementary Information PDF (srep16505-s1.pdf): Table S2 strains "
        "with Keio numbers and MDT families, Table S3 OD600 means and SDs without and "
        "with 0.5% (v/v) isoprenol",
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=SI_SOURCE_URL,
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": _SI_PDF_KEY},
            sha256=SI_PDF_SHA256,
            retrieved_at=_SI_PDF_RETRIEVED_AT,
        ),
    ),
)
#: ``{raw file name: pinned sha256}``, the build-time check of every consumed file.
DATA_SHA256: dict[str, str] = {f.name: f.sha256 for f in RAW_FILES}
RAW_FILES_BY_NAME: dict[str, RawFile] = {f.name: f for f in RAW_FILES}

#: What the paper releases that the loader deliberately does not consume.
NOT_MIRRORED = (
    "paper.pdf: not consumed; every per-strain value is in the SI PDF and every quote is "
    "anchored in the literature mirror's paper.md",
    "Figures 2, 3C, 4, 5, S3 and S5 (time courses, overexpression strains, BWdeltaacrAB "
    "and BWdeltaABC, 0.75% isoprenol, other alcohols, pT-tolC): released as figures only",
    "Table S4 (RT-qPCR fold changes of 9 transcripts): tabulated in the same SI PDF but "
    "not loaded; no phenotype class models a targeted qPCR panel",
)


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the sha256-pinned Wang ``paper.md``."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
        ),
    )


def _si(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the SI OCR (``si/si1.md``)."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI1_MD,
            citation_key=CITATION_KEY,
            sha256=SI1_MD_SHA256,
            method="MinerU OCR of the SI PDF (torchcell-library mirror)",
        ),
    )


def _baba(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to Baba 2006, the Keio paper Wang cites for the collection."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=BABA_KEY,
            sha256=BABA_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced values (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
_FIG = "Fig.  1C"  # the OCR keeps the PDF's no-break space after "Fig."
_SCREEN_QUOTE = (
    "cell growth was evaluated after $1 2 \\mathrm { ~ h ~ }$ exposure to "
    "$0 . 5 \\%$ (v/v) of isoprenol in a total of 44 MDT null mutants "
    f"({_FIG} and Supplementary Table S3)"
)
_FIG1C_QUOTE = (
    "MDT mutants were grown in 2YT medium with $0 . 5 \\%$ (v/v) of isoprenol at "
    "$3 0 ^ { \\circ } \\mathrm { C }$ for $1 2 \\mathrm { { h } }$ ."
)
_MEDIUM_PH_QUOTE = "adjusted to $\\mathrm { p H } 7 . 0 $ ) with agitation."

SOURCED_VALUES: dict[str, SourcedValue] = {
    "background_strain": _paper(
        "BW25113",
        "E. coli BW25113 $( \\mathrm { F ^ { - } } , \\ \\mathsf { \\lambda } ^ { - } , "
        "\\mathsf { \\lambda } ^ { - }$ , rrnB-3, ΔlacZ4787",
        note="Methods, 'Strains and culture condition'; the genotype the sentence goes on "
        "to list is BW25113's",
    ),
    "collection": _paper(
        "Keio collection",
        "rph-1) and its isogenic deletion mutants were obtained from Keio collection "
        "(National Institute of Genetics, Shizuoka, Japan) for screening of transporter "
        "candidates associated with isoprenol tolerance.",
        note="the stored collection string is this sentence's own wording",
    ),
    "keio_background": _baba(
        "BW25113",
        "The Keio collection is comprised of 3985 deletions in duplicate (7970 total) of "
        "E. coli K-12 strain BW25113 (Datsenko and Wanner, 2000)",
        note="corroborates the background: the collection Wang obtained its mutants from "
        "is built in BW25113",
    ),
    "cassette": _baba(
        "kanamycin cassette flanked by FLP recognition target sites",
        "Open-reading frame coding regions were replaced with a kanamycin cassette "
        "flanked by FLP recognition target sites",
        note="Wang cures a cassette only in BWdeltatolC before building BWdeltaABC (not "
        "loaded), so the screened collection strains are recorded with the cassette in "
        "place; the same wording as the Fuhrer 2017 loader",
    ),
    "data_location": _paper(
        "Supplementary Table S3",
        _SCREEN_QUOTE,
        note="the table the loader reads. Table S3 and Fig. 1C both carry 46 deletion "
        "strains (45 transporter genes plus tolC); the '44' of this sentence is an "
        "internal inconsistency of the paper, recorded and not resolved",
    ),
    "isoprenol_dose_percent_v_v": _paper(
        0.5,
        _SCREEN_QUOTE,
        note="the dose of every Table S3 measurement (column header '0.5% (v/v) "
        "isoprenol')",
    ),
    "duration_hours": _paper(12.0, _SCREEN_QUOTE),
    "medium": _paper(
        "YT_2X",
        _FIG1C_QUOTE,
        note="Fig. 1C caption, the figure Table S3 tabulates; the 2YT recipe is the "
        "MEDIA_LIBRARY key YT_2X, quoted from the same paper",
    ),
    "temperature_c": _paper(
        30.0,
        "Cultivations were carried out at $3 0 ^ { \\circ } \\mathrm { C }$",
        note="Methods, 'Determination of growth inhibition'; the Fig. 1C caption agrees",
    ),
    "medium_ph": _paper(
        7.0,
        _MEDIUM_PH_QUOTE,
        note="the 2YT recipe sentence; the acid or base used is not named (a typed gap on "
        "the pH perturbation's agent)",
    ),
    "aerobicity": _paper(
        "aerobic",
        _MEDIUM_PH_QUOTE,
        note="a READING of 'with agitation' (4 mL in a 55 mL tube, shaken); the paper "
        "never names an oxygen regime",
    ),
    "culture_format": _paper(
        "OD600 0.1 inoculum, 4 mL in a 55 mL tube",
        "Overnight culture was inoculated at an optical density $\\mathrm { ( O D _ { 6 0 0 "
        "} ) }$ of 0.1 into a PYREX $\\textcircled { \\mathscr { B } } _ { 5 5 \\mathrm { m "
        "L } }$ rimless culture tube (Corning, NY) containing $4 \\mathrm { m L }$ of fresh "
        "media with (or without) isoprenol",
        note="NOT stored: the bacterial experiment pair declares environment: "
        "Environment, so a CultureEnvironment would serialize as its base class",
    ),
    "growth_inhibition_definition": _paper(
        "growth inhibition = 1 - OD600(with) / OD600(without)",
        "Growth inhibition $( \\% )$ is defined as $( 1 - \\mathrm { O D } _ { 6 0 0 }$ with "
        "$\\mathrm { C } _ { \\mathrm { m } } \\mathrm { O H s }$ / $\\mathrm { O D } _ { 6 0 0 "
        "}$ without $\\mathrm { C } _ { \\mathrm { m } } \\mathrm { O H s } ,$ b $\\times "
        "1 0 0$ .",
    ),
    "relative_tolerance_definition": _paper(
        "relative tolerance capacity of A to B = (1 - GI_A) / (1 - GI_B)",
        "and relative tolerance capacity $( \\% )$ by [ $( 1 -$ growth inhibition of "
        "Strain A)/ $1 -$ growth inhibition of Strain $\\mathbf { B } ) ] \\times 1 0 0$ .",
        note="the stored value is the log2 of this ratio with B = BW25113 (the paper "
        "multiplies it by 100); 1 - GI = OD600(with) / OD600(without)",
    ),
    "n_samples": _paper(
        2,
        "The growth inhibition of the wild type strain is at approximate $5 0 \\%$ as a "
        "reference. Results are the means of two biological replicates.",
        note="Fig. 1C caption, the figure Table S3 tabulates (the text cites them together, "
        "SOURCED_VALUES['data_location']); Table S3 itself states no n",
    ),
    "uncertainty_type": _si(
        "sample_sd of each OD600 column",
        "Note: The results are presented as means $\\pm$ standard divisions.",
        note="'standard divisions' is in the publisher's text layer too (a typo for "
        "standard deviations, SOURCED_VALUES['uncertainty_type_corroboration']). The SDs "
        "belong to the four OD600 columns, not to the stored ratio, so none is stored",
    ),
    "uncertainty_type_corroboration": _paper(
        "standard deviation",
        "Error bars represent the standard deviations of two biological replicates.",
        note="Figs. 3C, 4 and 5: the same lab's +/- over the same replicate design is an SD",
    ),
    "reported_counts": _paper(
        {"above_52.5": 17, "above_57.5": 7},
        "As a result, growth inhibition over $5 2 . 5 \\%$ was observed in 17 mutants, "
        "that is, they are $5 \\%$ more susceptible to isoprenol than wild type E. coli BW "
        "25113. In particular, seven null mutants of 6 MDTs (AcrD, EmrAB, MacAB, MdtBC, "
        f"MdtJI and YdiM) ({_FIG}) showed growth inhibition of more than $5 7 . 5 \\%$",
        note="process() recomputes both counts from the parsed Table S3 and refuses a "
        "mismatch",
    ),
    "reported_tolerant_count": _paper(
        11,
        "Interestingly, eleven MDT mutants exhibited phenotypes with higher tolerance to "
        "isoprenol than wild type E. coli BW 25113",
        note="read as growth inhibition below 47.5%, Fig. 1C's 'resistant' and 'very "
        "resistant' bins, among Table S2 entries whose family is not OMP (tolC); "
        "process() checks it",
    ),
    "table_s2_mdt_count": _paper(
        45,
        "a library of 45 null mutants (Supplementary Table S2) associated with 37 MDTs "
        "was thus screened.",
        note="Table S2 lists 46 Keio entries, 45 of an MDT family plus tolC (OMP); "
        "process() checks the 45",
    ),
    "isoprenol_identity": _paper(
        "3-methyl-3-buten-1-ol",
        "represent isoprenol $( 3 \\mathrm { M } - 3 = \\mathrm { C } _ { 4 } \\mathrm { O H } "
        ",$ ) molecules.",
        note="'3M-3=C4OH' is 3-methyl, double bond at C3, C4 chain, 1-ol: SMILES "
        "C=C(C)CCO, whose RDKit InChIKey is ISOPRENOL_INCHIKEY (PubChem CID 12988 gives "
        "the same key)",
    ),
    "identity_evidence_acrA": _si(
        "acrB is transcribed in BWdeltaacrA",
        "<tr><td>acrB</td><td>2.66 ± 0.36</td><td>1.97 ± 0.09</td>",
        note="Table S4: the second value is the acrB transcript in the strain the paper "
        "calls BWdeltaacrA (columns BWdeltaacrA, BWdeltaacrB, BWdeltatolC). Table S2's "
        "JW0451 for that strain is acrB's Keio number on the BW25113 annotation; a strain "
        "transcribing acrB is not the acrB deletion",
    ),
    "identity_evidence_acrB": _si(
        "acrA is transcribed in BWdeltaacrB",
        "<tr><td>acrA</td><td>3.16 ± 0.87</td><td></td><td>1.59 ± 0.08</td>",
        note="Table S4: acrA transcript in the strain the paper calls BWdeltaacrB. Table "
        "S2 gives that strain 'JJW0452', whose well-formed reading JW0452 is acrA's Keio "
        "number; a strain transcribing acrA is not the acrA deletion",
    ),
    "doi": _paper(PAPER_DOI, "doi: 10.1038/srep16505 (2015)."),
}

TEMPERATURE_C = SOURCED_VALUES["temperature_c"]
DURATION_HOURS = SOURCED_VALUES["duration_hours"]
ISOPRENOL_DOSE = SOURCED_VALUES["isoprenol_dose_percent_v_v"]
MEDIUM_PH = SOURCED_VALUES["medium_ph"]
AEROBICITY = SOURCED_VALUES["aerobicity"]
N_SAMPLES = SOURCED_VALUES["n_samples"]
COLLECTION: str = SOURCED_VALUES["collection"].value
CASSETTE: str = SOURCED_VALUES["cassette"].value
REPORTED_COUNTS: dict[str, int] = SOURCED_VALUES["reported_counts"].value
REPORTED_TOLERANT_COUNT: int = SOURCED_VALUES["reported_tolerant_count"].value
TABLE_S2_MDT_COUNT: int = SOURCED_VALUES["table_s2_mdt_count"].value

#: Which sourced evidence speaks to a Table S2 disagreement, by strain gene name.
IDENTITY_EVIDENCE: dict[str, tuple[str, ...]] = {
    "acrA": ("identity_evidence_acrA",),
    "acrB": ("identity_evidence_acrB",),
}

#: The isoprenol label handed to ``resolved_compound`` (the Carruthers 2025 loader's).
ISOPRENOL_LABEL = "isoprenol"
#: RDKit InChIKey of ``ISOPRENOL_SMILES``; PubChem CID 12988 (3-methyl-3-buten-1-ol)
#: returns the same key. Not written onto the record: the compound-identity table is the
#: one authority, and its isoprenol row is a separate change.
ISOPRENOL_INCHIKEY = "CPJRRXSHAYUTGL-UHFFFAOYSA-N"
ISOPRENOL_SMILES = "C=C(C)CCO"
ISOPRENOL_PUBCHEM_CID = 12988

MEASUREMENT_TYPE = MeasurementType.log2_ratio
ASSAY_TYPE = AssayType.liquid_od_growth
RESPONSE_UNITS = (
    "log2 relative tolerance capacity to BW25113: log2[(OD600 with 0.5% (v/v) isoprenol "
    "/ OD600 without isoprenol) of the strain / (the same ratio of BW25113)], Table S3 "
    "means after 12 h in 2YT at 30 C"
)
PARENT_STRAIN = "BW25113"
STRAIN_PREFIX = "BWΔ"
THIS_STUDY = "This study"
OMP_FAMILY = "OMP"
#: Table S3's deletion rows. Fig. 1C draws the same 46 (8 ABC, 11 RND, 4 SMR, 21 MFS,
#: 1 MATE, 1 OMP); the text's "44" is recorded in SOURCED_VALUES['data_location'].
TABLE_S3_MUTANTS = 46
#: sha256 of the parsed Table S3 (``table_s3_digest``), measured on the pinned SI PDF
#: with poppler 21.01.0; any extraction drift changes it and stops the build.
TABLE_S3_SHA256 = "ce6d952c1058695fe5912353de563faa0224ad5ce8c16c5a8812e62da3b41093"
#: Checklist item 4: below this fraction of strain names resolving to BW25113 locus tags
#: the build stops. All 46 resolve on the pinned annotation, so a lower fraction (more
#: than 2 of 46 unresolved) would mean a parse error or the wrong genome.
MIN_RESOLVED_FRACTION = 0.95
#: Fig. 1C's bin edges (percent growth inhibition), used by the count checks.
SUSCEPTIBLE_EDGE = 52.5
VERY_SUSCEPTIBLE_EDGE = 57.5
RESISTANT_EDGE = 47.5

_LOOKED_IN = Provenance(
    source_uri=SI1_MD,
    citation_key=CITATION_KEY,
    sha256=SI1_MD_SHA256,
    method="Table S3 and its note, plus the full Methods and every figure caption of "
    f"{PAPER_MD} (sha256 {PAPER_MD_SHA256})",
    page="Supplementary Table S3",
)
_UNCERTAINTY_NOTE = (
    "Table S3 reports a sample SD for each of the four OD600 means the stored ratio is "
    "computed from, not for the ratio; the per-replicate OD600 values are not released, "
    "so the ratio's SD is not derivable without assuming the four cultures independent"
)
UNCERTAINTY_GAPS: tuple[ProvenanceGap, ...] = (
    ProvenanceGap(
        field="environment_response_uncertainty",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_LOOKED_IN,
        note=_UNCERTAINTY_NOTE,
    ),
    ProvenanceGap(
        field="environment_response_se",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_LOOKED_IN,
        note=_UNCERTAINTY_NOTE,
    ),
)
PH_AGENT_GAP = ProvenanceGap(
    field="agent",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=MEDIUM_PH.provenance,
    note="the 2YT recipe states 'adjusted to pH 7.0' and names no acid or base",
)
SOLVENT_GAP = ProvenanceGap(
    field="solvent",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=ISOPRENOL_DOSE.provenance,
    note="the paper doses isoprenol as % (v/v) of the medium and names no vehicle; "
    "added neat is the likely reading but is not stated",
)

PUBLICATION = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")


# --------------------------------------------------------------------------- #
# Raw mirror (the loader reads the mirror, never a live URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/wangDynamicInterplayMultidrug2015``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/wangDynamicInterplayMultidrug2015``."""
    return Path(data_root or _data_root()) / LIBRARY_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdftotext_version() -> str:
    """The ``pdftotext`` version line, recorded as the extraction's provenance."""
    result = subprocess.run(
        ["pdftotext", "-v"], capture_output=True, text=True, check=True
    )
    return (result.stderr or result.stdout).strip().split("\n")[0]


def extraction_record(input_sha256: str) -> ProcessingRecord:
    """How the loader turns the SI PDF into Tables S2 and S3 (tool, version, args)."""
    return ProcessingRecord(
        processor="torchcell.datasets.ecoli.wang2015.layout_text",
        tool="pdftotext",
        version=pdftotext_version(),
        params={"args": list(PDFTOTEXT_ARGS)},
        input_sha256=[input_sha256],
    )


def retrieve_raw_files(dest_dir: str | Path) -> dict[str, Path]:
    """Run each file's recorded retriever and write the verified bytes to ``dest_dir``.

    The recorded ``RetrievalRecord`` is what runs (``run_retriever``), so this is the
    re-runnable retrieval itself; a byte mismatch raises before anything is written.
    """
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

    ``sources`` maps every name in ``RAW_FILES`` to a local file. Idempotent by sha256:
    a mirror file with the pinned hash is left alone, and one with any other hash raises
    rather than being overwritten. The SI PDF's record carries its retrieval and the
    ``pdftotext`` extraction step.
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
                processing=extraction_record(raw.sha256),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=records,
        si_data_sources=[SI_SOURCE_URL],
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
# Parsing the SI tables
# --------------------------------------------------------------------------- #
TABLE_S2_MARKER = "Table S2."
TABLE_S3_MARKER = "Table S3."
TABLE_S4_MARKER = "Table S4."

#: One Table S2 entry: strain, Keio number (verbatim, so a malformed one survives to the
#: cross-check) or "This study", and MDT family. Each layout line carries two entries.
_S2_ENTRY = re.compile(r"BWΔ(\w+)\s+(This study|\S+)\s+(\S+)")
#: One Table S3 row: strain, then mean +/- SD without and with isoprenol.
_S3_ROW = re.compile(
    r"(BW25113|BWΔ\w+)\s+(\d+\.\d+)\s+±\s+(\d+\.\d+)\s+"
    r"(\d+\.\d+)\s+±\s+(\d+\.\d+)"
)


class TableExtractionError(ValueError):
    """The SI text layer does not parse into the tables as expected."""


class StrainSetMismatchError(ValueError):
    """Table S3's deletion strains are not exactly Table S2's Keio entries."""


class ReportedCountMismatchError(ValueError):
    """A count recomputed from Table S3 differs from the count the paper states."""


class CompoundIdentityConflictError(ValueError):
    """The compound-identity table resolves isoprenol to another InChIKey."""


class TableS2Entry(BaseModel):
    """One Table S2 row: a strain, its Keio number as printed, and its MDT family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gene_name: str
    keio_token: str
    mdt_family: str


class TableS3Row(BaseModel):
    """One Table S3 row: OD600 mean and SD without and with 0.5% (v/v) isoprenol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strain: str
    od600_without: float
    sd_without: float
    od600_with: float
    sd_with: float

    @property
    def gene_name(self) -> str:
        """The deleted gene of a ``BWΔ`` label (refused for the parent)."""
        if not self.strain.startswith(STRAIN_PREFIX):
            raise ValueError(f"{self.strain!r} is not a deletion strain label")
        return self.strain[len(STRAIN_PREFIX) :]

    @property
    def growth_ratio(self) -> float:
        """OD600 with isoprenol over OD600 without (1 minus the growth inhibition)."""
        return self.od600_with / self.od600_without

    @property
    def growth_inhibition_pct(self) -> float:
        """The paper's growth inhibition, in percent."""
        return (1.0 - self.growth_ratio) * 100.0


def layout_text(pdf_path: str | Path) -> str:
    """The SI PDF's text layer, as ``pdftotext -layout -enc UTF-8`` writes it."""
    result = subprocess.run(
        ["pdftotext", *PDFTOTEXT_ARGS, str(pdf_path), "-"],
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("utf-8")


def table_section(text: str, start: str, end: str) -> str:
    """The text between two table captions, each required exactly once and in order."""
    for marker in (start, end):
        found = text.count(marker)
        if found != 1:
            raise TableExtractionError(
                f"{marker!r} occurs {found} times in the SI text (expected 1)"
            )
    begin, stop = text.index(start), text.index(end)
    if stop <= begin:
        raise TableExtractionError(f"{end!r} precedes {start!r} in the SI text")
    return text[begin:stop]


def parse_table_s2(section: str) -> list[TableS2Entry]:
    """Every Table S2 entry, refusing a ``BWΔ`` label that does not parse or repeats."""
    entries = [
        TableS2Entry(gene_name=gene, keio_token=token, mdt_family=family)
        for gene, token, family in _S2_ENTRY.findall(section)
    ]
    labels = section.count(STRAIN_PREFIX)
    if len(entries) != labels:
        raise TableExtractionError(
            f"Table S2: {labels} strain labels but {len(entries)} parsed entries"
        )
    names = [e.gene_name for e in entries]
    if len(set(names)) != len(names):
        raise TableExtractionError("Table S2 repeats a strain")
    return entries


def parse_table_s3(section: str) -> list[TableS3Row]:
    """Every Table S3 row, refusing a strain label that does not parse."""
    rows = [
        TableS3Row(
            strain=strain,
            od600_without=float(od0),
            sd_without=float(sd0),
            od600_with=float(od1),
            sd_with=float(sd1),
        )
        for strain, od0, sd0, od1, sd1 in _S3_ROW.findall(section)
    ]
    labels = section.count(PARENT_STRAIN) + section.count(STRAIN_PREFIX)
    if len(rows) != labels:
        raise TableExtractionError(
            f"Table S3: {labels} strain labels but {len(rows)} parsed rows"
        )
    return rows


def table_s3_digest(rows: Sequence[TableS3Row]) -> str:
    """sha256 of the parsed rows, sorted by strain (independent of the layout order)."""
    payload = json.dumps(
        sorted((row.model_dump() for row in rows), key=lambda r: r["strain"]),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_parent(rows: Sequence[TableS3Row]) -> tuple[TableS3Row, list[TableS3Row]]:
    """The one BW25113 row and the deletion rows, sorted by gene name."""
    parents = [row for row in rows if row.strain == PARENT_STRAIN]
    if len(parents) != 1:
        raise TableExtractionError(f"Table S3 has {len(parents)} BW25113 rows")
    mutants = sorted(
        (row for row in rows if row.strain != PARENT_STRAIN),
        key=lambda row: row.gene_name,
    )
    names = [row.gene_name for row in mutants]
    if len(set(names)) != len(names):
        raise TableExtractionError("Table S3 repeats a strain")
    if len(mutants) != TABLE_S3_MUTANTS:
        raise TableExtractionError(
            f"Table S3 has {len(mutants)} deletion rows, expected {TABLE_S3_MUTANTS}"
        )
    return parents[0], mutants


def keio_entries_for(
    mutants: Sequence[TableS3Row], entries: Sequence[TableS2Entry]
) -> dict[str, TableS2Entry]:
    """Table S2's Keio entries by gene name, required to be exactly Table S3's strains."""
    keio = {e.gene_name: e for e in entries if e.keio_token != THIS_STUDY}
    measured = {row.gene_name for row in mutants}
    if measured != set(keio):
        raise StrainSetMismatchError(
            f"Table S3 only: {sorted(measured - set(keio))}; Table S2 Keio only: "
            f"{sorted(set(keio) - measured)}"
        )
    return keio


class CountCheck(BaseModel):
    """One count the paper states, recomputed from the parsed tables."""

    name: str
    stated: int
    measured: int


def reported_count_checks(
    mutants: Sequence[TableS3Row], keio: Mapping[str, TableS2Entry]
) -> list[CountCheck]:
    """Recompute the paper's four stated counts and refuse any mismatch."""
    inhibition = {row.gene_name: row.growth_inhibition_pct for row in mutants}
    mdt = {name for name, e in keio.items() if e.mdt_family != OMP_FAMILY}
    checks = [
        CountCheck(
            name=f"growth inhibition above {SUSCEPTIBLE_EDGE}%",
            stated=REPORTED_COUNTS["above_52.5"],
            measured=sum(v > SUSCEPTIBLE_EDGE for v in inhibition.values()),
        ),
        CountCheck(
            name=f"growth inhibition above {VERY_SUSCEPTIBLE_EDGE}%",
            stated=REPORTED_COUNTS["above_57.5"],
            measured=sum(v > VERY_SUSCEPTIBLE_EDGE for v in inhibition.values()),
        ),
        CountCheck(
            name=f"MDT strains with growth inhibition below {RESISTANT_EDGE}%",
            stated=REPORTED_TOLERANT_COUNT,
            measured=sum(inhibition[n] < RESISTANT_EDGE for n in mdt),
        ),
        CountCheck(
            name="Table S2 Keio entries of an MDT family",
            stated=TABLE_S2_MDT_COUNT,
            measured=len(mdt),
        ),
    ]
    bad = [c for c in checks if c.stated != c.measured]
    if bad:
        raise ReportedCountMismatchError(
            "; ".join(
                f"{c.name}: stated {c.stated}, measured {c.measured}" for c in bad
            )
        )
    return checks


# --------------------------------------------------------------------------- #
# Identifiers and the retention ledger
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the strains it removed, and why."""

    rule: str
    description: str
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    table_rows: int
    reference_rows: list[str]
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]


class StrainIdentity(BaseModel):
    """A Table S3 strain on its BW25113 locus, with Table S2's number when it agrees."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row: TableS3Row
    locus_tag: str
    symbol: str
    keio_token: str
    mdt_family: str
    keio_accession: str | None


class KeioDisagreement(BaseModel):
    """A strain whose Table S2 number does not name the locus its gene name names."""

    gene_name: str
    name_locus: str
    keio_token: str
    keio_resolution: str
    keio_locus_symbol: str | None
    evidence: list[str]


class IdentifierLedger(BaseModel):
    """Both reconciliations (names, Table S2 numbers) and where they disagree."""

    names: LocusTagReconciliation
    keio_numbers: LocusTagReconciliation
    min_resolved_fraction: float
    n_keio_agree: int
    keio_disagreements: list[KeioDisagreement]


def canonical_symbol(genome: EcoliK12Genome, tag: str) -> str:
    """The annotation's own gene symbol of ``tag`` when it resolves back to ``tag``,
    else the tag itself, so the stored pair always round-trips through the resolver.
    """
    symbol = genome.genbank.loci[tag].symbol
    if symbol is None:
        return tag
    resolution = genome.resolve_gene_name(symbol)
    return symbol if resolution.systematic_name == tag else tag


def resolve_strains(
    genome: EcoliK12Genome,
    mutants: Sequence[TableS3Row],
    keio: Mapping[str, TableS2Entry],
    *,
    label: str,
) -> tuple[list[StrainIdentity], DropRule, IdentifierLedger]:
    """Place each strain on BW25113 by its gene name and cross-check Table S2's number.

    Stops (``LocusTagResolutionError``) below ``MIN_RESOLVED_FRACTION``. A name the
    reconciler keeps as given (retired, ambiguous, or colliding with another name) is not
    a locus tag the deletion leaf accepts; its record is dropped with the reason. A Table
    S2 number that names another locus (or none) does not move the record: it is ledgered
    and not stored.
    """
    names = pd.Series([row.gene_name for row in mutants], dtype=object)
    stored, name_report = reconcile_locus_tags(
        genome, names, label=f"{label} Table S3 strain names"
    )
    name_report.require_resolved(MIN_RESOLVED_FRACTION)
    tokens = pd.Series(
        [keio[row.gene_name].keio_token for row in mutants], dtype=object
    )
    stored_tokens, keio_report = reconcile_locus_tags(
        genome, tokens, label=f"{label} Table S2 Keio numbers"
    )
    pattern = LOCUS_TAG_PATTERNS[name_report.gene_namespace]
    kept: list[StrainIdentity] = []
    dropped: list[str] = []
    disagreements: list[KeioDisagreement] = []
    for row, tag, token_tag in zip(
        mutants, stored.tolist(), stored_tokens.tolist(), strict=True
    ):
        entry = keio[row.gene_name]
        if pattern.match(tag) is None:
            resolution = genome.resolve_gene_name(row.gene_name)
            reason = (
                "kept as given on collision"
                if row.gene_name in name_report.kept_on_collision
                else resolution.status.value
            )
            dropped.append(
                f"{row.strain} (Table S2 {entry.keio_token}): {reason} "
                f"{resolution.systematic_name}"
            )
            continue
        agrees = token_tag == tag
        if not agrees:
            token_resolution = genome.resolve_gene_name(entry.keio_token)
            disagreements.append(
                KeioDisagreement(
                    gene_name=row.gene_name,
                    name_locus=tag,
                    keio_token=entry.keio_token,
                    keio_resolution=f"{token_resolution.status.value} "
                    f"{token_resolution.systematic_name}",
                    keio_locus_symbol=(
                        genome.genbank.loci[token_tag].symbol
                        if pattern.match(token_tag)
                        else None
                    ),
                    evidence=list(IDENTITY_EVIDENCE.get(row.gene_name, ())),
                )
            )
        kept.append(
            StrainIdentity(
                row=row,
                locus_tag=tag,
                symbol=canonical_symbol(genome, tag),
                keio_token=entry.keio_token,
                mdt_family=entry.mdt_family,
                keio_accession=entry.keio_token if agrees else None,
            )
        )
    rule = DropRule(
        rule="strain_name_not_on_a_bw25113_locus",
        description="the Table S3 strain name is not a symbol or synonym of exactly one "
        "BW25113 GenBank locus (retired, ambiguous, or shared with another strain "
        "name), so no locus tag of the pinned namespace names its deletion",
        n_records=len(dropped),
        items=dropped,
    )
    ledger = IdentifierLedger(
        names=name_report,
        keio_numbers=keio_report,
        min_resolved_fraction=MIN_RESOLVED_FRACTION,
        n_keio_agree=len(kept) - len(disagreements),
        keio_disagreements=disagreements,
    )
    return kept, rule, ledger


# --------------------------------------------------------------------------- #
# Record construction (pure, no files)
# --------------------------------------------------------------------------- #
def log2_relative_tolerance(row: TableS3Row, parent: TableS3Row) -> float:
    """log2 of the paper's relative tolerance capacity of ``row`` to ``parent``."""
    return math.log2(row.growth_ratio / parent.growth_ratio)


def response_phenotype(value: float) -> EnvironmentResponsePhenotype:
    """One strain's log2 relative tolerance; its uncertainty is a typed gap."""
    return EnvironmentResponsePhenotype(
        measurement_type=MEASUREMENT_TYPE,
        assay_type=ASSAY_TYPE,
        environment_response=value,
        n_samples=int(N_SAMPLES.value),
        sample_unit=SampleUnit.biological_replicate,
        units=RESPONSE_UNITS,
        provenance_gaps=list(UNCERTAINTY_GAPS),
    )


def isoprenol_compound() -> Compound:
    """Isoprenol through the shared compound-identity layer.

    With no table row the resolver returns the name with an ``inchikey`` gap; once a row
    exists, its InChIKey must be ``ISOPRENOL_INCHIKEY`` or the build stops.
    """
    compound = resolved_compound(ISOPRENOL_LABEL)
    if compound.inchikey is not None and compound.inchikey != ISOPRENOL_INCHIKEY:
        raise CompoundIdentityConflictError(
            f"the compound-identity table resolves {ISOPRENOL_LABEL!r} to "
            f"{compound.inchikey}, not {ISOPRENOL_INCHIKEY}"
        )
    return compound


def environment() -> Environment:
    """2YT at pH 7.0 and 30 C, shaken for 12 h, with 0.5% (v/v) isoprenol."""
    return Environment(
        media=YT_2X,
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        perturbations=[
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.ph,
                magnitude=Concentration(
                    value=float(MEDIUM_PH.value), unit=ConcentrationUnit.ph
                ),
                provenance_gaps=[PH_AGENT_GAP],
            ),
            SmallMoleculePerturbation(
                compound=isoprenol_compound(),
                concentration=Concentration(
                    value=float(ISOPRENOL_DOSE.value),
                    unit=ConcentrationUnit.percent_v_v,
                ),
                provenance_gaps=[SOLVENT_GAP],
            ),
        ],
        aerobicity=str(AEROBICITY.value),
        duration_hours=float(DURATION_HOURS.value),
    )


def deletion_genotype(identity: StrainIdentity) -> Genotype:
    """The one Keio deletion, on its BW25113 locus tag."""
    construction = (
        StrainConstruction(strain_accession=identity.keio_accession)
        if identity.keio_accession is not None
        else None
    )
    return Genotype(
        perturbations=[
            BacterialDeletionPerturbation(
                systematic_gene_name=identity.locus_tag,
                perturbed_gene_name=identity.symbol,
                gene_namespace=STRAIN_GENE_NAMESPACES["BW25113"],
                collection=COLLECTION,
                cassette=CASSETTE,
                construction=construction,
            )
        ]
    )


def build_experiment(
    dataset_name: str, identity: StrainIdentity, parent: TableS3Row, env: Environment
) -> BacterialEnvironmentResponseExperiment:
    """The record of one Keio strain under isoprenol."""
    return BacterialEnvironmentResponseExperiment(
        dataset_name=dataset_name,
        genotype=deletion_genotype(identity),
        environment=env,
        phenotype=response_phenotype(log2_relative_tolerance(identity.row, parent)),
    )


def build_reference(
    dataset_name: str, genome_reference: AssemblyReferenceGenome, env: Environment
) -> BacterialEnvironmentResponseExperimentReference:
    """BW25113 in the same environment: relative tolerance to itself, log2(1) = 0."""
    return BacterialEnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=env.model_copy(),
        phenotype_reference=response_phenotype(0.0),
    )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class EnvChemgenWang2015Dataset(ExperimentDataset):
    """Isoprenol tolerance of 46 Keio transporter deletions (Wang 2015, Table S3)."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``ecoli_genome`` is injected by the build entry points."""
        self.ecoli_genome = ecoli_genome
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
        """Link each mirror file into ``raw/`` after checking the manifest and sha256.

        The mirror plus ``DATA_SHA256`` is canonical; the PMC bucket URL is retrieval
        metadata that ``retrieve_raw_files`` re-runs, never a build input.
        """
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
        log.info("Wang 2015 raw files linked into %s (sha256 verified)", self.raw_dir)

    def _raw(self, name: str) -> str:
        return osp.join(self.raw_dir, name)

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache (a direct run);
        a genome of another assembly set is refused.
        """
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        expected = BACTERIAL_ASSEMBLY_SETS[self.REFERENCE_STRAIN]
        if self.ecoli_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.ecoli_genome.ASSEMBLY_SET}"
            )
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Parse Tables S2 and S3 of the SI PDF into per-strain records + LMDB."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        text = layout_text(self._raw(SI_PDF))
        entries = parse_table_s2(table_section(text, TABLE_S2_MARKER, TABLE_S3_MARKER))
        rows = parse_table_s3(table_section(text, TABLE_S3_MARKER, TABLE_S4_MARKER))
        digest = table_s3_digest(rows)
        if digest != TABLE_S3_SHA256:
            raise TableExtractionError(
                f"parsed Table S3 sha256 {digest}, pinned {TABLE_S3_SHA256}"
            )
        parent, mutants = split_parent(rows)
        keio = keio_entries_for(mutants, entries)
        checks = reported_count_checks(mutants, keio)

        genome = self._genome()
        resolved, unresolved, ledger = resolve_strains(
            genome, mutants, keio, label=self.name
        )
        drop_log = DropLog(
            dataset=self.name,
            table_rows=len(rows),
            reference_rows=[PARENT_STRAIN],
            source_records=len(mutants),
            kept_records=len(resolved),
            dropped_records=len(mutants) - len(resolved),
            rules=[unresolved],
        )
        if sum(r.n_records for r in drop_log.rules) != drop_log.dropped_records:
            raise RuntimeError("drop rules do not account for every dropped strain")
        log.info(
            "Wang 2015: %d Table S3 strains -> %d records; names %s; %d Table S2 Keio "
            "numbers name another locus or none",
            len(mutants),
            len(resolved),
            {s.value: n for s, n in ledger.names.status_histogram.items()},
            len(ledger.keio_disagreements),
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_ledgers(drop_log, ledger, checks, digest, parent, resolved)

        env = environment()
        reference = build_reference(
            self.name, assembly_reference(self.REFERENCE_STRAIN), env
        )
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, identity in enumerate(tqdm(resolved, desc="wang2015")):
                experiment = build_experiment(self.name, identity, parent, env)
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
        env_out.close()
        interned_env.close()
        log.info("Wrote %d Wang 2015 isoprenol-response experiments", len(resolved))

    def _write_ledgers(
        self,
        drop_log: DropLog,
        ledger: IdentifierLedger,
        checks: Sequence[CountCheck],
        digest: str,
        parent: TableS3Row,
        resolved: Sequence[StrainIdentity],
    ) -> None:
        """The drop log, identifier ledger, extraction checks and the strain table."""
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(drop_log.model_dump_json(indent=2))
        (out / "identifier_reconciliation.json").write_text(
            ledger.model_dump_json(indent=2)
        )
        (out / "extraction.json").write_text(
            json.dumps(
                {
                    "si_pdf_sha256": SI_PDF_SHA256,
                    "pdftotext": pdftotext_version(),
                    "pdftotext_args": list(PDFTOTEXT_ARGS),
                    "table_s3_sha256": digest,
                    "parent_growth_inhibition_pct": parent.growth_inhibition_pct,
                    "count_checks": [c.model_dump() for c in checks],
                },
                indent=2,
            )
        )
        table = [
            {
                "record": None,
                "strain": parent.strain,
                "mdt_family": None,
                "keio_token": None,
                "keio_accession": None,
                "locus_tag": None,
                "symbol": None,
                "od600_without": parent.od600_without,
                "sd_without": parent.sd_without,
                "od600_with": parent.od600_with,
                "sd_with": parent.sd_with,
                "growth_inhibition_pct": parent.growth_inhibition_pct,
                "log2_relative_tolerance": 0.0,
            }
        ] + [
            {
                "record": idx,
                "strain": item.row.strain,
                "mdt_family": item.mdt_family,
                "keio_token": item.keio_token,
                "keio_accession": item.keio_accession,
                "locus_tag": item.locus_tag,
                "symbol": item.symbol,
                "od600_without": item.row.od600_without,
                "sd_without": item.row.sd_without,
                "od600_with": item.row.od600_with,
                "sd_with": item.row.sd_with,
                "growth_inhibition_pct": item.row.growth_inhibition_pct,
                "log2_relative_tolerance": log2_relative_tolerance(item.row, parent),
            }
            for idx, item in enumerate(resolved)
        ]
        pd.DataFrame(table).astype({"record": "Int64"}).to_csv(
            out / "table_s3.csv", index=False
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled by ``build_experiment``."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built LMDB
# --------------------------------------------------------------------------- #
VERIFIER_PROVENANCE = Provenance(
    source_uri=f"{SI_SOURCE_URL} (Table S3)",
    citation_key=CITATION_KEY,
    method="log2 relative tolerance capacity to BW25113 from Table S3 OD600 means; "
    "reference = BW25113 (0)",
    page="Sci Rep 5:16505, Supplementary Table S3",
)


def run_verification(data_root: str | None = None) -> VerificationReport:
    """Run the environment-response family verifier (L0-L3) with the BW25113 resolver,
    the bacterial L4 containment, and the provenance audit of every ``SOURCED_VALUES``
    entry on the built dev LMDB; write ``preprocess/verification_report.json``.
    """
    from torchcell.verification.environment_response import (
        environment_response_gene_set,
        verify_environment_response_dataset,
    )
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _genome_for_reference,
        _write_report,
        load_records,
    )

    base = data_root or _data_root()
    abs_root = osp.join(base, DATASET_ROOT_REL)
    records = load_records(abs_root)
    drops = DropLog.model_validate_json(
        Path(abs_root, "preprocess", "dropped_records.json").read_text()
    )
    references = {
        json.dumps(r["reference"]["genome_reference"], sort_keys=True) for r in records
    }
    if len(references) != 1:
        raise ValueError(f"{len(references)} distinct genome references; expected 1")
    reference = json.loads(references.pop())
    genome = _genome_for_reference(reference, base)
    report = verify_environment_response_dataset(
        records,
        dataset_name=EnvChemgenWang2015Dataset.__name__,
        provenance=VERIFIER_PROVENANCE,
        expected_count=drops.kept_records,
        resolve_gene_name=genome.resolve_gene_name,
    )
    universe = _gene_set_for_reference(reference, base)
    deleted = environment_response_gene_set(records)
    missing = sorted(deleted - universe)
    report.add(
        LevelResult(
            level=Level.L4,
            name="gene_containment_bw25113_locus_tags",
            passed=not missing,
            message=f"{len(deleted) - len(missing)} of {len(deleted)} deleted loci are "
            "BW25113 GenBank gene rows",
            details={
                "n_deleted": len(deleted),
                "n_universe": len(universe),
                "missing_examples": missing[:20],
            },
        )
    )
    library = Path(base) / "torchcell-library"
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, library))
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` the dev LMDB, or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(prog="python -m torchcell.datasets.ecoli.wang2015")
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser(
        "deposit", help="retrieve (optional) and deposit the mirror"
    )
    deposit.add_argument("--download-dir", required=True)
    deposit.add_argument(
        "--retrieve",
        action="store_true",
        help="run the recorded PMC bucket retriever into --download-dir first",
    )
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "deposit":
        download = Path(args.download_dir)
        if args.retrieve:
            retrieve_raw_files(download)
        sources: dict[str, str | Path] = {name: download / name for name in DATA_SHA256}
        print(deposit_raw_mirror(sources=sources, data_root=data_root))
        return 0
    if args.command == "build":
        dataset = EnvChemgenWang2015Dataset(root=osp.join(data_root, DATASET_ROOT_REL))
        print(f"len = {len(dataset)}")
        return 0
    report = run_verification(data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
