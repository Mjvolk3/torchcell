# torchcell/datasets/pputida/kang2026
# [[torchcell.datasets.pputida.kang2026]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/kang2026
# Test file: tests/torchcell/datasets/pputida/test_kang2026.py
"""Kang 2026 isoprenyl acetate production campaign in P. putida KT2440.

Kang et al. 2026 (Metab. Eng. Commun., doi:10.1016/j.mec.2026.e00274) engineered the
isoprenol-producing KT2440 chassis PIPA into an isoprenyl acetate producer: a
heterologous alcohol acetyltransferase, three esterase deletions, an integrated xylose
isomerase pathway, two global-regulator deletions and an integrated acetyl-CoA module.
One dataset class, :class:`IsoprenylAcetateTiterKang2026Dataset`, serves every released
isoprenyl acetate titer as a ``ProductTiterExperiment``.

THREE TITER COLUMNS, NAMED, AND WHY ALL THREE ARE ONE MEASUREMENT. The paper releases
isoprenyl acetate as a number in exactly three places, all in mg/L and all quantified by
the same GC-FID method:

1. **Table 1, column ``Titer (mg/L); culture conditions``** -- the primary column and the
   engineering trajectory: seven strains, each the flask maximum under its own stated
   conditions ("Titers represent the highest isoprenyl acetate concentrations obtained in
   flask cultures for each strain under the indicated conditions"). Read from the pinned
   ``paper.md`` OCR mirror as sourced constants, one verbatim quote per row.
2. **Table S4, column ``IPA titer (mg/L)``** -- the five-member alcohol acyltransferase
   panel in the PIPA background, 5 mL tube cultures sampled at 48 h. Read from the
   deposited raw SI ``si/si1.docx``.
3. **Table S9, columns ``Isoprenyl acetate, aqueous / organic / off-gas (mg/L)``** -- the
   fed-batch time course of the final strain. The stored titer is the SUM of the three
   phases, which is the paper's own convention: the maximum of that sum over the seven
   sampled times is 1909.4 mg/L and the Results state "the PIPAxyl-E3-K3-O15 strain
   reached a final titer of 1.9 g/L isoprenyl acetate". The three phases are kept
   per-record in ``preprocess/titer_rows.csv``.

Nineteen records: 7 + 5 + 7. Nothing is dropped. The Table 1 strains that carry no titer
(``PIPAxyl``, ``PIPAxyl-O2``, ``PIPAxyl-O6``, ``PIPAxyl-O7`` and the integration variant
whose name the OCR renders ``014``) are not records; they are written to
``preprocess/strains_without_a_titer.csv``.

THE REPLICATE DESIGN IS SOURCED; THE UNCERTAINTY NUMBER IS NOT RELEASED. Every figure
caption in the paper and the SI states one design, verbatim: "Error bars indicate the
standard deviation of biological triplicates." So ``n_samples=3`` and
``sample_unit=biological_replicate`` are sourced and the uncertainty TYPE is a sample SD.
The SD VALUES exist only as error bars: there is no source-data workbook, no
per-replicate table and no SD column anywhere in the mirror. ``ProductTiterPhenotype``
forbids an unlabelled uncertainty (the number and its type are both set or both None), so
``titer_uncertainty`` and ``titer_uncertainty_type`` are typed ``ProvenanceGap``s naming
exactly that, with the design quote in the note. The fed-batch rows additionally gap
``n_samples`` and ``sample_unit``: Methods 2.8 describes one bioreactor run and never
states a replicate count.

THE CHASSIS. The reference every record is written against is **PIPA**, carried as a
``BacterialStrainBackground`` on an ``AssemblyReferenceGenome`` pinned to
``pputida_KT2440_ASM756v2`` / ``GCA_000007565.2``. Its parent is wild-type KT2440,
"Wild-type mt-2 derivative lacking the TOL plasmid pWW0". Table 1 states the genotype
WITH locus tags: "P. putida KT2440 ΔphaABC (PP_5003-5005) ΔmvaB (PP_3540) ∆hbdH
(PP_3073) ∆ldhA (PP_1649) ∆86kb". Everything built on PIPA is a perturbation in
``Genotype``: the pIY670 and pAAT plasmid genes and the two integrated cassettes as
``HeterologousPathwayPerturbation``, the esterase, regulator and integration-site
disruptions as ``BacterialDeletionPerturbation``.

WHAT THIS PAPER'S OWN LOCUS TAGS SETTLE THAT THE ANNOTATION CANNOT. Two symbols of this
background resolve to nothing in the pinned KT2440 annotation, and Table 1 names their
loci outright, so neither is a gap here:

- ``phaC``: the symbol is RETIRED against this assembly (measured), and Table 1's range
  ``PP_5003-5005`` places it at ``PP_5005``, which the annotation carries as ``phaC-II``.
  The mirrored Carruthers 2025 campaign records ``phaC`` as an unmappable gap for the
  sibling chassis; Kang's range closes it.
- ``crc``: the symbol is RETIRED and ``PP_5292`` carries no symbol in the annotation.
  Table 1 writes "Δcrc (PP_5292)".

THE Δ86kb DELETION STATES ITSELF THREE WAYS AND THEY DO NOT AGREE. One cell of Table S2
(and of Table 1) gives a label (``Δ86kb``), a coordinate span (``4,536,184-4,627,926``)
and a locus-tag range (``PP_4023-PP_4092``). Measured on GCA_000007565.2:

- the span is 91,743 bp, not 86 kb;
- 67 annotated loci lie entirely inside the span, and ``PP_4023`` straddles its left edge
  (``4,535,579..4,538,407``), so the deletion truncates that gene rather than removing it;
- the locus-tag range names 64 annotated tags, and four loci inside the span are not in it
  (``PP_5652``, ``PP_5653``, ``PP_5654``, ``PP_mr45`` -- later-added annotations whose
  tags sit out of coordinate order).

The typed alleles are the 63 loci both statements agree on (``full_deletion`` carrying
``deleted_span``) plus ``PP_4023`` as a ``partial_deletion``. The span-only loci and the
tag-range numbers this assembly does not annotate are written to
``preprocess/span_disagreement.csv`` rather than typed. Nothing is chosen by preference:
every statement is kept and the intersection is the conservative set.

TITER UNITS: THE SOURCE'S mg/L IS STORED VERBATIM AS ``ug/mL``. ``ConcentrationUnit`` has
no ``mg/L`` member and 1 mg/L is exactly 1 ug/mL, so no arithmetic is applied to a source
value. Adding ``mg_per_l`` is a served-closure decision and is raised in the PR.

THE PRODUCT HAS NO COMPOUND-IDENTITY ROW. ``resolved_compound("isoprenyl acetate")``
returns the canonical name with a typed ``ProvenanceGap`` on ``inchikey``; the InChIKey
this module records for the curation that will fill it is
:data:`ISOPRENYL_ACETATE_INCHIKEY`, DERIVED from the structure rather than read from a
mirror, and deliberately not asserted onto the ``Compound``.

ENVIRONMENT, AND WHAT THE SLOT CANNOT HOLD. The three Kang media already live in
``torchcell.datamodels.media`` (``M9_NREL_KANG2026``, ``M9_NREL_HIGH_N_KANG2026``,
``M9_MOPS_KANG2026``), each carbon-source free, so every sugar is an
``EnvironmentPhysicalPerturbation(factor=carbon_source)`` and every inducer, overlay and
supplement a ``SmallMoleculePerturbation``. ``ProductTiterExperiment.environment`` is
annotated ``Environment``, not ``CultureEnvironment``, so a vessel and working volume
would be serialized away; they are recorded in :data:`CULTURE_FORMATS` and in the note.
The 48 h sugar + nitrogen pulse that four conditions use has no ``Environment`` slot
either and is carried per record in ``preprocess/titer_rows.csv``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import shutil
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    verify_sha256,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import (
    M9_MOPS_KANG2026,
    M9_NREL_HIGH_N_KANG2026,
    M9_NREL_KANG2026,
)
from torchcell.datamodels.schema import (
    AlleleEdit,
    AssemblyReferenceGenome,
    BacterialAssemblySet,
    BacterialBackgroundAllele,
    BacterialDeletionPerturbation,
    BacterialGeneNamespace,
    BacterialReferenceStrain,
    BacterialStrainBackground,
    Compound,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    GenomicSpan,
    Genotype,
    HeterologousPathwayPerturbation,
    Media,
    PhysicalFactor,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
    ProductYieldUnit,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.verification.levels import (
    l0_structural,
    l1_count,
    l2_cross_method,
    l2_value_fidelity,
    l3_convention,
    l4_cross_source,
)
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1016/j.mec.2026.e00274"
PMCID = "PMC12996797"
TITLE = (
    "Multi-layered metabolic remodeling of Pseudomonas putida for efficient conversion "
    "of lignocellulosic sugars to the precursors of advanced aviation fuel"
)

CITATION_KEY = "kangMultilayeredMetabolicRemodeling2026"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

#: PMC Article Datasets bucket prefix of this article's open-access version.
PMC_PREFIX = f"{PMCID}.1"
#: The Supplementary Information document: Tables S1-S9 plus the figure captions. The
#: only released data file of this paper, and the one the loader consumes.
SI_DOCX_FILENAME = "si1.docx"
SI_DOCX_REL = f"si/{SI_DOCX_FILENAME}"
SI_DOCX_SOURCE_FILENAME = "mmc1.docx"
SI_DOCX_SHA256 = "c7d4567fae037c7392e39b855cc71f83b7b4a5d96699c3f18b991a25c57f09e0"
SI_RETRIEVED_AT = "2026-10-07"

#: Mirrored OCR the Table 1 values quote (torchcell-library, NOT the raw mirror).
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "894aee24472194d22c33ecc0a31994660b19bde41e4aadee2cb0080f0c016c63"

#: The mirrored sibling campaign this module defers to for the pIY670 part organisms.
CARRUTHERS_KEY = "carruthersAutomationMachineLearning2025"
CARRUTHERS_PAPER_MD_SHA256 = (
    "ca9a8a2593d2ae3ab3bacfb767e797ece2bdaa0228a4f798f1c38e1af73ef88d"
)

#: The campaign's proteomics deposit. No loader here consumes raw spectra.
PRIDE_ACCESSION = "PXD067010"
#: Where the strains and plasmids were deposited.
JBEI_REGISTRY_URL = "https://public-registry.jbei.org"

KT2440_NAMESPACE: BacterialGeneNamespace = "pputida_kt2440_locus_tag"
KT2440_ASSEMBLY_SET: BacterialAssemblySet = "pputida_KT2440_ASM756v2"
#: The single chromosome of GCA_000007565.2, as the assembly report names it.
KT2440_REPLICON = "AE015451.2"

#: The chassis every record is written against, and its sequenced parent.
CHASSIS_STRAIN = "PIPA"
PARENT_STRAIN: BacterialReferenceStrain = "KT2440"

#: 1-based inclusive coordinate span Table S2 states for the Δ86kb deletion.
SPAN_START = 4_536_184
SPAN_END = 4_627_926
#: The locus-tag range the SAME cell states for the same deletion.
SPAN_TAG_FIRST = 4023
SPAN_TAG_LAST = 4092

#: What one stored titer IS.
QUANTIFICATION_METHOD = "GC-FID"
#: Standard InChIKey of isoprenyl acetate (3-methyl-3-buten-1-yl acetate), DERIVED from
#: the structure SMILES ``CC(=C)CCOC(C)=O`` with ``rdkit.Chem.inchi.MolToInchiKey``
#: (rdkit 2026.03.6, run 2026-10-07). Recorded for the
#: ``compound_identity_table.json`` curation that will fill it, and NOT set on the
#: ``Compound``: that table is PubChem-sourced and curating a row is a human act.
ISOPRENYL_ACETATE_INCHIKEY = "OCUAPVNNQFAQSM-UHFFFAOYSA-N"
PRODUCT_NAME = "isoprenyl acetate"

#: Vessel and working volume per culture format. ``ProductTiterExperiment.environment``
#: is annotated ``Environment``, so these would be dumped away on the record.
CULTURE_FORMATS: dict[str, dict[str, Any]] = {
    "tube": {
        "vessel": "50 mL culture tube",
        "working_volume_ml": 5.0,
        "shaking_rpm": 200.0,
    },
    "flask": {
        "vessel": "250 mL unbaffled shake flask",
        "working_volume_ml": 50.0,
        "shaking_rpm": 200.0,
    },
    "bioreactor": {
        "vessel": "2 L DASGIP parallel bioreactor",
        "working_volume_ml": 1200.0,
        "shaking_rpm": None,
    },
}


# --------------------------------------------------------------------------- #
# The SI document reader
# --------------------------------------------------------------------------- #
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = f"{{{_W_NS}}}"


def _text_of(element: Any) -> str:
    """Whitespace-normalized concatenation of every ``w:t`` run under an element."""
    return " ".join("".join(node.text or "" for node in element.iter(f"{_W}t")).split())


def _cell_text(cell: Any) -> str:
    """One table cell: its paragraphs joined by a single space, normalized."""
    return " ".join(
        part for part in (_text_of(para) for para in cell.findall(f"{_W}p")) if part
    ).strip()


class SiTable(BaseModel):
    """One table of the SI document: the caption paragraph above it and its rows."""

    caption: str
    rows: list[list[str]]

    def header(self) -> list[str]:
        """The first row, which every consumed table uses as its column names."""
        if not self.rows:
            raise RuntimeError(f"{self.caption[:40]!r} has no rows")
        return self.rows[0]

    def column(self, name: str) -> int:
        """Index of one column by its exact header text."""
        header = self.header()
        if name not in header:
            raise RuntimeError(
                f"{self.caption[:40]!r} has no column {name!r}: {header}"
            )
        return header.index(name)


def read_si_tables(path: str | Path) -> list[SiTable]:
    """Every table of the SI ``.docx``, each paired with the paragraph above it.

    A ``.docx`` is a zip of OpenXML; the body's children are paragraphs and tables in
    document order, so the caption of a table is the last non-empty paragraph before it.
    Read with the standard library (``zipfile`` + ``ElementTree``) rather than a new
    dependency, and deterministic: the cell text every quote below is matched against is
    exactly what this function returns.
    """
    with ZipFile(path) as archive:
        document = archive.read("word/document.xml")
    body = ElementTree.fromstring(document).find(f"{_W}body")
    if body is None:
        raise RuntimeError(f"{path} has no w:body")
    tables: list[SiTable] = []
    caption = ""
    for child in body:
        if child.tag == f"{_W}p":
            text = _text_of(child)
            if text:
                caption = text
        elif child.tag == f"{_W}tbl":
            tables.append(
                SiTable(
                    caption=caption,
                    rows=[
                        [_cell_text(cell) for cell in row.findall(f"{_W}tc")]
                        for row in child.findall(f"{_W}tr")
                    ],
                )
            )
    return tables


def si_table(path: str | Path, number: int) -> SiTable:
    """The SI table whose caption begins ``Table S<number>.``, or a refusal."""
    prefix = f"Table S{number}."
    matches = [t for t in read_si_tables(path) if t.caption.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"{osp.basename(str(path))} holds {len(matches)} tables captioned "
            f"{prefix!r}; exactly one is expected"
        )
    return matches[0]


# --------------------------------------------------------------------------- #
# Sourced values: every number below quotes sha256-pinned mirrored bytes
# --------------------------------------------------------------------------- #
def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the pinned ``paper.md`` OCR mirror."""
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


def _si(value: Any, quote: str, *, page: str, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim cell of the deposited Supplementary Information docx."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI_DOCX_REL,
            citation_key=CITATION_KEY,
            sha256=SI_DOCX_SHA256,
            method="published Supplementary Information document (raw mirror), read by "
            "torchcell.datasets.pputida.kang2026.read_si_tables",
            page=page,
            retrieved=SI_RETRIEVED_AT,
        ),
    )


def _carruthers(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to the MIRRORED sibling campaign this paper defers pIY670 to.

    Kang's Table 1 cites pIY670 to Banerjee 2024, which is not mirrored; the mirrored
    Carruthers 2025 campaign carries the same plasmid and states the part organisms. The
    provenance chain (Kang's part string -> the same parts in Carruthers 2025) is the
    citation, per the deferral rule in ``CLAUDE.md``.
    """
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CARRUTHERS_KEY,
            sha256=CARRUTHERS_PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror of the "
            "deferred-to paper)",
            page=page,
        ),
    )


_METHODS_STRAINS = "Methods 2.1, 'Strains and plasmid construction'"
_METHODS_MEDIUM = "Methods 2.2, 'Reagents and growth medium'"
_METHODS_PRODUCTION = "Methods 2.4, 'Isoprenyl acetate production test'"
_METHODS_GC = (
    "Methods 2.5, 'Gas chromatography analysis for isoprenol and isoprenyl acetate'"
)
_METHODS_FEDBATCH = "Methods 2.8, 'Fed-batch cultivation'"
_TABLE1 = "Table 1, 'Key engineered strains and plasmids used in this study'"
_TABLE1_FOOTNOTE = "Table 1, footnote a"
_SI_TABLE2 = "Table S2, 'Strains and plasmids used in this study'"
_SI_TABLE4 = "Table S4, 'Comparison of alcohol acyltransferases (AATs)'"
_SI_TABLE5 = "Table S5, 'Putative esterases in P. putida'"
_SI_TABLE8 = (
    "Table S8, 'Summary of genes engineered to enhance acetyl-CoA biosynthesis'"
)
_SI_TABLE9 = "Table S9, 'Time-course analysis of sugar consumption and isoprenyl acetate partitioning in fed-batch fermentation'"

_Q_PARENT = "Wild-type mt-2 derivative lacking the TOL plasmid pWW0"
_Q_PIPA_SI = (
    "P. putida KT2440 ΔphaABC ΔmvaB ΔhbdH ΔldhA Δ86kb (4,536,184-4,627,926; "
    "PP_4023-PP_4092)"
)
_Q_PIPA_TABLE1 = (
    "P. putida KT2440 ΔphaABC (PP_5003-5005) ΔmvaB (PP_3540) ∆hbdH (PP_3073) ∆ldhA "
    "(PP_1649) ∆86kb (4,536,1844,627,926; PP_4023-PP_4092)"
)
_Q_PIPA_SOURCE = (
    "The P. putida strain used for isoprenol production was adapted from a previous "
    "study (Banerjee et al., 2024)."
)
_Q_PIPA_DELETIONS = (
    "Our base strain, PIPA, incorporated the GSMM-guided deletions of six genes "
    "(phaABC, mvaB, hbdH, and ldhA) from that study to redirect the metabolic flux "
    "towards isoprenol synthesis along with a deletion of the leucine degradation "
    "pathway to reduce native isoprenol degradation"
)
_Q_IPP_BYPASS = (
    "Additionally, five genes of the isopentenyl diphosphate (IPP)-bypass pathway "
    "(mvaS, mvaE, $M K _ { M m } ,$ $P M D _ { H K Q } ,$ and aphA) were expressed "
    "from a medium-copy plasmid (Kang et al., 2019)."
)
_Q_PIY670_PARTS = "pRK2-Kan-araC-PBAD-MvaSef-MvaEef-TrpoH-Ptrc1-O-MKmm-PMDHKQ-AphA"
_Q_APHA = (
    "The resulting IP is then dephosphorylated to isoprenol by the promiscuous "
    "monophosphatase AphA from E. coli, thereby supplying the C5 precursor for "
    "isoprenyl acetate, while the native MEP pathway is retained for essential "
    "isoprenoid biosynthesis."
)
_Q_PMD_CARRUTHERS = "promiscuous mevalonate decarboxylase (PMD\\*) from S. cerevisiae"
_Q_PAAT1 = "pRSF1010-Gm-NagR-PnagAa-ATF1-rrnB T1"
_Q_ATF = (
    "S. cerevisiae-derived ATF1 [NCBI: NP_015022.3] and ATF2 [NCBI: NP_011693.1] were "
    "amplified from JBEI-136483 and JBEI-136484, respectively (Carruthers et al., 2023)."
)
_Q_SAAT = (
    "The SAAT [GenBank: AAG13130.1] from Fragaria $\\times$ ananassa, codon-optimized "
    "for expression in E. coli, was obtained from JBEI-231873 (Carruthers et al., 2023)."
)
_Q_AAT_CAT = (
    "The R. hybrida-derived AAT [GenBank: AY850287.1] and E. coli-derived CAT "
    "[UniProtKB: P00484], carrying the Y20F mutation, were synthesized with P. "
    "putida-optimized codons by Twist Biosciences (Seo et al., 2021; Wang et al., 2022)."
)
_Q_XFPK = (
    "Additionally, codon-optimized xfpk [UniProtKB: A1A185], derived from "
    "Bifidobacterium adolescentis, was obtained from the JBEI registry."
)
_Q_XYL_GENES = (
    "We introduced codon-optimized E. coli genes xylE, xylA, and xylB, encoding a "
    "xylose transporter, isomerase, and kinase, respectively. Additionally, "
    "codon-optimized E. coli tal and tkt genes, which encode a transaldolase and "
    "transketolase, respectively, were introduced to support flux through the PP "
    "pathway."
)
_Q_XYL_CASSETTE = (
    "Finally, the complete expression cassette $( \\mathrm { P } _ { x y l E ^ { * - } "
    "} x y l E { : } \\mathrm { P } _ { \\mathrm { t a c } ^ { - } } x y l A B$ : "
    "talB:tktA) was integrated into the gcd locus, disrupting glucose dehydrogenase to "
    "prevent conversion of xylose to xylonate, thereby yielding strain PIPAxyl from the "
    "parental strain PIPA (Table 1)."
)
_Q_ESTERASES = (
    "Next, we minimized product degradation by disrupting endogenous esterases "
    "(PP_1127, PP_3812, and PP_4218) involved in the hydrolysis of"
)
_Q_ACS_EC = "Overexpression of E. coli acs"
_Q_TRIPLICATE = "Error bars indicate the standard deviation of biological triplicates."
_Q_TRIPLICATE_FIG2 = (
    "Error bars represent the standard deviation of biological triplicates."
)
_Q_GCFID = (
    "To analyze isoprenyl acetate and isoprenol concentrations, samples were processed "
    "using gas chromatography-flame ionization detection (GC-FID) (Thermo Focus GC, "
    "Thermo Scientific) equipped with a DB-WAX column"
)
_Q_TABLE1_FOOTNOTE = (
    "Titers represent the highest isoprenyl acetate concentrations obtained in flask "
    "cultures for each strain under the indicated conditions and correspond to the "
    "flask data points shown in Fig. 7c."
)
_Q_PULSE = (
    "“Pulse” indicates that at $^ { 4 8 \\mathrm { h } }$ an additional feed of "
    "glucose, xylose, and $\\mathrm { ( N H } _ { 4 } \\mathrm { ) } _ { 2 } S 0 _ { 4 "
    "}$ was supplied at the same concentrations as at $^ { 0 \\mathrm { h } }$ to "
    "maintain the initial C:N ratio."
)
_Q_TUBE_FLASK = (
    "Tube cultures were conducted in $5 \\mathrm { m L }$ of medium in $5 0 \\mathrm { "
    "m L }$ culture tubes, with samples collected at $4 8 \\mathrm { h }$ , while "
    "flask cultures were performed in $5 0 ~ \\mathrm { m L }$ of medium in 250 mL "
    "unbaffled shake flasks, with sampling every $2 4 \\ \\mathrm { h }$ ."
)
_Q_INDUCTION = (
    "At an $\\mathrm { O D } _ { 6 0 0 }$ of 0.6–0.8, cultures were induced, and a $2 "
    "0 \\%$ $\\mathbf { ( v / v ) }$ dodecane overlay was added."
)
_Q_TEMPERATURE = (
    "In each case, a single colony was inoculated into $5 ~ \\mathrm { m L }$ of LB "
    "medium with appropriate antibiotics in $5 0 ~ \\mathrm { m L }$ culture tubes and "
    "grown overnight at $3 0 ~ ^ { \\circ } \\mathrm { C }$ with shaking at $2 0 0 "
    "\\mathrm { r p m }$ ."
)
_Q_FEDBATCH_TEMPERATURE = (
    "The bioreactor was maintained at $3 0 ~ ^ { \\circ } \\mathrm { C }$ with "
    "dissolved oxygen (DO) controlled at $^ { 1 5 \\% }$ ."
)
_Q_FEDBATCH_VESSEL = (
    "Fed-batch cultivation was conducted in a $^ \\mathrm { ~ 2 ~ L ~ }$ DASGIP "
    "parallel bioreactor system (Eppendorf) with a working volume of $^ { 1 . 2 "
    "\\mathrm { ~ L ~ } }$ ."
)
_Q_FEDBATCH_INDUCTION = (
    "Induction was performed at $\\mathrm { O D } _ { 6 0 0 } 0 . 6$ by addition of $"
    "{ 2 g / \\mathrm { L } }$ Ara, $6 2 . 5 \\mu \\mathrm { M }$ SA, and $2 5 0 \\mu "
    "\\mathrm { M }$ 3 MB."
)
_Q_FEDBATCH_OVERLAY = (
    "To mitigate product loss, $2 0 \\%$ (v/v) Durasyn 164 (Univar Solutions) was used "
    "as an overlay."
)
_Q_FEDBATCH_MEDIUM = (
    "This adaptation step was repeated once using modified M9-MOPS containing $1 3 . 3 "
    "~ \\gimel$ glucose and $6 . 7 ~ { \\ g / \\mathrm { L } }$ xylose, followed by a "
    "third adaptation in $1 0 0 ~ \\mathrm { { m L } }$ of the same medium in $^ "
    "\\textrm { \\scriptsize 1 L }$ unbaffled shake flasks for $^ { 8 \\mathrm { h } }$ ."
)
_Q_FEDBATCH_FEED = (
    "The feeding solution consisted of $2 6 6 . 7 ~ \\mathrm { g / L }$ glucose and $1 "
    "3 3 . 3 ~ \\mathrm { g / L }$ xylose and was prepared using the same basal "
    "composition as the modified M9-MOPS culture medium with 1 to $5 ~ \\mathrm { g } / "
    "\\mathrm { L }$ yeast extract."
)
_Q_FEDBATCH_TITER = (
    "Under these fed-batch conditions, the PIPAxyl-E3-K3-O15 strain reached a final "
    "titer of $1 . 9 { \\ g } / \\mathrm { L }$ isoprenyl acetate from a total feed of "
    "28.0 $_ { 8 } / \\mathrm { L }$ glucose and $1 2 . 5 ~ \\mathrm { g / L }$ xylose, "
    "corresponding to a yield of $0 . 0 6 7 ~ \\mathrm { g } / \\mathrm { g }$ and $1 8 "
    ". 1 \\%$ of the theoretical maximum (Fig. 8b)."
)
_Q_FEDBATCH_SAMPLING = (
    "Samples of $5 \\mathrm { m L }$ to $1 0 ~ \\mathrm { m L }$ were collected "
    "approximately every $^ { 1 2 \\mathrm { ~ h ~ } }$ for analysis of $\\mathrm { O D "
    "} _ { 6 0 0 } .$ , residual sugars by HPLC, and isoprenyl acetate levels by GC-FID."
)
_Q_YEAST_EXTRACT = (
    "Cultures were grown in M9-MOPS medium supplemented with ${ } ^ { 1 } { \\ g } / "
    "\\mathrm { L }$ yeast extract."
)
_Q_FLASK_MAX_PIPA = (
    "Under these conditions, the triple knockout strain PIPA-E3 demonstrated the most "
    "promising performance in M9 medium containing $2 0 \\ g / \\mathrm { L }$ glucose, "
    "with titers increasing from 414 mg/"
)
_Q_K3_TIME = (
    "achieved the highest isoprenyl acetate titer of $5 9 9 \\mathrm { m g / L }$ at $1 "
    "4 4 \\mathrm { ~ h ~ }$ (Fig. 6b and Table 1), representing a"
)
_Q_PIPAXYL_BASELINE = (
    "Notably, these efforts enabled an over 10-fold increase in isoprenyl acetate "
    "production, from $1 8 1 ~ \\mathrm { m g / L }$ in the unoptimized PIPAxyl strain "
    "to $1 . 5 ~ \\gparallel \\mathrm { L }$ in shake flasks and $1 . 9 \\ g / \\mathrm "
    "{ L }$ under fed-batch cultivation with the consolidated PIPAxy"
)
_Q_AAT_PANEL_BEST = (
    "Among the AAT variants tested, the PIPA-AAT1 strain (i.e., PIPA harboring pIY670 "
    "and pAAT1) expressing ATF1 from S. cerevisiae achieved the highest isoprenyl "
    "acetate production with a titer of $4 0 5 ~ \\mathrm { m g / L }$ from ${ 2 0 ~ \\ "
    "g / \\mathrm { L } }$ glucose (Fig. 2b)."
)
_Q_OFFGAS = (
    "Notably, $5 2 \\%$ of the total isoprenyl acetate was recovered from the off-gas "
    "trap at the end of the 168-h cultivation, despite the inclusion of a $2 0 \\%$ "
    "Durasyn overlay in the culture (Table S9)."
)


#: The strain-description cell each strain's genotype is read from, verbatim. Table 1
#: carries the locus tags and is quoted wherever it names the strain; the five AAT-panel
#: strains appear only in Table S2, whose cells are the plain plasmid lists.
STRAIN_DESCRIPTIONS: dict[str, str] = {
    "PIPA-AAT1": "PIPA pIY670 pAAT1",
    "PIPA-AAT2": "PIPA pIY670 pAAT2",
    "PIPA-AAT3": "PIPA pIY670 pAAT3",
    "PIPA-AAT4": "PIPA pIY670 pAAT4",
    "PIPA-AAT5": "PIPA pIY670 pAAT5",
    "PIPA-E3": "PIPA ΔPP_1127 ΔPP_3812 ΔPP_4218 pIY670 pAAT1",
    "PIPAxyl-AAT1": "PIPAxyl pIY670 pAAT1",
    "PIPAxyl-E3": "PIPAxyl ΔPP_1127 ΔPP_3812 ΔPP_4218 pIY670 pAAT1",
    "PIPAxyl-E3-K3": (
        "PIPAxyl ΔPP_1127 ΔPP_3812 ΔPP_4218 Δcrc (PP_5292) ∆hexR (PP_1021) pIY670 pAAT1"
    ),
    "PIPAxyl-E3-O15": (
        "PIPAxyl ΔPP_1127 ΔPP_3812 ΔPP_4218 ΔampC (PP_2876):Pm-acSEc-xfpkBa_RBS1 "
        "pIY670 pAAT1"
    ),
    "PIPAxyl-E3-K3-O15": (
        "PIPAxyl ΔPP_1127 ΔPP_3812 ΔPP_4218 Δcrc (PP_5292) ΔhexR (PP_1021) ∆ampC "
        "(PP_2876):Pm-acSEc-xfpkBa_RBS1 pIY670 pAAT1"
    ),
}
#: Which mirror each strain description is quoted from.
STRAIN_DESCRIPTION_SOURCE: dict[str, str] = {
    "PIPA-AAT1": "table1",
    "PIPA-AAT2": "si",
    "PIPA-AAT3": "si",
    "PIPA-AAT4": "si",
    "PIPA-AAT5": "si",
    "PIPA-E3": "table1",
    "PIPAxyl-AAT1": "table1",
    "PIPAxyl-E3": "table1",
    "PIPAxyl-E3-K3": "table1",
    "PIPAxyl-E3-O15": "table1",
    "PIPAxyl-E3-K3-O15": "table1",
}

# --------------------------------------------------------------------------- #
# Sourced scalars
# --------------------------------------------------------------------------- #
PARENT_GENOTYPE = _si(PARENT_STRAIN, _Q_PARENT, page=_SI_TABLE2)
CHASSIS_GENOTYPE = _si(
    "KT2440 ΔphaABC ΔmvaB ΔhbdH ΔldhA Δ86kb (4,536,184-4,627,926; PP_4023-PP_4092)",
    _Q_PIPA_SI,
    page=_SI_TABLE2,
    note="the only statement of the Δ86kb coordinates with its hyphen intact; the "
    "paper.md OCR of the same cell renders the range as '4,536,1844,627,926'",
)
CHASSIS_GENOTYPE_TABLE1 = _paper(
    "KT2440 ΔphaABC (PP_5003-5005) ΔmvaB (PP_3540) ΔhbdH (PP_3073) ΔldhA (PP_1649) "
    "Δ86kb",
    _Q_PIPA_TABLE1,
    page=_TABLE1,
    note="the ONLY statement that gives a locus tag per deleted gene; it is what puts "
    "phaC at PP_5005 and crc at PP_5292, neither of which the pinned annotation's "
    "symbol layer resolves",
)
CHASSIS_SOURCE = _paper(
    "Banerjee et al. 2024",
    _Q_PIPA_SOURCE,
    page="Results 3.1, 'Production of isoprenyl acetate in P. putida'",
)
CHASSIS_DELETIONS = _paper(
    ("phaABC", "mvaB", "hbdH", "ldhA", "86kb leucine-degradation span"),
    _Q_PIPA_DELETIONS,
    page="Results 3.1, 'Production of isoprenyl acetate in P. putida'",
    note="the source calls phaABC+mvaB+hbdH+ldhA 'six genes', which is phaA, phaB, "
    "phaC, mvaB, hbdH and ldhA; the leucine-degradation deletion is the Δ86kb span",
)
JBEI_ACCESSIONS = _si(
    {"PIPA": "JBEI-233661", "pIY670": "JBx_233660", "pAAT1": "JBx_231759"},
    "JBEI-233661",
    page=_SI_TABLE2,
    note="the registry accessions Table S2 gives for the chassis and the two plasmids "
    "every record carries",
)
MEDIUM_SOURCES: dict[str, SourcedValue] = {
    "M9": _paper(
        M9_NREL_KANG2026.name,
        "P. putida strains were cultured in either LB medium, M9, modified M9, or "
        "M9-MOPS.",
        page=_METHODS_MEDIUM,
    ),
    "modified M9": _paper(
        M9_NREL_HIGH_N_KANG2026.name,
        "For experiments requiring modified nitrogen levels, the concentration of $"
        "\\mathrm { ( N H } _ { 4 } \\mathrm { ) } _ { 2 } S 0 _ { 4 }$ was increased to "
        "$4 0 ~ \\mathrm { m M }$ and is referred to as modified M9.",
        page=_METHODS_MEDIUM,
    ),
    "M9-MOPS": _paper(
        M9_MOPS_KANG2026.name,
        "M9-MOPS was prepared with the following components: M9 salts",
        page=_METHODS_MEDIUM,
    ),
}
TEMPERATURE_C = _paper(
    30.0,
    _Q_TEMPERATURE,
    page=_METHODS_PRODUCTION,
    note="30 C is stated for the seed and adaptation steps of the production test and "
    "again, explicitly, for the bioreactor ('" + _Q_FEDBATCH_TEMPERATURE + "'); the "
    "production culture's own temperature is never restated, and no other temperature "
    "appears anywhere in the Methods",
)
AEROBICITY = _paper(
    "aerobic",
    _Q_TUBE_FLASK,
    page=_METHODS_PRODUCTION,
    note="shaken tube and unbaffled-flask cultures, and a bioreactor held at 15% "
    "dissolved oxygen; the source never uses the word",
)
N_REPLICATES = _paper(
    3,
    _Q_TRIPLICATE,
    page="Fig. 3 caption (and Figs. 2, 4, 5, 6, 7 and Figs. S1-S10, identically)",
    note="biological triplicates. The SD VALUES are released only as error bars: no "
    "source-data workbook, per-replicate table or SD column exists in the mirror, so "
    "the uncertainty number and its type are typed gaps on every phenotype",
)
UNCERTAINTY_TYPE_STATED = _paper(
    "sample_sd",
    _Q_TRIPLICATE_FIG2,
    page="Fig. 2 caption",
    note="what the released error bars ARE, and therefore what titer_uncertainty_type "
    "WOULD be if the numbers were released. It is not set, because "
    "ProductTiterPhenotype forbids a type without a value",
)
QUANTIFICATION = _paper(QUANTIFICATION_METHOD, _Q_GCFID, page=_METHODS_GC)
TABLE1_RULE = _paper(
    "flask maximum under the stated conditions",
    _Q_TABLE1_FOOTNOTE,
    page=_TABLE1_FOOTNOTE,
    note="the scoring rule of the primary titer column: a MAXIMUM over a flask time "
    "course sampled every 24 h, not an endpoint, and an upward-biased order statistic",
)
PULSE_RULE = _paper(
    "48 h feed of glucose, xylose and (NH4)2SO4 at the 0 h concentrations",
    _Q_PULSE,
    page=_TABLE1_FOOTNOTE,
    note="Environment has no slot for a mid-culture feed event, so the flag is carried "
    "in preprocess/titer_rows.csv",
)
TUBE_FORMAT = _paper(CULTURE_FORMATS["tube"], _Q_TUBE_FLASK, page=_METHODS_PRODUCTION)
FLASK_FORMAT = _paper(CULTURE_FORMATS["flask"], _Q_TUBE_FLASK, page=_METHODS_PRODUCTION)
BIOREACTOR_FORMAT = _paper(
    CULTURE_FORMATS["bioreactor"], _Q_FEDBATCH_VESSEL, page=_METHODS_FEDBATCH
)
FEDBATCH_YIELD = _paper(
    0.067,
    _Q_FEDBATCH_TITER,
    page="Results 3.6, 'Isoprenyl acetate production in fed-batch cultivation'",
    note="g product per g sugar fed (28.0 g/L glucose + 12.5 g/L xylose). Stated once, "
    "for the run's maximum titer, so it is carried on the record whose three-phase sum "
    "is that maximum and on no other time point",
)
FEDBATCH_FINAL_TITER_G_PER_L = _paper(
    1.9,
    _Q_FEDBATCH_TITER,
    page="Results 3.6, 'Isoprenyl acetate production in fed-batch cultivation'",
    note="the cross-check the three-phase sum has to reproduce: max over Table S9 is "
    "1909.4 mg/L at 141.4 h, which is the 1.9 g/L the Results state. The source calls "
    "it 'final' although the 165.4 h end point is lower (1435.1 mg/L)",
)
OFFGAS_FRACTION_PERCENT = _paper(
    52.0,
    _Q_OFFGAS,
    page="Results 3.6, 'Isoprenyl acetate production in fed-batch cultivation'",
    note="Table S9's own 'Off-gas fraction (%)' column reads 51.9 at 165.4 h; that "
    "column is the independent oracle for the three-phase sum this loader stores",
)
PIY670_PARTS = _si(
    _Q_PIY670_PARTS,
    _Q_PIY670_PARTS,
    page=_SI_TABLE2,
    note="the part order the promoter assignment below is read from: araC-PBAD drives "
    "MvaSef and MvaEef up to the TrpoH terminator, and Ptrc1-O drives MKmm, PMDHKQ and "
    "AphA. The source does not draw the operon boundaries in prose",
)
PAAT1_PARTS = _si(
    _Q_PAAT1,
    _Q_PAAT1,
    page=_SI_TABLE2,
    note="the AAT plasmids are this backbone with the AAT swapped, so every pAAT gene "
    "is driven by PnagAa",
)
XYL_CASSETTE = _paper(
    "PxylE*-xylE:Ptac-xylAB:talB:tktA",
    _Q_XYL_CASSETTE,
    page="Results 3.4, 'Introduction of a xylose utilization pathway'",
    note="integrated at the gcd locus, which Table 1 names PP_1444; xylE keeps its "
    "native promoter with an ALE-identified single adenosine insertion at -10, which "
    "is what the asterisk in PxylE* means",
)
ACS_XFPK_CASSETTE = _paper(
    "Pm-acsEc-xfpkBa_RBS1",
    STRAIN_DESCRIPTIONS["PIPAxyl-E3-O15"],
    page=_TABLE1,
    note="Table 1 writes the integration as 'ΔampC (PP_2876):Pm-acSEc-xfpkBa_RBS1'; "
    "the capitalization of acSEc is OCR noise for acsEc, which Table S2 writes "
    "consistently. RBS1 is the first of five RBS variants (Fig. S7)",
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/kangMultilayeredMetabolicRemodeling2026``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/<citation key>``: the OCR the quotes cite."""
    return Path(data_root or _data_root()) / "torchcell-library" / CITATION_KEY


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pmc_cloud_key(filename: str = SI_DOCX_SOURCE_FILENAME) -> str:
    """Bucket key of the supplementary file in the PMC Article Datasets bucket."""
    return f"{PMC_PREFIX}/{filename}"


def deposit_raw_mirror(
    *,
    si_docx_path: str | Path,
    retrieved_at: str = SI_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror (the SI document) and its ``manifest.json``.

    Idempotent by sha256: an existing mirror file whose digest matches is left alone and
    a differing one raises rather than being overwritten. The file is verified BEFORE
    anything is written, so a refusal leaves no partial deposit. It comes from the PMC
    Article Datasets bucket, which is directly scriptable: the recorded retrieval
    re-runs as-is and reproduced the pinned digest on ``retrieved_at``.
    """
    verify_sha256(si_docx_path, SI_DOCX_SHA256)
    root = raw_mirror_dir(data_root)
    dest = root / SI_DOCX_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != SI_DOCX_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(si_docx_path, dest)
    key = pmc_cloud_key()
    url = f"https://pmc-oa-opendata.s3.amazonaws.com/{key}"
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=[
            ArtifactRecord(
                path=SI_DOCX_REL,
                role=ROLE_SI_DATA,
                bytes=dest.stat().st_size,
                sha256=SI_DOCX_SHA256,
                source=url,
                original_filename=SI_DOCX_SOURCE_FILENAME,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.pmc_cloud,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.pmc_cloud_object",
                    params={"key": key},
                    sha256=SI_DOCX_SHA256,
                    retrieved_at=retrieved_at,
                ),
            )
        ],
        si_data_sources=[
            url,
            f"https://www.ebi.ac.uk/pride/archive/projects/{PRIDE_ACCESSION}",
            JBEI_REGISTRY_URL,
        ],
        si_expected=[
            "si/si1.docx (publisher mmc1.docx) -- deposited; Tables S4 and S9 are the "
            "two released titer columns this loader reads out of it, and Tables S2, S5 "
            "and S8 are quoted for the strain, esterase and acetyl-CoA genotypes",
            "Table 1's 'Titer (mg/L); culture conditions' column lives in the article "
            "body, NOT in any released data file. It is carried as sourced constants "
            "quoting the sha256-pinned paper.md OCR in the torchcell-library mirror, "
            f"and the L4 level re-reads those bytes ({PAPER_MD_SHA256})",
            "the per-replicate titers and the error-bar standard deviations were NEVER "
            "released: this paper has no source-data file, and the figures are the only "
            "place the replicate spread appears. Nothing is fetchable here, which is "
            "why the uncertainty is a typed gap rather than a deferred one",
            f"PRIDE {PRIDE_ACCESSION} -- the campaign proteomics (raw DIA mass "
            "spectrometry). NOT deposited: no loader here consumes raw spectra, and no "
            "processed Top3 abundance table is released",
            f"{JBEI_REGISTRY_URL} -- the strains and plasmids are deposited in the JBEI "
            "Registry and are 'available upon request', so the sequence of each "
            "cassette is not retrievable by script; the genotype strings of Tables 1 "
            "and S2 are the authority this loader types against",
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


def verify_paper_quotes(data_root: str | None = None) -> int:
    """Assert every Table 1 quote is verbatim in the sha256-pinned ``paper.md``.

    Table 1's titer column is the one consumed column that is not in a released data
    file, so this is what makes it auditable: the OCR bytes are hashed, the hash is
    checked against :data:`PAPER_MD_SHA256`, and each row's quote must be a substring.
    Returns the number of quotes checked.
    """
    path = library_mirror_dir(data_root) / PAPER_MD
    digest = _sha256(path)
    if digest != PAPER_MD_SHA256:
        raise RuntimeError(
            f"{path} hashes {digest}, not the pinned {PAPER_MD_SHA256}; the OCR mirror "
            "moved and every Table 1 quote must be re-read before it is trusted"
        )
    text = path.read_text(encoding="utf-8")
    quotes = [row.condition_quote for row in TABLE1_ROWS]
    quotes.extend(
        STRAIN_DESCRIPTIONS[row.strain]
        for row in TABLE1_ROWS
        if STRAIN_DESCRIPTION_SOURCE[row.strain] == "table1"
    )
    quotes.append(_Q_TABLE1_FOOTNOTE)
    missing = [quote for quote in quotes if quote not in text]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(quotes)} Table 1 quotes are not verbatim in "
            f"{PAPER_MD}: {missing[:2]}"
        )
    return len(quotes)


# --------------------------------------------------------------------------- #
# The chassis background
# --------------------------------------------------------------------------- #
class ChassisDeletion(BaseModel):
    """One individually-named gene deletion of the PIPA background."""

    symbol: str
    locus_tag: str
    designation: str
    #: True when the genome's own symbol layer reproduces ``locus_tag``; False when only
    #: Table 1's locus range names it (``phaC``).
    symbol_resolves: bool


#: The six genes Table 1 names individually, each with the locus tag Table 1 gives it.
#: ``phaC`` is the one whose symbol the pinned annotation does not resolve: Table 1's
#: range ``PP_5003-5005`` is what places it, and the annotation carries PP_5005 as
#: ``phaC-II``.
CHASSIS_DELETIONS_TYPED: tuple[ChassisDeletion, ...] = (
    ChassisDeletion(
        symbol="phaA", locus_tag="PP_5003", designation="ΔphaABC", symbol_resolves=True
    ),
    ChassisDeletion(
        symbol="phaB", locus_tag="PP_5004", designation="ΔphaABC", symbol_resolves=True
    ),
    ChassisDeletion(
        symbol="phaC", locus_tag="PP_5005", designation="ΔphaABC", symbol_resolves=False
    ),
    ChassisDeletion(
        symbol="mvaB", locus_tag="PP_3540", designation="ΔmvaB", symbol_resolves=True
    ),
    ChassisDeletion(
        symbol="hbdH", locus_tag="PP_3073", designation="ΔhbdH", symbol_resolves=True
    ),
    ChassisDeletion(
        symbol="ldhA", locus_tag="PP_1649", designation="ΔldhA", symbol_resolves=True
    ),
)
#: The designation the Δ86kb loci carry, verbatim as Table S2 writes it.
SPAN_DESIGNATION = "Δ86kb (4,536,184-4,627,926; PP_4023-PP_4092)"


class SpanResolution(BaseModel):
    """The measured reconciliation of the Δ86kb cell's three statements."""

    span_length: int
    inside_span: list[str]
    tag_range: list[str]
    full_deletions: list[str]
    partial_deletions: list[str]
    span_only: list[str]
    unannotated_tag_numbers: list[int]

    #: Columns of ``preprocess/span_disagreement.csv``, so the file carries a header
    #: even when the three statements happen to agree on everything.
    DISAGREEMENT_COLUMNS: ClassVar[tuple[str, ...]] = ("item", "kind", "finding")

    def disagreement_rows(self) -> list[dict[str, Any]]:
        """One row per locus or tag number the three statements do not agree on."""
        rows: list[dict[str, Any]] = [
            {
                "item": tag,
                "kind": "locus",
                "finding": "inside the stated coordinate span but NOT in the stated "
                "locus-tag range (a later-added annotation out of tag order)",
            }
            for tag in self.span_only
        ]
        rows.extend(
            {
                "item": tag,
                "kind": "locus",
                "finding": "named by the locus-tag range and overlapping the span, but "
                "not fully inside it: the deletion truncates the gene",
            }
            for tag in self.partial_deletions
        )
        rows.extend(
            {
                "item": f"PP_{number:04d}",
                "kind": "tag_number",
                "finding": "inside the stated locus-tag range but not annotated as a "
                "gene on this assembly",
            }
            for number in self.unannotated_tag_numbers
        )
        return rows


def _tag_number(locus_tag: str) -> int | None:
    """The numeric part of a ``PP_<digits>`` tag, or None for a non-numeric tag."""
    suffix = locus_tag.removeprefix("PP_")
    return int(suffix) if suffix.isdigit() else None


def resolve_span(genome: PPutidaKT2440Genome) -> SpanResolution:
    """Measure which loci the Δ86kb cell's coordinate span and tag range each name.

    The two statements in that one cell do not describe the same locus set, and the
    label (``86kb``) matches neither: the span is 91,743 bp. The alleles this module
    types are the intersection, which is the conservative reading; everything the
    statements disagree on is reported rather than resolved by preference.
    """
    inside: list[str] = []
    partial: list[str] = []
    for tag in sorted(genome.genbank.loci):
        locus = genome[tag]
        if locus is None:
            raise RuntimeError(
                f"{tag} is listed among the assembly's loci but carries no feature; "
                "the span cannot be measured against a partial annotation"
            )
        if locus.start >= SPAN_START and locus.end <= SPAN_END:
            inside.append(tag)
        elif locus.start <= SPAN_END and locus.end >= SPAN_START:
            partial.append(tag)
    tag_range = [
        tag
        for tag in sorted(genome.genbank.loci)
        if (number := _tag_number(tag)) is not None
        and SPAN_TAG_FIRST <= number <= SPAN_TAG_LAST
    ]
    in_range = set(tag_range)
    annotated = {
        number for tag in tag_range if (number := _tag_number(tag)) is not None
    }
    return SpanResolution(
        span_length=SPAN_END - SPAN_START + 1,
        inside_span=inside,
        tag_range=tag_range,
        full_deletions=sorted(set(inside) & in_range),
        partial_deletions=sorted(tag for tag in partial if tag in in_range),
        span_only=sorted(set(inside) - in_range),
        unannotated_tag_numbers=sorted(
            set(range(SPAN_TAG_FIRST, SPAN_TAG_LAST + 1)) - annotated
        ),
    )


def _annotation_symbols(genome: PPutidaKT2440Genome) -> dict[str, str]:
    """Locus tag -> the gene symbol the pinned annotation carries for it."""
    exact, _ = genome.feature_index["symbol"]
    by_tag: dict[str, str] = {}
    for symbol, loci in exact.items():
        for locus in loci:
            by_tag.setdefault(str(locus), str(symbol))
    return by_tag


def chassis_background(
    genome: PPutidaKT2440Genome,
) -> tuple[BacterialStrainBackground, SpanResolution]:
    """PIPA: KT2440 with the six named deletions plus the Δ86kb span, and the span report.

    Every named allele's locus tag is Table 1's own, and for the five symbols the
    annotation does resolve the two must AGREE: a disagreement means either the paper's
    tag or the annotation moved, and the background refuses to be built rather than
    picking one. ``phaC`` is the documented exception and its symbol must still be
    RETIRED for the reason recorded here to stay true.
    """
    span = GenomicSpan(
        chromosome=KT2440_REPLICON,
        start=SPAN_START,
        end=SPAN_END,
        assembly=KT2440_ASSEMBLY_SET,
    )
    symbols = _annotation_symbols(genome)
    loci = set(genome.genbank.loci)
    alleles: list[BacterialBackgroundAllele] = []
    for entry in CHASSIS_DELETIONS_TYPED:
        if entry.locus_tag not in loci:
            raise RuntimeError(
                f"Table 1 places {entry.symbol!r} at {entry.locus_tag}, which is not a "
                f"locus of {KT2440_ASSEMBLY_SET}"
            )
        resolved = genome.resolve_gene_name(entry.symbol)
        if entry.symbol_resolves:
            if resolved.systematic_name != entry.locus_tag:
                raise RuntimeError(
                    f"the annotation resolves {entry.symbol!r} to "
                    f"{resolved.systematic_name} and Table 1 states {entry.locus_tag}; "
                    "the background cannot be typed while the two disagree"
                )
        elif resolved.status is not GeneNameStatus.RETIRED:
            raise RuntimeError(
                f"{entry.symbol!r} now resolves to {resolved.systematic_name} "
                f"({resolved.status}); it is documented as placed only by Table 1's "
                "locus range and must become a symbol-resolved allele"
            )
        alleles.append(
            BacterialBackgroundAllele(
                systematic_gene_name=entry.locus_tag,
                gene_namespace=KT2440_NAMESPACE,
                gene_name=entry.symbol,
                allele_name=entry.designation,
                edit=AlleleEdit.full_deletion,
                functional=False,
                provenance=[CHASSIS_GENOTYPE_TABLE1, CHASSIS_DELETIONS],
            )
        )
    named = {entry.locus_tag for entry in CHASSIS_DELETIONS_TYPED}
    report = resolve_span(genome)
    for tag, edit in [
        *((tag, AlleleEdit.full_deletion) for tag in report.full_deletions),
        *((tag, AlleleEdit.partial_deletion) for tag in report.partial_deletions),
    ]:
        if tag in named:
            continue
        alleles.append(
            BacterialBackgroundAllele(
                systematic_gene_name=tag,
                gene_namespace=KT2440_NAMESPACE,
                gene_name=symbols.get(tag, tag),
                allele_name=SPAN_DESIGNATION,
                edit=edit,
                functional=False,
                deleted_span=span,
                provenance=[CHASSIS_GENOTYPE, CHASSIS_GENOTYPE_TABLE1],
            )
        )
    background = BacterialStrainBackground(
        name=CHASSIS_STRAIN,
        reference_strain=PARENT_STRAIN,
        assembly_set=KT2440_ASSEMBLY_SET,
        parents=[PARENT_STRAIN],
        construction=(
            "in-frame deletions in P. putida KT2440 made by kanamycin-sucrose "
            "counterselection and catalytically inactive Cpf1 (Cpf1-D917N); the "
            "isoprenol-supplying pIY670 plasmid and every later cassette are recorded "
            "as perturbations rather than as part of the background"
        ),
        genotype_statement=_Q_PIPA_SI,
        alleles=alleles,
        provenance=[
            CHASSIS_GENOTYPE,
            CHASSIS_GENOTYPE_TABLE1,
            CHASSIS_SOURCE,
            CHASSIS_DELETIONS,
            PARENT_GENOTYPE,
            JBEI_ACCESSIONS,
        ],
    )
    return background, report


def chassis_reference(genome: PPutidaKT2440Genome) -> AssemblyReferenceGenome:
    """The assembly-pinned reference every record of this paper is written against."""
    background, _ = chassis_background(genome)
    return assembly_reference(PARENT_STRAIN, background=background)


# --------------------------------------------------------------------------- #
# Cassettes and the strain table
# --------------------------------------------------------------------------- #
PATHWAY_ISOPRENOL = "isoprenol via the IPP-bypass mevalonate pathway"
PATHWAY_ESTER = "isoprenyl acetate from isoprenol and acetyl-CoA"
PATHWAY_XYLOSE = "xylose isomerase pathway into the pentose phosphate pathway"
PATHWAY_ACETYL_COA = (
    "auxiliary acetyl-CoA supply (acetate assimilation + phosphoketolase)"
)


class CassetteGene(BaseModel):
    """One heterologous gene of a cassette, with its sourced organism and promoter."""

    token: str
    symbol: str
    organism: str
    promoter: str
    variant: str | None = None
    #: The substring of the cassette's construct string that carries this gene, when it
    #: is not the token itself. ``xylA`` and ``xylB`` share the operon shorthand
    #: ``xylAB``, which is how the source writes the two adjacent genes.
    construct_token: str | None = None
    #: The quote that states the organism, so a changed mirror cannot drift from it.
    organism_quote: str
    #: Which mirror the organism quote lives in.
    organism_source: Literal["paper", "si", "carruthers"]


class Cassette(BaseModel):
    """A plasmid or an integrated operon, and the genes it carries."""

    name: str
    construct_string: str
    pathway: str
    localization: Literal["episomal_plasmid", "chromosomal_integration"]
    integration_locus: str | None
    genes: tuple[CassetteGene, ...]


PIY670 = Cassette(
    name="pIY670",
    construct_string=_Q_PIY670_PARTS,
    pathway=PATHWAY_ISOPRENOL,
    localization="episomal_plasmid",
    integration_locus=None,
    genes=(
        CassetteGene(
            token="MvaSef",
            symbol="MvaS",
            organism="Enterococcus faecalis",
            promoter="PBAD",
            organism_quote=_Q_PIY670_PARTS,
            organism_source="si",
        ),
        CassetteGene(
            token="MvaEef",
            symbol="MvaE",
            organism="Enterococcus faecalis",
            promoter="PBAD",
            organism_quote=_Q_PIY670_PARTS,
            organism_source="si",
        ),
        CassetteGene(
            token="MKmm",
            symbol="MK",
            organism="Methanosarcina mazei",
            promoter="Ptrc1-O",
            organism_quote=_Q_PIY670_PARTS,
            organism_source="si",
        ),
        CassetteGene(
            token="PMDHKQ",
            symbol="PMD",
            organism="Saccharomyces cerevisiae",
            promoter="Ptrc1-O",
            variant="HKQ",
            organism_quote=_Q_PMD_CARRUTHERS,
            organism_source="carruthers",
        ),
        CassetteGene(
            token="AphA",
            symbol="AphA",
            organism="Escherichia coli",
            promoter="Ptrc1-O",
            organism_quote=_Q_APHA,
            organism_source="paper",
        ),
    ),
)
"""The isoprenol-supplying plasmid every record carries.

The organism of ``MvaSef``, ``MvaEef`` and ``MKmm`` is read from the part token's own
suffix in the quoted plasmid description (``ef`` = *Enterococcus faecalis*, ``mm`` =
*Methanosarcina mazei*); Kang never writes those organisms in prose. ``PMDHKQ`` carries
no organism suffix and no Kang statement at all, so it is sourced by DEFERRAL to the
mirrored Carruthers 2025 campaign, which carries the same pIY670 and names the
decarboxylase's organism. ``AphA`` Kang states itself.
"""

AAT_PROMOTER = "PnagAa"


def _aat_cassette(
    plasmid: str, token: str, organism: str, quote: str, variant: str | None = None
) -> Cassette:
    """One alcohol-acyltransferase plasmid of the Table S4 panel."""
    return Cassette(
        name=plasmid,
        construct_string=f"pRSF1010-Gm-NagR-{AAT_PROMOTER}-{token}-rrnB T1",
        pathway=PATHWAY_ESTER,
        localization="episomal_plasmid",
        integration_locus=None,
        genes=(
            CassetteGene(
                token=token,
                symbol=token,
                organism=organism,
                promoter=AAT_PROMOTER,
                variant=variant,
                organism_quote=quote,
                organism_source="paper",
            ),
        ),
    )


#: The five AAT plasmids, keyed by the Table S4 enzyme name. Each organism is stated in
#: Methods 2.1; ``CAT`` additionally carries the Y20F mutation the same sentence states.
AAT_CASSETTES: dict[str, Cassette] = {
    "ATF1": _aat_cassette("pAAT1", "ATF1", "Saccharomyces cerevisiae", _Q_ATF),
    "ATF2": _aat_cassette("pAAT2", "ATF2", "Saccharomyces cerevisiae", _Q_ATF),
    "SAAT": _aat_cassette("pAAT3", "SAAT", "Fragaria x ananassa", _Q_SAAT),
    "AAT": _aat_cassette("pAAT4", "AAT", "Rosa hybrida", _Q_AAT_CAT),
    "CAT": _aat_cassette(
        "pAAT5", "CAT", "Escherichia coli", _Q_AAT_CAT, variant="Y20F"
    ),
}

#: The integration site each chromosomal cassette replaces, with the locus tag Table 1
#: gives it and the gene symbol the annotation carries.
XYL_INTEGRATION_LOCUS = "PP_1444"
ACS_XFPK_INTEGRATION_LOCUS = "PP_2876"

XYL_CASSETTE_SPEC = Cassette(
    name="pXyl",
    construct_string="PxylE*-xylE:Ptac-xylAB:talB:tktA",
    pathway=PATHWAY_XYLOSE,
    localization="chromosomal_integration",
    integration_locus=XYL_INTEGRATION_LOCUS,
    genes=(
        CassetteGene(
            token="xylE",
            symbol="xylE",
            organism="Escherichia coli",
            promoter="PxylE*",
            organism_quote=_Q_XYL_GENES,
            organism_source="paper",
        ),
        CassetteGene(
            token="xylA",
            symbol="xylA",
            organism="Escherichia coli",
            promoter="Ptac",
            construct_token="xylAB",
            organism_quote=_Q_XYL_GENES,
            organism_source="paper",
        ),
        CassetteGene(
            token="xylB",
            symbol="xylB",
            organism="Escherichia coli",
            promoter="Ptac",
            construct_token="xylAB",
            organism_quote=_Q_XYL_GENES,
            organism_source="paper",
        ),
        CassetteGene(
            token="talB",
            symbol="tal",
            organism="Escherichia coli",
            promoter="Ptac",
            organism_quote=_Q_XYL_GENES,
            organism_source="paper",
        ),
        CassetteGene(
            token="tktA",
            symbol="tkt",
            organism="Escherichia coli",
            promoter="Ptac",
            organism_quote=_Q_XYL_GENES,
            organism_source="paper",
        ),
    ),
)

ACS_XFPK_CASSETTE_SPEC = Cassette(
    name="pO15",
    construct_string="Pm-acsEc-xfpkBa_RBS1",
    pathway=PATHWAY_ACETYL_COA,
    localization="chromosomal_integration",
    integration_locus=ACS_XFPK_INTEGRATION_LOCUS,
    genes=(
        CassetteGene(
            token="acsEc",
            symbol="acs",
            organism="Escherichia coli",
            promoter="Pm",
            organism_quote=_Q_ACS_EC,
            organism_source="paper",
        ),
        CassetteGene(
            token="xfpkBa",
            symbol="xfpk",
            organism="Bifidobacterium adolescentis",
            promoter="Pm",
            variant="RBS1",
            organism_quote=_Q_XFPK,
            organism_source="paper",
        ),
    ),
)

#: The chromosomal deletions each strain adds on top of PIPA, with the locus tag Table 1
#: gives it and the gene symbol the source itself uses (empty when the source names only
#: the tag, in which case the annotation's symbol is used).
DELETION_SYMBOLS: dict[str, str] = {
    "PP_1127": "",
    "PP_3812": "",
    "PP_4218": "",
    XYL_INTEGRATION_LOCUS: "gcd",
    "PP_5292": "crc",
    "PP_1021": "hexR",
    ACS_XFPK_INTEGRATION_LOCUS: "ampC",
}
#: Which quote states each deletion.
DELETION_QUOTES: dict[str, str] = {
    "PP_1127": _Q_ESTERASES,
    "PP_3812": _Q_ESTERASES,
    "PP_4218": _Q_ESTERASES,
    XYL_INTEGRATION_LOCUS: _Q_XYL_CASSETTE,
    "PP_5292": STRAIN_DESCRIPTIONS["PIPAxyl-E3-K3"],
    "PP_1021": STRAIN_DESCRIPTIONS["PIPAxyl-E3-K3"],
    ACS_XFPK_INTEGRATION_LOCUS: STRAIN_DESCRIPTIONS["PIPAxyl-E3-O15"],
}
ESTERASE_DELETIONS: tuple[str, ...] = ("PP_1127", "PP_3812", "PP_4218")
REGULATOR_DELETIONS: tuple[str, ...] = ("PP_5292", "PP_1021")


class StrainSpec(BaseModel):
    """One engineered strain: what it deletes and which cassettes it carries."""

    name: str
    deletions: tuple[str, ...]
    cassettes: tuple[str, ...]


#: Every strain this loader builds a record for, by name.
STRAIN_SPECS: dict[str, StrainSpec] = {
    "PIPA-AAT1": StrainSpec(
        name="PIPA-AAT1", deletions=(), cassettes=("pIY670", "pAAT1")
    ),
    "PIPA-AAT2": StrainSpec(
        name="PIPA-AAT2", deletions=(), cassettes=("pIY670", "pAAT2")
    ),
    "PIPA-AAT3": StrainSpec(
        name="PIPA-AAT3", deletions=(), cassettes=("pIY670", "pAAT3")
    ),
    "PIPA-AAT4": StrainSpec(
        name="PIPA-AAT4", deletions=(), cassettes=("pIY670", "pAAT4")
    ),
    "PIPA-AAT5": StrainSpec(
        name="PIPA-AAT5", deletions=(), cassettes=("pIY670", "pAAT5")
    ),
    "PIPA-E3": StrainSpec(
        name="PIPA-E3", deletions=ESTERASE_DELETIONS, cassettes=("pIY670", "pAAT1")
    ),
    "PIPAxyl-AAT1": StrainSpec(
        name="PIPAxyl-AAT1",
        deletions=(XYL_INTEGRATION_LOCUS,),
        cassettes=("pXyl", "pIY670", "pAAT1"),
    ),
    "PIPAxyl-E3": StrainSpec(
        name="PIPAxyl-E3",
        deletions=(XYL_INTEGRATION_LOCUS, *ESTERASE_DELETIONS),
        cassettes=("pXyl", "pIY670", "pAAT1"),
    ),
    "PIPAxyl-E3-K3": StrainSpec(
        name="PIPAxyl-E3-K3",
        deletions=(XYL_INTEGRATION_LOCUS, *ESTERASE_DELETIONS, *REGULATOR_DELETIONS),
        cassettes=("pXyl", "pIY670", "pAAT1"),
    ),
    "PIPAxyl-E3-O15": StrainSpec(
        name="PIPAxyl-E3-O15",
        deletions=(
            XYL_INTEGRATION_LOCUS,
            *ESTERASE_DELETIONS,
            ACS_XFPK_INTEGRATION_LOCUS,
        ),
        cassettes=("pXyl", "pO15", "pIY670", "pAAT1"),
    ),
    "PIPAxyl-E3-K3-O15": StrainSpec(
        name="PIPAxyl-E3-K3-O15",
        deletions=(
            XYL_INTEGRATION_LOCUS,
            *ESTERASE_DELETIONS,
            *REGULATOR_DELETIONS,
            ACS_XFPK_INTEGRATION_LOCUS,
        ),
        cassettes=("pXyl", "pO15", "pIY670", "pAAT1"),
    ),
}

#: Every cassette by name, so a strain spec names one and only one object.
CASSETTES: dict[str, Cassette] = {
    PIY670.name: PIY670,
    XYL_CASSETTE_SPEC.name: XYL_CASSETTE_SPEC,
    ACS_XFPK_CASSETTE_SPEC.name: ACS_XFPK_CASSETTE_SPEC,
    **{cassette.name: cassette for cassette in AAT_CASSETTES.values()},
}

#: The Table 1 strains that carry no titer, so they are not records.
STRAINS_WITHOUT_A_TITER: tuple[tuple[str, str], ...] = (
    ("PIPAxyl", "the xylose-pathway intermediate; Table 1 gives it no titer"),
    (
        "PIPAxyl-O2",
        "plasmid-borne rpe overexpression screen (Fig. 7b), no Table 1 titer",
    ),
    (
        "PIPAxyl-O6",
        "plasmid-borne acs overexpression screen (Fig. 7b), no Table 1 titer",
    ),
    (
        "PIPAxyl-O7",
        "plasmid-borne xfpk overexpression screen (Fig. 7b), no Table 1 titer",
    ),
    (
        "PIPAxyl-E3-O14",
        "the non-RBS-optimized integration variant; the OCR renders its name '014' and "
        "Table 1 gives it no titer",
    ),
)


# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #
def publication() -> Publication:
    """This paper, by DOI (no PubMed id is recorded in the mirror's manifest)."""
    return Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")


def _dose(
    compound: str, value: float, unit: ConcentrationUnit
) -> SmallMoleculePerturbation:
    """One added small molecule at a stated dose, through the shared compound layer."""
    return SmallMoleculePerturbation(
        compound=resolved_compound(compound),
        concentration=Concentration(value=value, unit=unit),
    )


def _carbon(compound: str, g_per_l: float) -> EnvironmentPhysicalPerturbation:
    """One sugar of the medium's carbon source, as a typed physical factor.

    The three Kang media are carbon-source FREE by construction (``media.py`` records
    that the sugar is the variable), so each sugar is a ``carbon_source`` factor naming
    the molecule it is realized by.
    """
    return EnvironmentPhysicalPerturbation(
        factor=PhysicalFactor.carbon_source,
        magnitude=Concentration(value=g_per_l, unit=ConcentrationUnit.g_per_l),
        agent=resolved_compound(compound),
    )


class ConditionSpec(BaseModel):
    """One stated culture condition: the medium, the sugars and the additions."""

    key: str
    medium_label: Literal["M9", "modified M9", "M9-MOPS"]
    culture_format: Literal["tube", "flask", "bioreactor"]
    glucose_g_per_l: float
    xylose_g_per_l: float | None
    arabinose_g_per_l: float
    salicylic_acid_um: float
    toluic_acid_um: float | None
    overlay: Literal["dodecane", "Durasyn 164"]
    yeast_extract_g_per_l: float | None
    ammonium_sulfate_mm: float | None
    pulse_at_48h: bool
    duration_hours: float | None
    quote: str


_MEDIA_BY_LABEL: dict[str, Media] = {
    "M9": M9_NREL_KANG2026,
    "modified M9": M9_NREL_HIGH_N_KANG2026,
    "M9-MOPS": M9_MOPS_KANG2026,
}

#: The two-strain and five-strain condition sets. ``duration_hours`` is set only where
#: the source states a time; a Table 1 maximum over a 24 h-sampled flask time course
#: states none, and that is a typed gap on the environment.
CONDITIONS: dict[str, ConditionSpec] = {
    "tube_m9_glc20": ConditionSpec(
        key="tube_m9_glc20",
        medium_label="M9",
        culture_format="tube",
        glucose_g_per_l=20.0,
        xylose_g_per_l=None,
        arabinose_g_per_l=2.0,
        salicylic_acid_um=125.0,
        toluic_acid_um=None,
        overlay="dodecane",
        yeast_extract_g_per_l=None,
        ammonium_sulfate_mm=None,
        pulse_at_48h=False,
        duration_hours=48.0,
        quote=_Q_TUBE_FLASK,
    ),
    "flask_m9_glc20": ConditionSpec(
        key="flask_m9_glc20",
        medium_label="M9",
        culture_format="flask",
        glucose_g_per_l=20.0,
        xylose_g_per_l=None,
        arabinose_g_per_l=2.0,
        salicylic_acid_um=125.0,
        toluic_acid_um=None,
        overlay="dodecane",
        yeast_extract_g_per_l=None,
        ammonium_sulfate_mm=None,
        pulse_at_48h=False,
        duration_hours=None,
        quote="414; M9, Glc 20 g/L, Ara 2 g/L SA 125 μM at OD600 0.60.8, Dod 20% (v/v)",
    ),
    "flask_m9hn_glcxyl_1010": ConditionSpec(
        key="flask_m9hn_glcxyl_1010",
        medium_label="modified M9",
        culture_format="flask",
        glucose_g_per_l=10.0,
        xylose_g_per_l=10.0,
        arabinose_g_per_l=2.0,
        salicylic_acid_um=125.0,
        toluic_acid_um=None,
        overlay="dodecane",
        yeast_extract_g_per_l=None,
        ammonium_sulfate_mm=None,
        pulse_at_48h=False,
        duration_hours=None,
        quote="181; M9, Glc/Xyl 10/10 g/L, (NH4)2SO4 40 mM, Ara 2 g/",
    ),
    "flask_m9hn_glcxyl_1010_pulse": ConditionSpec(
        key="flask_m9hn_glcxyl_1010_pulse",
        medium_label="modified M9",
        culture_format="flask",
        glucose_g_per_l=10.0,
        xylose_g_per_l=10.0,
        arabinose_g_per_l=2.0,
        salicylic_acid_um=125.0,
        toluic_acid_um=None,
        overlay="dodecane",
        yeast_extract_g_per_l=None,
        ammonium_sulfate_mm=None,
        pulse_at_48h=True,
        duration_hours=144.0,
        quote=(
            "599; M9, Glc/Xyl 10/10 g/L + pulse, (NH4)2SO4 40 mM, Ara 2 g/L SA 125 μM "
            "at OD600 0.60.8; Dod 20% (v/v)"
        ),
    ),
    "flask_mops_ye_glcxyl_1010_pulse": ConditionSpec(
        key="flask_mops_ye_glcxyl_1010_pulse",
        medium_label="M9-MOPS",
        culture_format="flask",
        glucose_g_per_l=10.0,
        xylose_g_per_l=10.0,
        arabinose_g_per_l=2.0,
        salicylic_acid_um=62.5,
        toluic_acid_um=250.0,
        overlay="Durasyn 164",
        yeast_extract_g_per_l=1.0,
        ammonium_sulfate_mm=40.0,
        pulse_at_48h=True,
        duration_hours=None,
        quote=(
            "819; M9-MOPS + YE, Glc/Xyl 10/10 g/L + pulse, (NH4)2SO4 40 mM, Ara 2 g/L "
            "at 0 h, SA 62.5 µM 3 MB 250"
        ),
    ),
    "flask_mops_ye_glcxyl_137_pulse": ConditionSpec(
        key="flask_mops_ye_glcxyl_137_pulse",
        medium_label="M9-MOPS",
        culture_format="flask",
        glucose_g_per_l=13.0,
        xylose_g_per_l=7.0,
        arabinose_g_per_l=2.0,
        salicylic_acid_um=62.5,
        toluic_acid_um=250.0,
        overlay="Durasyn 164",
        yeast_extract_g_per_l=1.0,
        ammonium_sulfate_mm=40.0,
        pulse_at_48h=True,
        duration_hours=None,
        quote=(
            "1500; M9-MOPS + YE, Glc/Xyl 13/7 g/L + pulse, (NH4)2SO4 40 mM, Ara 2 g/L "
            "at 0 h, SA 62.5 µM 3 MB 250"
        ),
    ),
    "fedbatch_mops_ye_glcxyl_1337": ConditionSpec(
        key="fedbatch_mops_ye_glcxyl_1337",
        medium_label="M9-MOPS",
        culture_format="bioreactor",
        glucose_g_per_l=13.3,
        xylose_g_per_l=6.7,
        arabinose_g_per_l=2.0,
        salicylic_acid_um=62.5,
        toluic_acid_um=250.0,
        overlay="Durasyn 164",
        yeast_extract_g_per_l=1.0,
        ammonium_sulfate_mm=None,
        pulse_at_48h=False,
        duration_hours=None,
        quote=_Q_FEDBATCH_MEDIUM,
    ),
}

#: Why ``duration_hours`` is None on a Table 1 record, and why the fed-batch medium is
#: the plain M9-MOPS object.
_DURATION_GAP_NOTE = (
    "the primary titer column is a MAXIMUM over a flask time course sampled every 24 h "
    f"('{_Q_TABLE1_FOOTNOTE}'), and Table 1 states the time for one strain only "
    "(PIPAxyl-E3-K3, 144 h); no time is released for the others"
)
_FEDBATCH_MEDIUM_NOTE = (
    "Methods 2.8 calls the fed-batch medium 'modified M9-MOPS' and never defines it; "
    "'modified M9' is defined for the NREL-type M9 only (ammonium sulfate raised to 40 "
    "mM). The served medium is therefore the plain M9-MOPS object and no raised "
    "ammonium level is asserted for this condition"
)


def environment(
    spec: ConditionSpec, *, duration_hours: float | None = None
) -> Environment:
    """The environment of one stated condition.

    A plain ``Environment``, not a ``CultureEnvironment``:
    ``ProductTiterExperiment.environment`` is annotated ``Environment`` and pydantic
    serializes by the DECLARED type, so a vessel or working volume would be dumped away
    without an error. They are carried in :data:`CULTURE_FORMATS` instead.
    """
    perturbations: list[Any] = [_carbon("D-glucose", spec.glucose_g_per_l)]
    if spec.xylose_g_per_l is not None:
        perturbations.append(_carbon("xylose", spec.xylose_g_per_l))
    if spec.ammonium_sulfate_mm is not None:
        perturbations.append(
            _dose(
                "ammonium sulfate",
                spec.ammonium_sulfate_mm,
                ConcentrationUnit.millimolar,
            )
        )
    if spec.yeast_extract_g_per_l is not None:
        perturbations.append(
            _dose(
                "yeast extract", spec.yeast_extract_g_per_l, ConcentrationUnit.g_per_l
            )
        )
    perturbations.append(
        _dose("L-arabinose", spec.arabinose_g_per_l, ConcentrationUnit.g_per_l)
    )
    perturbations.append(
        _dose("salicylic acid", spec.salicylic_acid_um, ConcentrationUnit.micromolar)
    )
    if spec.toluic_acid_um is not None:
        perturbations.append(
            _dose("m-toluic acid", spec.toluic_acid_um, ConcentrationUnit.micromolar)
        )
    perturbations.append(_dose(spec.overlay, 20.0, ConcentrationUnit.percent_v_v))
    hours = spec.duration_hours if duration_hours is None else duration_hours
    gaps: list[ProvenanceGap] = []
    if hours is None:
        gaps.append(
            ProvenanceGap(
                field="duration_hours",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=_DURATION_GAP_NOTE,
            )
        )
    return Environment(
        media=_MEDIA_BY_LABEL[spec.medium_label],
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        perturbations=perturbations,
        aerobicity=str(AEROBICITY.value),
        duration_hours=hours,
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# Phenotype
# --------------------------------------------------------------------------- #
def isoprenyl_acetate() -> Compound:
    """The product as a typed ``Compound`` through the shared compound-identity layer.

    ``isoprenyl acetate`` (3-methyl-3-buten-1-yl acetate) has no row in the committed
    ``compound_identity_table.json``, so the resolver returns the honest typed absence:
    the canonical name with a ``ProvenanceGap`` on ``inchikey``. The key the curation
    will fill is :data:`ISOPRENYL_ACETATE_INCHIKEY`; curating the row needs a PubChem
    call against the committed input lists and is a human act, so it is raised in the PR
    rather than done here.
    """
    return resolved_compound(PRODUCT_NAME)


def titer_phenotype(
    titer_mg_per_l: float, *, n_samples: int | None, product_yield: float | None = None
) -> ProductTiterPhenotype:
    """One released titer in ``ug/mL``, with the typed absences the paper forces.

    The released numbers are mg/L and 1 mg/L is exactly 1 ug/mL, so no arithmetic is
    applied to a source value. ``titer_uncertainty`` and ``titer_uncertainty_type`` are
    ALWAYS gaps: the design is sourced (biological triplicates, sample SD) but no SD
    number is released anywhere in the mirror, and the schema forbids a type without a
    value. ``n_samples`` is None only for the fed-batch rows, whose Methods describe one
    bioreactor run and state no replicate count.
    """
    gaps = [
        ProvenanceGap(
            field="titer_uncertainty",
            reason=ProvenanceGapReason.not_reported_by_primary,
            note="the replicate SPREAD is released only as error bars: the paper has no "
            f"source-data file and no SD column. The design is sourced ('{_Q_TRIPLICATE}'"
            ") and is carried in n_samples + sample_unit",
        ),
        ProvenanceGap(
            field="titer_uncertainty_type",
            reason=ProvenanceGapReason.not_reported_by_primary,
            note="the type the released error bars WOULD carry is a sample SD over "
            f"biological triplicates ('{_Q_TRIPLICATE_FIG2}'); ProductTiterPhenotype "
            "forbids a type without a value, so it is a typed absence rather than a "
            "half-filled pair",
        ),
        ProvenanceGap(
            field="productivity",
            reason=ProvenanceGapReason.not_reported_by_primary,
            note="no volumetric productivity is released for any strain or time point",
        ),
        ProvenanceGap(
            field="productivity_unit",
            reason=ProvenanceGapReason.not_reported_by_primary,
            note="no volumetric productivity is released for any strain or time point",
        ),
    ]
    if n_samples is None:
        gaps.extend(
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="Methods 2.8 describes a single fed-batch run in a 2 L bioreactor "
                "and never states a replicate count for it; the triplicate design is "
                "stated for the tube and flask figures only",
            )
            for field in ("n_samples", "sample_unit")
        )
    if product_yield is None:
        gaps.extend(
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="a yield on substrate is released once, for the fed-batch run's "
                f"maximum titer ('{_Q_FEDBATCH_TITER}'), and for no other record",
            )
            for field in ("product_yield", "product_yield_unit")
        )
    return ProductTiterPhenotype(
        product=isoprenyl_acetate(),
        titer=titer_mg_per_l,
        titer_unit=ConcentrationUnit.ug_per_ml,
        n_samples=n_samples,
        sample_unit=SampleUnit.biological_replicate if n_samples is not None else None,
        product_yield=product_yield,
        product_yield_unit=(
            ProductYieldUnit.g_per_g_substrate if product_yield is not None else None
        ),
        quantification_method=str(QUANTIFICATION.value),
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# The three titer columns
# --------------------------------------------------------------------------- #
class Table1Row(BaseModel):
    """One row of Table 1's ``Titer (mg/L); culture conditions`` column."""

    strain: str
    titer_mg_per_l: float
    condition: str
    #: The titer cell verbatim, exactly as the pinned ``paper.md`` OCR renders it.
    condition_quote: str


#: Table 1's titer column, in table order. Each ``condition_quote`` is the cell the
#: number and its conditions are read from; three of them (PIPAxyl-E3,
#: PIPAxyl-E3-O15, PIPAxyl-E3-K3-O15) are the FIRST of two OCR fragments of one cell,
#: because MinerU split the wrapped cell across two table rows. The fragment quoted is
#: always the one carrying the number.
TABLE1_ROWS: tuple[Table1Row, ...] = (
    Table1Row(
        strain="PIPA-AAT1",
        titer_mg_per_l=414.0,
        condition="flask_m9_glc20",
        condition_quote=(
            "414; M9, Glc 20 g/L, Ara 2 g/L SA 125 μM at OD600 0.60.8, Dod 20% (v/v)"
        ),
    ),
    Table1Row(
        strain="PIPA-E3",
        titer_mg_per_l=724.0,
        condition="flask_m9_glc20",
        condition_quote=(
            "724; M9, Glc 20 g/L, Ara 2 g/L SA 125 μM at OD600 0.60.8; Dod 20% (v/v)"
        ),
    ),
    Table1Row(
        strain="PIPAxyl-AAT1",
        titer_mg_per_l=181.0,
        condition="flask_m9hn_glcxyl_1010",
        condition_quote=(
            "181; M9, Glc/Xyl 10/10 g/L, (NH4)2SO4 40 mM, Ara 2 g/ L SA 125 μM at "
            "OD600 0.60.8; Dod 20% (v/v)"
        ),
    ),
    Table1Row(
        strain="PIPAxyl-E3",
        titer_mg_per_l=383.0,
        condition="flask_m9hn_glcxyl_1010",
        condition_quote="383; M9, Glc/Xyl 10/10 g/L, (NH4)2SO4 40 mM, Ara 2 g/",
    ),
    Table1Row(
        strain="PIPAxyl-E3-K3",
        titer_mg_per_l=599.0,
        condition="flask_m9hn_glcxyl_1010_pulse",
        condition_quote=(
            "599; M9, Glc/Xyl 10/10 g/L + pulse, (NH4)2SO4 40 mM, Ara 2 g/L SA 125 μM "
            "at OD600 0.60.8; Dod 20% (v/v)"
        ),
    ),
    Table1Row(
        strain="PIPAxyl-E3-O15",
        titer_mg_per_l=819.0,
        condition="flask_mops_ye_glcxyl_1010_pulse",
        condition_quote=(
            "819; M9-MOPS + YE, Glc/Xyl 10/10 g/L + pulse, (NH4)2SO4 40 mM, Ara 2 g/L "
            "at 0 h, SA 62.5 µM 3 MB 250"
        ),
    ),
    Table1Row(
        strain="PIPAxyl-E3-K3-O15",
        titer_mg_per_l=1500.0,
        condition="flask_mops_ye_glcxyl_137_pulse",
        condition_quote=(
            "1500; M9-MOPS + YE, Glc/Xyl 13/7 g/L + pulse, (NH4)2SO4 40 mM, Ara 2 g/L "
            "at 0 h, SA 62.5 µM 3 MB 250"
        ),
    ),
)

#: The column header of each consumed SI table, asserted before any value is read.
SI_TABLE4_COLUMN = "IPA titer (mg/L)"
SI_TABLE9_COLUMNS: tuple[str, ...] = (
    "Time (h)",
    "Glucose (g/L)",
    "Xylose (g/L)",
    "Total sugar (g/L)",
    "Isoprenol, aqueous (mg/L)",
    "Isoprenyl acetate, aqueous (mg/L)",
    "Isoprenyl acetate, organic (mg/L)",
    "Isoprenyl acetate, off-gas (mg/L)",
    "Off-gas fraction (%)",
)
#: A comma-free label for the fed-batch titer column, so the build ledger CSV reads
#: cleanly; the three source headers it sums are :data:`SI_TABLE9_PHASES`.
SI_TABLE9_SUM_LABEL = "Isoprenyl acetate aqueous + organic + off-gas (mg/L)"
#: How far three numbers each rounded to one decimal can disagree on a sum (0.05 each).
_SUGAR_ROUNDING_TOL = 0.15
#: The three phases of Table S9 whose sum is the fed-batch titer.
SI_TABLE9_PHASES: tuple[str, ...] = (
    "Isoprenyl acetate, aqueous (mg/L)",
    "Isoprenyl acetate, organic (mg/L)",
    "Isoprenyl acetate, off-gas (mg/L)",
)


#: Every quote that states part of one culture condition, so the ``Environment`` a
#: record carries is auditable quote by quote. Written to
#: ``preprocess/condition_provenance.json`` at build time.
AAT_PANEL_COLUMN = _si(SI_TABLE4_COLUMN, SI_TABLE4_COLUMN, page=_SI_TABLE4)
FEDBATCH_COLUMNS = _si(
    SI_TABLE9_PHASES,
    SI_TABLE9_PHASES[1],
    page=_SI_TABLE9,
    note="the three released phases whose sum is the stored fed-batch titer",
)
ESTERASE_FUNCTIONS = _si(
    {
        "PP_1127": "Carboxylesterase - EstC",
        "PP_3812": "Lipase",
        "PP_4218": "Lipase/esterase family protein",
    },
    "Carboxylesterase - EstC",
    page=_SI_TABLE5,
    note="the predicted function Table S5 gives each of the three deleted esterases; "
    "the deletions themselves are stated in the Results and in Table 1",
)
ACETYL_COA_GENES = _si(
    {
        "acs": "Convert acetate to acetyl-CoA",
        "xfpk": "Enable carbon-efficient acetyl-CoA synthesis from sugar phosphates",
    },
    "Enable carbon-efficient acetyl-CoA synthesis from sugar phosphates",
    page=_SI_TABLE8,
    note="why the integrated pO15 operon carries these two genes",
)
IPP_BYPASS_GENES = _paper(
    ("mvaS", "mvaE", "MK_Mm", "PMD_HKQ", "aphA"),
    _Q_IPP_BYPASS,
    page="Results 3.1, 'Production of isoprenyl acetate in P. putida'",
    note="the five pIY670 genes named in prose; the plasmid part string gives the same "
    "five in cassette order",
)
AAT_IDENTIFIERS = _paper(
    {
        "ATF1": "NCBI: NP_015022.3",
        "ATF2": "NCBI: NP_011693.1",
        "SAAT": "GenBank: AAG13130.1",
        "AAT": "GenBank: AY850287.1",
        "CAT": "UniProtKB: P00484",
    },
    _Q_ATF,
    page=_METHODS_STRAINS,
    note="Methods 2.1 states each AAT's accession and source organism; Table S4 repeats "
    "the same identifiers in its 'Source and identifier' column",
)
INDUCTION = _paper(
    "inducers and a 20% (v/v) dodecane overlay added at OD600 0.6-0.8",
    _Q_INDUCTION,
    page=_METHODS_PRODUCTION,
)
YEAST_EXTRACT_G_PER_L = _paper(1.0, _Q_YEAST_EXTRACT, page="Fig. 7 caption")
FEDBATCH_INDUCTION = _paper(
    {"L-arabinose_g_per_l": 2.0, "salicylic_acid_um": 62.5, "m_toluic_acid_um": 250.0},
    _Q_FEDBATCH_INDUCTION,
    page=_METHODS_FEDBATCH,
)
FEDBATCH_OVERLAY = _paper(
    {"overlay": "Durasyn 164", "percent_v_v": 20.0},
    _Q_FEDBATCH_OVERLAY,
    page=_METHODS_FEDBATCH,
)
FEDBATCH_FEED = _paper(
    {
        "glucose_g_per_l": 266.7,
        "xylose_g_per_l": 133.3,
        "yeast_extract_g_per_l": [1.0, 5.0],
    },
    _Q_FEDBATCH_FEED,
    page=_METHODS_FEDBATCH,
    note="the CONCENTRATED feed, not the batch medium; it is continuously fed, so no "
    "Environment slot holds it and the record carries the initial sugars only",
)

CONDITION_PROVENANCE: dict[str, tuple[SourcedValue, ...]] = {
    "tube_m9_glc20": (MEDIUM_SOURCES["M9"], TUBE_FORMAT, INDUCTION, AAT_PANEL_COLUMN),
    "flask_m9_glc20": (MEDIUM_SOURCES["M9"], FLASK_FORMAT, INDUCTION, TABLE1_RULE),
    "flask_m9hn_glcxyl_1010": (
        MEDIUM_SOURCES["modified M9"],
        FLASK_FORMAT,
        INDUCTION,
        TABLE1_RULE,
    ),
    "flask_m9hn_glcxyl_1010_pulse": (
        MEDIUM_SOURCES["modified M9"],
        FLASK_FORMAT,
        INDUCTION,
        TABLE1_RULE,
        PULSE_RULE,
    ),
    "flask_mops_ye_glcxyl_1010_pulse": (
        MEDIUM_SOURCES["M9-MOPS"],
        FLASK_FORMAT,
        YEAST_EXTRACT_G_PER_L,
        TABLE1_RULE,
        PULSE_RULE,
    ),
    "flask_mops_ye_glcxyl_137_pulse": (
        MEDIUM_SOURCES["M9-MOPS"],
        FLASK_FORMAT,
        YEAST_EXTRACT_G_PER_L,
        TABLE1_RULE,
        PULSE_RULE,
    ),
    "fedbatch_mops_ye_glcxyl_1337": (
        MEDIUM_SOURCES["M9-MOPS"],
        BIOREACTOR_FORMAT,
        FEDBATCH_INDUCTION,
        FEDBATCH_OVERLAY,
        FEDBATCH_FEED,
        FEDBATCH_COLUMNS,
    ),
}


class AatPanelRow(BaseModel):
    """One row of Table S4: an AAT enzyme, its strain and its tube titer."""

    enzyme: str
    strain: str
    organism: str
    identifier: str
    titer_mg_per_l: float


def read_aat_panel(path: str | Path) -> list[AatPanelRow]:
    """Table S4's ``IPA titer (mg/L)`` column, joined to the PIPA-AAT strain names.

    The table names the ENZYME; Table S2 names the plasmid that carries it
    (``pAAT1``-``pAAT5``) and the strain that carries the plasmid, and the Results name
    the join explicitly for ATF1 ("the PIPA-AAT1 strain (i.e., PIPA harboring pIY670 and
    pAAT1) expressing ATF1 from S. cerevisiae"). The plasmid order in Table S2 is the
    enzyme order of Table S4, so row n is ``pAAT<n>`` and strain ``PIPA-AAT<n>``.
    """
    table = si_table(path, 4)
    titer = table.column(SI_TABLE4_COLUMN)
    origin = table.column("Biological origin")
    identifier = table.column("Source and identifier")
    rows: list[AatPanelRow] = []
    for index, row in enumerate(table.rows[1:], start=1):
        enzyme = row[0].strip()
        if enzyme not in AAT_CASSETTES:
            raise RuntimeError(
                f"Table S4 row {index} names the enzyme {enzyme!r}, which is not one of "
                f"the five this module types ({sorted(AAT_CASSETTES)})"
            )
        cassette = AAT_CASSETTES[enzyme]
        if cassette.name != f"pAAT{index}":
            raise RuntimeError(
                f"Table S4 row {index} is {enzyme!r} -> {cassette.name}; the Table S2 "
                "plasmid order no longer matches the Table S4 enzyme order"
            )
        rows.append(
            AatPanelRow(
                enzyme=enzyme,
                strain=f"PIPA-AAT{index}",
                organism=row[origin].strip(),
                identifier=row[identifier].strip(),
                titer_mg_per_l=float(row[titer]),
            )
        )
    return rows


class FedBatchRow(BaseModel):
    """One sampled time of Table S9, with the three isoprenyl acetate phases."""

    time_hours: float
    glucose_g_per_l: float
    xylose_g_per_l: float
    total_sugar_g_per_l: float
    aqueous_mg_per_l: float
    organic_mg_per_l: float
    offgas_mg_per_l: float
    released_offgas_fraction_percent: float

    @property
    def titer_mg_per_l(self) -> float:
        """The stored titer: the sum of the three released phases."""
        return self.aqueous_mg_per_l + self.organic_mg_per_l + self.offgas_mg_per_l


def read_fed_batch(path: str | Path) -> list[FedBatchRow]:
    """Table S9's time course, with its header asserted column by column."""
    table = si_table(path, 9)
    header = table.header()
    if tuple(header) != SI_TABLE9_COLUMNS:
        raise RuntimeError(f"Table S9 header changed: {header}")
    index = {name: table.column(name) for name in SI_TABLE9_COLUMNS}
    return [
        FedBatchRow(
            time_hours=float(row[index["Time (h)"]]),
            glucose_g_per_l=float(row[index["Glucose (g/L)"]]),
            xylose_g_per_l=float(row[index["Xylose (g/L)"]]),
            total_sugar_g_per_l=float(row[index["Total sugar (g/L)"]]),
            aqueous_mg_per_l=float(row[index[SI_TABLE9_PHASES[0]]]),
            organic_mg_per_l=float(row[index[SI_TABLE9_PHASES[1]]]),
            offgas_mg_per_l=float(row[index[SI_TABLE9_PHASES[2]]]),
            released_offgas_fraction_percent=float(row[index["Off-gas fraction (%)"]]),
        )
        for row in table.rows[1:]
    ]


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, what it removed, and the items it removed."""

    rule: str
    scope: Literal["strain", "time_point"]
    description: str
    n_records: int
    items: list[str] = []


class BuildAccounting(BaseModel):
    """Everything a reader needs to audit one build's retention arithmetic."""

    dataset: str
    source_rows: int
    candidate_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule] = []
    reconciliation: LocusTagReconciliation | None = None
    span: SpanResolution | None = None
    notes: list[str] = []

    def check(self) -> None:
        """The rules must account for every candidate record that was not written."""
        accounted = sum(rule.n_records for rule in self.rules)
        if self.kept_records + self.dropped_records != self.candidate_records:
            raise RuntimeError(
                f"{self.dataset}: {self.kept_records} kept + {self.dropped_records} "
                f"dropped != {self.candidate_records} candidates"
            )
        if accounted != self.dropped_records:
            raise RuntimeError(
                f"{self.dataset}: rules total {accounted}, {self.dropped_records} "
                "records are missing from the build"
            )


def _write_accounting(accounting: BuildAccounting, preprocess_dir: str) -> None:
    """Validate and write ``preprocess/build_accounting.json``."""
    accounting.check()
    os.makedirs(preprocess_dir, exist_ok=True)
    with open(osp.join(preprocess_dir, "build_accounting.json"), "w") as handle:
        handle.write(accounting.model_dump_json(indent=2))


def _link_mirror_files(raw_dir: str, pins: Iterable[tuple[str, str, str]]) -> None:
    """Link each pinned mirror file into ``raw/`` after checking it against the manifest."""
    data_root = _data_root()
    manifest = load_manifest(data_root)
    os.makedirs(raw_dir, exist_ok=True)
    for relpath, filename, expected in pins:
        check_manifest_pin(relpath, manifest_sha256(manifest, relpath), expected)
        src = raw_mirror_dir(data_root) / relpath
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        link_verified(src, osp.join(raw_dir, filename), expected)


# --------------------------------------------------------------------------- #
# Genotype
# --------------------------------------------------------------------------- #
def deletion_perturbation(
    locus_tag: str, gene_name: str
) -> BacterialDeletionPerturbation:
    """One chromosomal deletion this campaign made on top of the PIPA chassis."""
    return BacterialDeletionPerturbation(
        systematic_gene_name=locus_tag,
        perturbed_gene_name=gene_name,
        gene_namespace=KT2440_NAMESPACE,
        cassette=None,
        collection=None,
    )


def cassette_perturbations(cassette: Cassette) -> list[HeterologousPathwayPerturbation]:
    """Every heterologous gene of one cassette, as its own typed perturbation.

    Each part token must appear in the cassette's quoted construct string, so the
    identifiers stay the source's own and a changed quote cannot drift away from them
    unnoticed.
    """
    for gene in cassette.genes:
        needle = gene.construct_token or gene.token
        if needle not in cassette.construct_string:
            raise RuntimeError(
                f"cassette part {needle!r} is not in the quoted {cassette.name} "
                f"description {cassette.construct_string!r}"
            )
    return [
        HeterologousPathwayPerturbation(
            systematic_gene_name=gene.token,
            perturbed_gene_name=gene.symbol,
            gene_namespace=KT2440_NAMESPACE,
            pathway_name=cassette.pathway,
            source_organism=gene.organism,
            is_heterologous=True,
            localization=cassette.localization,
            construct_name=cassette.name,
            integration_locus=cassette.integration_locus,
            variant=gene.variant,
            promoter_name=gene.promoter,
            copy_number=1.0,
        )
        for gene in cassette.genes
    ]


def strain_genotype(strain: str, symbols: dict[str, str]) -> Genotype:
    """The genotype of one engineered strain, as edits on the PIPA background."""
    spec = STRAIN_SPECS[strain]
    perturbations: list[Any] = []
    for locus_tag in spec.deletions:
        stated = DELETION_SYMBOLS[locus_tag]
        perturbations.append(
            deletion_perturbation(
                locus_tag, stated or symbols.get(locus_tag, locus_tag)
            )
        )
    for name in spec.cassettes:
        perturbations.extend(cassette_perturbations(CASSETTES[name]))
    return Genotype(perturbations=perturbations)


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
#: The record count this loader states: 7 Table 1 rows + 5 Table S4 rows + 7 Table S9
#: sampled times.
EXPECTED_RECORDS = len(TABLE1_ROWS) + len(AAT_CASSETTES) + 7
#: The baseline every family's reference is, and the quote that makes it the paper's own
#: fold-change denominator.
REFERENCE_BASELINES: dict[str, tuple[str, float, str, str]] = {
    "flask_pipa": ("PIPA-AAT1", 414.0, "flask_m9_glc20", _Q_FLASK_MAX_PIPA),
    "flask_pipaxyl": (
        "PIPAxyl-AAT1",
        181.0,
        "flask_m9hn_glcxyl_1010",
        _Q_PIPAXYL_BASELINE,
    ),
    "tube_pipa": ("PIPA-AAT1", 405.0, "tube_m9_glc20", _Q_AAT_PANEL_BEST),
    "fedbatch": (
        "PIPAxyl-E3-K3-O15",
        1500.0,
        "flask_mops_ye_glcxyl_137_pulse",
        _Q_FEDBATCH_TITER,
    ),
}
#: Which baseline each Table 1 row is measured against: the matching ``-AAT1`` strain in
#: its own background, which is the denominator the paper's own fold-changes use ("a
#: 1.7-fold increase", "a 2.1-fold increase over PIPAxyl-AAT1").
TABLE1_REFERENCE: dict[str, str] = {
    "PIPA-AAT1": "flask_pipa",
    "PIPA-E3": "flask_pipa",
    "PIPAxyl-AAT1": "flask_pipaxyl",
    "PIPAxyl-E3": "flask_pipaxyl",
    "PIPAxyl-E3-K3": "flask_pipaxyl",
    "PIPAxyl-E3-O15": "flask_pipaxyl",
    "PIPAxyl-E3-K3-O15": "flask_pipaxyl",
}


@register_dataset
class IsoprenylAcetateTiterKang2026Dataset(ExperimentDataset):
    """Kang 2026 isoprenyl acetate titers: Table 1, Table S4 and the Table S9 fed-batch."""

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"
    #: Every locus tag this campaign names must resolve to a locus of the pinned
    #: assembly. Measured on the pinned bytes: 13 of 13 (the 7 engineered deletions and
    #: the 6 chassis genes Table 1 places), so a value below 1.0 means the annotation or
    #: the released tags moved and the build stops.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 1.0

    def __init__(
        self,
        root: str = "data/torchcell/isoprenyl_acetate_titer_kang2026",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome resolves the chassis and deletion locus tags."""
        self.pputida_genome = pputida_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProductTiterExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProductTiterExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The deposited Supplementary Information document."""
        return [SI_DOCX_FILENAME]

    def download(self) -> None:
        """Link the SI document into ``raw/`` after verifying it against its pin."""
        _link_mirror_files(
            self.raw_dir, ((SI_DOCX_REL, SI_DOCX_FILENAME, SI_DOCX_SHA256),)
        )
        log.info("Kang 2026 SI artifact linked into %s", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or one opened from the genomes tier."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
        return self.pputida_genome

    @post_process
    def process(self) -> None:
        """Build one titer record per released strain and fed-batch time; write LMDB."""
        verify_raw_files(self.raw_dir, {SI_DOCX_FILENAME: SI_DOCX_SHA256})
        si_path = osp.join(self.raw_dir, SI_DOCX_FILENAME)
        n_quotes = verify_paper_quotes()
        panel = read_aat_panel(si_path)
        fed_batch = read_fed_batch(si_path)
        self._assert_fed_batch_oracles(fed_batch)
        self._assert_panel_agrees_with_table1(panel)

        genome = self._genome()
        tags = sorted(
            {tag for spec in STRAIN_SPECS.values() for tag in spec.deletions}
            | {entry.locus_tag for entry in CHASSIS_DELETIONS_TYPED}
        )
        stored, report = reconcile_locus_tags(genome, pd.Series(tags), label=self.name)
        report.require_resolved(self.MIN_RESOLVED_FRACTION)
        if report.outside_namespace:
            raise RuntimeError(
                f"{self.name}: locus tags outside {KT2440_NAMESPACE}: "
                f"{report.outside_namespace}"
            )
        if list(stored) != tags:
            raise RuntimeError(
                "a stated locus tag was remapped by the reconciliation; these are "
                f"current tags of the pinned assembly: {dict(zip(tags, stored))}"
            )
        symbols = _annotation_symbols(genome)
        background, span = chassis_background(genome)
        reference_genome = assembly_reference(PARENT_STRAIN, background=background)
        references = {
            key: ProductTiterExperimentReference(
                dataset_name=self.name,
                genome_reference=reference_genome,
                environment_reference=environment(CONDITIONS[condition]),
                phenotype_reference=titer_phenotype(
                    titer, n_samples=int(N_REPLICATES.value)
                ),
            )
            for key, (_, titer, condition, _) in REFERENCE_BASELINES.items()
        }
        pub = publication()

        rows: list[dict[str, Any]] = []
        for row in TABLE1_ROWS:
            rows.append(
                {
                    "source": "Table 1",
                    "column": "Titer (mg/L); culture conditions",
                    "strain": row.strain,
                    "condition": row.condition,
                    "titer_mg_per_l": row.titer_mg_per_l,
                    "time_hours": CONDITIONS[row.condition].duration_hours,
                    "n_samples": int(N_REPLICATES.value),
                    "product_yield_g_per_g": None,
                    "pulse_at_48h": CONDITIONS[row.condition].pulse_at_48h,
                    "reference": TABLE1_REFERENCE[row.strain],
                    "aqueous_mg_per_l": None,
                    "organic_mg_per_l": None,
                    "offgas_mg_per_l": None,
                    "quote": row.condition_quote,
                }
            )
        for entry in panel:
            rows.append(
                {
                    "source": "Table S4",
                    "column": SI_TABLE4_COLUMN,
                    "strain": entry.strain,
                    "condition": "tube_m9_glc20",
                    "titer_mg_per_l": entry.titer_mg_per_l,
                    "time_hours": 48.0,
                    "n_samples": int(N_REPLICATES.value),
                    "product_yield_g_per_g": None,
                    "pulse_at_48h": False,
                    "reference": "tube_pipa",
                    "aqueous_mg_per_l": None,
                    "organic_mg_per_l": None,
                    "offgas_mg_per_l": None,
                    "quote": f"{entry.enzyme} | {entry.organism} | {entry.identifier}",
                }
            )
        peak = max(point.titer_mg_per_l for point in fed_batch)
        for point in fed_batch:
            rows.append(
                {
                    "source": "Table S9",
                    "column": SI_TABLE9_SUM_LABEL,
                    "strain": "PIPAxyl-E3-K3-O15",
                    "condition": "fedbatch_mops_ye_glcxyl_1337",
                    "titer_mg_per_l": point.titer_mg_per_l,
                    "time_hours": point.time_hours,
                    "n_samples": None,
                    "product_yield_g_per_g": (
                        float(FEDBATCH_YIELD.value)
                        if point.titer_mg_per_l == peak
                        else None
                    ),
                    "pulse_at_48h": False,
                    "reference": "fedbatch",
                    "aqueous_mg_per_l": point.aqueous_mg_per_l,
                    "organic_mg_per_l": point.organic_mg_per_l,
                    "offgas_mg_per_l": point.offgas_mg_per_l,
                    "quote": _Q_FEDBATCH_SAMPLING,
                }
            )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for ledger_row in tqdm(rows, desc="kang2026-titer"):
                spec = CONDITIONS[str(ledger_row["condition"])]
                hours = ledger_row["time_hours"]
                replicates = ledger_row["n_samples"]
                experiment = ProductTiterExperiment(
                    dataset_name=self.name,
                    genotype=strain_genotype(str(ledger_row["strain"]), symbols),
                    environment=environment(
                        spec, duration_hours=None if hours is None else float(hours)
                    ),
                    phenotype=titer_phenotype(
                        float(ledger_row["titer_mg_per_l"]),
                        n_samples=None if replicates is None else int(replicates),
                        product_yield=ledger_row["product_yield_g_per_g"],
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(
                        experiment, references[str(ledger_row["reference"])], pub, itxn
                    ),
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(rows).to_csv(
            osp.join(self.preprocess_dir, "titer_rows.csv"), index=False
        )
        pd.DataFrame(
            [
                {"strain": name, "reason": reason}
                for name, reason in STRAINS_WITHOUT_A_TITER
            ]
        ).to_csv(
            osp.join(self.preprocess_dir, "strains_without_a_titer.csv"), index=False
        )
        with open(
            osp.join(self.preprocess_dir, "condition_provenance.json"), "w"
        ) as handle:
            json.dump(
                {
                    key: [value.model_dump(mode="json") for value in values]
                    for key, values in CONDITION_PROVENANCE.items()
                },
                handle,
                indent=2,
            )
        pd.DataFrame(
            span.disagreement_rows(), columns=list(SpanResolution.DISAGREEMENT_COLUMNS)
        ).to_csv(osp.join(self.preprocess_dir, "span_disagreement.csv"), index=False)
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=len(TABLE1_ROWS) + len(panel) + len(fed_batch),
                candidate_records=len(rows),
                kept_records=idx,
                dropped_records=len(rows) - idx,
                rules=[],
                reconciliation=report,
                span=span,
                notes=[
                    "nothing is dropped: every released isoprenyl acetate number in the "
                    "mirror is a record, from all three columns",
                    "the Table 1 strains with no titer are not records and are listed "
                    "in preprocess/strains_without_a_titer.csv",
                    f"{n_quotes} Table 1 quotes were re-read verbatim from the pinned "
                    "paper.md OCR before any value was used",
                    "the Δ86kb cell's three statements disagree; the typed alleles are "
                    "their intersection and preprocess/span_disagreement.csv lists "
                    "every item they disagree on",
                    _FEDBATCH_MEDIUM_NOTE,
                    f"the 48 h pulse has no Environment slot ({PULSE_RULE.value}); it "
                    "is carried per record in preprocess/titer_rows.csv",
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "Kang2026 titer: %d records (%d Table 1, %d Table S4, %d Table S9); "
            "%d chassis + deletion loci all current; Δ86kb span %d bp types %d full and "
            "%d partial deletions",
            idx,
            len(TABLE1_ROWS),
            len(panel),
            len(fed_batch),
            len(tags),
            span.span_length,
            len(span.full_deletions),
            len(span.partial_deletions),
        )

    @staticmethod
    def _assert_fed_batch_oracles(rows: Sequence[FedBatchRow]) -> None:
        """Table S9's own columns must confirm the three-phase sum this loader stores.

        Two independent checks on the deposited bytes: the released ``Off-gas fraction
        (%)`` equals off-gas / (the three-phase sum), which proves the sum is the
        denominator the authors used; and the released ``Total sugar (g/L)`` equals
        glucose + xylose, which proves the column arithmetic of the same table. The
        sugar columns are printed to one decimal, so three independently rounded numbers
        can disagree by :data:`_SUGAR_ROUNDING_TOL`; measured, one row does (93.4 h,
        2.1 + 3.2 = 5.3 against a released 5.2). The off-gas fraction is printed to one
        decimal on a percent scale, where the same rounding is far below 0.05.
        """
        if not rows:
            raise RuntimeError("Table S9 holds no sampled times")
        for row in rows:
            total = row.titer_mg_per_l
            expected = 0.0 if total == 0 else row.offgas_mg_per_l / total * 100.0
            if abs(expected - row.released_offgas_fraction_percent) > 0.05:
                raise RuntimeError(
                    f"Table S9 at {row.time_hours} h: off-gas / (aqueous + organic + "
                    f"off-gas) is {expected:.3f}% and the released column reads "
                    f"{row.released_offgas_fraction_percent}%; the three-phase sum is "
                    "not the total the authors used"
                )
            sugar = row.glucose_g_per_l + row.xylose_g_per_l
            if abs(sugar - row.total_sugar_g_per_l) > _SUGAR_ROUNDING_TOL:
                raise RuntimeError(
                    f"Table S9 at {row.time_hours} h: glucose + xylose is {sugar} and "
                    f"the released total reads {row.total_sugar_g_per_l}"
                )
        peak = max(row.titer_mg_per_l for row in rows)
        stated = float(FEDBATCH_FINAL_TITER_G_PER_L.value) * 1000.0
        if abs(peak - stated) > 50.0:
            raise RuntimeError(
                f"the maximum three-phase sum is {peak} mg/L and the Results state "
                f"{stated} mg/L for the same run"
            )

    @staticmethod
    def _assert_panel_agrees_with_table1(rows: Sequence[AatPanelRow]) -> None:
        """Table S4's own ATF1 titer must be the one the Results quote for PIPA-AAT1.

        The tube titer (405 mg/L) is NOT the flask titer Table 1 gives the same strain
        (414 mg/L); both are kept, and this asserts that the two columns are the two
        culture formats the Methods describe rather than two readings of one culture.
        """
        best = max(rows, key=lambda row: row.titer_mg_per_l)
        if best.enzyme != "ATF1":
            raise RuntimeError(
                f"Table S4's highest titer is {best.enzyme}, and the Results state ATF1 "
                "('achieved the highest isoprenyl acetate production')"
            )
        table1 = {row.strain: row.titer_mg_per_l for row in TABLE1_ROWS}
        if best.titer_mg_per_l >= table1["PIPA-AAT1"]:
            raise RuntimeError(
                f"Table S4's tube titer for PIPA-AAT1 is {best.titer_mg_per_l} mg/L and "
                f"Table 1's flask maximum is {table1['PIPA-AAT1']} mg/L; the tube "
                "endpoint is expected to be the lower of the two"
            )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification, L0 to L4, run from this module
# --------------------------------------------------------------------------- #
def _match_strain(record: dict[str, Any]) -> str:
    """The strain a stored record belongs to, matched on its genotype alone.

    A record names no strain (a ``Genotype`` is a set of typed edits), so the join back
    to :data:`STRAIN_SPECS` is by the cassette set and the deletion set, and exactly one
    spec must match. This is what lets the levels below check the store against the
    declarative strain table rather than against the build's own row order.
    """
    perturbations = record["experiment"]["genotype"]["perturbations"]
    constructs = {
        pert["construct_name"]
        for pert in perturbations
        if pert["perturbation_type"] == "heterologous_pathway"
    }
    deletions = {
        pert["systematic_gene_name"]
        for pert in perturbations
        if pert["perturbation_type"] == "bacterial_deletion"
    }
    matches = [
        name
        for name, spec in STRAIN_SPECS.items()
        if set(spec.cassettes) == constructs and set(spec.deletions) == deletions
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"a stored record matches {len(matches)} strain specs ({matches}); the "
            "genotype no longer identifies a strain"
        )
    return matches[0]


def _expected_perturbations(strain: str) -> int:
    """How many typed perturbations one strain's genotype must carry."""
    spec = STRAIN_SPECS[strain]
    return len(spec.deletions) + sum(
        len(CASSETTES[name].genes) for name in spec.cassettes
    )


def _record_signature(
    titer: float, duration_hours: float | None
) -> tuple[float, float]:
    """The key a record and its ledger row are joined on.

    ``(titer, duration_hours)`` is unique across the 19 records of this build, and both
    numbers are released values rather than positions: LMDB keys are strings, so the
    store's iteration order is lexicographic ("10" before "2") and a positional join
    would silently pair the wrong rows.
    """
    return (
        round(titer, 6),
        -1.0 if duration_hours is None else round(duration_hours, 6),
    )


def _records_by_source(
    records: Sequence[dict[str, Any]], rows: pd.DataFrame
) -> dict[str, list[dict[str, Any]]]:
    """Group the built records by the released column they came from."""
    if len(records) != len(rows):
        raise RuntimeError(
            f"{len(records)} records and {len(rows)} ledger rows; the build ledger and "
            "the store disagree"
        )
    by_signature: dict[tuple[float, float], str] = {}
    for _, row in rows.iterrows():
        hours = row["time_hours"]
        signature = _record_signature(
            float(row["titer_mg_per_l"]), None if pd.isna(hours) else float(hours)
        )
        if signature in by_signature:
            raise RuntimeError(
                f"two ledger rows share the signature {signature}; the join key is no "
                "longer unique and the verifier cannot pair records to rows"
            )
        by_signature[signature] = str(row["source"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        experiment = record["experiment"]
        signature = _record_signature(
            experiment["phenotype"]["titer"],
            experiment["environment"]["duration_hours"],
        )
        if signature not in by_signature:
            raise RuntimeError(
                f"a stored record has signature {signature}, which no ledger row carries"
            )
        grouped.setdefault(by_signature[signature], []).append(record)
    return grouped


def verify_build(dataset_root: str, data_root: str | None = None) -> VerificationReport:
    """Run L0 to L4 over a built tree and write ``preprocess/verification_report.json``.

    There is no ``run_product_titer`` runner in ``torchcell/verification/runners.py``
    yet, so the levels are assembled here, from this module, and the exact runner the
    plan's step 6 should add is named in the PR.

    L4 is a genuine cross-source join: the fed-batch records are re-derived from the
    deposited Table S9 bytes and compared to what the store holds, and the Table 1
    records are compared to the titers the Results prose states for the same strains
    (a DIFFERENT place in the same mirror from the table the values were read out of).
    """
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    ledger = pd.read_csv(osp.join(dataset_root, "preprocess", "titer_rows.csv"))
    grouped = _records_by_source(records, ledger)
    experiments = [record["experiment"] for record in records]
    phenotypes = [exp["phenotype"] for exp in experiments]
    report = VerificationReport(
        dataset_name=IsoprenylAcetateTiterKang2026Dataset.__name__,
        provenance=Provenance(
            source_uri=SI_DOCX_REL,
            citation_key=CITATION_KEY,
            sha256=SI_DOCX_SHA256,
            method=(
                "isoprenyl acetate titer in mg/L by GC-FID, stored verbatim as ug/mL: "
                "Table 1's flask maxima (from the pinned paper.md OCR), Table S4's "
                "48 h tube endpoints, and the sum of Table S9's three fed-batch phases"
            ),
            page="Table 1; Table S4; Table S9",
            retrieved=SI_RETRIEVED_AT,
        ),
    )
    report.add(l0_structural(experiments, ProductTiterExperiment.model_validate))
    report.add(l1_count(len(records), EXPECTED_RECORDS))
    report.add(l2_value_fidelity([p["titer"] for p in phenotypes], minimum=0.0))
    strains = [_match_strain(record) for record in records]
    report.add(
        l2_cross_method(
            [float(len(exp["genotype"]["perturbations"])) for exp in experiments],
            [float(_expected_perturbations(strain)) for strain in strains],
            tol=0.0,
        )
    )
    report.add(
        l3_convention(
            "titer_unit_is_the_sources_mg_per_l_as_ug_per_ml",
            all(
                p["titer_unit"] == ConcentrationUnit.ug_per_ml.value for p in phenotypes
            ),
            detail="1 mg/L == 1 ug/mL exactly, so the released number is stored verbatim",
        )
    )
    report.add(
        l3_convention(
            "every_uncertainty_is_a_typed_gap_not_a_guess",
            all(
                p["titer_uncertainty"] is None
                and p["titer_uncertainty_type"] is None
                and {"titer_uncertainty", "titer_uncertainty_type"}
                <= {gap["field"] for gap in p["provenance_gaps"]}
                for p in phenotypes
            ),
            detail="the replicate design is sourced (n=3 biological replicates, sample "
            "SD) but no SD number is released anywhere in the mirror",
        )
    )
    report.add(
        l3_convention(
            "every_genotype_carries_pIY670_and_one_aat",
            all(
                sum(
                    1
                    for pert in exp["genotype"]["perturbations"]
                    if pert["perturbation_type"] == "heterologous_pathway"
                    and pert["construct_name"] == PIY670.name
                )
                == len(PIY670.genes)
                and sum(
                    1
                    for pert in exp["genotype"]["perturbations"]
                    if pert["perturbation_type"] == "heterologous_pathway"
                    and pert["pathway_name"] == PATHWAY_ESTER
                )
                == 1
                for exp in experiments
            ),
            detail="PIPA + pIY670 supplies isoprenol and exactly one AAT esterifies it",
        )
    )
    report.add(
        l3_convention(
            "the_reference_is_the_pinned_PIPA_background",
            all(
                record["reference"]["genome_reference"]["strain"] == CHASSIS_STRAIN
                and record["reference"]["genome_reference"]["assembly_accession"]
                == "GCA_000007565.2"
                for record in records
            ),
            detail="every record is an edit of PIPA on GCA_000007565.2",
        )
    )
    report.add(_fed_batch_l4(grouped["Table S9"], data_root))
    report.add(_table1_l4(grouped["Table 1"], data_root))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def _fed_batch_l4(records: Sequence[dict[str, Any]], data_root: str | None) -> Any:
    """L4: the stored fed-batch titers against Table S9, re-read from the raw mirror."""
    path = raw_mirror_dir(data_root) / SI_DOCX_REL
    released = {row.time_hours: row.titer_mg_per_l for row in read_fed_batch(path)}
    shared = []
    for record in records:
        hours = record["experiment"]["environment"]["duration_hours"]
        if hours not in released:
            raise AssertionError(
                f"a fed-batch record is at {hours} h, which Table S9 does not sample"
            )
        shared.append(
            (hours, record["experiment"]["phenotype"]["titer"], released[hours])
        )
    if len(shared) != len(released):
        raise AssertionError(
            f"{len(shared)} fed-batch records for {len(released)} sampled times"
        )
    return l4_cross_source(shared, tol=1e-9).model_copy(
        update={"name": "fed_batch_titer_vs_table_s9_phases"}
    )


#: The titers the Results prose states for a Table 1 strain, each with its own quote.
#: An independent statement of the same measurement, in a different place of the same
#: pinned mirror from the table the loader reads.
TABLE1_PROSE_TITERS: dict[str, tuple[float, str]] = {
    "PIPA-AAT1": (414.0, _Q_FLASK_MAX_PIPA),
    "PIPA-E3": (724.0, "L to $7 2 4 ~ \\mathrm { m g / L }$ , a 1.7-fold increase"),
    "PIPAxyl-AAT1": (181.0, _Q_PIPAXYL_BASELINE),
    "PIPAxyl-E3": (
        383.0,
        "This engineered strain achieved a titer of $3 8 3 ~ \\mathrm { { \\ m g / L } }$",
    ),
    "PIPAxyl-E3-K3": (599.0, _Q_K3_TIME),
    "PIPAxyl-E3-O15": (
        819.0,
        "the PIPAxyl-E3- O15 strain produced $8 1 9 ~ \\mathrm { m g / L }$ of "
        "isoprenyl acetate",
    ),
}


def _table1_l4(records: Sequence[dict[str, Any]], data_root: str | None) -> Any:
    """L4: the stored Table 1 titers against the Results prose, quote by quote.

    Each quote must be verbatim in the sha256-pinned ``paper.md`` and must state the
    number the table gave, so a record cannot drift from either statement unnoticed.
    ``PIPAxyl-E3-K3-O15``'s 1500 mg/L is quoted as 1.5 g/L in the prose and is checked
    by :data:`FEDBATCH_FINAL_TITER_G_PER_L`'s sibling statement instead of here.
    """
    text = (library_mirror_dir(data_root) / PAPER_MD).read_text(encoding="utf-8")
    by_strain = {
        _match_strain(record): record["experiment"]["phenotype"]["titer"]
        for record in records
    }
    shared = []
    for strain, (prose_titer, quote) in sorted(TABLE1_PROSE_TITERS.items()):
        if quote not in text:
            raise AssertionError(
                f"the Results quote for {strain} is not verbatim in {PAPER_MD}: {quote!r}"
            )
        shared.append((strain, by_strain[strain], prose_titer))
    return l4_cross_source(shared, tol=1e-9).model_copy(
        update={"name": "table1_titer_vs_results_prose"}
    )


def main() -> None:
    """Build the dataset and run its verifier, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    root = osp.join(data_root, "data/torchcell/isoprenyl_acetate_titer_kang2026")
    genome = bacterial_genome("pputida", "KT2440", data_root)
    dataset = IsoprenylAcetateTiterKang2026Dataset(root=root, pputida_genome=genome)
    print(f"len = {len(dataset)}")
    accounting = json.loads(
        Path(osp.join(root, "preprocess/build_accounting.json")).read_text()
    )
    print(
        json.dumps(
            {
                key: accounting[key]
                for key in (
                    "source_rows",
                    "candidate_records",
                    "kept_records",
                    "dropped_records",
                    "notes",
                )
            },
            indent=2,
        )
    )
    print(verify_build(root, data_root).summary())


if __name__ == "__main__":
    main()
