# torchcell/datasets/ecoli/niu2019
# [[torchcell.datasets.ecoli.niu2019]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/niu2019
# Test file: tests/torchcell/datasets/ecoli/test_niu2019.py
"""Niu 2019: CRISPRa/CRISPRi of pinene-response genes in E. coli BW25113(PT5-dxs).

Niu et al. 2019 (Synth Syst Biotechnol 4(2):113-119, doi:10.1016/j.synbio.2019.05.001;
PMC6556621, citation key ``niuGenomicTranscriptionalChanges2019``) sequenced an evolved
pinene-tolerant isolate, read its transcriptome against the designed parent, and then
tested the up- and down-regulated genes one at a time in that parent with a
``dCas9*-MCPSoxS`` CRISPR activator and repressor. Two numbers per strain are released,
each a ratio against the no-guide control and each from its own culture: the OD600 after
12 h in LB plus 0.5% pinene, and the pinene made in a 72 h SBMSN shake-flask production
run. This module holds a store per column.

RECORD (growth) = one (target gene set x arm) ``BacterialEnvironmentResponseExperiment``:

- GENOTYPE: one ``BacterialReleasedCrisprPerturbation`` per targeted gene, keyed by the
  label the release writes (a gene symbol or an operon label), carrying the arm's
  released ``expression_direction``, the shared ``dCas9*-MCPSoxS`` construct and, when
  Suppl. Table 1 releases the matching ``N20-<gene>`` oligo, its guide spacer. The
  combination strains carry SIX leaves in one genotype, which is what they are.
- ENVIRONMENT: the one tolerance culture: LB plus 0.5% pinene at 37 C for 12 h, with the
  200 nM anhydrotetracycline that induces the effector carried as a medium COMPONENT.
- PHENOTYPE: ``EnvironmentResponsePhenotype`` with ``measurement_type=log2_ratio`` and
  ``assay_type=liquid_od_growth``. The reference carries 0.0, which is what a strain
  growing exactly like the no-guide control is.

RECORD (pinene) = one (target gene set x arm) ``ProductTiterExperiment``:

- GENOTYPE: the same released-construct leaves, keyed the same way.
- ENVIRONMENT: a ``CultureEnvironment`` for the PRODUCTION run: 50 mL SBMSN at the
  released amounts plus 1 mM IPTG and 200 nM anhydrotetracycline, a 20% dodecane overlay
  as the one environment perturbation, 72 h at 30 C and 130 rpm after induction, from an
  overnight 5 mL LB seed culture.
- PHENOTYPE: ``ProductTiterPhenotype`` on ``ConcentrationUnit.fold_change``, the released
  ratio with its released sample SD over three replicates, quantified by GC-FID. The
  reference carries 1.0, which is what the ratio's denominator is.

WHY THE CRISPR ARMS EXIST ONLY NOW. Issue #799: until
``BacterialCrisprActivationPerturbation`` landed, the activation arm had nowhere to go.
``CrisprActivationPerturbation`` inherits the R64 systematic-name validator and refuses
both ``b3417`` and ``dxs``, and the only bacterial expression leaves were CRISPRi, whose
direction is fixed to ``decreased``, and ``PromoterReplacementPerturbation``, which
asserts SO:1000032 ``delins`` -- a native promoter removed and a characterized part put
in its place -- an edit a guide-directed activator never makes. The measurement is
``experiments/036-dataset-fixes-before-kg-build/results/niu2019_release_loadability.json``.

THE HOST IS THE UNEVOLVED PARENT. Every record here is ``BW25113(PT5-dxs)`` carrying the
effector plasmid plus one guide plasmid, with the same parent carrying the empty
``pTargetA`` as the control. ``PT5-dxs`` is CONSTANT across the dataset and shared with
the control, so it is a ``BacterialStrainBackground`` on the reference rather than a
perturbation in the ``Genotype`` (the #507 rule: the genotype keeps only what the screen
varies).

THE EVOLVED ISOLATE ``YZFP`` HAS A WRITABLE GENOTYPE AND NO MEASURED PHENOTYPE, SO IT IS
NOT A RECORD HERE. Suppl. Table 2's 374 called variants are its subject, and since the
#835 leaves landed they are writable: 322 rows carry a b-number and take
``BacterialSequenceVariantPerturbation``, 48 are intergenic and take
``BacterialSiteVariantPerturbation``, which is 373 of 374 with the one blank-mutation-site
row refusing on its missing coordinate and 4 rows neither b-numbered nor intergenic
(counts from
``experiments/036-dataset-fixes-before-kg-build/results/niu2019_release_loadability.json``,
the #731 classification). What refuses is the RECORD, not the genotype. This release
measures no number of YZFP: its four SI tables are the primers, that variant list and the
two CRISPRa/i target tables, and every number in the latter two is measured in the
UNEVOLVED parent; the evolved strain's own tolerance and titer are attributed to reference
[8] ("we first improved pinene tolerance to 2.0% and pinene production to 9.9 mg/L from
5.6 mg/L ... to obtain the pinene tolerant strain Escherichia coli YZFP [8]"), which the
Caglar 2017 rule (#771) attributes to the study that first reported them; and the 182
qRT-PCR YZFP-over-parent transcript ratios are released as Fig. 2's bar panels with no
table behind them. A genotype with no measured phenotype is not an ``Experiment``, so the
373 writable leaves have nothing to hang on and the rows are counted in
``preprocess/not_loaded.json`` rather than stored.

WHY ``EnvironmentResponsePhenotype`` AND NOT ``FitnessPhenotype``. The released number is
a ratio against a control measured in the same run, which is an environment-response
quantity rather than a ko/wt fitness; and the stored log2 of it is signed by
construction, while ``FitnessPhenotype`` clamps non-positive values.

WHY THE STORED NUMBER IS A LOG2 AND THE RELEASED SD IS A TYPED GAP. The release gives a
RATIO ("a The ratio of OD600 with CRISPRa and without") with a sample standard deviation
over three replicates. ``MeasurementType`` has no fold-change member, and ``log2_ratio``
is the member whose definition the transformed number satisfies exactly, which is also
the encoding Lim 2025 landed for the same kind of quantity. The consequence is that the
released dispersion cannot ride along: a sample SD of a RATIO is not the SD of its
logarithm, and propagating it would be a first-order approximation the source never
made. ``environment_response_uncertainty`` therefore carries a typed gap, every released
ratio and SD is written verbatim to ``preprocess/released_growth_ratios.json``.
``ConcentrationUnit.fold_change`` landed in #1077 and the pinene column is now a store
of its own (``PineneFoldChangeNiu2019Dataset``, below), but ``MeasurementType`` still
has no fold-change member, so the growth column keeps its log2 and its typed gap.
``n_samples=3`` and
``sample_unit=biological_replicate`` ARE stored: they are facts about the measurement,
not about the scale ("All experiments were conducted in triplicate, and data were
averaged and presented as the means +- standard deviation").

TWO STORES, ONE PER COLUMN (counts in each store's ``preprocess/dropped_records.json``).
Released numeric cells: 43 activation growth ratios + 32 activation pinene ratios + 9
interference growth ratios + 6 interference pinene ratios + the 4 cells of the two
combination strains = 94. The growth store holds 54 of them and the pinene store holds
the other 40, so every released numeric cell of the paper is stored:

1. ``CrisprPineneToleranceNiu2019Dataset`` -- 54 records, the growth column: 43 + 9
   released ratios plus the two combination strains' growth cells. Its ledger's one
   rule, ``pinene_ratio_is_a_dimensionless_product_ratio``, states why the 40 pinene
   cells are not in THIS store: they are a product ratio measured in a different
   culture, so they are a product-titer record rather than a growth one.
2. ``PineneFoldChangeNiu2019Dataset`` -- 40 records, the pinene column: 32 + 6 released
   ratios plus the two combination strains' pinene cells, on
   ``ConcentrationUnit.fold_change`` with the empty-vector control at 1.0 (#770, closed
   by #1077's unit member).

Both stores key a record by the label the release writes, operon labels included
(``flgFGH``, ``sufBCDS``), on the released-construct leaf; no annotation is read at load
time.

Two further parts of the release are not cells of this grid and are refused for reasons
of their own, recorded in ``preprocess/not_loaded.json``:

- Suppl. Table 2's 374 called variants of the evolved isolate ``YZFP``, for the reason
  above: the genotype is writable on the #835 leaves and the release measures no
  phenotype of that strain.
- The 64 dashed cells. The tables' own footnote defines a dash ("-: means no change or
  negative effect."), and classifying each cell as numeric or dashed reproduces the main
  text's both / growth-only / pinene-only tallies exactly for both arms, so a dash is a
  MEASURED non-positive outcome rather than an absent measurement. It still has neither
  a number nor a single ``ResponseCategory``, because the footnote conflates "no change"
  with "negative effect", so it is left-censored and recorded rather than encoded.

PINENE HAS A CURATED ROW THAT RECORDS WHY IT STAYS A TYPED GAP. The #726 curator pass
queried it: PubChem's name index answers ``pinene`` with CID 15837102, whose structure is
neither alpha- nor beta-pinene, and the paper names alpha-pinene only as the standard its
production assay is quantified against, never as the dosed reagent. So the table's
``pinene`` row is UNRESOLVED_PUBLIC with that reason, the stress compound keeps its typed
gap on ``inchikey``, and ``alpha-pinene`` has its own row (CID 6654) for a source that
states it.

GUIDE SPACERS. Suppl. Table 1 releases 91 ``N20-<gene>`` oligos over 87 distinct names,
each of the form ``CGGGGTACC`` + spacer + ``gttttagagctagaaatag``; the spacer is the
middle, 20 nt in 90 of the 91 rows and 22 nt in the ``N20-prpR`` row, and is stored at
the length the release gives it rather than trimmed to the name's "N20". Two names
(``marA``, ``sspA``) appear twice with DIFFERENT spacers and the release gives no rule
for choosing, and one stored target (``opgB``) has no oligo row at all; those three
records carry ``guide_sequence=None`` and the reason is written to
``preprocess/guide_spacers.json``. A ``CrisprConstruct`` is not a gap carrier, so the
absence is recorded there and in the dendron note rather than on the record.

DATA SOURCE: ``mmc1.docx``, the article's single supplementary file, from the PMC Article
Datasets bucket (``pmc_cloud``, key ``PMC6556621.1/mmc1.docx``), deposited in
``$DATA_ROOT/torchcell-raw/niuGenomicTranscriptionalChanges2019/`` with a
``manifest.json``. The bucket carries exactly one supplementary object and the article's
JATS declares exactly one ``<supplementary-material>`` element, both measured, so one SI
file is the complete release rather than a partial capture. The two combination strains
are released in the MAIN tables only, which the OCR'd ``paper.md`` carries; the
interference one's six genes are read from the released row label and corroborated by
the Results prose that names them.
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
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, cast
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    AssayType,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialReleasedCrisprPerturbation,
    BacterialStrainBackground,
    Censoring,
    ComponentDefinition,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    CultureEnvironment,
    CultureFormat,
    EndpointRule,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    PreCulture,
    PreCultureSource,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import (
    STRAIN_GENE_NAMESPACES,
    assembly_reference,
    bacterial_genome,
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
from torchcell.sequence.genome.ecoli.k12 import (
    EcoliK12BW25113Genome,
    EcoliK12StrainName,
)
from torchcell.verification.levels import l3_convention
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
)

log = logging.getLogger(__name__)

DOI = "10.1016/j.synbio.2019.05.001"
PMCID = "PMC6556621"
TITLE = (
    "Genomic and transcriptional changes in response to pinene tolerance and "
    "overproduction in evolved Escherichia coli"
)
CITATION_KEY = "niuGenomicTranscriptionalChanges2019"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "43c733ed632ce1501fc734ac687a525b242d2d702e2c39e37adc0ac89c98dc80"

#: The article's one supplementary file, as the publisher deposited it. The mirror holds
#: it as ``si/si1.docx``; the raw mirror keeps the publisher's own filename.
SI_FILENAME = "mmc1.docx"
SI_SHA256 = "72099acb46abce07983265e559ec49596209b031869f081f40ae1e0bc0ce6fb9"
SI_LIBRARY_RELPATH = "si/si1.docx"
SI_BUCKET_KEY = f"{PMCID}.1/{SI_FILENAME}"
RAW_RETRIEVED_AT = "2026-10-09"

DATASET_ROOT_REL = "data/torchcell/crispr_pinene_tolerance_niu2019"

#: The namespace every record is written in: the host is BW25113 and every target is
#: resolved against the deposited BW25113 annotation.
BW25113_NAMESPACE = STRAIN_GENE_NAMESPACES["BW25113"]
REFERENCE_STRAIN_NAME: EcoliK12StrainName = "BW25113"

#: The two arms, as the SI's own table titles name them.
Arm = Literal["crispr_activation", "crispr_interference"]

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

#: Suppl. Table 1's own column header, verbatim.
TABLE_1_HEADER: tuple[str, ...] = ("Name", "Sequence", "Purpose")
#: Suppl. Tables 3 and 4 share this header, verbatim (the trailing letters are the
#: footnote markers the release prints inside the cell).
TARGET_TABLE_HEADER: tuple[str, ...] = (
    "Gene",
    "Protein",
    "Ratio of growtha",
    "Ratio of pinene concentrationb",
)
#: The SI's four table titles, in document order, verbatim.
SI_TABLE_TITLES: tuple[str, ...] = (
    "Suppl. Table 1 Primers used in this study",
    "Suppl. Table 2 Mutations in E. coli YZFP",
    "Suppl. Table 3 Effect of CRISPR activation of the up-regulated genes on growth "
    "and pinene production of pinene",
    "Suppl. Table 4 Effect of CRISPR interference of the down-regulated genes on "
    "growth and production of pinene",
)
#: The footnote that defines a ``-`` cell in Suppl. Tables 3 and 4, verbatim.
DASH_FOOTNOTE = "-: means no change or negative effect."

#: A value cell of the two target tables: ``1.24 ± 0.01``.
RATIO_CELL = re.compile(r"^(?P<ratio>\d+\.\d+)\s*±\s*(?P<sd>\d+\.\d+)$")
#: The constant flanks of every released ``N20`` oligo. The spacer is what lies between
#: them; both are required, so a row that carries neither cannot be read as a spacer.
OLIGO_PREFIX = "CGGGGTACC"
OLIGO_SUFFIX = "GTTTTAGAGCTAGAAATAG"

#: The two combination strains, released in the MAIN tables only. The label is the
#: released row label verbatim; ``genes`` is that label split on the hyphen, which the
#: Results prose corroborates gene by gene for both strains.
COMBINATION_LABELS: dict[Arm, str] = {
    "crispr_activation": "flgFGH-sufBCDS-dusB-rpoA-yehA-hslU",
    "crispr_interference": "ydiJ-yjbQ-prpR-marR-fabR-cedA",
}
#: The main-table growth cell of each combination strain, verbatim.
COMBINATION_GROWTH: dict[Arm, tuple[float, float]] = {
    "crispr_activation": (3.55, 0.03),
    "crispr_interference": (1.26, 0.02),
}

#: A released target label that is not one gene. Measured on the release: exactly these
#: two, both in the activation arm. They are STORED under the label the release writes
#: (the released-construct leaf takes an operon label as an identifier), and are kept
#: here because the note and the ledger name them.
MULTI_GENE_LABELS: frozenset[str] = frozenset({"sufBCDS", "flgFGH"})


class ReleaseContentError(RuntimeError):
    """The release does not hold what the paper says it holds."""


class TableFormatError(RuntimeError):
    """A consumed SI table's shape is not the one this loader reads."""


# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned Niu 2019 OCR mirror."""
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


_GROWTH_ASSAY = "Methods, 'Effect of the activation and repression on growth'"
_STATISTICS = "Methods, 'Statistical analysis'"
_ARMS = "Results, 'Effects of the activation and repression on growth and production'"

HOST_STRAIN = _paper(
    "BW25113(PT5-dxs)",
    "Thus, we applied this CRISPR-Cas-SoxS system [17] for activating and repressing "
    "target genes in E. coli $\\mathrm { B W } 2 5 1 1 3 ( \\mathrm { P _ { T 5 - d x "
    "s } } )$ to investigate their effects on growth and pinene production.",
    page=_ARMS,
    note="the host of every stored record: the UNEVOLVED designed parent, not the "
    "evolved isolate YZFP, whose called variants are Suppl. Table 2's subject and whose "
    "writable genotype has no measured phenotype in this release to carry it",
)
EFFECTOR = _paper(
    "dCas9*-MCPSoxS",
    "pBbB2K-dCas9\\*-MCPSoxS and pTargetA-X were co-transferred into E. coli BW25113 "
    "$( \\mathbf { P } _ { \\mathrm { T 5 - d x s } } )$ .",
    page=_ARMS,
    note="the one guide-directed effector of both arms: a dead-Cas9 fused to MCP-SoxS, "
    "which activates or represses according to where the guide puts it",
)
GROWTH_CULTURE = _paper(
    ("LB", 0.5, 200.0, 0.09, 37.0, 12.0),
    "The overnight cultures were inoculated into $5 0 \\mathrm { m L }$ of LB medium "
    "supplement with $0 . 5 \\%$ pinene and $2 0 0 \\mathrm { n M }$ anhydrous "
    "tetracycline with a starting $\\mathrm { \\ O D } _ { 6 0 0 }$ of 0.09. The "
    "cultures were incubated at $3 7 ^ { \\circ } \\mathrm { C }$ ( $2 0 0 \\mathrm { "
    "r p m } )$ for $1 2 \\mathrm { h }$ .",
    page=_GROWTH_ASSAY,
    note="the one culture every stored record was measured in: LB plus 0.5% pinene and "
    "200 nM anhydrotetracycline, 37 C, 12 h. The paper writes 0.5% with no w/v or v/v "
    "marker, so the dose is stored in the basis-free percent member",
)
CONTROL_STRAIN = _paper(
    "BW25113(PT5-dxs) (pBbB2K-dCas9*-MCPSoxS, pTargetA)",
    "The strain harboring the empty vector pTargetA, E. coli B $N 2 5 1 1 3 ( \\mathrm "
    "{ P _ { T 5 - d x s } } )$ (pBbB2K-${ \\mathrm { d } } \\mathsf { C a s 9 } ^ { * "
    "}$ -MCPSoxS, pTargetA), was set as the control.",
    page=_GROWTH_ASSAY,
    note="what the released ratio is read against, and why the reference phenotype is "
    "log2(1) = 0: the same parent carrying the same effector plasmid and an EMPTY "
    "guide vector",
)
GROWTH_RATIO_MEANING = _paper(
    "ratio of OD600 with the guide and without",
    "a The ratio of $\\mathrm { O D } _ { 6 0 0 }$ with CRISPRa and without. b The "
    "ratio of pinene production with CRISPRa and without.-: means no change or "
    "negative effect.",
    page="Table 3, footnotes",
    note="the stored quantity, and the reason the pinene column is refused: both are "
    "dimensionless ratios, and only the growth one has a MeasurementType member whose "
    "definition the transformed number satisfies",
)
TRIPLICATE = _paper(
    3,
    "All experiments were conducted in triplicate, and data were averaged and "
    "presented as the means $\\pm$ standard deviation.",
    page=_STATISTICS,
    note="n_samples and the kind of the released dispersion: a sample SD over three "
    "biological replicates. It is the SD of the RATIO, which is why it cannot be "
    "stored beside the log2 of that ratio",
)
ACTIVATION_TARGETS = _paper(
    57,
    "Of 96 up-regulated gens, a total of 57 up-regulated genes with higher "
    "transcription level compared to E. coli $\\mathrm { B W } 2 5 1 1 3 ( \\mathrm { "
    "P _ { T 5 - d x s } } )$ was selected to be activated with CRISPRa.",
    page=_ARMS,
    note="the activation arm's size, which Suppl. Table 3's data rows must equal",
)
INTERFERENCE_TARGETS = _paper(
    20,
    "All of 20 down-regulated genes were selected to be repressed with CRISPRi.",
    page=_ARMS,
    note="the interference arm's size, which Suppl. Table 4's data rows must equal",
)
ACTIVATION_COMBINATION = _paper(
    COMBINATION_LABELS["crispr_activation"],
    "The simultaneous activation of the flgFGH, sufBCDS, dusB, rpoA, yehA and hslU "
    "further improved growth and pinene production than single activation (Table 3).",
    page=_ARMS,
    note="the six genes of the activation combination strain, naming in prose what the "
    "main table's row label truncates. Two of them (flgFGH, sufBCDS) are operons, so "
    "the strain cannot be written as six gene-keyed leaves and is dropped",
)
INTERFERENCE_COMBINATION = _paper(
    COMBINATION_LABELS["crispr_interference"],
    "The same result of simultaneous repression of the ydiJ, yjbQ, prpR, marR, fabR "
    "and cedA was also observed (Table 4).",
    page=_ARMS,
    note="the six genes of the interference combination strain, every one a single "
    "gene, which is why this strain IS stored as a six-perturbation genotype",
)
DXS_PARENT_DESIGN = _paper(
    "PT5-dxs",
    "Are there synergies effects between these genes in the modular co-culture system "
    "of the whole-cell biocatalysis? All these should be needed further to investigate.",
    page="Conclusions",
    note="the parent's designed cassette appears throughout as the strain name "
    "BW25113(PT5-dxs) rather than in a construction sentence of its own, so the "
    "background's genotype_statement is the strain name the paper writes and its "
    "construction is a typed gap",
)

#: Every module-level sourced value, for the mirror audit.
SOURCED_VALUES: tuple[SourcedValue, ...] = (
    HOST_STRAIN,
    EFFECTOR,
    GROWTH_CULTURE,
    CONTROL_STRAIN,
    GROWTH_RATIO_MEANING,
    TRIPLICATE,
    ACTIVATION_TARGETS,
    INTERFERENCE_TARGETS,
    ACTIVATION_COMBINATION,
    INTERFERENCE_COMBINATION,
    DXS_PARENT_DESIGN,
)

N_ACTIVATION_TARGETS: int = int(ACTIVATION_TARGETS.value)
N_INTERFERENCE_TARGETS: int = int(INTERFERENCE_TARGETS.value)
N_SAMPLES: int = int(TRIPLICATE.value)


# --------------------------------------------------------------------------- #
# The host background, the medium and the one environment
# --------------------------------------------------------------------------- #
NIU2019_BACKGROUND = BacterialStrainBackground(
    name="BW25113(PT5-dxs)",
    reference_strain=REFERENCE_STRAIN_NAME,
    assembly_set=cast("Any", BACTERIAL_ASSEMBLY_SETS[REFERENCE_STRAIN_NAME]),
    parents=["BW25113"],
    construction=None,
    genotype_statement="BW25113(PT5-dxs)",
    alleles=[],
    provenance=[HOST_STRAIN, DXS_PARENT_DESIGN],
    provenance_gaps=[
        ProvenanceGap(
            field="construction",
            reason=ProvenanceGapReason.not_reported_by_primary,
            looked_in=Provenance(
                source_uri=PAPER_MD,
                citation_key=CITATION_KEY,
                sha256=PAPER_MD_SHA256,
                method="full Methods and Results read for a construction sentence for "
                "the PT5-dxs parent",
                page="Methods; Results",
            ),
            note="the paper names the parent as BW25113(PT5-dxs) throughout and never "
            "says how the PT5-dxs cassette was built or where it sits, so the "
            "construction is not asserted. The cassette is constant across the "
            "dataset and shared with the control strain, which is why it is a "
            "background at all rather than a perturbation",
        )
    ],
)
"""The designed parent every stored record is measured in, as a background.

NOT a ``Genotype`` perturbation: ``PT5-dxs`` is identical in every strain of the
dataset AND in the no-guide control the ratio is read against, so it is what the #507
background contract is for. Putting it in the genotype would make every record a
two-perturbation strain and would make the control's genotype non-empty.
"""

_NO_LB_AMOUNT = (
    "LB's identities are its definition; this paper states no formulation, and the "
    "project's own 'LB' is Miller for some rows and Lennox for others, so asserting "
    "either would fabricate three numbers"
)


def _lb_ingredients(provenance: SourcedValue | None = None) -> list[MediaComponent]:
    """LB's three ingredients with NO amounts, sourced to the culture that used it."""
    sourced = GROWTH_CULTURE if provenance is None else provenance
    return [
        MediaComponent(
            compound=resolved_compound("tryptone"),
            role=MediaComponentRole.complex_ingredient,
            concentration=None,
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[sourced],
            note=_NO_LB_AMOUNT,
        ),
        MediaComponent(
            compound=resolved_compound("yeast extract"),
            role=MediaComponentRole.complex_ingredient,
            concentration=None,
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[sourced],
            note=_NO_LB_AMOUNT,
        ),
        MediaComponent(
            compound=resolved_compound("sodium chloride"),
            role=MediaComponentRole.bulk_salt,
            concentration=None,
            provenance=[sourced],
            note=_NO_LB_AMOUNT,
        ),
    ]


NIU2019_LB_ATC = Media(
    name="LB with 200 nM anhydrotetracycline, formulation not stated (Niu 2019 "
    "tolerance assay), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium="LB",
    components=[
        *_lb_ingredients(),
        MediaComponent(
            compound=resolved_compound("anhydrotetracycline"),
            role=MediaComponentRole.other,
            concentration=Concentration(value=200.0, unit=ConcentrationUnit.nanomolar),
            provenance=[GROWTH_CULTURE],
            note="the inducer of the dCas9*-MCPSoxS effector. It is a medium component "
            "rather than an Environment.perturbation for the reason Rousset 2018's aTc "
            "is: the paper lists it as a supplement of the medium, it is constant "
            "across the dataset, and it switches the genome-side perturbation on "
            "rather than being the environmental variable the screen contrasts",
        ),
    ],
)
"""The tolerance assay's medium: LB plus the effector's inducer, pinene excluded.

Pinene is deliberately NOT a component here. It is the environmental challenge the
phenotype is about, so it is a ``SmallMoleculePerturbation`` on the environment, which
is where a consumer asking "what was this strain challenged with" will look.
"""


def pinene_challenge() -> SmallMoleculePerturbation:
    """0.5% pinene, the stress the released ratio is a tolerance to.

    The dose is stored in ``ConcentrationUnit.percent``, the basis-free member, because
    the paper writes "0.5% pinene" with no w/v or v/v marker and pinene is a liquid
    terpene whose percent could honestly be either. The compound comes back from the
    shared layer carrying a typed gap on its structure identifiers: the table's
    ``pinene`` row records that PubChem's name index answers the bare name with a
    structure that is neither alpha- nor beta-pinene (#726), and the paper never says
    which pinene it dosed, so nothing is guessed from a near-miss name.
    """
    return SmallMoleculePerturbation(
        compound=resolved_compound("pinene"),
        concentration=Concentration(value=0.5, unit=ConcentrationUnit.percent),
    )


def environment() -> Environment:
    """The one culture of this dataset."""
    perturbations: list[EnvironmentPerturbationType] = [pinene_challenge()]
    return Environment(
        media=NIU2019_LB_ATC,
        temperature=Temperature(value=37.0),
        perturbations=perturbations,
        aerobicity="aerobic",
        duration_hours=12.0,
    )


# --------------------------------------------------------------------------- #
# Phenotype
# --------------------------------------------------------------------------- #
UNITS = (
    "log2(OD600 after 12 h in LB with 0.5% pinene, with the guide / the same parent "
    "with an empty pTargetA); 0 = grows like the no-guide control"
)
UNITS_REFERENCE = "the no-guide control of the same culture (log2 ratio 0)"


def _uncertainty_gap(field: str) -> ProvenanceGap:
    """The released dispersion is a sample SD of the RATIO, not of its logarithm."""
    return ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="the release states a sample standard deviation over three replicates OF "
        "THE RATIO (for example '1.22 ± 0.02'), and the stored value is the log2 of "
        "that ratio. An SD does not transform with its statistic, and propagating it "
        "to the log scale would be a first-order approximation the source never made, "
        "so the released ratio and SD are written verbatim to "
        "preprocess/released_growth_ratios.json instead of being converted. A "
        "MeasurementType fold-change member would let the ratio and its SD be stored "
        "as released",
    )


def phenotype(ratio: float, arm: Arm) -> EnvironmentResponsePhenotype:
    """One released growth ratio, stored as its log2."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.liquid_od_growth,
        environment_response=math.log2(ratio),
        n_samples=N_SAMPLES,
        sample_unit=SampleUnit.biological_replicate,
        units=UNITS,
        screen_id=arm,
        provenance_gaps=[
            _uncertainty_gap("environment_response_uncertainty"),
            _uncertainty_gap("environment_response_se"),
        ],
    )


def reference_phenotype(arm: Arm) -> EnvironmentResponsePhenotype:
    """The no-guide control: a ratio of 1, whose log2 is 0."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.liquid_od_growth,
        environment_response=0.0,
        units=UNITS_REFERENCE,
        screen_id=arm,
        provenance_gaps=[
            ProvenanceGap(
                field="n_samples",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the 0 baseline is what a strain growing exactly like the "
                "empty-vector control is, not a measured set of control replicates",
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Genotype
# --------------------------------------------------------------------------- #
#: The released expression direction of each arm's dead-Cas effector.
DIRECTION_OF_ARM: dict[Arm, str] = {
    "crispr_activation": "increased",
    "crispr_interference": "decreased",
}

#: What ``systematic_gene_name`` is on every record, as the release writes it.
IDENTIFIER_KIND = (
    "target label, as Suppl. Table 3 or 4 writes it (a gene symbol or an operon label)"
)


class StoredTarget(BaseModel):
    """One targeted gene of a stored strain: its released label and its spacer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reported_symbol: str
    strain_label: str
    guide_sequence: str | None


def perturbation(target: StoredTarget, arm: Arm) -> Any:
    """One CRISPRa or CRISPRi leaf, keyed by the label the release names.

    Both arms take the one released-construct leaf (#1077), whose
    ``expression_direction`` is a released fact of the effector rather than a class
    default; the label is stored verbatim, so an operon label such as ``sufBCDS`` is a
    record identity like a gene symbol and no annotation is read.
    """
    return BacterialReleasedCrisprPerturbation(
        systematic_gene_name=target.reported_symbol,
        perturbed_gene_name=target.reported_symbol,
        gene_namespace=BW25113_NAMESPACE,
        strain_label=target.strain_label,
        identifier_kind=IDENTIFIER_KIND,
        expression_direction=DIRECTION_OF_ARM[arm],
        crispr=CrisprConstruct(
            effector=str(EFFECTOR.value), guide_sequence=target.guide_sequence
        ),
    )


def genotype(targets: Sequence[StoredTarget], arm: Arm) -> Genotype:
    """The strain's genotype: one leaf per targeted gene, six for a combination strain."""
    if not targets:
        raise ValueError("a stored strain targets at least one gene")
    return Genotype(perturbations=[perturbation(t, arm) for t in targets])


# --------------------------------------------------------------------------- #
# Reading the SI docx
# --------------------------------------------------------------------------- #
def _cell_text(tc: ET.Element) -> str:
    """A table cell's text: its paragraphs' runs, joined by a space."""
    return " ".join(
        "".join(t.text or "" for t in p.iter(W + "t")).strip()
        for p in tc.findall(W + "p")
    ).strip()


class SupplementaryTables(BaseModel):
    """The SI docx as read: its four tables and its paragraph text."""

    model_config = ConfigDict(extra="forbid")

    tables: list[list[list[str]]]
    paragraphs: list[str]


def read_si(path: str | Path) -> SupplementaryTables:
    """Read ``mmc1.docx``: the four tables in document order, and every paragraph.

    The four table titles and the dash footnote are asserted, so an SI whose tables
    moved stops the build instead of being read in the wrong order.
    """
    root = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    body = root.find(W + "body")
    if body is None:
        raise TableFormatError(f"{path}: word/document.xml has no w:body")
    tables: list[list[list[str]]] = []
    paragraphs: list[str] = []
    for child in body:
        if child.tag == W + "p":
            text = "".join(t.text or "" for t in child.iter(W + "t")).strip()
            if text:
                paragraphs.append(text)
        elif child.tag == W + "tbl":
            tables.append(
                [
                    [_cell_text(tc) for tc in tr.findall(W + "tc")]
                    for tr in child.findall(W + "tr")
                ]
            )
    if len(tables) != len(SI_TABLE_TITLES):
        raise TableFormatError(
            f"{path}: {len(tables)} tables, expected {len(SI_TABLE_TITLES)}"
        )
    for title in SI_TABLE_TITLES:
        if title not in paragraphs:
            raise TableFormatError(f"{path}: the SI title {title!r} is not present")
    if DASH_FOOTNOTE not in paragraphs:
        raise TableFormatError(f"{path}: the dash footnote is not present")
    return SupplementaryTables(tables=tables, paragraphs=paragraphs)


class GuideSpacers(BaseModel):
    """Suppl. Table 1's released ``N20`` oligos, read into per-gene spacers."""

    model_config = ConfigDict(extra="forbid")

    #: lowercased gene name -> the one released spacer, when the release gives one
    by_gene: dict[str, str]
    #: lowercased gene name -> the several DIFFERENT spacers the release gives it
    ambiguous: dict[str, list[str]]
    n_oligo_rows: int
    #: spacer length -> how many rows carry one that long
    length_histogram: dict[int, int]

    def spacer_of(self, symbol: str) -> str | None:
        """The released spacer of a gene, or ``None`` when the release gives none or two.

        Matched case-insensitively, because the release writes ``muts`` in Suppl. Table
        3 and ``N20-mutS`` in Suppl. Table 1 for the same gene; a case difference in one
        symbol is not a second gene.
        """
        return self.by_gene.get(symbol.lower())


def read_guide_spacers(rows: Sequence[Sequence[str]]) -> GuideSpacers:
    """Read the ``N20-<gene>`` rows of Suppl. Table 1 into per-gene guide spacers.

    Every released oligo is ``CGGGGTACC`` + spacer + ``gttttagagctagaaatag``; the spacer
    is what lies between those two constant flanks, taken at the length the release
    gives it. Both flanks are required: a row that carries neither is not an oligo of
    this design and raises rather than being read as a spacer.
    """
    header = tuple(rows[0])
    if header != TABLE_1_HEADER:
        raise TableFormatError(
            f"Suppl. Table 1 header is {header}, expected {TABLE_1_HEADER}"
        )
    seen: dict[str, set[str]] = {}
    lengths: Counter[int] = Counter()
    n_rows = 0
    for row in rows[1:]:
        name = row[0].strip()
        if not name.upper().startswith("N20"):
            continue
        n_rows += 1
        sequence = re.sub(r"\s+", "", row[1] if len(row) > 1 else "").upper()
        if not (sequence.startswith(OLIGO_PREFIX) and sequence.endswith(OLIGO_SUFFIX)):
            raise ReleaseContentError(
                f"Suppl. Table 1 row {name!r} is not an N20 oligo of the released "
                f"design ({OLIGO_PREFIX} + spacer + {OLIGO_SUFFIX.lower()})"
            )
        spacer = sequence[len(OLIGO_PREFIX) : -len(OLIGO_SUFFIX)]
        if not re.fullmatch(r"[ACGT]+", spacer):
            raise ReleaseContentError(
                f"Suppl. Table 1 row {name!r} has a non-ACGT spacer {spacer!r}"
            )
        lengths[len(spacer)] += 1
        gene = name.split("-", 1)[1] if "-" in name else name
        seen.setdefault(gene.lower(), set()).add(spacer)
    return GuideSpacers(
        by_gene={
            gene: next(iter(spacers))
            for gene, spacers in seen.items()
            if len(spacers) == 1
        },
        ambiguous={
            gene: sorted(spacers) for gene, spacers in seen.items() if len(spacers) > 1
        },
        n_oligo_rows=n_rows,
        length_histogram=dict(sorted(lengths.items())),
    )


class ReleasedTarget(BaseModel):
    """One row of Suppl. Table 3 or 4, with both value cells parsed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: str
    category: str
    label: str
    protein: str
    growth_ratio: float | None
    growth_sd: float | None
    growth_cell: str
    pinene_ratio: float | None
    pinene_sd: float | None
    pinene_cell: str

    @property
    def growth_is_censored(self) -> bool:
        """The growth cell is the footnote's dash, a measured non-positive outcome."""
        return self.growth_cell == "-"

    @property
    def pinene_is_censored(self) -> bool:
        """The pinene cell is the footnote's dash."""
        return self.pinene_cell == "-"


def read_target_table(rows: Sequence[Sequence[str]], arm: Arm) -> list[ReleasedTarget]:
    """One ``ReleasedTarget`` per data row of Suppl. Table 3 or 4.

    A category heading is a merged row carrying ONE filled cell, which is structural
    rather than textual; a value cell is either a ``ratio ± sd`` pair or the footnote's
    dash, and anything else raises.
    """
    header = tuple(rows[0])
    if header != TARGET_TABLE_HEADER:
        raise TableFormatError(
            f"{arm} table header is {header}, expected {TARGET_TABLE_HEADER}"
        )
    out: list[ReleasedTarget] = []
    category = ""
    for row in rows[1:]:
        filled = [cell for cell in row if cell.strip()]
        if len(filled) == 1:
            category = filled[0]
            continue
        growth, pinene = row[2].strip(), row[3].strip()
        parsed: dict[str, Any] = {
            "arm": arm,
            "category": category,
            "label": row[0].strip(),
            "protein": row[1].strip(),
            "growth_cell": growth,
            "pinene_cell": pinene,
        }
        for name, cell in (("growth", growth), ("pinene", pinene)):
            match = RATIO_CELL.match(cell)
            if match is None and cell != "-":
                raise ReleaseContentError(
                    f"{arm} row {parsed['label']!r}: {name} cell {cell!r} is neither a "
                    "ratio ± sd pair nor the footnote's dash"
                )
            parsed[f"{name}_ratio"] = (
                float(match.group("ratio")) if match is not None else None
            )
            parsed[f"{name}_sd"] = (
                float(match.group("sd")) if match is not None else None
            )
        out.append(ReleasedTarget(**parsed))
    return out


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #
RULE_PINENE_RATIO = "pinene_ratio_is_a_dimensionless_product_ratio"

DROP_RULE_DESCRIPTIONS: dict[str, str] = {
    RULE_PINENE_RATIO: "the released pinene value is a dimensionless product RATIO "
    "with no absolute titer anywhere in this arm, measured in the 72 h SBMSN "
    "shake-flask production culture rather than in this store's 12 h tolerance "
    "culture. EnvironmentResponsePhenotype is a fitness or growth readout, so storing "
    "a production ratio there would mislabel it. All 40 cells are stored by "
    "PineneFoldChangeNiu2019Dataset on ConcentrationUnit.fold_change (issue #770, "
    "closed by the unit member #1077 landed)"
}


class DropRule(BaseModel):
    """One retention rule, the cells it removed, and the items it removed them for."""

    model_config = ConfigDict(extra="forbid")

    rule: str
    description: str
    n_cells: int
    items: list[str] = []


class DropLog(BaseModel):
    """Which released numeric cells are stored, and the ledger that explains the rest."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    released_numeric_cells: int
    stored_records: int
    dropped_cells: int
    censored_cells: int
    rules: list[DropRule]


class Retention(BaseModel):
    """The strains to store, the ledger, and the released values kept verbatim."""

    model_config = ConfigDict(extra="forbid")

    #: (arm, released label) -> the targets of that strain, in released order
    strains: list[tuple[str, str, list[StoredTarget]]]
    #: (arm, released label) -> the released growth ratio
    ratios: dict[str, float]
    drop_log: DropLog
    spacers: GuideSpacers
    released: dict[str, list[dict[str, Any]]]
    spacer_absences: dict[str, str]


def _strain_key(arm: str, label: str) -> str:
    """``<arm>/<released label>``, the key of one stored strain."""
    return f"{arm}/{label}"


def retain(
    targets: Mapping[Arm, Sequence[ReleasedTarget]],
    spacers: GuideSpacers,
    *,
    dataset_name: str,
) -> Retention:
    """Decide which released cells are stored, and account for every one that is not."""
    for arm, expected in (
        ("crispr_activation", N_ACTIVATION_TARGETS),
        ("crispr_interference", N_INTERFERENCE_TARGETS),
    ):
        if len(targets[arm]) != expected:  # type: ignore[index]
            raise ReleaseContentError(
                f"{arm}: {len(targets[arm])} data rows, the paper states {expected}"  # type: ignore[index]
            )

    strains: list[tuple[str, str, list[StoredTarget]]] = []
    ratios: dict[str, float] = {}
    spacer_absences: dict[str, str] = {}
    n_pinene = 0
    n_censored = 0
    released: dict[str, list[dict[str, Any]]] = {}

    for arm in ("crispr_activation", "crispr_interference"):
        rows = list(targets[arm])  # type: ignore[index]
        released[arm] = [row.model_dump() for row in rows]
        for row in rows:
            n_censored += (1 if row.growth_is_censored else 0) + (
                1 if row.pinene_is_censored else 0
            )
            if row.pinene_ratio is not None:
                n_pinene += 1
            if row.growth_ratio is None:
                continue
            spacer = spacers.spacer_of(row.label)
            if spacer is None:
                spacer_absences[_strain_key(arm, row.label)] = (
                    "the release gives this gene two DIFFERENT N20 spacers and no rule "
                    "for choosing between them"
                    if row.label.lower() in spacers.ambiguous
                    else "Suppl. Table 1 releases no N20 oligo under this gene's name"
                )
            target = StoredTarget(
                reported_symbol=row.label, strain_label=row.label, guide_sequence=spacer
            )
            strains.append((arm, row.label, [target]))
            ratios[_strain_key(arm, row.label)] = row.growth_ratio

    # the two combination strains, released in the MAIN tables only
    for arm, label in COMBINATION_LABELS.items():
        genes = label.split("-")
        ratio, _sd = COMBINATION_GROWTH[arm]
        n_pinene += 1
        combination: list[StoredTarget] = []
        for gene in genes:
            spacer = spacers.spacer_of(gene)
            if spacer is None:
                spacer_absences[f"{_strain_key(arm, label)}:{gene}"] = (
                    "the release gives this gene two DIFFERENT N20 spacers and no rule "
                    "for choosing between them"
                    if gene.lower() in spacers.ambiguous
                    else "Suppl. Table 1 releases no N20 oligo under this gene's name"
                )
            combination.append(
                StoredTarget(
                    reported_symbol=gene, strain_label=label, guide_sequence=spacer
                )
            )
        strains.append((arm, label, combination))
        ratios[_strain_key(arm, label)] = ratio

    n_growth_numeric = len(
        [
            row
            for arm in ("crispr_activation", "crispr_interference")
            for row in targets[arm]  # type: ignore[index]
            if row.growth_ratio is not None
        ]
    ) + len(COMBINATION_GROWTH)
    released_numeric = n_growth_numeric + n_pinene
    rules = [
        DropRule(
            rule=RULE_PINENE_RATIO,
            description=DROP_RULE_DESCRIPTIONS[RULE_PINENE_RATIO],
            n_cells=n_pinene,
            items=[],
        )
    ]
    drop_log = DropLog(
        dataset=dataset_name,
        released_numeric_cells=released_numeric,
        stored_records=len(strains),
        dropped_cells=released_numeric - len(strains),
        censored_cells=n_censored,
        rules=rules,
    )
    accounted = sum(rule.n_cells for rule in rules)
    if accounted != drop_log.dropped_cells:
        raise RuntimeError(
            f"drop accounting mismatch: rules total {accounted}, "
            f"{drop_log.dropped_cells} cells missing from the build"
        )
    return Retention(
        strains=strains,
        ratios=ratios,
        drop_log=drop_log,
        spacers=spacers,
        released=released,
        spacer_absences=spacer_absences,
    )


# --------------------------------------------------------------------------- #
# What the release holds and this dataset does not load
# --------------------------------------------------------------------------- #
NOT_LOADED: tuple[dict[str, Any], ...] = (
    {
        "what": "Suppl. Table 2: 374 called variants of the evolved isolate YZFP",
        "n_rows": 374,
        "n_writable_on_the_variant_leaves": 373,
        "reason": "the genotype is writable and the PHENOTYPE does not exist. Since the "
        "#835 leaves landed, 322 b-numbered rows take "
        "BacterialSequenceVariantPerturbation and 48 intergenic rows take "
        "BacterialSiteVariantPerturbation (373 of 374; the one blank-mutation-site row "
        "refuses on its missing coordinate, and 4 rows are neither b-numbered nor "
        "intergenic). But this release measures no number of YZFP: the two CRISPRa/i "
        "target tables are measured in the UNEVOLVED parent, the evolved strain's own "
        "tolerance and titer are attributed to reference [8], and Fig. 2's 182 qRT-PCR "
        "YZFP-over-parent transcript ratios are released as bar panels with no table. A "
        "genotype with no measured phenotype is not an Experiment",
        "issue": 731,
    },
    {
        "what": "every dashed cell of Suppl. Tables 3 and 4",
        "n_cells": 64,
        "reason": "the footnote defines a dash as 'no change or negative effect', and "
        "classifying each cell as numeric or dashed reproduces the main text's own "
        "both / growth-only / pinene-only tallies exactly for both arms, so a dash is "
        "a MEASURED non-positive outcome rather than an absent measurement. It still "
        "has neither a number nor a single ResponseCategory, because the footnote "
        "conflates 'no change' with 'negative effect', so it is left-censored",
        "issue": None,
    },
    {
        "what": "Suppl. Table 1: the 319 non-N20 primer rows",
        "reason": "cloning and sequencing primers of the vectors; the only rows this "
        "loader consumes are the N20 guide oligos, read for their spacers",
        "issue": None,
    },
    {
        "what": "Suppl. Figures 1 and 2 and the paper's Figs 1-3",
        "reason": "the mCherry CRISPRa/i validation, the growth curves of the evolved "
        "and wild strains, and the co-culture biocatalysis titers; none releases a "
        "per-strain number this dataset's record shape could carry",
        "issue": None,
    },
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/niuGenomicTranscriptionalChanges2019``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_si_path(data_root: str | None = None) -> Path:
    """The literature mirror's copy of the SI, the bytes a deposit reads by default."""
    return Path(data_root or _data_root()) / LIBRARY_DIR_REL / SI_LIBRARY_RELPATH


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def si_rel() -> str:
    """Mirror-relative path of the consumed SI file."""
    return f"data/{SI_FILENAME}"


def si_url() -> str:
    """HTTPS URL of the SI object in the PMC Article Datasets bucket."""
    return pmc_cloud_url(SI_BUCKET_KEY)


def si_retrieval(retrieved_at: str = RAW_RETRIEVED_AT) -> RetrievalRecord:
    """The recorded retrieval of the SI: one PMC Article Datasets object."""
    return RetrievalRecord(
        method=RetrievalMethod.pmc_cloud,
        source_url=si_url(),
        retriever="torchcell.literature.retrieve.pmc_cloud_object",
        params={"key": SI_BUCKET_KEY},
        sha256=SI_SHA256,
        retrieved_at=retrieved_at,
    )


def deposit_raw_mirror(
    *,
    si_path: str | Path | None = None,
    retrieved_at: str = RAW_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from already-retrieved bytes, plus ``manifest.json``.

    Idempotent by sha256: an existing mirror file with the recorded hash is left alone
    and a differing one raises rather than being overwritten. The default source is the
    literature mirror's own capture of the same object, which is byte-identical by
    sha256 to what the recorded retrieval re-fetches.
    """
    root = raw_mirror_dir(data_root)
    source = Path(si_path) if si_path is not None else library_si_path(data_root)
    got = _sha256(source)
    if got != SI_SHA256:
        raise RuntimeError(f"{source} sha256 mismatch: got {got}, expected {SI_SHA256}")
    dest = root / si_rel()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != SI_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        dest.write_bytes(source.read_bytes())
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=[
            ArtifactRecord(
                path=si_rel(),
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=SI_SHA256,
                source=si_url(),
                retrieval=si_retrieval(retrieved_at),
            )
        ],
        si_data_sources=[si_url()],
        si_expected=[
            f"{SI_FILENAME}: the article's ONE supplementary file, holding Suppl. "
            "Tables 1-4 and Suppl. Figures 1-2. Measured: the PMC Article Datasets "
            f"bucket carries exactly one supplementary object under {PMCID}.1 and the "
            "article's JATS declares exactly one <supplementary-material> element, so "
            "this is the complete publisher deposit rather than a partial capture",
            "the two combination strains are released in the MAIN tables (Tables 3 and "
            "4 of the article body), not in the SI, and are read from the OCR'd "
            "paper.md in the literature mirror",
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
    """The sha256 the manifest records for one mirror-relative path."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the manifest")


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
#: Records of the full build: 41 activation growth ratios (43 numeric minus the two
#: operon labels), 9 interference growth ratios, and the six-target interference
#: combination strain.
EXPECTED_RECORDS = 54
#: Released numeric cells of the whole paper, which the ledger accounts for whole.
RELEASED_NUMERIC_CELLS = 94


@register_dataset
class CrisprPineneToleranceNiu2019Dataset(ExperimentDataset):
    """Niu 2019 CRISPRa/CRISPRi growth ratios under 0.5% pinene in BW25113(PT5-dxs)."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = REFERENCE_STRAIN_NAME

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize. No genome parameter: the loader reads no annotation."""
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
        """The one consumed SI file."""
        return [SI_FILENAME]

    def download(self) -> None:
        """Link the mirror's SI file into ``raw/`` after verifying its pinned sha256."""
        data_root = _data_root()
        mirror = raw_mirror_dir(data_root)
        manifest = load_manifest(data_root)
        check_manifest_pin(si_rel(), manifest_sha256(manifest, si_rel()), SI_SHA256)
        source = mirror / si_rel()
        if not source.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {source}")
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(source, osp.join(self.raw_dir, SI_FILENAME), SI_SHA256)
        log.info(
            "Niu 2019 %s linked into %s (sha256 verified)", SI_FILENAME, self.raw_dir
        )

    @post_process
    def process(self) -> None:
        """Parse the SI into one record per stored strain; write the LMDB."""
        verify_raw_files(self.raw_dir, {SI_FILENAME: SI_SHA256})
        si = read_si(osp.join(self.raw_dir, SI_FILENAME))
        spacers = read_guide_spacers(si.tables[0])
        targets: dict[Arm, Sequence[ReleasedTarget]] = {
            "crispr_activation": read_target_table(si.tables[2], "crispr_activation"),
            "crispr_interference": read_target_table(
                si.tables[3], "crispr_interference"
            ),
        }
        retention = retain(targets, spacers, dataset_name=self.name)
        culture = environment()
        arms: tuple[Arm, ...] = ("crispr_activation", "crispr_interference")
        references = {
            arm: BacterialEnvironmentResponseExperimentReference(
                dataset_name=self.name,
                genome_reference=assembly_reference(
                    self.REFERENCE_STRAIN, background=NIU2019_BACKGROUND
                ),
                environment_reference=culture,
                phenotype_reference=reference_phenotype(arm),
            )
            for arm in arms
        }
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        index = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for arm_label, label, strain in tqdm(retention.strains, desc="niu2019"):
                arm = cast("Arm", arm_label)
                experiment = BacterialEnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=genotype(strain, arm),
                    environment=culture,
                    phenotype=phenotype(retention.ratios[_strain_key(arm, label)], arm),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, references[arm], publication, itxn),
                )
                index += 1
        env.close()
        interned_env.close()
        if index != retention.drop_log.stored_records:
            raise RuntimeError(
                f"wrote {index} records, the ledger counted "
                f"{retention.drop_log.stored_records}"
            )
        self._write_reports(retention)
        log.info(
            "Niu2019: wrote %d records of %d released numeric cells; dropped %s, "
            "censored %d",
            index,
            retention.drop_log.released_numeric_cells,
            {rule.rule: rule.n_cells for rule in retention.drop_log.rules},
            retention.drop_log.censored_cells,
        )

    def _write_reports(self, retention: Retention) -> None:
        """The drop log, the released values verbatim, the spacers, the not-loaded list."""
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(
            retention.drop_log.model_dump_json(indent=2)
        )
        (out / "released_growth_ratios.json").write_text(
            json.dumps(
                {
                    "note": "every released cell of Suppl. Tables 3 and 4, verbatim. "
                    "The stored phenotype is the log2 of the growth ratio, so the "
                    "ratio and its sample SD are kept here: an SD does not transform "
                    "with its statistic",
                    "arms": retention.released,
                    "combination_strains": {
                        arm: {
                            "label": COMBINATION_LABELS[arm],
                            "growth_ratio": COMBINATION_GROWTH[arm][0],
                            "growth_sd": COMBINATION_GROWTH[arm][1],
                            "source": "the article's main Table 3 / Table 4, not the SI",
                        }
                        for arm in COMBINATION_LABELS
                    },
                },
                indent=2,
            )
        )
        (out / "guide_spacers.json").write_text(
            json.dumps(
                {
                    "n_oligo_rows": retention.spacers.n_oligo_rows,
                    "n_genes_with_one_spacer": len(retention.spacers.by_gene),
                    "spacer_length_histogram": retention.spacers.length_histogram,
                    "genes_with_two_different_spacers": retention.spacers.ambiguous,
                    "stored_strains_without_a_spacer": retention.spacer_absences,
                },
                indent=2,
            )
        )
        (out / "not_loaded.json").write_text(json.dumps(list(NOT_LOADED), indent=2))

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "CrisprPineneToleranceNiu2019Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built tree
# --------------------------------------------------------------------------- #
def stored_records(dataset_root: str) -> Iterator[dict[str, Any]]:
    """Every stored record of a built tree, in LMDB order."""
    from torchcell.verification.runners import stream_records

    yield from stream_records(dataset_root)


def verify_build(
    dataset_root: str,
    *,
    genome: EcoliK12BW25113Genome | None = None,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the environment-response L0-L4 verifier on a built tree and write its report.

    The L4 universe is every GenBank locus of the BW25113 assembly the records pin, read
    from the record's own reference rather than from a strain named here.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )

    if genome is None:
        genome = bacterial_genome("ecoli", REFERENCE_STRAIN_NAME)  # type: ignore[assignment]
    assert genome is not None
    report = verify_environment_response_dataset_streaming(
        stored_records(dataset_root),
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{si_rel()}",
            citation_key=CITATION_KEY,
            sha256=SI_SHA256,
            method="Suppl. Tables 1, 3 and 4 of the article's one supplementary file: "
            "the N20 guide oligos and the per-target growth and pinene ratios. One "
            "BacterialEnvironmentResponseExperiment per stored strain, a CRISPRa or "
            "CRISPRi genotype against BW25113, reference = the no-guide control at a "
            "log2 ratio of 0",
            page="Suppl. Tables 1, 3 and 4 (mmc1.docx)",
            retrieved=RAW_RETRIEVED_AT,
        ),
        expected_count=expected_count,
        sgd_genes=set(genome.genbank.loci),
        resolve_gene_name=genome.resolve_gene_name,
    )
    from torchcell.verification.runners import _write_report

    _write_report(report, osp.join(dataset_root, "preprocess"))
    return report


# --------------------------------------------------------------------------- #
# The pinene fold-change store: the 40 released product ratios
# --------------------------------------------------------------------------- #
# The sibling store above holds the growth column of Suppl. Tables 3 and 4. Its ledger
# accounts for the pinene column as a dimensionless product RATIO that no phenotype
# could carry: ``ProductTiterPhenotype`` required a ``ConcentrationUnit`` and the enum
# had no dimensionless member, so there was no unit under which the number was honest.
# ``ConcentrationUnit.fold_change`` (#1077) is that member, so the 40 released pinene
# cells are a product-titer store of their own rather than a ledger line.
#
# A SEPARATE store, not a second phenotype on the growth records, because the two
# columns were measured in two different cultures: the growth ratio is a 12 h LB plus
# 0.5% pinene tolerance assay, and the pinene ratio is a 72 h SBMSN shake-flask
# production run under a dodecane overlay. One record carrying both would state that
# one culture produced both numbers.
# --------------------------------------------------------------------------- #
_CRISPR_METHODS = "Methods, '2.4. CRISPRa and CRISPRi of the mutant genes'"
_PRODUCTION = "Methods, '2.5. Pinene production in shake flasks'"
_GC = "Methods, '2.7. GC analysis'"
_MAIN_TABLES = "Tables 3 and 4 of the article body"

PINENE_RATIO_MEANING = _paper(
    "ratio of pinene production with the guide and without",
    "a The ratio of $\\mathrm { O D } _ { 6 0 0 }$ with CRISPRa and without. b The "
    "ratio of pinene production with CRISPRa and without.-: means no change or "
    "negative effect.",
    page=_ARMS,
    note="the stored quantity of this store: footnote b's column, a dimensionless "
    "ratio of the strain's pinene over the no-guide control's, stored on "
    "ConcentrationUnit.fold_change",
)
PRODUCTION_PLASMIDS = _paper(
    "pBbB2K-dCas9*-MCPSoxS + pTargetA-X + pMEVIGPS",
    "To investigate the effect of the activation and repression on pinene "
    "production, pBbB2K-dCas9\\*-MCPSoxS, pTargetA-X and pMEVIGPS were co-"
    "transferred into E. coli ${ \\tt B W } 2 5 1 1 3 ( { \\mathrm P } _ { "
    "\\mathrm { T 5 - d x } s } )$ and then analyzed the production of pinene "
    "in shake flasks.",
    page=_CRISPR_METHODS,
    note="the production chassis: the effector plasmid, the guide plasmid and the "
    "mevalonate-plus-pinene-synthase pathway plasmid in BW25113(PT5-dxs). Only "
    "pTargetA-X differs between strains, which is why it is the perturbation and the "
    "other two are background",
)
PRODUCTION_CONTROL = _paper(
    "BW25113(PT5-dxs) (pBbB2K-dCas9*-MCPSoxS, pTargetA, pMEVIGPS)",
    "The strain harboring the empty vector pTargetA, E. coli $\\mathtt { B W 2"
    " 5 1 1 3 ( P _ { T 5 - d x s } ) }$ (pBbB2K-dCas9\\*-MCPSoxS, pTargetA, "
    "pMEVIGPS) was set as the control.",
    page=_CRISPR_METHODS,
    note="the denominator of every stored ratio: the same chassis carrying the EMPTY "
    "guide vector, so the pathway and the effector are in the control too and the "
    "ratio isolates the guide",
)
SEED_CULTURE = _paper(
    "overnight 5 mL LB falcon-tube culture at 37 C and 200 rpm",
    "A single colony was inoculated into $5 \\mathrm { m L }$ of LB medium in a "
    "falcon tube, which was cultured overnight at $3 7 ^ { \\circ } \\mathrm { C "
    "}$ and $2 0 0 \\mathrm { r p m }$ .",
    page=_PRODUCTION,
    note="the pre-culture of the production run, on PreCultureSource.overnight_culture; "
    "the paper writes 'overnight' and no hour count, so the duration is a typed gap",
)
SBMSN_FORMULATION = _paper(
    "sucrose 20, peptone 12, yeast extract 24, KH2PO4 1.7, K2HPO4 211.42, "
    "MgCl2.6H2O 1, ammonium oxalate 1.42 (g/L), Tween-80 2 mL",
    "SBMSN medium $\\mathrm { ( p H } 7 . 0 )$ containing the following $( "
    "\\mathbf { g } / \\mathbf { L } )$ : sucrose 20, peptone 12, yeast extract"
    " 24, $\\mathrm { K H _ { 2 } P O _ { 4 } } 1 . 7 ,$ $\\mathrm { K _ { 2 } "
    "H P O _ { 4 } }$ 211.42, $\\mathrm { M g C l _ { 2 } { \\cdot } 6 H _ { 2 "
    "} O }$ 1, ammonium oxalate 1.42, and Tween-$8 0 \\ 2 \\mathrm { m L }$ .",
    page=_PRODUCTION,
    note="the production medium, stored component by component at the released "
    "amounts. The K2HPO4 amount is the published one: 211.42 g/L is implausibly high "
    "for a medium and the publisher's own text layer carries the same digits, so it is "
    "stored verbatim with this note rather than corrected to a guess",
)
MAIN_CULTURE = _paper(
    "50 mL SBMSN inoculated to OD600 0.1, grown at 37 C and 200 rpm to OD600 0.8",
    "The main cultures were then incubated at $3 7 ^ { \\circ } \\mathrm { C }$"
    " and $2 0 0 \\mathrm { r p m }$ until an ${ \\mathrm { O D } } _ { 6 0 0 "
    "}$ of 0.8 was reached.",
    page=_PRODUCTION,
    note="the pre-induction growth phase. Environment.temperature holds the PRODUCTION "
    "phase's 30 C because that is the phase the titer accumulates in; this phase is "
    "recorded here and in the culture format's provenance",
)
INDUCTION = _paper(
    "1 mM IPTG and 200 nM anhydrotetracycline, with a 20% dodecane overlay",
    "Then, the cultures were induced with $^ { 1 \\mathrm { m M } }$ IPTG and $2"
    " 0 0 \\mathrm { n M }$ anhydrous tetracycline, and overlaid with $2 0 \\%$ "
    "dodecane to trap pinene.",
    page=_PRODUCTION,
    note="the two inducers, which are medium components for the reason the tolerance "
    "assay's aTc is (constant across the dataset, present in the control, and what "
    "switches the genome-side perturbation on), and the dodecane overlay, which is an "
    "environment perturbation on the Brunk 2016 precedent",
)
PRODUCTION_INCUBATION = _paper(
    72.0,
    "After induction, the cultures were incubated at $3 0 ^ { \\circ } \\mathrm"
    " { C }$ and $1 3 0 \\mathrm { r p m }$ for $^ { 7 2 \\mathrm { h } }$ .",
    page=_PRODUCTION,
    note="the production phase every stored ratio is read at the end of: 72 h at 30 C "
    "and 130 rpm, which is Environment.duration_hours, temperature and the culture "
    "format's shaking",
)
GC_FID = _paper(
    "GC-FID",
    "The samples were analyzed by GC-FID by using a standard curve of $\\alpha$ "
    "-pinene (Sigma Aldrich).",
    page=_GC,
    note="the quantification method stored on every phenotype. The standard is "
    "alpha-pinene while the released column is named 'pinene', so the product keeps "
    "the released name and its typed identity gap rather than being resolved to the "
    "standard's structure",
)
COMBINATION_PINENE_ACTIVATION = _paper(
    1.71,
    "<td>flgFGH-sufBCDS- dusB-rpoA-yehA-</td><td></td><td>3.55 ± 0.03</td>"
    "<td>1.71 ± 0.02</td>",
    page=_MAIN_TABLES,
    note="the activation combination strain's pinene cell, released in the main table "
    "only. The row label is truncated by the OCR; the six genes are the prose's, which "
    "ACTIVATION_COMBINATION quotes",
)
COMBINATION_PINENE_INTERFERENCE = _paper(
    1.13,
    "<td>ydiJ-yjbQ-prpR-marR-fabR-cedA</td><td></td><td>1.26 ± 0.02</td>"
    "<td>1.13 ± 0.02</td>",
    page=_MAIN_TABLES,
    note="the interference combination strain's pinene cell, released in the main "
    "table only",
)

#: Every sourced value of the pinene store, for the same mirror audit.
PINENE_SOURCED_VALUES: tuple[SourcedValue, ...] = (
    PINENE_RATIO_MEANING,
    PRODUCTION_PLASMIDS,
    PRODUCTION_CONTROL,
    SEED_CULTURE,
    SBMSN_FORMULATION,
    MAIN_CULTURE,
    INDUCTION,
    PRODUCTION_INCUBATION,
    GC_FID,
    COMBINATION_PINENE_ACTIVATION,
    COMBINATION_PINENE_INTERFERENCE,
)

#: The main-table pinene cell of each combination strain, verbatim (ratio, sample SD).
COMBINATION_PINENE: dict[Arm, tuple[float, float]] = {
    "crispr_activation": (float(COMBINATION_PINENE_ACTIVATION.value), 0.02),
    "crispr_interference": (float(COMBINATION_PINENE_INTERFERENCE.value), 0.02),
}


NIU2019_SBMSN_INDUCED = Media(
    name="SBMSN with 1 mM IPTG and 200 nM anhydrotetracycline (Niu 2019 pinene "
    "production), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium=None,
    components=[
        MediaComponent(
            compound=resolved_compound("sucrose"),
            role=MediaComponentRole.carbon_source,
            concentration=Concentration(value=20.0, unit=ConcentrationUnit.g_per_l),
            provenance=[SBMSN_FORMULATION],
        ),
        MediaComponent(
            compound=resolved_compound("peptone"),
            role=MediaComponentRole.complex_ingredient,
            concentration=Concentration(value=12.0, unit=ConcentrationUnit.g_per_l),
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[SBMSN_FORMULATION],
        ),
        MediaComponent(
            compound=resolved_compound("yeast extract"),
            role=MediaComponentRole.complex_ingredient,
            concentration=Concentration(value=24.0, unit=ConcentrationUnit.g_per_l),
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[SBMSN_FORMULATION],
        ),
        MediaComponent(
            compound=resolved_compound("potassium dihydrogen phosphate"),
            role=MediaComponentRole.buffer,
            concentration=Concentration(value=1.7, unit=ConcentrationUnit.g_per_l),
            provenance=[SBMSN_FORMULATION],
        ),
        MediaComponent(
            compound=resolved_compound("dipotassium hydrogen phosphate"),
            role=MediaComponentRole.buffer,
            concentration=Concentration(value=211.42, unit=ConcentrationUnit.g_per_l),
            provenance=[SBMSN_FORMULATION],
            note="the published amount, stored verbatim. 211.42 g/L is implausibly "
            "high for a buffer and the publisher's text layer carries the same digits, "
            "so nothing is corrected: a plausible-looking 11.42 would be our number, "
            "not theirs",
        ),
        MediaComponent(
            compound=resolved_compound("magnesium chloride hexahydrate"),
            role=MediaComponentRole.bulk_salt,
            concentration=Concentration(value=1.0, unit=ConcentrationUnit.g_per_l),
            provenance=[SBMSN_FORMULATION],
        ),
        MediaComponent(
            compound=resolved_compound("ammonium oxalate"),
            role=MediaComponentRole.other,
            concentration=Concentration(value=1.42, unit=ConcentrationUnit.g_per_l),
            provenance=[SBMSN_FORMULATION],
            note="the release states the amount and not the role; a complex medium "
            "already carrying peptone and yeast extract makes 'nitrogen source' an "
            "inference, so the role stays 'other'",
        ),
        MediaComponent(
            compound=resolved_compound("Tween-80"),
            role=MediaComponentRole.other,
            concentration=Concentration(value=0.2, unit=ConcentrationUnit.percent_v_v),
            provenance=[SBMSN_FORMULATION],
            note="the release writes '2 mL' inside a list whose stated basis is per "
            "litre, so 2 mL/L is 0.2% v/v by that basis. ConcentrationUnit has no "
            "volume-per-volume amount member, and Tween-80 is a liquid, so the v/v "
            "percent is the unit that carries the released number without a guess",
        ),
        MediaComponent(
            compound=resolved_compound("IPTG"),
            role=MediaComponentRole.other,
            concentration=Concentration(value=1.0, unit=ConcentrationUnit.millimolar),
            provenance=[INDUCTION],
            note="the inducer of the pMEVIGPS pathway, a medium component for the "
            "reason the tolerance assay's aTc is: it is constant across the dataset, "
            "it is in the control, and it switches the genome-side construct on rather "
            "than being the variable the screen contrasts",
        ),
        MediaComponent(
            compound=resolved_compound("anhydrotetracycline"),
            role=MediaComponentRole.other,
            concentration=Concentration(value=200.0, unit=ConcentrationUnit.nanomolar),
            provenance=[INDUCTION],
            note="the inducer of the dCas9*-MCPSoxS effector, on the same reasoning",
        ),
    ],
)
"""The production medium: SBMSN at the released amounts plus the two inducers.

``base_medium`` is None because SBMSN is released as a complete recipe and is not a
derivative of any medium the shared library holds, so naming a base would assert a
derivation the paper does not state.
"""

NIU2019_LB_SEED = Media(
    name="LB, formulation not stated (Niu 2019 production seed culture), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium="LB",
    components=_lb_ingredients(SEED_CULTURE),
)
"""The seed culture's medium: LB with no inducer and no stated formulation."""


def dodecane_overlay() -> SmallMoleculePerturbation:
    """The 20% dodecane overlay the pinene is trapped in and read out of.

    An environment perturbation rather than a medium component, on the Brunk 2016
    precedent: it is a second liquid phase added to the culture at induction, not a
    nutrient of the medium, and it is what the GC sample is drawn from.
    """
    return SmallMoleculePerturbation(
        compound=resolved_compound("dodecane"),
        concentration=Concentration(value=20.0, unit=ConcentrationUnit.percent_v_v),
    )


def production_culture_format() -> CultureFormat:
    """The shake-flask format: 50 mL, 130 rpm after induction, a fixed 72 h endpoint."""
    return CultureFormat(
        vessel="shake flask",
        working_volume_ul=50_000.0,
        shaking_rpm=130.0,
        inoculum_od600=0.1,
        endpoint=EndpointRule.fixed_duration,
        provenance=[MAIN_CULTURE, INDUCTION, PRODUCTION_INCUBATION],
    )


def production_pre_culture() -> PreCulture:
    """The overnight 5 mL LB seed culture; its duration is a typed gap."""
    return PreCulture(
        source=PreCultureSource.overnight_culture,
        medium=NIU2019_LB_SEED,
        duration_hours=None,
        provenance=[SEED_CULTURE],
        provenance_gaps=[
            ProvenanceGap(
                field="duration_hours",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=Provenance(
                    source_uri=PAPER_MD,
                    citation_key=CITATION_KEY,
                    sha256=PAPER_MD_SHA256,
                    method="section 2.5 read in full for an hour count of the seed "
                    "culture",
                    page=_PRODUCTION,
                ),
                note="the paper writes 'cultured overnight' and states no hours, so no "
                "duration is asserted",
            )
        ],
    )


def production_environment() -> CultureEnvironment:
    """The one production culture every stored ratio was measured in.

    ``temperature`` and ``duration_hours`` are the POST-induction phase (30 C, 72 h),
    because that is the phase the pinene accumulates in and the ratio is read at the
    end of. The 37 C pre-induction growth to OD600 0.8 is a sourced value on the
    culture format rather than a second temperature, since an Environment states one.
    """
    perturbations: list[EnvironmentPerturbationType] = [dodecane_overlay()]
    return CultureEnvironment(
        media=NIU2019_SBMSN_INDUCED,
        temperature=Temperature(value=30.0),
        culture_format=production_culture_format(),
        pre_culture=production_pre_culture(),
        perturbations=perturbations,
        aerobicity="aerobic",
        duration_hours=float(PRODUCTION_INCUBATION.value),
    )


NIU2019_PRODUCTION_BACKGROUND = BacterialStrainBackground(
    name="BW25113(PT5-dxs) (pBbB2K-dCas9*-MCPSoxS, pMEVIGPS)",
    reference_strain=REFERENCE_STRAIN_NAME,
    assembly_set=cast("Any", BACTERIAL_ASSEMBLY_SETS[REFERENCE_STRAIN_NAME]),
    parents=["BW25113(PT5-dxs)"],
    construction=None,
    genotype_statement="BW25113(PT5-dxs) (pBbB2K-dCas9*-MCPSoxS, pMEVIGPS)",
    alleles=[],
    provenance=[
        HOST_STRAIN,
        DXS_PARENT_DESIGN,
        PRODUCTION_PLASMIDS,
        PRODUCTION_CONTROL,
    ],
    provenance_gaps=[
        ProvenanceGap(
            field="construction",
            reason=ProvenanceGapReason.not_reported_by_primary,
            looked_in=Provenance(
                source_uri=PAPER_MD,
                citation_key=CITATION_KEY,
                sha256=PAPER_MD_SHA256,
                method="full Methods and Results read for a construction sentence for "
                "the PT5-dxs parent and the two shared plasmids",
                page="Methods; Results",
            ),
            note="the paper names the parent as BW25113(PT5-dxs) and lists the three "
            "co-transferred plasmids, and never says how the PT5-dxs cassette was "
            "built or where it sits",
        )
    ],
)
"""The production chassis as a background: the parent plus the two SHARED plasmids.

The guide plasmid pTargetA-X is deliberately absent: it is the one part that differs
between strains, so it is the perturbation, and the control carries the empty pTargetA
in its place. The effector plasmid and the pathway plasmid are in every strain of the
store AND in the control, which is what makes them background.
"""


PINENE_TITER_UNIT_DETAIL = (
    "the released column is a dimensionless ratio of the strain's pinene over the "
    "no-guide control's, so every titer is stored on ConcentrationUnit.fold_change "
    "with the control at 1.0; no absolute titer is released anywhere in this paper's "
    "CRISPRa/i arms"
)


def pinene_phenotype(ratio: float, sd: float) -> ProductTiterPhenotype:
    """One released pinene cell, stored as released: the ratio and its sample SD."""
    return ProductTiterPhenotype(
        product=resolved_compound("pinene"),
        titer=ratio,
        titer_unit=ConcentrationUnit.fold_change,
        titer_uncertainty=sd,
        titer_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=N_SAMPLES,
        sample_unit=SampleUnit.biological_replicate,
        quantification_method=str(GC_FID.value),
        titer_censoring=Censoring.uncensored,
    )


def pinene_reference_phenotype() -> ProductTiterPhenotype:
    """The no-guide control: the denominator of every stored ratio, which is 1.0."""
    return ProductTiterPhenotype(
        product=resolved_compound("pinene"),
        titer=1.0,
        titer_unit=ConcentrationUnit.fold_change,
        titer_censoring=Censoring.uncensored,
        quantification_method=str(GC_FID.value),
        provenance_gaps=[
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the 1.0 baseline is what the ratio's denominator IS, not a "
                "measured set of control replicates: the paper releases the control's "
                "dispersion and replicate design nowhere, and the empty-vector "
                "strain's absolute titer nowhere either",
            )
            for field in (
                "titer_uncertainty",
                "titer_uncertainty_type",
                "n_samples",
                "sample_unit",
            )
        ],
    )


class PineneLedger(BaseModel):
    """What the pinene store holds, and every released pinene cell it does not."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    released_numeric_cells: int
    stored_records: int
    dropped_cells: int
    censored_cells: int
    growth_cells_in_the_sibling_store: int
    rules: list[DropRule] = []


class PineneRetention(BaseModel):
    """The strains to store, their released cells, and the ledger."""

    model_config = ConfigDict(extra="forbid")

    #: (arm, released label, the strain's targets in released order)
    strains: list[tuple[str, str, list[StoredTarget]]]
    #: ``<arm>/<released label>`` -> (released ratio, released sample SD)
    titers: dict[str, tuple[float, float]]
    ledger: PineneLedger
    released: dict[str, list[dict[str, Any]]]
    spacer_absences: dict[str, str]


def retain_pinene(
    targets: Mapping[Arm, Sequence[ReleasedTarget]],
    spacers: GuideSpacers,
    *,
    dataset_name: str,
) -> PineneRetention:
    """Every released pinene cell that carries a number, and nothing else.

    A dash is not stored: the footnote defines it as 'no change or negative effect',
    which conflates a null with a decrease, so it is a MEASURED non-positive outcome
    with neither a number nor a single category. It is counted as censored here for the
    same reason the sibling store counts it.
    """
    for arm, expected in (
        ("crispr_activation", N_ACTIVATION_TARGETS),
        ("crispr_interference", N_INTERFERENCE_TARGETS),
    ):
        if len(targets[arm]) != expected:  # type: ignore[index]
            raise ReleaseContentError(
                f"{arm}: {len(targets[arm])} data rows, the paper states {expected}"  # type: ignore[index]
            )

    strains: list[tuple[str, str, list[StoredTarget]]] = []
    titers: dict[str, tuple[float, float]] = {}
    spacer_absences: dict[str, str] = {}
    released: dict[str, list[dict[str, Any]]] = {}
    n_censored = 0
    n_growth = 0

    for arm in ("crispr_activation", "crispr_interference"):
        rows = list(targets[arm])  # type: ignore[index]
        released[arm] = [row.model_dump() for row in rows]
        for row in rows:
            n_censored += 1 if row.pinene_is_censored else 0
            n_growth += 1 if row.growth_ratio is not None else 0
            if row.pinene_ratio is None:
                continue
            if row.pinene_sd is None:
                raise TableFormatError(
                    f"{arm} {row.label}: a numeric pinene cell with no SD"
                )
            spacer = spacers.spacer_of(row.label)
            if spacer is None:
                spacer_absences[_strain_key(arm, row.label)] = (
                    "the release gives this gene two DIFFERENT N20 spacers and no rule "
                    "for choosing between them"
                    if row.label.lower() in spacers.ambiguous
                    else "Suppl. Table 1 releases no N20 oligo under this gene's name"
                )
            strains.append(
                (
                    arm,
                    row.label,
                    [
                        StoredTarget(
                            reported_symbol=row.label,
                            strain_label=row.label,
                            guide_sequence=spacer,
                        )
                    ],
                )
            )
            titers[_strain_key(arm, row.label)] = (row.pinene_ratio, row.pinene_sd)

    # the two combination strains, whose pinene cell is in the MAIN tables only
    for arm, label in COMBINATION_LABELS.items():
        combination: list[StoredTarget] = []
        for gene in label.split("-"):
            spacer = spacers.spacer_of(gene)
            if spacer is None:
                spacer_absences[f"{_strain_key(arm, label)}:{gene}"] = (
                    "the release gives this gene two DIFFERENT N20 spacers and no rule "
                    "for choosing between them"
                    if gene.lower() in spacers.ambiguous
                    else "Suppl. Table 1 releases no N20 oligo under this gene's name"
                )
            combination.append(
                StoredTarget(
                    reported_symbol=gene, strain_label=label, guide_sequence=spacer
                )
            )
        strains.append((arm, label, combination))
        titers[_strain_key(arm, label)] = COMBINATION_PINENE[arm]

    ledger = PineneLedger(
        dataset=dataset_name,
        released_numeric_cells=len(strains),
        stored_records=len(strains),
        dropped_cells=0,
        censored_cells=n_censored,
        growth_cells_in_the_sibling_store=n_growth + len(COMBINATION_GROWTH),
        rules=[],
    )
    if ledger.dropped_cells != sum(rule.n_cells for rule in ledger.rules):
        raise RuntimeError("drop accounting mismatch in the pinene ledger")
    return PineneRetention(
        strains=strains,
        titers=titers,
        ledger=ledger,
        released=released,
        spacer_absences=spacer_absences,
    )


#: Records of the pinene store: 32 activation and 6 interference SI cells, plus the two
#: combination strains whose pinene cell is released in the main tables only.
PINENE_EXPECTED_RECORDS = 40
PINENE_DATASET_ROOT_REL = "data/torchcell/pinene_fold_change_niu2019"


@register_dataset
class PineneFoldChangeNiu2019Dataset(ExperimentDataset):
    """Niu 2019 CRISPRa/CRISPRi pinene production ratios in BW25113(PT5-dxs)."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = REFERENCE_STRAIN_NAME

    def __init__(
        self,
        root: str = PINENE_DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize. No genome parameter: the loader reads no annotation."""
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
        """The one consumed SI file."""
        return [SI_FILENAME]

    def download(self) -> None:
        """Link the mirror's SI file into ``raw/`` after verifying its pinned sha256."""
        data_root = _data_root()
        mirror = raw_mirror_dir(data_root)
        manifest = load_manifest(data_root)
        check_manifest_pin(si_rel(), manifest_sha256(manifest, si_rel()), SI_SHA256)
        source = mirror / si_rel()
        if not source.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {source}")
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(source, osp.join(self.raw_dir, SI_FILENAME), SI_SHA256)
        log.info(
            "Niu 2019 %s linked into %s (sha256 verified)", SI_FILENAME, self.raw_dir
        )

    @post_process
    def process(self) -> None:
        """One record per released pinene cell; write the LMDB."""
        verify_raw_files(self.raw_dir, {SI_FILENAME: SI_SHA256})
        si = read_si(osp.join(self.raw_dir, SI_FILENAME))
        spacers = read_guide_spacers(si.tables[0])
        targets: dict[Arm, Sequence[ReleasedTarget]] = {
            "crispr_activation": read_target_table(si.tables[2], "crispr_activation"),
            "crispr_interference": read_target_table(
                si.tables[3], "crispr_interference"
            ),
        }
        retention = retain_pinene(targets, spacers, dataset_name=self.name)
        culture = production_environment()
        reference = ProductTiterExperimentReference(
            dataset_name=self.name,
            genome_reference=assembly_reference(
                self.REFERENCE_STRAIN, background=NIU2019_PRODUCTION_BACKGROUND
            ),
            environment_reference=culture,
            phenotype_reference=pinene_reference_phenotype(),
        )
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        index = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for arm_label, label, strain in tqdm(
                retention.strains, desc="niu2019-pinene"
            ):
                arm = cast("Arm", arm_label)
                ratio, sd = retention.titers[_strain_key(arm, label)]
                experiment = ProductTiterExperiment(
                    dataset_name=self.name,
                    genotype=genotype(strain, arm),
                    environment=culture,
                    phenotype=pinene_phenotype(ratio, sd),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )
                index += 1
        env.close()
        interned_env.close()
        if index != retention.ledger.stored_records:
            raise RuntimeError(
                f"wrote {index} records, the ledger counted "
                f"{retention.ledger.stored_records}"
            )
        self._write_pinene_reports(retention)
        log.info(
            "Niu2019 pinene: wrote %d records of the release's %d numeric pinene "
            "cells; %d dashed cells censored; %d growth cells are the sibling store's",
            index,
            retention.ledger.released_numeric_cells,
            retention.ledger.censored_cells,
            retention.ledger.growth_cells_in_the_sibling_store,
        )

    def _write_pinene_reports(self, retention: PineneRetention) -> None:
        """The ledger, the released cells verbatim, and the not-loaded list."""
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(
            retention.ledger.model_dump_json(indent=2)
        )
        (out / "released_pinene_ratios.json").write_text(
            json.dumps(
                {
                    "note": "every released cell of Suppl. Tables 3 and 4, verbatim. "
                    "The stored titer is the pinene column's ratio and its sample SD, "
                    "both as released",
                    "arms": retention.released,
                    "combination_strains": {
                        arm: {
                            "label": COMBINATION_LABELS[arm],
                            "pinene_ratio": COMBINATION_PINENE[arm][0],
                            "pinene_sd": COMBINATION_PINENE[arm][1],
                            "source": "the article's main Table 3 / Table 4, not the SI",
                        }
                        for arm in COMBINATION_LABELS
                    },
                    "stored_strains_without_a_spacer": retention.spacer_absences,
                },
                indent=2,
            )
        )
        (out / "not_loaded.json").write_text(json.dumps(list(NOT_LOADED), indent=2))

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "PineneFoldChangeNiu2019Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built pinene tree
# --------------------------------------------------------------------------- #
def _record_strain_key(record: dict[str, Any]) -> str:
    """``<arm>/<released label>`` of one stored record, read back off its genotype."""
    perturbations = record["experiment"]["genotype"]["perturbations"]
    labels = {str(p["strain_label"]) for p in perturbations}
    directions = {str(p["expression_direction"]) for p in perturbations}
    if len(labels) != 1 or len(directions) != 1:
        raise RuntimeError(
            f"one record carries {sorted(labels)} and {sorted(directions)}: a stored "
            "strain is one released label in one arm"
        )
    direction = directions.pop()
    arm = next(key for key, value in DIRECTION_OF_ARM.items() if value == direction)
    return _strain_key(arm, labels.pop())


def _released_pinene_cells(data_root: str) -> dict[str, tuple[float, float]]:
    """Re-read every released pinene cell from the mirror bytes, keyed by strain."""
    si = read_si(raw_mirror_dir(data_root) / si_rel())
    cells: dict[str, tuple[float, float]] = {}
    for arm, table in (
        ("crispr_activation", si.tables[2]),
        ("crispr_interference", si.tables[3]),
    ):
        for row in read_target_table(table, cast("Arm", arm)):
            if row.pinene_ratio is None or row.pinene_sd is None:
                continue
            cells[_strain_key(arm, row.label)] = (row.pinene_ratio, row.pinene_sd)
    for arm, label in COMBINATION_LABELS.items():
        cells[_strain_key(arm, label)] = COMBINATION_PINENE[arm]
    return cells


def _l2_released_cells(
    records: Sequence[dict[str, Any]], data_root: str
) -> LevelResult:
    """L2: every stored titer and SD is its released cell, re-read from the mirror.

    An independent read: the records come out of the built LMDB and the cells are
    parsed again from the deposited ``mmc1.docx`` bytes, so a transcription that drifted
    between the parse and the store shows up here.
    """
    expected = _released_pinene_cells(data_root)
    problems: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        key = _record_strain_key(record)
        seen.add(key)
        phenotype = record["experiment"]["phenotype"]
        if key not in expected:
            problems.append({"index": index, "key": key, "problem": "not released"})
            continue
        ratio, sd = expected[key]
        if float(phenotype["titer"]) != ratio:
            problems.append(
                {
                    "index": index,
                    "key": key,
                    "stored": phenotype["titer"],
                    "released": ratio,
                }
            )
        if float(phenotype["titer_uncertainty"]) != sd:
            problems.append(
                {
                    "index": index,
                    "key": key,
                    "stored_sd": phenotype["titer_uncertainty"],
                    "released_sd": sd,
                }
            )
    missing = sorted(set(expected) - seen)
    passed = not problems and not missing
    return LevelResult(
        level=Level.L2,
        name="titers_match_the_released_pinene_cells",
        passed=passed,
        message=(
            f"{len(records)} records equal their Suppl. Table 3/4 or main-table pinene "
            "cell, ratio and sample SD, re-read from the deposited bytes"
            if passed
            else f"{len(problems)} mismatched cells and {len(missing)} released cells "
            "with no record"
        ),
        details={
            "n_released_cells": len(expected),
            "n_missing": len(missing),
            "missing": missing[:20],
            "worst": problems[:20],
        },
    )


def verify_pinene_build(
    dataset_root: str,
    data_root: str | None = None,
    *,
    expected_count: int = PINENE_EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the product-titer family gate plus this release's own cell-level L2.

    L4 is the family runner's: the perturbed-label containment against the pinned
    BW25113 assembly, which is a reported genome-reference finding and never a gate --
    two released labels (``flgFGH``, ``sufBCDS``) are operons, which is what the
    release keys those rows by.
    """
    from torchcell.verification.product_titer import verify_product_titer_dataset
    from torchcell.verification.runners import _write_report, load_records

    root = data_root if data_root is not None else _data_root()
    records = load_records(dataset_root)
    report = verify_product_titer_dataset(
        records,
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{si_rel()}",
            citation_key=CITATION_KEY,
            sha256=SI_SHA256,
            method="the pinene column of Suppl. Tables 3 and 4 (a ratio of the "
            "strain's 72 h shake-flask pinene over the no-guide control's, by GC-FID), "
            "plus the two combination strains' pinene cells from the article's main "
            "Tables 3 and 4. One ProductTiterExperiment per released cell on "
            "ConcentrationUnit.fold_change, reference = the empty-vector control at 1.0",
            page="Suppl. Tables 1, 3 and 4 (mmc1.docx); Tables 3 and 4",
            retrieved=RAW_RETRIEVED_AT,
        ),
        expected_count=expected_count,
        titer_unit=ConcentrationUnit.fold_change.value,
        titer_unit_detail=PINENE_TITER_UNIT_DETAIL,
        se_tol=1e-12,
        pathway_gene_counts={0},
        product_names={"pinene"},
    )
    report.add(_l2_released_cells(records, root))
    report.add(
        l3_convention(
            "the_reference_is_the_empty_vector_control_at_one",
            all(
                record["reference"]["phenotype_reference"]["titer"] == 1.0
                and record["reference"]["phenotype_reference"]["titer_unit"]
                == ConcentrationUnit.fold_change.value
                for record in records
            ),
            detail="the denominator of a released ratio is the control, so the "
            "reference titer is 1.0 fold change by construction and its dispersion is "
            "a typed gap rather than a number",
        )
    )
    statements = {
        (record["reference"]["genome_reference"]["background"] or {}).get(
            "genotype_statement"
        )
        for record in records
    }
    report.add(
        l3_convention(
            "the_pathway_plasmid_is_background_not_a_perturbation",
            statements == {NIU2019_PRODUCTION_BACKGROUND.genotype_statement},
            detail=f"stored background statements {sorted(map(str, statements))}; "
            "pMEVIGPS and pBbB2K-dCas9*-MCPSoxS are in every stored strain AND "
            "in the control, so they are the background; the guide plasmid pTargetA-X "
            "is the only part that differs and is the perturbation",
        )
    )
    _write_report(report, osp.join(dataset_root, "preprocess"))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI over the raw mirror and the module's two stores, growth and pinene."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(prog="python -m torchcell.datasets.ecoli.niu2019")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("deposit", help="deposit the raw mirror from the library capture")
    sub.add_parser("build", help="build (or load) the growth dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built growth dev-tree LMDB")
    sub.add_parser("build-pinene", help="build (or load) the pinene dev-tree LMDB")
    sub.add_parser("verify-pinene", help="run L0-L4 on the built pinene dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    root = osp.join(data_root, DATASET_ROOT_REL)
    pinene_root = osp.join(data_root, PINENE_DATASET_ROOT_REL)
    if args.command == "deposit":
        print(deposit_raw_mirror(data_root=data_root))
        return 0
    if args.command == "build":
        dataset = CrisprPineneToleranceNiu2019Dataset(root=root)
        print(f"len = {len(dataset)}")
        return 0
    if args.command == "build-pinene":
        pinene = PineneFoldChangeNiu2019Dataset(root=pinene_root)
        print(f"len = {len(pinene)}")
        return 0
    if args.command == "verify-pinene":
        pinene_report = verify_pinene_build(
            pinene_root, data_root, expected_count=PINENE_EXPECTED_RECORDS
        )
        print(pinene_report.summary())
        return 0 if pinene_report.passed else 1
    report = verify_build(root, data_root=data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
