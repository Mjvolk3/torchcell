# torchcell/datasets/scerevisiae/parsons2006
# [[torchcell.datasets.scerevisiae.parsons2006]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/parsons2006
# Test file: tests/torchcell/datasets/scerevisiae/test_parsons2006.py
r"""Parsons 2006 chemical-genetic compendium: env x geno -> barcode log2 ratio.

Parsons et al. 2006 (Cell 126:611, doi:10.1016/j.cell.2006.06.040, PMID 16901791)
screened the MATa haploid yeast deletion collection, pooled and read out by its uptag and
downtag barcodes on Agilent arrays, against 82 conditions: 75 compounds and 7 crude
natural-product extracts, each at the dose Supplementary Table 1 lists. The released
per-strain matrix is Supplementary Table 7 (``mmc8.xls``, "Raw data matrix (log2
ratios)"), 4,111 deletion strains x 82 conditions. The 75 compound conditions are stored;
the 7 crude extracts are held out (UNIDENTIFIED PERTURBING AGENTS, below).

WHAT THE NUMBER IS, AND ITS SIGN. The released value is log2(CONTROL / TREATED), the
opposite orientation of Vanacloig 2022's log2(inhibitor/control). The Methods: "ratios of
the control divided by sample intensities were computed for each tag" and the SI "The
ratio of control signal/treated sample signal corresponding to each barcode was averaged
for two fluor-reversed chips and reported as a log2 ratio". So a POSITIVE value is a
strain depleted under the drug (sensitive): "greater than 0.5, meaning that the drug
treatment leads to at least a 1.4-fold depletion of the deletion mutant relative to the
control". The value is stored AS RELEASED on ``environment_response`` with
``measurement_type=log2_ratio`` (the Hillenmeyer 2008 HET precedent for the same
control-over-treatment statistic), and the orientation is stated in ``units``; it is NOT
negated. Measured on the release: of the 5,550 cells whose Table S4 -log10 p exceeds 5,
99.5% carry a positive ratio, so the bytes agree with the stated orientation. A consumer
that pools this with a negative-is-sick dataset must orient first
(``[[torchcell.datasets.scerevisiae.parsons2006]]``).

The released value averages the up and down tags of one strain (the per-tag value is
itself the mean of two fluor-reversed chips). The Methods say a lone tag "was averaged
with the most probable value for a log2 ratio measurement (approximately zero)". Measured,
the 180 Table S7 strains with one Table S5 non-functional tag have a value SD of 0.530
against 0.471 for the rest (ratio 1.13); halving would put that ratio below 0.71 whatever
the correlation between tags, so the released values are NOT halved for those strains.
Hypothesis (untested): the halving step belongs to the clustering input only. The stored
value is the released one either way.

RECORD = one (deletion strain x condition) ``StrainEnvironmentResponseExperiment``:

- GENOTYPE: one ``BarcodedKanMxDeletionPerturbation`` (the pool is read by its barcodes;
  none is released, a typed gap) from "MATa haploid deletion mutants (~4800) ... obtained
  from Research Genetics"; the cassette kanMX4 is the YKO's (Giaever 2014, which also says
  the YKO strains were "made available to the yeast community by Research Genetics").
  Each strain is stored AS RELEASED, under Table S7's ORF cell, which is also its
  ``perturbed_gene_name`` (the table names strains by ORF only). No genome is consulted
  (#1045, owner rule 2026-10-12: "the data is the data"): an ORF the current annotation
  renames (YDL038C, YGR272C) or does not call a gene (14 ORFs, among them YER109C and
  YIL168W) keeps its released name and its records.
- REFERENCE GENOME: BY4741 on a ``StrainReferenceGenome``. The SI strain table states
  BY4741's genotype ("BY4741 MATa ura3∆0 leu2∆0 his3∆1 met15∆0"), which sources each
  allele; the paper never says the pool IS BY4741, so the background's ``provenance`` is a
  typed gap naming Giaever 2002 (the Hoepfner 2014 precedent).
- ENVIRONMENT: a ``CultureEnvironment``: liquid YPD (SC for sulfometuron methyl and
  brefeldin A, SC lacking lysine for thialysine), 30 C, shaken, inoculated at OD600 0.025
  from the frozen pool, read after 14 h + 24 h with fresh compound at T1, plus ONE
  ``SmallMoleculePerturbation``: the compound's PubChem identity from its Table S1 CAS
  number (``compound_identity_inputs/parsons2006.txt``) at its Table S1 screening
  concentration, in its Methods vehicle at the 1% of the solvent control.
- PHENOTYPE: ``EnvironmentResponsePhenotype``, log2 ratio as released;
  ``environment_response_p_value`` = 10^-x of the same cell of Supplementary Table 4
  ("Negative log10(p-value) of sensitive deletion strains", a modified t test of the
  strain in this condition against its other conditions; 300 is the release's cap);
  ``n_samples = 2`` (two fluor-reversed chips per tag), ``sample_unit=technical_replicate``.
- REFERENCE: per medium, the 1% solvent-control pool on the BY4741 background, log2 ratio 0.

CELLS WITH NO RECORD (counted in ``preprocess/dropped_records.json``), in rule order:

1. ``ratio_cell_blank_in_release``: the release's two-space blank, a cell whose tags were
   both removed as noise-dominated ("specific tags under a specific condition having an
   intensity smaller than median intensity of the background signal in the control ...
   were ignored"). 7,848 cells.
2. ``UNIDENTIFIED_COMPOUND_RULE`` (below): every released ratio of a crude-extract
   condition. 28,420 cells.

The ledger raises unless source cells = stored records + the cells of every rule.

UNIDENTIFIED PERTURBING AGENTS (database constitution principle 2, owner ruling
2026-10-15: "if you cannot tell me what goes into the flask so to speak, then we have
no hope of modeling!"; applied to these extracts by the owner 2026-10-16: a collection
code for an undefined mixture says nothing about what went into the flask). The
criterion is mechanical: a condition is unidentified iff its Table S1 name matches
``^Extract <collection code>$`` (``_EXTRACT_CODE``) and ``RELEASED_IDENTITIES`` (empty
today) has no entry for it; the loader refuses unless exactly the paper's seven crude
extracts match ("7 crude antifungal extracts, derived from different marine sponges and
microorganisms"; "Natural product extracts, pap B, agelasine E, and basiliskamide A were
from the collection of R. Andersen."; each Table 1 comment reads "antifungal natural
product extract"). The paper releases no composition and no accession for any of them,
only the code. The seven, with the released ratios held out (measured on the raw mirror
through the loader's own ``parse_ratio_matrix``, 2026-10-16): Extract 95-57 4,009,
Extract 6592 4,079, Extract 00-89 4,101, Extract 00-303C 4,111, Extract 00-243 4,075,
Extract 00-192 4,091, Extract 00-132 3,954; 28,420 in all. Their columns are never
built, and the ledger lists each under ``unidentified_compounds_held_out`` with its
Table S1 name, dose, vehicle, medium and record count, beside the quotes
(``unidentified_compound_sources``). Stored after the hold-out: 300,834 records over 75
conditions. The two compounds the paper purified from extracts 00-192 and 00-132
(stichloroside, theopalauamide) are their own conditions and stay stored.

STORED WITH A TYPED COMPOUND-IDENTITY GAP (2026.10.12, #1045 row 19): the two NAMED
natural products whose compound carries no InChIKey, CID or ChEBI id after resolution
(agelasine E with no PubChem compound, and stichloroside with no congener named) are
identified material and stay stored. ``resolved_compound`` attaches the typed
``inchikey`` gap it attaches to any uncurated label (``deferred_pending_source_review``,
the #726 route Piotrowski 2017 takes: "an uncurated label is a gap, not a drop"), and
``preprocess/dropped_records.json`` lists each condition under ``compound_identity_gaps``
with its record count (agelasine E 4,032, stichloroside 4,111).

NOT RELEASED AS RATIOS (counted in the same file, not drops): Table S4 lists 121 strains
more than Table S7, exactly Table S2's 121 multidrug-sensitive strains, which the authors
removed before clustering; they have p-values and no ratio, so they have no record.

DATA SOURCE: Supplementary Tables S1 (``mmc2.xls``), S2 (``mmc3.xls``), S4 (``mmc5.xls``)
and S7 (``mmc8.xls``), legacy BIFF workbooks retrieved from the Elsevier supplementary
store into the literature mirror and deposited from those bytes into
``$DATA_ROOT/torchcell-raw/parsonsExploringModeofActionBioactive2006/`` with a
``manifest.json``; the build reads only the raw mirror and verifies each sha256.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import os.path as osp
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import xlrd
from pydantic import BaseModel, ConfigDict, Field
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
from torchcell.datamodels.media import SC, YPD_LIQUID, dropout, restated
from torchcell.datamodels.schema import (
    AssayType,
    BackgroundAllele,
    BarcodedKanMxDeletionPerturbation,
    Compound,
    Concentration,
    ConcentrationUnit,
    CultureEnvironment,
    CultureFormat,
    EndpointRule,
    EnvironmentResponsePhenotype,
    EukaryoticStrainBackground,
    Experiment,
    ExperimentReference,
    Genotype,
    MatingType,
    MeasurementType,
    Media,
    PreCulture,
    PreCultureSource,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    Solvent,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    StrainReferenceGenome,
    Temperature,
)
from torchcell.datamodels.strain_background import (
    BRACHMANN_1998,
    GIAEVER_2002,
    KANMX4_CASSETTE,
    STANDARD_BY_GENOTYPES,
    pending_source_review,
    standard_allele,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.hoepfner2014 import ReleasedIdentity
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "parsonsExploringModeofActionBioactive2006"
DOI = "10.1016/j.cell.2006.06.040"
#: PubMed id: NCBI esearch on ``10.1016/j.cell.2006.06.040[doi]`` and on "Parsons AB[au]
#: AND Cell[jour] AND 2006[dp]" both return the single id 16901791 (2026-10-11), the
#: folder yp-data v1.0 curates this paper under.
PMID = "16901791"
TITLE = (
    "Exploring the Mode-of-Action of Bioactive Compounds by Chemical-Genetic Profiling "
    "in Yeast"
)
PII = "S0092867406009068"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
RETRIEVED_AT = "2026-10-11"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "dfc222acef11c4d66807946cb36f8e99f73de5d3af2715967674bc029193c886"
SI_MD = "si/si1.md"
SI_MD_SHA256 = "36fa1487de02e743eea74d6841d6de46a7fa4b411757eaf6598ee526900f854a"
GIAEVER_2014_KEY = "giaeverYeastDeletionCollection2014"
GIAEVER_2014_SHA256 = "4a177a8658be57938eb0e45e736762281793ac7799d358e3a3349af61acf1719"


class RawTable(BaseModel):
    """One released supplementary workbook: where it lives and what it must hash to."""

    model_config = ConfigDict(frozen=True)

    table: str
    filename: str
    library_rel: str
    sha256: str
    bytes: int
    title: str

    @property
    def raw_rel(self) -> str:
        """Path inside the raw mirror."""
        return f"data/{self.filename}"

    @property
    def source_url(self) -> str:
        """The Elsevier supplementary-object URL the literature mirror retrieved."""
        return f"https://ars.els-cdn.com/content/image/1-s2.0-{PII}-{self.filename}"


TABLE_S1 = RawTable(
    table="S1",
    filename="mmc2.xls",
    library_rel="si/si2.xls",
    sha256="959f14d2dab418a9b1dd1125e302b2e28c1de372d8d7ee249d369d30fb518353",
    bytes=45056,
    title="Supplementary Table 1.  Chemical compounds and extracts screened in this study.",
)
TABLE_S2 = RawTable(
    table="S2",
    filename="mmc3.xls",
    library_rel="si/si3.xls",
    sha256="766140a410419b19456235db2e4909e56b24e3523bcc0f6f7aa698ae13eb6870",
    bytes=27648,
    title="Supplementary Table 2.  Multi-drug sensitive deletion strains. ",
)
TABLE_S4 = RawTable(
    table="S4",
    filename="mmc5.xls",
    library_rel="si/si5.xls",
    sha256="0446eae63b5b08af7a1a94fa3b8d42d70f1c75b55a60e0aa6e03fea189753a01",
    bytes=5171712,
    title="Supplementary Table 4.  Negative log10(p-value) of sensitive deletion "
    "strains.  ",
)
TABLE_S7 = RawTable(
    table="S7",
    filename="mmc8.xls",
    library_rel="si/si8.xls",
    sha256="564e94c963424687edc24ed23b6e3c6882b70c246a2524057cf1dc5f393c848f",
    bytes=6239232,
    title="Supplementary Table 7.  Raw data matrix (log2 ratios).",
)
RAW_TABLES: tuple[RawTable, ...] = (TABLE_S1, TABLE_S2, TABLE_S4, TABLE_S7)

#: The released files this loader does NOT consume, and why, for ``si_expected``.
NOT_CONSUMED: tuple[str, ...] = (
    "mmc1.pdf -- the Supplemental Data (Experimental Procedures, figures, strain table); "
    "mirrored in the literature mirror (si/si1.pdf, OCR si/si1.md) and quoted, not read "
    "as data",
    "Table S3 (mmc4.xls) -- the top 50 tamoxifen-sensitive strains with GO terms: a "
    "ranking derived from Table S4, which this loader reads directly",
    "Tables S5 and S6 (mmc6.xls, mmc7.xls) -- the 204 non-functional tags and the 497 "
    "dubious ORFs the authors removed: lists of removals, not measurements",
    "Table S8 (mmc9.xls) -- the clustered matrix (3,418 strains, ratios below 0.5 set to "
    "0): derived from Table S7",
    "Table S9 (mmc10.xls) -- the 174 multidrug-resistant strains (166 in Table S7): a "
    "strain list derived from Table S7, kept as records and named in the dendron note",
)


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
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
            page="EXPERIMENTAL PROCEDURES / RESULTS",
        ),
    )


def _si(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned Supplemental Data OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI_MD,
            citation_key=CITATION_KEY,
            sha256=SI_MD_SHA256,
            method="MinerU 2.7.6 OCR of mmc1.pdf (torchcell-library mirror, si/si1.pdf)",
            page="Supplemental Experimental Procedures",
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced constants
# --------------------------------------------------------------------------- #
N_CONDITIONS = _paper(
    82,
    "We generated chemical-genetic profiles for 82 different conditions by screening "
    "them against the yeast haploid deletion collection.",
    note="Table S7 carries 82 condition columns and Table S1 82 compound rows",
)
N_STRAINS = _paper(
    4111,
    "applied to a data matrix X of 4111 deletion mutant strains by 82 compounds",
    note="the measured row count of Table S7; the PSMF input is the same matrix",
)
TABLE_S7_IS_THE_INPUT = _paper(
    "Table S7",
    "For the original data set used to generate the cluster see Table S7 and for the "
    "complete clustered data set see Table S8.",
)
RATIO_ORIENTATION = _paper(
    "log2(control / treated)",
    r"In each experiment, ${ \mathsf { l o g } } _ { 2 }$ ratios of the control divided "
    "by sample intensities were computed for each tag.",
)
SENSITIVE_IS_POSITIVE = _paper(
    "positive = depleted under treatment",
    r"the combined average ${ \mathsf { l o g } } _ { 2 }$ ratio (control/experiment) of "
    "both barcodes (uptag and downtag) corresponding to each strain is greater than "
    "0.5, meaning that the drug treatment leads to at least a 1.4-fold depletion of the "
    "deletion mutant relative to the control",
)
TAG_AVERAGE = _paper(
    "mean of the up and down tags",
    r"${ \mathsf { L o g } } _ { 2 }$ ratio measurements of up and down tags were "
    "averaged, when both were available.",
)
CHIP_REPLICATES = _si(
    2,
    "The ratio of control signal/treated sample signal corresponding to each barcode "
    r"was averaged for two fluor-reversed chips and reported as a $\log _ { 2 }$ ratio.",
    note="n_samples: the per-tag value is the mean of two fluor-reversed (dye-swapped) "
    "chips of ONE pooled culture, so the unit is a technical replicate; no biological "
    "replicate screen is released per condition",
)
NOISE_REMOVAL = _si(
    "noise-dominated tag measurements removed",
    "For each screen, individual, noise-dominated measurements were removed.",
    note="the blank cells of Table S7 (rule ratio_cell_blank_in_release)",
)
MULTIDRUG_SENSITIVE_REMOVED = _paper(
    121,
    "Prior to clustering, we processed the data by removing a set of 121 genes whose "
    "corresponding deletion mutant displayed statistically significant multidrug "
    "sensitivity (Table S2).",
    note="the 121 strains Table S4 lists and Table S7 does not; measured equal to Table "
    "S2's ORF set at build time",
)
SLOW_GROWER_CAVEAT = _paper(
    "control pool undergoes more doublings",
    "In our experiments, we run the control and drug-treated experiments for the same "
    "amount of time, such that the control pool undergoes more population doublings "
    "than the drug-treated (growth-inhibited) pool.",
    note="why slow growers read as resistant (negative); Table S9's 174 multidrug-"
    "resistant strains are kept as records and named in the dendron note",
)
P_VALUE_TEST = _paper(
    "modified t test against the strain's other conditions",
    r"a t test score and a corresponding p value were calculated based on the ${ "
    r"\mathsf { l o g } } _ { 2 }$ ratio measurements of growth of that strain in the "
    "presence of that particular drug, with respect to all other measurements obtained "
    "for that strain under all 82 conditions",
)
P_VALUE_CAP = _si(
    300.0,
    r"The p-values are reported as the negative $\log _ { 1 0 }$ of the pvalue, where $- "
    r"\log _ { 1 0 }$ (p-values) ${ > } 3 0 0$ are set to 300.",
    note="a released 300 is stored as 1e-300, the cap, not the true p",
)
SCREENING_CONCENTRATION = _paper(
    "Table S1 'Inhibitory Concentration'",
    "Inhibitory concentration refers to the screening concentration. Complete table "
    "including CAS numbers, FDA status, source, site of collection, and references in "
    "Table S1.",
)
POOL_SOURCE = _si(
    "MATa haploid deletion mutants (Research Genetics)",
    r"MATa haploid deletion mutants $( \sim 4 8 0 0 )$ were obtained from Research "
    "Genetics",
)
YKO_FROM_RESEARCH_GENETICS = SourcedValue(
    value="Research Genetics distributed the YKO strains",
    quote="directly made available to the yeast community by Research Genetics, SGTC, "
    "and ATCC",
    provenance=Provenance(
        source_uri="paper.md", citation_key=GIAEVER_2014_KEY, sha256=GIAEVER_2014_SHA256
    ),
    note="links the Research Genetics MATa collection to the YKO, whose cassette is "
    "KANMX4_CASSETTE (the same Giaever 2014 review)",
)
BY4741_GENOTYPE = _si(
    "BY4741",
    "BY4741 MATa ura3∆0 leu2∆0 his3∆1 met15∆0",
    note="the SI strain table's BY4741 line: sources each allele of the background. "
    "That the Research Genetics MATa pool is BY4741 is the literature-standard reading "
    "of Giaever 2002 (not mirrored), so the background's own provenance is a gap",
)
PRE_CULTURE_SOURCE = _si(
    PreCultureSource.frozen_stock,
    r"Strains were scraped from the plates, pooled in $\mathrm { Y P D } + 1 5 \%$ "
    r"glycerol, and frozen in aliquots at $- 8 0 \mathrm { { } ^ { \circ } C }$ .",
)
CULTURE_SENTENCE = (
    r"Aliquots were diluted to $\mathrm { O D } _ { 6 0 0 } = 0 . 0 2 5$ and incubated at "
    r"$3 0 \mathrm { { ^ \circ C } }$ shaking in the presence of compound or $1 \%$ "
    "solvent control in 5 or 10 mls YPD."
)
TEMPERATURE_C = _si(30.0, CULTURE_SENTENCE)
AEROBICITY = _si(
    "aerobic",
    CULTURE_SENTENCE,
    note="shaken liquid cultures; no anaerobic condition is described",
)
INOCULUM_OD600 = _si(
    0.025, CULTURE_SENTENCE, note="the same OD600 is restored at T1 (T1_T2_SENTENCE)"
)
SOLVENT_PERCENT = _si(
    1.0,
    CULTURE_SENTENCE,
    note="the stated level is the solvent CONTROL's; that the treated culture carries "
    "the same 1% of its vehicle is the matched-control reading, not a separate statement",
)
T1_T2_SENTENCE = (
    r"At 14 hours (T1) ODs were recorded and cultures where the growth of the "
    r"drug-treated culture was inhibited by $20 \mathrm { - } 3 0 \%$ compared to the "
    r"solvent control were diluted back to $\mathrm { O D } _ { 6 0 0 } = 0 . 0 2 5$ and "
    "incubated for a further 24 hours (T2) with fresh compound."
)
DURATION_HOURS = _si(
    38.0,
    T1_T2_SENTENCE,
    note="14 h to T1 plus 24 h to T2, when the cells were harvested for the barcode "
    "readout; the culture was diluted back once in between",
)
DOSE_ACCEPTANCE_RULE = _si(
    "20-30% growth inhibition at T1",
    T1_T2_SENTENCE,
    note="how the screening dose was accepted; DoseBasis has no IC20-IC30 member, so the "
    "rule is documented here and the dose itself is Table S1's released value",
)
SC_SENTENCE = (
    r"Sulfometuron methyl and brefeldin A were screened in synthetic complete media $( "
    r"0 . 6 7 \%$ yeast nitrogen base without amino acids, $0 . 2 \%$ amino acid add "
    "back) and thialysine was screened in synthetic complete media lacking lysine."
)
SC_MEDIUM = _si(
    "synthetic complete",
    SC_SENTENCE,
    note="no carbon source is stated for this SC; the shared SC composition (20 g/L "
    "D-glucose) is restated, so the glucose is the library's, not Parsons'",
)
SC_LYSINE_DROPOUT = _si("synthetic complete lacking lysine", SC_SENTENCE)
YPD_CULTURE = _si(
    "YPD",
    CULTURE_SENTENCE,
    note="the culture medium; the same section gives the YPD recipe for the pinning "
    "plates, 'YPD $2 \\%$ peptone, $1 \\%$ yeast extract, $2 \\%$ glucose) agar plates', "
    "the shared YPD_LIQUID composition",
)
SOLVENT_SENTENCE = (
    "Compounds were dissolved in DMSO except as noted: trifluoroperaine, pentamidine, "
    "nystatin, MMS, mitomycin C, hygromycin B, hydroxyurea, hydroxyethilhidrazine, "
    "hydrogen peroxide, fluconazole, emetine, doxycycline, caffeine, anisomycin, "
    "actinomycin, thialysin, sodium azide, and clomiphene were dissolved in water; "
    "phenantroline, brefeldin A, and artemisinin were dissolved in ethanol; wortmannin "
    "was dissolved in methanol; cisplatin was dissolved in a $0 . 9 \\%$ sodium chloride "
    "solution."
)
SOLVENTS = _paper("per-compound vehicle", SOLVENT_SENTENCE)
EXTRACT_SOLVENT = _paper("methanol", "Crude extracts were dissolved in methanol.")
BASILISKAMIDE_A = _paper(
    "basiliskamide A",
    "Natural product extracts, pap B, agelasine E, and basiliskamide A were from the "
    "collection of R. Andersen.",
    note="Table S1 says only 'Basiliskamide'; the screened congener is A",
)
STICHLOROSIDE_UNSPECIFIED = _paper(
    "stichloroside (congener not named)",
    "The compound isolated from extract 00-192 was identical to stichloroside "
    "(Kitagawa et al., 1981)",
    note="no congener and no CAS, so no structure can be assigned",
)
N_CRUDE_EXTRACTS = _paper(
    7,
    "Included in this group are 75 synthetic compounds and natural products, of which "
    "23 are FDA approved, and 7 crude antifungal extracts, derived from different "
    "marine sponges and microorganisms.",
)
EXTRACT_COLLECTION = _paper(
    "collection of R. Andersen",
    "Natural product extracts, pap B, agelasine E, and basiliskamide A were from the "
    "collection of R. Andersen.",
    note="the extracts are known only by the collection's codes; no composition or "
    "accession is released",
)
EXTRACT_TABLE_COMMENT = _paper(
    "antifungal natural product extract",
    "antifungal natural product extract",
    note="the Table 1 comment of each of the seven 'Extract <code>' rows",
)

SOURCED_VALUES: tuple[SourcedValue, ...] = (
    N_CONDITIONS,
    N_STRAINS,
    TABLE_S7_IS_THE_INPUT,
    RATIO_ORIENTATION,
    SENSITIVE_IS_POSITIVE,
    TAG_AVERAGE,
    CHIP_REPLICATES,
    NOISE_REMOVAL,
    MULTIDRUG_SENSITIVE_REMOVED,
    SLOW_GROWER_CAVEAT,
    P_VALUE_TEST,
    P_VALUE_CAP,
    SCREENING_CONCENTRATION,
    POOL_SOURCE,
    YKO_FROM_RESEARCH_GENETICS,
    BY4741_GENOTYPE,
    PRE_CULTURE_SOURCE,
    TEMPERATURE_C,
    AEROBICITY,
    INOCULUM_OD600,
    SOLVENT_PERCENT,
    DURATION_HOURS,
    DOSE_ACCEPTANCE_RULE,
    SC_MEDIUM,
    SC_LYSINE_DROPOUT,
    YPD_CULTURE,
    SOLVENTS,
    EXTRACT_SOLVENT,
    BASILISKAMIDE_A,
    STICHLOROSIDE_UNSPECIFIED,
    N_CRUDE_EXTRACTS,
    EXTRACT_COLLECTION,
    EXTRACT_TABLE_COMMENT,
)
"""Every module-level ``SourcedValue``; the test audits each quote against its mirror."""

MEASUREMENT_UNITS = (
    "log2(control / treated) barcode-array intensity ratio as released in Parsons 2006 "
    "Table S7: per tag the mean of two fluor-reversed chips, then the mean of the up "
    "and down tags; POSITIVE = the deletion strain is depleted in the drug-treated pool "
    "relative to the 1% solvent-control pool (sensitive), negative = enriched (slow "
    "growers read as resistant because the control pool undergoes more doublings). "
    "Pooled MATa haploid deletion collection, 38 h (14 h + 24 h with fresh compound "
    "after dilution at T1), 30 C shaken. environment_response_p_value = 10^-x of Table "
    "S4's -log10 p, a modified t test of this condition against the strain's other "
    "conditions (10% outliers excluded, tags weighted by their variance); -log10 p "
    "above 300 is released as 300"
)
REFERENCE_UNITS = (
    MEASUREMENT_UNITS
    + "; reference log2 ratio 0 is the 1% solvent-control pool itself (control / "
    "control), the same medium with no compound; the reference has no genotype slot, "
    "so its genome states the shared BY4741 background only"
)

# --------------------------------------------------------------------------- #
# The 82 conditions
# --------------------------------------------------------------------------- #
Vehicle = Literal["dmso", "water", "ethanol", "methanol", "saline"]
MediumKey = Literal["ypd", "sc", "sc_minus_lysine"]


class ConditionSpec(BaseModel):
    """One Table S7 column, its Table S1 row, its Methods vehicle and its medium."""

    model_config = ConfigDict(frozen=True)

    s7_label: str
    s1_name: str
    vehicle: Vehicle
    medium: MediumKey

    @property
    def lookup_label(self) -> str:
        """The Table S1 name without its footnote asterisks: the resolver's input."""
        return re.sub(r"\*+$", "", self.s1_name).strip()


def _c(
    s7: str, s1: str, vehicle: Vehicle = "dmso", medium: MediumKey = "ypd"
) -> ConditionSpec:
    return ConditionSpec(s7_label=s7, s1_name=s1, vehicle=vehicle, medium=medium)


#: Table S7 column order. ``s1_name`` is the Table S1 "Chemical Compound" cell, stripped;
#: ``vehicle`` follows SOLVENT_SENTENCE (DMSO unless named; crude extracts in methanol,
#: EXTRACT_SOLVENT); ``medium`` follows SC_SENTENCE (YPD unless named).
CONDITIONS: tuple[ConditionSpec, ...] = (
    _c("Sulfometuron methyl", "Sulfometuron methyl", medium="sc"),
    _c("MMS", "Methanesulfonic acid methyl ester (MMS)", "water"),
    _c("Clotrimazole", "Clotrimazole"),
    _c("Benomyl", "Benomyl"),
    _c("Plumbagin", "Plumbagin"),
    _c("Hydroxyurea", "Hydroxyurea", "water"),
    _c("Artemisinin", "Artemisinin", "ethanol"),
    _c("Amantadine hydrochloride", "Amantadine hydrochloride"),
    _c("4-Hydroxytamoxifen", "4-Hydroxytamoxifen"),
    _c("Usnic acid", "(+)-Usnic acid"),
    _c("Sodium Azide", "Sodium azide", "water"),
    _c("Nystatin", "Nystatin dihydrate", "water"),
    _c("Neomycin sulfate", "Neomycin B trisulfate salt hydrate"),
    _c("Caffeine", "Caffeine", "water"),
    _c("Menthol", "(1R, 2S, 5R) - ( - ) - Menthol"),
    _c("Verrucarin", "Verrucarin"),
    _c("Valinomycin", "Valinomycin"),
    _c("Trifluoroperazine", "Trifluoroperazine dihydrochloride", "water"),
    _c("Tamoxifen", "Tamoxifen"),
    _c("Raloxifene", "Raloxifene hydrochloride"),
    _c("Pentamidine", "Pentamidine isethionate", "water"),
    _c("Nigericin", "Nigericin"),
    _c("LY-294,002", "LY 294002 hydrochloride"),
    _c("Latrunculin B", "Latrunculin B"),
    _c("Hydroxyethilhidrazine", "2-Hydroxyethilhidrazine", "water"),
    _c("Hydrogen peroxide", "Hydrogen peroxide", "water"),
    _c("Hoechst", "Hoechst 33258"),
    _c("Harmine", "Harmine hydrochloride"),
    _c("Haloperidol", "Haloperidol"),
    _c("Fenpropimorph", "Fenpropimorph"),
    _c("Emetine", "Emetine dihydrochloride hydrate", "water"),
    _c("Dyclonine", "Dyclonine hydrochloride"),
    _c("Doxycycline", "Doxycycline hyclate", "water"),
    _c("Cyclopiazonic acid", "Cyclopiazonic acid"),
    _c("Clomiphene", "Clomiphene citrate", "water"),
    _c("Cisplatin", "cis-diamminedichloroplatinum (II) Cisplatin", "saline"),
    _c("Chlorpromazine", "Chlorpromazine hydrochloride"),
    _c("Cerulenin", "Cerulenin"),
    _c("Calcium ionophore", "Calcium ionophore A23187"),
    _c("Anisomycin", "Anisomycin", "water"),
    _c("Amphotericin", "Amphotericin B"),
    _c("Amiodarone", "Amiodarone hydrochloride"),
    _c("Alamethicin", "Alamethicin*"),
    _c("Actinomycin", "Actinomycin D", "water"),
    _c("Abietic acid", "( - )-Abietic acid"),
    _c("Wortmannin", "Wortmannin", "methanol"),
    _c("Staurosporine", "Staurosporine"),
    _c("Conine", "( +/- ) Coniine"),
    _c("Parthenolide", "Parthenolide"),
    _c("Radicicol", "Radicicol"),
    _c("Mitomycin C", "Mitomycin C", "water"),
    _c("Trichostatin A", "Trichostatin A"),
    _c("FK506", "FK506"),
    _c("Brefeldin A", "Brefeldin A", "ethanol", "sc"),
    _c("U73122", "U-73122"),
    _c("Tunicamycin", "Tunicamycin***"),
    _c("Thialysine", "Thialysine", "water", "sc_minus_lysine"),
    _c("Rapamycin", "Rapamycin"),
    _c("Phenylarsine oxide", "Phenylarsine oxide"),
    _c("Phenantroline", "1-10 Phenanthroline monohydrate", "ethanol"),
    _c("Oligomycin", "Oligomycin **"),
    _c("Nocodazole", "Nocodazole"),
    _c("Hygromycin B", "Hygromycin B", "water"),
    _c("Extract 95-57", "Extract 95-57", "methanol"),
    _c("Extract 6592", "Extract 6592", "methanol"),
    _c("Extract 00-89", "Extract 00-89", "methanol"),
    _c("Extract 00-303C", "Extract 00-303C", "methanol"),
    _c("Extract 00-243", "Extract 00-243", "methanol"),
    _c("Extract 00-192", "Extract 00-192", "methanol"),
    _c("Extract 00-132", "Extract 00-132", "methanol"),
    _c("Emodin", "Emodin"),
    _c("Desipramine", "Desipramine hydrochloride"),
    _c("Cytochalasin A", "Cytochalasin A"),
    _c("CG4-Theopalauamide", "Theopalauamide"),
    _c("Caspofungin", "Caspofungin"),
    _c("Camptothecin", "(S)-(+)-Camptothecin"),
    _c("Basiliskamide", "Basiliskamide"),
    _c("192A4-Stichloroside", "Stichloroside"),
    _c("Papuamide B", "Papuamide B"),
    _c("Agelasine E", "Agelasine E"),
    _c("Fluconazole", "Fluconazole", "water"),
    _c("Geldanamycin", "Geldanamycin"),
)

# --------------------------------------------------------------------------- #
# Unidentified perturbing agents (database constitution principle 2)
# --------------------------------------------------------------------------- #
#: Condition rule for an unidentified perturbing agent: the same rule string Hoepfner
#: 2014 and Hillenmeyer 2008 use.
UNIDENTIFIED_COMPOUND_RULE = (
    "compound_has_no_released_name_or_structure_not_representable"
)

UNIDENTIFIED_COMPOUND_POLICY = (
    "a condition whose Table S1 name is 'Extract <collection code>' and which has no "
    "RELEASED_IDENTITIES entry is a crude natural-product extract from the collection "
    "of R. Andersen known only by its collection code: an undefined mixture with no "
    "released composition and no accession, so what went into the flask cannot be "
    "stated and the experiment cannot be represented (database constitution principle "
    "2, owner rulings 2026-10-15 and 2026-10-16). Every released ratio of such a "
    "condition is HELD OUT, never stored under the bare code with an inchikey gap; the "
    "ledger lists each extract with its dose and record count, and the extract is "
    "admitted the day a released composition or a deposited accession fills "
    "RELEASED_IDENTITIES. A named natural product (agelasine E, stichloroside, "
    "theopalauamide, papuamide B, basiliskamide) is identified material and stays "
    "stored, with its typed inchikey gap where the table gives no key"
)

#: A Table S1 name that is a crude extract's collection code and nothing else.
_EXTRACT_CODE = re.compile(r"^Extract [0-9][0-9A-Z-]*$")

#: The ADMISSION TABLE, keyed by the Table S7 condition label: EMPTY today. An extract
#: condition is stored iff it has an entry here. Only a released composition of the
#: extract or a deposited accession for its material (a later paper, a repository
#: record), mirrored and sha256-pinned like any other artifact, fills it; an identity
#: inferred from profile similarity (00-192 resembles stichloroside, 00-132
#: theopalauamide) is a hypothesis and never goes here. A filled entry changes the
#: stored records, so it rebuilds the store. The ledger's
#: ``unidentified_compounds_held_out`` (every held-out extract with its dose and records)
#: is the request list.
RELEASED_IDENTITIES: dict[str, ReleasedIdentity] = {}


def compound_is_identified(spec: ConditionSpec) -> bool:
    """Whether a condition's agent is identified material (``UNIDENTIFIED_COMPOUND_RULE``).

    Unidentified iff the Table S1 name matches ``_EXTRACT_CODE`` (a crude extract known
    only by its collection code) and ``RELEASED_IDENTITIES`` has no entry for the label.
    Every other condition, a gapped named natural product included, is identified.
    """
    if spec.s7_label in RELEASED_IDENTITIES:
        return True
    return _EXTRACT_CODE.match(spec.lookup_label) is None


def unidentified_conditions() -> list[ConditionSpec]:
    """The conditions held out under ``UNIDENTIFIED_COMPOUND_RULE``, in Table S7 order.

    Refuses unless, with ``RELEASED_IDENTITIES`` empty, exactly the paper's seven crude
    extracts (``N_CRUDE_EXTRACTS``) match, so a relabelled column cannot slip through.
    """
    n_codes = sum(_EXTRACT_CODE.match(c.lookup_label) is not None for c in CONDITIONS)
    if n_codes != N_CRUDE_EXTRACTS.value:
        raise RuntimeError(
            f"{n_codes} Table S1 names are extract collection codes; the paper screened "
            f"{N_CRUDE_EXTRACTS.value} crude extracts"
        )
    return [c for c in CONDITIONS if not compound_is_identified(c)]


def _compound(spec: ConditionSpec) -> Compound:
    """The compound of a condition: the shared table, or an admission entry.

    A ``RELEASED_IDENTITIES`` entry supplies the name and SMILES the label lacks (its
    SMILES yields an RDKit InChIKey), and its released InChIKey fills a key the resolver
    could not, retiring that gap. Without an entry the label resolves as released.
    """
    released = RELEASED_IDENTITIES.get(spec.s7_label)
    if released is None:
        return resolved_compound(spec.lookup_label)
    compound = resolved_compound(
        released.name if released.name is not None else spec.lookup_label,
        smiles=released.smiles,
        derive_from_smiles=released.smiles is not None,
    )
    if released.inchikey is not None and compound.inchikey is None:
        compound = compound.model_copy(
            update={
                "inchikey": released.inchikey,
                "provenance_gaps": [
                    gap for gap in compound.provenance_gaps if gap.field != "inchikey"
                ],
            }
        )
    return compound


# --------------------------------------------------------------------------- #
# Doses (Table S1 "Inhibitory Concentration")
# --------------------------------------------------------------------------- #
_DOSE_RE = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mM|μM|nM|pM|μg/ml|%\s*\(v/v\))\s*$"
)
_DOSE_UNITS: dict[str, ConcentrationUnit] = {
    "mM": ConcentrationUnit.millimolar,
    "μM": ConcentrationUnit.micromolar,
    "nM": ConcentrationUnit.nanomolar,
    "μg/ml": ConcentrationUnit.ug_per_ml,
}


class S1Row(BaseModel):
    """One compound row of Table S1, verbatim."""

    model_config = ConfigDict(frozen=True)

    cas: str | None
    name: str
    dose_value: str | float
    dose_format: str


def parse_dose(row: S1Row) -> Concentration:
    """Table S1's screening concentration as a typed, exactly converted ``Concentration``.

    Strings carry their unit (``0.33 mM``, ``61 μM``, ``5% (v/v)``). A picomolar dose
    (rapamycin 5.5 pM, tunicamycin 71 pM) is stated in nM by an exact decimal division,
    since ``ConcentrationUnit`` has no pM member. The one numeric cell is MMS, an Excel
    percent-formatted fraction (``4e-05`` under ``0.000%``, which Table 1 prints as
    "0.004%"); its basis (v/v or w/v) is not stated, so it is the basis-free
    ``percent``. Anything else refuses.
    """
    if isinstance(row.dose_value, float):
        if not row.dose_format.endswith("%"):
            raise ValueError(
                f"{row.name}: numeric dose {row.dose_value!r} is not percent-formatted "
                f"({row.dose_format!r})"
            )
        percent = Decimal(repr(row.dose_value)) * 100
        return Concentration(value=float(percent), unit=ConcentrationUnit.percent)
    match = _DOSE_RE.match(row.dose_value)
    if match is None:
        raise ValueError(f"{row.name}: unparseable dose {row.dose_value!r}")
    value = Decimal(match["value"])
    unit = match["unit"]
    if unit == "pM":
        return Concentration(
            value=float(value / 1000), unit=ConcentrationUnit.nanomolar
        )
    if unit.startswith("%"):
        return Concentration(value=float(value), unit=ConcentrationUnit.percent_v_v)
    return Concentration(value=float(value), unit=_DOSE_UNITS[unit])


# --------------------------------------------------------------------------- #
# Media, vehicles, background
# --------------------------------------------------------------------------- #
PARSONS_YPD = restated(YPD_LIQUID, YPD_CULTURE)
PARSONS_SC = restated(SC, SC_MEDIUM)
PARSONS_SC_MINUS_LYSINE = dropout(
    SC,
    "L-lysine",
    name="SC - L-lysine (Parsons 2006 'synthetic complete media lacking lysine')",
    provenance=[SC_LYSINE_DROPOUT],
)
MEDIA: dict[str, Media] = {
    "ypd": PARSONS_YPD,
    "sc": PARSONS_SC,
    "sc_minus_lysine": PARSONS_SC_MINUS_LYSINE,
}


def vehicle(key: Vehicle) -> Solvent:
    """The Methods vehicle at the 1% of the solvent control.

    Water and the 0.9% saline are plain vehicles (``compound=None``): the pinned
    compound-identity table has no water row, and the saline is a solution, not a
    single species.
    """
    percent = SOLVENT_PERCENT.value
    if key == "dmso":
        return Solvent(
            name="DMSO",
            percent=percent,
            compound=resolved_compound("dimethyl sulfoxide"),
        )
    if key == "ethanol":
        return Solvent(
            name="ethanol", percent=percent, compound=resolved_compound("ethanol")
        )
    if key == "methanol":
        return Solvent(
            name="methanol", percent=percent, compound=resolved_compound("methanol")
        )
    if key == "water":
        return Solvent(name="water", percent=percent, compound=None)
    return Solvent(name="0.9% sodium chloride solution", percent=percent, compound=None)


def _background_allele(allele: BackgroundAllele) -> BackgroundAllele:
    """The allele with a pending-review gap on its (unpublished) deleted interval."""
    return BackgroundAllele.model_validate(
        {
            **allele.model_dump(),
            "provenance_gaps": [
                pending_source_review(
                    "deleted_span",
                    BRACHMANN_1998,
                    "the designation is quoted from the SI strain table; the edit kind "
                    "and the deleted interval are the literature-standard reading of "
                    "Brachmann 1998, not mirrored",
                ).model_dump()
            ],
        }
    )


BACKGROUND_STRAIN = "BY4741"


def parsons_background() -> EukaryoticStrainBackground:
    """BY4741 with its SI-quoted alleles; that the pool IS BY4741 is a typed gap."""
    genotype = STANDARD_BY_GENOTYPES[BACKGROUND_STRAIN]
    alleles = [
        _background_allele(
            standard_allele(allele_name, zygosity, provenance=[BY4741_GENOTYPE])
        )
        for allele_name, zygosity in genotype.alleles.items()
    ]
    return EukaryoticStrainBackground(
        reference_strain="S288C",
        assembly_set=SGD_S288C_R64,
        name=BACKGROUND_STRAIN,
        parents=list(genotype.parents),
        construction=None,
        mating_type=MatingType.a,
        ploidy="haploid",
        alleles=alleles,
        provenance=None,
        provenance_gaps=[
            ProvenanceGap(
                field="provenance",
                reason=ProvenanceGapReason.deferred_pending_source_review,
                looked_in=Provenance(
                    source_uri=SI_MD,
                    citation_key=CITATION_KEY,
                    sha256=SI_MD_SHA256,
                    page="Chemical-Genetic Profiling: 'MATa haploid deletion mutants "
                    "(~4800) were obtained from Research Genetics'; Yeast Strains Used in "
                    "This Study: the BY4741 line",
                ),
                resolve_with=GIAEVER_2002,
                note="the SI sources MATa and haploid for the pool and states BY4741's "
                "genotype, but never that the Research Genetics MATa collection is "
                "BY4741; the name is asserted pending Giaever 2002",
            )
        ],
    )


BY4741_GENOME = StrainReferenceGenome(
    species="Saccharomyces cerevisiae",
    strain=BACKGROUND_STRAIN,
    ploidy="haploid",
    background=parsons_background(),
    assembly_set=SGD_S288C_R64,
    assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/parsonsExploringModeofActionBioactive2006``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """The literature mirror of this citation key."""
    return Path(data_root or _data_root()) / "torchcell-library" / CITATION_KEY


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_retrieval(
    table: RawTable, retrieved_at: str = RETRIEVED_AT
) -> RetrievalRecord:
    """The recorded retrieval of one Elsevier supplementary object.

    The literature mirror holds the bytes under the digest the publisher served, with
    this retriever and its params recorded, so the raw mirror is written from those
    bytes; ``run_retriever`` on this record re-fetches them.
    """
    return RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=table.source_url,
        retriever="torchcell.literature.retrieve.elsevier_mmc",
        params={"pii": PII, "filename": table.filename},
        sha256=table.sha256,
        retrieved_at=retrieved_at,
    )


def deposit_raw_mirror(
    *, retrieved_at: str = RETRIEVED_AT, data_root: str | None = None
) -> Path:
    """Write the raw mirror (Tables S1, S2, S4, S7) from the literature mirror's bytes.

    Every source is verified BEFORE anything is written, so a refusal leaves no partial
    deposit. Idempotent by sha256: a mirror file with the pinned digest is left alone and
    one with other bytes raises rather than being overwritten.
    """
    library = library_dir(data_root)
    for table in RAW_TABLES:
        src = library / table.library_rel
        if not src.exists():
            raise RuntimeError(f"library mirror is missing {src}")
        got = _sha256(src)
        if got != table.sha256:
            raise RuntimeError(f"{src} sha256 {got} is not the pinned {table.sha256}")
    root = raw_mirror_dir(data_root)
    records: list[ArtifactRecord] = []
    for table in RAW_TABLES:
        dest = root / table.raw_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != table.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(library / table.library_rel, dest)
        size = dest.stat().st_size
        if size != table.bytes:
            raise RuntimeError(f"{dest} is {size} bytes, not the pinned {table.bytes}")
        records.append(
            ArtifactRecord(
                path=table.raw_rel,
                role=ROLE_RAW_DATA,
                bytes=size,
                sha256=table.sha256,
                source=table.source_url,
                original_filename=table.filename,
                retrieval=table_retrieval(table, retrieved_at),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=records,
        si_data_sources=[t.source_url for t in RAW_TABLES],
        si_expected=list(NOT_CONSUMED),
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    if not path.exists():
        raise RuntimeError(
            f"raw mirror manifest missing: {path}; run deposit_raw_mirror() first"
        )
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Reading the release (legacy BIFF .xls: title row 0, header row 2, body from row 3)
# --------------------------------------------------------------------------- #
HEADER_ROW = 2
S1_HEADER = (
    "CAS Number",
    "Chemical Compound",
    "FDA Approved",
    "Inhibitory Concentration",
    "Comments",
    "Origin",
    "Site of Collection",
    "Reference",
)
#: The cell the release writes for a removed ratio (two spaces).
BLANK_CELL = "  "


def read_sheet(path: str | Path) -> pd.DataFrame:
    """The workbook's only sheet as raw cells (no header inference)."""
    return pd.read_excel(path, sheet_name=0, header=None, engine="xlrd")


def _check_title(frame: pd.DataFrame, table: RawTable) -> None:
    title = frame.iloc[0, 0]
    if title != table.title:
        raise RuntimeError(
            f"Table {table.table} title is {title!r}, not {table.title!r}"
        )


def _labels(frame: pd.DataFrame, first: int) -> list[str]:
    """Header labels from column ``first`` on, stripped (the release pads them)."""
    return [str(v).strip() for v in frame.iloc[HEADER_ROW, first:].tolist()]


class RatioMatrix(BaseModel):
    """Table S7: ORF row order, condition labels, and each cell (None = released blank)."""

    orfs: list[str]
    labels: list[str]
    values: list[list[float | None]]

    @property
    def n_numeric(self) -> int:
        """Released ratios (non-blank cells)."""
        return sum(v is not None for row in self.values for v in row)

    @property
    def n_blank(self) -> int:
        """Released two-space blanks."""
        return sum(v is None for row in self.values for v in row)


def parse_ratio_matrix(frame: pd.DataFrame) -> RatioMatrix:
    """Table S7 cells: a float, or the release's two-space blank. Anything else refuses."""
    _check_title(frame, TABLE_S7)
    header = _labels(frame, 0)
    if header[0] != "ORF":
        raise RuntimeError(f"Table S7 first header cell is {header[0]!r}, not 'ORF'")
    labels = header[1:]
    expected = [c.s7_label for c in CONDITIONS]
    if labels != expected:
        raise RuntimeError(
            "Table S7 condition labels differ from CONDITIONS: "
            f"{[(a, b) for a, b in zip(labels, expected, strict=True) if a != b]}"
        )
    orfs: list[str] = []
    values: list[list[float | None]] = []
    for row in frame.iloc[HEADER_ROW + 1 :].itertuples(index=False):
        cells = list(row)
        orfs.append(str(cells[0]).strip())
        parsed: list[float | None] = []
        for label, cell in zip(labels, cells[1:], strict=True):
            if isinstance(cell, str):
                if cell != BLANK_CELL:
                    raise RuntimeError(
                        f"Table S7 {orfs[-1]} / {label}: unexpected cell {cell!r}"
                    )
                parsed.append(None)
                continue
            number = float(cell)
            if not math.isfinite(number):
                raise RuntimeError(
                    f"Table S7 {orfs[-1]} / {label}: non-finite {cell!r}"
                )
            parsed.append(number)
        values.append(parsed)
    if len(set(orfs)) != len(orfs):
        raise RuntimeError("Table S7 lists an ORF twice")
    return RatioMatrix(orfs=orfs, labels=labels, values=values)


class PValueMatrix(BaseModel):
    """Table S4: -log10 p per (ORF, condition label); a missing key is a blank cell."""

    orfs: list[str]
    labels: list[str]
    neg_log10: dict[str, list[float | None]]


def parse_p_value_matrix(frame: pd.DataFrame) -> PValueMatrix:
    """Table S4 cells: a finite non-negative -log10 p, or blank (NaN)."""
    _check_title(frame, TABLE_S4)
    header = _labels(frame, 0)
    if header[:2] != ["ORF", "Standard Name"]:
        raise RuntimeError(f"Table S4 header starts {header[:2]!r}")
    labels = header[2:]
    if labels != [c.s7_label for c in CONDITIONS]:
        raise RuntimeError("Table S4 condition labels differ from Table S7's")
    orfs: list[str] = []
    neg_log10: dict[str, list[float | None]] = {}
    for row in frame.iloc[HEADER_ROW + 1 :].itertuples(index=False):
        cells = list(row)
        orf = str(cells[0]).strip()
        parsed: list[float | None] = []
        for label, cell in zip(labels, cells[2:], strict=True):
            if pd.isna(cell):
                parsed.append(None)
                continue
            number = float(cell)
            if not (math.isfinite(number) and 0.0 <= number <= P_VALUE_CAP.value):
                raise RuntimeError(f"Table S4 {orf} / {label}: -log10 p {cell!r}")
            parsed.append(number)
        orfs.append(orf)
        neg_log10[orf] = parsed
    if len(neg_log10) != len(orfs):
        raise RuntimeError("Table S4 lists an ORF twice")
    return PValueMatrix(orfs=orfs, labels=labels, neg_log10=neg_log10)


def parse_orf_list(frame: pd.DataFrame, table: RawTable) -> list[str]:
    """The ORF column of a strain-list table (Table S2), stopping at the first blank."""
    _check_title(frame, table)
    if _labels(frame, 0)[0] != "ORF":
        raise RuntimeError(f"Table {table.table} first header cell is not 'ORF'")
    orfs: list[str] = []
    for cell in frame.iloc[HEADER_ROW + 1 :, 0].tolist():
        if pd.isna(cell):
            break
        orfs.append(str(cell).strip())
    return orfs


def read_s1_rows(path: str | Path) -> list[S1Row]:
    """Table S1's compound rows with each dose cell's Excel number format.

    Read through ``xlrd`` with ``formatting_info`` because one dose (MMS) is a
    percent-FORMATTED number, which only the cell's format string identifies. Rows stop
    at the first row whose name cell is blank (the release's footnotes follow it).
    """
    book = xlrd.open_workbook(str(path), formatting_info=True)
    sheet = book.sheet_by_index(0)
    if sheet.cell_value(0, 0) != TABLE_S1.title:
        raise RuntimeError(f"Table S1 title is {sheet.cell_value(0, 0)!r}")
    header = tuple(str(sheet.cell_value(HEADER_ROW, c)).strip() for c in range(8))
    if header != S1_HEADER:
        raise RuntimeError(f"Table S1 header is {header!r}")
    rows: list[S1Row] = []
    for r in range(HEADER_ROW + 1, sheet.nrows):
        name = sheet.cell_value(r, 1)
        if not str(name).strip():
            break
        cas = str(sheet.cell_value(r, 0)).strip() or None
        dose = sheet.cell(r, 3)
        fmt = book.format_map[book.xf_list[dose.xf_index].format_key].format_str
        rows.append(
            S1Row(
                cas=cas, name=str(name).strip(), dose_value=dose.value, dose_format=fmt
            )
        )
    return rows


def s1_by_name(rows: list[S1Row]) -> dict[str, S1Row]:
    """Table S1 rows keyed by name; every CONDITIONS row must appear exactly once."""
    by_name = {row.name: row for row in rows}
    if len(by_name) != len(rows):
        raise RuntimeError("Table S1 lists a compound name twice")
    wanted = [c.s1_name for c in CONDITIONS]
    missing = sorted(set(wanted) - set(by_name))
    unused = sorted(set(by_name) - set(wanted))
    if missing or unused or len(set(wanted)) != len(wanted):
        raise RuntimeError(
            f"Table S1 and CONDITIONS disagree: missing {missing}, unused {unused}"
        )
    return by_name


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the items it removed them for."""

    rule: str
    scope: Literal["cell", "strain", "condition"]
    description: str
    n_records: int
    items: list[str] = []


class ReleaseAccounting(BaseModel):
    """What the release carries outside the ratio matrix (no record can be built)."""

    table_s4_strains: int
    table_s4_p_values: int
    multidrug_sensitive_strains_without_ratios: list[str]
    multidrug_sensitive_p_values: int
    p_value_cells_without_a_ratio: int
    ratio_cells_without_a_p_value: int


class GappedCompoundCondition(BaseModel):
    """A condition stored with a typed compound-identity gap, and its record count."""

    condition: str = Field(description="The Table S7 condition label, verbatim.")
    compound_name: str = Field(description="The stored Compound.name.")
    gap_field: str = Field(description="The gapped Compound field (inchikey).")
    gap_reason: ProvenanceGapReason
    n_records: int


class UnidentifiedCompoundHeldOut(BaseModel):
    """One condition held out under ``UNIDENTIFIED_COMPOUND_RULE``, with what it carried."""

    model_config = ConfigDict(extra="forbid")

    condition: str = Field(description="The Table S7 condition label, verbatim.")
    s1_name: str = Field(description="The Table S1 'Chemical Compound' cell, verbatim.")
    dose: Concentration = Field(description="The Table S1 screening concentration.")
    vehicle: Vehicle
    medium: MediumKey
    n_records: int = Field(description="Released ratios of this column not written.")


class DropLog(BaseModel):
    """Every retention rule applied to a build, in order, plus the release accounting.

    ``compound_identity_gaps`` are not drops: those conditions are stored, each compound
    carrying its typed gap. ``unidentified_compounds_held_out`` is the per-condition
    breakdown of the ``UNIDENTIFIED_COMPOUND_RULE`` rule and the admission request list.
    """

    dataset: str
    source_records: int
    released_ratios: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]
    compound_identity_gaps: list[GappedCompoundCondition]
    unidentified_compound_rule: str
    unidentified_compound_policy: str
    unidentified_compound_sources: list[SourcedValue]
    unidentified_compounds_held_out: list[UnidentifiedCompoundHeldOut]
    release: ReleaseAccounting


@register_dataset
class EnvChemgenParsons2006Dataset(ExperimentDataset):
    """Parsons 2006 chemical-genetic compendium: env x geno -> log2(control/treated)."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_parsons2006",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (no genome: every strain is stored as released)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return StrainEnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return StrainEnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Tables S1, S2, S4 and S7 under their publisher filenames."""
        return [table.filename for table in RAW_TABLES]

    def download(self) -> None:
        """Link the four mirror files into ``raw/`` after verifying each against its pin.

        The pins are canonical; the manifest is the retrieval record and must carry the
        same digests, else ``ManifestPinMismatchError`` names both. The Elsevier URLs are
        retrieval metadata, never a build dependency.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for table in RAW_TABLES:
            check_manifest_pin(
                table.raw_rel, manifest_sha256(manifest, table.raw_rel), table.sha256
            )
            src = raw_mirror_dir(data_root) / table.raw_rel
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, table.filename), table.sha256)
        log.info(
            "Parsons 2006 raw tables linked into %s (sha256 verified)", self.raw_dir
        )

    # ---- readers (the file boundary; tests stub these) ------------------------- #
    def _read_ratio_matrix(self) -> RatioMatrix:
        return parse_ratio_matrix(read_sheet(osp.join(self.raw_dir, TABLE_S7.filename)))

    def _read_p_value_matrix(self) -> PValueMatrix:
        return parse_p_value_matrix(
            read_sheet(osp.join(self.raw_dir, TABLE_S4.filename))
        )

    def _read_multidrug_sensitive(self) -> list[str]:
        return parse_orf_list(
            read_sheet(osp.join(self.raw_dir, TABLE_S2.filename)), TABLE_S2
        )

    def _read_s1(self) -> list[S1Row]:
        return read_s1_rows(osp.join(self.raw_dir, TABLE_S1.filename))

    # ---- builders ---------------------------------------------------------------- #
    @staticmethod
    def _culture(medium: Media, perturbations: list[Any]) -> CultureEnvironment:
        """The 38 h shaken 30 C pooled culture from the frozen pool, read at T2."""
        return CultureEnvironment(
            media=medium,
            temperature=Temperature(value=TEMPERATURE_C.value),
            perturbations=perturbations,
            aerobicity=AEROBICITY.value,
            duration_hours=DURATION_HOURS.value,
            culture_format=CultureFormat(
                inoculum_od600=INOCULUM_OD600.value,
                endpoint=EndpointRule.fixed_duration,
                provenance=[INOCULUM_OD600, DURATION_HOURS],
                provenance_gaps=[
                    ProvenanceGap(
                        field="working_volume_ul",
                        reason=ProvenanceGapReason.not_reported_by_primary,
                        note="'in 5 or 10 mls YPD'; which volume a condition used is "
                        "not stated",
                    ),
                    ProvenanceGap(
                        field="shaking_rpm",
                        reason=ProvenanceGapReason.not_reported_by_primary,
                        note="'incubated at 30 C shaking'; no speed is stated",
                    ),
                    ProvenanceGap(
                        field="vessel",
                        reason=ProvenanceGapReason.not_reported_by_primary,
                        note="no culture vessel is named",
                    ),
                ],
            ),
            pre_culture=PreCulture(
                source=PRE_CULTURE_SOURCE.value, provenance=[PRE_CULTURE_SOURCE]
            ),
            provenance_gaps=[
                ProvenanceGap(
                    field="duration_generations",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note="the exposure is stated in hours (14 h + 24 h); no doubling "
                    "count is reported, and the control pool undergoes more doublings "
                    "than the treated one by design",
                )
            ],
        )

    def _environment(
        self, spec: ConditionSpec, compound: Compound, dose: Concentration
    ) -> CultureEnvironment:
        """The condition's culture: its medium plus the compound in its vehicle."""
        return self._culture(
            MEDIA[spec.medium],
            [
                SmallMoleculePerturbation(
                    compound=compound, concentration=dose, solvent=vehicle(spec.vehicle)
                )
            ],
        )

    def _reference(self, medium: Media) -> StrainEnvironmentResponseExperimentReference:
        """The 1% solvent-control pool in one medium: log2 ratio 0 by construction."""
        return StrainEnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=BY4741_GENOME,
            environment_reference=self._culture(medium, []),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.log2_ratio,
                assay_type=AssayType.pooled_competitive_growth_barcode,
                environment_response=0.0,
                units=REFERENCE_UNITS,
            ),
        )

    @staticmethod
    def _phenotype(
        ratio: float, neg_log10_p: float | None
    ) -> EnvironmentResponsePhenotype:
        """The released ratio, verbatim, with Table S4's p-value of the same cell."""
        return EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=ratio,
            environment_response_p_value=(
                None if neg_log10_p is None else 10.0 ** (-neg_log10_p)
            ),
            n_samples=CHIP_REPLICATES.value,
            sample_unit=SampleUnit.technical_replicate,
            units=MEASUREMENT_UNITS,
            provenance_gaps=[
                ProvenanceGap(
                    field=field,
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note="no per-cell dispersion is released: the two fluor-reversed "
                    "chips and the up and down tags are averaged into the one value",
                )
                for field in (
                    "environment_response_uncertainty",
                    "environment_response_se",
                )
            ],
        )

    @staticmethod
    def _genotype(orf: str) -> Genotype:
        """One barcoded kanMX4 deletion from the Research Genetics MATa collection,
        named by Table S7's ORF cell as released (no genome is consulted).
        """
        return Genotype(
            perturbations=[
                BarcodedKanMxDeletionPerturbation(
                    systematic_gene_name=orf,
                    perturbed_gene_name=orf,
                    collection=POOL_SOURCE.value,
                    cassette=KANMX4_CASSETTE.value,
                    provenance_gaps=[
                        pending_source_review(
                            "barcode",
                            GIAEVER_2002,
                            "the release publishes no UPTAG/DNTAG sequence; the YKO "
                            "barcode table gives them by ORF",
                        ),
                        pending_source_review(
                            "downtag_barcode",
                            GIAEVER_2002,
                            "the release publishes no DNTAG sequence",
                        ),
                    ],
                )
            ]
        )

    # ---- build ------------------------------------------------------------------- #
    @post_process
    def process(self) -> None:
        """Join Table S7's ratios to Table S4's p-values and Table S1's doses; write LMDB."""
        verify_raw_files(self.raw_dir, {t.filename: t.sha256 for t in RAW_TABLES})
        ratios = self._read_ratio_matrix()
        p_values = self._read_p_value_matrix()
        multidrug = self._read_multidrug_sensitive()
        s1 = s1_by_name(self._read_s1())

        missing_p = sorted(set(ratios.orfs) - set(p_values.neg_log10))
        if missing_p:
            raise RuntimeError(f"Table S7 ORFs absent from Table S4: {missing_p[:10]}")
        p_only = sorted(set(p_values.neg_log10) - set(ratios.orfs))
        if p_only != sorted(multidrug):
            raise RuntimeError(
                "the Table S4 strains Table S7 lacks are not Table S2's multidrug-"
                f"sensitive set ({len(p_only)} vs {len(multidrug)})"
            )

        held_out = {spec.s7_label: spec for spec in unidentified_conditions()}
        compounds: dict[str, Compound] = {}
        doses: dict[str, Concentration] = {}
        for spec in CONDITIONS:
            row = s1[spec.s1_name]
            doses[spec.s7_label] = parse_dose(row)
            if spec.s7_label not in held_out:
                compounds[spec.s7_label] = _compound(spec)
        # An identified compound with no structure identifier is stored with the typed
        # inchikey gap resolved_compound attaches (#726, #1045 row 19), never dropped.
        gapped = {
            label: compound
            for label, compound in compounds.items()
            if compound.inchikey is None
            and compound.pubchem_cid is None
            and compound.chebi_id is None
        }
        for label, compound in gapped.items():
            if [gap.field for gap in compound.provenance_gaps] != ["inchikey"]:
                raise RuntimeError(
                    f"{label}: a compound with no identifier must carry exactly the "
                    "typed inchikey gap"
                )
        n_gapped: dict[str, int] = dict.fromkeys(gapped, 0)
        names = [c.name for c in compounds.values()]
        if len(set(names)) != len(names):
            raise RuntimeError(
                "two conditions resolve to one canonical compound name, which would "
                "merge two conditions into one record key"
            )

        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )
        references = {key: self._reference(medium) for key, medium in MEDIA.items()}
        environments = {
            spec.s7_label: self._environment(
                spec, compounds[spec.s7_label], doses[spec.s7_label]
            )
            for spec in CONDITIONS
            if spec.s7_label not in held_out
        }

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_blank = 0
        n_held: dict[str, int] = dict.fromkeys(held_out, 0)
        n_without_p = 0
        n_p_without_ratio = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for source_orf, row_values in tqdm(
                zip(ratios.orfs, ratios.values, strict=True),
                total=len(ratios.orfs),
                desc="parsons2006",
            ):
                p_row = p_values.neg_log10[source_orf]
                genotype = self._genotype(source_orf)
                for spec, ratio, neg_log10_p in zip(
                    CONDITIONS, row_values, p_row, strict=True
                ):
                    if ratio is None:
                        n_blank += 1
                        n_p_without_ratio += neg_log10_p is not None
                        continue
                    n_without_p += neg_log10_p is None
                    if spec.s7_label in held_out:
                        n_held[spec.s7_label] += 1  # UNIDENTIFIED_COMPOUND_RULE
                        continue
                    if spec.s7_label in n_gapped:
                        n_gapped[spec.s7_label] += 1
                    experiment = StrainEnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=genotype,
                        environment=environments[spec.s7_label],
                        phenotype=self._phenotype(ratio, neg_log10_p),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(
                            experiment, references[spec.medium], publication, itxn
                        ),
                    )
                    idx += 1
        env.close()
        interned_env.close()

        source_records = len(ratios.orfs) * len(CONDITIONS)
        multidrug_p = sum(
            v is not None for orf in p_only for v in p_values.neg_log10[orf]
        )
        drop_log = DropLog(
            dataset=self.name,
            source_records=source_records,
            released_ratios=ratios.n_numeric,
            kept_records=idx,
            dropped_records=source_records - idx,
            rules=[
                DropRule(
                    rule="ratio_cell_blank_in_release",
                    scope="cell",
                    description=(
                        "the release writes a two-space blank where both tags of the "
                        "strain were removed as noise-dominated in that screen "
                        "(NOISE_REMOVAL); there is no ratio to store"
                    ),
                    n_records=n_blank,
                ),
                DropRule(
                    rule=UNIDENTIFIED_COMPOUND_RULE,
                    scope="condition",
                    description=(
                        "not representable: the condition's agent is a crude extract "
                        "known only by its collection code, with no released "
                        "composition or accession (UNIDENTIFIED_COMPOUND_POLICY)"
                    ),
                    n_records=sum(n_held.values()),
                    items=sorted(held_out),
                ),
            ],
            compound_identity_gaps=[
                GappedCompoundCondition(
                    condition=label,
                    compound_name=gapped[label].name,
                    gap_field=gapped[label].provenance_gaps[0].field,
                    gap_reason=gapped[label].provenance_gaps[0].reason,
                    n_records=n_gapped[label],
                )
                for label in sorted(gapped)
            ],
            unidentified_compound_rule=UNIDENTIFIED_COMPOUND_RULE,
            unidentified_compound_policy=UNIDENTIFIED_COMPOUND_POLICY,
            unidentified_compound_sources=[
                N_CRUDE_EXTRACTS,
                EXTRACT_COLLECTION,
                EXTRACT_TABLE_COMMENT,
            ],
            unidentified_compounds_held_out=[
                UnidentifiedCompoundHeldOut(
                    condition=label,
                    s1_name=spec.s1_name,
                    dose=doses[label],
                    vehicle=spec.vehicle,
                    medium=spec.medium,
                    n_records=n_held[label],
                )
                for label, spec in sorted(held_out.items())
            ],
            release=ReleaseAccounting(
                table_s4_strains=len(p_values.orfs),
                table_s4_p_values=sum(
                    v is not None for row in p_values.neg_log10.values() for v in row
                ),
                multidrug_sensitive_strains_without_ratios=p_only,
                multidrug_sensitive_p_values=multidrug_p,
                p_value_cells_without_a_ratio=n_p_without_ratio,
                ratio_cells_without_a_p_value=n_without_p,
            ),
        )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        accounted = sum(rule.n_records for rule in drop_log.rules)
        if accounted != drop_log.dropped_records:
            raise RuntimeError(
                f"drop accounting mismatch: rules total {accounted}, "
                f"{drop_log.dropped_records} records missing from the build"
            )
        log.info(
            "Parsons2006: wrote %d records over %d strains x %d conditions (%d released "
            "blank cells have no ratio; %d ratios of %d unidentified extracts held out; "
            "%d records on %d compounds stored with a typed identity gap)",
            idx,
            len(ratios.orfs),
            len(compounds),
            n_blank,
            sum(n_held.values()),
            len(held_out),
            sum(n_gapped.values()),
            len(gapped),
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/torchcell/env_chemgen_parsons2006")
    dataset = EnvChemgenParsons2006Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
