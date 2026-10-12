# torchcell/datasets/ecoli/rousset2018
# [[torchcell.datasets.ecoli.rousset2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/rousset2018
# Test file: tests/torchcell/datasets/ecoli/test_rousset2018.py
"""Rousset 2018: genome-wide CRISPR-dCas9 guide fitness under three phages.

Rousset et al. 2018 (PLoS Genetics 14:e1007749, doi:10.1371/journal.pgen.1007749;
PMID 30403660; PMC6242692) screened a pooled library of about 92,000 random-position
sgRNAs in two *E. coli* K-12 MG1655 derivatives carrying an aTc-inducible dCas9, and
released per-guide log2 fold changes for five independent screens: growth in rich medium
over 17 generations (S1 Table, the filtered library of about 59,000 guides), challenge by
phage lambda, T4 and 186cIts at MOI 1 (S4 Table, three columns on a library of about
17,200 guides), and recovery of the guide-carrying cosmid after a lambda transduction
assay (S6 Table, the same 17,200 guides).

THIS DATASET SERVES ALL FIVE RELEASED SCREENS (#905). The growth screen (S1 Table,
strain LC-E75) is stored as released, beside the four phage-derived screens, with its
own ``padj``: under the owner's rule of 2026-10-12 nothing is cross-referenced against
another source inside a loader. Before #905 the S1 rows were dropped as "served by" Cui
2018; that comparison is gone, and what it measured is below as a finding, not a rule.

RECORD = one (sgRNA x screen) ``BacterialEnvironmentResponseExperiment``:

- GENOTYPE: one ``BacterialReleasedCrisprPerturbation`` (#1077) keyed by the target
  gene's SYMBOL, exactly as the release names it, with the 20-nt spacer on the shared
  ``crispr`` construct (``effector="dCas9"``) and as ``strain_label``, and
  ``expression_direction="decreased"``. The loader reads no annotation (the owner's
  rule of 2026-10-12: the data is the data, and the genome is a separate layer joined
  at modeling time). Before that rule every symbol went through
  ``reconcile_locus_tags`` to a b-number and 444 records over 111 symbols the pinned
  GCA_000005845.2 retires, merges or cannot place were dropped; they are records now,
  and the L4 genome-reference rows report them (#1069).
- ENVIRONMENT: the screen's own culture, which is one medium for all four screens: LB
  plus 1 microM aTc, 0.2% maltose and 5 mM CaCl2 at 37 C for 2 h, each record carrying
  exactly one ``PhagePerturbation`` at ``multiplicity_of_infection=1``. The aTc that
  induces dCas9 is a COMPONENT of that medium, not an ``Environment.perturbation``,
  because the paper's own recipes put it there, twice, beside the maltose and the CaCl2;
  the full sourcing, including the one sentence that reads as a treatment instead, is on
  ``_atc``. That leaves the phage as the only environment perturbation any record
  carries, so the conf enables ``phage perturbation`` and not
  ``environment perturbation``, which would emit nothing. Since issue #756 the two lanes
  partition the environment's perturbations, so a conf may enable both and the
  placement of aTc is not an adapter decision in either direction.
- PHENOTYPE: ``EnvironmentResponsePhenotype`` with ``measurement_type=log2_ratio``. The
  reference carries 0.0, which is what no change in guide abundance is.

WHY ``EnvironmentResponsePhenotype`` AND NOT ``FitnessPhenotype``. The released value is
a signed DESeq2 ``log2FoldChange`` of guide abundance ("The log2FoldChange ... value
represents the enrichment or depletion of each sgRNA"), normalized on a non-targeting
control guide and paired against each sample's own initial condition. It is routinely
negative. Measured over the 68,436 phage-derived records stored before #1069: 89.15% of the transduction screen,
83.93% of T4, 44.83% of 186cIts and 27.70% of lambda. ``FitnessPhenotype.validate_fitness``
clamps every non-positive value to 0.0, which would erase that signal, and no
``MeasurementType`` member other than ``log2_ratio`` describes this number.

WHY ``PhagePerturbation``. Every stored screen doses a virion: "followed by infection
with phage lambda, T4 or 186cIts at a multiplicity of infection (MOI) of 1". That is the
typed leaf for a bacteriophage on the environment axis, and the dose is an MOI rather
than a ``Concentration``.

THE GROWTH SCREEN'S CULTURE AND HOST come from the deferral the paper states ("obtained
from our previous study [26]"): Cui 2018's pooled competition, LB with 1 nM aTc at 37 C
(``cui2018.screen_environment``), and Cui's own Supplementary Table 7 record of LC-E75;
the 17 generations and the triplicate are Rousset's own sentence
(``GROWTH_SCREEN_REPLICATES``). The 1 nM aTc is an environment perturbation there, which
is why the adapter conf enables ``environment perturbation`` beside ``phage perturbation``.

THE HOST STRAIN (phage screens). FR-E01, an MG1655 derivative written against the MG1655 assembly, so it
is a ``BacterialStrainBackground`` on the reference rather than a namespace: it carries
the same optimized dcas9 cassette as LC-E75 moved to the HK022 attB site, built so that
phage 186 could be studied without interference.

THE GROWTH SCREEN AND CUI 2018 (a finding, not a rule). [26] is Cui et al. 2018 (Nat
Commun 9:1912), served by ``CrispriKnockdownCui2018Dataset``. Measured once on the two
pinned tables (issue #878; the loader no longer reads Cui's table): 54,326 spacers in
common, median absolute difference 0.0000 and maximum 0.0066 against Cui's ``fit75``,
and only 426 bit-identical, so S1 Table is Rousset's own DESeq2 run over the same reads.
The two releases are two records of one screen under two publications and two screen ids;
which to use is a modeling-time decision. Hypothesis (untested): the S1 spacers Cui's
table lacks fall under Cui's 20-read floor.

S1 TABLE'S ``padj`` IS STORED (#905). It is the DESeq2 adjusted p-value of each guide's
log2FC, numeric on all 59,246 rows (0 to 0.999997, measured), on
``environment_response_p_value_adjusted`` with ``p_value_adjustment_method =
"benjamini_hochberg"``. The paper names its tool, not the correction ("computed the
log2-transformed fold change (log2FC) from DESeq2 R package [32]"); ref. 32, Love,
Huber and Anders 2014, states it ("adjusted for multiple testing using the procedure of
Benjamini and Hochberg"), and its PMC text is deposited in this raw mirror as the quote
anchor (``PADJ_CORRECTION``). No unadjusted p-value is released, so
``environment_response_p_value`` stays None; the L2 row
``released_test_where_released`` checks that pattern. S4 and S6 Tables release no test.

RECORDS DROPPED (rule + counts in ``preprocess/dropped_records.json``):

1. ``guide_targets_no_gene`` (S1 Table only, 5,063 guides): the released row carries no
   gene, so there is no target to write a gene perturbation against. S4 and S6 exclude
   these upstream.
2. ``guide_targets_the_template_strand`` (S1 Table only, 30,811 guides): ``coding`` is
   FALSE. dCas9 bound to the template strand does not block elongation, which is why the
   paper's own gene scores use coding-strand guides only ("For each gene, the median
   log2FC value of the sgRNAs targeting the coding strand was used for ranking") and why
   the phage screens exclude them ("guides targeting the template strand of genes or
   outside of genes were excluded from the analysis"). S2 Table makes the size of the
   difference explicit: the median ``median_template`` is the no-effect baseline while the
   median ``median_coding`` of an essential gene is -5.12. The only leaf available is
   ``BacterialCrisprInterferencePerturbation``, whose ``expression_direction`` is
   ``"decreased"`` with no slot for the target strand, so storing a template-strand guide
   would assert a knockdown the paper's own data says did not happen.
Rules 3 and 4, which dropped the remaining S1 rows as served by, or below the read floor
of, Cui 2018's released table, are GONE (#905): those 23,372 rows are the growth
screen's records.
The three symbol rules that used to follow (``gene_symbol_is_not_in_the_mg1655_annotation``
for a name GCA_000005845.2 retires, such as ``insA-7`` or ``lomR_1``;
``gene_symbol_collides_with_another_symbol_on_one_mg1655_locus`` for two names the
annotation merges, such as ``yaiT``/``yaiU``; and ``gene_symbol_is_ambiguous_in_mg1655``)
are GONE: they removed 444 records over 111 symbols, which are stored as released now,
and what the pinned assembly makes of a symbol is a verifier finding (#1069).

Retention arithmetic: 59,246 + 3 x 17,220 + 17,220 = 128,126 released (guide, screen)
cells. S1 Table: 59,246 - 5,063 - 30,811 = 23,372 records; every one of the 68,880
phage-derived cells is a record; 92,252 records over 3,944 released symbols. The
candidate table's 236,000 is 59,000
guides x 4 conditions, which counts every guide in every condition; the phage and
transduction screens were released on the filtered 17,220-guide library.

SOURCED VALUES (module-level ``SourcedValue``s anchored to the sha256 of
``roussetGenomewideCRISPRdCas9Screens2018/paper.md`` or of
``cuiCRISPRiScreenColi2018/paper.md``, or typed ``ProvenanceGap``s):

- ``n_samples = 3``, ``sample_unit=biological_replicate`` for every record: "The phage
  screen was performed in triplicates as follows".
- Uncertainty: a ``not_reported_by_primary`` gap on
  ``environment_response_uncertainty``. The released tables carry one log2FC per guide
  and screen; DESeq2's ``lfcSE`` is not among the released columns.
- ``multiplicity_of_infection = 1.0`` and ``host_of_propagation = "MG1655"`` for all three
  phages; no phage family, genome type, taxon or accession is stated, so those stay None.
- ``titer_pfu_per_ml``: the stocks are 10^7 pfu/microL but the infected culture's volume
  at MOI 1 is not stated, so the in-culture titer is not computed.
- ``Temperature(37.0)``, ``duration_hours = 2.0`` and the medium (including its 1 microM
  aTc) from the phage-screen paragraph; ``aerobic`` from shaken flask cultures with no gas
  control described.
- The growth screen's culture, host and replicate count: Cui 2018's LB, 1 nM aTc,
  37 C and Supplementary Table 7 LC-E75 record (by the stated deferral), and this
  paper's own "17 generations in triplicates" (``GROWTH_SCREEN_REPLICATES``).
- The padj correction, from the deposited DESeq2 reference (``PADJ_CORRECTION``).

DATA SOURCE: S1, S4 and S6 Tables (``pgen.1007749.s011.csv``, ``.s014.csv``,
``.s016.csv``) from the PMC Article Datasets bucket (``pmc_cloud``, prefix
``PMC6242692.1``), deposited in
``$DATA_ROOT/torchcell-raw/roussetGenomewideCRISPRdCas9Screens2018/`` with a
``manifest.json``, together with the DESeq2 reference's PMC text (``ref/``). No other
source's table is read. S2, S5 and S7 Tables are the
gene-level medians and model estimates derived from the three above, so they are not
consumed; S3 Table, S8 to S10 Tables and the ten SI figures carry no per-guide value. The
ENA BioProject PRJEB28256 holds the raw reads the log2FC values were computed from and is
recorded as the upstream accession, not mirrored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
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
    AssayType,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialReleasedCrisprPerturbation,
    BacterialStrainBackground,
    ComponentDefinition,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    Environment,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    PhagePerturbation,
    Publication,
    SampleUnit,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    STRAIN_GENE_NAMESPACES,
    assembly_reference,
    bacterial_genome,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.ecoli import cui2018
from torchcell.literature.manifest import (
    ROLE_PAPER_TEXT,
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.retrieve import pmc_cloud_url
from torchcell.sequence.genome.ecoli.k12 import EcoliK12MG1655Genome, EcoliK12StrainName
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

DOI = "10.1371/journal.pgen.1007749"
PMID = "30403660"
PMCID = "PMC6242692"
TITLE = (
    "Genome-wide CRISPR-dCas9 screens in E. coli identify essential genes and phage "
    "host factors"
)

CITATION_KEY = "roussetGenomewideCRISPRdCas9Screens2018"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "46ea72979c7f11855477b557824fb62baa7a4937ae4787d930e706cd2b93db3f"

#: The method paper the growth screen defers to ("obtained from our previous study
#: [26]"), mirrored, so the deferral is followed rather than recorded as a gap.
CUI2018_KEY = "cuiCRISPRiScreenColi2018"
CUI2018_DOI = "10.1038/s41467-018-04209-5"
CUI2018_PAPER_MD_SHA256 = (
    "b8e28e16f0cbb4f8d4061f79505409c1a9aa03ed4ba8349ba4dca2327d26d22d"
)
CUI2018_DATASET = "crispri_knockdown_cui2018"
CUI2018_DATASET_CLASS = "CrispriKnockdownCui2018Dataset"
CUI2018_SCREEN_ID = "LC-E75"

#: The reference the paper's fold changes defer to ("computed the log2-transformed fold
#: change (log2FC) from DESeq2 R package [32]"): Love, Huber and Anders 2014, Genome
#: Biology 15:550 (PMC4302049). Its PMC plain-text object is deposited in THIS raw mirror
#: (#905), by the same scriptable pmc_cloud route as the tables, as the quote anchor for
#: the correction behind S1 Table's ``padj``. It is a cited reference, not a dataset, so
#: it does not go into Zotero.
DESEQ2_REFERENCE_DOI = "10.1186/s13059-014-0550-8"
DESEQ2_REFERENCE_FILENAME = "PMC4302049.1.txt"
DESEQ2_REFERENCE_REL = f"ref/{DESEQ2_REFERENCE_FILENAME}"
DESEQ2_REFERENCE_KEY = f"PMC4302049.1/{DESEQ2_REFERENCE_FILENAME}"
DESEQ2_REFERENCE_SHA256 = (
    "f5debc17b0b777c08387c966301aa5080cd6caf1da3fe92030e1b93ac4e1c2dd"
)
DESEQ2_REFERENCE_RETRIEVED_AT = "2026-10-12"

#: The ENA BioProject holding the raw reads behind every released log2FC. Recorded as
#: the upstream accession; the loader consumes the computed tables, not the reads.
ENA_BIOPROJECT = "PRJEB28256"

RAW_RETRIEVED_AT = "2026-10-07"

GROWTH_TABLE = "pgen.1007749.s011.csv"
PHAGE_TABLE = "pgen.1007749.s014.csv"
TRANSDUCTION_TABLE = "pgen.1007749.s016.csv"

#: Each consumed table: its mirror-relative path, its sha256, and its PMC bucket key.
TABLE_SHA256: dict[str, str] = {
    GROWTH_TABLE: "015ebda56925fee97cd2c44557b146799299f883b2255ed22836ebab61875ebb",
    PHAGE_TABLE: "3ee887e7596d2917f1fde1ce41dfcea0b79538ece0f0d654e01dad31ec1686a2",
    TRANSDUCTION_TABLE: (
        "c960a214350944f6c98e945410ee0a7d09b18e587f3ee9cf16a7b6e4cd360b2f"
    ),
}
TABLE_LABEL: dict[str, str] = {
    GROWTH_TABLE: "S1 Table",
    PHAGE_TABLE: "S4 Table",
    TRANSDUCTION_TABLE: "S6 Table",
}


def raw_sha256() -> dict[str, str]:
    """Every raw file the build consumes and verifies, with its pinned sha256.

    Rousset's three tables. No other source's table is read (#905): the growth screen
    is stored from S1 Table as released, never cross-referenced against Cui 2018's.
    """
    return dict(TABLE_SHA256)


def table_rel(filename: str) -> str:
    """Mirror-relative path of one consumed SI table."""
    return f"data/{filename}"


def table_key(filename: str) -> str:
    """PMC Article Datasets bucket key of one consumed SI table."""
    return f"{PMCID}.1/{filename}"


def table_url(filename: str) -> str:
    """HTTPS URL of one consumed SI table in the PMC Article Datasets bucket."""
    return pmc_cloud_url(table_key(filename))


#: Fraction of the released gene symbols that must resolve to one MG1655 locus
#: (plan checklist item 4). Measured on the release: 3,912 of 3,944 (0.9919).

#: The b-number namespace every record is written in (both host strains are MG1655
#: derivatives written against the MG1655 assembly).
MG1655_NAMESPACE = STRAIN_GENE_NAMESPACES["MG1655"]
REFERENCE_STRAIN_NAME: EcoliK12StrainName = "MG1655"


# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned Rousset 2018 OCR mirror."""
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


def _cui(value: Any, quote: str, *, page: str, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned Cui 2018 OCR mirror.

    Rousset's growth screen defers its culture detail to this paper, so the value is
    sourced from THAT mirror and the deferral chain is the citation (CLAUDE.md, "Follow
    deferrals").
    """
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CUI2018_KEY,
            sha256=CUI2018_PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page=page,
        ),
    )


_STRAINS = "Materials and Methods, 'E. coli strain construction'"
_LIBRARY = "Materials and Methods, 'CRISPRi library design and assembly'"
_SCREENS = "Materials and Methods, 'High-throughput screens'"
_ANALYSIS = "Materials and Methods, 'Data analysis'"
_PHAGE_STOCKS = "Materials and Methods, 'Phage strains and stocks'"
_SI = "Supporting information"
_CUI_SEQUENCING = "Methods, 'Library sequencing'"
_CUI_FOLD_CHANGE = "Methods, 'Fold-change computation'"

LIBRARY_TARGETS_MG1655 = _paper(
    "MG1655",
    "These sgRNAs target 20-nt regions adjacent to NGG sites in E. coli K-12 MG1655 "
    "(NC_000913.2) and were chosen randomly among the total pool of possible sgRNAs in "
    "this strain.",
    page=_LIBRARY,
    note="the guide library, and therefore every target gene the release names, is "
    "defined against MG1655; both screened strains are MG1655 derivatives",
)
FILTERED_LIBRARY = _paper(
    59000,
    "yielding a library of \\~ 59,000 guides used to perform the analyses below "
    "(S1 Table)",
    page="Results, 'Identifying essential genes'",
    note="the growth screen's released library; S1 Table carries 59,246 rows, and its "
    "coding-strand in-gene guides are stored as the growth_LC-E75 screen (#905)",
)
LC_E75_CASSETTE = _paper(
    "LC-E75",
    "This strain expresses an optimized dcas9 cassette under the control of an "
    "aTc-inducible pTet promoter integrated at the phage 186 attB site.",
    page=_STRAINS,
    note="the growth screen's host: MG1655 plus a chromosomal aTc-inducible dcas9 "
    "cassette at the phage 186 attB site; the stored background is Cui 2018's own "
    "LC-E75 record (Supplementary Table 7), the study this screen defers to",
)
FR_E01_CASSETTE = _paper(
    "FR-E01",
    "a new strain FR-E01 was constructed with the same cassette integrated at the HK022 "
    "attB site to avoid any interference",
    page=_STRAINS,
    note="the phage screens' host: the same cassette moved to the HK022 attB site so "
    "that phage 186 can be screened",
)
FR_E01_PARENT = _paper(
    "MG1655",
    "The resulting vector was electroporated into strain MG1655.",
    page=_STRAINS,
    note="FR-E01 is built in MG1655; the pOSIP backbone is then removed with pE-FLP",
)
EFFECTOR = _paper(
    "dCas9",
    "A fragment containing dcas9 under the control of pTet promoter was amplified by "
    "Phusion PCR (ThermoScientific)",
    page=_STRAINS,
    note="the guide-directed effector of every record; the paper's title names the "
    "screens CRISPR-dCas9",
)
GROWTH_SCREEN_DEFERRAL = _paper(
    CUI2018_KEY,
    "The data for the screen performed with strain LC-E75 grown in rich medium was "
    "obtained from our previous study [26]. This screen was performed over 17 "
    "generations in triplicates from independent aliquots of the library generated from "
    "3 independent transformations into strain LC-E75.",
    page=_SCREENS,
    note="reference [26] is Cui 2018 (mirrored); the culture (LB, 1 nM aTc, 37 C) is "
    "sourced from it by this deferral, and the 17 generations and the triplicate are "
    "this sentence's own. S1 Table is Rousset's own DESeq2 run over that screen's "
    "reads and is stored as released (#905): the record cites this paper, carries its "
    "own padj, and is never compared with Cui's released fit75 inside the loader",
)
PHAGE_SCREEN_TRIPLICATE = _paper(
    3,
    "The phage screen was performed in triplicates as follows: FR-E01 was grown at "
    "$3 7 ^ { \\circ } \\mathrm { C }$ from 1 mL aliquots stored at "
    "$- 8 0 ~ ^ { \\circ } \\mathrm { C }$ into $5 0 0 ~ \\mathrm { m L }$ LB.",
    page=_SCREENS,
    note="the three phage challenges and the transduction assay are arms of this one "
    "triplicate series, so every record carries n_samples = 3",
)
PHAGE_SCREEN_TEMPERATURE = _paper(
    37.0,
    "FR-E01 was grown at $3 7 ^ { \\circ } \\mathrm { C }$ from 1 mL aliquots stored at",
    page=_SCREENS,
)
PHAGE_SCREEN_INDUCTION = _paper(
    (1.0, "uM"),
    "dCas9 expression was induced by addition of $1 \\mu \\mathrm { M }$ aTc (Acros "
    "Organics) to trigger the silencing of the target genes.",
    page=_SCREENS,
)
PHAGE_SCREEN_MEDIUM = _paper(
    ("LB", 0.2, 5.0),
    "diluted 100-fold in LB containing $1 \\mu \\mathrm { M }$ aTc, $0 . 2 \\%$ Maltose "
    "and $5 \\mathrm { m M } \\mathrm { C a C l } _ { 2 }$ .",
    page=_SCREENS,
    note="the medium the library is in when it is infected: LB plus 0.2% maltose and "
    "5 mM CaCl2, with the inducer carried over at 1 microM",
)
PHAGE_INFECTION = _paper(
    (1.0, 2.0),
    "The culture was then infected with $1 \\mathrm { m L }$ of $\\lambda$ , T4 or "
    "$1 8 6 c \\mathrm { I }$ -ts stocks "
    "$\\left( 1 0 ^ { 7 } \\mathrm { p f u } / \\mu \\mathrm { L } \\right)$ to reach a "
    "MOI of 1 ensuring a high infection rate while limiting double infections. After "
    "$^ { 2 \\mathrm { h } }$ at $3 7 ^ { \\circ } \\mathrm { C } ,$ the cultures were "
    "harvested",
    page=_SCREENS,
    note="MOI 1 and a 2 h challenge; the stock titer is 10^7 pfu/microL but the "
    "infected culture's volume is not stated, so no in-culture titer is computed",
)
PHAGE_NAMES = _paper(
    ("lambda", "T4", "186cIts"),
    "followed by infection with phage λ, T4 or 186cIts at a multiplicity of infection "
    "(MOI) of 1 (Fig 3A)",
    page="Results, 'A CRISPR-dCas9 screen to identify phage host factors'",
    note="the S4 Table columns name the same three phages log2FC_lambda, log2FC_T4 and "
    "log2FC_186; 'lambda' is the released table's own spelling of the prose glyph",
)
PHAGE_PROPAGATION_HOST = _paper(
    "MG1655",
    "All the liquid stocks were further propagated in MG1655 grown in LB supplemented "
    "with maltose $0 . 2 \\%$ (Sigma) and $\\mathrm { C a C l } _ { 2 }$ 5 mM (Sigma) at "
    "a multiplicity of infection (MOI) of 1.",
    page=_PHAGE_STOCKS,
)
TRANSDUCTION_READOUT = _paper(
    "cosmid recovered after lambda transduction",
    "the cell lysate containing a mixture of $\\lambda$ and cosmid particles was used "
    "to transduce strain $\\mathrm { M G } 1 6 5 5 { \\because } { \\lambda }$ and thus "
    "recover guides in the library that do not affect the infection process. The "
    "distribution of sgRNAs recovered after transduction was compared to the initial "
    "pool to identify depleted sgRNAs corresponding to bacterial genes required for the "
    "production of functional capsids (S6 Table).",
    page="Results, 'Identifying host factors required for the production of functional "
    "phages'",
    note="the readout is the guide-carrying cosmid packaged by lambda, not the surviving "
    "cell pool, so this arm's assay_type is 'other' while its environment is the same "
    "lambda challenge at MOI 1",
)
LOG2FC_STATISTIC = _paper(
    "log2 fold change of sgRNA abundance",
    "We used the number of reads as a measure of the abundance of each guide in the "
    "library and computed the log2-transformed fold change (log2FC) from DESeq2 R "
    "package [32] as a measure of relative sgRNA fitness.",
    page="Results, 'Identifying essential genes'",
)
LOG2FC_NORMALIZATION = _paper(
    "non-targeting control guide, paired against each sample's initial condition",
    "were used as the normalization factor. A paired analysis was performed to compare "
    "each sample to its initial condition. The log2FoldChange "
    "$( \\log 2 \\mathrm { F C } )$ value represents the enrichment or depletion of each "
    "sgRNA.",
    page=_ANALYSIS,
    note="what the stored number is: a signed enrichment or depletion, so the reference "
    "value is 0",
)
CODING_STRAND_RULE = _paper(
    "coding strand",
    "For each gene, the median log2FC value of the sgRNAs targeting the coding strand "
    "was used for ranking (S2 and S5 Tables).",
    page=_ANALYSIS,
    note="the paper's own gene scores use only coding-strand guides, which is the rule "
    "the template-strand drop follows",
)
PHAGE_SCREEN_FILTER = _paper(
    17200,
    "For the phage screen, guides targeting the template strand of genes or outside of "
    "genes were excluded from the analysis as well as well guides with insufficient "
    "number of reads (BaseMean $< 1 0 $ ), yielding a library of \\~ 17,200 sgRNAs.",
    page=_ANALYSIS,
    note="S4 and S6 Tables carry 17,220 rows each, on the same guide set",
)
RELEASED_TABLES = _paper(
    ("S1 Table", "S4 Table", "S6 Table"),
    "The lists of all sgRNAs with computed log2FC values after the growth-based screen, "
    "after phage screens and after transduction assay are provided as S1, S4 and S6 "
    "Tables respectively.",
    page=_ANALYSIS,
    note="the three guide-level tables this dataset is built from; S2, S5 and S7 Tables "
    "are gene-level summaries derived from them",
)
GROWTH_TABLE_COLUMNS = _paper(
    ("log2FC", "padj", "gamma"),
    "1 Table. List of sgRNAs with log2FC values after the growth-based screen in strain "
    "LC-E75. Each sgRNA is indexed with its position, orientation (ori), target strand "
    "(coding) and computed fold change (log2FC, padj and gamma).",
    page=_SI,
    note="the caption names gamma but neither this paper nor Cui 2018 defines it, so it "
    "is not stored",
)
PHAGE_TABLE_COLUMNS = _paper(
    ("log2FC_lambda", "log2FC_T4", "log2FC_186"),
    "S4 Table. List of sgRNAs with log2FC values after phage screens. Each sgRNA is "
    "indexed with its position, orientation (ori), strand (coding), information on the "
    "targeted gene (gene, essential, gene_left, gene_right, gene_ori) and computed fold "
    "change for each phage (log2F-C_lambda, log2FC_T4 and log2FC_186).",
    page=_SI,
)
TRANSDUCTION_TABLE_COLUMNS = _paper(
    ("log2FC",),
    "S6 Table. List of sgRNAs with log2FC values after transduction screen. Each sgRNA "
    "is indexed with its position, orientation (ori), strand (coding), information on "
    "the targeted gene (gene, essential, gene_left, gene_right, gene_ori) and computed "
    "fold change after transduction (log2FC).",
    page=_SI,
)
GROWTH_SCREEN_REPLICATES = _paper(
    3,
    "This screen was performed over 17 generations in triplicates from independent "
    "aliquots of the library generated from 3 independent transformations into strain "
    "LC-E75.",
    page=_SCREENS,
    note="n_samples = 3 biological replicates on every growth-screen record",
)
GROWTH_PADJ = _paper(
    "padj",
    "computed fold change (log2FC, padj and gamma)",
    page=_SI,
    note="S1 Table's padj column, numeric on all 59,246 rows (0 to 0.999997, measured "
    "2026-10-12), is stored verbatim on environment_response_p_value_adjusted. No "
    "unadjusted p-value is released beside it, so environment_response_p_value stays "
    "None",
)
RAW_READS_ACCESSION = _paper(
    ENA_BIOPROJECT,
    "Raw sequencing files are available on the European Nucleotide Archive "
    "(https://www.ebi.ac. uk/ena) with the accession number PRJEB28256.",
    page=_ANALYSIS,
)


def _deseq2(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the deposited DESeq2 reference (ref. 32)."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=DESEQ2_REFERENCE_REL,
            citation_key=CITATION_KEY,
            sha256=DESEQ2_REFERENCE_SHA256,
            method="PMC plain-text object of Love, Huber and Anders 2014 "
            f"(doi:{DESEQ2_REFERENCE_DOI}), deposited in this raw mirror",
            page=page,
        ),
    )


#: The correction behind S1 Table's padj, by the deferral this paper states
#: (its ref. 32 is DESeq2).
P_VALUE_ADJUSTMENT_METHOD = "benjamini_hochberg"
PADJ_CORRECTION = _deseq2(
    P_VALUE_ADJUSTMENT_METHOD,
    "The Wald test P values from the subset of genes that pass an independent filtering "
    "step, described in the next section, are adjusted for multiple testing using the "
    "procedure of Benjamini and Hochberg [21].",
    page="Results and discussion, 'Hypothesis tests for differential expression'",
    note="Rousset computes its fold changes 'from DESeq2 R package [32]' "
    "(LOG2FC_STATISTIC); ref. 32 is this paper, which states the correction DESeq2 "
    "applies. Hypothesis (unverified): Rousset ran DESeq2's default adjustment; the "
    "paper does not name a non-default one",
)

#: Every module-level sourced value, for the mirror audit.
SOURCED_VALUES: tuple[SourcedValue, ...] = (
    LIBRARY_TARGETS_MG1655,
    FILTERED_LIBRARY,
    LC_E75_CASSETTE,
    FR_E01_CASSETTE,
    FR_E01_PARENT,
    EFFECTOR,
    GROWTH_SCREEN_DEFERRAL,
    PHAGE_SCREEN_TRIPLICATE,
    PHAGE_SCREEN_TEMPERATURE,
    PHAGE_SCREEN_INDUCTION,
    PHAGE_SCREEN_MEDIUM,
    PHAGE_INFECTION,
    PHAGE_NAMES,
    PHAGE_PROPAGATION_HOST,
    TRANSDUCTION_READOUT,
    LOG2FC_STATISTIC,
    LOG2FC_NORMALIZATION,
    CODING_STRAND_RULE,
    PHAGE_SCREEN_FILTER,
    RELEASED_TABLES,
    GROWTH_TABLE_COLUMNS,
    PHAGE_TABLE_COLUMNS,
    TRANSDUCTION_TABLE_COLUMNS,
    RAW_READS_ACCESSION,
    GROWTH_SCREEN_REPLICATES,
    GROWTH_PADJ,
)


# --------------------------------------------------------------------------- #
# Media and strain backgrounds
# --------------------------------------------------------------------------- #
def _lb_ingredients(provenance: SourcedValue) -> list[MediaComponent]:
    """LB's three ingredients with NO amounts.

    ``LB`` in the media library carries the Miller amounts, corroborated by four other
    papers. Rousset 2018 states no formulation (and the project's own "LB" is Miller for
    some rows and Lennox for others), so asserting either would fabricate three numbers.
    The identities are LB's definition; the amounts are ``None``, which is what an
    unsourced amount means.
    """
    return [
        MediaComponent(
            compound=resolved_compound("tryptone"),
            role=MediaComponentRole.complex_ingredient,
            concentration=None,
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[provenance],
            note="LB ingredient; no amount is stated by the paper",
        ),
        MediaComponent(
            compound=resolved_compound("yeast extract"),
            role=MediaComponentRole.complex_ingredient,
            concentration=None,
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[provenance],
            note="LB ingredient; no amount is stated by the paper",
        ),
        MediaComponent(
            compound=resolved_compound("sodium chloride"),
            role=MediaComponentRole.bulk_salt,
            concentration=None,
            provenance=[provenance],
            note="LB ingredient; no amount is stated by the paper, and the Miller "
            "and Lennox formulations differ in exactly this component",
        ),
    ]


def _atc(
    value: float, unit: ConcentrationUnit, provenance: SourcedValue
) -> MediaComponent:
    """The anhydrotetracycline that induces dCas9, as a component of the medium.

    It is in the medium, not on ``Environment.perturbations``, because the paper puts it
    there. The Methods write aTc twice in recipe form, as an item of a medium's
    composition beside components this medium already carries: "diluted 100-fold in LB
    containing 1 microM aTc, 0.2% Maltose and 5 mM CaCl2" (High-throughput screens, the
    very culture the sgRNA distributions are sampled from, ``PHAGE_SCREEN_MEDIUM``) and
    "Strains were grown overnight and diluted 100-fold in LB medium containing 0.2%
    maltose, 1 microM aTc, 5 mM CaCl2 and kanamycin" (Infection dynamics, paper.md line
    210, sha256 46ea72979c7f11855477b557824fb62baa7a4937ae4787d930e706cd2b93db3f).

    The paper ALSO writes it once as an event, in the sentence immediately before the
    first of those: "dCas9 expression was induced by addition of 1 microM aTc (Acros
    Organics) to trigger the silencing of the target genes" (``PHAGE_SCREEN_INDUCTION``,
    kept beside the recipe quote so both readings are on the record). That sentence is
    the initiating addition in the pre-infection outgrowth; every culture downstream of
    it, including the one that is sampled, is described as a medium containing aTc. The
    recipe reading is therefore the one that covers the stored records, and it is the
    reading this loader follows.

    Not a reason, a consequence: aTc is 1 microM in every record and in the reference, so
    on the environment axis it would be a perturbation nothing in the dataset contrasts.
    The adapter does not enter into it. Until issue #756 the served
    ``_environment_perturbation_node`` emitted phages too, so a conf could enable either
    the phage lane or the environment lane and not both; the two lanes now partition the
    leaves, and this placement would be unchanged either way.

    The dose is stored rather than noted because it is what switches the knockdown on,
    and because it is three orders of magnitude above the 1 nM Cui 2018 used on the same
    cassette.
    """
    return MediaComponent(
        compound=resolved_compound("anhydrotetracycline"),
        role=MediaComponentRole.other,
        concentration=Concentration(value=value, unit=unit),
        provenance=[provenance],
        note="inducer of the chromosomal pTet-dcas9 cassette; it switches the "
        "perturbation on rather than acting on the cell's metabolism, which is why its "
        "role is 'other' rather than a nutrient role",
    )


ROUSSET2018_LB_MALTOSE_CACL2 = Media(
    name="LB with 1 uM aTc, 0.2% maltose and 5 mM CaCl2, formulation not stated "
    "(Rousset 2018 phage screens), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium="LB",
    components=[
        *_lb_ingredients(PHAGE_SCREEN_MEDIUM),
        _atc(
            PHAGE_SCREEN_INDUCTION.value[0],
            ConcentrationUnit.micromolar,
            PHAGE_SCREEN_INDUCTION,
        ),
        MediaComponent(
            compound=resolved_compound("maltose"),
            role=MediaComponentRole.carbon_source,
            concentration=Concentration(value=0.2, unit=ConcentrationUnit.percent_w_v),
            provenance=[PHAGE_SCREEN_MEDIUM],
            note="induces the lambda receptor LamB; the paper writes 0.2% with no w/v "
            "or v/v marker and maltose is a solid, so it is read as w/v",
        ),
        MediaComponent(
            compound=resolved_compound("calcium chloride"),
            role=MediaComponentRole.bulk_salt,
            concentration=Concentration(value=5.0, unit=ConcentrationUnit.millimolar),
            provenance=[PHAGE_SCREEN_MEDIUM],
            note="required for phage adsorption",
        ),
    ],
    provenance=[PHAGE_SCREEN_MEDIUM, PHAGE_SCREEN_INDUCTION],
)
"""The phage screens' medium: LB plus the aTc, maltose and CaCl2 the paper lists."""

FR_E01_BACKGROUND = BacterialStrainBackground(
    name="FR-E01",
    reference_strain=REFERENCE_STRAIN_NAME,
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    parents=["MG1655"],
    construction="MG1655 carrying the same pTet-dcas9 cassette integrated at the HK022 "
    "attB site, so that phage 186 can be screened without interference; the pOSIP "
    "backbone was removed with pE-FLP",
    genotype_statement="MG1655 HK022attB::pTet-dcas9 (optimized cassette)",
    provenance=[FR_E01_CASSETTE, FR_E01_PARENT],
)
"""The phage screens' host strain, as a background on the MG1655 assembly."""


# --------------------------------------------------------------------------- #
# The four stored screens
# --------------------------------------------------------------------------- #
#: The two hosts: FR-E01 for the four phage-derived screens and LC-E75 for the growth
#: screen (#905). The label stays a ``Literal`` so a condition cannot name a strain with
#: no background. LC-E75's background is Cui 2018's own record of that strain
#: (Supplementary Table 7), the study the growth screen defers to, so the two datasets
#: name one strain the same way.
StrainLabel = Literal["FR-E01", "LC-E75"]

BACKGROUNDS: dict[StrainLabel, BacterialStrainBackground] = {
    "FR-E01": FR_E01_BACKGROUND,
    "LC-E75": cui2018.strain_background("LC-E75"),
}


class ScreenCondition(BaseModel):
    """One released screen: which table and column it is, and what the culture was.

    ``screen_id`` goes onto the phenotype, where it joins the environment-response
    verifier's condition signature. That is what keeps the lambda challenge and the
    lambda transduction assay, whose cultures are identical, from reading as duplicate
    measurements of one condition.
    """

    model_config = ConfigDict(frozen=True)

    screen_id: str
    table: str
    column: str
    strain: StrainLabel
    #: ``None`` for the growth screen, which challenges no phage.
    phage: str | None
    assay_type: AssayType
    units: str


#: The growth screen's id (#905): S1 Table, strain LC-E75.
GROWTH_SCREEN_ID = "growth_LC-E75"

CONDITIONS: tuple[ScreenCondition, ...] = (
    ScreenCondition(
        screen_id=GROWTH_SCREEN_ID,
        table=GROWTH_TABLE,
        column="log2FC",
        strain="LC-E75",
        phage=None,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        units="log2(sgRNA abundance after 17 generations of dCas9 induction in LC-E75 "
        "/ initial pool), Rousset 2018's own DESeq2 run (S1 Table) over the screen of "
        "Cui 2018, normalized on a non-targeting control guide; padj is DESeq2's "
        "Benjamini-Hochberg adjusted p-value of the same test",
    ),
    ScreenCondition(
        screen_id="phage_lambda",
        table=PHAGE_TABLE,
        column="log2FC_lambda",
        strain="FR-E01",
        phage="lambda",
        assay_type=AssayType.pooled_competitive_growth_barcode,
        units="log2(sgRNA abundance in the surviving cell pool after 2 h of phage "
        "lambda at MOI 1 / pool before infection), DESeq2, normalized on a "
        "non-targeting control guide",
    ),
    ScreenCondition(
        screen_id="phage_T4",
        table=PHAGE_TABLE,
        column="log2FC_T4",
        strain="FR-E01",
        phage="T4",
        assay_type=AssayType.pooled_competitive_growth_barcode,
        units="log2(sgRNA abundance in the surviving cell pool after 2 h of phage T4 at "
        "MOI 1 / pool before infection), DESeq2, normalized on a non-targeting control "
        "guide",
    ),
    ScreenCondition(
        screen_id="phage_186cIts",
        table=PHAGE_TABLE,
        column="log2FC_186",
        strain="FR-E01",
        phage="186cIts",
        assay_type=AssayType.pooled_competitive_growth_barcode,
        units="log2(sgRNA abundance in the surviving cell pool after 2 h of phage "
        "186cIts at MOI 1 / pool before infection), DESeq2, normalized on a "
        "non-targeting control guide",
    ),
    ScreenCondition(
        screen_id="lambda_transduction",
        table=TRANSDUCTION_TABLE,
        column="log2FC",
        strain="FR-E01",
        phage="lambda",
        assay_type=AssayType.other,
        units="log2(sgRNA abundance in the cosmid packaged by phage lambda and "
        "transduced into MG1655::lambda / pool before infection), DESeq2, normalized on "
        "a non-targeting control guide; the readout is packaged cosmid, not the "
        "surviving cell pool",
    ),
)

#: The released columns each table's records read: S1 Table's log2FC and padj, and the
#: log2FC columns of S4 and S6 Tables.
TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    GROWTH_TABLE: ("log2FC", "padj"),
    **{
        table: tuple(c.column for c in CONDITIONS if c.table == table)
        for table in (PHAGE_TABLE, TRANSDUCTION_TABLE)
    },
}


def phage_perturbation(name: str) -> PhagePerturbation:
    """One phage challenge at MOI 1, propagated on MG1655.

    No family, genome type, NCBI taxon or genome accession is stated for any of the three
    phages, so those stay None. The stock titer (10^7 pfu/microL) is stated but the
    infected culture's volume is not, so ``titer_pfu_per_ml`` is left unset rather than
    back-computed.
    """
    return PhagePerturbation(
        name=name,
        multiplicity_of_infection=1.0,
        host_of_propagation=PHAGE_PROPAGATION_HOST.value,
    )


def environment(condition: ScreenCondition) -> Environment:
    """The culture one screen was run in.

    The growth screen's is Cui 2018's pooled competition (LB, 1 nM aTc, 37 C, 17
    generations), by the deferral this paper states (``GROWTH_SCREEN_DEFERRAL``); a phage
    screen's is the one phage-screen medium plus its phage.
    """
    if condition.phage is None:
        return cui2018.screen_environment()
    return Environment(
        media=ROUSSET2018_LB_MALTOSE_CACL2,
        temperature=Temperature(value=PHAGE_SCREEN_TEMPERATURE.value),
        perturbations=[phage_perturbation(condition.phage)],
        aerobicity="aerobic",
        duration_hours=PHAGE_INFECTION.value[1],
    )


def _uncertainty_gap() -> ProvenanceGap:
    """The released tables carry no dispersion for a guide's log2FC."""
    return ProvenanceGap(
        field="environment_response_uncertainty",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="each table releases one log2FC per guide and screen; DESeq2's lfcSE is "
        "not a released column. S1 Table also releases padj (stored on the adjusted "
        "p-value field, a test and not a dispersion) and gamma (named in the S1 Table "
        "caption and defined in neither mirrored paper, so not stored)",
    )


def phenotype(
    value: float, condition: ScreenCondition, padj: float | None = None
) -> EnvironmentResponsePhenotype:
    """One released log2FC cell, with S1 Table's padj where the row releases one."""
    replicates = (
        GROWTH_SCREEN_REPLICATES if condition.phage is None else PHAGE_SCREEN_TRIPLICATE
    )
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=condition.assay_type,
        environment_response=value,
        environment_response_p_value_adjusted=padj,
        p_value_adjustment_method=(
            None if padj is None else str(PADJ_CORRECTION.value)
        ),
        n_samples=replicates.value,
        sample_unit=SampleUnit.biological_replicate,
        units=condition.units,
        screen_id=condition.screen_id,
        provenance_gaps=[_uncertainty_gap()],
    )


def reference_phenotype(condition: ScreenCondition) -> EnvironmentResponsePhenotype:
    """The baseline: a guide whose abundance did not change has log2FC 0."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=condition.assay_type,
        environment_response=0.0,
        units=condition.units,
        screen_id=condition.screen_id,
        provenance_gaps=[
            ProvenanceGap(
                field="n_samples",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the 0 baseline is what the DESeq2 paired analysis against each "
                "sample's own initial condition puts a guide of unchanged abundance "
                "at, not a measured set of control replicates",
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/roussetGenomewideCRISPRdCas9Screens2018``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_retrieval(
    filename: str, retrieved_at: str = RAW_RETRIEVED_AT
) -> RetrievalRecord:
    """The recorded retrieval of one SI table: one PMC Article Datasets object.

    ``run_retriever`` on this record re-fetches the bytes; the pinned sha256 is the
    anchor a rebuild verifies against.
    """
    return RetrievalRecord(
        method=RetrievalMethod.pmc_cloud,
        source_url=table_url(filename),
        retriever="torchcell.literature.retrieve.pmc_cloud_object",
        params={"key": table_key(filename)},
        sha256=TABLE_SHA256[filename],
        retrieved_at=retrieved_at,
    )


def deposit_raw_mirror(
    *,
    table_paths: Mapping[str, str | Path],
    retrieved_at: str = RAW_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from the already-retrieved SI tables plus ``manifest.json``.

    Idempotent by sha256: an existing mirror file with the recorded hash is left alone
    and a differing one raises rather than being overwritten. Each file's bytes are
    verified against ``TABLE_SHA256`` before anything is written.
    """
    missing = sorted(set(TABLE_SHA256) - set(table_paths))
    if missing:
        raise RuntimeError(f"no retrieved bytes given for {missing}")
    root = raw_mirror_dir(data_root)
    records: list[ArtifactRecord] = []
    for filename in TABLE_SHA256:
        source = table_paths[filename]
        got = _sha256(source)
        if got != TABLE_SHA256[filename]:
            raise RuntimeError(
                f"{source} sha256 mismatch: got {got}, expected {TABLE_SHA256[filename]}"
            )
        dest = root / table_rel(filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != TABLE_SHA256[filename]:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            dest.write_bytes(Path(source).read_bytes())
        records.append(
            ArtifactRecord(
                path=table_rel(filename),
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=TABLE_SHA256[filename],
                source=table_url(filename),
                retrieval=table_retrieval(filename, retrieved_at),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=records,
        si_data_sources=[table_url(filename) for filename in TABLE_SHA256],
        si_expected=[
            f"{TABLE_LABEL[GROWTH_TABLE]} ({GROWTH_TABLE}): per-sgRNA log2FC after the "
            "growth-based screen in strain LC-E75",
            f"{TABLE_LABEL[PHAGE_TABLE]} ({PHAGE_TABLE}): per-sgRNA log2FC after the "
            "phage lambda, T4 and 186cIts screens in strain FR-E01",
            f"{TABLE_LABEL[TRANSDUCTION_TABLE]} ({TRANSDUCTION_TABLE}): per-sgRNA "
            "log2FC after the lambda transduction screen",
            "S2, S5 and S7 Tables: gene-level medians and transduction model estimates "
            "DERIVED from the three tables above, so NOT mirrored",
            f"ENA BioProject {ENA_BIOPROJECT}: the raw reads the log2FC values were "
            "computed from, NOT mirrored (the loader consumes the computed tables)",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def deposit_deseq2_reference(data_root: str | None = None) -> Path:
    """Deposit the DESeq2 reference's PMC text into this raw mirror (#905).

    Runs the recorded ``pmc_cloud`` retrieval, refuses bytes that are not the pin, and
    adds one record to ``manifest.json`` without rewriting the records already there.
    Idempotent: a present file with the pinned hash and a present record are left alone.
    """
    from torchcell.literature.retrieve import pmc_cloud_object

    root = raw_mirror_dir(data_root)
    dest = root / DESEQ2_REFERENCE_REL
    if not dest.exists():
        payload = pmc_cloud_object(DESEQ2_REFERENCE_KEY)
        got = hashlib.sha256(payload).hexdigest()
        if got != DESEQ2_REFERENCE_SHA256:
            raise RuntimeError(
                f"{DESEQ2_REFERENCE_KEY} yields {got}, the pin is "
                f"{DESEQ2_REFERENCE_SHA256}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
    if _sha256(dest) != DESEQ2_REFERENCE_SHA256:
        raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    manifest = load_manifest(data_root)
    if any(record.path == DESEQ2_REFERENCE_REL for record in manifest.files):
        return dest
    record = ArtifactRecord(
        path=DESEQ2_REFERENCE_REL,
        role=ROLE_PAPER_TEXT,
        bytes=dest.stat().st_size,
        sha256=DESEQ2_REFERENCE_SHA256,
        source=pmc_cloud_url(DESEQ2_REFERENCE_KEY),
        original_filename=DESEQ2_REFERENCE_FILENAME,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=pmc_cloud_url(DESEQ2_REFERENCE_KEY),
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": DESEQ2_REFERENCE_KEY},
            sha256=DESEQ2_REFERENCE_SHA256,
            retrieved_at=DESEQ2_REFERENCE_RETRIEVED_AT,
        ),
    )
    updated = manifest.model_copy(
        update={
            "files": [*manifest.files, record],
            "si_expected": [
                *manifest.si_expected,
                f"{DESEQ2_REFERENCE_REL}: the PMC text of reference 32 (DESeq2, "
                f"doi:{DESEQ2_REFERENCE_DOI}), the quote anchor for the padj "
                "correction (#905)",
            ],
        }
    )
    (root / "manifest.json").write_text(updated.model_dump_json(indent=2))
    return dest


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    return load_manifest_of(CITATION_KEY, data_root)


def load_manifest_of(citation_key: str, data_root: str | None = None) -> Manifest:
    """Read one raw mirror's ``manifest.json``.

    Two mirrors are read: Rousset's own, and Cui 2018's, whose screen table the
    growth-screen de-duplication rule is defined by. Each file is checked against the
    manifest of the mirror it comes from, so neither mirror's pin stands in for the
    other's.
    """
    path = Path(data_root or _data_root()) / f"torchcell-raw/{citation_key}"
    return Manifest.model_validate_json((path / "manifest.json").read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Reading the three tables
# --------------------------------------------------------------------------- #
#: The header of each consumed table, verbatim and in release order. S6 Table prints
#: ``pos`` where S1 and S4 print ``position``, and prints ``gene_right`` before
#: ``gene_left``; both are the release's own spellings and are pinned rather than fixed.
TABLE_HEADERS: dict[str, tuple[str, ...]] = {
    GROWTH_TABLE: (
        "target",
        "position",
        "ori",
        "coding",
        "gene",
        "essential",
        "gene_left",
        "gene_right",
        "gene_ori",
        "log2FC",
        "padj",
        "gamma",
    ),
    PHAGE_TABLE: (
        "target",
        "position",
        "ori",
        "gene",
        "essential",
        "gene_left",
        "gene_right",
        "gene_ori",
        "log2FC_lambda",
        "log2FC_T4",
        "log2FC_186",
    ),
    TRANSDUCTION_TABLE: (
        "target",
        "pos",
        "ori",
        "gene",
        "essential",
        "gene_right",
        "gene_left",
        "gene_ori",
        "log2FC",
    ),
}


def read_table(path: str | Path, filename: str) -> pd.DataFrame:
    """Read one released table, refusing a header or a guide set that is not the pinned one.

    The ``gene`` column stays as read (``NaN`` for a guide that targets no gene) and the
    ``target`` spacer must be a distinct 20-nt string in every row.
    """
    frame = pd.read_csv(path)
    expected = list(TABLE_HEADERS[filename])
    if list(frame.columns) != expected:
        raise ValueError(f"{filename} header {list(frame.columns)} is not {expected}")
    spacers = frame["target"].astype(str)
    if not (spacers.str.len() == 20).all():
        raise ValueError(f"{filename}: every target spacer must be 20 nt")
    repeated = spacers[spacers.duplicated()].tolist()
    if repeated:
        raise ValueError(f"{filename} repeats spacers {repeated[:10]}")
    for column in TABLE_COLUMNS[filename]:
        if frame[column].isna().any():
            raise ValueError(f"{filename}: column {column} has empty cells")
    return frame.reset_index(drop=True)


def coding_strand(frame: pd.DataFrame, filename: str) -> pd.Series:
    """Which rows target the CODING strand of their gene.

    S1 Table releases the call as a ``coding`` column; S4 and S6 do not, because their
    library is already filtered to coding-strand guides. The column equals
    ``ori != gene_ori`` on every one of S1's 54,183 in-gene rows, so the same comparison
    is the rule for S4 and S6, where it must hold for every row.
    """
    derived = frame["ori"] != frame["gene_ori"]
    if filename == GROWTH_TABLE:
        released = frame["coding"] == True  # noqa: E712  # NaN on the intergenic rows
        in_gene = frame["gene"].notna()
        if not (derived[in_gene] == released[in_gene]).all():
            raise ValueError(
                f"{filename}: the released coding column disagrees with "
                "(ori != gene_ori) on an in-gene row"
            )
        return released & in_gene
    if not derived.all():
        raise ValueError(
            f"{filename}: the phage-screen library is filtered to coding-strand guides, "
            f"but {int((~derived).sum())} rows target the template strand"
        )
    return derived


# --------------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------------- #
#: What ``systematic_gene_name`` is on every record, as the release writes it.
IDENTIFIER_KIND = "gene symbol, as the released S4 and S6 gene column writes it"
#: The same for the growth screen's records (#905), whose symbol is S1 Table's.
GROWTH_IDENTIFIER_KIND = "gene symbol, as the released S1 gene column writes it"


class StoredGuide(BaseModel):
    """One kept (guide, gene) pair, identified as the release identifies it."""

    model_config = ConfigDict(frozen=True)

    spacer: str
    reported_symbol: str
    identifier_kind: str = IDENTIFIER_KIND


def genotype(stored: StoredGuide) -> Genotype:
    """One guide-bearing strain: CRISPRi of its released target gene by its spacer."""
    return Genotype(
        perturbations=[
            BacterialReleasedCrisprPerturbation(
                systematic_gene_name=stored.reported_symbol,
                perturbed_gene_name=stored.reported_symbol,
                gene_namespace=MG1655_NAMESPACE,
                strain_label=stored.spacer,
                identifier_kind=stored.identifier_kind,
                expression_direction="decreased",
                crispr=CrisprConstruct(
                    effector=EFFECTOR.value, guide_sequence=stored.spacer, n_guides=1
                ),
            )
        ]
    )


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the items it removed them for.

    ``served_by`` is the dataset that holds the dropped measurements instead, for a rule
    that drops a record because another release owns it. It is None for a rule that
    drops a record outright, so a reader can tell the two apart without reading the
    description.
    """

    rule: str
    scope: Literal["guide", "gene"]
    description: str
    n_records: int
    items: list[str] = []
    served_by: str | None = None


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]


class IdentifierReport(BaseModel):
    """What the records are keyed by, written to ``preprocess/``.

    No annotation is read: every released symbol of a stored row is a stored
    identifier, and what the pinned assembly makes of it is a verifier finding.
    """

    dataset: str
    released_symbols: int
    stored_symbols: int


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class CrispriScreenRousset2018Dataset(ExperimentDataset):
    """Rousset 2018 per-sgRNA log2FC: growth, three phage challenges, transduction."""

    organism = Organism.ecoli

    #: Both host strains (LC-E75, FR-E01) are MG1655 derivatives and every released
    #: identifier is an MG1655 gene symbol, so one assembly serves the whole dataset.
    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = REFERENCE_STRAIN_NAME

    def __init__(
        self,
        root: str = "data/torchcell/ecoli_crispri_rousset2018",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; the MG1655 genome is injected or opened in process."""
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
        """Rousset's three SI tables."""
        return list(raw_sha256())

    def download(self) -> None:
        """Link the mirror files into ``raw/`` after verifying their pinned sha256s.

        The mirror plus the pins is canonical; the PMC bucket URLs are retrieval
        metadata ``deposit_raw_mirror`` records, never a live build dependency. Every
        file is checked against the manifest and found on disk BEFORE ``raw/`` is
        created, so an incomplete mirror leaves no half-populated raw directory behind.
        """
        data_root = _data_root()
        mirror = raw_mirror_dir(data_root)
        manifest = load_manifest(data_root)
        for filename, sha256 in raw_sha256().items():
            relpath = table_rel(filename)
            check_manifest_pin(relpath, manifest_sha256(manifest, relpath), sha256)
            if not (mirror / relpath).exists():
                raise RuntimeError(
                    f"required raw artifact missing from mirror: {mirror / relpath}"
                )
        os.makedirs(self.raw_dir, exist_ok=True)
        for filename, sha256 in raw_sha256().items():
            link_verified(
                mirror / table_rel(filename), osp.join(self.raw_dir, filename), sha256
            )
        log.info(
            "Rousset 2018 S1/S4/S6 Tables linked into %s (sha256 verified)",
            self.raw_dir,
        )

    @post_process
    def process(self) -> None:
        """Parse the three tables into per-(guide, screen) records; write the LMDB."""
        verify_raw_files(self.raw_dir, raw_sha256())
        tables = {
            filename: read_table(osp.join(self.raw_dir, filename), filename)
            for filename in TABLE_SHA256
        }
        coding = {
            filename: coding_strand(frame, filename)
            for filename, frame in tables.items()
        }
        record_tables = {c.table for c in CONDITIONS}
        symbols = sorted(
            {
                str(name)
                for filename, frame in tables.items()
                if filename in record_tables
                for name in frame.loc[coding[filename], "gene"].dropna()
            }
        )

        environments = {c.screen_id: environment(c) for c in CONDITIONS}
        references = {
            c.screen_id: BacterialEnvironmentResponseExperimentReference(
                dataset_name=self.name,
                genome_reference=assembly_reference(
                    self.REFERENCE_STRAIN, background=BACKGROUNDS[c.strain]
                ),
                environment_reference=environments[c.screen_id],
                phenotype_reference=reference_phenotype(c),
            )
            for c in CONDITIONS
        }
        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for condition in tqdm(CONDITIONS, desc="rousset2018"):
                frame = tables[condition.table]
                kept = frame.loc[coding[condition.table]]
                condition_environment = environments[condition.screen_id]
                reference = references[condition.screen_id]
                for row in kept.itertuples(index=False):
                    symbol = str(row.gene)
                    experiment = BacterialEnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=genotype(
                            StoredGuide(
                                spacer=str(row.target),
                                reported_symbol=symbol,
                                identifier_kind=GROWTH_IDENTIFIER_KIND
                                if condition.table == GROWTH_TABLE
                                else IDENTIFIER_KIND,
                            )
                        ),
                        environment=condition_environment,
                        phenotype=phenotype(
                            float(getattr(row, condition.column)),
                            condition,
                            float(getattr(row, "padj"))
                            if condition.table == GROWTH_TABLE
                            else None,
                        ),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(experiment, reference, publication, itxn),
                    )
                    idx += 1
        env.close()
        interned_env.close()

        self._write_reports(tables, coding, symbols, idx)
        log.info(
            "Rousset2018: wrote %d records over %d released symbols and %d screens",
            idx,
            len(symbols),
            len(CONDITIONS),
        )

    def _write_reports(
        self,
        tables: Mapping[str, pd.DataFrame],
        coding: Mapping[str, pd.Series],
        symbols: Sequence[str],
        kept_records: int,
    ) -> None:
        """Write the drop accounting and the identifier report to ``preprocess/``.

        Every released cell is a source record: S1 Table's rows once (one screen) and
        each S4 / S6 column's rows once. No S1 row is attributed to another dataset
        (#905); what is not stored is counted under the two rules below.
        """
        growth = tables[GROWTH_TABLE]
        n_no_gene = int(growth["gene"].isna().sum())
        n_template = int((~coding[GROWTH_TABLE] & growth["gene"].notna()).sum())
        source_records = sum(len(tables[c.table]) for c in CONDITIONS)

        rules = [
            DropRule(
                rule="guide_targets_no_gene",
                scope="guide",
                description=(
                    "the released S1 Table row carries no gene (the guide targets an "
                    "intergenic position), so there is no target gene to write a "
                    "CRISPRi perturbation against; S4 and S6 Tables exclude these "
                    "upstream"
                ),
                n_records=n_no_gene,
            ),
            DropRule(
                rule="guide_targets_the_template_strand",
                scope="guide",
                description=(
                    "the S1 Table row's coding column is FALSE. dCas9 on the template "
                    "strand does not block elongation, which is why the paper's gene "
                    "scores use coding-strand guides only and the phage screens exclude "
                    "template-strand guides; BacterialCrisprInterferencePerturbation "
                    "fixes expression_direction to 'decreased' and has no slot for the "
                    "target strand, so storing one would assert a knockdown the paper's "
                    "own S2 Table says did not happen"
                ),
                n_records=n_template,
            ),
        ]
        drop_log = DropLog(
            dataset=self.name,
            source_records=source_records,
            kept_records=kept_records,
            dropped_records=source_records - kept_records,
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
        identifiers = IdentifierReport(
            dataset=self.name,
            released_symbols=len(symbols),
            stored_symbols=len(symbols),
        )
        with open(
            osp.join(self.preprocess_dir, "identifier_reconciliation.json"), "w"
        ) as handle:
            handle.write(identifiers.model_dump_json(indent=2))

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built tree
# --------------------------------------------------------------------------- #
#: The frozen per-screen record-count oracle, from the drop accounting of the release:
#: every coding-strand in-gene row of S1 Table (#905) and of the phage library, each
#: under its released gene symbol.
EXPECTED_SCREEN_CENSUS: dict[str, int] = {
    GROWTH_SCREEN_ID: 23372,
    "phage_lambda": 17220,
    "phage_T4": 17220,
    "phage_186cIts": 17220,
    "lambda_transduction": 17220,
}

#: The frozen record-count oracle: the five stored screens together.
EXPECTED_RECORDS = sum(EXPECTED_SCREEN_CENSUS.values())

#: The frozen gene-set oracle: distinct RELEASED gene symbols across the kept records
#: (3,944 in S1 Table, a superset of the phage library's 3,708).
EXPECTED_GENES = 3944


def screen_census(records: Iterable[Mapping[str, Any]]) -> LevelResult:
    """SUPPLEMENTARY L1: the record count of each of the five screens.

    The shared ``count`` row checks the dataset total, which one screen's rows could
    cover for another's. This row pins the per-``screen_id`` split, so a column read from
    the wrong table shows up as a changed census rather than as the same total. It takes
    an ITERABLE so it can run over a second streaming pass rather than a materialized
    list.
    """
    census = Counter(
        str(record["experiment"]["phenotype"]["screen_id"]) for record in records
    )
    expected = EXPECTED_SCREEN_CENSUS
    passed = dict(census) == expected
    return LevelResult(
        level=Level.L1,
        name="screen_census",
        passed=passed,
        message=(
            f"SUPPLEMENTARY: {len(census)} screens, census {dict(census)}"
            if passed
            else f"SUPPLEMENTARY: census {dict(census)} is not {expected}"
        ),
        details={"census": dict(census), "expected": expected},
    )


#: The released-test fields of ``EnvironmentResponsePhenotype`` (#863).
RELEASED_TEST_FIELDS: tuple[str, ...] = (
    "environment_response_p_value",
    "environment_response_p_value_adjusted",
    "p_value_adjustment_method",
)


def released_test_where_released(records: Iterable[Mapping[str, Any]]) -> LevelResult:
    """SUPPLEMENTARY L2: a p-value sits exactly where the release put one (#905).

    S1 Table releases ``padj`` and no unadjusted p-value, so every growth-screen record
    carries an adjusted p-value with its correction and no unadjusted one; S4 and S6
    Tables release log2FC only, so no phage-derived record and no reference carries any
    of ``RELEASED_TEST_FIELDS``. Counts every departure from that pattern.
    """
    wrong: Counter[str] = Counter()
    n_growth = 0
    n_records = 0
    for record in records:
        n_records += 1
        phenotype = record["experiment"]["phenotype"]
        growth = phenotype["screen_id"] == GROWTH_SCREEN_ID
        n_growth += int(growth)
        expected = {
            "environment_response_p_value": False,
            "environment_response_p_value_adjusted": growth,
            "p_value_adjustment_method": growth,
        }
        for name, present in expected.items():
            if (phenotype[name] is not None) != present:
                wrong[f"experiment.{name}"] += 1
        if growth and phenotype["p_value_adjustment_method"] != str(
            PADJ_CORRECTION.value
        ):
            wrong["experiment.p_value_adjustment_method_value"] += 1
        reference = record["reference"]["phenotype_reference"]
        for name in RELEASED_TEST_FIELDS:
            if reference[name] is not None:
                wrong[f"reference.{name}"] += 1
    passed = not wrong
    return LevelResult(
        level=Level.L2,
        name="released_test_where_released",
        passed=passed,
        message=(
            f"SUPPLEMENTARY: {n_growth} of {n_records} records carry S1 Table's padj "
            f"({PADJ_CORRECTION.value}); the rest carry no test, as released"
            if passed
            else f"SUPPLEMENTARY: released-test pattern broken {dict(wrong)}"
        ),
        details={"n_records": n_records, "n_growth": n_growth, "wrong": dict(wrong)},
    )


def verify_build(
    dataset_root: str,
    *,
    genome: EcoliK12MG1655Genome | None = None,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the environment-response L0-L4 verifier on a built tree and write its report.

    The LMDB is STREAMED three times: once for the verifier and once for each
    SUPPLEMENTARY row. The 92,252 records are never materialized, which is the choice
    Price 2018 and Borchert 2024 make for the two other large bacterial stores (an eager
    ``load_records`` of this store takes tens of minutes; three streaming passes take a
    fraction of that).

    Every record is checked against the MG1655 genome its references pin: the resolver of
    the canonical-name rule, and as the L4 universe every GenBank locus of the assembly
    (4,651, pseudogenes and RNA tags included; the same set as the verification runners'
    ``_ecoli_k12_gene_set``). Two SUPPLEMENTARY rows are appended: the per-screen census
    and ``released_test_where_released``, each over its own streaming pass.
    The report is written to ``preprocess/verification_report.json``.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records

    if genome is None:
        opened = bacterial_genome("ecoli", REFERENCE_STRAIN_NAME, data_root)
        if not isinstance(opened, EcoliK12MG1655Genome):
            raise TypeError(f"expected the MG1655 genome, got {type(opened).__name__}")
        genome = opened
    report = verify_environment_response_dataset_streaming(
        stream_records(dataset_root),
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/data/",
            citation_key=CITATION_KEY,
            sha256=TABLE_SHA256[PHAGE_TABLE],
            method=(
                "S1, S4 and S6 Tables (PMC Article Datasets "
                f"{PMCID}.1): one BacterialEnvironmentResponseExperiment per (sgRNA, "
                "screen), DESeq2 log2FoldChange of guide abundance normalized on a "
                "non-targeting control guide and paired against each sample's initial "
                "condition; coding-strand in-gene guides only, each keyed by the gene "
                "SYMBOL the release names (no annotation read). The S1 Table growth "
                "screen (LC-E75) is stored with its released padj, adjusted by "
                "Benjamini-Hochberg per DESeq2 (ref. 32, deposited)"
            ),
            page=(
                f"{TABLE_LABEL[GROWTH_TABLE]} ({GROWTH_TABLE}), "
                f"{TABLE_LABEL[PHAGE_TABLE]} ({PHAGE_TABLE}), "
                f"{TABLE_LABEL[TRANSDUCTION_TABLE]} ({TRANSDUCTION_TABLE})"
            ),
            retrieved=RAW_RETRIEVED_AT,
        ),
        expected_count=expected_count,
        resolve_gene_name=genome.resolve_gene_name,
        sgd_genes=set(genome.genbank.loci),
    )
    report.add(screen_census(stream_records(dataset_root)))
    report.add(released_test_where_released(stream_records(dataset_root)))
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build the dataset and verify it, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    root = osp.join(data_root, "data/torchcell/ecoli_crispri_rousset2018")
    dataset = CrispriScreenRousset2018Dataset(root=root)
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
    print(verify_build(root, data_root=data_root).summary())


if __name__ == "__main__":
    main()
