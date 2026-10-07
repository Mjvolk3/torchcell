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

THIS DATASET SERVES THE FOUR PHAGE-DERIVED SCREENS. The growth screen is Cui et al.
2018's screen, released again; it is served by ``CrispriKnockdownCui2018Dataset`` and is
accounted for in this build's retention ledger rather than stored a second time. The
measurement behind that, and the reason the non-overlapping remainder is not kept either,
is below under THE GROWTH SCREEN IS CUI 2018'S SCREEN.

RECORD = one (sgRNA x screen) ``BacterialEnvironmentResponseExperiment``:

- GENOTYPE: one ``BacterialCrisprInterferencePerturbation`` keyed by the target gene's
  MG1655 b-number, carrying the 20-nt spacer on the shared ``crispr`` construct
  (``effector="dCas9"``). The release names genes by SYMBOL, so every symbol is resolved
  to a current locus through ``reconcile_locus_tags``; the symbol it came from is kept on
  ``identifier_mapping`` (``route="gene_symbol"``), never dropped.
- ENVIRONMENT: the screen's own culture, which is one medium for all four screens: LB
  plus 1 microM aTc, 0.2% maltose and 5 mM CaCl2 at 37 C for 2 h, each record carrying
  exactly one ``PhagePerturbation`` at ``multiplicity_of_infection=1``. The aTc that
  induces dCas9 is a COMPONENT of that medium, not an ``Environment.perturbation``: the
  paper puts it there ("diluted 100-fold in LB containing 1 microM aTc, 0.2% Maltose and
  5 mM CaCl2" lists it beside the two components the medium already carries), and it is
  constant across the dataset rather than the varied condition. That leaves the phage as
  the only environment perturbation any record carries, which is what lets the adapter
  conf enable ``phage perturbation`` and not ``environment perturbation``: the served
  ``_environment_perturbation_node`` does not filter phages out, so a conf enabling both
  would emit each phage twice under two labels on one id.
- PHENOTYPE: ``EnvironmentResponsePhenotype`` with ``measurement_type=log2_ratio``. The
  reference carries 0.0, which is what no change in guide abundance is.

WHY ``EnvironmentResponsePhenotype`` AND NOT ``FitnessPhenotype``. The released value is
a signed DESeq2 ``log2FoldChange`` of guide abundance ("The log2FoldChange ... value
represents the enrichment or depletion of each sgRNA"), normalized on a non-targeting
control guide and paired against each sample's own initial condition. It is routinely
negative. Measured over the 68,436 STORED records: 89.15% of the transduction screen,
83.93% of T4, 44.83% of 186cIts and 27.70% of lambda. ``FitnessPhenotype.validate_fitness``
clamps every non-positive value to 0.0, which would erase that signal, and no
``MeasurementType`` member other than ``log2_ratio`` describes this number.

WHY ``PhagePerturbation``. Every stored screen doses a virion: "followed by infection
with phage lambda, T4 or 186cIts at a multiplicity of infection (MOI) of 1". That is the
typed leaf for a bacteriophage on the environment axis, and the dose is an MOI rather
than a ``Concentration``.

THE HOST STRAIN. FR-E01, an MG1655 derivative written against the MG1655 assembly, so it
is a ``BacterialStrainBackground`` on the reference rather than a namespace: it carries
the same optimized dcas9 cassette as LC-E75 moved to the HK022 attB site, built so that
phage 186 could be studied without interference.

THE GROWTH SCREEN IS CUI 2018'S SCREEN, AND IS NOT STORED HERE. Rousset says so outright:
"The data for the screen performed with strain LC-E75 grown in rich medium was obtained
from our previous study [26]", and [26] is Cui et al. 2018 (Nat Commun 9:1912,
doi:10.1038/s41467-018-04209-5), mirrored as ``cuiCRISPRiScreenColi2018`` and served by
``CrispriKnockdownCui2018Dataset``. Measured on the two pinned tables, joined on the
20-nt spacer (Rousset's ``target``, Cui's ``guide``):

- Rousset S1 ``log2FC`` against Cui ``fit75``: 54,326 spacers in common, median absolute
  difference 0.0000, maximum 0.0066, Pearson r 1.0000. The same measurements to released
  precision; the 0.0066 is rounding, not re-analysis.
- Against Cui ``fit18``, the other dose regime: r 0.8018, median absolute difference
  0.4151. A different screen, and Cui's alone.
- Cui's table carries 78,137 distinct guides, 23,811 of which Rousset's S1 Table does not
  carry at all. Cui is therefore the primary release: the study Rousset cites, covering
  both dose regimes and more of the library.

Neither release is a subset of the other: 4,920 of Rousset's 59,246 released spacers are
absent from Cui's table. Those are NOT kept as a remainder, because measured on the
pinned bytes they are the library's low-abundance tail rather than a screen:

- 6.6% of their coding-strand members appear in the 17,220-guide phage library, against
  78.9% of the shared coding-strand guides. That library's gate is Rousset's own
  ``BaseMean < 10`` exclusion on this same library, so 93.4% of them fail an abundance
  floor the paper itself applied.
- At matched effect size they carry no statistical power: in the 0.25 to 0.5 ``|log2FC|``
  band 0 of 1,306 reach ``padj < 0.05``, against 29.8% of 16,452 shared guides; in the
  0.5 to 1.0 band 0.2% against 65.8%. Their median ``|log2FC|`` (0.380) matches the
  shared set's (0.377), so the deficit is dispersion, not effect size.
- They are not a region Cui excluded: the median distance from one to the nearest Cui
  guide is 24 bp (90th percentile 82 bp, maximum 396 bp), so they are interspersed
  through the chromosome.
- The storable remainder would be 1,687 records over 1,284 genes, a median of 1 guide per
  gene against 5 for the shared set, and the paper's own reading rule is that "the
  effects of genes on a given phenotype should ideally not be inferred from the effect of
  a single guide". Adding it to the shared set flips 10 genes' essentiality calls under
  the paper's own ``median log2FC < -2`` rule, moving one gene median by 2.922.

Hypothesis (untested, and not testable from either release because neither publishes read
counts): the 4,920 fall below Cui's 20-read floor because Cui's table reports both dose
arms and the LC-E18 arm was sequenced 2.3-fold shallower, while Rousset's growth analysis
applied the same floor to the LC-E75 arm alone.

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
3. ``growth_screen_measurement_is_served_by_cui2018`` (21,685 guides): the remaining S1
   rows whose spacer Cui's table carries. The rule names the dataset that serves them
   (``served_by``), so the drop is attributable rather than a disappearance.
4. ``growth_screen_guide_is_below_cui2018_read_floor`` (1,687 guides): the remaining S1
   rows whose spacer Cui's table does not carry, the low-abundance tail measured above.
5. ``gene_symbol_is_not_in_the_mg1655_annotation`` (S4/S6; RETIRED against
   GCA_000005845.2, the insertion-sequence and cryptic-prophage names ``insA-7``,
   ``lomR_1``, ``C0299``, ``G0-10699``, ...).
6. ``gene_symbol_collides_with_another_symbol_on_one_mg1655_locus``: two released symbols
   resolve to ONE current locus (the annotation merges the pseudogene fragments
   ``yaiT``/``yaiU``, ``ydeK``/``ydeU``, ...). ``reconcile_locus_tags`` keeps both as
   given so the records stay distinct, and a bare symbol is not a b-number, so neither
   member can be stored. The group is dropped whole: unlike Tong 2020's b-number case, no
   member IS the locus tag.
7. ``gene_symbol_is_ambiguous_in_mg1655``.

Retention arithmetic: 59,246 + 3 x 17,220 + 17,220 = 128,126 released (guide, screen)
cells. The S1 Table's 59,246 are accounted for whole by rules 1 to 4
(5,063 + 30,811 + 21,685 + 1,687), and rules 5 to 7 remove 444 of the 68,880 phage-derived
cells, leaving 68,436 records over 3,671 genes. The candidate table's 236,000 is 59,000
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
- The two Cui 2018 values the de-duplication rules cite: its 20-read floor
  (``CUI_READ_FLOOR``) and its two arms' sequencing depth (``CUI_SCREEN_DEPTH``).

DATA SOURCE: S1, S4 and S6 Tables (``pgen.1007749.s011.csv``, ``.s014.csv``,
``.s016.csv``) from the PMC Article Datasets bucket (``pmc_cloud``, prefix
``PMC6242692.1``), deposited in
``$DATA_ROOT/torchcell-raw/roussetGenomewideCRISPRdCas9Screens2018/`` with a
``manifest.json``. S1 Table is still read and sha256-verified although no record comes
from it: the de-duplication ledger is derived from its bytes rather than asserted. The
build also reads ONE column of Cui 2018's own mirrored screen table
(``41467_2018_4209_MOESM8_ESM.csv``, verified against Cui's manifest), because the
de-duplication rule is defined by Cui's released guide set. S2, S5 and S7 Tables are the
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
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    AssayType,
    BacterialCrisprInterferencePerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialStrainBackground,
    ComponentDefinition,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    DerivedIdentifierMapping,
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
from torchcell.literature.retrieve import pmc_cloud_url
from torchcell.sequence.genome.bacterial import BacterialGenome
from torchcell.sequence.genome.ecoli.k12 import (
    EcoliK12Genome,
    EcoliK12MG1655Genome,
    EcoliK12StrainName,
)
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

#: Cui 2018's released screen table (Supplementary Data 5, MOESM8), pinned by the same
#: sha256 its own loader pins, read here for ONE column: the 20-nt ``guide`` spacer. The
#: growth-screen de-duplication rule is DEFINED by that guide set, so it is read from the
#: bytes rather than asserted as a count.
CUI2018_SCREEN_FILENAME = "41467_2018_4209_MOESM8_ESM.csv"
CUI2018_SCREEN_SHA256 = (
    "95ebaa5a0c92c63849617f48889e2d28b7805fdffdd960527f8a501381143c1e"
)
CUI2018_RAW_DIR_REL = f"torchcell-raw/{CUI2018_KEY}"
CUI2018_SCREEN_HEADER: tuple[str, ...] = (
    "guide",
    "gene",
    "essential",
    "pos",
    "ori",
    "coding",
    "fit18",
    "fit75",
    "ntargets",
    "seq",
)

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

    Rousset's three tables, plus Cui 2018's screen table, whose guide set the
    growth-screen de-duplication rule is defined by. The Cui file is read from ITS OWN
    mirror and is never deposited into Rousset's.
    """
    return {**TABLE_SHA256, CUI2018_SCREEN_FILENAME: CUI2018_SCREEN_SHA256}


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
MIN_RESOLVED_FRACTION = 0.95

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
    note="the growth screen's released library; S1 Table carries 59,246 rows, which "
    "this dataset accounts for in its retention ledger and does not store",
)
LC_E75_CASSETTE = _paper(
    "LC-E75",
    "This strain expresses an optimized dcas9 cassette under the control of an "
    "aTc-inducible pTet promoter integrated at the phage 186 attB site.",
    page=_STRAINS,
    note="the host of the SUBSUMED growth screen: MG1655 plus a chromosomal "
    "aTc-inducible dcas9 cassette at the phage 186 attB site. No record here carries "
    "this background; it identifies the screen the ledger attributes to Cui 2018",
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
    note="reference [26] is Cui 2018 (mirrored), so the growth screen's released "
    "log2FC values ARE Cui's fit75 column rather than a re-analysis of it. Measured on "
    "the two pinned tables, joined on the 20-nt spacer: 54,326 spacers in common, "
    "median absolute difference 0.0000, maximum 0.0066, Pearson r 1.0000 against "
    "fit75, against r 0.8018 and a median absolute difference of 0.4151 for Cui's "
    "other screen (fit18). This dataset therefore stores no growth-screen record",
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
RAW_READS_ACCESSION = _paper(
    ENA_BIOPROJECT,
    "Raw sequencing files are available on the European Nucleotide Archive "
    "(https://www.ebi.ac. uk/ena) with the accession number PRJEB28256.",
    page=_ANALYSIS,
)

CUI_READ_FLOOR = _cui(
    20,
    "Guides with a total number of reads across samples ${ < } 2 0$ were discarded "
    "from the analysis.",
    page=_CUI_FOLD_CHANGE,
    note="the floor Cui's released table is filtered on, and the reason 4,920 of "
    "Rousset's 59,246 released spacers do not appear in it at all. Rousset's own growth "
    "analysis states the same count ('Guides with fewer than 20 reads in total were "
    "discarded') over the LC-E75 arm alone, while Cui's table reports both arms",
)
CUI_SCREEN_DEPTH = _cui(
    (7.5e6, 17.0e6),
    "we obtained on average 7.5 million and 17 million reads per experimental condition "
    "for LC-E18 and LC-E75, respectively.",
    page=_CUI_SEQUENCING,
    note="the two arms Cui's table reports are sequenced 2.3-fold apart. Hypothesis "
    "(untested, and not testable from either release because neither publishes counts): "
    "the shallower LC-E18 arm is what put the 4,920 spacers under the 20-read floor. "
    "What IS measured is that those spacers are the library's low-abundance tail",
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
    CUI_READ_FLOOR,
    CUI_SCREEN_DEPTH,
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
    there: "diluted 100-fold in LB containing 1 microM aTc, 0.2% Maltose and 5 mM CaCl2"
    lists aTc beside the two components this medium already carries. It is also constant
    across the dataset rather than the varied condition, so the environment axis would
    hold a factor nothing in the dataset contrasts. The dose is stored rather than noted
    because it is what switches the knockdown on, and because it is three orders of
    magnitude above the 1 nM Cui 2018 used on the same cassette.
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
#: Only FR-E01 appears: the LC-E75 growth screen is Cui 2018's and is not stored. The
#: label stays a ``Literal`` so a condition cannot name a strain with no background.
StrainLabel = Literal["FR-E01"]

BACKGROUNDS: dict[StrainLabel, BacterialStrainBackground] = {
    "FR-E01": FR_E01_BACKGROUND
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
    phage: str
    assay_type: AssayType
    units: str


CONDITIONS: tuple[ScreenCondition, ...] = (
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

#: The released column each condition reads, keyed by table. S1 Table is still read and
#: validated, for the retention ledger that attributes it to Cui 2018, so its one
#: measured column is listed although no condition reads it.
TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    GROWTH_TABLE: ("log2FC",),
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
    """The culture one screen was run in: the one phage-screen medium, plus its phage."""
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
        "not a released column. S1 Table also releases padj (the adjusted p-value of "
        "the DESeq2 test, for which the schema has no slot) and gamma (named in the S1 "
        "Table caption and defined in neither mirrored paper); neither is an "
        "uncertainty, so neither is stored",
    )


def phenotype(value: float, condition: ScreenCondition) -> EnvironmentResponsePhenotype:
    """One released log2FC cell."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=condition.assay_type,
        environment_response=value,
        n_samples=PHAGE_SCREEN_TRIPLICATE.value,
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


def cui2018_spacers(path: str | Path) -> frozenset[str]:
    """Every 20-nt spacer Cui 2018 released a fitness value for.

    The de-duplication rule is defined by this set, so it is read from Cui's own pinned
    bytes rather than pinned as a count here. One row of that table is a (guide, target
    position) pair, so the guide column repeats for a multi-target guide; the set is what
    the rule needs.
    """
    frame = pd.read_csv(path, usecols=["guide"])
    if list(frame.columns) != ["guide"]:
        raise ValueError(f"{path}: expected a guide column, got {list(frame.columns)}")
    spacers = frame["guide"].astype(str)
    if not (spacers.str.len() == 20).all():
        raise ValueError(f"{path}: every Cui 2018 guide must be 20 nt")
    return frozenset(spacers)


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
class StoredGuide(BaseModel):
    """One kept (guide, gene) pair with the identity its records are stored under."""

    model_config = ConfigDict(frozen=True)

    spacer: str
    reported_symbol: str
    systematic_gene_name: str
    perturbed_gene_name: str


class SymbolResolution(BaseModel):
    """The one symbol-to-b-number map the whole dataset is stored under.

    ONE reconciliation over the union of every symbol the three tables name, so a gene
    has a single identity across the five screens. Reconciling per table would let a
    symbol that collides in one table and not in another be stored two different ways.
    """

    stored: dict[str, str]
    retired: tuple[str, ...]
    collided: tuple[str, ...]
    ambiguous: tuple[str, ...]
    report: LocusTagReconciliation

    @property
    def unstorable(self) -> frozenset[str]:
        """Symbols that cannot be written as a b-number, so their records are dropped."""
        return (
            frozenset(self.retired)
            | frozenset(self.collided)
            | frozenset(self.ambiguous)
        )


def genotype(stored: StoredGuide) -> Genotype:
    """One guide-bearing strain: CRISPRi of its target gene by its 20-nt spacer."""
    return Genotype(
        perturbations=[
            BacterialCrisprInterferencePerturbation(
                systematic_gene_name=stored.systematic_gene_name,
                perturbed_gene_name=stored.perturbed_gene_name,
                gene_namespace=MG1655_NAMESPACE,
                crispr=CrisprConstruct(
                    effector=EFFECTOR.value, guide_sequence=stored.spacer, n_guides=1
                ),
                identifier_mapping=DerivedIdentifierMapping(
                    source_identifier=stored.reported_symbol, route="gene_symbol"
                ),
            )
        ]
    )


def canonical_symbol(genome: BacterialGenome[Any], tag: str) -> str:
    """The genome's own gene symbol for ``tag`` when it resolves back to ``tag``.

    One spelling per locus is what keeps a perturbation from splitting into two graph
    nodes, so the symbol comes from the pinned annotation rather than from the release's
    ``gene`` column. A locus with no symbol, or whose symbol resolves elsewhere, is named
    by its tag.
    """
    symbol = genome.genbank.loci[tag].symbol
    if symbol is None:
        return tag
    resolved = genome.resolve_gene_name(symbol).systematic_name
    return symbol if resolved == tag else tag


def resolve_symbols(
    symbols: Sequence[str], genome: EcoliK12MG1655Genome, *, label: str
) -> SymbolResolution:
    """Map every released gene symbol to one MG1655 b-number.

    ``reconcile_locus_tags`` keeps a retired, colliding or ambiguous name as given; a
    bare gene symbol is not a b-number, so those names have no storable identity and are
    reported here for the drop accounting. Every other symbol is remapped to its locus,
    and the remapped tags must be distinct.
    """
    names = pd.Series(sorted(set(symbols)), dtype=str)
    stored, report = reconcile_locus_tags(genome, names, label=label)
    report.require_resolved(MIN_RESOLVED_FRACTION)
    mapping = dict(zip(names, stored, strict=True))
    unstorable = frozenset(report.outside_namespace)
    resolution = SymbolResolution(
        stored={k: v for k, v in mapping.items() if v not in unstorable},
        retired=report.retired_kept,
        collided=report.kept_on_collision,
        ambiguous=tuple(sorted(report.ambiguous_kept)),
        report=report,
    )
    accounted = resolution.unstorable
    unmapped = {k for k, v in mapping.items() if v in unstorable}
    if accounted != unmapped:
        raise RuntimeError(
            f"{label}: symbols with no storable b-number {sorted(unmapped)} are not the "
            f"retired/colliding/ambiguous set {sorted(accounted)}"
        )
    tags = sorted(resolution.stored.values())
    if len(tags) != len(set(tags)):
        raise RuntimeError(f"{label}: two symbols stored under one b-number")
    return resolution


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
    """What the identifier step did, written to ``preprocess/``."""

    dataset: str
    released_symbols: int
    stored_symbols: int
    stored_b_numbers: int
    retired: list[str]
    collided: list[str]
    ambiguous: dict[str, list[str]]
    reconciliation: LocusTagReconciliation


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class CrispriScreenRousset2018Dataset(ExperimentDataset):
    """Rousset 2018 per-sgRNA log2FC: growth, three phage challenges, transduction."""

    #: Both host strains (LC-E75, FR-E01) are MG1655 derivatives and every released
    #: identifier is an MG1655 gene symbol, so one assembly serves the whole dataset.
    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = REFERENCE_STRAIN_NAME

    def __init__(
        self,
        root: str = "data/torchcell/ecoli_crispri_rousset2018",
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; the MG1655 genome is injected or opened in process."""
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
        """Rousset's three SI tables, plus Cui 2018's screen table."""
        return list(raw_sha256())

    def download(self) -> None:
        """Link the mirror files into ``raw/`` after verifying their pinned sha256s.

        The mirror plus the pins is canonical; the PMC bucket URLs are retrieval
        metadata ``deposit_raw_mirror`` records, never a live build dependency. Every
        file is checked against the manifest and found on disk BEFORE ``raw/`` is
        created, so an incomplete mirror leaves no half-populated raw directory behind.
        """
        data_root = _data_root()
        sources = {
            **{
                filename: (raw_mirror_dir(data_root), CITATION_KEY)
                for filename in TABLE_SHA256
            },
            CUI2018_SCREEN_FILENAME: (
                Path(data_root) / CUI2018_RAW_DIR_REL,
                CUI2018_KEY,
            ),
        }
        manifests = {
            key: load_manifest_of(key, data_root) for key in (CITATION_KEY, CUI2018_KEY)
        }
        for filename, sha256 in raw_sha256().items():
            mirror, key = sources[filename]
            relpath = table_rel(filename)
            check_manifest_pin(
                relpath, manifest_sha256(manifests[key], relpath), sha256
            )
            if not (mirror / relpath).exists():
                raise RuntimeError(
                    f"required raw artifact missing from mirror: {mirror / relpath}"
                )
        os.makedirs(self.raw_dir, exist_ok=True)
        for filename, sha256 in raw_sha256().items():
            mirror, _ = sources[filename]
            link_verified(
                mirror / table_rel(filename), osp.join(self.raw_dir, filename), sha256
            )
        log.info(
            "Rousset 2018 S1/S4/S6 Tables and the Cui 2018 screen table linked into %s "
            "(sha256 verified)",
            self.raw_dir,
        )

    def _genome(self) -> EcoliK12MG1655Genome:
        """The MG1655 genome: injected by the build entry points, or opened here."""
        if self.ecoli_genome is None:  # a direct run; the build entry points inject it
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        genome = self.ecoli_genome
        if not isinstance(genome, EcoliK12MG1655Genome):
            raise TypeError(
                f"{type(self).__name__} needs the MG1655 genome, got "
                f"{type(genome).__name__}"
            )
        return genome

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
        genome = self._genome()
        record_tables = {c.table for c in CONDITIONS}
        symbols = sorted(
            {
                str(name)
                for filename, frame in tables.items()
                if filename in record_tables
                for name in frame.loc[coding[filename], "gene"].dropna()
            }
        )
        resolution = resolve_symbols(symbols, genome, label=f"{self.name} gene symbols")
        canonical = {
            tag: canonical_symbol(genome, tag) for tag in resolution.stored.values()
        }

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
        dropped_by_symbol: Counter[str] = Counter()
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for condition in tqdm(CONDITIONS, desc="rousset2018"):
                frame = tables[condition.table]
                kept = frame.loc[coding[condition.table]]
                condition_environment = environments[condition.screen_id]
                reference = references[condition.screen_id]
                for row in kept.itertuples(index=False):
                    symbol = str(row.gene)
                    tag = resolution.stored.get(symbol)
                    if tag is None:
                        dropped_by_symbol[symbol] += 1
                        continue
                    experiment = BacterialEnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=genotype(
                            StoredGuide(
                                spacer=str(row.target),
                                reported_symbol=symbol,
                                systematic_gene_name=tag,
                                perturbed_gene_name=canonical[tag],
                            )
                        ),
                        environment=condition_environment,
                        phenotype=phenotype(
                            float(getattr(row, condition.column)), condition
                        ),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(experiment, reference, publication, itxn),
                    )
                    idx += 1
        env.close()
        interned_env.close()

        self._write_reports(
            tables,
            coding,
            resolution,
            dropped_by_symbol,
            idx,
            cui2018_spacers(osp.join(self.raw_dir, CUI2018_SCREEN_FILENAME)),
        )
        log.info(
            "Rousset2018: wrote %d records over %d genes and %d screens",
            idx,
            len(resolution.stored),
            len(CONDITIONS),
        )

    def _write_reports(
        self,
        tables: Mapping[str, pd.DataFrame],
        coding: Mapping[str, pd.Series],
        resolution: SymbolResolution,
        dropped_by_symbol: Mapping[str, int],
        kept_records: int,
        cui_spacers: frozenset[str],
    ) -> None:
        """Write the drop accounting and the identifier report to ``preprocess/``."""
        growth = tables[GROWTH_TABLE]
        n_no_gene = int(growth["gene"].isna().sum())
        n_template = int((~coding[GROWTH_TABLE] & growth["gene"].notna()).sum())
        storable = growth.loc[coding[GROWTH_TABLE], "target"].astype(str)
        in_cui = storable.isin(cui_spacers)
        n_served_by_cui = int(in_cui.sum())
        n_below_read_floor = int((~in_cui).sum())
        source_records = len(growth) + sum(len(tables[c.table]) for c in CONDITIONS)

        def by_rule(items: Sequence[str]) -> int:
            return sum(dropped_by_symbol.get(symbol, 0) for symbol in items)

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
            DropRule(
                rule="growth_screen_measurement_is_served_by_cui2018",
                scope="guide",
                description=(
                    "the S1 Table row's 20-nt spacer is in Cui 2018's released screen "
                    "table, whose fit75 column IS this value. Rousset states the "
                    "deferral outright ('The data for the screen performed with strain "
                    "LC-E75 grown in rich medium was obtained from our previous study "
                    "[26]'), and the two releases agree on the 54,326 spacers they "
                    "share to a median absolute difference of 0.0000 (maximum 0.0066, "
                    "Pearson r 1.0000); against Cui's other screen, fit18, the same "
                    "join gives r 0.8018 and a median absolute difference of 0.4151. "
                    "Cui 2018 is the primary release: it is the study Rousset cites, it "
                    "covers both dCas9 dose regimes, and it carries 23,811 spacers "
                    "Rousset's table does not. Storing these rows here would store one "
                    "measurement twice under two dataset names"
                ),
                n_records=n_served_by_cui,
                served_by=f"{CUI2018_DATASET_CLASS} ({CUI2018_DATASET}), "
                f"screen_id {CUI2018_SCREEN_ID}",
            ),
            DropRule(
                rule="growth_screen_guide_is_below_cui2018_read_floor",
                scope="guide",
                description=(
                    "the S1 Table row's spacer is absent from Cui 2018's table "
                    "entirely, so no other dataset serves it, and it is NOT stored "
                    "either: these guides are the library's low-abundance tail rather "
                    "than a screen. Measured on the pinned bytes, over the 4,920 "
                    "released spacers Cui's table does not carry: 93.4% of their "
                    "coding-strand members fail the BaseMean >= 10 floor Rousset's own "
                    "phage analysis applied to this same library (6.6% appear in the "
                    "17,220-guide phage library, against 78.9% of the shared "
                    "coding-strand guides), and at matched effect size they carry no "
                    "statistical power (0 of 1,306 reach padj < 0.05 in the 0.25 to 0.5 "
                    "|log2FC| band, against 29.8% of 16,452 shared guides; 0.2% against "
                    "65.8% in the 0.5 to 1.0 band). The storable remainder is 1 guide "
                    "per gene at the median against 5 for the shared set, and the "
                    "paper's own reading rule is that a gene's phenotype should not be "
                    "inferred from one guide. Keeping it would store the screen's noise "
                    "floor under the name of a genome-wide screen"
                ),
                n_records=n_below_read_floor,
            ),
            DropRule(
                rule="gene_symbol_is_not_in_the_mg1655_annotation",
                scope="gene",
                description=(
                    "the released gene symbol is no locus tag, symbol or synonym of "
                    "GCA_000005845.2 (RETIRED), so no b-number can be written for it"
                ),
                n_records=by_rule(resolution.retired),
                items=list(resolution.retired),
            ),
            DropRule(
                rule="gene_symbol_collides_with_another_symbol_on_one_mg1655_locus",
                scope="gene",
                description=(
                    "two released symbols resolve to ONE current MG1655 locus (the "
                    "annotation merges pseudogene fragments such as yaiT/yaiU). "
                    "reconcile_locus_tags keeps both as given so the records stay "
                    "distinct, and a bare symbol is not a b-number, so neither member "
                    "has a storable identity; unlike a b-number collision, no member IS "
                    "the locus tag, so the group is dropped whole"
                ),
                n_records=by_rule(resolution.collided),
                items=list(resolution.collided),
            ),
            DropRule(
                rule="gene_symbol_is_ambiguous_in_mg1655",
                scope="gene",
                description=(
                    "the released gene symbol matches more than one MG1655 locus"
                ),
                n_records=by_rule(resolution.ambiguous),
                items=list(resolution.ambiguous),
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
            released_symbols=resolution.report.unique_names,
            stored_symbols=len(resolution.stored),
            stored_b_numbers=len(set(resolution.stored.values())),
            retired=list(resolution.retired),
            collided=list(resolution.collided),
            ambiguous={
                name: list(candidates)
                for name, candidates in resolution.report.ambiguous_kept.items()
            },
            reconciliation=resolution.report,
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
#: The frozen per-screen record-count oracle, from the drop accounting of the release.
#: The growth screen is absent: it is Cui 2018's screen and is accounted for in the
#: retention ledger rather than stored.
EXPECTED_SCREEN_CENSUS: dict[str, int] = {
    "phage_lambda": 17109,
    "phage_T4": 17109,
    "phage_186cIts": 17109,
    "lambda_transduction": 17109,
}

#: The frozen record-count oracle: the four phage-derived screens together.
EXPECTED_RECORDS = sum(EXPECTED_SCREEN_CENSUS.values())

#: The frozen gene-set oracle: distinct MG1655 b-numbers across the kept records.
EXPECTED_GENES = 3671


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


def verify_build(
    dataset_root: str,
    *,
    genome: EcoliK12MG1655Genome | None = None,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the environment-response L0-L4 verifier on a built tree and write its report.

    The LMDB is STREAMED, twice: once for the verifier and once for the census row. The
    68,436 records are never materialized, which is the choice Price 2018 and Borchert
    2024 make for the two other large bacterial stores (an eager ``load_records`` of this
    store takes tens of minutes; two streaming passes take a fraction of that).

    Every record is checked against the MG1655 genome its references pin: the resolver of
    the canonical-name rule, and as the L4 universe every GenBank locus of the assembly
    (4,651, pseudogenes and RNA tags included; the same set as the verification runners'
    ``_ecoli_k12_gene_set``). One SUPPLEMENTARY row, the per-screen census, is appended.
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
                "S4 and S6 Tables (PMC Article Datasets "
                f"{PMCID}.1): one BacterialEnvironmentResponseExperiment per (sgRNA, "
                "screen), DESeq2 log2FoldChange of guide abundance normalized on a "
                "non-targeting control guide and paired against each sample's initial "
                "condition; coding-strand in-gene guides only. The S1 Table growth "
                "screen is Cui 2018's and is not stored"
            ),
            page=(
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
