# torchcell/datasets/ecoli/rousset2018
# [[torchcell.datasets.ecoli.rousset2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/rousset2018
# Test file: tests/torchcell/datasets/ecoli/test_rousset2018.py
"""Rousset 2018: genome-wide CRISPR-dCas9 guide fitness, growth and three phages.

Rousset et al. 2018 (PLoS Genetics 14:e1007749, doi:10.1371/journal.pgen.1007749;
PMID 30403660; PMC6242692) screened a pooled library of about 92,000 random-position
sgRNAs in two *E. coli* K-12 MG1655 derivatives carrying an aTc-inducible dCas9, and
released per-guide log2 fold changes for five independent screens: growth in rich medium
over 17 generations (S1 Table, the filtered library of about 59,000 guides), challenge by
phage lambda, T4 and 186cIts at MOI 1 (S4 Table, three columns on a library of about
17,200 guides), and recovery of the guide-carrying cosmid after a lambda transduction
assay (S6 Table, the same 17,200 guides).

RECORD = one (sgRNA x screen) ``BacterialEnvironmentResponseExperiment``:

- GENOTYPE: one ``BacterialCrisprInterferencePerturbation`` keyed by the target gene's
  MG1655 b-number, carrying the 20-nt spacer on the shared ``crispr`` construct
  (``effector="dCas9"``). The release names genes by SYMBOL, so every symbol is resolved
  to a current locus through ``reconcile_locus_tags``; the symbol it came from is kept on
  ``identifier_mapping`` (``route="gene_symbol"``), never dropped.
- ENVIRONMENT: the screen's own culture. The growth screen is LB plus 1 nM aTc for 17
  generations; the three phage challenges and the transduction assay are LB plus 0.2%
  maltose, 5 mM CaCl2 and 1 microM aTc at 37 C for 2 h, each phage challenge carrying one
  ``PhagePerturbation`` at ``multiplicity_of_infection=1``. aTc is an added small
  molecule, not an LB ingredient, so it is a ``SmallMoleculePerturbation`` in every
  environment rather than a medium component.
- PHENOTYPE: ``EnvironmentResponsePhenotype`` with ``measurement_type=log2_ratio``. The
  reference carries 0.0, which is what no change in guide abundance is.

WHY ``EnvironmentResponsePhenotype`` AND NOT ``FitnessPhenotype``. The released value is
a signed DESeq2 ``log2FoldChange`` of guide abundance ("The log2FoldChange ... value
represents the enrichment or depletion of each sgRNA"), normalized on a non-targeting
control guide and paired against each sample's own initial condition. It is routinely
negative. Measured over the 91,609 STORED records: 92.51% of the growth screen (minimum
-11.9475), 89.16% of the transduction screen, 83.93% of T4, 44.82% of 186cIts and 27.70%
of lambda. ``FitnessPhenotype.validate_fitness`` clamps every non-positive value to 0.0,
which would erase that signal, and no ``MeasurementType`` member other than
``log2_ratio`` describes this number.

WHY ``PhagePerturbation``. Three of the five screens dose a virion: "followed by
infection with phage lambda, T4 or 186cIts at a multiplicity of infection (MOI) of 1".
That is the typed leaf for a bacteriophage on the environment axis, and the dose is an
MOI rather than a ``Concentration``. The growth screen has no phage and carries none.

THE TWO HOST STRAINS. Both are MG1655 derivatives and both are written against the
MG1655 assembly, so they differ as a ``BacterialStrainBackground`` on the reference, not
as a namespace: LC-E75 carries the optimized dcas9 cassette at the phage 186 attB site
and ran the growth screen; FR-E01 carries the same cassette at the HK022 attB site, built
so that phage 186 could be studied without interference, and ran the four phage-derived
screens.

THE GROWTH SCREEN'S DEFERRAL. "The data for the screen performed with strain LC-E75 grown
in rich medium was obtained from our previous study [26]", which is Cui et al. 2018 (Nat
Commun 9:1912, doi:10.1038/s41467-018-04209-5), mirrored as
``cuiCRISPRiScreenColi2018``. Its "dCas9 knockdown assay" paragraph is where that screen's
medium (LB), inducer dose (1 nM aTc) and 17-generation serial-dilution design are sourced
from; Rousset's own Methods state only "rich medium" and the generation count. Neither
paper states the growth screen's incubation temperature, so it is a typed
``ProvenanceGap`` on ``temperature``, never the 37 C both papers state for OTHER assays.

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
3. ``gene_symbol_is_not_in_the_mg1655_annotation`` (31 symbols, 307 records): RETIRED
   against GCA_000005845.2 (the insertion-sequence and cryptic-prophage names
   ``insA-7``, ``lomR_1``, ``C0299``, ``G0-10699``, ...).
4. ``gene_symbol_collides_with_another_symbol_on_one_mg1655_locus`` (16 symbols, 321
   records): two released symbols resolve to ONE current locus (the annotation merges the
   pseudogene fragments ``yaiT``/``yaiU``, ``ydeK``/``ydeU``, ...).
   ``reconcile_locus_tags`` keeps both as given so the records stay distinct, and a bare
   symbol is not a b-number, so neither member can be stored. The group is dropped whole:
   unlike Tong 2020's b-number case, no member IS the locus tag.
5. ``gene_symbol_is_ambiguous_in_mg1655`` (``rffT`` -> b3793, b4481; 15 records).

Retention arithmetic: 59,246 + 3 x 17,220 + 17,220 = 128,126 released (guide, screen)
cells, minus 5,063 + 30,811 + 307 + 321 + 15 = 36,517 dropped, leaves 91,609 records over
3,896 genes. The candidate table's 236,000 is 59,000 guides x 4 conditions, which counts
every guide in every condition; the phage and transduction screens were released on the
filtered 17,220-guide library, and the growth screen's template-strand and intergenic
guides are not gene perturbations.

SOURCED VALUES (module-level ``SourcedValue``s anchored to the sha256 of
``roussetGenomewideCRISPRdCas9Screens2018/paper.md`` or of
``cuiCRISPRiScreenColi2018/paper.md``, or typed ``ProvenanceGap``s):

- ``n_samples = 3``, ``sample_unit=biological_replicate`` for every record. Growth screen:
  "This screen was performed over 17 generations in triplicates from independent aliquots
  of the library generated from 3 independent transformations into strain LC-E75." Phage
  and transduction screens: "The phage screen was performed in triplicates as follows".
- Uncertainty: a ``not_reported_by_primary`` gap on
  ``environment_response_uncertainty``. The released tables carry one log2FC per guide
  and screen; DESeq2's ``lfcSE`` is not among the released columns. S1 Table additionally
  releases ``padj`` and ``gamma``, which are not uncertainties: ``padj`` is the adjusted
  p-value of the DESeq2 test and the schema has no p-value slot, and ``gamma`` is named in
  the S1 Table caption but defined nowhere in either mirrored paper (solving
  ``log2FC / log2(gamma)`` over the 20,540 rows with ``|log2FC| > 0.5`` gives an implied
  generation count of 11.41 to 15.58, median 12.07, so it is not log2FC rescaled by the
  stated 17 generations either). Both are left out rather than stored under a field whose
  meaning they would misstate.
- ``multiplicity_of_infection = 1.0`` and ``host_of_propagation = "MG1655"`` for all three
  phages; no phage family, genome type, taxon or accession is stated, so those stay None.
- ``titer_pfu_per_ml``: the stocks are 10^7 pfu/microL but the infected culture's volume
  at MOI 1 is not stated, so the in-culture titer is not computed.
- ``Temperature(37.0)``, ``duration_hours = 2.0`` and the phage-screen medium from the
  phage-screen paragraph; ``duration_generations = 17.0`` and the growth-screen medium
  from Rousset plus the Cui 2018 deferral; ``aerobic`` from shaken flask cultures with no
  gas control described.

DATA SOURCE: S1, S4 and S6 Tables (``pgen.1007749.s011.csv``, ``.s014.csv``,
``.s016.csv``) from the PMC Article Datasets bucket (``pmc_cloud``, prefix
``PMC6242692.1``), deposited in
``$DATA_ROOT/torchcell-raw/roussetGenomewideCRISPRdCas9Screens2018/`` with a
``manifest.json``. S2, S5 and S7 Tables are the gene-level medians and model estimates
derived from these three, so they are not consumed; S3 Table, S8 to S10 Tables and the
ten SI figures carry no per-guide value. The ENA BioProject PRJEB28256 holds the raw
reads the log2FC values were computed from and is recorded as the upstream accession, not
mirrored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
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
    SmallMoleculePerturbation,
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
_CUI_MEDIA = "Methods, 'Bacterial strains and media'"
_CUI_ASSAY = "Methods, 'dCas9 knockdown assay'"

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
    note="the growth screen's released library; S1 Table carries 59,246 rows",
)
LC_E75_CASSETTE = _paper(
    "LC-E75",
    "This strain expresses an optimized dcas9 cassette under the control of an "
    "aTc-inducible pTet promoter integrated at the phage 186 attB site.",
    page=_STRAINS,
    note="the growth screen's host: MG1655 plus a chromosomal aTc-inducible dcas9 "
    "cassette at the phage 186 attB site",
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
GROWTH_SCREEN_DESIGN = _paper(
    (17.0, 3),
    "The data for the screen performed with strain LC-E75 grown in rich medium was "
    "obtained from our previous study [26]. This screen was performed over 17 "
    "generations in triplicates from independent aliquots of the library generated from "
    "3 independent transformations into strain LC-E75.",
    page=_SCREENS,
    note="17 generations of exposure and 3 independent library transformations, which "
    "is the replicate unit; reference [26] is Cui 2018 (mirrored), whose 'dCas9 "
    "knockdown assay' paragraph carries the medium and the inducer dose",
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

GROWTH_MEDIUM = _cui(
    "LB",
    "Cells were grown in Luria-Bertani (LB) broth.",
    page=_CUI_MEDIA,
    note="Rousset calls the growth screen's medium 'rich medium' and defers the screen "
    "to this paper; neither states the LB formulation, so the three LB ingredients are "
    "listed with no amount",
)
GROWTH_INDUCTION = _cui(
    (1.0, "nM"),
    "The expression of dCas9 was then induced by addition of aTc to a final "
    "concentration of $1 \\mathrm { n M }$ .",
    page=_CUI_ASSAY,
    note="the growth screen's inducer dose, three orders of magnitude below the phage "
    "screen's 1 microM; dCas9 was deliberately kept low in LC-E75",
)
GROWTH_SERIAL_DILUTION = _cui(
    17.0,
    "Cells were grown for 17 generations by diluting the culture 100-fold once it "
    "reached OD600 2.2–2.5.",
    page=_CUI_ASSAY,
    note="the exposure the 17-generation count measures; Rousset states the same count",
)
GROWTH_REPLICATE_UNIT = _cui(
    "biological_replicate",
    "The experiment was performed in triplicates starting from independent aliquots of "
    "the library generated from independent electroporation assays.",
    page=_CUI_ASSAY,
    note="independent library transformations, which is a biological replicate rather "
    "than a technical one",
)

#: Every module-level sourced value, for the mirror audit.
SOURCED_VALUES: tuple[SourcedValue, ...] = (
    LIBRARY_TARGETS_MG1655,
    FILTERED_LIBRARY,
    LC_E75_CASSETTE,
    FR_E01_CASSETTE,
    FR_E01_PARENT,
    EFFECTOR,
    GROWTH_SCREEN_DESIGN,
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
    GROWTH_MEDIUM,
    GROWTH_INDUCTION,
    GROWTH_SERIAL_DILUTION,
    GROWTH_REPLICATE_UNIT,
)


# --------------------------------------------------------------------------- #
# Media and strain backgrounds
# --------------------------------------------------------------------------- #
def _lb_ingredients(provenance: SourcedValue) -> list[MediaComponent]:
    """LB's three ingredients with NO amounts.

    ``LB`` in the media library carries the Miller amounts, corroborated by four other
    papers. Neither Rousset 2018 nor Cui 2018 states a formulation (and the project's own
    "LB" is Miller for some rows and Lennox for others), so asserting either would
    fabricate three numbers. The identities are LB's definition; the amounts are
    ``None``, which is what an unsourced amount means.
    """
    return [
        MediaComponent(
            compound=resolved_compound("tryptone"),
            role=MediaComponentRole.complex_ingredient,
            concentration=None,
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[provenance],
            note="LB ingredient; no amount is stated by either paper",
        ),
        MediaComponent(
            compound=resolved_compound("yeast extract"),
            role=MediaComponentRole.complex_ingredient,
            concentration=None,
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[provenance],
            note="LB ingredient; no amount is stated by either paper",
        ),
        MediaComponent(
            compound=resolved_compound("sodium chloride"),
            role=MediaComponentRole.bulk_salt,
            concentration=None,
            provenance=[provenance],
            note="LB ingredient; no amount is stated by either paper, and the Miller "
            "and Lennox formulations differ in exactly this component",
        ),
    ]


ROUSSET2018_LB = Media(
    name="LB, formulation not stated (Rousset 2018 growth-based screen), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium="LB",
    components=_lb_ingredients(GROWTH_MEDIUM),
    provenance=[GROWTH_MEDIUM, GROWTH_SERIAL_DILUTION],
)
"""The growth screen's medium: LB, through the Cui 2018 deferral, with no amounts."""

ROUSSET2018_LB_MALTOSE_CACL2 = Media(
    name="LB with 0.2% maltose and 5 mM CaCl2, formulation not stated "
    "(Rousset 2018 phage screens), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium="LB",
    components=[
        *_lb_ingredients(PHAGE_SCREEN_MEDIUM),
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
    provenance=[PHAGE_SCREEN_MEDIUM],
)
"""The phage screens' medium: LB plus the maltose and CaCl2 the infections need."""

LC_E75_BACKGROUND = BacterialStrainBackground(
    name="LC-E75",
    reference_strain=REFERENCE_STRAIN_NAME,
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    parents=["MG1655"],
    construction="MG1655 carrying an optimized aTc-inducible dcas9 cassette integrated "
    "at the phage 186 attB site; dCas9 expression was tuned down to limit the bad-seed "
    "and off-target effects",
    genotype_statement="MG1655 186attB::pTet-dcas9 (optimized cassette)",
    provenance=[LC_E75_CASSETTE, LIBRARY_TARGETS_MG1655],
)
"""The growth screen's host strain, as a background on the MG1655 assembly."""

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
# The five screens
# --------------------------------------------------------------------------- #
StrainLabel = Literal["LC-E75", "FR-E01"]

BACKGROUNDS: dict[StrainLabel, BacterialStrainBackground] = {
    "LC-E75": LC_E75_BACKGROUND,
    "FR-E01": FR_E01_BACKGROUND,
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
    phage: str | None
    assay_type: AssayType
    units: str


CONDITIONS: tuple[ScreenCondition, ...] = (
    ScreenCondition(
        screen_id="growth_17_generations",
        table=GROWTH_TABLE,
        column="log2FC",
        strain="LC-E75",
        phage=None,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        units="log2(sgRNA abundance after 17 generations of dCas9 induction / initial "
        "pool), DESeq2, normalized on a non-targeting control guide",
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

#: The released column each condition reads, keyed by table.
TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    table: tuple(c.column for c in CONDITIONS if c.table == table)
    for table in (GROWTH_TABLE, PHAGE_TABLE, TRANSDUCTION_TABLE)
}


def _atc_perturbation(
    value: float, unit: ConcentrationUnit
) -> SmallMoleculePerturbation:
    """The aTc that induces dCas9, as the added small molecule it is.

    aTc is not an LB ingredient: it is the inducer dosed on top of the medium, so it is
    an environment perturbation rather than a medium component. Its dose differs between
    the two screens (1 nM in the growth screen, 1 microM in the phage screens), which is
    part of what makes those environments distinct.
    """
    return SmallMoleculePerturbation(
        compound=resolved_compound("anhydrotetracycline"),
        concentration=Concentration(value=value, unit=unit),
    )


def _growth_temperature_gap() -> ProvenanceGap:
    """Neither paper states the growth screen's incubation temperature."""
    return ProvenanceGap(
        field="temperature",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=Provenance(
            source_uri=PAPER_MD,
            citation_key=CUI2018_KEY,
            sha256=CUI2018_PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page=_CUI_ASSAY,
        ),
        note="Rousset's 'High-throughput screens' paragraph states 37 C for the PHAGE "
        "screen only, and the Cui 2018 'dCas9 knockdown assay' paragraph the growth "
        "screen defers to states no temperature; both papers state 37 C for other "
        "assays, which is not this one",
    )


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
    """The culture one screen was run in."""
    if condition.phage is None:
        return Environment(
            media=ROUSSET2018_LB,
            temperature=None,
            perturbations=[
                _atc_perturbation(
                    GROWTH_INDUCTION.value[0], ConcentrationUnit.nanomolar
                )
            ],
            aerobicity="aerobic",
            duration_generations=GROWTH_SERIAL_DILUTION.value,
            provenance_gaps=[_growth_temperature_gap()],
        )
    return Environment(
        media=ROUSSET2018_LB_MALTOSE_CACL2,
        temperature=Temperature(value=PHAGE_SCREEN_TEMPERATURE.value),
        perturbations=[
            _atc_perturbation(
                PHAGE_SCREEN_INDUCTION.value[0], ConcentrationUnit.micromolar
            ),
            phage_perturbation(condition.phage),
        ],
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
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


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
    """One retention rule, the records it removed, and the items it removed them for."""

    rule: str
    scope: Literal["guide", "gene"]
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
        """The three mirrored SI tables."""
        return list(TABLE_SHA256)

    def download(self) -> None:
        """Link the mirror files into ``raw/`` after verifying their pinned sha256s.

        The mirror plus the pins is canonical; the PMC bucket URLs are retrieval
        metadata ``deposit_raw_mirror`` records, never a live build dependency. Every
        file is checked against the manifest and found on disk BEFORE ``raw/`` is
        created, so an incomplete mirror leaves no half-populated raw directory behind.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        mirror = raw_mirror_dir(data_root)
        for filename, sha256 in TABLE_SHA256.items():
            relpath = table_rel(filename)
            check_manifest_pin(relpath, manifest_sha256(manifest, relpath), sha256)
            if not (mirror / relpath).exists():
                raise RuntimeError(
                    f"required raw artifact missing from mirror: {mirror / relpath}"
                )
        os.makedirs(self.raw_dir, exist_ok=True)
        for filename, sha256 in TABLE_SHA256.items():
            link_verified(
                mirror / table_rel(filename), osp.join(self.raw_dir, filename), sha256
            )
        log.info(
            "Rousset 2018 S1/S4/S6 Tables linked into %s (sha256 verified)",
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
        verify_raw_files(self.raw_dir, dict(TABLE_SHA256))
        tables = {
            filename: read_table(osp.join(self.raw_dir, filename), filename)
            for filename in TABLE_SHA256
        }
        coding = {
            filename: coding_strand(frame, filename)
            for filename, frame in tables.items()
        }
        genome = self._genome()
        symbols = sorted(
            {
                str(name)
                for filename, frame in tables.items()
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

        self._write_reports(tables, coding, resolution, dropped_by_symbol, idx)
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
    ) -> None:
        """Write the drop accounting and the identifier report to ``preprocess/``."""
        growth = tables[GROWTH_TABLE]
        n_no_gene = int(growth["gene"].isna().sum())
        n_template = int((~coding[GROWTH_TABLE] & growth["gene"].notna()).sum())
        source_records = sum(len(tables[c.table]) for c in CONDITIONS)

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
EXPECTED_SCREEN_CENSUS: dict[str, int] = {
    "growth_17_generations": 23209,
    "phage_lambda": 17100,
    "phage_T4": 17100,
    "phage_186cIts": 17100,
    "lambda_transduction": 17100,
}

#: The frozen record-count oracle: the five screens together.
EXPECTED_RECORDS = sum(EXPECTED_SCREEN_CENSUS.values())

#: The frozen gene-set oracle: distinct MG1655 b-numbers across the kept records.
EXPECTED_GENES = 3896


def screen_census(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    """SUPPLEMENTARY L1: the record count of each of the five screens.

    The shared ``count`` row checks the dataset total, which one screen's rows could
    cover for another's. This row pins the per-``screen_id`` split, so a column read from
    the wrong table shows up as a changed census rather than as the same total.
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

    Every record is checked against the MG1655 genome its references pin: the resolver of
    the canonical-name rule, and as the L4 universe every GenBank locus of the assembly
    (4,651, pseudogenes and RNA tags included; the same set as the verification runners'
    ``_ecoli_k12_gene_set``). One SUPPLEMENTARY row, the per-screen census, is appended.
    The report is written to ``preprocess/verification_report.json``.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset,
    )
    from torchcell.verification.runners import load_records

    if genome is None:
        opened = bacterial_genome("ecoli", REFERENCE_STRAIN_NAME, data_root)
        if not isinstance(opened, EcoliK12MG1655Genome):
            raise TypeError(f"expected the MG1655 genome, got {type(opened).__name__}")
        genome = opened
    records = load_records(dataset_root)
    report = verify_environment_response_dataset(
        records,
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/data/",
            citation_key=CITATION_KEY,
            sha256=TABLE_SHA256[GROWTH_TABLE],
            method=(
                "S1, S4 and S6 Tables (PMC Article Datasets "
                f"{PMCID}.1): one BacterialEnvironmentResponseExperiment per (sgRNA, "
                "screen), DESeq2 log2FoldChange of guide abundance normalized on a "
                "non-targeting control guide and paired against each sample's initial "
                "condition; coding-strand in-gene guides only"
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
    report.add(screen_census(records))
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
