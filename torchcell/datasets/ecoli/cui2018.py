# torchcell/datasets/ecoli/cui2018
# [[torchcell.datasets.ecoli.cui2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/cui2018
# Test file: tests/torchcell/datasets/ecoli/test_cui2018.py
r"""Cui 2018 genome-wide dCas9 knockdown screen in E. coli K-12 MG1655.

Cui et al. 2018 (Nat Commun 9, 1912; doi:10.1038/s41467-018-04209-5; PMID 29765036)
electroporated a library of about 92,000 single-guide RNAs targeting random NGG-PAM
positions of the MG1655 chromosome into two MG1655 derivatives that differ only in how
much dCas9 they express, grew each pool for 17 generations under anhydrotetracycline
induction, and released the per-guide log2 fold change in guide abundance.
:class:`CrispriKnockdownCui2018Dataset` serves that released readout as one
``BacterialEnvironmentResponseExperiment`` per (guide, screened strain).

WHAT THE RELEASE REPORTS, AND WHAT IT DOES NOT SEPARATE. This paper is known for the
"bad-seed" effect: guides sharing certain five PAM-proximal bases kill E. coli
"regardless of the other 15 nucleotides of guide sequence", so a guide's measured
fitness defect is not necessarily its target gene's phenotype. The released screen
table (Supplementary Data 5) does NOT separate the two. Its ``fit18`` and ``fit75``
columns are the RAW per-guide log2 fold change in each strain, and the decomposition
into on-target repression, off-target binding at 9-or-more-nt seed matches, and
sequence-intrinsic toxicity is a MODEL output in the paper (a locally connected network
trained on guides in neutral regions, Pearson r 0.56) with neither the predictions nor
any corrected per-guide value released as a column. So the measured log2FC is what this
loader stores, and it CONFOUNDS those three causes. The paper's own design rule (v) is
the honest reading of a single record: "the effects of genes on a given phenotype should
ideally not be inferred from the effect of a single guide but rather from the statistical
analyses of several guides."

THE TWO SCREENS ARE TWO CONFOUND REGIMES, NOT REPLICATES. ``screen_id`` is the screened
strain. ``LC-E18`` carries the original Ptet-dCas9 cassette; ``LC-E75`` carries a
fine-tuned cassette expressing dCas9 "2.6-time lower", where the bad-seed effect is
"largely alleviated ... but not abolished" (130 of 1024 seeds significant in LC-E18, 14
in LC-E75) while coding-strand targets of essential genes still deplete. The two strains
are also distinct ``BacterialStrainBackground`` objects on the reference, from
Supplementary Table 7, so the regime is readable from the record rather than only from
the screen label.

THE PER-SEED STATISTICS ARE NOT RECORDS, AND ARE NOT LOADED. Supplementary Data 3 gives
the mean, SD, n and Bonferroni-corrected p-value of each of 1,022 five-base seeds in
each strain. Its unit of observation is a five-nucleotide SEQUENCE, not a strain, so it
is not a genotype-by-environment-to-phenotype record and no phenotype here can carry it.
It is named in the raw mirror's ``si_expected`` as a released file this loader does not
consume.

A ROW OF THE RELEASE IS A (GUIDE, TARGET POSITION) PAIR; A RECORD IS A (GUIDE, STRAIN).
Measured on the pinned bytes: 85,381 rows over 78,137 distinct guides, with each guide
appearing exactly ``ntargets`` times, once per perfect chromosomal match, and ``fit18``
and ``fit75`` CONSTANT across a guide's rows (0 guides carry two values of either). The
measurement therefore belongs to the guide, not to the position, and writing one record
per row would replicate one measurement across up to 47 positions. A multi-target guide
instead becomes ONE record whose genotype carries one knockdown perturbation per
distinct targeted gene: the repeated-element guides are genuine multi-locus knockdowns
(the seven rRNA operons, the insL/insH IS copies, the rhsABC paralogs, the glnU/glnW
tRNA pair).

POSITIONS ARE NOT STORED, AND THE REASON IS A GENOME VERSION. The library was designed
"around the genome of E. coli strain MG1655 (NC_000913.2)" while the pinned assembly set
is GenBank ASM584v2 (U00096.3), so the released ``pos`` column is in coordinates one
annotation release older than the records' own assembly. Nothing is lost: the 20-nt
spacer is stored on every record's ``CrisprConstruct.guide_sequence``, which determines
the target and the strand against whatever assembly a consumer pins. The ``pos``,
``ori``, ``coding`` and ``essential`` cells are kept verbatim in
``preprocess/guide_retention.csv``.

THE NON-TARGETING CONTROL GUIDE IS THE REFERENCE, AND IT IS NOT A ROW. Fold changes were
"normalized to the control guide 5'-TGAGACCAGTCTAGGTCTCG-3'", and that guide is absent
from the released table (measured: 0 rows). It is therefore typed as the
``phenotype_reference``: the same environment, the same strain, and
``environment_response=0.0``, which is what the normalization makes the control guide's
value by construction. No record claims the control guide as a genotype, because no
genotype it perturbs was released.

WHICH MEDIUM, AND THE ONE VALUE THIS PAPER DOES NOT PIN. The Methods state "Cells were
grown in Luria-Bertani (LB) broth" and print no amounts, so the formulation is UNSTATED
and neither the Miller (10 g/L NaCl) nor the Lennox (5 g/L NaCl) object can be excluded
from the text. The loader takes the shared ``MEDIA_LIBRARY["LB"]`` object, the library's
unqualified-LB entry that the rows naming "LB Miller" without amounts also take, so this
screen joins them; no media entry is added, because ``media.py`` is a fingerprinted
shared value file. This is the one environment value the source does not pin and it is
recorded here, in ``preprocess/build_accounting.json`` and in the note.

THE DATA-AVAILABILITY STATEMENT MISNUMBERS ITS OWN FILE. The paper says "The screen
results are provided as Supplementary Data 4", while the Description of Additional
Supplementary Files names Supplementary Data 4 "Plasmid sequences" and Supplementary
Data 5 "Screen results". The bytes this loader consumes are MOESM8, whose columns are
the screen results, so Supplementary Data 5 is the file and the Data availability
sentence is off by one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import re
import shutil
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data.experiment_dataset import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import LB
from torchcell.datamodels.schema import (
    AssayType,
    AssemblyReferenceGenome,
    BacterialCrisprInterferencePerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialGeneNamespace,
    BacterialStrainBackground,
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
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome
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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paper identity and the pinned artifacts
# --------------------------------------------------------------------------- #
DOI = "10.1038/s41467-018-04209-5"
PMID = "29765036"
PMCID = "PMC5954155"
TITLE = "A CRISPRi screen in E. coli reveals sequence-specific toxicity of dCas9"
CITATION_KEY = "cuiCRISPRiScreenColi2018"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

#: PMC Article Datasets bucket prefix for this article's supplementary files.
PMC_PREFIX = f"{PMCID}.1"

#: Supplementary Data 5, the screen-results table: the only released file this loader
#: consumes. Published as MOESM8; the Data availability sentence calls it Data 4.
SCREEN_FILENAME = "41467_2018_4209_MOESM8_ESM.csv"
SCREEN_REL = f"data/{SCREEN_FILENAME}"
SCREEN_SHA256 = "95ebaa5a0c92c63849617f48889e2d28b7805fdffdd960527f8a501381143c1e"
SCREEN_BYTES = 12080844
SCREEN_RETRIEVED_AT = "2026-10-07"

#: The article OCR in the torchcell-library mirror: quoted here, never parsed.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "b8e28e16f0cbb4f8d4061f79505409c1a9aa03ed4ba8349ba4dca2327d26d22d"
#: The Supplementary Information OCR in the same mirror, which carries Supplementary
#: Table 7 (the strain constructions) and the Supplementary Figure 1 legend (37 C).
SI1_MD = "si/si1.md"
SI1_MD_SHA256 = "fd8cbfc076133e745eb5f31e46626869e1baf368d1e41e78fdfe853370f56f95"

# --------------------------------------------------------------------------- #
# Assembly, namespace and the released columns
# --------------------------------------------------------------------------- #
MG1655_NAMESPACE: BacterialGeneNamespace = "ecoli_k12_mg1655_bnumber"
MG1655_STRAIN: Literal["MG1655"] = "MG1655"

#: Every column of Supplementary Data 5, in order; the build refuses any other header.
SCREEN_COLUMNS: tuple[str, ...] = (
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
#: The cell the release writes where a target position lies in no annotated gene.
NO_GENE = "NA"
#: A 20-nt spacer over the four unambiguous bases; the build refuses anything else.
SPACER_RE = re.compile(r"^[ACGT]{20}$")
#: A b-number of the pinned MG1655 assembly.
BNUMBER_RE = re.compile(r"^b\d{4}$")

#: The two screens: ``screen_id``, the released fitness column, and the strain whose
#: background the records of that screen pin.
SCREENS: tuple[tuple[str, str], ...] = (("LC-E18", "fit18"), ("LC-E75", "fit75"))

#: Frozen oracles, measured on the pinned bytes. The build refuses any other count.
EXPECTED_SOURCE_ROWS = 85381
EXPECTED_GUIDES = 78137
EXPECTED_RECORDS = 141542
#: Distinct b-numbers the retained records perturb.
EXPECTED_TARGETS = 4263

#: Drop reasons, each a rule this loader declares; the counts are measured, not stated.
DROP_NO_GENE = "no_target_in_a_gene"
DROP_MIXED = "target_outside_a_gene"
DROP_RETIRED = "gene_retired_in_the_pinned_assembly"
DROP_COLLISION = "gene_symbol_shared_with_another_source_name"
DROP_AMBIGUOUS = "gene_symbol_ambiguous_in_the_pinned_assembly"
#: Guides dropped per reason, measured on the pinned bytes (x2 screens = measurements).
EXPECTED_DROPS: dict[str, int] = {
    DROP_NO_GENE: 6268,
    DROP_MIXED: 173,
    DROP_RETIRED: 619,
    DROP_COLLISION: 283,
    DROP_AMBIGUOUS: 23,
}

# --------------------------------------------------------------------------- #
# Sourced values
# --------------------------------------------------------------------------- #
_PAGE_RESULTS_POSITION = "Results, Effect of dCas9-binding position and orientation"
_PAGE_RESULTS_SEED = "Results, Machine-learning approach reveals toxic seed sequences"
_PAGE_RESULTS_LOW_DCAS9 = (
    "Results, The bad-seed effect is alleviated at low dCas9 concentrations"
)
_PAGE_DISCUSSION = "Discussion"
_PAGE_METHODS_MEDIA = "Methods, Bacterial strains and media"
_PAGE_METHODS_LIBRARY = "Methods, Library construction"
_PAGE_METHODS_ASSAY = "Methods, dCas9 knockdown assay"
_PAGE_METHODS_FOLD_CHANGE = "Methods, Fold-change computation"
_PAGE_METHODS_WESTERN = "Methods, Western blot"
_PAGE_DATA_AVAILABILITY = "Data availability"
_PAGE_SI_FIG1 = "Supplementary Figure 1 legend"
_PAGE_SI_TABLE7 = "Supplementary Table 7. Strains made using the OSIP system."


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
    """Bind a value to a verbatim quote in the pinned Supplementary Information OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI1_MD,
            citation_key=CITATION_KEY,
            sha256=SI1_MD_SHA256,
            method="MinerU OCR of the Supplementary Information PDF (library mirror)",
            page=page,
        ),
    )


_Q_LIBRARY_MG1655 = (
    "We designed a library of ${ \\sim } 9 2 { , } 0 0 0$ unique guide RNAs targeting "
    "random positions along the genome of $E$ . coli MG1655, with the simple "
    "requirement of a “NGG” PAM. The library contains an average of 19 "
    "targets per gene."
)
_Q_LIBRARY_DESIGN = (
    "The library was designed by randomly choosing targets with a proper NGG PAM "
    "around the genome of $E$ . coli strain MG1655 (NC_000913.2). A pool of 92,919 "
    "oligonucleotides (synthesized by CustomArray) was amplified with primers LC296 "
    "and LC297"
)
_Q_LC_E18 = (
    "A pool of guide RNAs obtained through onchip oligo synthesis was cloned under the "
    "control of a constitutive promoter on plasmid psgRNA and electroporated in strain "
    "LC-E18 carrying the dCas9 gene under the control of a Ptet promoter in the "
    "chromosome (Supplementary Fig. 1)."
)
_Q_LB = (
    "Cells were grown in Luria-Bertani (LB) broth. LB agar $1 . 5 \\%$ was used as "
    "solid medium."
)
_Q_ATC = (
    "The expression of dCas9 was then induced by addition of aTc to a final "
    "concentration of $1 \\mathrm { n M }$ ."
)
_Q_GENERATIONS = (
    "Cells were grown for 17 generations by diluting the culture 100-fold once it "
    "reached OD600 2.2–2.5."
)
_Q_TRIPLICATE = (
    "The experiment was performed in triplicates starting from independent aliquots of "
    "the library generated from independent electroporation assays."
)
_Q_DESEQ2 = (
    "The fold change in abundance of each guide RNA was computed from read counts "
    "using $\\mathrm { D E S e q } \\breve { 2 } ^ { 2 7 }$ using data from the three "
    "replicates and normalized to the control guide $5 ^ { \\prime }$ "
    "-TGAGACCAGTCTAGGTCTCG- $3 ^ { \\prime }$ ."
)
_Q_READ_FILTER = (
    "Guides with a total number of reads across samples ${ < } 2 0$ were discarded "
    "from the analysis."
)
_Q_SUPP_DATA5 = (
    "A list of all targets with computed fold-change values is provided as "
    "Supplementary Data 5."
)
_Q_LOG2FC = (
    "The effect of each guide on the cell fitness can be measured as the fold change "
    "in abundance (log2FC) of the guide RNA in the library during the course of the "
    "experiment, as measured through deep sequencing of the library."
)
_Q_LC_E75_SCREEN = (
    "Strain LC-E75 carrying this fine-tuned Ptet-dCas9 cassette was then used to "
    "perform a genome-wide dCas9 knockdown screen following the same protocol as the "
    "screen previously performed with strain LC-E18."
)
_Q_EXPRESSION_RATIO = (
    "The expression cassette selected in this manner displayed an expression level "
    "2.6-time lower than the original strain LC-E18 and was integrated in strain "
    "LC-E75."
)
_Q_BAD_SEED = (
    "Using a machine-learning approach, we reveal that guide RNAs sharing specific "
    "5-nucleotide seed sequences can produce strong fitness defects or even kill E. "
    "coli regardless of the other 15 nucleotides of guide sequence."
)
_Q_BAD_SEED_130 = (
    "All in all, 130 out of the 1024 possible combinations of 5 nucleotides show a "
    "significantly reduced fitness compared to the mean (single sample $t$ -test, $ { "
    "p } < 0 . 0 1$ after Bonferroni correctio"
)
_Q_SEVERAL_GUIDES = (
    "For the reasons described above, the effects of genes on a given phenotype should "
    "ideally not be inferred from the effect of a single guide but rather from the "
    "statistical analyses of several guides."
)
_Q_OFF_TARGET_MEDIAN = (
    "Guides that target the chromosome of $E _ { \\ast }$ . coli MG1655 have a median "
    "of 4 off-targets that carry a perfect match of 9 nt or more with the seed "
    "sequence and a NGG PAM motif"
)
_Q_SPCAS9 = (
    "Rabbit monoclonal antibodies to SpCas9 (ab189380, Abcam, diluted 10,000-fold)"
)
_Q_DATA_AVAILABILITY = (
    "Data availability. The screen results are provided as Supplementary Data 4."
)
_Q_SI_37C = (
    "Cells were grown at $3 7 ^ { \\circ } \\mathsf { C }$ in LB supplemented with aTc "
    "and the psgRNA library was extracted and sequenced at the beginning and at the "
    "end of the experiment."
)
#: Supplementary Table 7's own row for each strain. The OCR renders the table as HTML,
#: so the verbatim quote is the HTML row: a prettier rendering would not be a substring
#: of the pinned bytes and could not be audited against them.
_Q_SI_TABLE7_LC_E18 = (
    "<tr><td rowspan=1 colspan=1>LC-E18</td><td rowspan=1 colspan=1>MG1655</td>"
    "<td rowspan=1 colspan=1>pOSIP-KL-sulA-GFP</td><td rowspan=1 colspan=1>N/A</td>"
    "<td rowspan=1 colspan=1>pOSIP-KH-RBS2-dCas9</td></tr>"
)
_Q_SI_TABLE7_LC_E75 = (
    "<tr><td rowspan=1 colspan=1>LC-E75</td><td rowspan=1 colspan=1>MG1655</td>"
    "<td rowspan=1 colspan=1>pOSIP-KL-mcherry</td>"
    "<td rowspan=1 colspan=1>pOSIP-CO-RBS-library-dCas9 (2-3)*</td>"
    "<td rowspan=1 colspan=1>N/A</td></tr>"
)

REFERENCE_STRAIN_SOURCE = _paper(
    MG1655_STRAIN,
    _Q_LIBRARY_DESIGN,
    page=_PAGE_METHODS_LIBRARY,
    note="the library's targets were chosen on the MG1655 chromosome and both screened "
    "strains are MG1655 derivatives (Supplementary Table 7), so MG1655 is the "
    "reference assembly; the release names NO b-numbers, so no ECK crosswalk is used",
)
GENOME_VERSION_DRIFT = _paper(
    "NC_000913.2",
    _Q_LIBRARY_DESIGN,
    page=_PAGE_METHODS_LIBRARY,
    note="the released pos column is in the coordinates of this annotation release, one "
    "older than the pinned GenBank ASM584v2 (U00096.3); positions are therefore not "
    "stored and the 20-nt spacer carries the target instead",
)
EFFECTOR = _paper(
    "dCas9",
    _Q_LC_E18,
    page=_PAGE_RESULTS_POSITION,
    note="the paper's own term throughout; the Western blot identifies it as SpCas9 "
    f"('{_Q_SPCAS9}'), which is not restated as the effector string",
)
READOUT = _paper(
    "log2_ratio",
    _Q_LOG2FC,
    page=_PAGE_RESULTS_POSITION,
    note="a signed log2 fold change in guide abundance, routinely negative, which is "
    "why the record is an EnvironmentResponsePhenotype and not a FitnessPhenotype",
)
ASSAY = _paper(
    "pooled_competitive_growth_barcode",
    _Q_LOG2FC,
    page=_PAGE_RESULTS_POSITION,
    note="guide abundance in one pooled culture read by deep sequencing at the start "
    "and the end of the competition",
)
NORMALIZATION = _paper(
    "TGAGACCAGTCTAGGTCTCG",
    _Q_DESEQ2,
    page=_PAGE_METHODS_FOLD_CHANGE,
    note="the non-targeting control guide every fold change is normalized to, which is "
    "why the phenotype_reference carries environment_response=0.0; measured on the "
    "pinned bytes: this spacer is in 0 of the 85,381 released rows",
)
READ_FILTER = _paper(
    20,
    _Q_READ_FILTER,
    page=_PAGE_METHODS_FOLD_CHANGE,
    note="the authors' own filter, already applied to the released table; this loader "
    "adds no abundance filter of its own",
)
N_REPLICATES = _paper(
    3,
    _Q_TRIPLICATE,
    page=_PAGE_METHODS_ASSAY,
    note="three independent pooled cultures from independent electroporations, which "
    f"DESeq2 pooled into the released fold change ('{_Q_DESEQ2}')",
)
SAMPLE_UNIT = _paper(
    "biological_replicate",
    _Q_TRIPLICATE,
    page=_PAGE_METHODS_ASSAY,
    note="independent aliquots from independent electroporation assays, the paper's "
    "own description of one replicate",
)
UNCERTAINTY_UNRELEASED = _paper(
    None,
    _Q_SUPP_DATA5,
    page=_PAGE_METHODS_FOLD_CHANGE,
    note="measured on the pinned bytes: the ten released columns are guide, gene, "
    "essential, pos, ori, coding, fit18, fit75, ntargets, seq. DESeq2 computes an "
    "lfcSE per guide and the release carries none, so the uncertainty fields are typed "
    "ProvenanceGaps rather than a reconstructed number",
)
MEDIUM = _paper(
    "LB",
    _Q_LB,
    page=_PAGE_METHODS_MEDIA,
    note="the Methods print no amounts, so the formulation is UNSTATED and neither "
    "MEDIA_LIBRARY['LB'] (Miller, 10 g/L NaCl) nor MEDIA_LIBRARY['LB_LENNOX'] (5 g/L) "
    "can be excluded from the text; the unqualified-LB object is taken for the join "
    "and this is the one environment value the source does not pin",
)
TEMPERATURE = _si(
    37.0,
    _Q_SI_37C,
    page=_PAGE_SI_FIG1,
    note="the screen temperature, stated in the Supplementary Figure 1 legend that "
    "describes this experimental scheme; the Methods state it for no step",
)
INDUCER_NM = _paper(
    1.0,
    _Q_ATC,
    page=_PAGE_METHODS_ASSAY,
    note="anhydrotetracycline induces the chromosomal Ptet-dCas9 cassette, which is "
    "what makes the knockdown happen; both screens used the same dose",
)
GENERATIONS = _paper(
    17.0,
    _Q_GENERATIONS,
    page=_PAGE_METHODS_ASSAY,
    note="the competition length; the paper states generations and no wall-clock hours",
)
LIBRARY_SIZE = _paper(
    92919,
    _Q_LIBRARY_DESIGN,
    page=_PAGE_METHODS_LIBRARY,
    note="the oligonucleotide pool; the released table holds 78,137 distinct guides "
    "after the authors' own read filter, and the Discussion calls the library 84,215",
)
BAD_SEED_CONFOUND = _paper(
    "guide_sequence_toxicity_not_separated",
    _Q_BAD_SEED,
    page="Abstract",
    note="the release carries the raw per-guide log2FC and no corrected value, so a "
    "stored response confounds on-target repression, off-target binding and the "
    f"sequence-intrinsic bad-seed effect ('{_Q_BAD_SEED_130}')",
)
SEVERAL_GUIDES = _paper(
    "aggregate_several_guides_per_gene",
    _Q_SEVERAL_GUIDES,
    page=_PAGE_DISCUSSION,
    note="the paper's design rule (v): a gene-level call is a statistic over several "
    "guides, never one record",
)
OFF_TARGET_BURDEN = _paper(
    4,
    _Q_OFF_TARGET_MEDIAN,
    page=_PAGE_DISCUSSION,
    note="median off-targets per guide with a 9-or-more-nt seed match and an NGG PAM; "
    "the released ntargets column counts PERFECT full-length matches only, which is "
    "the only multiplicity this loader can type as perturbations",
)
LOW_DCAS9 = _paper(
    2.6,
    _Q_EXPRESSION_RATIO,
    page=_PAGE_RESULTS_LOW_DCAS9,
    note="LC-E75 expresses dCas9 2.6-fold lower than LC-E18, which is the whole "
    "difference between the two screens",
)
DATA_AVAILABILITY_MISNUMBER = _paper(
    "Supplementary Data 5",
    _Q_DATA_AVAILABILITY,
    page=_PAGE_DATA_AVAILABILITY,
    note="the Data availability sentence says Data 4; the Description of Additional "
    "Supplementary Files names Data 4 'Plasmid sequences' and Data 5 'Screen results', "
    "and MOESM8 carries the screen columns, so Data 5 is the file",
)
#: Every sourced value, so a reader can audit the sourcing without reading the code.
SOURCED_VALUES: dict[str, SourcedValue] = {
    "reference_strain": REFERENCE_STRAIN_SOURCE,
    "genome_version_drift": GENOME_VERSION_DRIFT,
    "effector": EFFECTOR,
    "readout": READOUT,
    "assay_type": ASSAY,
    "normalization_control_guide": NORMALIZATION,
    "read_filter": READ_FILTER,
    "n_samples": N_REPLICATES,
    "sample_unit": SAMPLE_UNIT,
    "uncertainty": UNCERTAINTY_UNRELEASED,
    "medium": MEDIUM,
    "temperature": TEMPERATURE,
    "inducer_nM": INDUCER_NM,
    "duration_generations": GENERATIONS,
    "library_size": LIBRARY_SIZE,
    "bad_seed_confound": BAD_SEED_CONFOUND,
    "aggregation_rule": SEVERAL_GUIDES,
    "off_target_burden": OFF_TARGET_BURDEN,
    "low_dcas9_ratio": LOW_DCAS9,
    "screen_results_file": DATA_AVAILABILITY_MISNUMBER,
}

#: Supplementary Table 7's own row for each screened strain, and a one-line reading of
#: it. The integrations sit at phage attachment sites and the paper names no disrupted
#: locus, so each background carries no typed allele.
STRAIN_BACKGROUNDS: dict[str, tuple[str, str]] = {
    "LC-E18": (
        _Q_SI_TABLE7_LC_E18,
        "MG1655 with pOSIP-KH-RBS2-dCas9 integrated at the HK022 attB site (the "
        "original Ptet-dCas9 cassette) and pOSIP-KL-sulA-GFP at the lambda attB site",
    ),
    "LC-E75": (
        _Q_SI_TABLE7_LC_E75,
        "MG1655 with pOSIP-CO-RBS-library-dCas9 (2-3) integrated at the primary 186 "
        "attB site (the fine-tuned Ptet-dCas9 cassette expressing dCas9 2.6-fold lower "
        "than LC-E18) and pOSIP-KL-mcherry at the lambda attB site",
    ),
}


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/cuiCRISPRiScreenColi2018``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pmc_cloud_key(filename: str) -> str:
    """Bucket key of one supplementary file in the PMC Article Datasets bucket."""
    return f"{PMC_PREFIX}/{filename}"


def deposit_raw_mirror(
    *,
    screen_path: str | Path,
    retrieved_at: str = SCREEN_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror (Supplementary Data 5) and its ``manifest.json``.

    Idempotent by sha256: an existing mirror file whose digest matches is left alone and
    a differing one raises rather than being overwritten. The file is verified BEFORE
    anything is written, so a refusal leaves no partial deposit. The bytes come from the
    PMC Article Datasets bucket, which is directly scriptable through
    ``torchcell.literature.retrieve.pmc_cloud_object`` and reproduced the pinned digest
    and the pinned byte count on ``retrieved_at``.
    """
    root = raw_mirror_dir(data_root)
    digest = _sha256(screen_path)
    if digest != SCREEN_SHA256:
        raise RuntimeError(
            f"{screen_path} has sha256 {digest}, not the pinned {SCREEN_SHA256}"
        )
    dest = root / SCREEN_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != SCREEN_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(screen_path, dest)
    size = dest.stat().st_size
    if size != SCREEN_BYTES:
        raise RuntimeError(f"{dest} is {size} bytes, not the pinned {SCREEN_BYTES}")
    key = pmc_cloud_key(SCREEN_FILENAME)
    url = f"https://pmc-oa-opendata.s3.amazonaws.com/{key}"
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=[
            ArtifactRecord(
                path=SCREEN_REL,
                role=ROLE_RAW_DATA,
                bytes=size,
                sha256=SCREEN_SHA256,
                source=url,
                original_filename=SCREEN_FILENAME,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.pmc_cloud,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.pmc_cloud_object",
                    params={"key": key},
                    sha256=SCREEN_SHA256,
                    retrieved_at=retrieved_at,
                ),
            )
        ],
        si_data_sources=[url, f"https://doi.org/{DOI}"],
        si_expected=[
            "Supplementary Data 5 (MOESM8) -- the screen-results table, deposited. It "
            "is the ONLY released file this loader consumes: 85,381 rows over 78,137"
            " distinct guides, columns guide, gene, essential, pos, ori, coding, "
            "fit18, fit75, ntargets, seq. The Data availability sentence calls it "
            "Supplementary Data 4, which the Description of Additional Supplementary "
            "Files contradicts",
            "Supplementary Data 3 (MOESM6) -- the per-seed statistics: mean, SD, n and "
            "Bonferroni-corrected p-value of 1,022 five-base seed sequences in each "
            "strain. NOT deposited: its unit of observation is a five-nucleotide "
            "SEQUENCE, not a strain, so it is not a genotype-by-environment record and "
            "no phenotype in the schema can carry it",
            "Supplementary Data 1 and 2 (MOESM4, MOESM5) -- the 64 essential-gene "
            "promoter operons and the reverse-polar-effect gene list. NOT deposited: "
            "both are analysis gene lists over the same screen, carrying no per-strain "
            "measurement this loader does not already read from Data 5",
            "Supplementary Data 4 (MOESM7) -- the psgRNA plasmid sequences. NOT "
            "deposited: no loader consumes plasmid sequence today; it is the artifact a "
            "future full-plasmid CrisprConstruct.effector_plasmid_ref would pin",
            "Supplementary Data 6 (MOESM9) -- the row indices of the paper's own "
            "train/validation/test split. NOT deposited: a modeling split over Data 5, "
            "not a measurement",
            "NO SEQUENCE-READ ACCESSION IS RELEASED. The Data availability statement "
            "names Supplementary Data and then defers: 'Other relevant data supporting "
            "the findings of the study are available in this article and its "
            "Supplementary Information files or from the corresponding author upon "
            "request.' There is no SRA, ENA or GEO accession for the guide-abundance "
            "reads or for the six resequenced bad-seed-suppressor genomes, so the "
            "released fold changes cannot be recomputed from reads",
            "The machine-learning code is a Jupyter notebook at "
            "https://gitlab.pasteur.fr/dbikard/badSeed_public, which the rebuttal also "
            "names as where 'the content of the tables is also extensively described'. "
            "NOT deposited: no loader runs it, and the column meanings this loader "
            "needs are established from the paper's own Methods and from the released "
            "values (see the dendron note)",
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
# The released table
# --------------------------------------------------------------------------- #
def read_screen_table(path: str) -> pd.DataFrame:
    """Supplementary Data 5 as strings, with its header and shape checked.

    Everything is read as ``str`` with ``keep_default_na=False`` so the release's own
    ``NA`` cells stay the token they are: a position in no annotated gene is a FACT
    about the target, not a missing value, and letting pandas turn it into a float NaN
    would make it indistinguishable from an unreported one.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != SCREEN_COLUMNS:
        raise RuntimeError(
            f"screen table header is {tuple(frame.columns)!r}, not {SCREEN_COLUMNS!r}"
        )
    if len(frame) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(
            f"screen table has {len(frame)} rows, not the pinned {EXPECTED_SOURCE_ROWS}"
        )
    return frame


class GuideMeasurement(BaseModel):
    """One guide's released measurement and the genes its perfect matches lie in.

    ``responses`` is the guide's log2FC per ``screen_id``, which the release carries as
    one value per strain for the whole guide rather than per target position.
    """

    guide: str
    n_targets: int
    source_genes: tuple[str, ...]
    positions: tuple[str, ...]
    responses: dict[str, float]


def collapse_to_guides(frame: pd.DataFrame) -> list[GuideMeasurement]:
    """One :class:`GuideMeasurement` per distinct guide, in first-appearance order.

    Asserts the release's own structure rather than assuming it: a guide appears exactly
    ``ntargets`` times, its ``fit18`` and ``fit75`` are constant across those rows, and
    its spacer is a 20-nt sequence over ACGT. Each assertion is what makes "the
    measurement belongs to the guide, not the position" a checked fact.
    """
    guides: list[GuideMeasurement] = []
    for guide, rows in frame.groupby("guide", sort=False):
        if SPACER_RE.match(str(guide)) is None:
            raise RuntimeError(f"guide {guide!r} is not a 20-nt ACGT spacer")
        n_targets = {int(value) for value in rows["ntargets"]}
        if len(n_targets) != 1:
            raise RuntimeError(f"guide {guide!r} carries ntargets {sorted(n_targets)}")
        (declared,) = n_targets
        if declared != len(rows):
            raise RuntimeError(
                f"guide {guide!r} has {len(rows)} rows but declares ntargets {declared}"
            )
        responses: dict[str, float] = {}
        for screen_id, column in SCREENS:
            values = {str(value) for value in rows[column]}
            if len(values) != 1:
                raise RuntimeError(
                    f"guide {guide!r} carries {len(values)} distinct {column} values; "
                    "the released fold change is per guide, not per position"
                )
            responses[screen_id] = float(values.pop())
        guides.append(
            GuideMeasurement(
                guide=str(guide),
                n_targets=declared,
                source_genes=tuple(str(value) for value in rows["gene"]),
                positions=tuple(str(value) for value in rows["pos"]),
                responses=responses,
            )
        )
    if len(guides) != EXPECTED_GUIDES:
        raise RuntimeError(
            f"{len(guides)} distinct guides, not the pinned {EXPECTED_GUIDES}"
        )
    return guides


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #
class RetentionRule(BaseModel):
    """One guide's retention verdict: its stored targets, or why it is dropped."""

    guide: str
    n_targets: int
    source_genes: tuple[str, ...]
    stored_tags: tuple[str, ...]
    drop_reason: str


def classify_guide(
    guide: GuideMeasurement,
    stored: Mapping[str, str],
    *,
    retired: frozenset[str],
    collided: frozenset[str],
    ambiguous: frozenset[str],
) -> RetentionRule:
    """The retention verdict for one guide, by the loader's declared drop rules.

    A guide is retained when EVERY perfect chromosomal match lies in an annotated gene
    whose source symbol resolves to a b-number of the pinned assembly. The rules, in
    the order they are applied:

    - no match lies in a gene: there is no gene perturbation to write, so the
      measurement has no genotype (``no_target_in_a_gene``);
    - some matches lie in a gene and some do not: the genotype would name a strict
      subset of the loci dCas9 actually binds, which asserts a genotype the release
      does not support (``target_outside_a_gene``);
    - a targeted symbol is retired, shares its locus with another source symbol, or is
      ambiguous in the pinned assembly: the bacterial perturbation leaf refuses a
      ``systematic_gene_name`` outside its namespace, so the record cannot be written
      at all (the three ``gene_*`` reasons).

    The last three are the cost of reading a library designed on NC_000913.2 against
    ASM584v2, and they are reported as such rather than smoothed over.
    """
    in_gene = [name for name in guide.source_genes if name != NO_GENE]
    if not in_gene:
        return RetentionRule(
            guide=guide.guide,
            n_targets=guide.n_targets,
            source_genes=guide.source_genes,
            stored_tags=(),
            drop_reason=DROP_NO_GENE,
        )
    if len(in_gene) != len(guide.source_genes):
        return RetentionRule(
            guide=guide.guide,
            n_targets=guide.n_targets,
            source_genes=guide.source_genes,
            stored_tags=(),
            drop_reason=DROP_MIXED,
        )
    reason = ""
    if any(name in retired for name in in_gene):
        reason = DROP_RETIRED
    elif any(name in collided for name in in_gene):
        reason = DROP_COLLISION
    elif any(name in ambiguous for name in in_gene):
        reason = DROP_AMBIGUOUS
    tags = tuple(sorted({stored[name] for name in in_gene})) if not reason else ()
    if not reason and any(BNUMBER_RE.match(tag) is None for tag in tags):
        raise RuntimeError(
            f"guide {guide.guide!r} resolves to {tags}, which is not all b-numbers; "
            "the reconciliation report named no reason for it"
        )
    return RetentionRule(
        guide=guide.guide,
        n_targets=guide.n_targets,
        source_genes=guide.source_genes,
        stored_tags=tags,
        drop_reason=reason,
    )


class BuildAccounting(BaseModel):
    """Everything a reader needs to audit one build's retention arithmetic."""

    dataset: str
    source_rows: int
    source_guides: int
    screens: tuple[str, ...]
    candidate_records: int
    kept_records: int
    dropped_records: int
    dropped_guides_by_reason: dict[str, int]
    distinct_targets: int
    perturbations_written: int
    reconciliation: LocusTagReconciliation
    notes: list[str] = []

    def check(self) -> None:
        """Nothing may vanish between the released table and the written store."""
        if self.kept_records + self.dropped_records != self.candidate_records:
            raise RuntimeError(
                f"{self.dataset}: {self.kept_records} kept + {self.dropped_records} "
                f"dropped != {self.candidate_records} candidates"
            )
        n_screens = len(self.screens)
        if self.candidate_records != self.source_guides * n_screens:
            raise RuntimeError(
                f"{self.dataset}: {self.source_guides} guides x {n_screens} screens "
                f"!= {self.candidate_records} candidates"
            )
        dropped = sum(self.dropped_guides_by_reason.values()) * n_screens
        if dropped != self.dropped_records:
            raise RuntimeError(
                f"{self.dataset}: the per-reason drops total {dropped}, not "
                f"{self.dropped_records}"
            )
        if self.dropped_guides_by_reason != EXPECTED_DROPS:
            raise RuntimeError(
                f"{self.dataset}: drops {self.dropped_guides_by_reason} are not the "
                f"pinned {EXPECTED_DROPS}"
            )


def _write_accounting(accounting: BuildAccounting, preprocess_dir: str) -> None:
    """Validate and write ``preprocess/build_accounting.json``."""
    accounting.check()
    os.makedirs(preprocess_dir, exist_ok=True)
    path = osp.join(preprocess_dir, "build_accounting.json")
    with open(path, "w") as handle:
        handle.write(accounting.model_dump_json(indent=2))


# --------------------------------------------------------------------------- #
# Genotype and environment builders
# --------------------------------------------------------------------------- #
def publication() -> Publication:
    """This paper, by PubMed id and DOI."""
    return Publication(
        pubmed_id=PMID,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
        doi=DOI,
        doi_url=f"https://doi.org/{DOI}",
    )


def strain_background(screen_id: str) -> BacterialStrainBackground:
    """The MG1655 derivative one screen was run in, from Supplementary Table 7.

    ``alleles`` is empty and that is a statement, not an omission: both cassettes are
    pOSIP integrations at phage attachment sites and the paper names no disrupted MG1655
    locus, so there is no allele of a b-numbered gene to type. What the source does give
    -- the strain name, its parent and the plasmid integrated at each attB site -- is
    carried verbatim in ``genotype_statement`` and read once in ``construction``.
    """
    quote, construction = STRAIN_BACKGROUNDS[screen_id]
    return BacterialStrainBackground(
        name=screen_id,
        reference_strain=MG1655_STRAIN,
        assembly_set="ecoli_K12_MG1655_ASM584v2",
        parents=["MG1655"],
        construction=construction,
        genotype_statement=quote,
        alleles=[],
        provenance=[
            _si(screen_id, quote, page=_PAGE_SI_TABLE7, note=construction),
            _paper(
                screen_id,
                _Q_LC_E18 if screen_id == "LC-E18" else _Q_LC_E75_SCREEN,
                page=_PAGE_RESULTS_POSITION
                if screen_id == "LC-E18"
                else _PAGE_RESULTS_LOW_DCAS9,
                note="the strain the pooled library was screened in",
            ),
        ],
    )


def screen_reference_genome(
    screen_id: str, data_root: str | None = None
) -> AssemblyReferenceGenome:
    """MG1655 pinned to its GenBank assembly, carrying one screen's strain background."""
    return assembly_reference(
        MG1655_STRAIN, background=strain_background(screen_id), data_root=data_root
    )


def screen_environment() -> Environment:
    """The pooled competition both screens were run in, identical for the two.

    LB at 37 C with anhydrotetracycline at 1 nM for 17 generations. The two screens
    differ in the STRAIN, which rides on the reference's background, not in the
    environment, which is why one object serves both.
    """
    return Environment(
        media=LB,
        temperature=Temperature(value=float(TEMPERATURE.value)),
        perturbations=[
            SmallMoleculePerturbation(
                compound=resolved_compound("anhydrotetracycline"),
                concentration=Concentration(
                    value=float(INDUCER_NM.value), unit=ConcentrationUnit.nanomolar
                ),
            )
        ],
        aerobicity="aerobic",
        duration_generations=float(GENERATIONS.value),
    )


#: The gaps every phenotype carries: DESeq2's per-guide standard error is not a released
#: column, so nothing here can carry a dispersion.
_UNCERTAINTY_FIELDS: tuple[str, ...] = (
    "environment_response_se",
    "environment_response_uncertainty",
    "environment_response_uncertainty_type",
)

_UNITS = (
    "log2 fold change in single-guide-RNA abundance over 17 generations of dCas9 "
    "induction, computed by DESeq2 from three biological replicates and normalized to "
    "the non-targeting control guide TGAGACCAGTCTAGGTCTCG; negative = the guide's "
    "clone was depleted from the pool"
)


def screen_phenotype(
    screen_id: str, response: float | None
) -> EnvironmentResponsePhenotype:
    """One guide's released log2FC in one screen, or the control guide's 0.0 reference.

    ``response=None`` builds the reference phenotype, whose value is 0.0 because the
    normalization makes the control guide's log2FC zero by construction. The three
    uncertainty fields are typed absences in both: DESeq2 fits a standard error per
    guide and the release carries no such column.
    """
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=0.0 if response is None else response,
        n_samples=int(N_REPLICATES.value),
        sample_unit=SampleUnit(str(SAMPLE_UNIT.value)),
        units=_UNITS,
        screen_id=screen_id,
        provenance_gaps=[
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=str(UNCERTAINTY_UNRELEASED.note),
            )
            for field in _UNCERTAINTY_FIELDS
        ],
    )


def crispri_perturbations(
    rule: RetentionRule, stored: Mapping[str, str], symbols: Mapping[str, str]
) -> list[BacterialCrisprInterferencePerturbation]:
    """One dCas9 knockdown per distinct gene the guide's perfect matches lie in.

    ``n_guides=1`` is a property of the construct, not of the library: one psgRNA
    plasmid carries one spacer, and the library's "average of 19 targets per gene" is
    the number of RECORDS a gene has, not the number of guides in any one cell. Each
    perturbation carries a ``DerivedIdentifierMapping`` naming the released gene symbol,
    because the release names no b-number and every stored tag is derived from a symbol.
    """
    construct = CrisprConstruct(
        effector=str(EFFECTOR.value), guide_sequence=rule.guide, n_guides=1
    )
    by_tag: dict[str, str] = {}
    for name in rule.source_genes:
        if name != NO_GENE:
            by_tag.setdefault(stored[name], name)
    return [
        BacterialCrisprInterferencePerturbation(
            systematic_gene_name=tag,
            perturbed_gene_name=symbols.get(tag, tag),
            gene_namespace=MG1655_NAMESPACE,
            crispr=construct,
            identifier_mapping=DerivedIdentifierMapping(
                source_identifier=by_tag[tag], route="gene_symbol"
            ),
        )
        for tag in rule.stored_tags
    ]


def annotation_symbols(genome: EcoliK12Genome, tags: Iterable[str]) -> dict[str, str]:
    """Locus tag -> the annotation's own gene symbol, falling back to the tag.

    The symbol comes from the pinned assembly so one gene carries one spelling across
    datasets; the release's own symbol is kept on the perturbation's
    ``identifier_mapping`` and in ``preprocess/guide_retention.csv``.
    """
    exact, _ = genome.feature_index["symbol"]
    by_tag: dict[str, str] = {}
    for symbol, loci in exact.items():
        for locus in loci:
            by_tag.setdefault(str(locus), str(symbol))
    return {tag: by_tag.get(tag, tag) for tag in tags}


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class CrispriKnockdownCui2018Dataset(ExperimentDataset):
    """Cui 2018 per-guide dCas9 knockdown log2 fold changes in two MG1655 strains."""

    REFERENCE_STRAIN: ClassVar[Literal["MG1655"]] = "MG1655"
    #: Measured on the pinned bytes: 4,286 of the 4,360 released gene symbols resolve to
    #: a single locus of ASM584v2 (0.983). The floor is stated below that so a small
    #: annotation move reports rather than stops, and far above it so a large one stops.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 0.95

    def __init__(
        self,
        root: str = "data/torchcell/crispri_knockdown_cui2018",
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the MG1655 genome resolves the released gene symbols to loci."""
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
        """Supplementary Data 5, the released screen-results table."""
        return [SCREEN_FILENAME]

    def download(self) -> None:
        """Link the pinned table into ``raw/`` after verifying it against the manifest."""
        _link_mirror_files(
            self.raw_dir, ((SCREEN_REL, SCREEN_FILENAME, SCREEN_SHA256),)
        )
        log.info("Cui 2018 screen table linked into %s", self.raw_dir)

    def _genome(self) -> EcoliK12Genome:
        """The injected MG1655 genome, or one opened from the genomes tier."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Build one record per (retained guide, screen); write LMDB."""
        verify_raw_files(self.raw_dir, {SCREEN_FILENAME: SCREEN_SHA256})
        frame = read_screen_table(osp.join(self.raw_dir, SCREEN_FILENAME))
        guides = collapse_to_guides(frame)
        genome = self._genome()

        names = sorted({name for name in frame["gene"].unique() if name != NO_GENE})
        resolved, report = reconcile_locus_tags(
            genome, pd.Series(names), label=self.name
        )
        report.require_resolved(self.MIN_RESOLVED_FRACTION)
        stored = dict(zip(names, resolved, strict=True))
        rules = [
            classify_guide(
                guide,
                stored,
                retired=frozenset(report.retired_kept),
                collided=frozenset(report.kept_on_collision),
                ambiguous=frozenset(report.ambiguous_kept),
            )
            for guide in guides
        ]
        kept = [rule for rule in rules if not rule.drop_reason]
        tags = sorted({tag for rule in kept for tag in rule.stored_tags})
        if len(tags) != EXPECTED_TARGETS:
            raise RuntimeError(
                f"{len(tags)} distinct targets, not the pinned {EXPECTED_TARGETS}"
            )
        symbols = annotation_symbols(genome, tags)

        environment = screen_environment()
        references = {
            screen_id: BacterialEnvironmentResponseExperimentReference(
                dataset_name=self.name,
                genome_reference=screen_reference_genome(screen_id),
                environment_reference=environment,
                phenotype_reference=screen_phenotype(screen_id, None),
            )
            for screen_id, _ in SCREENS
        }
        pub = publication()
        by_guide = {guide.guide: guide for guide in guides}

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_perturbations = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for screen_id, _ in SCREENS:
                for rule in tqdm(kept, desc=f"cui2018-{screen_id}"):
                    # Annotated list[Any]: Genotype.perturbations is an invariant list
                    # over the whole perturbation union, so a narrower list is refused.
                    perturbations: list[Any] = crispri_perturbations(
                        rule, stored, symbols
                    )
                    experiment = BacterialEnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=Genotype(perturbations=perturbations),
                        environment=environment,
                        phenotype=screen_phenotype(
                            screen_id, by_guide[rule.guide].responses[screen_id]
                        ),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(
                            experiment, references[screen_id], pub, itxn
                        ),
                    )
                    idx += 1
                    n_perturbations += len(perturbations)
        env.close()
        interned_env.close()
        if idx != EXPECTED_RECORDS:
            raise RuntimeError(
                f"{idx} records written, not the pinned {EXPECTED_RECORDS}"
            )

        pd.DataFrame(
            [
                {
                    "guide": rule.guide,
                    "n_targets": rule.n_targets,
                    "source_genes": ";".join(rule.source_genes),
                    "positions": ";".join(by_guide[rule.guide].positions),
                    "stored_tags": ";".join(rule.stored_tags),
                    "fit18": by_guide[rule.guide].responses["LC-E18"],
                    "fit75": by_guide[rule.guide].responses["LC-E75"],
                    "drop_reason": rule.drop_reason,
                }
                for rule in rules
            ]
        ).to_csv(osp.join(self.preprocess_dir, "guide_retention.csv"), index=False)
        dropped_by_reason = {
            reason: sum(1 for rule in rules if rule.drop_reason == reason)
            for reason in EXPECTED_DROPS
        }
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=len(frame),
                source_guides=len(guides),
                screens=tuple(screen_id for screen_id, _ in SCREENS),
                candidate_records=len(guides) * len(SCREENS),
                kept_records=idx,
                dropped_records=len(guides) * len(SCREENS) - idx,
                dropped_guides_by_reason=dropped_by_reason,
                distinct_targets=len(tags),
                perturbations_written=n_perturbations,
                reconciliation=report,
                notes=[
                    "a released row is a (guide, target position) pair and a record is "
                    "a (guide, screen): the released fold change is constant across a "
                    "guide's rows, which collapse_to_guides asserts rather than "
                    "assumes, so one measurement is never replicated across positions",
                    "a multi-target guide becomes ONE record carrying one knockdown "
                    "perturbation per distinct targeted gene; the repeated-element "
                    "guides are genuine multi-locus knockdowns",
                    "the stored response is the RAW per-guide log2FC and confounds "
                    "on-target repression, off-target binding at 9-or-more-nt seed "
                    "matches, and the sequence-intrinsic bad-seed effect; the paper's "
                    "decomposition is a model output with no released column, and its "
                    "own design rule (v) is to aggregate several guides per gene",
                    "the two screens are two dCas9-dose regimes, not replicates: "
                    "LC-E75 expresses dCas9 2.6-fold lower than LC-E18 and 14 rather "
                    "than 130 seed sequences stay significant; they are kept apart by "
                    "screen_id AND by distinct strain backgrounds on the reference",
                    "target positions are NOT stored: the release gives them in "
                    "NC_000913.2 coordinates while the records pin GenBank ASM584v2 "
                    "(U00096.3). The 20-nt spacer is stored instead and determines the "
                    "target and the strand against any assembly a consumer pins",
                    "the non-targeting control guide TGAGACCAGTCTAGGTCTCG is the "
                    "normalizer and is absent from the released table (0 of 85,381 "
                    "rows); it is typed as the phenotype_reference at "
                    "environment_response=0.0 and is never a record's genotype",
                    "the medium's formulation is UNSTATED: the Methods say "
                    "'Luria-Bertani (LB) broth' and print no amounts, so neither the "
                    "Miller nor the Lennox object is excluded by the text; the shared "
                    "unqualified-LB object is taken for the join and no media entry is "
                    "added",
                    "no dispersion is stored: DESeq2 fits a standard error per guide "
                    "and the release carries no such column, so all three uncertainty "
                    "fields are typed ProvenanceGaps",
                    "Supplementary Data 3's per-seed statistics are NOT loaded: their "
                    "unit of observation is a five-nucleotide sequence, not a strain",
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "Cui2018 CRISPRi: %d records over %d distinct targets (%d perturbations, "
            "%d guides dropped: %s)",
            idx,
            len(tags),
            n_perturbations,
            sum(dropped_by_reason.values()),
            dropped_by_reason,
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification (L0-L4)
# --------------------------------------------------------------------------- #
class StoreSummary(BaseModel):
    """What one memory-bounded pass over a built store sees.

    The store holds 141,542 records, so the supplementary rows are computed from this
    summary rather than from a materialized record list: the distinct targets and the
    malformed spacers are small, and the spacer count is a count.
    """

    n_records: int
    targets: tuple[str, ...]
    n_spacers: int
    malformed_spacers: tuple[str, ...]
    per_screen: dict[str, int]
    pins: tuple[tuple[str, str, str, str], ...]


def summarize_store(records: Iterable[Mapping[str, Any]]) -> StoreSummary:
    """Accumulate the supplementary rows' inputs in one pass over the records."""
    targets: set[str] = set()
    spacers: set[str] = set()
    per_screen: dict[str, int] = {}
    pins: set[tuple[str, str, str, str]] = set()
    n_records = 0
    for record in records:
        n_records += 1
        experiment = record["experiment"]
        for perturbation in experiment["genotype"]["perturbations"]:
            targets.add(str(perturbation["systematic_gene_name"]))
            spacers.add(str(perturbation["crispr"]["guide_sequence"]))
        screen_id = str(experiment["phenotype"]["screen_id"])
        per_screen[screen_id] = per_screen.get(screen_id, 0) + 1
        reference = record["reference"]["genome_reference"]
        background = reference.get("background") or {}
        pins.add(
            (
                screen_id,
                str(reference.get("assembly_set")),
                str(reference.get("assembly_accession")),
                str(background.get("name")),
            )
        )
    return StoreSummary(
        n_records=n_records,
        targets=tuple(sorted(targets)),
        n_spacers=len(spacers),
        malformed_spacers=tuple(
            sorted(s for s in spacers if SPACER_RE.match(s) is None)
        ),
        per_screen=dict(sorted(per_screen.items())),
        pins=tuple(sorted(pins)),
    )


def stored_tags_are_loci(summary: StoreSummary, genome: EcoliK12Genome) -> LevelResult:
    """L1 SUPPLEMENTARY: every stored knockdown target resolves to itself.

    The shared ``canonical_gene_names`` rule requires status ``current``, which a
    pseudogene locus never has (the bacterial resolver returns ``non_gene_feature``
    naming the same tag). This row accepts a gene or a pseudogene locus that resolves to
    itself. It is added beside the shared row, never in its place.
    """
    elsewhere = [
        tag
        for tag in summary.targets
        if (resolution := genome.resolve_gene_name(tag)).systematic_name != tag
        or resolution.status
        not in (GeneNameStatus.CURRENT, GeneNameStatus.NON_GENE_FEATURE)
    ]
    return LevelResult(
        level=Level.L1,
        name="stored_targets_are_loci_of_the_pinned_assembly",
        passed=not elsewhere,
        message=(
            f"SUPPLEMENTARY: {len(summary.targets)} stored knockdown targets; "
            f"{len(elsewhere)} do not resolve to themselves"
        ),
        details={"n_targets": len(summary.targets), "not_a_locus": elsewhere[:20]},
    )


def spacers_are_twenty_nt(summary: StoreSummary) -> LevelResult:
    """L2 SUPPLEMENTARY: every record's guide spacer is a 20-nt ACGT sequence.

    The spacer is the perturbation's identity here and the only thing that carries the
    target position, so a malformed one is a record that cannot be re-mapped.
    """
    return LevelResult(
        level=Level.L2,
        name="guide_spacers_are_twenty_nt_acgt",
        passed=not summary.malformed_spacers,
        message=(
            f"SUPPLEMENTARY: {summary.n_spacers} distinct spacers; "
            f"{len(summary.malformed_spacers)} malformed"
        ),
        details={
            "n_spacers": summary.n_spacers,
            "malformed": list(summary.malformed_spacers[:20]),
        },
    )


def screens_are_balanced(summary: StoreSummary) -> LevelResult:
    """L3 SUPPLEMENTARY: both screens hold the same retained guides, strain-pinned.

    Every retained guide was measured in both strains, so the two ``screen_id`` groups
    must be the same size, and each group's reference must pin the MG1655 GenBank
    assembly with that screen's own strain background. A group that drifted would mean
    a drop rule had fired per screen, which no rule here does.
    """
    expected_pins = {
        (screen_id, "ecoli_K12_MG1655_ASM584v2", "GCA_000005845.2", screen_id)
        for screen_id, _ in SCREENS
    }
    expected_screens = sorted(screen_id for screen_id, _ in SCREENS)
    balanced = (
        sorted(summary.per_screen) == expected_screens
        and len(set(summary.per_screen.values())) == 1
    )
    return LevelResult(
        level=Level.L3,
        name="both_screens_are_balanced_and_strain_pinned",
        passed=balanced and set(summary.pins) == expected_pins,
        message=(
            f"SUPPLEMENTARY: records per screen {summary.per_screen}; "
            f"{len(summary.pins)} distinct (screen, assembly, background) pins"
        ),
        details={
            "per_screen": summary.per_screen,
            "pins": [list(pin) for pin in summary.pins],
        },
    )


def verify_build(dataset_root: str, data_root: str | None = None) -> VerificationReport:
    """Run the environment-response verifier on a built tree and write its report.

    The streaming verifier is used because the store holds 141,542 records: it is a
    single memory-bounded pass over the same L0-L4 rules, and the three SUPPLEMENTARY
    rows come from a second such pass through :func:`summarize_store` rather than from a
    materialized record list. The verifier's own rows keep their verdicts.
    ``background_genes`` is empty on purpose: the two strains' dCas9 cassettes are pOSIP
    integrations at phage attB sites with no b-numbered locus to exclude, and they ride
    on the reference's background rather than in any genotype.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records

    genome = bacterial_genome("ecoli", MG1655_STRAIN, data_root)
    report = verify_environment_response_dataset_streaming(
        stream_records(dataset_root),
        dataset_name="CrispriKnockdownCui2018Dataset",
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{SCREEN_REL}",
            citation_key=CITATION_KEY,
            sha256=SCREEN_SHA256,
            method=(
                "Supplementary Data 5 (published as MOESM8), the per-guide screen "
                "results of the two genome-wide dCas9 knockdown screens. Label = the "
                "released fit18 (strain LC-E18) or fit75 (strain LC-E75) log2 fold "
                "change in guide-RNA abundance over 17 generations, computed by DESeq2 "
                "from three biological replicates and normalized to the non-targeting "
                "control guide TGAGACCAGTCTAGGTCTCG; the control guide is the "
                "phenotype_reference at 0.0 and appears in no released row. One record "
                "per (guide, screen) over the 70,771 guides whose every perfect "
                "chromosomal match lies in a gene resolving to an ASM584v2 b-number; "
                "a multi-target guide carries one knockdown per distinct targeted "
                "gene. The response is RAW and confounds on-target repression, "
                "off-target binding at 9-or-more-nt seed matches and the "
                "sequence-intrinsic bad-seed effect, which the release does not "
                "separate. n_samples=3 biological replicates; DESeq2's per-guide "
                "standard error is not a released column, so all three uncertainty "
                "fields are typed ProvenanceGaps. Environment: LB (formulation "
                "unstated by the source) at 37 C with anhydrotetracycline at 1 nM for "
                "17 generations. Dropped: 6,268 guides with no match in a gene, 173 "
                "with a match outside one, and 925 whose targeted symbol is retired, "
                "shared or ambiguous in ASM584v2 (the library was designed on "
                "NC_000913.2)"
            ),
            page="Nat Commun 9:1912; Supplementary Data 5 (41467_2018_4209_MOESM8)",
            retrieved=SCREEN_RETRIEVED_AT,
        ),
        expected_count=EXPECTED_RECORDS,
        sgd_genes=set(genome.genbank.loci),
        resolve_gene_name=genome.resolve_gene_name,
    )
    summary = summarize_store(stream_records(dataset_root))
    report.add(stored_tags_are_loci(summary, genome))
    report.add(spacers_are_twenty_nt(summary))
    report.add(screens_are_balanced(summary))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build the dataset and verify it, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    root = osp.join(data_root, "data/torchcell/crispri_knockdown_cui2018")
    genome = bacterial_genome("ecoli", MG1655_STRAIN, data_root)
    dataset = CrispriKnockdownCui2018Dataset(root=root, ecoli_genome=genome)
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
                    "source_guides",
                    "candidate_records",
                    "kept_records",
                    "dropped_records",
                    "dropped_guides_by_reason",
                    "distinct_targets",
                    "perturbations_written",
                )
            },
            indent=2,
        )
    )
    report = verify_build(root, data_root)
    print(report.summary())
    for result in report.results:
        flag = "PASS" if result.passed else "FAIL"
        print(f"  [{flag}] L{int(result.level)} {result.name}: {result.message}")


if __name__ == "__main__":
    main()
