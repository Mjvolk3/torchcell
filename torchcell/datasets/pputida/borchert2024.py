# torchcell/datasets/pputida/borchert2024
# [[torchcell.datasets.pputida.borchert2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/borchert2024
# Test file: tests/torchcell/datasets/pputida/test_borchert2024.py
"""Borchert 2024 fModules: the P. putida KT2440 RB-TnSeq gene-fitness compendium.

Borchert et al. 2024 (mSystems 9:e00942-23, doi:10.1128/msystems.00942-23, citation key
``borchertMachineLearningAnalysis2024``) assembled 332 RB-TnSeq samples of the KT2440
``Putida_ML5`` transposon library into one gene-by-sample fitness matrix (4,732 genes) and
decomposed it with ICA. Rank 8 of the fifty bacterial rows, marked ``aggregation``: the
matrix is the SUPERSET of the other KT2440 RB-TnSeq rows, so it is loaded here and the
subsumed rows become per-record source-study provenance (plan section 4, item 7).

SOURCE. The paper's Data Availability names the release: "Gene fitness values, associated
statistics, and metadata for each sample are available at https:// github.com/beckham-lab/
fModule." That repository holds one file, ``fModule_Metadata.xlsx`` (sheets ``metadata``,
``fitness_measurements``, ``T-like_statistics``), retrieved at commit ``30eaef39`` through
``torchcell.literature.retrieve.direct_url`` and sha256-pinned in the raw mirror
``$DATA_ROOT/torchcell-raw/borchertMachineLearningAnalysis2024/``.

RECORDS. One ``BacterialEnvironmentResponseExperiment`` per (gene, sample): the genotype is
a ``TransposonInsertionPerturbation`` of the gene's GenBank locus tag in the sample's mutant
library, the environment is the sample's medium plus its carbon / nitrogen / stress
variable, and the phenotype is the gene fitness. Replicate samples are NOT averaged (the
paper: "Biological and technical replicates present in the data were not averaged"), so
each sample is its own record set and ``screen_id`` carries the sample name.

PHENOTYPE CLASS. RB-TnSeq gene fitness is a signed log2 ratio centered on zero ("The gene
fitness values are normalized so that the typical gene has a fitness of zero").
``FitnessPhenotype.validate_fitness`` clamps every value at or below zero to 0.0, which
would erase every fitness defect, so the record family is the assembly-pinned
environment-response pair with ``measurement_type=log2_ratio`` and
``assay_type=pooled_competitive_growth_barcode``. The plan's FitnessPhenotype mapping is
blocked by that validator; the schema change it would need is stated in the PR body.

UNCERTAINTY. The release carries Wetmore 2015's moderated t, ``t = f / sqrt(0.1^2 +
max(Ve, Vn))``, a test statistic rather than an uncertainty, so the verbatim uncertainty
field is a typed gap. Price 2018 defines the Fitness Browser's standard error as "the
maximum of two estimates", and the Price 2018 E. coli loader measured ``t = f / sqrt(0.1^2
+ max(se_obs, se_naive)^2)`` to 6.4e-13, so for the 254 Fitness Browser samples the
standard error is ``sqrt((f / t)^2 - 0.1^2)``. It is stored as the derived
``environment_response_se`` when the release's three-decimal rounding, propagated through
the subtraction, bounds its error at 5 % or less; every build re-checks the identity's
implied floor ``|f / t| >= 0.1`` on every kept Fitness Browser value. For the 78 Beckham
samples the t column is not that statistic (measured: its sign disagrees with the fitness
for about half of all genes and is inverted on the strongly negative ones), so no standard
error is derived there.

DROPPED (``preprocess/dropped_records.json``): the 20 samples on ``RCH2_defined_noCarbon``,
a medium the shared library does not carry, and the 22 bioreactor and adaptation samples,
whose distinguishing variables (dissolved-oxygen set point, feed regime, sampling time,
solvent overlay) have no field on ``Environment``.

Design, sourcing and the superset finding: [[torchcell.datasets.pputida.borchert2024]].
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import os.path as osp
import pickle
import shutil
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    file_sha256,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import MOPS_MINIMAL
from torchcell.datamodels.schema import (
    AssayType,
    AssemblyReferenceGenome,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Compound,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    PhysicalFactor,
    Publication,
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
    SourceCheck,
)
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "borchertMachineLearningAnalysis2024"
DOI = "10.1128/msystems.00942-23"
TITLE = (
    "Machine learning analysis of RB-TnSeq fitness data predicts functional gene "
    "modules in Pseudomonas putida KT2440"
)
#: sha256 of the mirrored OCR every Borchert 2024 quote is a literal substring of.
PAPER_MD_SHA256 = "9de6b0772124fb77795764ce2f44187ec95737de7dcefb7335941543e75ce2f4"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

#: The release the paper names, pinned to the commit that added the file.
DATA_REPOSITORY = "https://github.com/beckham-lab/fModule"
DATA_COMMIT = "30eaef39a4609335f0f10c2d25a8eb69d426b0ce"
DATA_FILE = "fModule_Metadata.xlsx"
DATA_RELPATH = f"data/{DATA_FILE}"
DATA_URL = (
    f"https://raw.githubusercontent.com/beckham-lab/fModule/{DATA_COMMIT}/{DATA_FILE}"
)
DATA_SHA256 = "4d649385ac06684482396a125f135df22a2a5060da73485b2cd14468f8cc8be1"
#: git blob id of the file in the commit's tree (``git hash-object`` of the bytes).
DATA_GIT_BLOB = "bce51f3614fe4146b036f69edd5339cf78e042e7"
DATA_BYTES = 23261645
RAW_RETRIEVED_AT = "2026-10-07"

METADATA_SHEET = "metadata"
FITNESS_SHEET = "fitness_measurements"
T_SHEET = "T-like_statistics"
#: The metadata sheet's header, verbatim; the reader refuses any other layout.
METADATA_COLUMNS: tuple[str, ...] = (
    "orgId",
    "expName",
    "set",
    "expDesc",
    "timeZeroSet",
    "expGroup",
    "expDescLong",
    "classifier",
    "total_rep",
    "rep",
    "mutantLibrary",
    "person",
    "dateStarted",
    "setName",
    "seqindex",
    "Inoculum media type",
    "media",
    "temperature",
    "pH",
    "vessel",
    "aerobic",
    "liquid",
    "shaking",
    "condition_1",
    "units_1",
    "concentration_1",
    "condition_2",
    "units_2",
    "concentration_2",
)
#: The five leading columns of both value sheets, verbatim.
GENE_COLUMNS: tuple[str, ...] = ("orgId", "locusId", "sysName", "geneName", "desc")
N_GENES = 4732
N_SAMPLES = 332

#: Mirrored source papers whose Methods a sample is attributed or sourced by.
BORCHERT2023_KEY = "borchertRBTnSeqIdentifiesGenetic2023"
THOMPSON2020_KEY = "thompsonFattyAcidAlcohol2020"
SCHMIDT2022_KEY = "schmidtNitrogenMetabolismPseudomonas2022"
WETMORE2015_KEY = "wetmoreRapidQuantificationMutant2015"

_PAPER = Provenance(
    source_uri="paper.md", citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
)
_BORCHERT2023 = Provenance(
    source_uri="paper.md",
    citation_key=BORCHERT2023_KEY,
    sha256="e156d3b2139add2feb8b3ffb1b319f9a7d369d6ee9b6454f67c38d33c924265a",
)
_THOMPSON2020 = Provenance(
    source_uri="paper.md",
    citation_key=THOMPSON2020_KEY,
    sha256="389d0d6cd196f9f159d0485f6b8ed09743dfbe3b6682369db489d7356a469b45",
)
_SCHMIDT2022 = Provenance(
    source_uri="paper.md",
    citation_key=SCHMIDT2022_KEY,
    sha256="acc5c8bfaa588f5dda7767127f523c1721496cffa6e010acfe7e7865ede36c66",
)
_WETMORE2015 = Provenance(
    source_uri="paper.md",
    citation_key=WETMORE2015_KEY,
    sha256="ca3e7ef27a22a2a28e52ccbe5fdbb60890fea93d03b18b46e1d2b77b033b3cdb",
)
_PRICE2018 = Provenance(
    source_uri="paper.md",
    citation_key="priceMutantPhenotypesThousands2018",
    sha256="f3443cdcb2f722b5e6aa6d999f67d68a6f845eb9eb45ad0bac1cc4c24ea53e2d",
)
_RELEASE = Provenance(
    source_uri=DATA_RELPATH,
    citation_key=CITATION_KEY,
    sha256=DATA_SHA256,
    method=f"raw mirror $DATA_ROOT/{RAW_DIR_REL}/{DATA_RELPATH}",
)

# --------------------------------------------------------------------------- #
# Verbatim quotes (literal substrings of the pinned paper.md files above)
# --------------------------------------------------------------------------- #
Q_COMPENDIUM = (
    "An initial RB-TnSeq fitness data compendium was generated by collecting 332 publicly "
    "available P. putida RB-TnSeq data sets collected between 2017 and 2022, consisting "
    "of a mixture of duplicate or triplicate samples spanning 183 unique growth "
    "conditions."
)
Q_FB_SHARE = (
    "The Fitness Browser (https://fit.genomics.lbl.gov/cgi-bin/myFrontPage.cgi) was used "
    "to obtain data for 254/332 of the samples, and the remaining data were generated by "
    "the Beckham group, available through the NCBI Sequence Read Archive (SRA) with "
    "accession numbers PRJNA809672, PRJNA856070, and PRJNA1011287 (22, 38)."
)
Q_GENE_FILTER = (
    "In instances where gene fitness data for a particular gene did not exist across all "
    "332 data sets, the gene was eliminated from analysis."
)
Q_NOT_AVERAGED = (
    "Biological and technical replicates present in the data were not averaged or "
    "normalized in any other way prior to analysis."
)
Q_FINAL = (
    "This resulted in a final data set, where 4,732/5,564 protein-coding genes from P. "
    "putida contained fitness data for the 332 samples (36)."
)
Q_DATA_LOCATION = (
    "Gene fitness values, associated statistics, and metadata for each sample are "
    "available at https:// github.com/beckham-lab/fModule."
)
Q_LIBRARY = (
    "All RB-TnSeq data in the ICA data set were generated with a previously described, "
    "randomly barcoded transposon mutant library in P. putida KT2440 (Putida_ML5) (88)."
)
Q_MARINER = (
    "Also of note, the mariner transposon used in the KT2440 library does not contain an "
    "outward-facing promoter"
)
Q_ABSTRACT_CONDITIONS = (
    "In this work, independent component analysis (ICA) was applied to a compendium of "
    "existing fitness data from randomly barcoded transposon insertion sequencing "
    "(RB-TnSeq) of P. putida KT2440 grown in 179 unique experimental conditions."
)
Q_SOURCES_CITED = (
    "Gene fitness values were derived from a variety of selection conditions, including "
    "growth on single carbon sources, growth on single nitrogen sources, metabolite and "
    "osmotic stress, and volumetric scales ranging from microtiter plates to 2-L "
    "bioreactors (20–23, 37, 38)."
)
Q_WETMORE_ROUGHLY = (
    "Roughly, strain fitness is the normalized $\\log _ { 2 }$ ratio of counts between "
    "the treatment sample (i.e., after growth in a certain medium) and the reference "
    "“time-zero” sample. Gene fitness is the weighted average of the strain fitness, and "
    "a t score is computed based on the consistency of the strain fitness values for "
    "each gene."
)
Q_WETMORE_FEBA = (
    "Given a table of bar codes, where they map in the genome, and their counts in each "
    "sample, we estimate strain fitness and gene fitness values and their reliability "
    "with a custom R script (FEBA.R)."
)
Q_WETMORE_T_INTRO = (
    "(iv) $\\pmb { t }$ -like test statistic. To estimate the reliability of the fitness "
    "measurement for each gene $f ,$ we use a moderated $t$ statistic:"
)
Q_WETMORE_SIGMA = (
    "where $\\sigma$ is a small constant (we use 0.1) that represents uncertainty in the "
    "normalization for small fitness values, $V _ { e }$ represents the estimated "
    "variance, and $V _ { n }$ represents the naive variance."
)
Q_PRICE_STANDARD_ERROR = (
    "To estimate the reliability of the fitness value for a gene in a specific "
    "experiment, we use a $t$ -like test statistic, which is the gene’s fitness divided "
    "by the standard error9 . The standard error is the maximum of two estimates. The "
    "first estimate is based on the consistency of the fitness for the strains in that "
    "gene. The second estimate is based on the number of reads for the gene."
)
Q_WETMORE_TYPICAL_ZERO = (
    "The fitness data are normalized so that the typical gene has a fitness of zero "
    "(see Materials and Methods)."
)
Q_WETMORE_CENTRAL = (
    "Only strains that lie within the central 10 to $9 0 \\%$ of a gene are considered."
)
Q_THOMPSON_CONDITIONS = (
    "Global fitness analyses of transposon libraries grown on 13 fatty acids and 10 "
    "alcohols produced strong phenotypes for hundreds of genes."
)
Q_THOMPSON_ALCOHOLS = (
    "transposon libraries were grown on a number of short $n$ -alcohols (ethanol, "
    "butanol, and pentanol), diols (1,2- propanediol, 1,3-butanediol, 1,4-butanediol, "
    "and 1,5-pentanediol), and branched-chain alcohols (isopentanol, isoprenol, and "
    "2-methyl-1-butanol)."
)
Q_THOMPSON_PROPIONATE = (
    "Unsurprisingly, the MCC appeared to be absolutely required for growth on propionate "
    "$( \\mathsf { C } _ { 3 } ) ,$ , valerate"
)
Q_THOMPSON_RBTNSEQ = (
    "The libraries were then washed once in MOPS minimal medium with no carbon source "
    "and then diluted 1:50 in MOPS minimal medium with $1 0 ~ \\mathsf { m M }$ each "
    "carbon source tested. Cells were grown in $1 0 ~ \\mathsf { m } \\mathsf { I }$ of "
    "medium in test tubes at $3 0 ^ { \\circ } \\mathsf { C }$ with shaking at $2 0 0 { "
    "\\mathsf { r p m } }$ ."
)
Q_THOMPSON_FITNESS = (
    "The fitness of a strain is defined here as the normalized ${ \\mathsf { l o g } } _ "
    "{ 2 }$ ratio of barcode reads in the experimental sample to barcode reads in the "
    "time zero sample."
)
Q_THOMPSON_T = (
    "The primary statistical t value represents the form of fitness divided by the "
    "estimated variance across different mutants of the same gene."
)
Q_THOMPSON_DEFER = (
    "A more detailed explanation of calculating fitness scores can be found in a "
    "previous study by Wetmore et al. (40)."
)
Q_THOMPSON_DUPLICATE = (
    "All experiments were conducted in biological duplicate, and all fitness data are "
    "publicly available at http://fit.genomics.lbl.gov."
)
Q_THOMPSON_JBEI1 = (
    "RB–Tn-Seq experiments utilized the P. putida library JBEI-1, which has been "
    "described previously, with slight modification (18)."
)
Q_THOMPSON_MODIFIED_MOPS = (
    "When indicated, P. putida and E. coli were grown on modified MOPS "
    "(morpholinepropanesulfonic acid) minimal medium, which is comprised of"
)
Q_SCHMIDT_CONDITIONS = (
    "we identified genes and proteins involved in the assimilation of 52 different "
    "nitrogen containing compounds. To assay amino acid biosynthesis, 19 amino acid "
    "drop-out conditions were also tested. From these 71 conditions,"
)
Q_SCHMIDT_DROPOUT = (
    "By supplying all but one of the 20 proteinogenic amino acids, we created conditions "
    "where biosynthesis of amino acids is essential for growth."
)
Q_SCHMIDT_MEDIUM = (
    "a library of barcoded transposon insertion mutants was cultured in minimal media "
    "with glucose and a variety of sole nitrogen sources (Fig. S1)."
)
Q_SCHMIDT_BARSEQ = (
    "Experiments were conducted in 24-well plates; each well contained $2 ~ \\mathrm { m "
    "L }$ of nitrogen-free MOPS minimal medium with 10 mM each tested nitrogen source."
)
Q_SCHMIDT_DURATION = (
    "samples were collected after 24 to $7 2 \\ \\mathrm { h } ,$ depending on when "
    "cultures appeared sufficiently turbid for DNA extraction."
)
Q_SCHMIDT_FITNESS = (
    "Strain fitness is defined as the normalized $\\mathsf { l o g } _ { 2 }$ ratio of "
    "the barcode reads in the experimental sample to the barcode reads in the time zero "
    "sample."
)
Q_SCHMIDT_DEFER = (
    "A more detailed explanation of fitness score calculations can be found in Wetmore "
    "et al. (143)."
)
Q_SCHMIDT_DUPLICATE = (
    "Experiments were conducted in biological duplicates, and the fitness data are "
    "publicly available at http://fit.genomics.lbl.gov."
)
Q_BORCHERT23_CONDITIONS = (
    "M9 minimal medium with $2 0 \\mathrm { m M }$ glucose and supplemented with either "
    "nothing, 60 mM 4-coumarate, $6 0 ~ \\mathrm { m M }$ ferulate, $6 0 ~ \\mathrm { m M "
    "}$ 4-hydroxybenzoate, $6 0 ~ \\mathrm { m M }$ vanillate"
)
Q_BORCHERT23_PCA_DAY = (
    "The protocatechuate enrichment experiment (with corresponding $\\mathbf { M } 9 + 2 "
    "0 $ mM glucose alone enrichment) was performed on a separate day from all other "
    "experiments."
)
Q_BORCHERT23_M9 = (
    "Modified M9 $\\left( 6 . 7 8 \\ g / \\mathrm { L } \\right)$ ${ \\mathrm { N a } } _ { "
    "2 } { \\mathrm { H P O } } _ { 4 }$ $3 . 0 0 { \\ g } / { \\mathrm { L } }$ $\\mathrm { "
    "K _ { 2 } H P O _ { 4 } }$ $0 . 5 0 g / \\mathrm { L }$ NaCl, $1 . 6 6 ~ \\mathrm { g "
    "/ L N H _ { 4 } C l }$ $0 . 2 4 \\mathrm { g } / \\mathrm { L M g } \\mathrm { S O } _ "
    "{ 4 }$ ,0.01 $g / \\mathrm { L }$ $\\mathrm { C a C l _ { 2 } } ,$ and $\\mathbf { 0 "
    ". 0 0 2  g / L F e S O _ { 4 } }$ supplemented with the indicated carbon source(s) "
    "was used as minimal medium."
)
Q_BORCHERT23_FITNESS = (
    "Transposon insertion counts were not trimmed from gene ends for fitness "
    "calculations. Gene fitness was calculated as the weighted average of the strain "
    "fitness for all transposon insertions at that locus and normalized by subtracting "
    "the median unnormalized fitness within a 251 gene sliding window."
)
Q_BORCHERT23_TRIPLICATE = (
    "Gene data were excluded if fitness values were not obtained for all three "
    "biological replicates from both conditions."
)
Q_BORCHERT23_SRA = (
    "Sequencing data (fastq files) were deposited at the NCBI Sequence Read Archive "
    "(SRA) with accession number SRP385031."
)
Q_BORCHERT23_LIBRARY = (
    "Experiments employed a KT2440 library (ML-5) harboring randomly-barcoded mariner "
    "(Tc1) transposon insertions, as described previously (Rand et al., 2017)."
)


def _sv(
    source: Provenance, value: Any, quote: str, note: str | None = None
) -> SourcedValue:
    """A value quoted from one pinned artifact."""
    return SourcedValue(value=value, provenance=source, quote=quote, note=note)


#: Every value this loader hardcodes, each with its verbatim quote and pinned sha256.
SOURCED_VALUES: dict[str, SourcedValue] = {
    "data_location": _sv(_PAPER, DATA_REPOSITORY, Q_DATA_LOCATION),
    "n_samples_total": _sv(_PAPER, N_SAMPLES, Q_COMPENDIUM),
    "fitness_browser_share": _sv(
        _PAPER,
        {"fitness_browser": 254, "beckham_group": 78},
        Q_FB_SHARE,
        note="the 78 are the release's sets set100 and set101 (person Andrew Borchert / "
        "Alissa Bleem, mutantLibrary Putida_ML5_JBEI)",
    ),
    "n_genes": _sv(_PAPER, N_GENES, Q_FINAL),
    "gene_filter": _sv(
        _PAPER,
        "a gene without fitness in every one of the 332 samples was removed",
        Q_GENE_FILTER,
        note="so genes with a condition-specific value only are absent from the release",
    ),
    "replicates_not_averaged": _sv(
        _PAPER,
        "one record per sample; replicate samples are separate records",
        Q_NOT_AVERAGED,
    ),
    "strain": _sv(
        _PAPER,
        "KT2440",
        Q_LIBRARY,
        note="the reference strain and the library; the release's mutantLibrary column "
        "names Putida_ML5 and its regrown JBEI-1 stock Putida_ML5_JBEI",
    ),
    "jbei1_is_putida_ml5": _sv(
        _THOMPSON2020,
        "JBEI-1 is the Putida_ML5 library of ref (18), regrown",
        Q_THOMPSON_JBEI1,
    ),
    "transposon": _sv(_BORCHERT2023, "mariner (Tc1)", Q_BORCHERT23_LIBRARY),
    "transposon_corroboration": _sv(_PAPER, "mariner", Q_MARINER),
    "fitness_definition": _sv(
        _WETMORE2015, "log2 ratio vs time zero", Q_WETMORE_ROUGHLY
    ),
    "fitness_pipeline": _sv(_WETMORE2015, "FEBA.R", Q_WETMORE_FEBA),
    "central_10_90": _sv(_WETMORE2015, (0.1, 0.9), Q_WETMORE_CENTRAL),
    "typical_gene_zero": _sv(
        _WETMORE2015,
        0.0,
        Q_WETMORE_TYPICAL_ZERO,
        note="the reference phenotype of every record is therefore 0.0",
    ),
    "t_statistic": _sv(
        _WETMORE2015,
        "t = f / sqrt(sigma^2 + max(Ve, Vn))",
        Q_WETMORE_T_INTRO,
        note="a test statistic, not an uncertainty; the standard error is recovered as "
        "sqrt((f / t)^2 - sigma^2)",
    ),
    "t_sigma": _sv(_WETMORE2015, 0.1, Q_WETMORE_SIGMA),
    "standard_error_definition": _sv(
        _PRICE2018,
        "max(se_obs, se_naive)",
        Q_PRICE_STANDARD_ERROR,
        note="the Fitness Browser pipeline's standard error; the Price 2018 E. coli "
        "loader measured t = f / sqrt(0.1^2 + max(se_obs, se_naive)^2) to 6.4e-13 over "
        "613,818 released values, which is Wetmore's t with V = se^2",
    ),
    "thompson_fitness": _sv(
        _THOMPSON2020, "log2 ratio vs time zero", Q_THOMPSON_FITNESS
    ),
    "thompson_t": _sv(_THOMPSON2020, "fitness over its variance", Q_THOMPSON_T),
    "thompson_defers_to_wetmore": _sv(_THOMPSON2020, "Wetmore 2015", Q_THOMPSON_DEFER),
    "thompson_duplicate": _sv(_THOMPSON2020, 2, Q_THOMPSON_DUPLICATE),
    "thompson_conditions": _sv(
        _THOMPSON2020, {"fatty_acids": 13, "alcohols": 10}, Q_THOMPSON_CONDITIONS
    ),
    "thompson_alcohols": _sv(
        _THOMPSON2020,
        [
            "ethanol",
            "butanol",
            "pentanol",
            "1,2-propanediol",
            "1,3-butanediol",
            "1,4-butanediol",
            "1,5-pentanediol",
            "isopentanol",
            "isoprenol",
            "2-methyl-1-butanol",
        ],
        Q_THOMPSON_ALCOHOLS,
    ),
    "thompson_propionate": _sv(_THOMPSON2020, "propionate (C3)", Q_THOMPSON_PROPIONATE),
    "thompson_protocol": _sv(
        _THOMPSON2020,
        {"carbon_mM": 10, "vessel": "test tube", "shaking_rpm": 200},
        Q_THOMPSON_RBTNSEQ,
        note="set12's Thompson conditions are labeled '96 deep-well microplate', 700 rpm "
        "and 1% DMSO in the release, which this protocol does not describe",
    ),
    "thompson_modified_mops": _sv(
        _THOMPSON2020,
        "modified MOPS, 'when indicated'",
        Q_THOMPSON_MODIFIED_MOPS,
        note="its trace metals are ten times Neidhardt's; the release labels the "
        "samples with the Fitness Browser medium 'MOPS minimal media_noCarbon', which "
        "this loader follows (MOPS_MINIMAL)",
    ),
    "schmidt_conditions": _sv(
        _SCHMIDT2022,
        {"nitrogen_compounds": 52, "amino_acid_dropouts": 19, "conditions": 71},
        Q_SCHMIDT_CONDITIONS,
    ),
    "schmidt_dropout": _sv(
        _SCHMIDT2022, "the 20 proteinogenic amino acids minus one", Q_SCHMIDT_DROPOUT
    ),
    "schmidt_medium": _sv(_SCHMIDT2022, "glucose, nitrogen varied", Q_SCHMIDT_MEDIUM),
    "schmidt_barseq": _sv(
        _SCHMIDT2022, "nitrogen-free MOPS + 10 mM nitrogen source", Q_SCHMIDT_BARSEQ
    ),
    "schmidt_duration": _sv(_SCHMIDT2022, (24, 72), Q_SCHMIDT_DURATION),
    "schmidt_fitness": _sv(_SCHMIDT2022, "log2 ratio vs time zero", Q_SCHMIDT_FITNESS),
    "schmidt_defers_to_wetmore": _sv(_SCHMIDT2022, "Wetmore 2015", Q_SCHMIDT_DEFER),
    "schmidt_duplicate": _sv(_SCHMIDT2022, 2, Q_SCHMIDT_DUPLICATE),
    "borchert23_conditions": _sv(
        _BORCHERT2023,
        "M9 + 20 mM glucose with one of 11 stressors or nothing",
        Q_BORCHERT23_CONDITIONS,
    ),
    "borchert23_protocatechuate_day": _sv(
        _BORCHERT2023,
        "protocatechuate with its own glucose control",
        Q_BORCHERT23_PCA_DAY,
    ),
    "borchert23_m9": _sv(_BORCHERT2023, "modified M9", Q_BORCHERT23_M9),
    "borchert23_fitness": _sv(
        _BORCHERT2023, "untrimmed, 251-gene window median", Q_BORCHERT23_FITNESS
    ),
    "borchert23_triplicate": _sv(_BORCHERT2023, 3, Q_BORCHERT23_TRIPLICATE),
    "borchert23_sra": _sv(_BORCHERT2023, "SRP385031", Q_BORCHERT23_SRA),
    "conditions_179": _sv(_PAPER, 179, Q_ABSTRACT_CONDITIONS),
    "sources_cited": _sv(_PAPER, "refs 20-23, 37, 38", Q_SOURCES_CITED),
}

# --------------------------------------------------------------------------- #
# Source studies (checklist item 7): which paper first reported each sample
# --------------------------------------------------------------------------- #
SourceStudyKey = Literal["thompson2020", "schmidt2022", "borchert2023", "borchert2024"]


class SourceStudy(BaseModel):
    """A paper a sample of the compendium is attributed to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: SourceStudyKey
    citation: str
    doi: str
    citation_key: str = Field(
        description="the mirrored library key the evidence quotes"
    )
    fifty_rank: int = Field(description="rank of the paper's row among the fifty")

    @property
    def publication(self) -> Publication:
        """The ``Publication`` every record attributed to this study stores."""
        return Publication(doi=self.doi, doi_url=f"https://doi.org/{self.doi}")


SOURCE_STUDIES: dict[SourceStudyKey, SourceStudy] = {
    "thompson2020": SourceStudy(
        key="thompson2020",
        citation="Thompson MG et al. 2020, Appl Environ Microbiol 86:e01665-20",
        doi="10.1128/AEM.01665-20",
        citation_key=THOMPSON2020_KEY,
        fifty_rank=28,
    ),
    "schmidt2022": SourceStudy(
        key="schmidt2022",
        citation="Schmidt M et al. 2022, Appl Environ Microbiol 88:e0243021",
        doi="10.1128/aem.02430-21",
        citation_key=SCHMIDT2022_KEY,
        fifty_rank=24,
    ),
    "borchert2023": SourceStudy(
        key="borchert2023",
        citation="Borchert AJ, Bleem A, Beckham GT 2023, Metab Eng 77:208-218",
        doi="10.1016/j.ymben.2023.04.007",
        citation_key=BORCHERT2023_KEY,
        fifty_rank=29,
    ),
    "borchert2024": SourceStudy(
        key="borchert2024",
        citation="Borchert AJ et al. 2024, mSystems 9:e00942-23 (the compendium release)",
        doi=DOI,
        citation_key=CITATION_KEY,
        fifty_rank=8,
    ),
}

AttributionBasis = Literal["source_study_quote", "compendium_release"]


class SampleAttribution(BaseModel):
    """Which paper first reported one sample, and on what evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exp_name: str
    set_name: str
    study: SourceStudyKey
    basis: AttributionBasis
    evidence: tuple[str, ...] = Field(
        description="SOURCED_VALUES keys whose quotes name this sample's condition"
    )
    hypothesis: str | None = Field(
        default=None,
        description="an UNTESTED candidate source for a compendium-only sample, from "
        "the person, set and condition columns; never stored on a record",
    )
    note: str | None = None


#: The five Thompson 2020 conditions that sit in set12 (the deep-well DMSO set) rather
#: than in its own sets 15 and 16.
THOMPSON2020_SET12_CONDITIONS: frozenset[str] = frozenset(
    {
        "Sodium propionate",
        "1,2-Propanediol",
        "1,4-Butanediol",
        "1,5-Pentanediol",
        "2-methyl-1-butanol",
    }
)
#: UNTESTED candidates for the compendium-only samples, keyed by set (the note's table).
#: Each names a paper Borchert 2024 cites or the expansion document lists; none of them
#: is mirrored, so none becomes a record's publication.
HYPOTHESES: dict[str, str] = {
    "set1": "unattributed (person 'Kelly', Putida_ML5, 2020)",
    "set5": "Rand 2017 (doi 10.1038/s41564-017-0028-z, ref 88): levulinic acid on the "
    "Putida_ML5 library",
    "set6": "Thompson 2019 valerolactam (doi 10.1016/j.mec.2019.e00098) for the "
    "2-piperidinone sample",
    "set7": "Thompson 2019 lysine (doi 10.1128/mBio.02577-18, ref 20) and Thompson 2019 "
    "valerolactam",
    "set8": "Eng 2021 (doi 10.1016/j.ymben.2021.04.015, ref 37), the bioreactor screen",
    "set9": "Eng 2021 (doi 10.1016/j.ymben.2021.04.015, ref 37), the bioreactor screen",
    "set10": "unattributed (the 20-amino-acid control and six sole carbon sources)",
    "set12": "Incha 2020 (doi 10.1016/j.mec.2019.e00119, ref 21) for the aromatics",
    "set29": "unattributed (D-galacturonic acid as carbon source)",
    "set100": "unattributed (200 mM cis,cis-muconate; not in Borchert 2023's Methods)",
    "set101": "Borchert 2022 (doi 10.1021/acssynbio.2c00119, ref 38) or the Borchert "
    "2024 deposit PRJNA1011287",
}


# --------------------------------------------------------------------------- #
# The release
# --------------------------------------------------------------------------- #
class SampleMetadata(BaseModel):
    """One row of the release's ``metadata`` sheet (the fields the loader reads)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exp_name: str
    set_name: str
    exp_desc: str
    exp_group: str
    mutant_library: str
    person: str
    media: str
    temperature: float
    aerobic: str
    total_rep: int
    rep: int
    condition_1: str | None
    units_1: str | None
    concentration_1: float | None
    condition_2: str | None
    units_2: str | None
    concentration_2: float | None

    @property
    def is_beckham(self) -> bool:
        """A Beckham-group sample (sets 100 and 101), not from the Fitness Browser."""
        return self.set_name in BECKHAM_SETS

    @property
    def column(self) -> str:
        """The sample's header in the two value sheets: ``<expName> <expDesc>``."""
        return f"{self.exp_name} {self.exp_desc}"


#: The release's sets generated by the Beckham group (the paper's "data sets 100 and 101").
BECKHAM_SETS: frozenset[str] = frozenset({"set100", "set101"})


class Release(BaseModel):
    """The parsed release: sample metadata, gene rows and the two value matrices."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    samples: tuple[SampleMetadata, ...]
    genes: tuple[str, ...] = Field(description="locusId, in sheet order")
    gene_names: tuple[str | None, ...] = Field(description="geneName, in sheet order")
    fitness: npt.NDArray[np.float64] = Field(description="genes x samples")
    t: npt.NDArray[np.float64] = Field(description="genes x samples")


def _cell(value: Any) -> str | None:
    """A text cell, ``None`` when empty."""
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _float(value: Any) -> float | None:
    """A numeric cell, ``None`` when empty."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return float(value)


def read_release(path: str | Path) -> Release:
    """Parse ``fModule_Metadata.xlsx``, refusing any layout other than the pinned one.

    Both value sheets must list the same genes in the same order and one column per
    metadata row, headed ``<expName> <expDesc>`` in the metadata order; every value is a
    number (the release has no missing cell).
    """
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if workbook.sheetnames != [METADATA_SHEET, FITNESS_SHEET, T_SHEET]:
        raise ValueError(f"unexpected sheets {workbook.sheetnames}")
    meta_rows = list(workbook[METADATA_SHEET].iter_rows(values_only=True))
    if tuple(meta_rows[0]) != METADATA_COLUMNS:
        raise ValueError(f"unexpected metadata header {meta_rows[0]}")
    samples: list[SampleMetadata] = []
    for row in meta_rows[1:]:
        cell = dict(zip(METADATA_COLUMNS, row, strict=True))
        if cell["orgId"] != "Putida":
            raise ValueError(f"metadata row for orgId {cell['orgId']!r}")
        samples.append(
            SampleMetadata(
                exp_name=str(cell["expName"]),
                set_name=str(cell["set"]),
                exp_desc=str(cell["expDesc"]),
                exp_group=str(cell["expGroup"]),
                mutant_library=str(cell["mutantLibrary"]),
                person=str(cell["person"]),
                media=str(cell["media"]),
                temperature=float(cell["temperature"]),
                aerobic=str(cell["aerobic"]),
                total_rep=int(cell["total_rep"]),
                rep=int(cell["rep"]),
                condition_1=_cell(cell["condition_1"]),
                units_1=_cell(cell["units_1"]),
                concentration_1=_float(cell["concentration_1"]),
                condition_2=_cell(cell["condition_2"]),
                units_2=_cell(cell["units_2"]),
                concentration_2=_float(cell["concentration_2"]),
            )
        )
    columns = [s.column for s in samples]
    matrices: list[npt.NDArray[np.float64]] = []
    gene_rows: list[tuple[tuple[str, ...], tuple[str | None, ...]]] = []
    for sheet in (FITNESS_SHEET, T_SHEET):
        rows = list(workbook[sheet].iter_rows(values_only=True))
        header = rows[0]
        if tuple(header[: len(GENE_COLUMNS)]) != GENE_COLUMNS:
            raise ValueError(f"{sheet}: unexpected gene columns {header[:5]}")
        if list(header[len(GENE_COLUMNS) :]) != columns:
            raise ValueError(f"{sheet}: value columns are not the metadata samples")
        body = rows[1:]
        for row in body:
            if row[0] != "Putida" or row[1] != row[2]:
                raise ValueError(f"{sheet}: row {row[:3]} is not a Putida locusId row")
        gene_rows.append(
            (tuple(str(r[1]) for r in body), tuple(_cell(r[3]) for r in body))
        )
        matrices.append(
            np.array([r[len(GENE_COLUMNS) :] for r in body], dtype=np.float64)
        )
    workbook.close()
    if gene_rows[0][0] != gene_rows[1][0]:
        raise ValueError("the two value sheets list different genes")
    for name, matrix in zip((FITNESS_SHEET, T_SHEET), matrices, strict=True):
        if np.isnan(matrix).any():
            raise ValueError(f"{name} has an empty cell")
    return Release(
        samples=tuple(samples),
        genes=gene_rows[0][0],
        gene_names=gene_rows[0][1],
        fitness=matrices[0],
        t=matrices[1],
    )


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the repo-root ``.env``."""
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/borchertMachineLearningAnalysis2024``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _retrieval(
    sha256: str, retrieved_at: str, check: SourceCheck | None
) -> RetrievalRecord:
    return RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=DATA_URL,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": DATA_URL},
        sha256=sha256,
        retrieved_at=retrieved_at,
        last_check=check,
    )


def deposit_raw_mirror(
    source: str | Path | None = None,
    *,
    data_root: str | None = None,
    verify_source: bool = False,
    retrieved_at: str = RAW_RETRIEVED_AT,
) -> Path:
    """Deposit ``fModule_Metadata.xlsx`` into the raw mirror with its manifest.

    ``source`` is an already-retrieved copy; with ``source=None`` the recorded retrieval
    runs (``direct_url`` on the commit-pinned URL). The bytes must hash to
    ``DATA_SHA256``. Idempotent by sha256: an existing mirror file is kept when it
    matches and refused when it differs, never overwritten. ``verify_source`` re-runs the
    retrieval and records the comparison as the file's ``last_check``.
    """
    from torchcell.literature.retrieve import direct_url

    root = raw_mirror_dir(data_root)
    dest = root / DATA_RELPATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    staged = root / f".{DATA_FILE}.incoming"
    if source is None:
        staged.write_bytes(direct_url(DATA_URL))
        source = staged
    sha = file_sha256(source)
    if sha != DATA_SHA256:
        raise RuntimeError(f"{source} hashes to {sha}, the pin is {DATA_SHA256}")
    check: SourceCheck | None = None
    if verify_source:
        produced = hashlib.sha256(direct_url(DATA_URL)).hexdigest()
        if produced != sha:
            raise RuntimeError(
                f"{DATA_URL} now yields sha256 {produced}; the deposited bytes are {sha}"
            )
        check = SourceCheck(
            checked_at=date.today().isoformat(), produced_sha256=produced, matches=True
        )
    elif (root / "manifest.json").exists():
        # A re-deposit without a re-check keeps the last check of the same bytes.
        previous = load_manifest(data_root).files[0].retrieval
        if previous is not None and previous.sha256 == sha:
            check = previous.last_check
    if dest.exists():
        if file_sha256(dest) != sha:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(source, dest)
    if staged.exists():
        staged.unlink()
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=[
            ArtifactRecord(
                path=DATA_RELPATH,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=sha,
                source=f"{DATA_REPOSITORY}/blob/{DATA_COMMIT}/{DATA_FILE} "
                f"(git blob {DATA_GIT_BLOB})",
                retrieval=_retrieval(sha, retrieved_at, check),
            )
        ],
        si_data_sources=[DATA_REPOSITORY],
        si_expected=[
            "fModule_Metadata.xlsx: sheets metadata, fitness_measurements and "
            "T-like_statistics for the 332 samples (Data Availability)"
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    return Manifest.model_validate_json(
        (raw_mirror_dir(data_root) / "manifest.json").read_text()
    )


def manifest_sha256(manifest: Manifest, relpath: str = DATA_RELPATH) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the Borchert 2024 raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
def attribute_sample(sample: SampleMetadata) -> SampleAttribution:
    """The paper that first reported ``sample``, on quoted evidence only.

    A sample is attributed to a mirrored source study when that study's Methods or
    Results name its condition and the release's set, group and condition agree; every
    other sample is attributed to the compendium release itself, with an untested
    candidate recorded beside it.
    """
    common = {"exp_name": sample.exp_name, "set_name": sample.set_name}
    if sample.set_name in ("set15", "set16"):
        return SampleAttribution(
            **common,
            study="thompson2020",
            basis="source_study_quote",
            evidence=("thompson_conditions", "thompson_alcohols", "thompson_duplicate"),
        )
    if (
        sample.set_name == "set12"
        and sample.condition_1 in THOMPSON2020_SET12_CONDITIONS
    ):
        return SampleAttribution(
            **common,
            study="thompson2020",
            basis="source_study_quote",
            evidence=("thompson_alcohols", "thompson_propionate", "thompson_protocol"),
            note="the condition is in Thompson 2020's list; the release's protocol for "
            "set12 (deep-well plate, 1% DMSO) is not the paper's tube protocol",
        )
    if sample.set_name in ("set27", "set28", "set29") and (
        sample.exp_group == "nitrogen source"
    ):
        return SampleAttribution(
            **common,
            study="schmidt2022",
            basis="source_study_quote",
            evidence=("schmidt_conditions", "schmidt_barseq", "schmidt_duplicate"),
        )
    if sample.set_name == "set10" and (sample.condition_2 or "").startswith(
        "20AA_mix_minus_"
    ):
        return SampleAttribution(
            **common,
            study="schmidt2022",
            basis="source_study_quote",
            evidence=("schmidt_conditions", "schmidt_dropout"),
        )
    if sample.set_name == "set100" and sample.condition_2 != "cis,cis-muconate":
        return SampleAttribution(
            **common,
            study="borchert2023",
            basis="source_study_quote",
            evidence=("borchert23_conditions", "borchert23_triplicate"),
        )
    if sample.set_name == "set101" and (
        sample.condition_2 == "Protecatechuic acid"
        or (sample.condition_1 == "D-Glucose" and sample.condition_2 is None)
    ):
        return SampleAttribution(
            **common,
            study="borchert2023",
            basis="source_study_quote",
            evidence=("borchert23_conditions", "borchert23_protocatechuate_day"),
            note="the 30 mM protocatechuate enrichment and its separate-day glucose "
            "control",
        )
    return SampleAttribution(
        **common,
        study="borchert2024",
        basis="compendium_release",
        evidence=("fitness_browser_share",),
        hypothesis=HYPOTHESES[sample.set_name],
    )


# --------------------------------------------------------------------------- #
# Drops
# --------------------------------------------------------------------------- #
DROP_MEDIUM_NOT_IN_LIBRARY = "medium_not_in_media_library"
DROP_REACTOR_PROCESS = "reactor_process_not_representable"
DROP_RULES: dict[str, str] = {
    DROP_MEDIUM_NOT_IN_LIBRARY: "the release's medium 'RCH2_defined_noCarbon' has no "
    "MEDIA_LIBRARY entry and no library base it derives from, so its records would join "
    "nothing; Price 2018 Table S18 states the recipe, and the entry is proposed in the "
    "PR rather than invented here",
    DROP_REACTOR_PROCESS: "bioreactor and adaptation samples differ by dissolved-oxygen "
    "set point, batch or fed-batch feed rate, sampling time, solvent overlay and "
    "adaptation day, none of which has a field on Environment, so distinct conditions "
    "would collapse into one environment",
}
REACTOR_GROUPS: frozenset[str] = frozenset({"reactor", "reactor_pregrowth"})


def drop_reason(sample: SampleMetadata) -> str | None:
    """The drop rule a sample falls under, or ``None`` when it is kept."""
    if sample.exp_group in REACTOR_GROUPS:
        return DROP_REACTOR_PROCESS
    if sample.media == "RCH2_defined_noCarbon":
        return DROP_MEDIUM_NOT_IN_LIBRARY
    return None


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #
_GL = ConcentrationUnit.g_per_l
_SALT = MediaComponentRole.bulk_salt
_RELEASE_MEDIA_NOTE = (
    "the release's 'media' column names this medium for the sample (raw mirror "
    f"{DATA_RELPATH}, sheet {METADATA_SHEET})"
)


def _borchert23_component(
    name: str, role: MediaComponentRole, grams_per_l: float, read: str
) -> MediaComponent:
    return MediaComponent(
        compound=resolved_compound(name),
        role=role,
        concentration=Concentration(value=grams_per_l, unit=_GL),
        provenance=[_sv(_BORCHERT2023, read, Q_BORCHERT23_M9)],
    )


BORCHERT2023_M9 = Media(
    name="Modified M9 (Borchert 2023): 6.78 g/L Na2HPO4, 3.00 g/L K2HPO4, 0.50 g/L NaCl, "
    "1.66 g/L NH4Cl, 0.24 g/L MgSO4, 0.01 g/L CaCl2, 0.002 g/L FeSO4; carbon source per "
    "condition",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        _borchert23_component("disodium hydrogen phosphate", _SALT, 6.78, "6.78 g/L"),
        _borchert23_component(
            "dipotassium hydrogen phosphate", _SALT, 3.0, "3.00 g/L K2HPO4"
        ),
        _borchert23_component("sodium chloride", _SALT, 0.5, "0.50 g/L"),
        _borchert23_component(
            "ammonium chloride", MediaComponentRole.nitrogen_source, 1.66, "1.66 g/L"
        ),
        _borchert23_component("magnesium sulfate", _SALT, 0.24, "0.24 g/L"),
        _borchert23_component("calcium chloride", _SALT, 0.01, "0.01 g/L"),
        _borchert23_component(
            "iron(II) sulfate", MediaComponentRole.trace_element, 0.002, "0.002 g/L"
        ),
    ],
    dropouts=[resolved_compound("potassium dihydrogen phosphate")],
    provenance=[
        _sv(
            _BORCHERT2023,
            "modified M9, carbon source per condition",
            Q_BORCHERT23_M9,
            note="the phosphate is K2HPO4 in both the OCR and the PDF text layer, so the "
            "M9 base's KH2PO4 is recorded as replaced (a dropout). Borchert 2023 states "
            "this recipe for the 42 samples it reports; the other 36 Beckham samples "
            "carry the same release label 'M9_medium', which is the only evidence for "
            "them",
        )
    ],
)
"""The Beckham group's RB-TnSeq medium, the release's ``M9_medium`` (sets 100 and 101)."""

_AMMONIUM_CHLORIDE = resolved_compound("ammonium chloride").name
FB_MOPS_GLUCOSE_NO_NITROGEN = Media(
    name="MOPS minimal media_Glucose_noNitrogen (Fitness Browser): MOPS_MINIMAL without "
    "ammonium chloride, plus D-glucose; nitrogen source per condition",
    state="liquid",
    is_synthetic=True,
    base_medium="MOPS_MINIMAL",
    components=[
        *(c for c in MOPS_MINIMAL.components if c.compound.name != _AMMONIUM_CHLORIDE),
        MediaComponent(
            compound=resolved_compound("D-glucose"),
            role=MediaComponentRole.carbon_source,
            concentration=None,
            provenance=[_sv(_SCHMIDT2022, "glucose", Q_SCHMIDT_MEDIUM)],
            note="the amount is not stated in a mirrored file: Schmidt 2022 refers the "
            "medium to its Fig. S1, which is not mirrored",
        ),
    ],
    dropouts=[resolved_compound("ammonium chloride")],
    provenance=[
        _sv(_SCHMIDT2022, "nitrogen-free MOPS minimal medium", Q_SCHMIDT_BARSEQ),
        _sv(_SCHMIDT2022, "minimal medium with glucose", Q_SCHMIDT_MEDIUM),
    ],
)
"""The Fitness Browser's nitrogen-source medium, derived from ``MOPS_MINIMAL``."""

#: The release's medium label of every kept sample, to its medium object.
MEDIA_BY_LABEL: dict[str, Media] = {
    "MOPS minimal media_noCarbon": MOPS_MINIMAL,
    "MOPS minimal media_Glucose_noNitrogen": FB_MOPS_GLUCOSE_NO_NITROGEN,
    "M9_medium": BORCHERT2023_M9,
}
# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
#: Release condition labels repaired before compound resolution, each with its reason.
CONDITION_LABEL_FIXES: dict[str, tuple[str, str]] = {
    "\u00c3\u0178-ketoadipate": (
        "beta-ketoadipate",
        "the release stores the UTF-8 bytes of 'ß' decoded as Latin-1 ('Ã' 'Ÿ')",
    ),
    "Protecatechuic acid": (
        "Protocatechuic Acid",
        "a misspelling in set101; set12 spells the same compound 'Protocatechuic Acid'",
    ),
}
#: The release's three-letter amino-acid codes in the ``20AA_mix_minus_<X>`` labels.
AMINO_ACIDS: dict[str, str] = {
    "Ala": "L-alanine",
    "Arg": "L-arginine",
    "Asn": "L-asparagine",
    "Asp": "L-aspartic acid",
    "Cys": "L-cysteine",
    "Gln": "L-glutamine",
    "Glu": "L-glutamic acid",
    "Gly": "glycine",
    "His": "L-histidine",
    "Ile": "L-isoleucine",
    "Leu": "L-leucine",
    "Lys": "L-lysine",
    "Met": "L-methionine",
    "Phe": "L-phenylalanine",
    "Pro": "L-proline",
    "Ser": "L-serine",
    "Thr": "L-threonine",
    "Trp": "L-tryptophan",
    "Tyr": "L-tyrosine",
    "Val": "L-valine",
}
AMINO_ACID_MIX = "20AA_mix"
_UNITS: dict[str, ConcentrationUnit] = {
    "mM": ConcentrationUnit.millimolar,
    "vol%": ConcentrationUnit.percent_v_v,
}
_FACTOR_BY_GROUP: dict[str, PhysicalFactor] = {
    "carbon source": PhysicalFactor.carbon_source,
    "nitrogen source": PhysicalFactor.nitrogen_source,
}


def condition_compound(label: str) -> Compound:
    """The compound a release condition label names, through the shared identity table."""
    fixed = CONDITION_LABEL_FIXES.get(label, (label.strip(), ""))[0]
    return resolved_compound(fixed)


def amino_acid_mix() -> Compound:
    """The release's 20-amino-acid supplement: a mixture, so it has no InChIKey."""
    return Compound(
        name=AMINO_ACID_MIX,
        provenance_gaps=[
            ProvenanceGap(
                field="inchikey",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="a mixture of the 20 proteinogenic amino acids has no single "
                "InChIKey; its per-amino-acid amounts are not in a mirrored file",
            )
        ],
    )


def _dose(value: float | None, units: str | None) -> Concentration:
    if value is None or units not in _UNITS:
        raise ValueError(f"unsupported dose {value!r} {units!r}")
    return Concentration(value=value, unit=_UNITS[units])


def _duration_gaps() -> list[ProvenanceGap]:
    return [
        ProvenanceGap(
            field=name,
            reason=ProvenanceGapReason.not_reported_by_primary,
            looked_in=_RELEASE,
            note="the release has no growth-time column; Schmidt 2022 harvested at 24 "
            "to 72 h by turbidity and Borchert 2023 at OD600 1.0",
        )
        for name in ("duration_hours", "duration_generations")
    ]


def build_environment(sample: SampleMetadata) -> Environment:
    """The sample's medium, temperature and condition variables.

    ``condition_1`` is the varied carbon or nitrogen source (an
    ``EnvironmentPhysicalPerturbation`` with the compound as agent). ``condition_2`` is
    a second added species: a stressor on fixed glucose (mM), the DMSO vehicle (vol%), or
    the 20-amino-acid supplement (``X``), whose ``_minus_<X>`` suffix is a
    ``nutrient_dropout`` of that amino acid.
    """
    if sample.aerobic != "Aerobic":
        raise ValueError(f"{sample.exp_name}: aerobic={sample.aerobic!r}")
    if sample.condition_1 is None:
        raise ValueError(f"{sample.exp_name}: no condition_1")
    perturbations: list[EnvironmentPerturbationType] = [
        EnvironmentPhysicalPerturbation(
            factor=_FACTOR_BY_GROUP[sample.exp_group],
            agent=condition_compound(sample.condition_1),
            magnitude=_dose(sample.concentration_1, sample.units_1),
        )
    ]
    second = sample.condition_2
    if second is not None and sample.units_2 == "X":
        if not second.startswith(AMINO_ACID_MIX):
            raise ValueError(f"{sample.exp_name}: 'X' units on {second!r}")
        perturbations.append(
            SmallMoleculePerturbation(
                compound=amino_acid_mix(),
                concentration=Concentration(basis=DoseBasis.fixed),
                description=f"the release's 20-amino-acid mix at "
                f"{sample.concentration_2:g} X (no ConcentrationUnit for 'X')",
            )
        )
        dropped = second.removeprefix(AMINO_ACID_MIX)
        if dropped:
            perturbations.append(
                EnvironmentPhysicalPerturbation(
                    factor=PhysicalFactor.nutrient_dropout,
                    agent=resolved_compound(
                        AMINO_ACIDS[dropped.removeprefix("_minus_")]
                    ),
                )
            )
    elif second is not None:
        perturbations.append(
            SmallMoleculePerturbation(
                compound=condition_compound(second),
                concentration=_dose(sample.concentration_2, sample.units_2),
            )
        )
    return Environment(
        media=MEDIA_BY_LABEL[sample.media],
        temperature=Temperature(value=sample.temperature),
        perturbations=perturbations,
        aerobicity="aerobic",
        provenance_gaps=_duration_gaps(),
    )


# --------------------------------------------------------------------------- #
# Phenotype
# --------------------------------------------------------------------------- #
#: Wetmore's normalization constant in the t denominator ("we use 0.1").
T_SIGMA = 0.1
#: Half a unit in the last place of the release's three-decimal f and t.
RELEASE_HALF_ULP = 0.0005
#: Largest relative error the rounding may put on the derived standard error for it to
#: be stored (first order, propagated through sqrt(q^2 - sigma^2)).
SE_ROUNDING_TOLERANCE = 0.05
UNITS_FITNESS_BROWSER = (
    "RB-TnSeq gene fitness (Fitness Browser, Wetmore 2015): log2(sample/time-zero) "
    "barcode ratio, weighted mean over central 10-90% insertions, typical gene = 0; "
    "se = sqrt((fitness/t)^2 - 0.1^2) = max(se_obs, se_naive)"
)
UNITS_BECKHAM = (
    "RB-TnSeq gene fitness (Beckham group, Borchert 2023): log2(sample/time-zero) "
    "barcode ratio, weighted mean over all insertions, 251-gene median = 0"
)
UNITS_REFERENCE = "the typical gene of this sample (gene fitness is normalized to 0)"


def _rounding_bound(fitness: float, t: float) -> float:
    """First-order relative error of ``|f / t|`` from rounding f and t to 3 decimals."""
    return RELEASE_HALF_ULP / abs(fitness) + RELEASE_HALF_ULP / abs(t)


def derived_se(fitness: float, t: float) -> float | None:
    """The standard error ``max(se_obs, se_naive) = sqrt((f / t)^2 - 0.1^2)``.

    ``None`` when f or t is 0, when ``|f / t|`` sits at or below the 0.1 floor (only
    rounding puts it there), or when the rounding error of ``q = |f / t|``, propagated
    through the subtraction as ``bound * q^2 / (q^2 - 0.1^2)``, exceeds the tolerance.
    """
    if fitness == 0.0 or t == 0.0:
        return None
    q = abs(fitness / t)
    variance = q * q - T_SIGMA * T_SIGMA
    if variance <= 0.0:
        return None
    if _rounding_bound(fitness, t) * q * q / variance > SE_ROUNDING_TOLERANCE:
        return None
    return math.sqrt(variance)


def floor_violations(
    fitness: npt.NDArray[np.float64], t: npt.NDArray[np.float64]
) -> int:
    """How many nonzero (f, t) pairs put ``|f / t|`` below 0.1 beyond their rounding.

    The identity ``t = f / sqrt(0.1^2 + se^2)`` forces ``|f / t| >= 0.1``; a pair below
    it even at its rounding's upper bound cannot come from that statistic.
    """
    nonzero = (fitness != 0) & (t != 0)
    f = np.abs(fitness[nonzero])
    s = np.abs(t[nonzero])
    upper = (f / s) * (1.0 + RELEASE_HALF_ULP / f + RELEASE_HALF_ULP / s)
    return int((upper < T_SIGMA).sum())


_GAP_UNCERTAINTY = ProvenanceGap(
    field="environment_response_uncertainty",
    reason=ProvenanceGapReason.not_reported_by_primary,
)
_GAP_SE = ProvenanceGap(
    field="environment_response_se", reason=ProvenanceGapReason.not_reported_by_primary
)
_GAP_N = ProvenanceGap(
    field="n_samples", reason=ProvenanceGapReason.not_carried_by_curation
)
_GAP_UNIT = ProvenanceGap(
    field="sample_unit", reason=ProvenanceGapReason.not_carried_by_curation
)


def build_phenotype(
    fitness: float, t: float, sample: SampleMetadata
) -> EnvironmentResponsePhenotype:
    """One gene's fitness in one sample.

    The verbatim uncertainty is always a gap (the release carries a t statistic, not an
    uncertainty). ``n_samples`` and ``sample_unit`` are gaps because the strains averaged
    into a gene fitness are counted upstream and dropped by the compendium.
    """
    se = None if sample.is_beckham else derived_se(fitness, t)
    gaps = [_GAP_UNCERTAINTY, _GAP_N, _GAP_UNIT]
    if se is None:
        gaps.append(_GAP_SE)
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=fitness,
        environment_response_se=se,
        screen_id=sample.exp_name,
        units=UNITS_BECKHAM if sample.is_beckham else UNITS_FITNESS_BROWSER,
        provenance_gaps=gaps,
    )


def build_reference(
    dataset_name: str,
    sample: SampleMetadata,
    environment: Environment,
    genome_reference: AssemblyReferenceGenome,
) -> BacterialEnvironmentResponseExperimentReference:
    """The typical gene of the same sample: fitness 0 by the normalization."""
    return BacterialEnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=environment,
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            screen_id=sample.exp_name,
            units=UNITS_REFERENCE,
        ),
    )


# --------------------------------------------------------------------------- #
# Genotype
# --------------------------------------------------------------------------- #
TRANSPOSON = "mariner (Tc1)"


def perturbed_gene_names(
    genome: PPutidaKT2440Genome, tags: list[str]
) -> dict[str, str]:
    """Each locus tag's GenBank gene symbol when it names that locus alone, else the tag.

    A symbol is used only when the genome's resolver maps it back to the same tag as a
    current gene, so the stored common name always resolves to the stored locus.
    """
    names: dict[str, str] = {}
    for tag in tags:
        symbol = genome.genbank.loci[tag].symbol
        if symbol:
            resolution = genome.resolve_gene_name(symbol)
            if resolution.systematic_name == tag and resolution.status in (
                GeneNameStatus.CURRENT,
                GeneNameStatus.RENAMED,
            ):
                names[tag] = symbol
                continue
        names[tag] = tag
    return names


def build_genotype(tag: str, name: str, library: str) -> Genotype:
    """A transposon insertion in one gene of one mutant library (gene-level aggregate)."""
    return Genotype(
        perturbations=[
            TransposonInsertionPerturbation(
                systematic_gene_name=tag,
                perturbed_gene_name=name,
                gene_namespace="pputida_kt2440_locus_tag",
                transposon=TRANSPOSON,
                library_pool=library,
            )
        ]
    )


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
class StudyCoverage(BaseModel):
    """How many samples and conditions of one source study the release carries."""

    model_config = ConfigDict(extra="forbid")

    study: SourceStudyKey
    doi: str
    fifty_rank: int
    n_samples: int
    n_conditions: int
    n_kept_samples: int
    sets: list[str]


def condition_key(sample: SampleMetadata) -> tuple[Any, ...]:
    """A sample's condition from its typed columns, not its free-text description.

    ``expDesc`` spells one condition two ways in set101 (a doubled space before
    ``(C)``), so the condition is the medium plus both condition columns.
    """
    return (
        sample.media,
        sample.condition_1,
        sample.concentration_1,
        sample.condition_2,
        sample.concentration_2,
    )


def coverage(
    samples: tuple[SampleMetadata, ...], attributions: dict[str, SampleAttribution]
) -> list[StudyCoverage]:
    """The superset finding: samples and distinct conditions per source study."""
    out: list[StudyCoverage] = []
    for key, study in SOURCE_STUDIES.items():
        members = [s for s in samples if attributions[s.exp_name].study == key]
        out.append(
            StudyCoverage(
                study=key,
                doi=study.doi,
                fifty_rank=study.fifty_rank,
                n_samples=len(members),
                n_conditions=len({condition_key(s) for s in members}),
                n_kept_samples=sum(1 for s in members if drop_reason(s) is None),
                sets=sorted({s.set_name for s in members}),
            )
        )
    return out


def t_sign_agreement(release: Release) -> dict[str, dict[str, float]]:
    """Per set: the fraction of nonzero (f, t) pairs whose signs disagree, and the
    median per-sample Pearson correlation of f with t.

    Wetmore's t has the sign of its fitness by construction, so a set whose pairs
    disagree is carrying another statistic.
    """
    by_set: dict[str, list[int]] = {}
    for index, sample in enumerate(release.samples):
        by_set.setdefault(sample.set_name, []).append(index)
    out: dict[str, dict[str, float]] = {}
    for set_name, columns in sorted(by_set.items()):
        f = release.fitness[:, columns]
        t = release.t[:, columns]
        nonzero = (f != 0) & (t != 0)
        disagree = nonzero & (np.sign(f) != np.sign(t))
        correlations = [
            float(np.corrcoef(release.fitness[:, c], release.t[:, c])[0, 1])
            for c in columns
        ]
        out[set_name] = {
            "sign_disagreement": float(disagree.sum() / nonzero.sum()),
            "median_corr_f_t": float(np.median(correlations)),
            "floor_violations": float(floor_violations(f, t)),
        }
    return out


def check_identity_floor(release: Release) -> int:
    """Re-check ``|f / t| >= 0.1`` on every kept Fitness Browser value; refuse a breach.

    Returns the number of nonzero pairs checked. A breach beyond rounding means the
    sample's t is not ``f / sqrt(0.1^2 + se^2)``, so no standard error may be derived
    from it.
    """
    columns = [
        column
        for column, sample in enumerate(release.samples)
        if drop_reason(sample) is None and not sample.is_beckham
    ]
    f = release.fitness[:, columns]
    t = release.t[:, columns]
    breaches = floor_violations(f, t)
    if breaches:
        raise ValueError(
            f"{breaches} Fitness Browser (f, t) pairs fall below |f/t| = {T_SIGMA} "
            "beyond rounding"
        )
    return int(((f != 0) & (t != 0)).sum())


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class RbTnseqBorchert2024Dataset(ExperimentDataset):
    """KT2440 RB-TnSeq gene fitness, 4,732 genes x the kept samples of 332."""

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"

    def __init__(
        self,
        root: str = "data/torchcell/rbtnseq_borchert2024",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
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
        """The one release file this dataset reads."""
        return [DATA_FILE]

    def download(self) -> None:
        """Link the mirror's release file into ``raw/`` and verify its sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        recorded = manifest_sha256(manifest)
        check_manifest_pin(DATA_RELPATH, recorded, DATA_SHA256)
        src = raw_mirror_dir(data_root) / DATA_RELPATH
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(src, osp.join(self.raw_dir, DATA_FILE), recorded)

    def _genome(self) -> PPutidaKT2440Genome:
        if (
            self.pputida_genome is None
        ):  # a direct run; the build entry points inject it
            self.pputida_genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
        return self.pputida_genome

    def _pointer(self, obj: Any, hint: str, itxn: Any) -> Any:
        """``obj``'s dump, interned exactly as ``_intern_record`` interns it."""
        holder = {"value": obj.model_dump()}
        self._maybe_intern(holder, "value", obj, hint, itxn)
        return holder["value"]

    @post_process
    def process(self) -> None:
        """Build one record per (gene, kept sample)."""
        verify_raw_files(self.raw_dir, {DATA_FILE: DATA_SHA256})
        release = read_release(osp.join(self.raw_dir, DATA_FILE))
        if (len(release.genes), len(release.samples)) != (N_GENES, N_SAMPLES):
            raise ValueError(
                f"release is {len(release.genes)} x {len(release.samples)}, the paper "
                f"states {N_GENES} x {N_SAMPLES}"
            )
        genome = self._genome()
        stored, reconciliation = reconcile_locus_tags(
            genome, pd.Series(release.genes), label=self.name
        )
        reconciliation.require_resolved(0.99)
        identity_checked = check_identity_floor(release)
        tags = list(stored)
        names = perturbed_gene_names(genome, tags)
        genome_reference = assembly_reference(self.REFERENCE_STRAIN)
        attributions = {s.exp_name: attribute_sample(s) for s in release.samples}

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        genotypes: dict[tuple[str, str], dict[str, Any]] = {}
        se_counts: Counter[str] = Counter()
        idx = 0
        for column, sample in enumerate(tqdm(release.samples, desc=self.name)):
            if drop_reason(sample) is not None:
                continue
            environment = build_environment(sample)
            reference = build_reference(
                self.name, sample, environment, genome_reference
            )
            publication = SOURCE_STUDIES[
                attributions[sample.exp_name].study
            ].publication
            with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
                env_ptr = self._pointer(environment, environment.media.name, itxn)
                ref_ptr = self._pointer(reference, reference.dataset_name, itxn)
                pub_ptr = self._pointer(publication, "publication", itxn)
                for row, tag in enumerate(tags):
                    key = (tag, sample.mutant_library)
                    if key not in genotypes:
                        genotypes[key] = build_genotype(
                            tag, names[tag], sample.mutant_library
                        ).model_dump()
                    fitness = float(release.fitness[row, column])
                    phenotype = build_phenotype(
                        fitness, float(release.t[row, column]), sample
                    )
                    se_counts[
                        "beckham"
                        if sample.is_beckham
                        else (
                            "derived"
                            if phenotype.environment_response_se is not None
                            else "rounding_gap"
                        )
                    ] += 1
                    record = {
                        "experiment": {
                            "experiment_type": "bacterial_environment_response",
                            "dataset_name": self.name,
                            "genotype": genotypes[key],
                            "environment": env_ptr,
                            "phenotype": phenotype.model_dump(),
                        },
                        "reference": ref_ptr,
                        "publication": pub_ptr,
                    }
                    if row == 0:
                        self._check_record(
                            record,
                            genotypes[key],
                            environment,
                            phenotype,
                            reference,
                            publication,
                            itxn,
                        )
                    txn.put(f"{idx}".encode(), pickle.dumps(record))
                    idx += 1
        env.close()
        interned_env.close()
        self._write_reports(
            release, attributions, reconciliation, se_counts, idx, identity_checked
        )
        log.info("Wrote %d %s records to LMDB", idx, self.name)

    def _check_record(
        self,
        record: dict[str, Any],
        genotype: dict[str, Any],
        environment: Environment,
        phenotype: EnvironmentResponsePhenotype,
        reference: BacterialEnvironmentResponseExperimentReference,
        publication: Publication,
        itxn: Any,
    ) -> None:
        """The fast-path record equals what ``_intern_record`` writes for the same objects."""
        experiment = BacterialEnvironmentResponseExperiment(
            dataset_name=self.name,
            genotype=Genotype.model_validate(genotype),
            environment=environment,
            phenotype=phenotype,
        )
        expected = pickle.loads(
            self._intern_record(experiment, reference, publication, itxn)
        )
        if expected != record:
            raise AssertionError(
                f"{self.name}: assembled record differs from _intern_record's"
            )

    def _write_reports(
        self,
        release: Release,
        attributions: dict[str, SampleAttribution],
        reconciliation: LocusTagReconciliation,
        se_counts: Counter[str],
        kept_records: int,
        identity_checked: int,
    ) -> None:
        """Write the drop, source-study, identifier and uncertainty reports."""
        dropped: dict[str, dict[str, Any]] = {}
        for sample in release.samples:
            rule = drop_reason(sample)
            if rule is None:
                continue
            bucket = dropped.setdefault(
                rule, {"samples": [], "n_records_not_written": 0}
            )
            bucket["samples"].append(
                f"{sample.exp_name} ({sample.media}: {sample.exp_desc})"
            )
            bucket["n_records_not_written"] += len(release.genes)
        reports: dict[str, Any] = {
            "dropped_records.json": {
                "dataset": self.name,
                "kept_records": kept_records,
                "kept_samples": kept_records // len(release.genes),
                "rules": DROP_RULES,
                "by_rule": dropped,
            },
            "source_studies.json": {
                "studies": {k: v.model_dump() for k, v in SOURCE_STUDIES.items()},
                "coverage": [
                    c.model_dump() for c in coverage(release.samples, attributions)
                ],
                "samples": [a.model_dump() for a in attributions.values()],
            },
            "locus_tag_reconciliation.json": reconciliation.model_dump(mode="json"),
            "uncertainty.json": {
                "se_rule": f"sqrt((fitness/t)^2 - {T_SIGMA}^2) when the rounding bound "
                f"(0.0005/|f| + 0.0005/|t|) * q^2 / (q^2 - {T_SIGMA}^2) <= "
                f"{SE_ROUNDING_TOLERANCE}, Fitness Browser samples only",
                "identity_floor_checked_pairs": identity_checked,
                "record_counts": dict(se_counts),
                "t_sign_agreement_by_set": t_sign_agreement(release),
            },
        }
        for name, payload in reports.items():
            with open(osp.join(self.preprocess_dir, name), "w") as handle:
                json.dump(payload, handle, indent=2)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "RbTnseqBorchert2024Dataset builds records in process()"
        )


def main() -> None:
    """Build the dataset under ``DATA_ROOT`` and print its length and first record."""
    dataset = RbTnseqBorchert2024Dataset(
        root=osp.join(_data_root(), "data/torchcell/rbtnseq_borchert2024")
    )
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
