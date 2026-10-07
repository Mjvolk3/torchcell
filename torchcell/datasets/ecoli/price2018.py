# torchcell/datasets/ecoli/price2018
# [[torchcell.datasets.ecoli.price2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/price2018
# Test file: tests/torchcell/datasets/ecoli/test_price2018.py
"""Price 2018 Fitness Browser compendium, E. coli BW25113 (``Keio``): RB-TnSeq gene fitness.

Price et al. 2018 (Nature, doi:10.1038/s41586-018-0124-0, citation key
``priceMutantPhenotypesThousands2018``) is the genome-wide mutant fitness compendium of 32
bacteria. This loader takes ONLY its E. coli BW25113 arm (``orgId`` ``Keio``, the Tn5
library KEIO_ML9): the 162 successful samples Supplementary Table 5 lists, whose gene
fitness is the compendium release's ``fit_logratios_good.tab`` (statistics version 1.0.3).
The KT2440 arm of the same compendium is the Borchert 2024 loader's.

SUPERSET (plan checklist item 7). The compendium subsumes Wetmore 2015 (rank 2): 92 of the
162 samples are Wetmore's, named by ``wetmore2015.subsumption_record().carried``, and each
record of those samples cites Wetmore 2015 as its publication. The other 70 cite Price
2018, which re-analyzes "385 successful experiments from Wetmore et al.9 and 36 successful
experiments from Melnyk et al.12" and states "The other 4,449 successful fitness assays are
described here for the first time". Table S20 attributes KEIO_ML9 to PMID 25968644
(Wetmore 2015) and lists no other E. coli library, so treating the 70 as first described
in Price 2018 is an inference from that table, not a per-sample statement. Table S5's Keio
samples come from sets set1, set2 and set6 only.

RECORDS. One record per (gene, sample): a sample is one BarSeq experiment, and replicates
are separate samples ("Samples with the same value are replicates"), so replication lies
across records and ``screen_id`` = ``Keio:<sample name>`` keeps them distinct.

* Phenotype: ``EnvironmentResponsePhenotype(measurement_type=log2_ratio,
  assay_type=pooled_competitive_growth_barcode)`` in the assembly-pinned
  ``BacterialEnvironmentResponseExperiment`` family. Gene fitness is a signed log2 ratio,
  which ``FitnessPhenotype`` would clamp at 0.
* Uncertainty: the paper's standard error, ``UncertaintyType.standard_error``. Price 2018:
  "The standard error is the maximum of two estimates", the strain-consistency estimate
  (``fit_standard_error_obs.tab``) and the read-count estimate
  (``fit_standard_error_naive.tab``). The stored value is their maximum, which is what the
  released t statistic is computed from: ``t = fit / sqrt(0.1^2 + SE^2)`` holds for every
  value of the 162 samples (Wetmore's moderated t, sigma 0.1), checked at build time. The
  naive estimate is the larger one for 35.75% of the values, so storing the
  strain-consistency column alone would overstate precision there. The t statistic itself
  has no slot on the phenotype and is not stored.
* ``n_samples`` (usable strains per gene) and ``sample_unit`` are typed gaps: the count is
  released only inside the R image (``fit.image``, field ``n``); Table S20 gives only its
  median, 12.
* Genotype: one ``TransposonInsertionPerturbation`` per gene (gene-level, so barcode and
  insertion site are ``None``), ``transposon="Tn5"``, ``library_pool="KEIO_ML9"``.
  The release names MG1655 b-numbers for this BW25113 library; each is mapped to a
  ``BW25113_`` locus tag through its one-to-one ECK pair (``eck_crosswalk``), and the
  leaf's ``description`` says on every record that the tag is DERIVED that way. 3,768 of
  the release's 3,789 genes map (one pair, b0018 to BW25113_4412, has numbers that
  differ); the 21 without a one-to-one pair are dropped and listed.
* Reference: per sample, the typical gene of that sample (fitness 0 by the
  normalization), in the sample's own environment.
* Environment: Table S5's medium through the media library (``LB`` is ``LB_LENNOX``, the
  M9 and MOPS media are Price's own Table S18 entries), its temperature, and its
  Condition_1: a carbon or nitrogen source is
  ``EnvironmentPhysicalPerturbation(factor=carbon_source | nitrogen_source)`` with the
  compound as ``agent``; a stress compound is a ``SmallMoleculePerturbation`` whose
  vehicle is a typed gap. Compounds the pinned identity table does not resolve carry the
  resolver's typed ``inchikey`` gap.

DROPPED SAMPLES (counted in ``preprocess/dropped_records.json``): the four sucrose and
D-mannitol samples the authors' 2021 note withdraws; the four ``MOPS Rich Defined
media_noCarbon`` samples (no media-library entry); the seven soft-agar motility samples
(no media-library entry for 0.3% LB Lennox agar, and the readout is a spatial cut, not
growth in a condition).

RAW DATA. Four release files are already deposited in the Wetmore 2015 raw mirror
(``fit_logratios_good.tab``, the Keio page, the compendium page, ``fit_quality.tab``) and
are consumed there by their sha256. The three this loader adds (the two standard-error
tables and ``fit_t.tab``) are deposited in
``$DATA_ROOT/torchcell-raw/priceMutantPhenotypesThousands2018/`` by
:func:`deposit_raw_mirror`; Table S5 is ``si/si3.xlsx`` of the literature mirror.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import os.path as osp
import shutil
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

import numpy as np
import numpy.typing as npt
import openpyxl
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

import torchcell.datasets.ecoli.wetmore2015 as wetmore
from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    verify_sha256,
    write_verified,
)
from torchcell.datamodels.compound_identity import (
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.media import (
    LB_LENNOX,
    M9_NOCARBON_PRICE2018,
    M9_NONITROGEN_PRICE2018,
    MOPS_MINIMAL,
)
from torchcell.datamodels.schema import (
    AssayType,
    AssemblyReferenceGenome,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    PhysicalFactor,
    Publication,
    SmallMoleculePerturbation,
    Temperature,
    TransposonInsertionPerturbation,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import (
    STRAIN_GENE_NAMESPACES,
    LocusTagReconciliation,
    LocusTagResolutionError,
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
    sha256_file,
)
from torchcell.literature.retrieve import direct_url
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import (
    EckCrosswalk,
    EcoliK12BW25113Genome,
    EcoliK12Genome,
    EcoliK12MG1655Genome,
    EcoliK12StrainName,
)
from torchcell.verification.report import Level, LevelResult, Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "priceMutantPhenotypesThousands2018"
PAPER_DOI = "10.1038/s41586-018-0124-0"
PAPER_TITLE = "Mutant phenotypes for thousands of bacterial genes of unknown function"
LIBRARY_DIR_REL = "torchcell-library"
RAW_ROOT_REL = "torchcell-raw"
RAW_DIR_REL = f"{RAW_ROOT_REL}/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "f3443cdcb2f722b5e6aa6d999f67d68a6f845eb9eb45ad0bac1cc4c24ea53e2d"
#: The Supplementary Tables workbook (Tables S1 to S22, one sheet each).
SUPPLEMENTARY_TABLES = "si/si3.xlsx"
SUPPLEMENTARY_TABLES_SHA256 = (
    "e5dbf3d5c97cfc12f49d7fd83f84bc95c16cbe963309a561fff20f442788b879"
)
TABLE_S5_SHEET = "TableS5_Experiments"
TABLE_S14_SHEET = "TableS14_RB_TnSeq_Bacteria"
TABLE_S20_SHEET = "TableS20_Mutagenesis"

#: The compendium's organism id for E. coli BW25113 (Table S5 ``orgId``).
ORG_ID: Final = "Keio"
#: The library strain; the records pin ``assembly_reference(REFERENCE_STRAIN)``.
REFERENCE_STRAIN: Final = "BW25113"
#: The library's own name (Table S20 "Mutant library name").
MUTANT_LIBRARY: Final = "KEIO_ML9"
#: The transposon, verbatim from Table S20's "Transposon" column.
TRANSPOSON: Final = "Tn5"

# --------------------------------------------------------------------------- #
# Raw files
# --------------------------------------------------------------------------- #
RAW_RETRIEVED_AT = "2026-10-07"
_BIGFIT = "https://genomics.lbl.gov/supplemental/bigfit"
_BIGFIT_KEIO = f"{_BIGFIT}/html/Keio"
SE_OBS = "data/bigfit/html/Keio/fit_standard_error_obs.tab"
SE_NAIVE = "data/bigfit/html/Keio/fit_standard_error_naive.tab"
T_STAT = "data/bigfit/html/Keio/fit_t.tab"


class RawFile(BaseModel):
    """One release file this loader deposits in its own raw mirror."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relpath: str
    url: str
    sha256: str
    purpose: str


RAW_FILES: dict[str, RawFile] = {
    raw.relpath: raw
    for raw in (
        RawFile(
            relpath=SE_OBS,
            url=f"{_BIGFIT_KEIO}/fit_standard_error_obs.tab",
            sha256="1ab5a4f016f2e6abd9f87667e2a6991360820a63785e0f4957636f267e3237ab",
            purpose="the strain-consistency standard-error estimate (the release's "
            "'se'), one of the two the paper's standard error is the maximum of",
        ),
        RawFile(
            relpath=SE_NAIVE,
            url=f"{_BIGFIT_KEIO}/fit_standard_error_naive.tab",
            sha256="b1260e6833dd17acad7612099a1ba5b9dfe09a304c311e925fa2036dad55b7ee",
            purpose="the read-count standard-error estimate (the release's 'sdNaive'), "
            "the other of the two",
        ),
        RawFile(
            relpath=T_STAT,
            url=f"{_BIGFIT_KEIO}/fit_t.tab",
            sha256="444f4cac23cd1028bf200a87a4d2d0c8842e0a0ba77f4235ca2c90cbbfa577c8",
            purpose="the released moderated t; the build checks that it equals "
            "fit / sqrt(0.1^2 + SE^2) for every stored value, which pins which standard "
            "error the authors used",
        ),
    )
}


class ReferencedRawFile(BaseModel):
    """A file of ANOTHER key's raw mirror this loader consumes there, by its sha256.

    The compendium's Keio fitness table, quality table and pages were deposited with the
    Wetmore 2015 provenance record (the same bytes this loader needs), so they are read
    from that mirror, verified against that mirror's manifest, never copied twice.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    citation_key: str
    relpath: str
    sha256: str
    purpose: str


REFERENCED_RAW_FILES: dict[str, ReferencedRawFile] = {
    relpath: ReferencedRawFile(
        citation_key=wetmore.CITATION_KEY,
        relpath=relpath,
        sha256=wetmore.RAW_FILES[relpath].sha256,
        purpose=purpose,
    )
    for relpath, purpose in (
        (
            wetmore.SUPERSET_FITNESS,
            "the stored values: gene fitness of the 162 successful Keio samples, "
            "statistics version 1.0.3",
        ),
        (wetmore.SUPERSET_PAGE, "quotes: table definitions and the statistics version"),
        (wetmore.SUPERSET_TOP_PAGE, "quote: the 2021 sucrose / D-mannitol withdrawal"),
        (
            wetmore.SUPERSET_QUALITY,
            "read by wetmore2015.subsumption_record for the 92 carried samples",
        ),
    )
}

#: ``raw/`` file name -> (mirror, path in it, sha256). The names are distinct.
RAW_FILE_NAMES: dict[str, tuple[Literal["price", "wetmore", "library"], str, str]] = {
    "fit_logratios_good.tab": (
        "wetmore",
        wetmore.SUPERSET_FITNESS,
        wetmore.RAW_FILES[wetmore.SUPERSET_FITNESS].sha256,
    ),
    "fit_standard_error_obs.tab": ("price", SE_OBS, RAW_FILES[SE_OBS].sha256),
    "fit_standard_error_naive.tab": ("price", SE_NAIVE, RAW_FILES[SE_NAIVE].sha256),
    "fit_t.tab": ("price", T_STAT, RAW_FILES[T_STAT].sha256),
    "si3.xlsx": ("library", SUPPLEMENTARY_TABLES, SUPPLEMENTARY_TABLES_SHA256),
}


# --------------------------------------------------------------------------- #
# Sourced values
# --------------------------------------------------------------------------- #
_OCR_METHOD = "MinerU OCR of the publisher PDF (torchcell-library mirror)"
_XLSX_METHOD = (
    "row rendering of the Supplementary Tables workbook: a row's non-empty cells joined "
    "by ' | ', consecutive rows by ' / ' (xlsx_text)"
)
_RELEASE_METHOD = "the authors' data-release page, retrieved verbatim (torchcell-raw)"
_FITNESS_METHODS = "Methods, 'Computation of fitness values'"


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote of the pinned Price 2018 OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method=_OCR_METHOD,
            page=page,
        ),
    )


def _tables(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a row rendering of the pinned Supplementary Tables workbook."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SUPPLEMENTARY_TABLES,
            citation_key=CITATION_KEY,
            sha256=SUPPLEMENTARY_TABLES_SHA256,
            method=_XLSX_METHOD,
            page=page,
        ),
    )


def _keio_page(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a quote of the compendium's Keio page (Wetmore raw mirror).

    The page resolves under ``$DATA_ROOT/torchcell-raw`` with the Wetmore key, where it
    was deposited, so audit these with that root.
    """
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=wetmore.SUPERSET_PAGE,
            citation_key=wetmore.CITATION_KEY,
            sha256=wetmore.RAW_FILES[wetmore.SUPERSET_PAGE].sha256,
            method=_RELEASE_METHOD,
            page=page,
            retrieved=wetmore.RAW_RETRIEVED_AT,
        ),
    )


# The strain and the library --------------------------------------------------- #
STRAIN = _tables(
    REFERENCE_STRAIN,
    "Escherichia coli BW25113 | Coli genetic stock center | Derivative of strain K12 | "
    "Coli genetic stock center 7636 | PMID 10829079",
    page="Supplementary Table 14, the E. coli row",
    note="BW25113, not MG1655; Table S5 names the organism of every Keio sample "
    "'Escherichia coli BW25113', and Wetmore 2015 (wetmore2015.STRAIN) agrees",
)
LIBRARY = _tables(
    {
        "mutant_library": MUTANT_LIBRARY,
        "transposon": TRANSPOSON,
        "unique_barcodes": 152018,
        "median_strains_per_gene": 12,
        "selection_medium": "LB",
        "selection_temperature_c": 37,
        "antibiotic_ug_per_ml": ("Kanamycin", 50),
        "reference_pmid": "25968644",
    },
    "Escherichia coli BW25113 | KEIO_ML9 | Tn5 | 152018 | 12 | electroporation | NA | NA "
    "| NA | LB | 37 | Kanamycin; 50 | PMID 25968644",
    page="Supplementary Table 20, the E. coli row",
    note="PMID 25968644 is Wetmore 2015, the library's construction paper",
)
LIBRARY_COLUMNS = _tables(
    "Table S20 columns",
    "Strain | Mutant library name | Transposon1 | Number of unique barcodes2 | Mutants "
    "used in fitness calculation per gene3 | Method of delivery",
    page="Supplementary Table 20, header row",
)
MEDIAN_STRAINS_PER_GENE = _tables(
    12,
    "3 | The median number of usable strains for calculating gene fitness per gene in "
    "each bacterium",
    page="Supplementary Table 20, note 3",
    note="a median over genes, so never an n_samples value (N_SAMPLES_GAP)",
)

# Which experiments, and from which paper --------------------------------------- #
EXPERIMENT_TABLE = _tables(
    "Table S5 lists the successful experiments",
    "This table lists all of the genome-wide fitness experiments that met our quality "
    "standards.",
    page="Supplementary Table 5, preamble",
)
EXPERIMENT_NAME = _tables(
    "orgId:name is the experiment identity",
    "name – an internal identifier for this experiment; these are unique only within "
    "each organism",
    page="Supplementary Table 5, preamble",
    note="so screen_id is 'Keio:<name>'",
)
EXPERIMENT_TEMPERATURE = _tables(
    "Temperature column, degrees Celsius",
    "Temperature – the temperature that the mutant library was grown at",
    page="Supplementary Table 5, preamble",
)
EXPERIMENT_CONDITION = _tables(
    "Condition_1 is the added compound",
    "Condition_1 – usually a compound that was added to the media",
    page="Supplementary Table 5, preamble",
)
NEW_EXPERIMENTS = _paper(
    "the samples not carried from Wetmore 2015 are first described here",
    "The other 4,449 successful fitness assays are described here for the first time.",
    page="Methods, 'Mutant fitness assays' (paper.md line 177)",
    note="beside wetmore2015.SUPERSET_INCLUDES_THIS_PAPER (385 from Wetmore, 36 from "
    "Melnyk). INFERENCE, not a per-sample statement: Table S20 gives KEIO_ML9 one "
    "reference (PMID 25968644, Wetmore 2015) and no other E. coli library, so a Keio "
    "sample not carried from Wetmore is taken as new here",
)
MEDIA_TABLE = _paper(
    "Supplementary Table 18",
    "A full list of the media used in this study and their components is given in "
    "Supplementary Table 18.",
    page="Methods, 'Media and standard culturing conditions'",
    note="the MEDIA_LIBRARY entries LB_LENNOX, M9_NOCARBON_PRICE2018, "
    "M9_NONITROGEN_PRICE2018 and MOPS_MINIMAL quote that table",
)

# Fitness and its statistics ---------------------------------------------------- #
FITNESS_DEFERRAL = _paper(
    wetmore.CITATION_KEY,
    "Fitness data was analysed as previously described9 .",
    page=_FITNESS_METHODS,
    note="ref 9 is Wetmore 2015; its statistics are wetmore2015's SourcedValues",
)
GENE_FITNESS = _paper(
    MeasurementType.log2_ratio,
    "In brief, the fitness value of each strain (an individual transposon mutant) is the "
    "normalized log2(strain barcode abundance at end of experiment/strain barcode "
    "abundance at start of experiment). The fitness value of each gene is the weighted "
    "average of the fitness of its strains.",
    page=_FITNESS_METHODS,
)
GENE_FITNESS_IS_LOG2 = _keio_page(
    MeasurementType.log2_ratio,
    "<P>Gene fitness is a log<SUB>2</SUB> ratio.",
    page="Documentation, 'Gene Fitness'",
    note="signed and centered on zero, so EnvironmentResponsePhenotype, not the clamping "
    "FitnessPhenotype",
)
SAME_STRAINS_EVERY_EXPERIMENT = _paper(
    "every gene has a value in every experiment",
    "Because we include the same set of strains in the analysis of each experiment, "
    "there are no instances where a gene has a fitness value in one experiment but not "
    "in another.",
    page=_FITNESS_METHODS,
)
STRAINS_PER_GENE_RANGE = _paper(
    (3, 26),
    "we used 3–26 mutant strains for the typical protein-coding gene in each bacterium "
    "(Supplementary Table 20)",
    page=_FITNESS_METHODS,
    note="across the 32 bacteria; E. coli's median is MEDIAN_STRAINS_PER_GENE",
)
STRESS_NOT_CORRECTED = _paper(
    "a stress value is not relative to the unstressed condition",
    "Our fitness calculations for stress experiments do not correct for the fitness "
    "values in the unstressed condition.",
    page="Methods, 'Cofitness and conserved cofitness'",
    note="so a record's 0 is the typical gene of the SAME sample (the reference), not "
    "an unstressed control",
)
STANDARD_ERROR = _paper(
    UncertaintyType.standard_error,
    "which is the gene’s fitness divided by the standard error9 . The standard error is "
    "the maximum of two estimates. The first estimate is based on the consistency of the "
    "fitness for the strains in that gene. The second estimate is based on the number "
    "of reads for the gene.",
    page="Methods, 'Computation of fitness values'",
    note="the two estimates are the release's fit_standard_error_obs.tab and "
    "fit_standard_error_naive.tab (SE_OBS_TABLE, SE_NAIVE_TABLE); the stored SE is their "
    "maximum, and the build checks the released t against it",
)
SE_OBS_TABLE = _keio_page(
    "fit_standard_error_obs.tab",
    '<LI>Or <A HREF="fit_standard_error_obs.tab">estimated standard error</A> (based on '
    "variation across strains)",
    page="Tables, 'Genes'",
)
SE_NAIVE_TABLE = _keio_page(
    "fit_standard_error_naive.tab",
    '<LI>Or <A HREF="fit_standard_error_naive.tab">naive standard error</A> (based on '
    "total counts)",
    page="Tables, 'Genes'",
)
SE_FIELD = _keio_page(
    "se",
    "<LI><B>se</B> -- an estimate of how noisy this gene's measurement is (se is short "
    "for standard error)",
    page="Documentation, 'The R image', per-gene information",
)
SD_NAIVE_FIELD = _keio_page(
    "sdNaive",
    "<LI><B>sdNaive</B> -- the best-case of how noisy this gene's measurement would be, "
    "based on the total counts",
    page="Documentation, 'The R image', per-gene information",
)
T_TABLE = _keio_page(
    "fit_t.tab",
    '<LI><A HREF="fit_t.tab">t-like test statistic</A> (based on consistency of '
    "measurements for the gene)",
    page="Tables, 'Genes'",
)
N_STRAINS_FIELD = _keio_page(
    "n",
    "<LI><B>n</B> -- number of usable strains for each gene",
    page="Documentation, 'The R image', per-gene information",
    note="released only inside the R image, never in a tab-delimited table",
)
REPLICATE_DEFINITION = _keio_page(
    "samples sharing a 'short' description are replicates",
    "<LI><B>short</B> -- shortened description. Samples with the same value are "
    "replicates.",
    page="Documentation, the quality-score table",
    note="each replicate is its own sample and its own records",
)
REPLICATE_AVERAGING = _paper(
    "the paper averages replicates for its phenotype calls",
    "We averaged fitness values from exact replicate experiments.",
    page="Methods, 'Genes with statistically significant phenotypes'",
    note="an analysis step for the paper's calls; the release keeps one column per "
    "sample, and so do the records",
)
STATISTICS_VERSIONS = _paper(
    ("1.0.3", "1.1.0", "1.1.1"),
    "we used statistics versions 1.0.3, 1.1.0, or 1.1.1 of the code",
    page="Methods, 'Data and code availability'",
    note="Keio is 1.0.3 (wetmore2015.SUPERSET_RELEASE)",
)
R_IMAGE = _paper(
    "the R image carries everything but the per-strain values",
    "The R image contains all of the information in the Fitness Browser except for the "
    "per-strain fitness values.",
    page="Methods, 'Data and code availability'",
)

#: Every SourcedValue this module defines, by name (the data tests audit each).
SOURCED_VALUES: dict[str, SourcedValue] = {
    name: value
    for name, value in globals().items()
    if isinstance(value, SourcedValue) and name.isupper()
}

#: Wetmore 2015's values this loader relies on, by name.
DEFERRED_VALUES: dict[str, SourcedValue] = {
    "STRAIN": wetmore.STRAIN,
    "T_STATISTIC": wetmore.T_STATISTIC,
    "T_STATISTIC_SIGMA": wetmore.T_STATISTIC_SIGMA,
    "NORMALIZATION": wetmore.NORMALIZATION,
    "SUPERSET_RELEASE": wetmore.SUPERSET_RELEASE,
    "SUPERSET_INCLUDES_THIS_PAPER": wetmore.SUPERSET_INCLUDES_THIS_PAPER,
    "STOCK_SOLUTION_CORRECTION": wetmore.STOCK_SOLUTION_CORRECTION,
}

N_SAMPLES_GAP = ProvenanceGap(
    field="n_samples",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=Provenance(
        source_uri=f"{_BIGFIT_KEIO}/fit_logratios_good.tab",
        citation_key=CITATION_KEY,
        sha256=wetmore.RAW_FILES[wetmore.SUPERSET_FITNESS].sha256,
        method=_RELEASE_METHOD,
        page="the tab-delimited per-gene tables carry no strain-count column",
    ),
    resolve_with=Provenance(
        source_uri=f"{_BIGFIT_KEIO}/fit.image",
        citation_key=CITATION_KEY,
        method="R image of the compendium release, field fit$n",
        page="per-gene information, 'n -- number of usable strains for each gene'",
    ),
    note="a gene fitness value averages the gene's usable insertion strains within one "
    "sample; the count is released only in the R image, and Table S20 gives its median "
    "(12). The stored SE is the larger of the two released estimates and needs no n",
)
SAMPLE_UNIT_GAP = ProvenanceGap(
    field="sample_unit",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=N_SAMPLES_GAP.looked_in,
    resolve_with=N_SAMPLES_GAP.resolve_with,
    note="travels with n_samples: one unit would be a usable insertion strain, for "
    "which SampleUnit has no member",
)
SOLVENT_GAP = ProvenanceGap(
    field="solvent",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=Provenance(
        source_uri=SUPPLEMENTARY_TABLES,
        citation_key=CITATION_KEY,
        sha256=SUPPLEMENTARY_TABLES_SHA256,
        method=_XLSX_METHOD,
        page="Supplementary Table 5 (Condition_1, Concentration_1, Units_1) and Table 4",
    ),
    note="no vehicle is stated per stress compound in the paper, Table S4 or Table S5",
)


@functools.cache
def xlsx_text(path: str) -> str:
    """Every sheet's non-empty rows, cells joined by ' | ', rows by ' / '.

    The rendering the workbook quotes are cut from, as for the xlsx quotes of
    ``media.py``; one line per sheet.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workbook = openpyxl.load_workbook(path, read_only=True)
    sheets = []
    for sheet in workbook.worksheets:
        rows = [
            " | ".join(str(cell) for cell in row if cell is not None)
            for row in sheet.iter_rows(values_only=True)
            if any(cell is not None for cell in row)
        ]
        sheets.append(" / ".join(rows))
    return "\n".join(sheets)


# --------------------------------------------------------------------------- #
# Paths, pins, retrieval
# --------------------------------------------------------------------------- #
def _data_root(data_root: str | None = None) -> str:
    """``data_root`` when given, else ``DATA_ROOT`` (the repo-root ``.env``)."""
    if data_root is not None:
        return data_root
    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/priceMutantPhenotypesThousands2018``."""
    return Path(_data_root(data_root)) / RAW_DIR_REL


def load_manifest(data_root: str | None = None) -> Manifest:
    """This loader's raw mirror ``manifest.json``."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The sha256 a manifest records for one file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the {manifest.citation_key} manifest")


def source_path(name: str, data_root: str | None = None) -> Path:
    """The mirror file behind ``raw/<name>``, after checking its manifest pin and bytes.

    The loader's own files are checked against this mirror's manifest, the referenced
    ones through ``wetmore2015.raw_path`` (the Wetmore mirror's manifest), and Table S5
    through the literature mirror's manifest.
    """
    root = _data_root(data_root)
    mirror, relpath, pin = RAW_FILE_NAMES[name]
    if mirror == "wetmore":
        return wetmore.raw_path(relpath, root)
    if mirror == "library":
        return wetmore.library_path(CITATION_KEY, relpath, pin, root)
    check_manifest_pin(relpath, manifest_sha256(load_manifest(root), relpath), pin)
    path = raw_mirror_dir(root) / relpath
    verify_sha256(path, pin)
    return path


def retrieve_raw_files(dest_dir: str | Path) -> Path:
    """Run the recorded retrieval for this loader's three files into ``dest_dir``.

    A changed upstream file raises ``RawSha256MismatchError`` and writes nothing for it,
    so drift is detected rather than deposited.
    """
    dest = Path(dest_dir)
    for raw in RAW_FILES.values():
        target = dest / raw.relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        write_verified(direct_url(raw.url), target, raw.sha256, raw.url)
    return dest


def _raw_manifest(root: Path, retrieved_at: str, created_at: str) -> Manifest:
    """The manifest the raw mirror carries, built from the deposited bytes."""
    referenced = "; ".join(
        f"{ref.relpath} (sha256 {ref.sha256}): {ref.purpose}"
        for ref in REFERENCED_RAW_FILES.values()
    )
    return Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=[
            ArtifactRecord(
                path=raw.relpath,
                role=ROLE_RAW_DATA,
                bytes=(root / raw.relpath).stat().st_size,
                sha256=raw.sha256,
                source=raw.url,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=raw.url,
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"url": raw.url},
                    sha256=raw.sha256,
                    retrieved_at=retrieved_at,
                ),
            )
            for raw in RAW_FILES.values()
        ],
        si_data_sources=[f"{_BIGFIT}/", f"{_BIGFIT_KEIO}/"],
        si_expected=[
            "consumed from the torchcell-raw/wetmoreRapidQuantificationMutant2015 "
            f"mirror, where they were deposited first: {referenced}",
            "Supplementary Table 5 is si/si3.xlsx of the literature mirror",
            "not mirrored: the R image fit.image (the only release of the per-gene "
            "strain count n), the per-strain tables and expsUsed (its metadata columns "
            "equal Table S5's for all 162 Keio samples)",
        ],
        provenance_complete=True,
        created_at=created_at,
    )


def deposit_raw_mirror(
    *,
    source_dir: str | Path,
    retrieved_at: str = RAW_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Copy the retrieved release files into the raw mirror and write its manifest.

    Every source file is checked against its pin, and every file already in the mirror
    against the same pin, BEFORE anything is written, so a refusal leaves the mirror as
    it was. Idempotent by sha256: a matching mirror file is left alone, a differing one
    raises. An existing manifest is kept when it records the same files and refused when
    it records others; it is never overwritten.
    """
    source = Path(source_dir)
    root = raw_mirror_dir(data_root)
    for raw in RAW_FILES.values():
        got = sha256_file(source / raw.relpath)
        if got != raw.sha256:
            raise RuntimeError(
                f"{source / raw.relpath} sha256 mismatch: got {got}, expected "
                f"{raw.sha256}"
            )
    for raw in RAW_FILES.values():
        dest = root / raw.relpath
        if dest.exists() and sha256_file(dest) != raw.sha256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    manifest_path = root / "manifest.json"
    existing = (
        Manifest.model_validate_json(manifest_path.read_text())
        if manifest_path.exists()
        else None
    )
    for raw in RAW_FILES.values():
        dest = root / raw.relpath
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / raw.relpath, dest)
    manifest = _raw_manifest(root, retrieved_at, datetime.now(UTC).isoformat())
    if existing is not None:
        if existing.files != manifest.files:
            raise RuntimeError(
                f"{manifest_path} records other files or retrievals; refusing"
            )
        return root
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return root


# --------------------------------------------------------------------------- #
# The release tables
# --------------------------------------------------------------------------- #
#: Largest |t - fit / sqrt(sigma^2 + SE^2)| accepted; 6.4e-13 is measured on the release.
T_IDENTITY_TOLERANCE = 1e-9


class ReleaseTables(BaseModel):
    """The compendium's Keio gene tables, aligned gene by gene and sample by sample.

    ``fitness`` and ``standard_error`` are genes x samples. ``standard_error`` is the
    paper's: the larger of the strain-consistency and read-count estimates.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    locus_ids: tuple[int, ...]
    b_numbers: tuple[str, ...]
    samples: tuple[str, ...]
    fitness: npt.NDArray[np.float64]
    standard_error: npt.NDArray[np.float64]
    n_values: int
    n_naive_larger: int = Field(
        description="values whose read-count estimate exceeds the strain-consistency one"
    )
    t_max_abs_deviation: float = Field(
        description="max |t - fit / sqrt(sigma^2 + SE^2)| over every value"
    )


def sample_columns(table: pd.DataFrame, label: str) -> dict[str, str]:
    """Sample name (the header's first token) to its column, refusing a repeat."""
    columns: dict[str, str] = {}
    for column in table.columns:
        name = str(column).split(" ")[0]
        if not name.startswith("set"):
            continue
        if name in columns:
            raise ValueError(f"{label}: sample {name} has two columns")
        columns[name] = str(column)
    return columns


def align_release(
    fitness: pd.DataFrame,
    se_obs: pd.DataFrame,
    se_naive: pd.DataFrame,
    t: pd.DataFrame,
    samples: Sequence[str],
    sigma: float,
) -> ReleaseTables:
    """Align the four gene tables on ``samples`` and check them against each other.

    Refused: tables whose gene rows (locusId, sysName) differ; a sample missing from a
    table or whose header text differs between tables; ``samples`` that are not exactly
    the fitness table's samples; a missing value; a non-positive standard error; and a
    released t that is not ``fit / sqrt(sigma^2 + max(se_obs, se_naive)^2)``.
    """
    keys = ["locusId", "sysName"]
    for label, table in (("se_obs", se_obs), ("se_naive", se_naive), ("t", t)):
        if not table[keys].equals(fitness[keys]):
            raise ValueError(f"{label}: gene rows differ from the fitness table's")
    fit_columns = sample_columns(fitness, "fitness")
    if sorted(fit_columns) != sorted(samples):
        raise ValueError(
            f"fitness samples != the experiment table's: only in fitness "
            f"{sorted(set(fit_columns) - set(samples))}, only in the table "
            f"{sorted(set(samples) - set(fit_columns))}"
        )
    frames = {"se_obs": se_obs, "se_naive": se_naive, "t": t}
    arrays: dict[str, npt.NDArray[np.float64]] = {}
    for label, table in frames.items():
        columns = sample_columns(table, label)
        missing = sorted(set(samples) - set(columns))
        if missing:
            raise ValueError(f"{label}: samples absent {missing}")
        differing = sorted(s for s in samples if columns[s] != fit_columns[s])
        if differing:
            raise ValueError(f"{label}: header text differs for {differing}")
        arrays[label] = table[[columns[s] for s in samples]].to_numpy(dtype=np.float64)
    values = fitness[[fit_columns[s] for s in samples]].to_numpy(dtype=np.float64)
    for label, array in {"fitness": values, **arrays}.items():
        if np.isnan(array).any():
            raise ValueError(f"{label}: {int(np.isnan(array).sum())} missing values")
    if (arrays["se_obs"] <= 0).any() or (arrays["se_naive"] <= 0).any():
        raise ValueError("a standard-error estimate is not positive")
    standard_error = np.maximum(arrays["se_obs"], arrays["se_naive"])
    deviation = float(
        np.max(np.abs(arrays["t"] - values / np.sqrt(sigma**2 + standard_error**2)))
    )
    if deviation > T_IDENTITY_TOLERANCE:
        raise ValueError(
            f"the released t is not fit / sqrt({sigma}^2 + SE^2): max deviation "
            f"{deviation:.3g}"
        )
    return ReleaseTables(
        locus_ids=tuple(int(v) for v in fitness["locusId"]),
        b_numbers=tuple(str(v) for v in fitness["sysName"]),
        samples=tuple(samples),
        fitness=values,
        standard_error=standard_error,
        n_values=int(values.size),
        n_naive_larger=int((arrays["se_naive"] > arrays["se_obs"]).sum()),
        t_max_abs_deviation=deviation,
    )


def read_release(raw_dir: str, samples: Sequence[str]) -> ReleaseTables:
    """:func:`align_release` on the four tables linked into ``raw_dir``."""

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(osp.join(raw_dir, name), sep="\t")

    return align_release(
        fitness=read("fit_logratios_good.tab"),
        se_obs=read("fit_standard_error_obs.tab"),
        se_naive=read("fit_standard_error_naive.tab"),
        t=read("fit_t.tab"),
        samples=samples,
        sigma=float(wetmore.T_STATISTIC_SIGMA.value),
    )


# --------------------------------------------------------------------------- #
# Samples: source paper, environment, retention
# --------------------------------------------------------------------------- #
class SampleSource(StrEnum):
    """The paper that first reported a sample (its records' publication)."""

    wetmore2015 = wetmore.CITATION_KEY
    price2018 = CITATION_KEY


PUBLICATIONS: dict[SampleSource, Publication] = {
    SampleSource.wetmore2015: Publication(
        doi=wetmore.PAPER_DOI, doi_url=f"https://doi.org/{wetmore.PAPER_DOI}"
    ),
    SampleSource.price2018: Publication(
        doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}"
    ),
}

DROP_DISREGARDED = "authors_withdrew_sucrose_and_mannitol_2021"
DROP_MEDIUM_NOT_IN_LIBRARY = "medium_not_in_media_library"
DROP_MOTILITY = "motility_soft_agar_assay"
DROP_NO_ECK_PAIR = "no_one_to_one_eck_pair"

DROP_RULES: dict[str, str] = {
    DROP_DISREGARDED: "the compendium page's 2021 note: "
    f"{wetmore.STOCK_SOLUTION_CORRECTION.quote!r}; the condition is sucrose or "
    "D-mannitol (wetmore2015.DISREGARDED_CONDITIONS)",
    DROP_MEDIUM_NOT_IN_LIBRARY: "'MOPS Rich Defined media_noCarbon' has no MEDIA_LIBRARY "
    "entry (Table S18 states it; the bacterial media tranche left it out), and a "
    "free-text medium joins nothing",
    DROP_MOTILITY: "an LB 0.3% soft-agar plate cut into 'outer' or 'inner' regions: no "
    "MEDIA_LIBRARY entry for LB Lennox soft agar (LB_LENNOX is liquid, LB_AGAR is Miller "
    "at 2%), and the readout is a spatial selection, not growth in a condition",
    DROP_NO_ECK_PAIR: "the release's MG1655 b-number has no one-to-one ECK pair with a "
    "BW25113 locus (absent from the MG1655 annotation, no ECK synonym, or an ECK id "
    "BW25113 lacks or carries on several loci); per gene in identifier_mapping.json",
}

#: Table S5 ``Media`` -> the library object. Anything else is refused.
MEDIA_BY_LABEL: dict[str, Media] = {
    "LB": LB_LENNOX,
    "M9 minimal media_noCarbon": M9_NOCARBON_PRICE2018,
    "M9 minimal media_noNitrogen": M9_NONITROGEN_PRICE2018,
    "MOPS minimal media_noCarbon": MOPS_MINIMAL,
}
#: Table S5 ``Media`` values that are dropped, not refused.
UNLISTED_MEDIA: frozenset[str] = frozenset({"MOPS Rich Defined media_noCarbon"})

GroupKind = Literal["carbon_source", "nitrogen_source", "stress", "plain"]
#: Table S5 ``Group`` -> how its Condition_1 enters the environment.
GROUP_KINDS: dict[str, GroupKind] = {
    "carbon source": "carbon_source",
    "nitrogen source": "nitrogen_source",
    "stress": "stress",
    "lb": "plain",
}
#: The media a group may name; a pairing outside this is refused.
GROUP_MEDIA: dict[GroupKind, frozenset[str]] = {
    "carbon_source": frozenset(
        {"M9 minimal media_noCarbon", "MOPS minimal media_noCarbon"}
    ),
    "nitrogen_source": frozenset({"M9 minimal media_noNitrogen"}),
    "stress": frozenset({"LB"}),
    "plain": frozenset({"LB"}),
}
#: Table S5 ``Units_1`` -> the typed unit. ``mg/ml`` IS g/L (1 mg/mL = 1 g/L), so the
#: value is kept as printed.
UNITS_BY_LABEL: dict[str, ConcentrationUnit] = {
    "mM": ConcentrationUnit.millimolar,
    "mg/ml": ConcentrationUnit.g_per_l,
    "g/L": ConcentrationUnit.g_per_l,
    "vol%": ConcentrationUnit.percent_v_v,
}


class SampleSpec(BaseModel):
    """One Table S5 Keio row: what the sample was, who reported it, and its fate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    group: str
    short: str
    media_label: str
    condition: str | None
    concentration: float | None
    units: str | None
    temperature_c: float
    source: SampleSource
    drop_rule: str | None

    @property
    def screen_id(self) -> str:
        """``Keio:<name>``: names are unique only within an organism (EXPERIMENT_NAME)."""
        return f"{ORG_ID}:{self.name}"


def _optional_str(value: Any) -> str | None:
    """A spreadsheet cell as text, or None when empty."""
    return None if pd.isna(value) else str(value)


def _optional_float(value: Any) -> float | None:
    """A spreadsheet cell as a float, or None when empty."""
    return None if pd.isna(value) else float(value)


def classify_samples(
    table_s5: pd.DataFrame, carried: frozenset[str]
) -> tuple[SampleSpec, ...]:
    """Every Keio row of Table S5 as a :class:`SampleSpec`, in table order.

    ``carried`` names the samples Wetmore 2015 first reported. Each row is aerobic (else
    refused) and either kept or dropped by exactly one rule, checked in this order: the
    2021 withdrawal, a medium outside the library, the motility assay. A kept row must
    name a library medium and a known group that the medium fits.
    """
    names = [str(n) for n in table_s5["name"]]
    if len(set(names)) != len(names):
        raise ValueError("Table S5 repeats a Keio sample name")
    unknown = sorted(carried - set(names))
    if unknown:
        raise ValueError(f"carried samples absent from Table S5: {unknown}")
    specs: list[SampleSpec] = []
    for _, row in table_s5.iterrows():
        if row["Aerobic_v_Anaerobic"] != "Aerobic":
            raise ValueError(
                f"{row['name']}: not aerobic ({row['Aerobic_v_Anaerobic']})"
            )
        condition = _optional_str(row["Condition_1"])
        media_label = str(row["Media"])
        group = str(row["Group"])
        drop: str | None = None
        if condition in wetmore.DISREGARDED_CONDITIONS:
            drop = DROP_DISREGARDED
        elif media_label in UNLISTED_MEDIA:
            drop = DROP_MEDIUM_NOT_IN_LIBRARY
        elif group == "motility":
            drop = DROP_MOTILITY
        else:
            if media_label not in MEDIA_BY_LABEL:
                raise ValueError(f"{row['name']}: unknown medium {media_label!r}")
            if group not in GROUP_KINDS:
                raise ValueError(f"{row['name']}: unknown group {group!r}")
            if media_label not in GROUP_MEDIA[GROUP_KINDS[group]]:
                raise ValueError(
                    f"{row['name']}: group {group!r} on medium {media_label!r}"
                )
        name = str(row["name"])
        specs.append(
            SampleSpec(
                name=name,
                group=group,
                short=str(row["short"]),
                media_label=media_label,
                condition=condition,
                concentration=_optional_float(row["Concentration_1"]),
                units=_optional_str(row["Units_1"]),
                temperature_c=float(row["Temperature"]),
                source=(
                    SampleSource.wetmore2015
                    if name in carried
                    else SampleSource.price2018
                ),
                drop_rule=drop,
            )
        )
    return tuple(specs)


def _dose(spec: SampleSpec) -> Concentration:
    """The sample's Condition_1 dose with its typed unit."""
    if spec.concentration is None or spec.units is None:
        raise ValueError(f"{spec.name}: {spec.condition!r} has no dose")
    if spec.units not in UNITS_BY_LABEL:
        raise ValueError(f"{spec.name}: unknown unit {spec.units!r}")
    return Concentration(value=spec.concentration, unit=UNITS_BY_LABEL[spec.units])


def build_environment(spec: SampleSpec) -> Environment:
    """The environment of a kept sample: library medium, temperature, Condition_1.

    A carbon or nitrogen source is the varied factor of a medium that leaves it out
    (``CARBON_FREE_MEDIA``), so it is an ``EnvironmentPhysicalPerturbation`` with the
    compound as ``agent``; a stress compound is an added ``SmallMoleculePerturbation``
    whose vehicle is a typed gap; the plain LB samples carry no perturbation.
    """
    if spec.drop_rule is not None:
        raise ValueError(f"{spec.name} is dropped ({spec.drop_rule})")
    kind = GROUP_KINDS[spec.group]
    perturbations: list[EnvironmentPerturbationType]
    if kind == "plain":
        if spec.condition is not None:
            raise ValueError(f"{spec.name}: a plain sample names {spec.condition!r}")
        perturbations = []
    elif spec.condition is None:
        raise ValueError(f"{spec.name}: a {spec.group} sample names no condition")
    elif kind == "stress":
        perturbations = [
            SmallMoleculePerturbation(
                compound=resolved_compound(spec.condition),
                concentration=_dose(spec),
                solvent=None,
                provenance_gaps=[SOLVENT_GAP],
            )
        ]
    else:
        perturbations = [
            EnvironmentPhysicalPerturbation(
                factor=(
                    PhysicalFactor.carbon_source
                    if kind == "carbon_source"
                    else PhysicalFactor.nitrogen_source
                ),
                magnitude=_dose(spec),
                agent=resolved_compound(spec.condition),
            )
        ]
    return Environment(
        media=MEDIA_BY_LABEL[spec.media_label],
        temperature=Temperature(value=spec.temperature_c),
        perturbations=perturbations,
        aerobicity="aerobic",
    )


# --------------------------------------------------------------------------- #
# Genes: the ECK route from MG1655 b-numbers to BW25113 locus tags
# --------------------------------------------------------------------------- #
#: The ECK route must place at least this fraction of the release's genes; it places
#: 3,768 of 3,789 (0.9945) on the deposited annotations.
MIN_ECK_ROUTE_FRACTION = 0.99

UnmappedReason = Literal[
    "b_number_not_in_mg1655_annotation",
    "mg1655_locus_has_no_eck_synonym",
    "eck_absent_from_bw25113",
    "eck_not_one_to_one",
]


class GeneMapping(BaseModel):
    """One release gene placed on its BW25113 locus through a one-to-one ECK pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    b_number: str
    eck: str
    locus_tag: str
    perturbed_gene_name: str
    numerics_agree: bool


class UnmappedGene(BaseModel):
    """A release gene the ECK route cannot place, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    b_number: str
    reason: UnmappedReason
    eck: tuple[str, ...]


class IdentifierReport(BaseModel):
    """The ECK route over the release's genes, with the reconciler's histograms."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_source_genes: int
    n_mapped: int
    mapped_fraction: float
    min_fraction: float
    numeric_disagreements: tuple[tuple[str, str, str], ...] = Field(
        description="(b-number, BW25113 tag, ECK) of mapped pairs whose numbers differ"
    )
    unmapped: tuple[UnmappedGene, ...]
    unmapped_by_reason: dict[str, int]
    reconcile_status_histogram: dict[str, int] = Field(
        description="reconcile_locus_tags over the mapped genes' ECK ids, on BW25113"
    )
    reconcile_layer_histogram: dict[str, int]
    n_symbol_names: int = Field(
        description="perturbed_gene_name is the BW25113 gene symbol"
    )
    n_tag_names: int = Field(
        description="perturbed_gene_name is the locus tag (symbol absent or not unique)"
    )


def eck_route(
    b_numbers: Sequence[str],
    crosswalk: EckCrosswalk,
    mg1655_ecks: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, tuple[str, str]], tuple[UnmappedGene, ...]]:
    """Each b-number's (ECK, BW25113 tag) through its one-to-one pair, or why not.

    ``mg1655_ecks`` maps every MG1655 locus tag to its ECK synonyms. Returns the placed
    genes keyed by b-number and the unplaced ones, each with its reason.
    """
    by_b = {pair.mg1655: pair for pair in crosswalk.pairs}
    absent = set(crosswalk.mg1655_only)
    not_one_to_one = set(crosswalk.not_one_to_one)
    placed: dict[str, tuple[str, str]] = {}
    unplaced: list[UnmappedGene] = []
    for b in b_numbers:
        if b in by_b:
            placed[b] = (by_b[b].eck, by_b[b].bw25113)
            continue
        if b not in mg1655_ecks:
            reason: UnmappedReason = "b_number_not_in_mg1655_annotation"
            ecks: tuple[str, ...] = ()
        else:
            ecks = mg1655_ecks[b]
            if not ecks:
                reason = "mg1655_locus_has_no_eck_synonym"
            elif any(eck in not_one_to_one for eck in ecks):
                reason = "eck_not_one_to_one"
            elif all(eck in absent for eck in ecks):
                reason = "eck_absent_from_bw25113"
            else:
                raise ValueError(f"{b}: ECK {ecks} is paired, absent and not shared")
        unplaced.append(UnmappedGene(b_number=b, reason=reason, eck=ecks))
    return placed, tuple(unplaced)


def gene_symbol(
    symbol: str | None, tag: str, resolve: Callable[[str], GeneNameResolution]
) -> str:
    """The locus's GenBank gene symbol when it resolves back to that locus, else the tag.

    The stored common name then always resolves to the stored locus tag.
    """
    if symbol:
        resolution = resolve(symbol)
        if resolution.systematic_name == tag and resolution.status in (
            GeneNameStatus.CURRENT,
            GeneNameStatus.RENAMED,
        ):
            return symbol
    return tag


#: ``reconcile_locus_tags`` bound to a genome and a label: ECK ids in, tags out.
Reconcile = Callable[[pd.Series], tuple[pd.Series, LocusTagReconciliation]]


def assemble_mapping(
    b_numbers: Sequence[str],
    crosswalk: EckCrosswalk,
    mg1655_ecks: Mapping[str, tuple[str, ...]],
    reconcile: Reconcile,
    name_of: Callable[[str], str],
    *,
    label: str,
) -> tuple[dict[str, GeneMapping], IdentifierReport]:
    """Place the release's b-numbers on BW25113 locus tags and report how.

    ``reconcile`` runs the placed genes' ECK ids through ``reconcile_locus_tags`` on
    BW25113 and must return the crosswalk's tag for every one; ``name_of`` gives a
    tag's stored common name. The route must place at least ``MIN_ECK_ROUTE_FRACTION``
    of the genes, else the build stops (checklist item 4) rather than dropping more.
    """
    placed, unplaced = eck_route(b_numbers, crosswalk, mg1655_ecks)
    fraction = len(placed) / len(b_numbers)
    if fraction < MIN_ECK_ROUTE_FRACTION:
        raise LocusTagResolutionError(
            f"{label}: the ECK route places {len(placed)} of {len(b_numbers)} genes "
            f"({fraction:.4f}), below {MIN_ECK_ROUTE_FRACTION}"
        )
    order = [b for b in b_numbers if b in placed]
    stored, reconciliation = reconcile(pd.Series([placed[b][0] for b in order]))
    differing = [b for b, tag in zip(order, stored, strict=True) if tag != placed[b][1]]
    if differing:
        raise ValueError(f"{label}: reconciler and crosswalk disagree on {differing}")
    mapping = {
        b: GeneMapping(
            b_number=b,
            eck=placed[b][0],
            locus_tag=placed[b][1],
            perturbed_gene_name=name_of(placed[b][1]),
            numerics_agree=(
                b.removeprefix("b") == placed[b][1].removeprefix("BW25113_")
            ),
        )
        for b in order
    }
    n_symbol = sum(1 for m in mapping.values() if m.perturbed_gene_name != m.locus_tag)
    report = IdentifierReport(
        n_source_genes=len(b_numbers),
        n_mapped=len(mapping),
        mapped_fraction=fraction,
        min_fraction=MIN_ECK_ROUTE_FRACTION,
        numeric_disagreements=tuple(
            (m.b_number, m.locus_tag, m.eck)
            for m in mapping.values()
            if not m.numerics_agree
        ),
        unmapped=unplaced,
        unmapped_by_reason=dict(Counter(u.reason for u in unplaced)),
        reconcile_status_histogram={
            status.value: n for status, n in reconciliation.status_histogram.items()
        },
        reconcile_layer_histogram=dict(reconciliation.layer_histogram),
        n_symbol_names=n_symbol,
        n_tag_names=len(mapping) - n_symbol,
    )
    return mapping, report


def map_genes(
    b_numbers: Sequence[str],
    mg1655: EcoliK12MG1655Genome,
    bw25113: EcoliK12BW25113Genome,
    *,
    label: str,
) -> tuple[dict[str, GeneMapping], IdentifierReport]:
    """:func:`assemble_mapping` on the deposited MG1655 and BW25113 annotations."""
    mg1655_ecks = {
        tag: tuple(s for s in locus.synonyms if s.startswith("ECK"))
        for tag, locus in mg1655.genbank.loci.items()
    }
    loci = bw25113.genbank.loci

    def reconcile(ecks: pd.Series) -> tuple[pd.Series, LocusTagReconciliation]:
        return reconcile_locus_tags(bw25113, ecks, label=f"{label} ECK ids")

    return assemble_mapping(
        b_numbers,
        eck_crosswalk(mg1655, bw25113),
        mg1655_ecks,
        reconcile,
        lambda tag: gene_symbol(loci[tag].symbol, tag, bw25113.resolve_gene_name),
        label=label,
    )


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
UNITS = (
    "RB-TnSeq gene fitness, Fitness Browser compendium release (statistics version "
    "1.0.3): normalized log2 ratio of the gene's insertion-strain barcode abundance at "
    "the end of growth in the condition over the time-zero sample, weighted over the "
    "gene's usable strains, typical gene = 0; SE = max(strain-consistency, read-count) "
    "estimate"
)
UNITS_REFERENCE = "the typical gene of this sample (gene fitness is normalized to 0)"
DESCRIPTION = (
    "Tn5 transposon insertion disrupting a gene; gene-level call over the gene's usable "
    "insertion strains (no per-strain barcode or mapped site). Locus tag DERIVED: the "
    "release names an MG1655 b-number, mapped to this BW25113 locus through its "
    "one-to-one ECK pair (eck_crosswalk)"
)


def build_genotype(mapping: GeneMapping) -> Genotype:
    """A Tn5 insertion in one gene of KEIO_ML9, the derived mapping in its description."""
    return Genotype(
        perturbations=[
            TransposonInsertionPerturbation(
                systematic_gene_name=mapping.locus_tag,
                perturbed_gene_name=mapping.perturbed_gene_name,
                gene_namespace=STRAIN_GENE_NAMESPACES[REFERENCE_STRAIN],
                description=DESCRIPTION,
                transposon=TRANSPOSON,
                library_pool=MUTANT_LIBRARY,
            )
        ]
    )


def build_phenotype(
    fitness: float, standard_error: float, screen_id: str
) -> EnvironmentResponsePhenotype:
    """One gene's fitness in one sample, with the paper's standard error."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=fitness,
        environment_response_uncertainty=standard_error,
        environment_response_uncertainty_type=UncertaintyType.standard_error,
        screen_id=screen_id,
        units=UNITS,
        provenance_gaps=[N_SAMPLES_GAP, SAMPLE_UNIT_GAP],
    )


def build_reference(
    dataset_name: str,
    environment: Environment,
    genome_reference: AssemblyReferenceGenome,
    screen_id: str,
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
            screen_id=screen_id,
            units=UNITS_REFERENCE,
        ),
    )


class SampleRecords(BaseModel):
    """A kept sample with everything its records share."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    spec: SampleSpec
    column: int
    environment: Environment
    reference: BacterialEnvironmentResponseExperimentReference
    publication: Publication


def sample_records(
    dataset_name: str,
    specs: Sequence[SampleSpec],
    release: ReleaseTables,
    genome_reference: AssemblyReferenceGenome,
) -> list[SampleRecords]:
    """The kept samples, in Table S5 order, each with its release column."""
    columns = {name: j for j, name in enumerate(release.samples)}
    out: list[SampleRecords] = []
    for spec in specs:
        if spec.drop_rule is not None:
            continue
        environment = build_environment(spec)
        out.append(
            SampleRecords(
                spec=spec,
                column=columns[spec.name],
                environment=environment,
                reference=build_reference(
                    dataset_name, environment, genome_reference, spec.screen_id
                ),
                publication=PUBLICATIONS[spec.source],
            )
        )
    return out


def iter_records(
    dataset_name: str,
    samples: Sequence[SampleRecords],
    release: ReleaseTables,
    mapping: Mapping[str, GeneMapping],
) -> Iterator[
    tuple[
        BacterialEnvironmentResponseExperiment,
        BacterialEnvironmentResponseExperimentReference,
        Publication,
    ]
]:
    """Every record, sample by sample then gene by gene in release order."""
    rows = [
        (i, build_genotype(mapping[b]))
        for i, b in enumerate(release.b_numbers)
        if b in mapping
    ]
    for sample in samples:
        j = sample.column
        for i, genotype in rows:
            yield (
                BacterialEnvironmentResponseExperiment(
                    dataset_name=dataset_name,
                    genotype=genotype,
                    environment=sample.environment,
                    phenotype=build_phenotype(
                        float(release.fitness[i, j]),
                        float(release.standard_error[i, j]),
                        sample.spec.screen_id,
                    ),
                ),
                sample.reference,
                sample.publication,
            )


#: Records the build writes: 147 kept samples x 3,768 mapped genes (measured 2026-10-07).
EXPECTED_SAMPLES = 147
EXPECTED_GENES = 3768
EXPECTED_RECORDS = EXPECTED_SAMPLES * EXPECTED_GENES


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
class DropRuleCount(BaseModel):
    """One retention rule, what it removed, and the records that removed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str
    scope: Literal["sample", "gene"]
    description: str
    items: tuple[str, ...]
    n_records: int


class Inventory(BaseModel):
    """The sample and record accounting the dendron note states."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table_s5_samples: int
    samples_by_source: dict[str, int]
    kept_samples_by_source: dict[str, int]
    kept_samples_by_group: dict[str, int]
    kept_samples_by_medium: dict[str, int]
    source_genes: int
    kept_genes: int
    source_records: int
    kept_records: int
    drops: tuple[DropRuleCount, ...]
    unidentified_compounds: dict[str, int] = Field(
        description="Condition_1 labels of kept samples the identity table does not "
        "resolve -> kept samples naming them (records carry the typed inchikey gap)"
    )


def inventory(specs: Sequence[SampleSpec], identifiers: IdentifierReport) -> Inventory:
    """Count samples, genes and records kept and dropped, by rule and by source."""
    kept = [s for s in specs if s.drop_rule is None]
    genes = identifiers.n_mapped
    drops = [
        DropRuleCount(
            rule=rule,
            scope="sample",
            description=DROP_RULES[rule],
            items=tuple(s.name for s in specs if s.drop_rule == rule),
            n_records=sum(1 for s in specs if s.drop_rule == rule)
            * identifiers.n_source_genes,
        )
        for rule in (DROP_DISREGARDED, DROP_MEDIUM_NOT_IN_LIBRARY, DROP_MOTILITY)
    ]
    drops.append(
        DropRuleCount(
            rule=DROP_NO_ECK_PAIR,
            scope="gene",
            description=DROP_RULES[DROP_NO_ECK_PAIR],
            items=tuple(u.b_number for u in identifiers.unmapped),
            n_records=len(identifiers.unmapped) * len(kept),
        )
    )
    unidentified = Counter(
        s.condition
        for s in kept
        if s.condition is not None
        and not resolve_compound_identity(name=s.condition).identified
    )
    return Inventory(
        table_s5_samples=len(specs),
        samples_by_source=dict(Counter(s.source.value for s in specs)),
        kept_samples_by_source=dict(Counter(s.source.value for s in kept)),
        kept_samples_by_group=dict(Counter(s.group for s in kept)),
        kept_samples_by_medium=dict(Counter(s.media_label for s in kept)),
        source_genes=identifiers.n_source_genes,
        kept_genes=genes,
        source_records=len(specs) * identifiers.n_source_genes,
        kept_records=len(kept) * genes,
        drops=tuple(drops),
        unidentified_compounds=dict(sorted(unidentified.items())),
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class RbTnseqPrice2018EcoliDataset(ExperimentDataset):
    """Price 2018 compendium, E. coli BW25113 KEIO_ML9: gene fitness per sample."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = "data/torchcell/rbtnseq_price2018_ecoli",
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``ecoli_genome`` is the BW25113 genome the build entry points inject."""
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
        """The release tables and Table S5, linked into ``raw/`` by ``download``."""
        return list(RAW_FILE_NAMES)

    def download(self) -> None:
        """Link each pinned mirror file into ``raw/`` after verifying its sha256."""
        os.makedirs(self.raw_dir, exist_ok=True)
        for name, (_, _, pin) in RAW_FILE_NAMES.items():
            link_verified(source_path(name), osp.join(self.raw_dir, name), pin)
        log.info("Price 2018 Keio raw files linked into %s", self.raw_dir)

    def _bw25113(self) -> EcoliK12BW25113Genome:
        """The injected genome, or the BW25113 genome reopened read-only (direct runs)."""
        genome = (
            bacterial_genome("ecoli", REFERENCE_STRAIN)
            if self.ecoli_genome is None
            else self.ecoli_genome
        )
        if not isinstance(genome, EcoliK12BW25113Genome):
            raise TypeError(f"expected the BW25113 genome, got {type(genome).__name__}")
        self.ecoli_genome = genome
        return genome

    @post_process
    def process(self) -> None:
        """Build one record per (mapped gene, kept sample) and the preprocess reports."""
        data_root = _data_root()
        verify_raw_files(
            self.raw_dir, {name: pin for name, (_, _, pin) in RAW_FILE_NAMES.items()}
        )
        table_s5 = wetmore.read_superset_experiments(osp.join(self.raw_dir, "si3.xlsx"))
        subsumption = wetmore.subsumption_record(data_root)
        specs = classify_samples(table_s5, frozenset(subsumption.carried))
        withdrawn = {s.name for s in specs if s.drop_rule == DROP_DISREGARDED}
        if withdrawn != set(subsumption.disregarded):
            raise ValueError(
                f"withdrawn samples {sorted(withdrawn)} != Wetmore's "
                f"{sorted(subsumption.disregarded)}"
            )
        release = read_release(self.raw_dir, [s.name for s in specs])
        mg1655 = bacterial_genome("ecoli", "MG1655", data_root)
        if not isinstance(mg1655, EcoliK12MG1655Genome):
            raise TypeError(f"expected the MG1655 genome, got {type(mg1655).__name__}")
        mapping, identifiers = map_genes(
            release.b_numbers, mg1655, self._bw25113(), label=self.name
        )
        samples = sample_records(
            self.name, specs, release, assembly_reference(REFERENCE_STRAIN)
        )
        counts = inventory(specs, identifiers)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        batch_size = 100_000
        txn = env.begin(write=True)
        itxn = interned_env.begin(write=True)
        for experiment, reference, publication in tqdm(
            iter_records(self.name, samples, release, mapping),
            total=counts.kept_records,
            desc="Price 2018 Keio",
        ):
            txn.put(
                f"{idx}".encode(),
                self._intern_record(experiment, reference, publication, itxn),
            )
            idx += 1
            if idx % batch_size == 0:
                itxn.commit()
                txn.commit()
                txn = env.begin(write=True)
                itxn = interned_env.begin(write=True)
        itxn.commit()
        txn.commit()
        env.close()
        interned_env.close()
        if idx != counts.kept_records:
            raise RuntimeError(
                f"wrote {idx} records, inventory says {counts.kept_records}"
            )
        self._write_reports(specs, release, identifiers, counts)
        log.info("Wrote %d Price 2018 Keio records", idx)

    def _write_reports(
        self,
        specs: Sequence[SampleSpec],
        release: ReleaseTables,
        identifiers: IdentifierReport,
        counts: Inventory,
    ) -> None:
        """``dropped_records.json``, ``identifier_mapping.json``, ``samples.json`` and
        ``standard_error.json`` in ``preprocess/``.
        """
        payloads: dict[str, Any] = {
            "dropped_records.json": counts.model_dump(mode="json"),
            "identifier_mapping.json": identifiers.model_dump(mode="json"),
            "samples.json": [s.model_dump(mode="json") for s in specs],
            "standard_error.json": {
                "rule": STANDARD_ERROR.quote,
                "n_values": release.n_values,
                "n_naive_larger": release.n_naive_larger,
                "t_max_abs_deviation": release.t_max_abs_deviation,
                "sigma": wetmore.T_STATISTIC_SIGMA.value,
            },
        }
        for name, payload in payloads.items():
            with open(osp.join(self.preprocess_dir, name), "w") as handle:
                json.dump(payload, handle, indent=2)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "Price 2018 builds its records in process(); see iter_records"
        )


# --------------------------------------------------------------------------- #
# Verification (L0 to L4) and the evidence report
# --------------------------------------------------------------------------- #
VERIFY_PROVENANCE = Provenance(
    source_uri=f"{_BIGFIT_KEIO}/fit_logratios_good.tab",
    citation_key=CITATION_KEY,
    sha256=wetmore.RAW_FILES[wetmore.SUPERSET_FITNESS].sha256,
    method=(
        "compendium gene fitness (statistics version 1.0.3) of Table S5's Keio samples; "
        "SE = max(fit_standard_error_obs, fit_standard_error_naive); MG1655 b-numbers "
        "placed on BW25113 through one-to-one ECK pairs"
    ),
    page="Supplementary Table 5 (si/si3.xlsx), orgId Keio",
    retrieved=wetmore.RAW_RETRIEVED_AT,
)


def stored_tags_are_loci(
    tags: set[str], resolve_gene_name: Callable[[str], GeneNameResolution]
) -> LevelResult:
    """SUPPLEMENTARY L1: every stored tag resolves to itself as a locus of BW25113.

    The shared ``canonical_gene_names`` row requires status ``current``, which a
    pseudogene locus never has (the bacterial resolver answers ``non_gene_feature``,
    naming the same tag). This row accepts a gene or a pseudogene that resolves to
    itself and counts both; it is added beside the shared row, never in its place.
    """
    statuses: Counter[str] = Counter()
    elsewhere: list[str] = []
    for tag in sorted(tags):
        resolution = resolve_gene_name(tag)
        statuses[resolution.status.value] += 1
        if resolution.systematic_name != tag or resolution.status not in (
            GeneNameStatus.CURRENT,
            GeneNameStatus.NON_GENE_FEATURE,
        ):
            elsewhere.append(tag)
    return LevelResult(
        level=Level.L1,
        name="stored_tags_are_loci_of_the_pinned_assembly",
        passed=not elsewhere,
        message=(
            f"SUPPLEMENTARY: {len(tags)} stored tags, statuses {dict(statuses)}; "
            f"{len(elsewhere)} do not resolve to themselves"
        ),
        details={"statuses": dict(statuses), "not_a_locus": elsewhere[:20]},
    )


def verify(data_root: str | None = None) -> Any:
    """Run the environment-response verifier on the dev build; write its report.

    The gene universe and resolver are BW25113's (every GenBank locus, pseudogenes
    included), selected by the records' own assembly pin. The report is
    ``preprocess/verification_report.json`` with one SUPPLEMENTARY row
    (:func:`stored_tags_are_loci`).
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records

    root = _data_root(data_root)
    abs_root = osp.join(root, "data/torchcell/rbtnseq_price2018_ecoli")
    genome = bacterial_genome("ecoli", REFERENCE_STRAIN, root)
    report = verify_environment_response_dataset_streaming(
        stream_records(abs_root),
        dataset_name=RbTnseqPrice2018EcoliDataset.__name__,
        provenance=VERIFY_PROVENANCE,
        expected_count=EXPECTED_RECORDS,
        sgd_genes=set(genome.genbank.loci),
        min_containment=1.0,
        resolve_gene_name=genome.resolve_gene_name,
    )
    tags = {
        str(p["systematic_gene_name"])
        for record in stream_records(abs_root)
        for p in record["experiment"]["genotype"]["perturbations"]
    }
    report.add(stored_tags_are_loci(tags, genome.resolve_gene_name))
    out = osp.join(abs_root, "preprocess", "verification_report.json")
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def report(data_root: str | None = None) -> dict[str, Any]:
    """Every number the dendron note states, recomputed from the pinned files."""
    root = _data_root(data_root)
    table_s5 = wetmore.read_superset_experiments(source_path("si3.xlsx", root))
    subsumption = wetmore.subsumption_record(root)
    specs = classify_samples(table_s5, frozenset(subsumption.carried))

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(source_path(name, root), sep="\t")

    release = align_release(
        fitness=read("fit_logratios_good.tab"),
        se_obs=read("fit_standard_error_obs.tab"),
        se_naive=read("fit_standard_error_naive.tab"),
        t=read("fit_t.tab"),
        samples=[s.name for s in specs],
        sigma=float(wetmore.T_STATISTIC_SIGMA.value),
    )
    mg1655 = bacterial_genome("ecoli", "MG1655", root)
    bw25113 = bacterial_genome("ecoli", REFERENCE_STRAIN, root)
    if not isinstance(mg1655, EcoliK12MG1655Genome) or not isinstance(
        bw25113, EcoliK12BW25113Genome
    ):
        raise TypeError("the K-12 genomes resolved to the wrong classes")
    _, identifiers = map_genes(release.b_numbers, mg1655, bw25113, label="report")
    return {
        "inventory": inventory(specs, identifiers).model_dump(mode="json"),
        "identifiers": identifiers.model_dump(mode="json"),
        "standard_error": {
            "n_values": release.n_values,
            "n_naive_larger": release.n_naive_larger,
            "fraction_naive_larger": release.n_naive_larger / release.n_values,
            "t_max_abs_deviation": release.t_max_abs_deviation,
        },
        "set_prefixes": sorted({s.name.split("IT")[0] for s in specs}),
    }


def main(argv: list[str] | None = None) -> None:
    """``retrieve`` / ``deposit`` the release files, print the ``report``, or ``verify``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    retrieve = commands.add_parser("retrieve", help="fetch the release files")
    retrieve.add_argument("--dest", required=True)
    deposit = commands.add_parser("deposit", help="deposit fetched files")
    deposit.add_argument("--source-dir", required=True)
    commands.add_parser("report", help="print the evidence as JSON")
    commands.add_parser("verify", help="verify the built dev LMDB (L0 to L4)")
    args = parser.parse_args(argv)
    if args.command == "retrieve":
        print(retrieve_raw_files(args.dest))
    elif args.command == "deposit":
        print(deposit_raw_mirror(source_dir=args.source_dir))
    elif args.command == "report":
        print(json.dumps(report(), indent=2))
    else:
        print(verify().summary())


if __name__ == "__main__":
    main()
