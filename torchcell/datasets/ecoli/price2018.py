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

TWO DATASETS, from two different analyses of the same TnSeq data.
:class:`RbTnseqPrice2018EcoliDataset` is the BarSeq gene fitness per (gene, sample).
:class:`GeneEssentialityPrice2018EcoliDataset` is Supplementary Table 1's likely-essential
gene list, computed from the insertion maps WITHOUT the barcodes ("We did not consider the
DNA barcodes in this analysis of essential genes") and over the genes the fitness analysis
could NOT value, so the two gene sets are disjoint by construction and the build proves it.

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
  vehicle is Supplementary Table 4's ``Solvent`` for that compound (water, Dimethyl
  Sulfoxide or Ethanol), matched to Condition_1 by case-insensitive exact compound name.
  That solvent is the one Table S4 records for the WILD-TYPE IC50 prescreen, and the
  paper never says the mutant fitness assays drew on those stocks, so every stress
  perturbation's ``description`` says so and ``Solvent.percent`` stays ``None`` (the
  final vehicle fraction is not released). Compounds the pinned identity table does not
  resolve carry the resolver's typed ``inchikey`` gap, the solvent ``water`` among them.

DROPPED SAMPLES (counted in ``preprocess/dropped_records.json``): the four sucrose and
D-mannitol samples the authors' 2021 note withdraws; the four ``MOPS Rich Defined
media_noCarbon`` samples (no media-library entry); the seven soft-agar motility samples
(no media-library entry for 0.3% LB Lennox agar, and the readout is a spatial cut, not
growth in a condition).

ESSENTIALITY (:class:`GeneEssentialityPrice2018EcoliDataset`). Supplementary Table 1's
324 ``orgId == "Keio"`` rows, one record each, as
``GeneEssentialityPhenotype(is_essential=True)`` in
``BacterialGeneEssentialityExperiment``.

* THE LABEL IS NOT PLAIN ESSENTIALITY, and the records must not be read as though it
  were. The paper's own label is "essential or important for growth (nearly essential)",
  the condition is the library-isolation condition ("growth on LB plates" at 37 C, not
  the fitness assays' conditions), and Supplementary Note 1 puts the E. coli
  false-discovery rate somewhere between 6% and 16% (21% against the PEC / Keio list
  before 15 cases the note argues are nearly essential). ``GeneEssentialityPhenotype``
  has one boolean and no slot for any of that, so the three quotes are
  ``ESSENTIAL_LABEL`` / ``ESSENTIAL_CONDITION`` / ``ESSENTIAL_FDR``, written verbatim to
  ``preprocess/essentiality_label.json`` with the quantities that have no field, and the
  caveat is restated on every record's perturbation ``description``.
* The call is a no-insertion call, not a measurement: "Protein-coding genes were
  considered essential or important for growth (nearly essential) if we did not estimate
  fitness values for the gene and both the normalized insertion density and the
  normalized read density were under 0.2". Table S1's coverage columns (``GC``,
  ``nReads``, ``normreads``, ``nPosCentral``, ``dens``) are the evidence the call is
  computed from, not a phenotype, and are kept in ``preprocess/essential_genes.csv``.
* 320 of the 324 rows become records. The four dropped are araA, araB, rhaA and rhaB
  (``b0062``, ``b0063``, ``b3903``, ``b3904``), whose ECK ids no BW25113 locus carries:
  BW25113 DELETES araBAD and rhaBAD, so a library built in it can have no insertion
  there and the release's own rule calls them essential. Table S1 corroborates this by
  leaving their ``locus_tag`` (its only BW25113 identifier) empty, and they are the only
  four Keio rows without one. The drop rule is the fitness loader's
  ``no_one_to_one_eck_pair``.
* Reference: the unperturbed BW25113 parent on the same selection plates, viable
  (``is_essential=False``) -- the library was built in it and selected there.
* Environment: a medium DERIVED from ``LB_LENNOX`` (``base_medium`` ``LB``, so it joins
  there), solid, plus agar at an unstated amount and kanamycin at Table S20's 50 ug/mL,
  at 37 C. ``LB_AGAR``'s 2% agar is another paper's bench value and is not asserted
  here. Duration is a typed gap in both forms.

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
    BW25113_BACKGROUND_GENOTYPE,
    BW25113_BACKGROUND_LESIONS,
    AssayType,
    AssemblyReferenceGenome,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialGeneEssentialityExperiment,
    BacterialGeneEssentialityExperimentReference,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    GeneEssentialityPhenotype,
    Genotype,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    PhysicalFactor,
    Publication,
    SmallMoleculePerturbation,
    Solvent,
    Temperature,
    TransposonInsertionPerturbation,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import (
    LOCUS_TAG_PATTERNS,
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
from torchcell.verification.levels import l0_structural, l1_count
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
#: The Supplementary Information OCR: Supplementary Notes 1 to 6, Note 1 being the
#: essentiality validation (the PEC / Keio benchmark and the E. coli FDR range).
SUPPLEMENTARY_NOTES = "si/si1.md"
SUPPLEMENTARY_NOTES_SHA256 = (
    "1c68b123ebb7516b6c7163985d3c9aee7f17f13651e844c3565f9bddb21b350e"
)
#: The Supplementary Tables workbook (Tables S1 to S22, one sheet each).
SUPPLEMENTARY_TABLES = "si/si3.xlsx"
SUPPLEMENTARY_TABLES_SHA256 = (
    "e5dbf3d5c97cfc12f49d7fd83f84bc95c16cbe963309a561fff20f442788b879"
)
TABLE_S1_SHEET = "TableS1_LikelyEssentialGenes"
TABLE_S4_SHEET = "TableS4_Stress"
TABLE_S5_SHEET = "TableS5_Experiments"
TABLE_S14_SHEET = "TableS14_RB_TnSeq_Bacteria"
TABLE_S20_SHEET = "TableS20_Mutagenesis"
#: First header cell of each sheet read here; every sheet carries a free-text preamble
#: above its header row, which is FOUND by this cell (``wetmore.read_superset_experiments``
#: finds Table S5's by ``orgId`` the same way) rather than pinned by row number.
HEADER_CELLS: dict[str, str] = {TABLE_S1_SHEET: "organism", TABLE_S4_SHEET: "Compound"}

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


def _notes(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote of the pinned Supplementary Information OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SUPPLEMENTARY_NOTES,
            citation_key=CITATION_KEY,
            sha256=SUPPLEMENTARY_NOTES_SHA256,
            method=_OCR_METHOD,
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

# The stress compounds' vehicle --------------------------------------------- #
SOLVENT_COLUMN = _tables(
    "Solvent",
    "Compound | CAS | CoreSet_forMutantFitnessAssays | Stock solution | Stock solution "
    "units | Solvent | Maximum concentration tested | Minimum concentration tested",
    page="Supplementary Table 4, header row",
    note="Solvent sits between the stock's units and the tested range, so it is the "
    "solvent OF THE STOCK SOLUTION; 55 rows, one per stress compound",
)
SOLVENT_ROWS = _tables(
    {"Kanamycin sulfate": "water", "Chloramphenicol": "Ethanol"},
    "Kanamycin sulfate | 25839-94-0 | no | 100 | mg/ml | water | 2 | 0.00390625",
    page="Supplementary Table 4, the Kanamycin sulfate row",
    note="the three values the column takes are water, Dimethyl Sulfoxide and Ethanol; "
    "Chloramphenicol | 56-75-7 | no | 34 | mg/ml | Ethanol is the one Ethanol row",
)
SOLVENT_ROWS_ETHANOL = _tables(
    "Ethanol",
    "Chloramphenicol | 56-75-7 | no | 34 | mg/ml | Ethanol | 0.68 | 0.001328125",
    page="Supplementary Table 4, the Chloramphenicol row",
)
SOLVENT_IS_THE_PRESCREEN_STOCK = _paper(
    "Table S4 is the wild-type IC50 prescreen",
    "For each compound, we grew the wildtype bacterium across a 1,000-fold range of "
    "inhibitor concentrations in a rich medium.",
    page="Methods, 'High-throughput growth assays of wild-type bacteria'",
    note="so Table S4's stock, its units and its Solvent describe THAT assay's stocks; "
    "whether the mutant fitness assays drew on the same stocks is never stated "
    "(paper.md contains none of 'stock solution', 'dissolved', 'solvent', 'DMSO' or "
    "'dimethyl'), which is what STRESS_DESCRIPTION says on every stress record",
)
SOLVENT_CONCENTRATION_CAVEAT = _tables(
    "the prescreen doses are not the fitness-assay doses",
    "The concentrations reported here are not necessarily the concentrations used for "
    "the mutant fitness assays. This may be due to differences in the growth conditions "
    "used for  the different assays.",
    page="Supplementary Table 4, preamble",
    note="the sheet's own caveat is about the CONCENTRATION column, which this loader "
    "never reads (the dose comes from Table S5's Concentration_1); it is quoted here "
    "because it is the closest the release comes to saying whether the stocks were "
    "shared, and it does not say so",
)

# Supplementary Table 1: the likely-essential genes --------------------------- #
ESSENTIAL_TABLE = _tables(
    TABLE_S1_SHEET,
    "This table lists all of the likely-essential protein-coding genes in the 32 "
    "bacteria",
    page="Supplementary Table 1, preamble",
)
ESSENTIAL_TABLE_COLUMNS = _tables(
    "Table S1 columns",
    "organism | orgId | locusId | sysName | locus_tag | protein_id | uniprotId | "
    "scaffoldId | begin | end | strand | name | desc | GC | nReads | normreads | "
    "nPosCentral | dens | geneClass",
    page="Supplementary Table 1, header row",
    note="sysName holds the MG1655 b-number (all 324 Keio rows match b\\d{4}), the same "
    "identifier the fitness tables name, so the stored locus tag comes through the same "
    "one-to-one ECK pair",
)
ESSENTIAL_LABEL = _paper(
    "essential or important for growth (nearly essential)",
    "Genes that lack insertions or that have very low coverage in the start samples are "
    "likely to be essential or important for growth (nearly essential) in rich medium, "
    "as except for S. elongatus, pools of mutants were produced and recovered in medium "
    "that contained yeast extract.",
    page="Methods, 'Identifying essential or nearly essential genes'",
    note="the label the records carry is THIS, not unconditional essentiality: a gene "
    "with no insertions in a rich-medium library. GeneEssentialityPhenotype has one "
    "boolean and no slot for 'nearly', so the distinction lives in this quote, in "
    "preprocess/essentiality_label.json and in ESSENTIAL_DESCRIPTION",
)
ESSENTIAL_RULE = _paper(
    0.2,
    "Protein-coding genes were considered essential or important for growth (nearly "
    "essential) if we did not estimate fitness values for the gene and both the "
    "normalized insertion density and the normalized read density were under 0.2.",
    page="Methods, 'Identifying essential or nearly essential genes'",
    note="a no-insertion call over genes the fitness analysis could not value, which is "
    "why the essential set and the fitness set are disjoint by construction; the two "
    "densities are Table S1's dens and normreads, kept in preprocess/essential_genes.csv",
)
ESSENTIAL_IGNORES_BARCODES = _paper(
    "TnSeq only, no barcodes",
    "We did not consider the DNA barcodes in this analysis of essential genes.",
    page="Methods, 'Identifying essential or nearly essential genes'",
    note="so this dataset is NOT a BarSeq measurement and shares no value with the "
    "fitness dataset, although both come from one library",
)
FITNESS_GENES_ARE_NON_ESSENTIAL = _paper(
    123255,
    "In this study, we restricted our analysis to the 123,255 different non-essential "
    "protein-coding genes for which we collected gene fitness data.",
    page="Methods, 'Computation of fitness values'",
    note="across the 32 bacteria; the build refuses a release whose essential Keio "
    "b-numbers meet its fitness Keio b-numbers at all",
)
ESSENTIAL_PER_ORGANISM_RANGE = _paper(
    (289, 614),
    "We identified 289–614 genes per bacterium that are likely to encode essential "
    "proteins",
    page="Results, the 32-bacterium survey",
    note="E. coli's 324 Keio rows of Table S1 sit inside the stated range; the same "
    "sentence calls them 'likely to encode essential proteins', never 'essential'",
)
ESSENTIAL_CONDITION = _notes(
    "growth on LB plates",
    'However, some of these "false positives" are likely to be essential, or nearly so, '
    "in the condition that we used to isolate our mutant library, namely growth on LB "
    "plates.",
    page="Supplementary Note 1, the E. coli validation",
    note="the condition of the call is the LIBRARY-ISOLATION condition, not any of the "
    "147 fitness-assay conditions; plates, so the stored medium is solid",
)
ESSENTIAL_CONDITION_TEMPERATURE = _notes(
    37.0,
    "but there may be other genes in our list that are nearly-essential for growth on "
    r"LB plates at $3 7 ^ { \circ } \mathsf { C }$ .",
    page="Supplementary Note 1, the E. coli validation",
    note="agrees with Table S20's 'Temperature for selecting mutants' of 37 for "
    "KEIO_ML9 (LIBRARY)",
)
ESSENTIAL_FDR = _notes(
    (0.06, 0.16),
    "So, we expect that the true rate of false positives in our list of $E .$ coli "
    r"proteins that are essential, or nearly so, for growth in rich media is somewhere "
    r"between $6 \%$ and $16 \%$ .",
    page="Supplementary Note 1, the E. coli validation",
    note="a DATASET-level false-discovery rate with no per-record slot on "
    "GeneEssentialityPhenotype; it is written to preprocess/essentiality_label.json "
    "verbatim and must travel with any use of these 320 records",
)
ESSENTIAL_FDR_NAIVE = _notes(
    0.21,
    "Our list of essential genes also includes 67 non-essential genes, which "
    r"corresponds to a false discovery rate (FDR) of $21 \%$ .",
    page="Supplementary Note 1, the E. coli validation",
    note="the naive rate against the PEC / Keio list, before the note argues 15 of the "
    "67 are nearly essential in the library-isolation condition; 21% and 6% are the two "
    "ends ESSENTIAL_FDR reconciles",
)
ESSENTIAL_BENCHMARK = _notes(
    (330, 257),
    "By combining the profiling of E. coli chromosome database (PEC) and the results of "
    r"systematically attempting to delete every gene in E. coli 1,6, we obtained a list "
    r"of $3 3 0 \ E .$ coli genes that were previously reported to be essential. 257 of "
    r"these 330 genes $( 7 8 \% )$ were in our list of likely-essential genes from TnSeq "
    "analysis.",
    page="Supplementary Note 1, the E. coli validation",
    note="the external benchmark (Baba 2006 / Yamamoto 2009 / PEC) is NOT in this "
    "repo's mirror, so the 78% agreement cannot be recomputed here",
)
SELECTION_MEDIA_COLUMN = _tables(
    "LB plates with kanamycin",
    "5 | Media used for for both the conjugation (with supplemented diaminopimelic "
    "acid) and for the selection of transposon mutants (with supplemented kanamycin), "
    "except for Synechococcus elongatus PCC 7942, which was conjugated on LB plates "
    "with DAP.",
    page="Supplementary Table 20, note 5",
    note="E. coli's row gives Media 'LB', 'Temperature for selecting mutants' 37 and "
    "'Antibiotic; concentration (in ug/mL)' 'Kanamycin; 50' (LIBRARY); delivery was "
    "electroporation, so no DAP conjugation step applies to KEIO_ML9",
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
ESSENTIAL_DURATION_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=PAPER_MD,
        citation_key=CITATION_KEY,
        sha256=PAPER_MD_SHA256,
        method=_OCR_METHOD,
        page="Methods, 'Constructing pools of randomly barcoded transposon mutants' "
        "(the plate step is 'After growth', with no time); Table S20 gives the medium "
        "and the temperature, not a duration",
    ),
    note="how long the KEIO_ML9 selection plates were incubated is never stated",
)
ESSENTIAL_GENERATIONS_GAP = ProvenanceGap(
    field="duration_generations",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=ESSENTIAL_DURATION_GAP.looked_in,
    note="travels with duration_hours: colony growth on a selection plate is not "
    "reported in doublings either, and the paper's generation counts were measured "
    "for six bacteria, E. coli not among them",
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


def read_below_header(path: str | Path, sheet_name: str) -> pd.DataFrame:
    """One sheet's data rows, the header row FOUND by its first cell (``HEADER_CELLS``).

    Every sheet of the workbook carries a free-text preamble above its header, and some
    of those preamble lines start in column 0 too, so the row is found by an exact match
    on the header's own first cell and a second match is refused.
    """
    header_cell = HEADER_CELLS[sheet_name]
    sheet = pd.read_excel(path, sheet_name=sheet_name, header=None)
    rows = sheet.index[sheet[0] == header_cell]
    if len(rows) != 1:
        raise ValueError(
            f"{sheet_name}: expected one {header_cell!r} header row, found {len(rows)}"
        )
    start = int(rows[0])
    table = sheet.iloc[start + 1 :].copy()
    table.columns = sheet.iloc[start].tolist()
    return table.loc[table[header_cell].notna()].reset_index(drop=True)


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
#: What every stress record says about the vehicle it stores (SOLVENT_COLUMN,
#: SOLVENT_IS_THE_PRESCREEN_STOCK): the solvent is Table S4's, recorded for the
#: wild-type IC50 prescreen, and the release never ties it to the fitness assays.
STRESS_DESCRIPTION = (
    "Stress compound added to LB. Vehicle DERIVED: Supplementary Table 4's Solvent for "
    "this compound, which is the solvent of the stock used for the wild-type IC50 "
    "prescreen; the paper does not state that the mutant fitness assays drew on those "
    "stocks. The final vehicle fraction in the medium is not released, so "
    "Solvent.percent is None"
)


def read_solvents(table_s4: str | Path) -> dict[str, str]:
    """Supplementary Table 4's ``Solvent`` per compound, keyed by lowercased name.

    The key is the compound name casefolded and stripped, because Table S5's Condition_1
    and Table S4's Compound differ in case for 6 of the 35 kept stress labels (``benzoic
    acid``, ``methylglyoxal``, ...). A sheet with an empty Solvent cell, or with two
    compounds differing only in case, is refused rather than resolved by a rule.
    """
    table = read_below_header(table_s4, TABLE_S4_SHEET)
    solvents: dict[str, str] = {}
    for _, row in table.iterrows():
        compound = str(row["Compound"]).strip()
        if pd.isna(row["Solvent"]):
            raise ValueError(f"{TABLE_S4_SHEET}: {compound!r} has no Solvent")
        key = compound.lower()
        if key in solvents:
            raise ValueError(f"{TABLE_S4_SHEET}: two rows name {key!r}")
        solvents[key] = str(row["Solvent"]).strip()
    return solvents


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


def stress_solvent(spec: SampleSpec, solvents: Mapping[str, str]) -> Solvent:
    """The vehicle of a stress sample's compound, from Table S4's ``Solvent``.

    A kept stress sample whose Condition_1 names no Table S4 compound is refused: all 55
    match by this rule, so a miss means the release changed, not that a vehicle is
    unknown. ``water`` is not in the pinned compound-identity table, so its ``Compound``
    carries the resolver's typed ``inchikey`` gap, as the unresolved stress labels do.
    """
    if spec.condition is None:
        raise ValueError(f"{spec.name}: a stress sample names no condition")
    key = spec.condition.strip().lower()
    if key not in solvents:
        raise ValueError(
            f"{spec.name}: {spec.condition!r} is not a {TABLE_S4_SHEET} compound"
        )
    name = solvents[key]
    return Solvent(name=name, percent=None, compound=resolved_compound(name))


def build_environment(spec: SampleSpec, solvents: Mapping[str, str]) -> Environment:
    """The environment of a kept sample: library medium, temperature, Condition_1.

    A carbon or nitrogen source is the varied factor of a medium that leaves it out
    (``CARBON_FREE_MEDIA``), so it is an ``EnvironmentPhysicalPerturbation`` with the
    compound as ``agent``; a stress compound is an added ``SmallMoleculePerturbation``
    carrying ``solvents``'s vehicle for it (:func:`stress_solvent`); the plain LB samples
    carry no perturbation.
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
                description=STRESS_DESCRIPTION,
                compound=resolved_compound(spec.condition),
                concentration=_dose(spec),
                solvent=stress_solvent(spec, solvents),
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
    min_fraction: float | None = None,
) -> tuple[dict[str, GeneMapping], IdentifierReport]:
    """Place the release's b-numbers on BW25113 locus tags and report how.

    ``reconcile`` runs the placed genes' ECK ids through ``reconcile_locus_tags`` on
    BW25113 and must return the crosswalk's tag for every one; ``name_of`` gives a
    tag's stored common name. The route must place at least ``min_fraction`` of the
    genes, else the build stops (checklist item 4) rather than dropping more. The
    essentiality table has its own, lower floor
    (``MIN_ESSENTIAL_ECK_ROUTE_FRACTION``): four of its 324 genes are the operons
    BW25113 deletes. ``min_fraction`` defaults to ``MIN_ECK_ROUTE_FRACTION`` READ AT
    CALL TIME, not bound into the signature, so a test can move the constant.
    """
    minimum = MIN_ECK_ROUTE_FRACTION if min_fraction is None else min_fraction
    placed, unplaced = eck_route(b_numbers, crosswalk, mg1655_ecks)
    fraction = len(placed) / len(b_numbers)
    if fraction < minimum:
        raise LocusTagResolutionError(
            f"{label}: the ECK route places {len(placed)} of {len(b_numbers)} genes "
            f"({fraction:.4f}), below {minimum}"
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
        min_fraction=minimum,
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
    min_fraction: float | None = None,
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
        min_fraction=min_fraction,
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
    solvents: Mapping[str, str],
) -> list[SampleRecords]:
    """The kept samples, in Table S5 order, each with its release column."""
    columns = {name: j for j, name in enumerate(release.samples)}
    out: list[SampleRecords] = []
    for spec in specs:
        if spec.drop_rule is not None:
            continue
        environment = build_environment(spec, solvents)
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
    stress_solvent_by_sample: dict[str, int] = Field(
        description="Table S4 Solvent -> kept stress samples whose compound names it"
    )
    stress_solvent_by_compound: dict[str, int] = Field(
        description="Table S4 Solvent -> distinct kept stress Condition_1 labels"
    )


def inventory(
    specs: Sequence[SampleSpec],
    identifiers: IdentifierReport,
    solvents: Mapping[str, str],
) -> Inventory:
    """Count samples, genes and records kept and dropped, by rule and by source."""
    kept = [s for s in specs if s.drop_rule is None]
    stress = [s for s in kept if GROUP_KINDS[s.group] == "stress"]
    by_sample = Counter(stress_solvent(s, solvents).name for s in stress)
    by_compound = Counter(
        solvents[c.strip().lower()]
        for c in {s.condition for s in stress if s.condition is not None}
    )
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
        stress_solvent_by_sample=dict(sorted(by_sample.items())),
        stress_solvent_by_compound=dict(sorted(by_compound.items())),
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
        solvents = read_solvents(osp.join(self.raw_dir, "si3.xlsx"))
        samples = sample_records(
            self.name, specs, release, assembly_reference(REFERENCE_STRAIN), solvents
        )
        counts = inventory(specs, identifiers, solvents)

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

    The shared ``canonical_gene_names`` row used to require status ``current``, which a
    pseudogene locus never has (the bacterial resolver answers ``non_gene_feature``,
    naming the same tag), so this row was written to carry the real check. That row now
    accepts a non-gene feature that resolves to ITSELF and counts them, so this one is
    its stricter restatement over the stored tags alone: it is added beside the shared
    row, never in its place.
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
    solvents = read_solvents(source_path("si3.xlsx", root))
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
        "inventory": inventory(specs, identifiers, solvents).model_dump(mode="json"),
        "identifiers": identifiers.model_dump(mode="json"),
        "standard_error": {
            "n_values": release.n_values,
            "n_naive_larger": release.n_naive_larger,
            "fraction_naive_larger": release.n_naive_larger / release.n_values,
            "t_max_abs_deviation": release.t_max_abs_deviation,
        },
        "set_prefixes": sorted({s.name.split("IT")[0] for s in specs}),
    }


# --------------------------------------------------------------------------- #
# Supplementary Table 1: the likely-essential genes
# --------------------------------------------------------------------------- #
ESSENTIAL_DATASET_ROOT_REL = "data/torchcell/gene_essentiality_price2018_ecoli"
#: Table S1's ``orgId == "Keio"`` rows, and the records they become (measured
#: 2026-10-07): 324 released rows, 320 of them on a BW25113 locus.
ESSENTIAL_SOURCE_GENES = 324
ESSENTIAL_EXPECTED_RECORDS = 320
#: The ECK route must place at least this fraction of Table S1's Keio genes. It places
#: 320 of 324 (0.9877), BELOW the fitness tables' 0.99: the four it cannot place are
#: araA, araB, rhaA and rhaB, whose operons BW25113 deletes, so they can carry no
#: insertion and the release's no-insertion rule calls them essential. A floor of 0.98
#: still stops a wrong annotation while admitting those four.
MIN_ESSENTIAL_ECK_ROUTE_FRACTION = 0.98
#: The four Table S1 Keio rows BW25113 has no locus for: the deleted araBAD and rhaBAD
#: operons (measured; their Table S1 ``name`` values are araA, araB, rhaA and rhaB).
#: Table S1 leaves their ``locus_tag`` empty and fills it for the other 320, which
#: :func:`essentiality_inventory` checks against the ECK route's own misses. The reason
#: is not asserted from the literature: ``BW25113_BACKGROUND_LESIONS`` carries
#: ``(araBAD)567`` and ``(rhaBAD)568`` verbatim from the ``source`` feature of the
#: sha256-pinned ``ecoli_K12_BW25113_ASM75055v1`` GenBank file, and the inventory ties
#: each dropped gene to the lesion that accounts for it.
DELETED_OPERON_B_NUMBERS: Final = ("b0062", "b0063", "b3903", "b3904")

ESSENTIAL_DESCRIPTION = (
    "Tn5 transposon library; gene-level TnSeq call, NOT a measured value: this gene got "
    "no fitness value and both its normalized insertion density and its normalized read "
    "density were under 0.2 (ESSENTIAL_RULE). The paper's label is 'essential or "
    "important for growth (nearly essential)' in the library-isolation condition (LB "
    "plates at 37 C), and its own false-discovery rate for the E. coli list is 6% to "
    "16% (ESSENTIAL_LABEL, ESSENTIAL_CONDITION, ESSENTIAL_FDR). The barcodes are not "
    "used in this analysis, so no strain is identified. Locus tag DERIVED: the release "
    "names an MG1655 b-number, mapped to this BW25113 locus through its one-to-one ECK "
    "pair (eck_crosswalk)"
)


#: The quotes written verbatim to ``preprocess/essentiality_label.json``: everything the
#: one stored boolean does not say.
ESSENTIALITY_LABEL_VALUES: Final = (
    "ESSENTIAL_TABLE",
    "ESSENTIAL_LABEL",
    "ESSENTIAL_RULE",
    "ESSENTIAL_IGNORES_BARCODES",
    "ESSENTIAL_CONDITION",
    "ESSENTIAL_CONDITION_TEMPERATURE",
    "ESSENTIAL_FDR",
    "ESSENTIAL_FDR_NAIVE",
    "ESSENTIAL_BENCHMARK",
    "ESSENTIAL_PER_ORGANISM_RANGE",
    "FITNESS_GENES_ARE_NON_ESSENTIAL",
)


class EssentialGene(BaseModel):
    """One Table S1 ``orgId == "Keio"`` row: the call and the coverage behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    row: int = Field(description="1-based position among the Keio rows")
    locus_id: str = Field(description="the compendium's internal gene id")
    b_number: str = Field(description="sysName, an MG1655 b-number")
    refseq_locus_tag: str | None = Field(
        description="locus_tag, a BW25113 RefSeq tag; empty for the deleted operons"
    )
    name: str
    desc: str
    gene_class: str
    gc: float
    n_reads: int
    normreads: float
    n_pos_central: int
    dens: float


def read_essential_genes(table_s1: str | Path) -> tuple[EssentialGene, ...]:
    """Table S1's ``Keio`` rows in sheet order, with the coverage columns.

    Every Keio ``sysName`` is an MG1655 b-number (``ESSENTIAL_TABLE_COLUMNS``), so a
    value that is not raises rather than being placed by some other route. ``GC``,
    ``nReads``, ``normreads``, ``nPosCentral`` and ``dens`` are the evidence the call is
    computed from, kept for ``preprocess/essential_genes.csv`` and stored on no record.
    """
    table = read_below_header(table_s1, TABLE_S1_SHEET)
    keio = table.loc[table["orgId"] == ORG_ID]
    if keio.empty:
        raise ValueError(f"{TABLE_S1_SHEET}: no {ORG_ID} rows")
    pattern = LOCUS_TAG_PATTERNS[STRAIN_GENE_NAMESPACES["MG1655"]]
    genes: list[EssentialGene] = []
    for row, (_, record) in enumerate(keio.iterrows(), start=1):
        b_number = str(record["sysName"])
        if not pattern.match(b_number):
            raise ValueError(f"{TABLE_S1_SHEET} row {row}: {b_number!r} is no b-number")
        genes.append(
            EssentialGene(
                row=row,
                locus_id=str(record["locusId"]),
                b_number=b_number,
                refseq_locus_tag=_optional_str(record["locus_tag"]),
                name=str(record["name"]),
                desc=str(record["desc"]),
                gene_class=str(record["geneClass"]),
                gc=float(record["GC"]),
                n_reads=int(record["nReads"]),
                normreads=float(record["normreads"]),
                n_pos_central=int(record["nPosCentral"]),
                dens=float(record["dens"]),
            )
        )
    return tuple(genes)


def selection_medium() -> Media:
    """The KEIO_ML9 selection plates: Price's LB (Lennox) plus agar and kanamycin.

    Supplementary Note 1 says the E. coli library was isolated by "growth on LB plates",
    so the medium is SOLID; Table S20 gives the medium (``LB``, which Price's Table S18
    states at 5 g/L NaCl, hence ``LB_LENNOX``'s components) and the antibiotic with its
    dose ("Kanamycin; 50", the column's own unit being ug/mL). The agar amount is never
    stated, so its concentration is ``None``: ``LB_AGAR``'s 2% is Menasalvas 2025's and
    Schmidt 2016's bench value and asserting it here would fabricate a number. The
    object derives from the ``LB`` library key, so it joins there.
    """
    return Media(
        name="LB Lennox agar with kanamycin (Price 2018 KEIO_ML9 transposon-mutant "
        "selection plates; agar amount unstated)",
        state="solid",
        is_synthetic=False,
        base_medium="LB",
        components=[
            *LB_LENNOX.components,
            MediaComponent(
                compound=resolved_compound("agar"),
                role=MediaComponentRole.gelling_agent,
                concentration=None,
                provenance=[ESSENTIAL_CONDITION],
                note="the plates are solid ('growth on LB plates'); no agar amount is "
                "stated in the paper or in Table S18",
            ),
            MediaComponent(
                compound=resolved_compound("kanamycin"),
                role=MediaComponentRole.selection_agent,
                concentration=Concentration(
                    value=50.0, unit=ConcentrationUnit.ug_per_ml
                ),
                provenance=[SELECTION_MEDIA_COLUMN, LIBRARY],
                note="selects for the transposon's kanamycin-resistance marker; Table "
                "S20's 'Antibiotic; concentration (in ug/mL)' reads 'Kanamycin; 50'",
            ),
        ],
        provenance=[SELECTION_MEDIA_COLUMN, ESSENTIAL_CONDITION, MEDIA_TABLE],
    )


def essentiality_environment() -> Environment:
    """The library-isolation condition: the selection plates at 37 C.

    This is NOT one of the 147 fitness-assay environments: the call is made on the
    library as isolated (``ESSENTIAL_CONDITION``). Both durations are typed gaps, and
    ``aerobicity`` keeps the field default, since plates incubated in air are never
    stated and the field cannot be ``None``.
    """
    return Environment(
        media=selection_medium(),
        temperature=Temperature(value=float(ESSENTIAL_CONDITION_TEMPERATURE.value)),
        perturbations=[],
        provenance_gaps=[ESSENTIAL_DURATION_GAP, ESSENTIAL_GENERATIONS_GAP],
    )


class EssentialityInventory(BaseModel):
    """What Table S1 released for E. coli, what became a record, and the disjointness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_genes: int
    kept_genes: int
    dropped_genes: tuple[str, ...]
    dropped_gene_names: tuple[str, ...] = Field(
        description="Table S1's own `name` for each dropped gene (araA, araB, rhaA, rhaB)"
    )
    drop_rule: str
    drop_description: str
    gene_class_histogram: dict[str, int]
    rows_without_a_refseq_locus_tag: tuple[str, ...] = Field(
        description="Keio rows whose Table S1 locus_tag is empty: the release's own "
        "corroboration that BW25113 carries no locus for them"
    )
    dropped_genes_explained_by_background: dict[str, str] = Field(
        description="each dropped gene's Table S1 name -> the BW25113 background lesion "
        "(BW25113_BACKGROUND_LESIONS, verbatim from the pinned assembly's source "
        "feature) that accounts for the locus being absent"
    )
    fitness_genes: int = Field(description="genes of fit_logratios_good.tab")
    shared_with_fitness: tuple[str, ...] = Field(
        description="b-numbers in BOTH the essential list and the fitness table; empty "
        "by the release's own rule (ESSENTIAL_RULE, FITNESS_GENES_ARE_NON_ESSENTIAL)"
    )
    unencodable_quantities: dict[str, str] = Field(
        description="what Table S1 and Supplementary Note 1 state that no field of "
        "GeneEssentialityPhenotype can hold -> the verbatim quote stating it"
    )


def essentiality_inventory(
    genes: Sequence[EssentialGene],
    identifiers: IdentifierReport,
    fitness_b_numbers: Sequence[str],
) -> EssentialityInventory:
    """Count the released rows and the records, and prove the two gene sets disjoint.

    The disjointness is the release's OWN rule read back off the files: a gene is called
    likely-essential only where the fitness analysis produced no value for it
    (``ESSENTIAL_RULE``), and that analysis covers "non-essential protein-coding genes"
    only (``FITNESS_GENES_ARE_NON_ESSENTIAL``). An overlap means the two files
    disagree, so the build refuses rather than storing one gene under two contradicting
    phenotypes. The genes the ECK route cannot place must likewise be exactly the rows
    Table S1 leaves without a BW25113 ``locus_tag``; a difference means the route lost a
    gene the release does place, which is a mapping bug, not a deleted operon.
    """
    essential = {gene.b_number for gene in genes}
    shared = tuple(sorted(essential & set(fitness_b_numbers)))
    if shared:
        raise ValueError(
            f"{len(shared)} genes are both likely-essential and valued by the fitness "
            f"release: {shared[:10]}"
        )
    unmapped = tuple(u.b_number for u in identifiers.unmapped)
    without_tag = tuple(g.b_number for g in genes if g.refseq_locus_tag is None)
    if set(unmapped) != set(without_tag):
        raise ValueError(
            f"the genes the ECK route cannot place {sorted(unmapped)} are not the rows "
            f"Table S1 leaves without a locus_tag {sorted(without_tag)}"
        )
    names = {gene.b_number: gene.name for gene in genes}
    explained = {
        names[b]: lesion
        for b in unmapped
        for lesion in BW25113_BACKGROUND_LESIONS
        if names[b][:3].lower() in lesion.lower()
    }
    unexplained = sorted(set(names[b] for b in unmapped) - set(explained))
    if unexplained:
        raise ValueError(
            f"{unexplained} are absent from BW25113 for a reason its background "
            f"genotype does not state ({BW25113_BACKGROUND_GENOTYPE})"
        )
    return EssentialityInventory(
        source_genes=len(genes),
        kept_genes=identifiers.n_mapped,
        dropped_genes=unmapped,
        dropped_gene_names=tuple(names[b] for b in unmapped),
        drop_rule=DROP_NO_ECK_PAIR,
        drop_description=DROP_RULES[DROP_NO_ECK_PAIR],
        gene_class_histogram=dict(sorted(Counter(g.gene_class for g in genes).items())),
        rows_without_a_refseq_locus_tag=without_tag,
        dropped_genes_explained_by_background=explained,
        fitness_genes=len(set(fitness_b_numbers)),
        shared_with_fitness=shared,
        unencodable_quantities={
            "the label is 'nearly essential' too": ESSENTIAL_LABEL.quote,
            "the condition is the library isolation, not an assay": (
                ESSENTIAL_CONDITION.quote
            ),
            "the list's own false-discovery rate": ESSENTIAL_FDR.quote,
            "the naive rate against the PEC and Keio list": ESSENTIAL_FDR_NAIVE.quote,
            "the call rule and its threshold": ESSENTIAL_RULE.quote,
            "the coverage evidence behind each call": ESSENTIAL_TABLE_COLUMNS.quote,
        },
    )


def build_essentiality_genotype(mapping: GeneMapping) -> Genotype:
    """A Tn5 insertion in one gene of KEIO_ML9, the call's caveat in its description."""
    return Genotype(
        perturbations=[
            TransposonInsertionPerturbation(
                systematic_gene_name=mapping.locus_tag,
                perturbed_gene_name=mapping.perturbed_gene_name,
                gene_namespace=STRAIN_GENE_NAMESPACES[REFERENCE_STRAIN],
                description=ESSENTIAL_DESCRIPTION,
                transposon=TRANSPOSON,
                library_pool=MUTANT_LIBRARY,
            )
        ]
    )


def build_essentiality_experiment(
    dataset_name: str, mapping: GeneMapping, environment: Environment
) -> BacterialGeneEssentialityExperiment:
    """One likely-essential gene: ``is_essential=True``, with the label's caveat."""
    return BacterialGeneEssentialityExperiment(
        dataset_name=dataset_name,
        genotype=build_essentiality_genotype(mapping),
        environment=environment,
        phenotype=GeneEssentialityPhenotype(is_essential=True),
    )


def build_essentiality_reference(
    dataset_name: str,
    genome_reference: AssemblyReferenceGenome,
    environment: Environment,
) -> BacterialGeneEssentialityExperimentReference:
    """The unperturbed BW25113 parent on the same plates: viable, by construction.

    The library was built in that parent and isolated on those plates
    (``ESSENTIAL_CONDITION``), so its viability there is a fact of the experiment rather
    than an inference.
    """
    return BacterialGeneEssentialityExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=environment.model_copy(),
        phenotype_reference=GeneEssentialityPhenotype(is_essential=False),
    )


@register_dataset
class GeneEssentialityPrice2018EcoliDataset(ExperimentDataset):
    """Price 2018 Supplementary Table 1: E. coli BW25113 likely-essential genes."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = "data/torchcell/gene_essentiality_price2018_ecoli",
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
        return BacterialGeneEssentialityExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialGeneEssentialityExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Table S1's workbook and the fitness table the disjointness check reads."""
        return ["si3.xlsx", "fit_logratios_good.tab"]

    def download(self) -> None:
        """Link each pinned mirror file into ``raw/`` after verifying its sha256."""
        os.makedirs(self.raw_dir, exist_ok=True)
        for name in self.raw_file_names:
            pin = RAW_FILE_NAMES[name][2]
            link_verified(source_path(name), osp.join(self.raw_dir, name), pin)
        log.info("Price 2018 Table S1 raw files linked into %s", self.raw_dir)

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
        """Build one record per mapped Table S1 Keio gene and the preprocess reports."""
        data_root = _data_root()
        verify_raw_files(
            self.raw_dir,
            {name: RAW_FILE_NAMES[name][2] for name in self.raw_file_names},
        )
        genes = read_essential_genes(osp.join(self.raw_dir, "si3.xlsx"))
        fitness = pd.read_csv(
            osp.join(self.raw_dir, "fit_logratios_good.tab"),
            sep="\t",
            usecols=["sysName"],
        )
        mg1655 = bacterial_genome("ecoli", "MG1655", data_root)
        if not isinstance(mg1655, EcoliK12MG1655Genome):
            raise TypeError(f"expected the MG1655 genome, got {type(mg1655).__name__}")
        mapping, identifiers = map_genes(
            [gene.b_number for gene in genes],
            mg1655,
            self._bw25113(),
            label=self.name,
            min_fraction=MIN_ESSENTIAL_ECK_ROUTE_FRACTION,
        )
        counts = essentiality_inventory(
            genes, identifiers, [str(v) for v in fitness["sysName"]]
        )

        environment = essentiality_environment()
        reference = build_essentiality_reference(
            self.name, assembly_reference(REFERENCE_STRAIN), environment
        )
        publication = PUBLICATIONS[SampleSource.price2018]
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for gene in tqdm(genes, desc="Price 2018 Table S1"):
                placed = mapping.get(gene.b_number)
                if placed is None:
                    continue
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(
                        build_essentiality_experiment(self.name, placed, environment),
                        reference,
                        publication,
                        itxn,
                    ),
                )
                idx += 1
        env.close()
        interned_env.close()
        if idx != counts.kept_genes:
            raise RuntimeError(
                f"wrote {idx} records, inventory says {counts.kept_genes}"
            )
        self._write_essentiality_reports(genes, mapping, identifiers, counts)
        log.info("Wrote %d Price 2018 likely-essential gene records", idx)

    def _write_essentiality_reports(
        self,
        genes: Sequence[EssentialGene],
        mapping: Mapping[str, GeneMapping],
        identifiers: IdentifierReport,
        counts: EssentialityInventory,
    ) -> None:
        """``dropped_records.json``, ``identifier_mapping.json``,
        ``essentiality_label.json`` and the per-row ``essential_genes.csv``.

        ``essentiality_label.json`` is where the label's caveat lives in full: the
        verbatim label, condition and FDR quotes with their source and sha256, so a
        reader of the store can find what the stored boolean does not say.
        """
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(counts.model_dump_json(indent=2))
        (out / "identifier_mapping.json").write_text(
            identifiers.model_dump_json(indent=2)
        )
        (out / "essentiality_label.json").write_text(
            json.dumps(
                {
                    name: SOURCED_VALUES[name].model_dump(mode="json")
                    for name in ESSENTIALITY_LABEL_VALUES
                },
                indent=2,
            )
        )
        rows: list[dict[str, Any]] = []
        record = 0
        for gene in genes:
            placed = mapping.get(gene.b_number)
            rows.append(
                {
                    **gene.model_dump(mode="json"),
                    "locus_tag": None if placed is None else placed.locus_tag,
                    "eck": None if placed is None else placed.eck,
                    "perturbed_gene_name": (
                        None if placed is None else placed.perturbed_gene_name
                    ),
                    "drop_rule": None if placed is not None else DROP_NO_ECK_PAIR,
                    "record": None if placed is None else record,
                }
            )
            if placed is not None:
                record += 1
        pd.DataFrame(rows).astype({"record": "Int64"}).to_csv(
            out / "essential_genes.csv", index=False
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled by ``build_essentiality_experiment``."""
        raise NotImplementedError(
            "Price 2018 essentiality builds its records in process()"
        )


# --------------------------------------------------------------------------- #
# Essentiality verification (L0 to L4) and its evidence report
# --------------------------------------------------------------------------- #
ESSENTIAL_VERIFY_PROVENANCE = Provenance(
    source_uri=SUPPLEMENTARY_TABLES,
    citation_key=CITATION_KEY,
    sha256=SUPPLEMENTARY_TABLES_SHA256,
    method=(
        "Supplementary Table 1's likely-essential protein-coding genes, orgId Keio: a "
        "TnSeq no-insertion call ('essential or important for growth (nearly "
        "essential)'), FDR 6% to 16%; MG1655 b-numbers placed on BW25113 through "
        "one-to-one ECK pairs"
    ),
    page=f"{TABLE_S1_SHEET}, orgId Keio",
)


def essentiality_calls_match_table(
    records: Sequence[Mapping[str, Any]], genes: Sequence[EssentialGene]
) -> LevelResult:
    """L2: one single-insertion ``is_essential=True`` record per stored locus.

    Table S1 lists only likely-essential genes, so there is no False to check against;
    what the records can get wrong is storing a locus twice, storing a False, carrying
    more than one perturbation, or a reference that is not the viable parent.
    """
    stored = Counter(
        p["systematic_gene_name"]
        for r in records
        for p in r["experiment"]["genotype"]["perturbations"]
    )
    problems: list[str] = []
    repeated = sorted(tag for tag, n in stored.items() if n > 1)
    if repeated:
        problems.append(f"{len(repeated)} loci repeat: {repeated[:10]}")
    not_true = sum(
        1 for r in records if not r["experiment"]["phenotype"]["is_essential"]
    )
    if not_true:
        problems.append(f"{not_true} records store is_essential=False")
    multi = sum(
        1 for r in records if len(r["experiment"]["genotype"]["perturbations"]) != 1
    )
    if multi:
        problems.append(f"{multi} records carry more than one perturbation")
    references = {
        r["reference"]["phenotype_reference"]["is_essential"] for r in records
    }
    if references != {False}:
        problems.append(f"the reference phenotype is {references}, expected viable")
    if len(stored) != len(records):
        problems.append(f"{len(records)} records over {len(stored)} loci")
    return LevelResult(
        level=Level.L2,
        name="calls_match_table_s1",
        passed=not problems,
        message=(
            f"{len(records)} records, one likely-essential locus each, from "
            f"{len(genes)} released Keio rows"
            if not problems
            else "; ".join(problems)
        ),
        details={
            "n_records": len(records),
            "n_loci": len(stored),
            "n_released_rows": len(genes),
            "problems": problems,
        },
    )


def essential_genes_are_not_fitness_genes(
    genes: Sequence[EssentialGene], fitness_b_numbers: Sequence[str]
) -> LevelResult:
    """L3: the essential and fitness gene sets are disjoint, as the release's rule says.

    Re-read from the two files rather than from the build's own report, so the row is
    evidence and not an echo of it.
    """
    shared = sorted({g.b_number for g in genes} & set(fitness_b_numbers))
    return LevelResult(
        level=Level.L3,
        name="disjoint_from_the_fitness_genes",
        passed=not shared,
        message=(
            f"{len(genes)} likely-essential genes and {len(set(fitness_b_numbers))} "
            "genes with fitness values share none"
            if not shared
            else f"{len(shared)} genes are in both sets: {shared[:10]}"
        ),
        details={
            "n_essential": len(genes),
            "n_fitness": len(set(fitness_b_numbers)),
            "shared": shared[:20],
            "rule": ESSENTIAL_RULE.quote,
        },
    )


def essentiality_label_is_qualified(
    records: Sequence[Mapping[str, Any]],
) -> LevelResult:
    """L3: every record restates the label's caveat on its perturbation description.

    ``GeneEssentialityPhenotype`` is one boolean, so a record read without this text
    would say "essential" where the paper says "essential or important for growth
    (nearly essential)" at a 6% to 16% false-discovery rate. The row fails if any record
    drops it.
    """
    missing = sum(
        1
        for r in records
        for p in r["experiment"]["genotype"]["perturbations"]
        if p["description"] != ESSENTIAL_DESCRIPTION
    )
    return LevelResult(
        level=Level.L3,
        name="label_caveat_on_every_record",
        passed=not missing,
        message=(
            f"all {len(records)} records carry the 'nearly essential' label and the "
            "6% to 16% FDR on their perturbation description"
            if not missing
            else f"{missing} perturbations do not carry the label caveat"
        ),
        details={"n_records": len(records), "n_missing": missing},
    )


def verify_essentiality_records(
    records: Sequence[Mapping[str, Any]],
    *,
    genes: Sequence[EssentialGene],
    fitness_b_numbers: Sequence[str],
    universe: set[str],
    resolve_gene_name: Callable[[str], GeneNameResolution],
    expected_count: int,
    dataset_name: str = "GeneEssentialityPrice2018EcoliDataset",
) -> VerificationReport:
    """The L0 to L4 gate over built essentiality records, plus the shared rules."""
    from torchcell.verification.common import shared_rule_results

    report = VerificationReport(
        dataset_name=dataset_name, provenance=ESSENTIAL_VERIFY_PROVENANCE
    )
    report.add(
        l0_structural(
            (r["experiment"] for r in records),
            BacterialGeneEssentialityExperiment.model_validate,
        )
    )
    report.add(l1_count(len(records), expected_count))
    report.add(essentiality_calls_match_table(records, genes))
    report.add(essential_genes_are_not_fitness_genes(genes, fitness_b_numbers))
    report.add(essentiality_label_is_qualified(records))
    for result in shared_rule_results(
        records,
        resolve_gene_name=resolve_gene_name,
        sgd_genes=universe,
        gene_universe_label="BW25113",
        min_containment=1.0,
    ):
        report.add(result)
    tags = {
        str(p["systematic_gene_name"])
        for r in records
        for p in r["experiment"]["genotype"]["perturbations"]
    }
    report.add(stored_tags_are_loci(tags, resolve_gene_name))
    return report


def verify_essentiality(data_root: str | None = None) -> VerificationReport:
    """Verify the built essentiality LMDB (L0 to L4) and write its report.

    The gene universe and resolver are BW25113's, as for the fitness dataset. Table S1
    and the fitness table are re-read from this dataset's own ``raw/``, so the L2 and L3
    rows check the records against the pinned files rather than against the build.
    """
    from torchcell.verification.runners import load_records

    root = _data_root(data_root)
    abs_root = osp.join(root, ESSENTIAL_DATASET_ROOT_REL)
    genes = read_essential_genes(osp.join(abs_root, "raw", "si3.xlsx"))
    fitness = pd.read_csv(
        osp.join(abs_root, "raw", "fit_logratios_good.tab"),
        sep="\t",
        usecols=["sysName"],
    )
    genome = bacterial_genome("ecoli", REFERENCE_STRAIN, root)
    report = verify_essentiality_records(
        load_records(abs_root),
        genes=genes,
        fitness_b_numbers=[str(v) for v in fitness["sysName"]],
        universe=set(genome.genbank.loci),
        resolve_gene_name=genome.resolve_gene_name,
        expected_count=ESSENTIAL_EXPECTED_RECORDS,
    )
    out = osp.join(abs_root, "preprocess", "verification_report.json")
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def refseq_route_agreement(
    genes: Sequence[EssentialGene],
    mapping: Mapping[str, GeneMapping],
    resolve_gene_name: Callable[[str], GeneNameResolution],
) -> dict[str, Any]:
    """Table S1's own ``locus_tag`` route against the ECK route, gene by gene.

    A second, independent identifier route. Table S1 gives each gene a BW25113 RefSeq
    locus tag, which the deposited BW25113 annotation carries as its own RefSeq layer,
    so where that tag resolves it must name the locus the ECK route chose. Where it does
    not resolve (a tag this annotation release retired) the route is silent rather than
    contradicting. Reported, not enforced: the RefSeq layer is a second release of the
    same assembly, and the stored namespace is the GenBank one.
    """
    agree: list[str] = []
    unresolved: list[str] = []
    disagree: list[tuple[str, str | None, str]] = []
    for gene in genes:
        placed = mapping.get(gene.b_number)
        if placed is None or gene.refseq_locus_tag is None:
            continue
        resolution = resolve_gene_name(gene.refseq_locus_tag)
        if resolution.systematic_name == placed.locus_tag:
            agree.append(gene.b_number)
        elif resolution.status is GeneNameStatus.RETIRED:
            unresolved.append(gene.b_number)
        else:
            disagree.append(
                (gene.b_number, resolution.systematic_name, placed.locus_tag)
            )
    return {
        "n_agree": len(agree),
        "n_unresolved_refseq_tag": len(unresolved),
        "unresolved": unresolved,
        "disagree": disagree,
    }


def essentiality_report(data_root: str | None = None) -> dict[str, Any]:
    """Every essentiality number the dendron note states, from the pinned files."""
    root = _data_root(data_root)
    genes = read_essential_genes(source_path("si3.xlsx", root))
    fitness = pd.read_csv(
        source_path("fit_logratios_good.tab", root), sep="\t", usecols=["sysName"]
    )
    mg1655 = bacterial_genome("ecoli", "MG1655", root)
    bw25113 = bacterial_genome("ecoli", REFERENCE_STRAIN, root)
    if not isinstance(mg1655, EcoliK12MG1655Genome) or not isinstance(
        bw25113, EcoliK12BW25113Genome
    ):
        raise TypeError("the K-12 genomes resolved to the wrong classes")
    mapping, identifiers = map_genes(
        [gene.b_number for gene in genes],
        mg1655,
        bw25113,
        label="essentiality report",
        min_fraction=MIN_ESSENTIAL_ECK_ROUTE_FRACTION,
    )
    counts = essentiality_inventory(
        genes, identifiers, [str(v) for v in fitness["sysName"]]
    )
    return {
        "inventory": counts.model_dump(mode="json"),
        "identifiers": identifiers.model_dump(mode="json"),
        "refseq_route_agreement": refseq_route_agreement(
            genes, mapping, bw25113.resolve_gene_name
        ),
        "table_s1_names_differing_from_the_genome": sorted(
            f"{gene.b_number} {gene.name} -> {mapping[gene.b_number].perturbed_gene_name}"
            for gene in genes
            if gene.b_number in mapping
            and gene.name != mapping[gene.b_number].perturbed_gene_name
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """``retrieve`` / ``deposit`` the release files, or ``report`` / ``verify`` either
    dataset (``essentiality-report``, ``essentiality-verify`` for Table S1's).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    retrieve = commands.add_parser("retrieve", help="fetch the release files")
    retrieve.add_argument("--dest", required=True)
    deposit = commands.add_parser("deposit", help="deposit fetched files")
    deposit.add_argument("--source-dir", required=True)
    commands.add_parser("report", help="print the fitness evidence as JSON")
    commands.add_parser("verify", help="verify the built fitness LMDB (L0 to L4)")
    commands.add_parser(
        "essentiality-report", help="print the Table S1 evidence as JSON"
    )
    commands.add_parser(
        "essentiality-verify", help="verify the built essentiality LMDB (L0 to L4)"
    )
    args = parser.parse_args(argv)
    if args.command == "retrieve":
        print(retrieve_raw_files(args.dest))
    elif args.command == "deposit":
        print(deposit_raw_mirror(source_dir=args.source_dir))
    elif args.command == "report":
        print(json.dumps(report(), indent=2))
    elif args.command == "essentiality-report":
        print(json.dumps(essentiality_report(), indent=2))
    elif args.command == "essentiality-verify":
        print(verify_essentiality().summary())
    else:
        print(verify().summary())


if __name__ == "__main__":
    main()
