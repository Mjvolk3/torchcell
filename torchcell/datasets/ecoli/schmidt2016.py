# torchcell/datasets/ecoli/schmidt2016
# [[torchcell.datasets.ecoli.schmidt2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/schmidt2016
# Test file: tests/torchcell/datasets/ecoli/test_schmidt2016.py
"""Schmidt 2016, the quantitative condition-dependent E. coli proteome (row 32).

Schmidt et al. 2016 (Nat Biotechnol 34:104, doi:10.1038/nbt.3418, PMID 26641532)
measured absolute protein abundances for more than 2,300 proteins of *E. coli* K-12
BW25113 across 22 growth conditions in biological triplicates, and repeated the glucose
and LB conditions in MG1655 and NCM3722. :class:`ProteomeSchmidt2016Dataset` serves one
``BacterialProteinAbundanceExperiment`` per loaded BW25113 growth condition.

WHICH PER-PROTEIN QUANTITY IS STORED, AND ITS DEFINITION. The release carries three
per-protein quantities per condition and they are NOT interchangeable: ``Protein
copies/cell``, ``Protein Mass (fg) / Cell`` (the same number times the molecular
weight), and ``Coefficient of Variance (%) Between Biological Triplicates``. This
loader stores **protein copies per cell**, the source's own primitive: "Based on the
number of cells counted by fluorescenceactivated cell sorting for each sample, absolute
abundances for the selected proteins (in copies/cell) could be calculated across all
samples in both data sets (Supplementary Tables 2 and 3)." It is read from the ``Protein
copies/cell`` block of Table S6, the release's "Final table with combined global
absolute abundance estimations from both datasets". The mass block is not stored,
because it is derived from the stored number and the released molecular weight.

HOW THE UNCERTAINTY IS OBTAINED, AND WHY IT IS AN EXACT SE RATHER THAN THE RELEASED CV.
``ProteinAbundancePhenotype.protein_abundance_se`` is a standard error, and the release
publishes a coefficient of variation instead. Three facts, all measured on the pinned
workbook, turn one into the other without an assumption:

1. ``medianNormInt_<group>`` of Table S8 equals the median of that group's three
   ``normInt_`` replicate columns for every one of the 47,334 released cells, exactly.
2. ``cv_<group>`` equals ``100 * stdev / mean`` of those same three columns, with the
   99.9th percentile of the relative deviation at 5.6e-8 (the export carries ten
   significant digits). The only cells above 1e-6 are the 23 belonging to ``P63284``,
   which Table S6 files on two rows and this loader drops.
3. Within one condition the stored copies/cell is a per-protein multiple of that
   protein's ``medianNormInt``: ``copies[i, c] / copies[i, glucose]`` equals
   ``medianNormInt[i, c] / medianNormInt[i, glucose]`` for all 2,057 shared proteins and
   all 21 condition pairs, to a maximum relative deviation of 9.3e-10.

So ``scale[i, c] = copies[i, c] / medianNormInt[i, c]`` converts that protein's
replicate spread into copies per cell, and the stored SE is
``scale[i, c] * stdev(three normInt replicates) / sqrt(3)``. Facts 1 and 2 are asserted
at build time against the bytes; a drift in either stops the build.

``n_replicates`` IS PER PROTEIN AND THE RELEASE STATES IT. Table S6 combines two
LC-MS experiments and names which one each row came from in its ``Dataset`` column:
2,058 of 2,359 rows are dataset 2, which was "prepared in biological triplicates", and
301 are dataset 1, of which the release says "since this dataset contains no replicate
measurements, no statical tests were performed". A dataset-2 protein therefore stores
``n_replicates = 3`` and an exact SE; a dataset-1 protein stores ``n_replicates = 1``
and ``nan``, which is what no replicate measurement means.

EVERY LOADED CONDITION IS WILD-TYPE BW25113: NONE IS A DELETION STRAIN. The three Keio
deletion strains in the paper (``rimI``, ``rimJ``, ``rimL``) appear only in the
N-alpha-acetylation tables (Supplementary Tables 20, 21 and 24) and carry no per-protein
abundance at all, so no abundance record has a genotype perturbation. ``Genotype`` is
therefore empty for all 14 records and the varying axis is entirely the environment,
which is why the glucose arm is the ``phenotype_reference`` rather than a record: the
release defines every fold change against "BW25113 grown in minimal medium (glucose)".

FOURTEEN RECORDS FROM TWENTY-TWO RELEASED CONDITIONS, AND EACH DROP IS STRUCTURAL.
Seven of Table S6's 22 conditions are not loaded and one more is the reference:

- ``Glycerol + AA`` is M9 with glycerol plus 21 named supplements at stated doses, which
  has no ``MEDIA_LIBRARY`` entry. ``media.py`` is a value-surface file this branch does
  not edit, so the condition is left out and ``M9_GLYCEROL_AA_SCHMIDT2016`` is named in
  the PR as the addition it needs.
- The four ``Chemostat`` arms differ from each other ONLY in dilution rate, and
  ``Environment`` has no culture-mode or dilution-rate slot. Loading them would give
  four records one byte-identical environment, which is the rule the landed Lamoureux
  2023 loader already states as ``culture_not_batch``.
- ``Stationary phase 1 day`` and ``Stationary phase 3 days`` share medium, carbon
  source, temperature and oxygen regime with the exponential ``Glucose`` arm, and
  ``Environment`` has no growth-phase slot, so all three would collapse onto one
  environment.

That the remaining 15 environments are pairwise distinct is asserted at build time by
comparing their serialized bytes, so the rule is checked rather than claimed.

THE IDENTIFIER ROUTE IS DERIVED, SINGLE, AND THE RELEASED B-NUMBER IS NOT IT. The
records are BW25113 and the release's ``Bnumber`` column is an MG1655 identifier:
measured against the deposited annotations, 2,285 of 2,285 released b-numbers are
MG1655 locus tags and 0 of 2,285 are BW25113 locus tags. Keying on them would write
MG1655 identities onto BW25113 records, and ``ygbT``'s row shows the hazard directly --
it carries ``b2755`` while being an *Erwinia amylovora* protein. The stored key is
instead the released ``Gene`` symbol resolved against the pinned BW25113 GenBank
annotation through :func:`reconcile_locus_tags`: 2,336 of 2,347 unique host symbols
resolve (0.9953), 2,259 through the gene-symbol layer and 77 through a gene synonym,
with no ambiguity and no collision. Each b-number is kept in
``preprocess/protein_identifiers.csv`` as the source's cross-reference.

THE RETENTION LEDGER, WITH ITS ARITHMETIC. 2,359 released rows become 2,329 protein
keys: 2,359 - 5 - 14 - 11 = 2,329.

- 5 rows are not *E. coli* proteins. The search database was built that way -- "The
  database consists of 4,431 E. coli proteins as well as known contaminants" -- and the
  released ``Description`` names each row's organism in its ``OS=`` token: *Bacillus
  subtilis* (``addB``), *Erwinia amylovora* (``ygbT``), *Vibrio harveyi* (``cgtA``),
  *Peptoniphilus sp.* (``cas1``) and *Methylobacter tundripaludum*
  (``MettuDRAFT_4149``). Dropping them first is what leaves ``obgE`` a clean unique
  resolution, since four of the five share an E. coli gene synonym.
- 14 rows carry one of 7 gene symbols that the release files on two rows each
  (``nrdA``, ``rmlA``, ``mrcB``, ``rpmE``, ``clpB``, ``bioD``, ``glsA``). One symbol
  cannot name two protein groups. The release does state a preference for dataset 2
  that would pick a row for four of the seven, and it is NOT taken: ``rmlA`` and
  ``glsA`` are two distinct paralogs under one symbol in BOTH rows, so a uniform drop
  is the only rule that never mis-attributes a measurement.
- 11 symbols resolve to no BW25113 locus and are kept as given by the retain-all
  policy, so they have no gene to key an abundance to: ``JW58``, ``NA``, ``adeP``,
  ``coaB``, ``comR``, ``ghxP``, ``insC``, ``insH``, ``trmB``, ``trpG`` and ``zapE``.

Three records carry 2,034 keys rather than 2,329: ``Xylose``, ``Mannose`` and
``Fructose`` are the conditions dataset 1 did not cover, so the 295 kept dataset-1
proteins are released as ``NA`` there and are dropped from those records. Every
record's reference is key-matched to that record.

A HEADER FAULT IN THE RELEASED TABLE, FOUND BY MEASUREMENT AND NOT CONSUMED. Table S6's
CV block holds LB's coefficient of variation under the header ``Glucose`` and glucose's
under ``LB``; its remaining 20 CV headers are correct, and its copies/cell and mass
headers are all correct. Recomputing each CV from Table S8's replicate columns proves
the swap for all 2,058 dataset-2 rows, with zero exceptions. This loader reads no CV
column, so nothing stored depends on it; the swap is asserted at build time so the
finding stays pinned to the bytes.

TWO SIBLING LOADERS READ THE SAME WORKBOOK, FROM THEIR OWN MODULES.
:mod:`torchcell.datasets.ecoli.schmidt2016_srm` serves Tables S2 and S3, the 41-protein
SRM + stable-isotope-dilution panel the abundances above were anchored on (a different
assay of proteins this block also covers, under its own ``measurement_type``), and
:mod:`torchcell.datasets.ecoli.schmidt2016_growth_rate` serves Table S24, the
rim-deletion growth rates, which are this paper's only gene-perturbation phenotype. They
are separate MODULES on purpose: ``build_manifest`` keys a built store's staleness on the
schema closure of the loader module's own ``torchcell.datamodels`` imports, so adding
``FitnessPhenotype`` or ``BacterialDeletionPerturbation`` here would mark the already-served
``proteome_schmidt2016`` store stale for a change that touches none of its records. They
import the pinned artifact, the condition table, :func:`build_environment`,
:func:`check_environments_distinct`, :func:`publication` and the three supplementary
verification rules from this module, so each is stated once.

DATA. One file is consumed: ``si2.xlsx``, the publisher's Supplementary tables, from the
PMC OA Cloud bucket (``PMC4888949.1/NIHMS65833-supplement-Supplementary_tables.xlsx``),
deposited under ``$DATA_ROOT/torchcell-raw/<citation key>/data/si2.xlsx``. The retrieval
is scriptable and recorded, so ``retrieve_raw_files`` re-runs it. The raw mass spectra
live in ProteomeXchange PXD000498, which is recorded in the mirror manifest and which no
loader reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import os.path as osp
import pickle
import re
import shutil
import statistics
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import openpyxl
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    write_verified,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import LB, M9_SCHMIDT2016
from torchcell.datamodels.schema import (
    BacterialProteinAbundanceExperiment,
    BacterialProteinAbundanceExperimentReference,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    Genotype,
    Media,
    PhysicalFactor,
    ProteinAbundancePhenotype,
    Publication,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    BACTERIAL_ASSEMBLY_SETS,
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
from torchcell.literature.provenance import run_retriever
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import SourcedValue

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# The pinned artifacts
# --------------------------------------------------------------------------- #
CITATION_KEY = "schmidtQuantitativeConditiondependentEscherichia2016"
PAPER_DOI = "10.1038/nbt.3418"
PAPER_TITLE = "The quantitative and condition-dependent Escherichia coli proteome"
#: Resolved from the PMC id converter, which returns this PMID and the DOI above for
#: ``PMC4888949`` (the PMC record the mirrored SI was retrieved from).
PUBMED_ID = "26641532"
PMC_ID = "PMC4888949"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "67bedae8f934421086c23b7fa9582b0950a07206bffe7c4ffdd11e4d003a8710"

#: The one consumed file: the publisher's Supplementary tables workbook.
SI2 = "si2.xlsx"
SI2_SHA256 = "3280a13ff67a73f25440cff6ee73fb99b5ce3ef57854213dbbf6272be241912f"
SI2_BYTES = 17128596
SI2_PMC_KEY = f"{PMC_ID}.1/NIHMS65833-supplement-Supplementary_tables.xlsx"
SI2_URL = f"https://pmc-oa-opendata.s3.amazonaws.com/{SI2_PMC_KEY}"
SI2_RETRIEVED_AT = "2026-10-07T11:42:57.070583+00:00"
SI2_MIRROR_RELPATH = f"data/{SI2}"
#: ``{raw file name: pinned sha256}``, re-checked at the start of ``process()``.
DATA_SHA256: dict[str, str] = {SI2: SI2_SHA256}

SI2_RETRIEVAL = RetrievalRecord(
    method=RetrievalMethod.pmc_cloud,
    source_url=SI2_URL,
    retriever="torchcell.literature.retrieve.pmc_cloud_object",
    params={"key": SI2_PMC_KEY},
    sha256=SI2_SHA256,
    retrieved_at=SI2_RETRIEVED_AT,
)

#: Released locations this loader does NOT mirror, with why.
NOT_MIRRORED = (
    "ProteomeXchange PXD000498 (raw mass spectra and assigned MS/MS spectra, the "
    "deposit Supplementary Table 17 names): no loader reads spectra",
    "si/si1.pdf (Supplementary figures and notes): the sourcing quotes come from "
    "paper.md and from si2.xlsx, and no value is read from the figures",
)


# --------------------------------------------------------------------------- #
# Verbatim quotes. Every one is a substring of the pinned ``paper.md`` or of a
# ``si2.xlsx`` row rendering (its non-empty cells joined by " | "), OCR artifacts and
# the source's own misspellings included.
# --------------------------------------------------------------------------- #
_Q_STRAIN = (
    "The Escherichia coli K-12 strain BW25113 (genotype: F-, "
    "∆(araD-araB)567, ∆lacZ4787(:rrnB-3), λ-, rph-1, ∆(rhaD-rhaB)568, "
    "hsdR514)19 was used to generate the proteome map for all 22 "
    "conditions."
)
_Q_OTHER_STRAINS = (
    "Additionally, the proteome for the glucose and LB condition was also"
    " determined for the strains MG1655 (genotype: F-, λ, rph- $1 ) ^ { 2"
    " 0 }$ and NCM3722 (genotype: $\\mathrm { F } + ) ^ { 2 1 }$ ."
)
_Q_22_CONDITIONS = (
    "We grew E. coli BW25113 (ref. 19) under 22 different growth "
    "conditions in biological triplicates."
)
_Q_CARBON_SOURCES = (
    "The following carbon sources and concentrations were used: acetate "
    "(sodium acetate, $3 . 5 \\mathrm { g } / \\mathrm { L }$ ), fumarate"
    " (disodium fumarate, $2 . 8 \\mathrm { g } / \\mathrm { L }$ , "
    "galactose $( 2 . 3 \\mathrm { g } / \\mathrm { L } )$ , glucose $( 5"
    " \\mathrm { g } / \\mathrm { L } )$ , glucosamine $( 2 . 1 \\mathrm "
    "{ g } / \\mathrm { L } )$ , glycerol $( 2 . 2 \\mathrm { g } / "
    "\\mathrm { L } )$ , pyruvate (sodium pyruvate, $3 . 3 \\mathrm { g /"
    " L }$ ), sucnate (disodium succinate hexahydrate, $5 . 7 \\mathrm { "
    "g / L }$ , fructose $( 5 \\mathrm { g } / \\mathrm { L } )$ , "
    "mannose $( 5 \\mathrm { g } / \\mathrm { L } ) \\mathrm { g }$ and "
    "xylose $( 5 \\mathrm { g } / \\mathrm { L } )$ ."
)
_Q_CHEMOSTAT_GLUCOSE = (
    "For chemostat growth only $1 \\mathrm { g / L }$ of glucose was used."
)
_Q_OSMOTIC = (
    "Glucose minimal medium for the cells grown with osmotic stress was "
    "supplemented with $\\mathrm { { N a C l } }$ to a concentration of "
    "$5 0 \\mathrm { m M }$"
)
_Q_PH6 = (
    "for the cells grown with $\\mathrm { p H }$ stress, fuming "
    "hydrochloric acid was titrated to the medium until a $\\mathrm { p H"
    " }$ of 6 was reached."
)
_Q_BATCH_37C = (
    "For the batch cultures, the cells from a preculture were "
    "re-inoculated into $5 0 ~ \\mathrm { m l }$ of the appropriate "
    "pre-warmed medium in a $5 0 0 \\mathrm { - m l }$ unbaffled "
    "wide-neck Erlenmeyer flask covered by a $3 8 \\mathrm { - m m }$ "
    "silicone sponge closure (BellCo glass) and grown at $3 7 ^ { \\circ "
    "} \\mathrm { C } ,$ orbital shaking at $3 0 0 \\mathrm { r . p . m "
    "}$ . and 5-cm shaking diameter (ISF-4-V, Kühner)."
)
_Q_42C = (
    "The cells undergoing temperature stress were grown at $4 2 ^ { "
    "\\circ } \\mathrm { C }$"
)
_Q_TRIPLICATES = "All samples of data set 2 were prepared in biological triplicates."
_Q_COPIES_PER_CELL = (
    "Based on the number of cells counted by fluorescenceactivated cell "
    "sorting for each sample, absolute abundances for the selected "
    "proteins (in copies/cell) could be calculated across all samples in"
    " both data sets (Supplementary Tables 2 and 3)."
)
_Q_PROTEOME_WIDE = (
    "The absolute protein concentrations determined for 41 glycolytic "
    "proteins were aligned with the summed protein intensities as "
    "provided by the Progenesis LC-MS software (v4.0, Nonlinear Dynamics "
    "Limited) divided by the number of expected tryptic peptides as "
    "recently specified4,25."
)
_Q_DATASET2_PREFERRED = (
    "Owing to the higher number of quantified membrane proteins, higher "
    "number of growth conditions included and the analysis in biological "
    "triplicates (Supplementary Fig. 4), protein quantities obtained from"
    " data set 2 were employed for all quantitative analysis carry out in"
    " this study."
)
_Q_SEARCH_DATABASE = (
    "The generated mgf-files were searched using MASCOT against a decoy "
    "database (consisting of forward and reverse protein sequences) of "
    "the predicted proteome from E. coli (UniProt, download date: "
    "2012/07/20). The database consists of 4,431 E. coli proteins as well"
    " as known contaminants such as porcine trypsin, human keratins and "
    "high abundant bovine serum proteins (Uniprot), resulting in a total "
    "of 10,388 protein sequences."
)
_Q_STATIONARY = (
    "Starved cells were continuously shaken after reaching stationary "
    "phase for either 1 or $^ 3 \\mathrm { d }$ ."
)
_Q_CHEMOSTAT = (
    "Cells grown in a chemostat were inoculated from a preculture to an "
    "OD of 0.1 and allowed to grow in batch mode to an OD of around 0.8 "
    "before dilution (rates: 0.12, 0.2, 0.35, 0.5) was started52."
)
_Q_GLYCEROL_AA = (
    "The glycerol $^ +$ amino acid medium was made by supplementing the "
    "media with glycerol to a concentration of $2 . 2 \\ \\mathrm { g / L"
    " }$ individual amino acids: alanine, asparagine, cysteine, "
    "glutamate, glycine, proline and serine to final concentrations of "
    "alanine $1 . 0 \\mathrm { m g / L }$ ${ 0 . 0 } \\mathrm { m M } )$ "
    ", adenine $1 0 . 2 \\mathrm { m g / L ( 0 . 1 m M ) }$ , arginine $5"
    " 1 . 1 \\mathrm { m g / L } \\left( 0 . 3 \\mathrm { m M } \\right)$"
    " , asparagine $\\mathrm { 1 . 6 m g / L ( 0 . 0 1 m M ) }$ , "
    "aspartic acid $8 1 . 8 ~ \\mathrm { m g / L }$ $\\mathrm { { 0 . 6 ~"
    " m M } }$ , cysteine $1 . 2 ~ \\mathrm { m g / L }$ $( 0 . 0 1 \\ "
    "\\mathrm { m M } )$ , glutamate $1 5 . 2 ~ \\mathrm { m g / L }$ $0 "
    ". 1 \\mathrm { \\ m M }$ , glutamine $1 3 . 9 ~ \\mathrm { m g / L "
    "}$ $\\mathrm { { ( 0 . 1 ~ m M } }$ , glycine $0 . 4 ~ \\mathrm { m "
    "g / L }$ $\\mathrm { { ( 0 . 0 1 m M ) } }$ , histidine $2 0 . 5 "
    "\\mathrm { m g / L ( 0 . 1 m M ) }$ , isoleucine $5 1 . 1 \\mathrm {"
    " m g / L } \\left( 0 . 4 \\mathrm { m M } \\right)$ leucine $1 0 2 ."
    " 3 ~ \\mathrm { m g / L }$ $( 0 . 8 \\ \\mathrm { m M } )$ , lysine "
    "$5 1 . 1 ~ \\mathrm { m g / L }$ $\\mathrm { 0 . 4 ~ m M }$ , "
    "methionine $2 0 . 5 ~ \\mathrm { m g / L }$ $( 0 . 1 4 \\mathrm { m "
    "M } )$ , phenylalanine $5 1 . 1 \\mathrm { m g / L }$ $0 . 3 "
    "\\mathrm { m M }$ , proline $5 . 2 \\mathrm { m g / L }$ $( 0 . 0 5 "
    "\\mathrm { m M } )$ , serine $9 . 2 ~ \\mathrm { m g / L }$ $( 0 . 1"
    " \\mathrm { \\ m M }$ , threonine $1 0 2 . 3 ~ \\mathrm { m g / L }$"
    " $\\langle 0 . 9 \\mathrm { ~ m M } \\rangle$ , tryptophan $5 1 . 1 "
    "\\mathrm { m g / L }$ (0.3 mM), tyrosine $5 1 . 1 \\mathrm { m g / L"
    " }$ $\\left( 0 . 3 \\mathrm { m M } \\right)$ , valine $1 4 3 . 2 "
    "\\mathrm { m g / L } \\left( 1 . 2 \\mathrm { m M } \\right)$ and "
    "uracil $2 0 . 5 \\mathrm { m g / L }$ (0.2 mM)."
)
_Q_TABLE_S6 = (
    "Table S6 | Final table with combined global absolute abundance "
    "estimations from both datasets including functional annotations "
    "using cluster of orthologues groups (COG)"
)
_Q_TABLE_S7 = (
    "Table S7 | Global relative quantification of all proteins identifed "
    "in dataset 1 (see supplemental Figure 4 for dataset details) using "
    "label-free quantification and SafeQuant data analysis (since this "
    "dataset contains no replicate measurements, no statical tests were "
    "performed. Therefore, protein quantities in dataset 2 are of higher "
    "confidence and should be preferred)"
)
_Q_TABLE_S8 = (
    "Table S8 | Global relative quantification of all proteins identifed "
    "in dataset 2 (see supplemental Figure 4 for dataset details) "
    "inlcuding statiscal analysis from biological triplicates using "
    "label-free quantification and SafeQuant data analysis and the "
    "coefficient of variation for each protein across the growth "
    "conditons"
)
_Q_TABLE_S25 = (
    "Table S25 | Sample names for the individual replicates analyzed for "
    "each growth condition and strain included in this study"
)

PAPER = Provenance(
    source_uri=PAPER_MD, citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
)
SI2_SOURCE = Provenance(
    source_uri=SI2_MIRROR_RELPATH, citation_key=CITATION_KEY, sha256=SI2_SHA256
)

#: Quotes whose provenance is ``paper.md``, by the constant's own name. The test module
#: asserts every one is a substring of the pinned bytes.
PAPER_QUOTES: dict[str, str] = {
    "strain": _Q_STRAIN,
    "other_strains": _Q_OTHER_STRAINS,
    "twenty_two_conditions": _Q_22_CONDITIONS,
    "carbon_sources": _Q_CARBON_SOURCES,
    "chemostat_glucose": _Q_CHEMOSTAT_GLUCOSE,
    "osmotic": _Q_OSMOTIC,
    "ph6": _Q_PH6,
    "batch_37c": _Q_BATCH_37C,
    "temperature_42c": _Q_42C,
    "triplicates": _Q_TRIPLICATES,
    "copies_per_cell": _Q_COPIES_PER_CELL,
    "proteome_wide": _Q_PROTEOME_WIDE,
    "dataset2_preferred": _Q_DATASET2_PREFERRED,
    "search_database": _Q_SEARCH_DATABASE,
    "stationary": _Q_STATIONARY,
    "chemostat": _Q_CHEMOSTAT,
    "glycerol_amino_acids": _Q_GLYCEROL_AA,
}
#: Quotes whose provenance is a ``si2.xlsx`` row rendering.
SI2_QUOTES: dict[str, str] = {
    "table_s6": _Q_TABLE_S6,
    "table_s7": _Q_TABLE_S7,
    "table_s8": _Q_TABLE_S8,
    "table_s25": _Q_TABLE_S25,
}


def _paper(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    """A ``SourcedValue`` quoting the pinned ``paper.md``."""
    return SourcedValue(value=value, provenance=PAPER, quote=quote, note=note)


def _si2(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    """A ``SourcedValue`` quoting the pinned ``si2.xlsx``."""
    return SourcedValue(value=value, provenance=SI2_SOURCE, quote=quote, note=note)


SOURCED_VALUES: dict[str, SourcedValue] = {
    "reference_strain": _paper("BW25113", _Q_STRAIN),
    "background_genotype": _paper(
        "F-, ∆(araD-araB)567, ∆lacZ4787(:rrnB-3), λ-, rph-1, ∆(rhaD-rhaB)568, hsdR514",
        _Q_STRAIN,
        note="the lesions are already in the deposited BW25113 GenBank annotation this "
        "record pins, so they are not a BacterialStrainBackground on top of it",
    ),
    "temperature_c": _paper(37.0, _Q_BATCH_37C),
    "temperature_stress_c": _paper(42.0, _Q_42C),
    "aerobicity": _paper(
        "aerobic",
        _Q_BATCH_37C,
        note="orbital shaking at 300 r.p.m. in a sponge-closed Erlenmeyer flask",
    ),
    "osmotic_nacl_mm": _paper(50.0, _Q_OSMOTIC),
    "ph_stress": _paper(6.0, _Q_PH6),
    "n_biological_replicates": _paper(3, _Q_TRIPLICATES),
    "stored_quantity": _paper("protein copies per cell", _Q_COPIES_PER_CELL),
    "abundance_model": _paper(
        "summed protein intensity divided by the number of expected tryptic peptides, "
        "aligned to the 41 proteins quantified absolutely by stable isotope dilution",
        _Q_PROTEOME_WIDE,
    ),
    "dataset1_no_replicates": _si2(
        1,
        _Q_TABLE_S7,
        note="dataset 1 has no replicate measurement, so a dataset-1 protein stores "
        "n_replicates = 1 and no standard error",
    ),
    "combined_table": _si2("Table S6", _Q_TABLE_S6),
    "replicate_table": _si2("Table S8", _Q_TABLE_S8),
    "sample_names": _si2("Table S25", _Q_TABLE_S25),
    "search_database": _paper(
        "UniProt E. coli predicted proteome (2012/07/20) plus known contaminants",
        _Q_SEARCH_DATABASE,
        note="why five released rows are proteins of other organisms",
    ),
}


# --------------------------------------------------------------------------- #
# Workbook geometry. Blocks are located by the group label the merged row 2 carries,
# never by a hard-coded column index, and the located block's own headers are checked
# against the declared condition order.
# --------------------------------------------------------------------------- #
SHEET_S6 = "Table S6"
SHEET_S8 = "Table S8"
SHEET_S25 = "Table S25"
#: Row 1 is the table title, row 2 the block labels, row 3 the column headers.
HEADER_ROW = 3
BLOCK_COPIES = "Protein copies/cell"
BLOCK_MASS = "Protein Mass (fg) / Cell"
BLOCK_CV = (
    "Coefficient of Variance (%) Between Biological Triplicates (only available for "
    "Dataset 2)"
)
COL_UNIPROT = "Uniprot Accession"
COL_DESCRIPTION = "Description"
COL_GENE = "Gene"
COL_DATASET = "Dataset"
COL_MW = "Molecular weight (Da)"
COL_BNUMBER = "Bnumber"
COL_PEPTIDES = "Peptides.used.for.quantitation"
#: The release writes a missing value as this literal.
NOT_AVAILABLE = "NA"
#: ``OS=<organism>`` of the released UniProt description, up to the next ``KEY=`` token.
ORGANISM_RE = re.compile(r"OS=(.*?)(?= [A-Z][A-Za-z]*=|$)")
HOST_ORGANISM_PREFIX = "Escherichia coli"

#: The strain every loaded record is written against, as the Methods states it.
SCHMIDT_REFERENCE_STRAIN: EcoliK12StrainName = "BW25113"
#: What one stored abundance IS.
MEASUREMENT_TYPE = "absolute_protein_copies_per_cell_label_free_sid_anchored"
#: The ``Dataset`` value of the LC-MS experiment that ran in biological triplicates.
DATASET_WITH_REPLICATES = 2
#: Replicates per sample in dataset 2, and in dataset 1.
N_REPLICATES_DATASET2 = 3
N_REPLICATES_DATASET1 = 1
#: Relative tolerance of the two build-time identities against the released bytes.
IDENTITY_RTOL = 1e-6
#: Measured on the pinned workbook: 2,336 of 2,347 unique host gene symbols resolve to a
#: BW25113 locus (0.9953). The floor sits just below, so a real change in the release's
#: keying stops the build instead of silently shrinking the abundance map.
MIN_RESOLVED_FRACTION = 0.99

_GL = ConcentrationUnit.g_per_l
_MM = ConcentrationUnit.millimolar
_PH = ConcentrationUnit.ph


class DropReason(BaseModel):
    """Why one released growth condition is not a record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    description: str
    needed_addition: str | None = None


DROP_MEDIUM_NOT_IN_LIBRARY = DropReason(
    rule="medium_has_no_media_library_entry",
    description=(
        "the glycerol + amino acid medium is M9 with glycerol plus 21 named supplements "
        "at stated doses, which no MEDIA_LIBRARY key states; media.py is a value-surface "
        "file this branch does not edit, so the condition is left out rather than "
        "written against a medium that joins nothing"
    ),
    needed_addition="M9_GLYCEROL_AA_SCHMIDT2016",
)
DROP_CULTURE_NOT_BATCH = DropReason(
    rule="culture_not_batch",
    description=(
        "a chemostat culture: Environment has no culture-mode or dilution-rate slot, so "
        "the four chemostat arms, which differ from one another ONLY in dilution rate, "
        "would carry one byte-identical environment. This is the rule the landed "
        "Lamoureux 2023 loader states under the same name"
    ),
    needed_addition="a culture-mode / dilution-rate slot on Environment (closure-changing)",
)
DROP_GROWTH_PHASE = DropReason(
    rule="growth_phase_not_representable",
    description=(
        "the culture was harvested 1 or 3 days into stationary phase; Environment has no "
        "growth-phase slot, and the medium, carbon source, temperature and oxygen regime "
        "are those of the exponential Glucose arm, so the two stationary arms and the "
        "Glucose arm would collapse onto one environment"
    ),
    needed_addition="a growth-phase slot on Environment (closure-changing)",
)


class ConditionSpec(BaseModel):
    """One released growth condition of Table S6, and how it becomes an environment."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    s6_column: str = Field(description="Table S6 header, stripped of padding space.")
    s8_suffix: str = Field(description="Table S8 column suffix of the same group.")
    s25_label: str = Field(description="Table S25 growth-condition label, verbatim.")
    s25_files: tuple[str, str, str] = Field(
        description="Table S25 file names of the three biological replicates."
    )
    media: Media
    carbon_source: str | None = None
    carbon_g_per_l: float | None = None
    temperature_c: float = 37.0
    nacl_mm: float | None = None
    ph: float | None = None
    is_reference: bool = False
    drop: DropReason | None = None


def _m9(
    s6: str,
    s8: str,
    label: str,
    files: tuple[str, str, str],
    carbon: str,
    g_per_l: float,
    **kwargs: Any,
) -> ConditionSpec:
    """A minimal-medium condition: the Schmidt M9 base plus one carbon source."""
    return ConditionSpec(
        s6_column=s6,
        s8_suffix=s8,
        s25_label=label,
        s25_files=files,
        media=M9_SCHMIDT2016,
        carbon_source=carbon,
        carbon_g_per_l=g_per_l,
        **kwargs,
    )


#: Every condition Table S6 releases, in its own column order. The carbon-source names
#: are the weighed reagents the Methods names ("acetate (sodium acetate, 3.5 g/L)"), so
#: the stated g/L is the amount of the compound the magnitude carries.
CONDITIONS: tuple[ConditionSpec, ...] = (
    _m9(
        "Glucose",
        "Glucose.1",
        "glucose",
        ("A14-07036", "A14-07037", "A14-07038"),
        "glucose",
        5.0,
        is_reference=True,
    ),
    ConditionSpec(
        s6_column="LB",
        s8_suffix="LB",
        s25_label="LB",
        s25_files=("A14-07032", "A14-07033", "A14-07034"),
        media=LB,
    ),
    _m9(
        "Glycerol + AA",
        "Glycerin.AA",
        "glycerol + AA",
        ("A14-07040", "A14-07041", "A14-07042"),
        "glycerol",
        2.2,
        drop=DROP_MEDIUM_NOT_IN_LIBRARY,
    ),
    _m9(
        "Acetate",
        "Acetate",
        "acetate",
        ("A14-07044", "A14-07045", "A14-07046"),
        "sodium acetate",
        3.5,
    ),
    _m9(
        "Fumarate",
        "Fumarate",
        "fumarate",
        ("A14-07048", "A14-07049", "A14-07050"),
        "disodium fumarate",
        2.8,
    ),
    _m9(
        "Glucosamine",
        "Glucoseamine",
        "glucosamine",
        ("A14-07052", "A14-07053", "A14-07054"),
        "glucosamine",
        2.1,
    ),
    _m9(
        "Glycerol",
        "Glycerol",
        "glycerol",
        ("A14-07056", "A14-07057", "A14-07058"),
        "glycerol",
        2.2,
    ),
    _m9(
        "Pyruvate",
        "Pyruvate",
        "pyruvate",
        ("A14-07060", "A14-07061", "A14-07062"),
        "sodium pyruvate",
        3.3,
    ),
    _m9(
        "Chemostat µ=0.5",
        "Chemstat.0.5",
        "chemostat µ=0.5",
        ("A14-07064", "A14-07065", "A14-07066"),
        "glucose",
        1.0,
        drop=DROP_CULTURE_NOT_BATCH,
    ),
    _m9(
        "Chemostat µ=0.35",
        "Chemstat.0.35",
        "chemostat µ=0.35",
        ("A14-07068", "A14-07069", "A14-07070"),
        "glucose",
        1.0,
        drop=DROP_CULTURE_NOT_BATCH,
    ),
    _m9(
        "Chemostat µ=0.20",
        "Chemstat.0.2",
        "chemostat µ=0.20",
        ("A14-07072", "A14-07073", "A14-07074"),
        "glucose",
        1.0,
        drop=DROP_CULTURE_NOT_BATCH,
    ),
    _m9(
        "Chemostat µ=0.12",
        "Chemstat.0.12",
        "chemostat µ=0.12",
        ("A14-07076", "A14-07077", "A14-07078"),
        "glucose",
        1.0,
        drop=DROP_CULTURE_NOT_BATCH,
    ),
    _m9(
        "Stationary phase 1 day",
        "Stat.1day",
        "stationary 1 day",
        ("A14-07080", "A14-07081", "A14-07082"),
        "glucose",
        5.0,
        drop=DROP_GROWTH_PHASE,
    ),
    _m9(
        "Stationary phase 3 days",
        "Stat.3days",
        "stationary 3 days",
        ("A14-07084", "A14-07085", "A14-07086"),
        "glucose",
        5.0,
        drop=DROP_GROWTH_PHASE,
    ),
    _m9(
        "Osmotic-stress glucose",
        "OSM",
        "50 mM NaCl",
        ("A14-07092", "A14-07093", "A14-07094"),
        "glucose",
        5.0,
        nacl_mm=50.0,
    ),
    _m9(
        "42°C glucose",
        "42.C",
        "42°C",
        ("A14-07096", "A14-07097", "A14-07098"),
        "glucose",
        5.0,
        temperature_c=42.0,
    ),
    _m9(
        "pH6 glucose",
        "pH6",
        "pH 6",
        ("A14-07100", "A14-07101", "A14-07102"),
        "glucose",
        5.0,
        ph=6.0,
    ),
    _m9(
        "Xylose",
        "Xylose",
        "xylose",
        ("A14-07109", "A14-07110", "A14-07111"),
        "xylose",
        5.0,
    ),
    _m9(
        "Mannose",
        "Mannose",
        "mannose",
        ("A14-07113", "A14-07114", "A14-07115"),
        "mannose",
        5.0,
    ),
    _m9(
        "Galactose",
        "Galactose",
        "galactose",
        ("A14-07117", "A14-07118", "A14-07119"),
        "galactose",
        2.3,
    ),
    _m9(
        "Succinate",
        "Succinate",
        "succinate",
        ("A14-07121", "A14-07122", "A14-07123"),
        "disodium succinate hexahydrate",
        5.7,
    ),
    _m9(
        "Fructose",
        "Fructose",
        "fructose",
        ("A14-07125", "A14-07126", "A14-07127"),
        "fructose",
        5.0,
    ),
)
CONDITIONS_BY_COLUMN: dict[str, ConditionSpec] = {c.s6_column: c for c in CONDITIONS}
REFERENCE_CONDITION = "Glucose"
#: 22 released conditions - 1 reference - 7 structurally unrepresentable = 14.
EXPECTED_RECORDS = 14
#: Protein keys on a record of a condition both LC-MS experiments covered, and on one of
#: the three conditions only dataset 2 covered.
EXPECTED_PROTEIN_KEYS = 2329
EXPECTED_PROTEIN_KEYS_DATASET2_ONLY = 2034
#: The second BW25113 glucose triplicate (``Glucose.2``, Table S25 A14-07088..90), which
#: Table S6 does not combine: it is the reproducibility arm of Supplementary Figure 7.
GLUCOSE_REPRODUCIBILITY_FILES = ("A14-07088", "A14-07089", "A14-07090")


# --------------------------------------------------------------------------- #
# Reading the pinned workbook
# --------------------------------------------------------------------------- #
class ProteinRow(BaseModel):
    """One Table S6 row: its identifiers, its dataset, and its copies/cell per column."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_number: int = Field(description="1-based row in the sheet, for the ledger.")
    uniprot: str
    description: str
    gene: str
    dataset: int
    molecular_weight_da: float
    peptides_used: int
    bnumber: str | None
    organism: str
    copies: dict[str, float | None] = Field(
        description="condition column -> copies/cell, None where the release says NA"
    )

    @property
    def is_host(self) -> bool:
        """True when the released UniProt description names an E. coli organism."""
        return self.organism.startswith(HOST_ORGANISM_PREFIX)

    @property
    def n_replicates(self) -> int:
        """Replicates behind this row: three for dataset 2, one for dataset 1."""
        return (
            N_REPLICATES_DATASET2
            if self.dataset == DATASET_WITH_REPLICATES
            else N_REPLICATES_DATASET1
        )


class ReplicateCell(BaseModel):
    """One Table S8 (protein, condition) cell: its three replicates and its statistics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    replicates: tuple[float, float, float]
    median_released: float
    cv_released: float

    @property
    def median(self) -> float:
        """Median of the three released normalized abundances."""
        return statistics.median(self.replicates)

    @property
    def cv(self) -> float:
        """``100 * stdev / mean`` of the three released normalized abundances."""
        return (
            100.0
            * statistics.stdev(self.replicates)
            / statistics.fmean(self.replicates)
        )

    def standard_error(self, copies: float) -> float:
        """The SE of ``copies`` per cell, from this cell's own replicate spread.

        ``copies`` is a per-protein multiple of ``median_released`` within a condition
        (measured, see the module docstring), so that ratio converts the replicate
        standard deviation into copies per cell; the SE divides by sqrt(3).
        """
        scale = copies / self.median_released
        return (
            scale * statistics.stdev(self.replicates) / math.sqrt(N_REPLICATES_DATASET2)
        )


def _header_index(
    header: Sequence[Any], columns: Sequence[int] | None = None
) -> dict[str, int]:
    """``{stripped header: column index}`` over ``columns``; a repeat is refused.

    Table S6 repeats a condition name once per quantity block, so a whole-row index
    would collide; the caller passes the identifier block and the trailing annotation
    block separately.
    """
    wanted = range(len(header)) if columns is None else columns
    out: dict[str, int] = {}
    for index in wanted:
        cell = header[index]
        if cell is None or not str(cell).strip():
            continue
        name = str(cell).strip()
        if name in out:
            raise RuntimeError(f"header {name!r} appears twice in one row")
        out[name] = index
    return out


def _block_start(labels: Sequence[Any], label: str) -> int:
    """Index of the merged row-2 cell carrying ``label``."""
    for index, cell in enumerate(labels):
        if cell is not None and str(cell).strip() == label.strip():
            return index
    raise RuntimeError(f"Table S6 row 2 carries no block labelled {label!r}")


def _number(value: Any) -> float | None:
    """A released cell as a float, or None where the release writes ``NA``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip() == NOT_AVAILABLE:
        return None
    raise RuntimeError(f"{value!r} is neither a number nor {NOT_AVAILABLE!r}")


def read_table_s6(path: str) -> tuple[list[ProteinRow], list[tuple[str, float, float]]]:
    """Read Table S6's rows plus its first two CV columns (for the header assertion).

    The copies/cell block is located by its row-2 label and its 22 headers must be the
    declared condition order, so a re-export that moves or renames a column stops the
    build rather than shifting a condition's values onto another condition.
    """
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(book[SHEET_S6].iter_rows(values_only=True))
    finally:
        book.close()
    labels, header = rows[HEADER_ROW - 2], rows[HEADER_ROW - 1]
    copies_start = _block_start(labels, BLOCK_COPIES)
    mass_start = _block_start(labels, BLOCK_MASS)
    cv_start = _block_start(labels, BLOCK_CV)
    tail_start = cv_start + len(CONDITIONS)
    index = _header_index(header, range(copies_start))
    tail = _header_index(header, range(tail_start, len(header)))
    if mass_start - copies_start != len(CONDITIONS):
        raise RuntimeError(
            f"the copies/cell block spans {mass_start - copies_start} columns, not "
            f"{len(CONDITIONS)}"
        )
    found = tuple(str(header[copies_start + i]).strip() for i in range(len(CONDITIONS)))
    declared = tuple(spec.s6_column for spec in CONDITIONS)
    if found != declared:
        raise RuntimeError(
            f"Table S6's copies/cell headers are {found}, the module declares {declared}"
        )

    out: list[ProteinRow] = []
    cv_pairs: list[tuple[str, float, float]] = []
    for offset, row in enumerate(rows[HEADER_ROW:]):
        uniprot = str(row[index[COL_UNIPROT]]).strip()
        description = str(row[index[COL_DESCRIPTION]])
        match = ORGANISM_RE.search(description)
        if match is None:
            raise RuntimeError(f"{uniprot}: released description names no OS= organism")
        bnumber = row[tail[COL_BNUMBER]]
        out.append(
            ProteinRow(
                row_number=HEADER_ROW + 1 + offset,
                uniprot=uniprot,
                description=description,
                gene=str(row[index[COL_GENE]]).strip(),
                dataset=int(row[index[COL_DATASET]]),
                molecular_weight_da=float(row[index[COL_MW]]),
                peptides_used=int(row[index[COL_PEPTIDES]]),
                bnumber=None if bnumber is None else str(bnumber).strip(),
                organism=match.group(1).strip(),
                copies={
                    spec.s6_column: _number(row[copies_start + i])
                    for i, spec in enumerate(CONDITIONS)
                },
            )
        )
        first, second = row[cv_start], row[cv_start + 1]
        if isinstance(first, (int, float)) and isinstance(second, (int, float)):
            cv_pairs.append((uniprot, float(first), float(second)))
    return out, cv_pairs


def read_table_s25(path: str) -> list[tuple[str, str, str]]:
    """``(file name, growth condition, strain)`` of every LC-MS sample, in sheet order."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(book[SHEET_S25].iter_rows(values_only=True))
    finally:
        book.close()
    out: list[tuple[str, str, str]] = []
    for row in rows[HEADER_ROW:]:
        if row[0] is None or not str(row[0]).strip():
            continue
        out.append((str(row[0]).strip(), str(row[1]).strip(), str(row[2]).strip()))
    return out


def check_sample_map(samples: Sequence[tuple[str, str, str]]) -> None:
    """Every condition's three declared replicate files are what Table S25 lists.

    The check is per file, not per label: Table S25 lists ``glucose`` / ``BW25113``
    twice, because the second triplicate is the reproducibility arm Table S6 leaves out.
    """
    by_file = {name: (condition, strain) for name, condition, strain in samples}
    for spec in CONDITIONS:
        for name in spec.s25_files:
            if name not in by_file:
                raise RuntimeError(f"Table S25 lists no sample {name!r}")
            condition, strain = by_file[name]
            if (condition, strain) != (spec.s25_label, SCHMIDT_REFERENCE_STRAIN):
                raise RuntimeError(
                    f"Table S25 files {name!r} under {(condition, strain)}; the module "
                    f"declares {(spec.s25_label, SCHMIDT_REFERENCE_STRAIN)}"
                )
    for name in GLUCOSE_REPRODUCIBILITY_FILES:
        if by_file.get(name) != ("glucose", SCHMIDT_REFERENCE_STRAIN):
            raise RuntimeError(
                f"Table S25 no longer files {name!r} as a BW25113 glucose replicate"
            )


def _s8_column(file_name: str) -> str:
    """Table S8's normalized-abundance column of one Table S25 file name."""
    return f"normInt_{file_name.replace('-', '.')}.1"


def read_table_s8(
    path: str, columns: Sequence[str]
) -> tuple[dict[str, dict[str, ReplicateCell]], tuple[str, ...]]:
    """Per-protein replicate cells for each requested Table S6 condition column.

    Returns ``{uniprot: {condition column: ReplicateCell}}`` and the accessions Table S8
    files on more than one row, whose cells are not returned at all: a repeated
    accession's statistics cannot be attributed to either row.
    """
    specs = [CONDITIONS_BY_COLUMN[name] for name in columns]
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(book[SHEET_S8].iter_rows(values_only=True))
    finally:
        book.close()
    index = _header_index(rows[HEADER_ROW - 1])
    plan: list[tuple[str, int, int, tuple[int, int, int]]] = []
    for spec in specs:
        replicates = tuple(index[_s8_column(name)] for name in spec.s25_files)
        if len(replicates) != N_REPLICATES_DATASET2:
            raise RuntimeError(f"{spec.s6_column}: {len(replicates)} replicate columns")
        plan.append(
            (
                spec.s6_column,
                index[f"medianNormInt_{spec.s8_suffix}"],
                index[f"cv_{spec.s8_suffix}"],
                (replicates[0], replicates[1], replicates[2]),
            )
        )

    out: dict[str, dict[str, ReplicateCell]] = {}
    seen: dict[str, int] = {}
    for row in rows[HEADER_ROW:]:
        uniprot = str(row[0]).strip()
        seen[uniprot] = seen.get(uniprot, 0) + 1
        cells: dict[str, ReplicateCell] = {}
        for column, median_col, cv_col, replicate_cols in plan:
            values = [row[i] for i in replicate_cols]
            median, cv = row[median_col], row[cv_col]
            if not all(isinstance(v, (int, float)) for v in [*values, median, cv]):
                continue
            cells[column] = ReplicateCell(
                replicates=(float(values[0]), float(values[1]), float(values[2])),
                median_released=float(median),
                cv_released=float(cv),
            )
        out[uniprot] = cells
    duplicates = tuple(sorted(key for key, count in seen.items() if count > 1))
    for key in duplicates:
        out.pop(key, None)
    return out, duplicates


# --------------------------------------------------------------------------- #
# The retention ledger
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One rule that removed released rows or columns, with the items it removed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    scope: str = Field(description="'condition', 'protein_row' or 'protein_key'")
    description: str
    n_items: int
    items: list[str]
    needed_addition: str | None = None


class DropLog(BaseModel):
    """Every released condition and protein row, and what became of it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    source_conditions: int
    reference_conditions: list[str]
    candidate_records: int
    kept_records: int
    dropped_records: int
    source_protein_rows: int
    kept_protein_keys: int
    dropped_protein_rows: int
    rules: list[DropRule]
    reconciliation: LocusTagReconciliation
    notes: list[str]

    def check(self) -> None:
        """Refuse a ledger whose rules do not account for every drop."""
        conditions = sum(r.n_items for r in self.rules if r.scope == "condition")
        if self.kept_records + conditions + len(self.reference_conditions) != (
            self.source_conditions
        ):
            raise RuntimeError(
                f"{self.dataset}: {self.kept_records} records + {conditions} dropped "
                f"conditions + {len(self.reference_conditions)} references != "
                f"{self.source_conditions} released conditions"
            )
        rows = sum(r.n_items for r in self.rules if r.scope == "protein_row")
        if self.kept_protein_keys + rows != self.source_protein_rows:
            raise RuntimeError(
                f"{self.dataset}: {self.kept_protein_keys} kept keys + {rows} dropped "
                f"rows != {self.source_protein_rows} released rows"
            )
        if self.dropped_protein_rows != rows:
            raise RuntimeError(
                f"{self.dataset}: {self.dropped_protein_rows} dropped rows stated, "
                f"{rows} accounted by rules"
            )


class ProteinSelection(BaseModel):
    """The kept protein rows, their stored locus tags, and every drop that got there."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kept: list[ProteinRow]
    locus_tag: dict[str, str] = Field(description="uniprot accession -> locus tag")
    reconciliation: LocusTagReconciliation
    rules: list[DropRule]

    @property
    def dropped_rows(self) -> int:
        """Released rows removed by the row-scoped rules."""
        return sum(rule.n_items for rule in self.rules if rule.scope == "protein_row")


def select_proteins(
    rows: Sequence[ProteinRow], genome: EcoliK12Genome, *, label: str
) -> ProteinSelection:
    """Apply the three protein-row rules, then reconcile the surviving gene symbols.

    Order matters and is not arbitrary: the non-host rows go first, because four of the
    five share an E. coli gene synonym with a host row and would otherwise make that
    host row's symbol collide and be kept as given.
    """
    non_host = [row for row in rows if not row.is_host]
    host = [row for row in rows if row.is_host]

    counts: dict[str, int] = {}
    for row in host:
        counts[row.gene] = counts.get(row.gene, 0) + 1
    repeated = {gene for gene, count in counts.items() if count > 1}
    duplicated = [row for row in host if row.gene in repeated]
    unique = [row for row in host if row.gene not in repeated]

    stored, report = reconcile_locus_tags(
        genome, pd.Series([row.gene for row in unique]), label=label
    )
    report.require_resolved(MIN_RESOLVED_FRACTION)
    outside = set(report.outside_namespace)
    kept: list[ProteinRow] = []
    locus_tag: dict[str, str] = {}
    unresolved: list[ProteinRow] = []
    for row, tag in zip(unique, stored, strict=True):
        if tag in outside:
            unresolved.append(row)
            continue
        kept.append(row)
        locus_tag[row.uniprot] = str(tag)
    if len({*locus_tag.values()}) != len(kept):
        raise RuntimeError(f"{label}: two kept rows share one locus tag")

    rules = [
        DropRule(
            rule="source_organism_is_not_escherichia_coli",
            scope="protein_row",
            description=(
                "the released UniProt description names another organism in its OS= "
                "token; the search database was built to carry contaminants (the "
                "search_database sourced value quotes the Methods), so the row is no "
                "E. coli protein and has no gene to key an abundance to"
            ),
            n_items=len(non_host),
            items=[f"{row.gene} ({row.uniprot}, {row.organism})" for row in non_host],
        ),
        DropRule(
            rule="gene_symbol_filed_on_two_released_rows",
            scope="protein_row",
            description=(
                "the release files this gene symbol on two rows with different UniProt "
                "accessions or isoforms, so one key would name two protein groups. The "
                "release does state a dataset-2 preference that would pick a row for "
                "four of the seven symbols, and it is deliberately not applied: rmlA "
                "and glsA are two distinct paralogs under one symbol in both of their "
                "rows, so only a uniform drop never mis-attributes a measurement"
            ),
            n_items=len(duplicated),
            items=[
                f"{row.gene} ({row.uniprot}, dataset {row.dataset}, row {row.row_number})"
                for row in duplicated
            ],
        ),
        DropRule(
            rule="gene_symbol_resolves_to_no_locus_of_the_pinned_assembly",
            scope="protein_row",
            description=(
                "the symbol resolves to no locus of "
                f"{BACTERIAL_ASSEMBLY_SETS[SCHMIDT_REFERENCE_STRAIN]} through any "
                "resolver layer and is kept as given by the retain-all policy, so it "
                "names no gene node"
            ),
            n_items=len(unresolved),
            items=[f"{row.gene} ({row.uniprot})" for row in unresolved],
        ),
    ]
    return ProteinSelection(
        kept=kept, locus_tag=locus_tag, reconciliation=report, rules=rules
    )


# --------------------------------------------------------------------------- #
# Environment, phenotype, record
# --------------------------------------------------------------------------- #
def build_environment(spec: ConditionSpec) -> Environment:
    """The environment of one growth condition."""
    perturbations: list[EnvironmentPerturbationType] = []
    if spec.carbon_source is not None:
        perturbations.append(
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.carbon_source,
                magnitude=Concentration(value=spec.carbon_g_per_l, unit=_GL),
                agent=resolved_compound(spec.carbon_source),
            )
        )
    if spec.nacl_mm is not None:
        perturbations.append(
            SmallMoleculePerturbation(
                compound=resolved_compound("sodium chloride"),
                concentration=Concentration(value=spec.nacl_mm, unit=_MM),
            )
        )
    if spec.ph is not None:
        perturbations.append(
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.ph,
                magnitude=Concentration(value=spec.ph, unit=_PH),
                agent=resolved_compound("hydrochloric acid"),
            )
        )
    return Environment(
        media=spec.media,
        temperature=Temperature(value=spec.temperature_c),
        perturbations=perturbations,
        aerobicity=str(SOURCED_VALUES["aerobicity"].value),
    )


def check_environments_distinct(environments: Mapping[str, Environment]) -> None:
    """Refuse two conditions whose environments serialize to the same bytes.

    This is what makes the ``culture_not_batch`` and ``growth_phase_not_representable``
    rules checked rather than asserted: whatever survives them must be distinguishable.
    """
    seen: dict[str, str] = {}
    for name, environment in environments.items():
        key = environment.model_dump_json()
        if key in seen:
            raise RuntimeError(
                f"{name!r} and {seen[key]!r} serialize to one environment, so their "
                "records would not be distinguishable"
            )
        seen[key] = name


def build_phenotype(
    rows: Sequence[ProteinRow],
    locus_tag: Mapping[str, str],
    cells: Mapping[str, Mapping[str, ReplicateCell]],
    column: str,
) -> ProteinAbundancePhenotype:
    """The abundance profile of one condition, over the rows the release quantified.

    A row the release writes as ``NA`` in this column is not in the profile: dataset 1
    did not cover the xylose, mannose and fructose conditions, so its 295 kept proteins
    have no value there. A released 0 IS a value and is kept verbatim.
    """
    abundance: dict[str, float] = {}
    standard_error: dict[str, float] = {}
    n_replicates: dict[str, int] = {}
    for row in rows:
        copies = row.copies[column]
        if copies is None:
            continue
        tag = locus_tag[row.uniprot]
        abundance[tag] = copies
        n_replicates[tag] = row.n_replicates
        cell = cells.get(row.uniprot, {}).get(column)
        standard_error[tag] = (
            cell.standard_error(copies) if cell is not None else float("nan")
        )
    return ProteinAbundancePhenotype(
        protein_abundance=abundance,
        protein_abundance_se=standard_error,
        n_replicates=n_replicates,
        measurement_type=MEASUREMENT_TYPE,
    )


def restrict(
    phenotype: ProteinAbundancePhenotype, keys: Sequence[str]
) -> ProteinAbundancePhenotype:
    """The same profile over ``keys`` only, for a key-matched reference."""
    wanted = set(keys)
    missing = wanted - set(phenotype.protein_abundance)
    if missing:
        raise RuntimeError(
            f"the reference condition quantifies none of {sorted(missing)[:5]}"
        )
    return ProteinAbundancePhenotype(
        protein_abundance={
            k: v for k, v in phenotype.protein_abundance.items() if k in wanted
        },
        protein_abundance_se={
            k: v
            for k, v in (phenotype.protein_abundance_se or {}).items()
            if k in wanted
        },
        n_replicates={k: v for k, v in phenotype.n_replicates.items() if k in wanted},
        measurement_type=phenotype.measurement_type,
    )


def publication() -> Publication:
    """The paper, as the PMC id converter resolves it from ``PMC4888949``."""
    return Publication(
        pubmed_id=PUBMED_ID,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
        doi=PAPER_DOI,
        doi_url=f"https://doi.org/{PAPER_DOI}",
    )


def check_released_statistics(
    cells: Mapping[str, Mapping[str, ReplicateCell]],
) -> dict[str, Any]:
    """Both released statistics must be what the replicate columns say they are.

    ``medianNormInt`` is the median of the three replicates and ``cv`` is
    ``100 * stdev / mean`` of the same three. The stored SE is derived from those
    replicates, so a drift in either identity means the columns no longer mean what the
    derivation assumes, and the build stops.
    """
    worst_median = 0.0
    worst_cv = 0.0
    n = 0
    for uniprot, by_column in cells.items():
        for column, cell in by_column.items():
            n += 1
            for released, computed, label in (
                (cell.median_released, cell.median, "median"),
                (cell.cv_released, cell.cv, "cv"),
            ):
                deviation = (
                    abs(computed - released) / abs(released)
                    if released
                    else abs(computed)
                )
                if label == "median":
                    worst_median = max(worst_median, deviation)
                else:
                    worst_cv = max(worst_cv, deviation)
                if deviation > IDENTITY_RTOL:
                    raise RuntimeError(
                        f"{uniprot}/{column}: released {label} {released} against "
                        f"{computed} computed from the three replicates "
                        f"(relative deviation {deviation:.3e} > {IDENTITY_RTOL})"
                    )
    return {"n_cells": n, "worst_median_rtol": worst_median, "worst_cv_rtol": worst_cv}


def check_cv_header_swap(
    cv_pairs: Sequence[tuple[str, float, float]],
    cells: Mapping[str, Mapping[str, ReplicateCell]],
) -> dict[str, Any]:
    """Table S6's first two CV headers are swapped, and this pins that finding.

    Its CV column headed ``Glucose`` carries LB's coefficient of variation and the one
    headed ``LB`` carries glucose's. No stored value reads a Table S6 CV column; the
    check exists so the finding stays attached to the bytes rather than to a note.
    """
    checked = 0
    for uniprot, first, second in cv_pairs:
        by_column = cells.get(uniprot)
        if by_column is None or "LB" not in by_column or "Glucose" not in by_column:
            continue
        for released, cell in (
            (first, by_column["LB"]),
            (second, by_column["Glucose"]),
        ):
            deviation = abs(cell.cv_released - released) / abs(released)
            if deviation > IDENTITY_RTOL:
                raise RuntimeError(
                    "Table S6's first two CV columns no longer hold LB then Glucose: "
                    f"{uniprot} gives {released} against {cell.cv_released}"
                )
        checked += 1
    if not checked:
        raise RuntimeError("no row could check the Table S6 CV header order")
    return {"n_rows_checked": checked, "swapped": True}


# --------------------------------------------------------------------------- #
# The raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (both mirrors and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/schmidtQuantitativeConditiondependentEscherichia2016``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/schmidtQuantitativeConditiondependentEscherichia2016``."""
    return Path(data_root or _data_root()) / LIBRARY_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve_raw_files(dest_dir: str | Path) -> dict[str, Path]:
    """Re-run the recorded retrieval and write the verified bytes into ``dest_dir``.

    The ``RetrievalRecord`` is what runs, so this IS the recorded retrieval rather than
    a description of it; a byte mismatch raises before anything is written.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / SI2
    write_verified(run_retriever(SI2_RETRIEVAL), path, SI2_SHA256, SI2_URL)
    return {SI2: path}


def deposit_raw_mirror(*, source: str | Path, data_root: str | None = None) -> Path:
    """Write the raw mirror from an already-retrieved ``si2.xlsx`` plus its manifest.

    Idempotent by sha256: a mirror file that already hashes to the pin is left alone,
    and one with any other hash raises rather than being overwritten.
    """
    src = Path(source)
    observed = _sha256(src)
    if observed != SI2_SHA256:
        raise RuntimeError(
            f"{src} sha256 mismatch: got {observed}, expected {SI2_SHA256}"
        )
    root = raw_mirror_dir(data_root)
    dest = root / SI2_MIRROR_RELPATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != SI2_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(src, dest)
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=[
            ArtifactRecord(
                path=SI2_MIRROR_RELPATH,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=SI2_SHA256,
                source=SI2_URL,
                retrieval=SI2_RETRIEVAL,
            )
        ],
        si_data_sources=[
            SI2_URL,
            f"https://pmc.ncbi.nlm.nih.gov/articles/{PMC_ID}/",
            "https://proteomecentral.proteomexchange.org (PXD000498)",
        ],
        si_expected=list(NOT_MIRRORED),
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
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class ProteomeSchmidt2016Dataset(ExperimentDataset):
    """Absolute BW25113 proteome, one record per loaded growth condition."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = SCHMIDT_REFERENCE_STRAIN

    def __init__(
        self,
        root: str = "data/torchcell/proteome_schmidt2016",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``ecoli_genome`` is injected by the build entry points."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialProteinAbundanceExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialProteinAbundanceExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The one consumed file, linked from the raw mirror."""
        return [SI2]

    def download(self) -> None:
        """Link the mirrored workbook into ``raw/`` after checking manifest and sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        check_manifest_pin(
            SI2_MIRROR_RELPATH,
            manifest_sha256(manifest, SI2_MIRROR_RELPATH),
            SI2_SHA256,
        )
        src = raw_mirror_dir(data_root) / SI2_MIRROR_RELPATH
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(src, osp.join(self.raw_dir, SI2), SI2_SHA256)
        log.info("Schmidt 2016 raw file linked into %s (sha256 verified)", self.raw_dir)

    def compute_gene_set(self) -> GeneSet:
        """The BW25113 loci the stored abundance profiles are keyed by.

        Every record is wild type, so no genotype names a gene and the base class's
        genotype scan would return the empty set it refuses. The dataset's genes are the
        loci it measures, which is how the landed Caglar 2017 loader states the same
        situation for its REL606 panel.
        """
        if self.env is None:
            self._init_db()
        genes = GeneSet()
        with self.env.begin() as txn:
            for _, value in txn.cursor():
                record = pickle.loads(value)
                genes.update(record["experiment"]["phenotype"]["protein_abundance"])
        self.close_lmdb()
        return genes

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        expected = BACTERIAL_ASSEMBLY_SETS[self.REFERENCE_STRAIN]
        if self.ecoli_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.ecoli_genome.ASSEMBLY_SET}"
            )
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Build one abundance record per loaded condition and write the LMDB."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        path = osp.join(self.raw_dir, SI2)
        rows, cv_pairs = read_table_s6(path)
        check_sample_map(read_table_s25(path))
        genome = self._genome()
        selection = select_proteins(rows, genome, label=f"{self.name} gene symbols")

        loaded = [spec for spec in CONDITIONS if spec.drop is None]
        cells, repeated = read_table_s8(path, [spec.s6_column for spec in loaded])
        statistics_check = check_released_statistics(cells)
        header_check = check_cv_header_swap(cv_pairs, cells)
        environments = {spec.s6_column: build_environment(spec) for spec in loaded}
        check_environments_distinct(environments)

        phenotypes = {
            spec.s6_column: build_phenotype(
                selection.kept, selection.locus_tag, cells, spec.s6_column
            )
            for spec in loaded
        }
        reference_phenotype = phenotypes[REFERENCE_CONDITION]
        reference_genome = assembly_reference(self.REFERENCE_STRAIN)
        reference_environment = environments[REFERENCE_CONDITION]
        pub = publication()
        records = [spec for spec in loaded if not spec.is_reference]
        if len(records) != EXPECTED_RECORDS:
            raise RuntimeError(
                f"{len(records)} records, the module states {EXPECTED_RECORDS}"
            )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        condition_rows: list[dict[str, Any]] = []
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, spec in enumerate(tqdm(records, desc="schmidt2016")):
                phenotype = phenotypes[spec.s6_column]
                experiment = BacterialProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(perturbations=[]),
                    environment=environments[spec.s6_column],
                    phenotype=phenotype,
                )
                reference = BacterialProteinAbundanceExperimentReference(
                    dataset_name=self.name,
                    genome_reference=reference_genome,
                    environment_reference=reference_environment,
                    phenotype_reference=restrict(
                        reference_phenotype, list(phenotype.protein_abundance)
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                condition_rows.append(
                    {
                        "s6_column": spec.s6_column,
                        "s25_label": spec.s25_label,
                        "media": spec.media.name,
                        "carbon_source": spec.carbon_source,
                        "carbon_g_per_l": spec.carbon_g_per_l,
                        "temperature_c": spec.temperature_c,
                        "n_protein_keys": len(phenotype.protein_abundance),
                        "n_with_se": sum(
                            1
                            for v in (phenotype.protein_abundance_se or {}).values()
                            if not math.isnan(v)
                        ),
                    }
                )
        env.close()
        interned_env.close()

        self._write_ledgers(
            rows,
            selection,
            loaded,
            phenotypes,
            condition_rows,
            repeated,
            statistics_check,
            header_check,
        )
        log.info(
            "Schmidt 2016: %d records (+ the %s reference) x %d protein keys from %d "
            "released rows; %d rows dropped; %d of 22 conditions left out",
            len(records),
            REFERENCE_CONDITION,
            len(reference_phenotype.protein_abundance),
            len(rows),
            selection.dropped_rows,
            len(CONDITIONS) - len(loaded),
        )

    def _write_ledgers(
        self,
        rows: Sequence[ProteinRow],
        selection: ProteinSelection,
        loaded: Sequence[ConditionSpec],
        phenotypes: Mapping[str, ProteinAbundancePhenotype],
        condition_rows: Sequence[Mapping[str, Any]],
        repeated: Sequence[str],
        statistics_check: Mapping[str, Any],
        header_check: Mapping[str, Any],
    ) -> None:
        """The drop log, the sourcing table, the identifier table, the condition table."""
        out = Path(self.preprocess_dir)
        kept_keys = len(phenotypes[REFERENCE_CONDITION].protein_abundance)
        condition_rules = [
            DropRule(
                rule=reason.rule,
                scope="condition",
                description=reason.description,
                n_items=len(
                    [s for s in CONDITIONS if s.drop is not None and s.drop == reason]
                ),
                items=[
                    s.s6_column
                    for s in CONDITIONS
                    if s.drop is not None and s.drop == reason
                ],
                needed_addition=reason.needed_addition,
            )
            for reason in (
                DROP_MEDIUM_NOT_IN_LIBRARY,
                DROP_CULTURE_NOT_BATCH,
                DROP_GROWTH_PHASE,
            )
        ]
        log_model = DropLog(
            dataset=self.name,
            source_conditions=len(CONDITIONS),
            reference_conditions=[REFERENCE_CONDITION],
            candidate_records=len(loaded) - 1,
            kept_records=len(condition_rows),
            dropped_records=len(CONDITIONS) - len(loaded),
            source_protein_rows=len(rows),
            kept_protein_keys=kept_keys,
            dropped_protein_rows=selection.dropped_rows,
            rules=[*condition_rules, *selection.rules],
            reconciliation=selection.reconciliation,
            notes=[
                f"{len(rows)} released rows - "
                + " - ".join(
                    f"{rule.n_items} ({rule.rule})" for rule in selection.rules
                ),
                f"= {kept_keys} protein keys on every record of a condition both LC-MS "
                "experiments covered",
                f"{EXPECTED_PROTEIN_KEYS_DATASET2_ONLY} keys on the Xylose, Mannose and "
                "Fructose records: dataset 1 did not cover those three conditions, so "
                "its kept proteins are released as NA there",
                f"the {REFERENCE_CONDITION} condition is the phenotype_reference, not a "
                "record, because the release defines every fold change against BW25113 "
                "in glucose minimal medium and every record's genotype is identical",
                "the second BW25113 glucose triplicate (Table S25 "
                f"{', '.join(GLUCOSE_REPRODUCIBILITY_FILES)}, Table S8 Glucose.2) is "
                "the reproducibility arm and is not in Table S6, so it is not a record",
                f"Table S8 files {len(repeated)} accession(s) on more than one row "
                f"({', '.join(repeated) or 'none'}); their replicate cells are not read",
            ],
        )
        log_model.check()
        (out / "dropped_records.json").write_text(log_model.model_dump_json(indent=2))
        (out / "released_statistics_check.json").write_text(
            json.dumps(
                {
                    "median_and_cv_from_replicates": dict(statistics_check),
                    "table_s6_cv_header_swap": dict(header_check),
                    "rtol": IDENTITY_RTOL,
                },
                indent=2,
            )
        )
        (out / "sourced_values.json").write_text(
            json.dumps(
                {
                    name: value.model_dump(mode="json")
                    for name, value in SOURCED_VALUES.items()
                },
                indent=2,
            )
        )
        pd.DataFrame(list(condition_rows)).to_csv(out / "conditions.csv", index=False)
        pd.DataFrame(
            [
                {
                    "uniprot": row.uniprot,
                    "released_gene": row.gene,
                    "released_bnumber": row.bnumber,
                    "stored_locus_tag": selection.locus_tag[row.uniprot],
                    "dataset": row.dataset,
                    "n_replicates": row.n_replicates,
                    "molecular_weight_da": row.molecular_weight_da,
                    "peptides_used": row.peptides_used,
                }
                for row in selection.kept
            ]
        ).to_csv(out / "protein_identifiers.csv", index=False)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# L0-L4 verification
# --------------------------------------------------------------------------- #
def assembly_pin_rule(records: Sequence[Mapping[str, Any]]) -> Any:
    """SUPPLEMENTARY L3: every record pins the BW25113 GenBank assembly."""
    from torchcell.verification.report import Level, LevelResult

    pins = {
        (
            record["reference"]["genome_reference"].get("assembly_set"),
            record["reference"]["genome_reference"].get("assembly_accession"),
        )
        for record in records
    }
    expected = {(BACTERIAL_ASSEMBLY_SETS[SCHMIDT_REFERENCE_STRAIN], "GCA_000750555.1")}
    return LevelResult(
        level=Level.L3,
        name="assembly_pin",
        passed=pins == expected,
        message=f"SUPPLEMENTARY: assembly pins {sorted(map(str, pins))}",
        details={"pins": sorted(map(str, pins))},
    )


def environment_uniqueness_rule(records: Sequence[Mapping[str, Any]]) -> Any:
    """SUPPLEMENTARY L1: each record carries its own environment.

    Every record's genotype is the same empty wild-type genotype, so the environment IS
    the record identity; two records sharing one environment would be indistinguishable.
    """
    from torchcell.verification.report import Level, LevelResult

    keys = [
        json.dumps(record["experiment"]["environment"], sort_keys=True)
        for record in records
    ]
    distinct = len(set(keys))
    return LevelResult(
        level=Level.L1,
        name="environment_uniqueness",
        passed=distinct == len(records),
        message=(
            f"SUPPLEMENTARY: {distinct} distinct environments over {len(records)} records"
        ),
        details={"n_records": len(records), "n_distinct": distinct},
    )


def replicate_rule(records: Sequence[Mapping[str, Any]]) -> Any:
    """SUPPLEMENTARY L2: a protein has an SE exactly when it has three replicates.

    Dataset 2 ran in biological triplicates and dataset 1 ran once, so a finite SE must
    accompany ``n_replicates == 3`` and ``nan`` must accompany ``n_replicates == 1``.
    """
    from torchcell.verification.report import Level, LevelResult

    bad = 0
    counts: dict[int, int] = {}
    for record in records:
        phenotype = record["experiment"]["phenotype"]
        errors = phenotype.get("protein_abundance_se") or {}
        for key, n in phenotype["n_replicates"].items():
            counts[int(n)] = counts.get(int(n), 0) + 1
            finite = not math.isnan(float(errors[key]))
            if finite != (int(n) == N_REPLICATES_DATASET2):
                bad += 1
    return LevelResult(
        level=Level.L2,
        name="se_matches_replicate_count",
        passed=bad == 0,
        message=(
            f"SUPPLEMENTARY: {bad} of {sum(counts.values())} values disagree with their "
            "replicate count"
        ),
        details={"n_bad": bad, "n_by_replicates": counts},
    )


def gene_containment_rule(
    records: Sequence[Mapping[str, Any]], universe: set[str]
) -> Any:
    """L4: every stored protein key is a locus of the pinned BW25113 assembly."""
    from torchcell.verification.report import Level, LevelResult

    measured: set[str] = set()
    for record in records:
        measured.update(record["experiment"]["phenotype"]["protein_abundance"])
        measured.update(record["reference"]["phenotype_reference"]["protein_abundance"])
    outside = sorted(measured - universe)
    return LevelResult(
        level=Level.L4,
        name="gene_containment_bw25113",
        passed=bool(measured) and not outside,
        message=(
            f"{len(measured)} measured protein keys; {len(outside)} outside the "
            f"{BACTERIAL_ASSEMBLY_SETS[SCHMIDT_REFERENCE_STRAIN]} locus universe"
        ),
        details={"outside": outside[:20], "n_universe": len(universe)},
    )


def verify_build(
    dataset_root: str,
    data_root: str | None = None,
    *,
    genome: EcoliK12Genome | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run this module's L0-L4 gate over a built tree and write the report.

    The shared ``verify_protein_dataset`` supplies L0 to L3 (structure, count, value
    fidelity, a key-matched finite reference, one measurement type); three
    SUPPLEMENTARY rows and the host-aware L4 containment are added here, because the
    yeast deletion-collection overlap that ``run_protein`` adds would say nothing about
    a BW25113 locus tag. The report is written to
    ``preprocess/verification_report.json``.
    """
    from torchcell.verification.protein import verify_protein_dataset
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    if genome is None:
        genome = bacterial_genome("ecoli", SCHMIDT_REFERENCE_STRAIN, data_root)
    report = verify_protein_dataset(
        records,
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{SI2_MIRROR_RELPATH}",
            citation_key=CITATION_KEY,
            sha256=SI2_SHA256,
            method=(
                "Table S6 'Protein copies/cell' block: one "
                "BacterialProteinAbundanceExperiment per loaded BW25113 growth "
                "condition, with SE = (copies / medianNormInt) * stdev(three Table S8 "
                "normInt replicates) / sqrt(3) for the dataset-2 proteins"
            ),
            page="si2.xlsx sheets 'Table S6', 'Table S8', 'Table S25'",
            retrieved=SI2_RETRIEVED_AT,
        ),
        expected_count=expected_count,
    )
    report.add(environment_uniqueness_rule(records))
    report.add(replicate_rule(records))
    report.add(assembly_pin_rule(records))
    report.add(gene_containment_rule(records, set(genome.genbank.loci)))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Deposit the raw mirror, or build and verify the dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deposit",
        action="store_true",
        help="re-run the recorded retrieval and write the raw mirror, then exit",
    )
    parser.add_argument(
        "--root",
        default="data/torchcell/proteome_schmidt2016",
        help="build tree, relative to DATA_ROOT",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    if args.deposit:
        staging = raw_mirror_dir(data_root) / "_staging"
        retrieved = retrieve_raw_files(staging)
        root = deposit_raw_mirror(source=retrieved[SI2], data_root=data_root)
        shutil.rmtree(staging)
        print(f"deposited {SI2} into {root}")
        return

    build_root = osp.join(data_root, args.root)
    dataset = ProteomeSchmidt2016Dataset(root=build_root)
    print(f"len = {len(dataset)}")
    ledger = json.loads(
        Path(build_root, "preprocess", "dropped_records.json").read_text()
    )
    print(
        json.dumps(
            {
                k: ledger[k]
                for k in (
                    "source_conditions",
                    "kept_records",
                    "dropped_records",
                    "source_protein_rows",
                    "kept_protein_keys",
                    "dropped_protein_rows",
                    "notes",
                )
            },
            indent=2,
        )
    )
    print(verify_build(build_root, data_root).summary())


if __name__ == "__main__":
    main()
