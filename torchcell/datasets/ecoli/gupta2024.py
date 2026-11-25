# torchcell/datasets/ecoli/gupta2024
# [[torchcell.datasets.ecoli.gupta2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/gupta2024
# Test file: tests/torchcell/datasets/ecoli/test_gupta2024.py
r"""Gupta 2024 global protein turnover in E. coli NCM3722, the 13 released conditions.

Gupta et al. 2024 (Nat Commun 15:5890, doi:10.1038/s41467-024-49920-8) switched the feed
of a steady-state culture from light to 15N ammonium and followed the shift of the
monoisotopic peak with TMTproC complement-reporter proteomics, fitting one turnover
parameter per protein in each of 13 growth conditions. This module is the first consumer
of ``ProteinTurnoverPhenotype``: one ``ProteinTurnoverExperiment`` per released
BIOLOGICAL REPLICATE of each condition (26 records), keyed on MG1655 b-numbers.

ONE RECORD PER BIOLOGICAL REPLICATE (#1174, #1178). The paper measured "13 different
growth conditions, each with two biological replicates" (Results), and the Reporting
Summary says what a replicate is: "Replication A in duplicates. These are biological
replicates measures months apart." Supplementary Data 1 releases each replicate as its
own column (``Half-life (hrs) in <condition>, replicate 1`` / ``..., replicate 2``) beside
the authors' ``Average of half-lives (hrs) in <condition>``, so every replicate column is
one record: ``replicate_id`` is the number the header prints (``"1"`` or ``"2"``), every
``n_replicates`` value is 1, and ``degradation_rate_se`` is ``None`` with a typed
``not_applicable_single_measurement`` gap. A replicate's number is ONE chemostat (or batch)
time course: eight TMTpro-multiplexed time points and every quantified peptide are fitted
together into one half-life, so the technical level (time points, peptides) is combined
inside the record by the authors' own fit, which is what makes the value valid at all.
The Reporting Summary states "No data was excluded from analyses.", so no replicate cell
is withdrawn and none is dropped. The authors' mean is no longer a record: grouping the
26 records by condition and averaging ``half_life`` reproduces it (the build asserts the
mean column equals the replicate mean for every key), and it survives as the
``phenotype_reference`` summary, the N-lim 6 h wild-type aggregate with
``replicate_id=None``.

WHAT THE SOURCE RELEASES, AND WHAT THE RECORD STORES. The released per-protein number is
a TOTAL HALF-LIFE in hours, not a rate: Supplementary Data 1 is "Half-lives for 3262
proteins across 13 growth conditions". The fitted parameter behind it is the total
turnover rate: the Supplementary Note fits "a free parameter $\\mathsf { k } _ { \\mathsf
{ D } } + \\mathsf { D } )$" and states the identity "Since $\\mathsf { k } _ { \\mathsf {
t o t a l } } = \\ln 2 / \\mathrm { T } _ { 1 / 2 }$". So:

- ``half_life`` holds the released replicate cell VERBATIM, in hours (the trailing ``*``
  ceiling flag stripped into ``censoring``). The two replicate columns of each condition
  are what this loader consumes; the "Average of half-lives (hrs)" column is read as the
  oracle and for the reference.
- ``degradation_rate`` is ``ln(2) / half_life``, per hour: an exact algebraic transform
  of the consumed cell through the Note's own identity, so the record is internally
  consistent (``degradation_rate == ln 2 / half_life`` for every key).
- ``measurement_type`` names the assay and its unit only
  (``n15_ammonium_tmtproc_total_turnover_rate_per_hour``): it is the TOTAL turnover rate,
  active degradation PLUS dilution. The dilution term itself is the environment's, and it
  is carried there as ``Environment.dilution_rate_per_hour`` rather than spelled into
  this string.

The ACTIVE degradation rate ``k_D`` is deliberately NOT stored: ``k_D = k_total - D``
goes negative wherever the fitted total half-life exceeds the doubling time, which the
release is full of (that is what a stable protein looks like), and
``ProteinTurnoverPhenotype`` requires a non-negative rate. Clipping it at zero would
manufacture a value the fit did not give.

THE REFERENCE'S STANDARD ERROR IS THE PAPER'S OWN REPLICATE FORMULA. (Since the
per-replicate conversion this statistic lives only on the ``phenotype_reference``
aggregate; a replicate record has none.) The Note's active-degradation
test uses "$\\mathsf { t } = ( \\bar { \\mathsf { x } } - 6 ) / \\sqrt { \\frac { \\sigma
^ { 2 } } { 2 } }$ where $\\bar { \\mathsf X }$ is the sample mean and $\\sigma ^ { 2 }$
is the sample variance calculated using the two replicates", i.e. sd / sqrt(n) with
n = 2. ``degradation_rate_se`` is that statistic on the RATE scale (the scale the label
is stored on), computed from the two replicate rates ``ln 2 / T_i``; it is ``nan`` for a
protein with one replicate in that condition or with a ceiling-flagged replicate, the
landed convention for "this key has no SE" (the validator admits NaN there).

CEILING CELLS ARE KEPT AND THEY TRAVEL AS CENSORED. 2,084 of the 61,811 released
replicate cells carry a trailing ``*``, which Supplementary Data 1 glosses as "* Protein
total half-life was set to ceiling for this dilution rate" and Supplementary Data 7
marks ``Undetermined``; 2,082 of them are cells of a stored key. They are right-censored,
not absent: the authors include them in their own "Average of half-lives" cell (verified:
every mean column equals the arithmetic mean of its available replicates, 0 mismatches in
28,610 two-replicate cells) and in Table 1's per-condition protein counts, which this
loader reproduces exactly as an L1 oracle. Dropping them would delete the stable half of
the proteome and would break that oracle.

The flag now travels on the record. ``censoring`` carries a ``Censoring`` for EVERY
stored key of every record, which the source's own oracle makes honest: Supplementary
Data 1's ``*`` and Supplementary Data 7's ``Undetermined`` agree on all 61,811 released
cells (0 disagreements, asserted at build time), so the release resolves the question for
every cell and absence would be the wrong answer rather than a cheaper one. In a
replicate record a key is ``right`` exactly when ITS cell is ceiling-flagged: 2,082 of
the 61,787 stored replicate keys are ``right`` and 59,705 are ``uncensored``. (On the
aggregate reference a key is ``right`` when ANY of its replicate cells is capped, because
the authors' mean then has a capped value in it; over the 13 condition means that was
1,989 of 33,187 keys, 93 of them with both replicates capped.) The side is stated on the RELEASED quantity, the
total half-life ``half_life`` stores verbatim, which is what the cap was applied to;
``degradation_rate`` is ``ln 2`` over it, a decreasing map, so the same key bounds the
rate from ABOVE. ``preprocess/ceiling_cells.csv`` is still written, because it carries
what the record does not: the replicate number and the measured ceiling of each doubling
time (2 h at 42 min, 4 h at 3 h, 8 h at 6 h, 16 h at 12 h).

THE IDENTIFIER ROUTE IS TWO LAYERS, BOTH MEASURED AND BOTH TYPED. The release keys on
UniProt entries (``sp|A5A614|YCIZ_ECOLI``) plus a gene-name column, neither of which is a
locus tag.

1. The pinned GenBank assembly carries UniProt accessions itself, as
   ``/db_xref="UniProtKB/Swiss-Prot:<acc>"`` on its CDS features (4,275 accessions, none
   mapping to two loci). 3,225 of the 3,262 released accessions resolve through it, on
   ``DerivedIdentifierRoute`` member ``uniprot_db_xref``.
2. The remaining rows go through ``reconcile_locus_tags`` on the released gene name,
   which the genome resolves at its symbol and synonym layers: 35 rows, ``gene_symbol``.

Every resolved row's route is now a typed ``DerivedIdentifierMapping``, built and
validated at read time -- all 3,262 released keys pass ``UNIPROT_ACCESSION_PATTERN``
through ``DerivedIdentifierMapping.uniprot_accession()`` -- and
``preprocess/identifier_route.csv`` records that enum value, not a free string. The
mapping cannot be attached per key, because a dict-keyed phenotype has no
``identifier_mapping`` field; it is the CSV and the build accounting that carry it.

Together the two layers reach 3,260 of 3,262 unique b-numbers, with no two rows claiming
one locus. Layer 1 takes precedence, and the one row where the two layers DISAGREE shows
why: ``sp|P0A6E9|BIOD2_ECOLI`` carries the gene name ``bioD``, whose symbol resolves to
``b0778`` (bioD1), while the accession resolves to ``b1593`` (bioD2) -- the UniProt entry
name is BIOD2 and the table carries ``bioD1`` as a separate row, so the symbol column is
the imprecise one.

The DELETION perturbations keep ``route="gene_symbol"``, measured rather than assumed:
the paper releases its knockouts only as symbols ("The ΔclpP, Δlon, ΔhslV single mutants
were generated by P1 transduction", "we knocked out the smpB gene"), never as
accessions, and those four symbols resolve through the genome's name layers. A
``uniprot_db_xref`` route there would have to invent an accession the source does not
release for the genotype.

TWO PROTEIN KEYS ARE DROPPED, and they are the only drops. ``sp|P07363-2|CHEA_ECOLI`` and
``sp|P63284-2|CLPB_ECOLI`` are alternative-isoform accessions whose canonical forms
(``P07363`` cheA, ``P63284`` clpB) are already separate rows of the same table, and whose
gene-name cell is empty. Resolving them would put two values on one locus; a
gene-level record cannot hold that, so they are not keys of any record.

THE CHEMOSTAT DILUTION RATE IS A TYPED FIELD OF THE ENVIRONMENT. Twelve of the 13
conditions are continuous culture, where "we can control the cell doubling time and
enforce steady state" and ``D`` is "the known variable $D$ (dilution rate)". No numeric
``D`` is printed per condition, but the paper fixes it twice over: Fig. 2's legend says
"The vertical line marks the dilution limit set by the 6-h doubling time", i.e. the
dilution-limited total half-life IS the doubling time, and the Supplementary Note's
identity ``k_total = ln 2 / T_half`` turns that half-life into a rate. So
``dilution_rate_per_hour`` is ``ln 2`` over the doubling time Table 1 publishes for that
condition: 0.231049 h^-1 at 3 h, 0.115525 at 6 h, 0.057762 at 12 h. The 42 min condition
is batch, so it carries no dilution rate at all. This is exact arithmetic on two quoted
sentences, not a model, and it is the same identity ``degradation_rate`` already uses.

THE GENOTYPE AXIS IS NOT PURELY WILD TYPE. Table 1's conditions 1-8 are the unperturbed
strain, so those records carry ``Genotype(perturbations=[])`` -- the environment is what
varies and the genotype says so rather than inventing a perturbation. Conditions 9-13 are
protease and SsrA-pathway knockouts in the same strain and carry one
``DeletionStrainPerturbation`` per deleted gene (three for the triple knockout).

THE STRAIN IS NCM3722 AND THE ASSEMBLY PIN IS MG1655, WHICH IS A DERIVED CHOICE. The
paper names only "E. coli strain NCM3722"; no NCM3722 assembly is in the genomes tier,
and the paper states no genotype for it relative to any sequenced K-12. NCM3722 is
carried as a ``StrainBackground`` on MG1655 with a ``ProvenanceGap`` on
``genotype_statement`` and on ``construction``, because the paper carries neither. The
pin is MG1655 because that is the assembly the released identifiers are written in: every
one of the 3,262 UniProt entry names ends in ``_ECOLI``, UniProt's K-12 mnemonic, and
3,260 of them reach a b-number of ``ecoli_K12_MG1655_ASM584v2``. The deletions came from
the Keio collection, which is BW25113, but by P1 transduction INTO NCM3722, so the
measured strain is an NCM3722 derivative and not a Keio clone.

THE PUBLISHED INTERVAL IS STORED FOR EXACTLY THE KEYS IT IS AN INTERVAL OF.
Supplementary Data 7 publishes a per-REPLICATE 95% confidence interval on the total
half-life, from ``curve_fit``'s parameter variance times the t quantile at
``dof = i x 8 - 1`` with ``i`` the protein's peptide count.

A replicate record's value IS that replicate's fitted half-life, so the released
interval of the same replicate cell is the record's own interval, and every replicate
record stores it for every determined cell: 59,621 bound pairs over the 26 records. The
84 cells whose released upper half-life endpoint is zero or negative are refused (no
finite non-negative rate bound exists; clamping would manufacture one) and listed in the
build accounting, and the 2,082 ceiling cells carry ``Undetermined`` and no bound. This is
a statistic the SOURCE releases (the fit's own uncertainty of one measurement), not a
dispersion over replicates, so the per-replicate policy keeps it. The rules below still
describe the aggregate ``phenotype_reference``:

- A ONE-REPLICATE key's stored value IS that replicate's fitted half-life, so the
  released interval is the record's interval and is stored losslessly. 4,587 of the
  33,187 stored keys have one replicate; 292 of those are ceiling-flagged and their
  interval cell reads ``Undetermined``; of the 4,295 that remain, 4,270 are stored. The
  bounds are inverted onto the RATE scale, which is exact arithmetic through the Note's
  own identity (``r = ln 2 / T`` is decreasing, so the two bounds swap), and the build
  asserts that every stored pair brackets the stored rate.
- The other 25 are REFUSED and listed in the build accounting: their released upper
  half-life bound is zero or negative (the rate interval's lower bound crossed zero), so
  no finite non-negative rate bound exists for them. Clamping one at zero would
  manufacture a bound the fit did not give.
- A TWO-REPLICATE key gets no bounds. The interval is published per replicate, and the
  interval of the MEAN of two fits is not published; combining the two needs ``i``, the
  per-protein peptide count, which the release does not carry. ``degradation_rate_lower``
  and ``degradation_rate_upper`` are ragged by design and a key absent from them is
  exactly "the source publishes no interval for this key", so no per-key gap restates it.

``confidence_level`` is 0.95 and ``interval_method`` names the construction, both
sourced. ``degradation_rate_se`` stays the paper's own two-replicate ``sd / sqrt(n)``: a
replicate SE and a fitted interval are different statistics, and this release carries
each one for the keys the other cannot cover.

WHAT IS STILL GAPPED RATHER THAN GUESSED. There is no per-protein uncertainty of the
released MEAN for a two-replicate key, because ``i`` is unreleased and the interval does
not invert back to the standard deviation it came from; and no per-protein synthesis rate
exists at all, since the model fits one free parameter. The latter is a typed
``ProvenanceGap`` on every phenotype; a replicate record carries a second one, on
``degradation_rate_se``, with reason ``not_applicable_single_measurement``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import os.path as osp
import shutil
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import openpyxl
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
from torchcell.datamodels.media import MOPS_MINIMAL
from torchcell.datamodels.schema import (
    BacterialAssemblySet,
    BacterialGeneNamespace,
    Censoring,
    Concentration,
    ConcentrationUnit,
    DeletionStrainPerturbation,
    DerivedIdentifierMapping,
    DerivedIdentifierRoute,
    Environment,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    PhysicalFactor,
    ProteinTurnoverExperiment,
    ProteinTurnoverExperimentReference,
    ProteinTurnoverPhenotype,
    Publication,
    ReferenceGenome,
    StrainBackground,
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
from torchcell.literature.retrieve import pmc_cloud_url
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12MG1655Genome
from torchcell.verification.levels import (
    l0_structural,
    l1_count,
    l2_value_fidelity,
    l3_convention,
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

__all__ = [
    "CITATION_KEY",
    "CONDITIONS",
    "EXPECTED_RECORDS",
    "ProteinTurnoverGupta2024Dataset",
    "deposit_raw_mirror",
    "load_manifest",
    "manifest_sha256",
    "raw_mirror_dir",
    "verify_build",
]

# --------------------------------------------------------------------------- #
# Paper, mirror and raw-file pins
# --------------------------------------------------------------------------- #
CITATION_KEY = "guptaGlobalProteinTurnover2024"
DOI = "10.1038/s41467-024-49920-8"
TITLE = (
    "Global protein turnover quantification in Escherichia coli reveals cytoplasmic "
    "recycling under nitrogen limitation"
)
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
PMCID = "PMC11246515"
PMC_PREFIX = f"{PMCID}.1"
PRIDE_ACCESSION = "PXD042444"
CODE_DOI = "10.5281/zenodo.10895828"

#: Supplementary Data 1 (MOESM4): the half-life matrix every record is built from.
HALF_LIVES_FILENAME = "41467_2024_49920_MOESM4_ESM.xlsx"
HALF_LIVES_REL = f"si/{HALF_LIVES_FILENAME}"
HALF_LIVES_SHA256 = "546cc57b8386a15217dd76c10077844073ac083755cdecbd99f9284c1cbe0acc"
HALF_LIVES_SHEET = "TableS1"
#: Supplementary Data 7 (MOESM10): the per-replicate 95% CIs, consumed as the oracle
#: that an ``Undetermined`` interval is exactly a ceiling-flagged half-life.
CONFIDENCE_FILENAME = "41467_2024_49920_MOESM10_ESM.xlsx"
CONFIDENCE_REL = f"si/{CONFIDENCE_FILENAME}"
CONFIDENCE_SHA256 = "149a361b9b4c783c046f702364ba083730a294a1ef5ef3b6bcfc73a9ff27d4fe"
CONFIDENCE_SHEET = "TableS7"
RAW_RETRIEVED_AT = "2026-10-07"

#: Mirrored OCR files the sourced values quote (torchcell-library, not the raw mirror).
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "1be6b9d958235a80ce2421c552299dd6bf2371cccfed28a24bd813a30d3d35a2"
SI1_MD = "si/si1.md"
SI1_MD_SHA256 = "5f913f0b846865ad0bcac57cfe7a9f1895d04dd2ec7b16db11bf28be10d1062a"
SI3_MD = "si/si3.md"
SI3_MD_SHA256 = "87a1760129b2feb312a009dbe1486e3fbbc75fe6fcdbe0711b52576bf28ddc14"
#: The Nature Portfolio Reporting Summary (MOESM12), which says what a replicate is.
SI12_MD = "si/si12.md"
SI12_MD_SHA256 = "e7fd2a450b11d86d3fde4d55eb2ca4e2f7f7d8808baf506961194e6c89708582"

MG1655_STRAIN: Literal["MG1655"] = "MG1655"
MG1655_ASSEMBLY_SET: BacterialAssemblySet = "ecoli_K12_MG1655_ASM584v2"
MG1655_NAMESPACE: BacterialGeneNamespace = "ecoli_k12_mg1655_bnumber"
#: The strain every record was measured in; an NCM3722 assembly is not in the tier.
MEASURED_STRAIN = "NCM3722"

#: Released rows, and the two whose key is an alternative isoform of another row's.
SOURCE_ROWS = 3262
DROPPED_PROTEIN_KEYS = ("sp|P07363-2|CHEA_ECOLI", "sp|P63284-2|CLPB_ECOLI")
#: Records: one per released biological replicate, two per growth condition.
EXPECTED_RECORDS = 26
#: The two replicate numbers each condition's column headers print ("..., replicate 1").
REPLICATE_IDS: tuple[str, str] = ("1", "2")
#: Measured on the pinned workbook: 3,260 of 3,262 released protein keys reach a
#: b-number of the pinned assembly (0.99939). The threshold sits just below, because
#: the two that do not are the isoform rows above; a real drop means the keying changed.
MIN_RESOLVED_FRACTION = 0.998

_UNIPROT_XREF_PREFIXES = ("UniProtKB/Swiss-Prot:", "UniProtKB/TrEMBL:")


# --------------------------------------------------------------------------- #
# Sourced values: every number below quotes sha256-pinned mirrored bytes
# --------------------------------------------------------------------------- #
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


def _si_note(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the pinned Supplementary Note OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI1_MD,
            citation_key=CITATION_KEY,
            sha256=SI1_MD_SHA256,
            method="MinerU OCR of the Supplementary Information PDF (mirror)",
            page=page,
        ),
    )


def _si_legend(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to the pinned Supplementary Data description sheet (MOESM3)."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI3_MD,
            citation_key=CITATION_KEY,
            sha256=SI3_MD_SHA256,
            method="MinerU OCR of the Supplementary Data description PDF (mirror)",
            page=page,
        ),
    )


def _reporting_summary(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the pinned Reporting Summary OCR (MOESM12)."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI12_MD,
            citation_key=CITATION_KEY,
            sha256=SI12_MD_SHA256,
            method="MinerU OCR of the Reporting Summary PDF (mirror)",
            page=page,
        ),
    )


def _workbook(
    value: Any,
    quote: str,
    *,
    relpath: str,
    sha256: str,
    note: str | None = None,
    page: str | None = None,
) -> SourcedValue:
    """Bind a value to a verbatim cell of one deposited supplementary workbook."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{relpath}",
            citation_key=CITATION_KEY,
            sha256=sha256,
            method="published Supplementary Data workbook (raw mirror)",
            page=page or f"{relpath} cell A2/A3",
        ),
    )


_PAGE_METHODS_CHEMOSTAT = "Methods, 'Chemostat growth and labeling'"
_PAGE_METHODS_BATCH = "Methods, 'Minimal media batch cultures'"
_PAGE_METHODS_STRAINS = "Methods, 'Strain construction'"
_PAGE_METHODS_STATS = "Methods, 'Statistics and reproducibility'"
_PAGE_RESULTS = "Results"
_PAGE_SI_FITTING = "Supplementary Note, 'Protein half-life fitting' (lines 217-278)"
_PAGE_SI_SCALING = "Supplementary Note, degradation-rate scaling derivation"
_PAGE_SI_LEGENDS = "Supplementary Data file descriptions"

STRAIN = _paper(
    MEASURED_STRAIN,
    "E. coli strain NCM3722 was grown in continuous culture chemostats at",
    page=_PAGE_METHODS_CHEMOSTAT,
    note="the only strain designation the paper gives for the turnover measurements; "
    "no NCM3722 assembly is deposited, so the records pin MG1655 and carry NCM3722 as "
    "a StrainBackground",
)
TEMPERATURE_CHEMOSTAT_C = _paper(
    37.0,
    "continuous culture chemostats at $3 7 ^ { \\circ } \\mathrm { C }$",
    page=_PAGE_METHODS_CHEMOSTAT,
)
TEMPERATURE_BATCH_C = _paper(
    37.0,
    "NCM3722 cells were grown at $3 7 ^ { \\circ } \\mathrm { C }$ in "
    "$4 0 \\mathsf { m M }$ MOPS media",
    page=_PAGE_METHODS_BATCH,
)
CHEMOSTAT_PH = _paper(
    7.2,
    "pH was maintained at $7 . 2 \\pm 0 . 1$",
    page=_PAGE_METHODS_CHEMOSTAT,
    note="controlled in the chemostat only; the batch culture's pH is not stated, so "
    "the batch environment carries no pH factor",
)
BASE_MEDIUM = _paper(
    "40 mM MOPS (Teknova M2120) with glucose, ammonium and phosphate added separately",
    "MOPS media (M2120, Teknova) was used with glucose",
    page=_PAGE_METHODS_CHEMOSTAT,
    note="MEDIA_LIBRARY key MOPS_MINIMAL is the same Neidhardt MOPS base with the same "
    "9.5 mM ammonium chloride and 1.32 mM dipotassium hydrogen phosphate and no carbon "
    "source; this paper's four media are that base with the stated glucose added and "
    "the limiting salt reduced",
)
FULL_LEVELS = _paper(
    {"glucose_percent_w_v": 0.4, "ammonium_mM": 9.5, "phosphate_mM": 1.32},
    "glucose $( 0 . 4 \\% \\mathsf { w } /$ v, Sigma G8270), ammonium "
    "$9 . 5 \\mathsf { m M }$",
    page=_PAGE_METHODS_CHEMOSTAT,
)
REDUCED_C_AND_N = _paper(
    {"glucose_percent_w_v": 0.08, "ammonium_mM": 1.9},
    "For Cand N-limiting media, glucose and ammonium concentrations were reduced by "
    "fivefold",
    page=_PAGE_METHODS_CHEMOSTAT,
    note="the OCR runs 'For C- and' together as 'For Cand'; the two reduced values are "
    "quoted separately on their components",
)
REDUCED_VALUES = _paper(
    {"glucose_percent_w_v": 0.08, "ammonium_mM": 1.9},
    "$0 . 0 8 \\%$ and $1 . 9 \\mathsf { m M }$",
    page=_PAGE_METHODS_CHEMOSTAT,
)
REDUCED_P = _paper(
    0.132, "The P-limiting medium contains", page=_PAGE_METHODS_CHEMOSTAT
)
REDUCED_P_VALUE = _paper(
    0.132, "$0 . 1 3 2 \\mathsf { m M }$", page=_PAGE_METHODS_CHEMOSTAT
)
REPLICATES = _paper(
    2,
    "13 different growth conditions, each with two biological replicates",
    page=_PAGE_RESULTS,
    note="the released matrix carries two replicate columns per condition; each "
    "column is one record (replicate_id = the number its header prints), and a protein "
    "quantified in only one of them is a key of that replicate's record only",
)
REPLICATES_ARE_BIOLOGICAL = _reporting_summary(
    "biological",
    "Replication A in duplicates. These are biological replicates measures months "
    "apart.",
    page="Reporting Summary, 'Life sciences study design', Replication",
    note="[sic, OCR]; the two replicate columns are two independent cultures measured "
    "months apart, not two injections of one culture, so each is its own record",
)
NO_DATA_EXCLUDED = _reporting_summary(
    "none withdrawn",
    "No data was excluded from analyses.",
    page="Reporting Summary, 'Life sciences study design', Data exclusions",
    note="no released replicate cell is a withdrawn sample, so none is dropped",
)
REPLICATE_COLUMNS = _workbook(
    REPLICATE_IDS,
    "Half-life (hrs) in minimal media batch, doubling time 42 mins, replicate 1",
    relpath=HALF_LIVES_REL,
    sha256=HALF_LIVES_SHA256,
    note="the header row (row 4) prints the replicate number on each of the 26 "
    "replicate columns; every Condition pins its two headers verbatim",
    page=f"{HALF_LIVES_REL} sheet TableS1, header row 4, column D",
)
SINGLE_REPLICATE_SE_NOTE = (
    "one record is one biological replicate's fitted half-life (one chemostat or batch "
    "time course), so a standard error over replicates does not exist for it; the "
    "paper's two-replicate sd / sqrt(2) is carried by the aggregate phenotype_reference"
)
REPLICATION_SCOPE = _paper(
    "every strain and condition",
    "All protein turnover rate measurements were replicated for each strain and "
    "condition.",
    page=_PAGE_METHODS_STATS,
)
SINGLE_KO_CONSTRUCTION = _paper(
    "P1 transduction from the Keio collection into NCM3722",
    "The ΔclpP, Δlon, ΔhslV single mutants were generated by P1 "
    "transduction from the Keio collection87 into E. coli strain NCM3722.",
    page=_PAGE_METHODS_STRAINS,
    note="the deletion alleles are Keio (BW25113) alleles moved into NCM3722, so the "
    "measured strain is an NCM3722 derivative and the collection field records where "
    "the allele came from",
)
TRIPLE_KO_SOURCE = _paper(
    "Basan lab",
    "The ΔclpPΔlonΔhslV triple knockout was provided by the Basan lab88.",
    page=_PAGE_METHODS_STRAINS,
    note="a gift strain, so its three deletions carry no Keio collection attribution",
)
SMPB_KO = _paper(
    "smpB",
    "we knocked out the smpB gene (codes for a protein in the SsrA tagging "
    "complex75), and measured gene-by-gene protein turnover in nitrogen limitation "
    "with a 6-h doubling time in duplicate.",
    page=_PAGE_RESULTS,
)
SAMPLING_TABLE = _paper(
    "Table 2",
    "Table 2 | The time point used for the sample collection for all doubling times "
    "analyzed in this study",
    page="Methods, Table 2",
    note="duration_hours is the LAST sampling time of a doubling time's series, in "
    "hours: the window from the feed switch to the final proteomics sample",
)
TOTAL_TURNOVER_IS_THE_FITTED_PARAMETER = _paper(
    "k_D + D",
    "(total turnover rate) is obtained by fitting this model to the experimentally "
    "measured signal of peptides.",
    page=_PAGE_RESULTS,
)
HALF_LIFE_IDENTITY = _si_note(
    "k_total = ln 2 / T_half",
    "Since $\\mathsf { k } _ { \\mathsf { t o t a l } } = \\ln 2 / \\mathrm { T } _ "
    "{ 1 / 2 }$",
    page=_PAGE_SI_SCALING,
    note="the identity that turns the released total half-life into the stored total "
    "turnover rate; the transform is exact, not a model",
)
FIT_PARAMETER = _si_note(
    "k_D + D",
    "for each protein is obtained by minimizing the least square difference",
    page=_PAGE_SI_FITTING,
)
CONFIDENCE_INTERVALS = _si_note(
    "95% CI per replicate",
    "265 We also assign confidence intervals to the fitted half-lives.",
    page=_PAGE_SI_FITTING,
    note="the interval is curve_fit's parameter standard deviation times the t "
    "quantile at dof = i x 8 - 1 with i the protein's peptide count; i is not "
    "released, so the interval does not invert back to a standard deviation",
)
CONFIDENCE_LEVEL_SOURCE = _si_note(
    0.95,
    "This is the error with $67 \\%$ confidence interval. To get the $9 5 \\%$ CI, we "
    "should",
    page=_PAGE_SI_FITTING,
    note="the level the stored bounds are at; the workbook's own column header says it "
    "too ('Total half-life 95% confidence interval in <condition>, replicate 1')",
)
INTERVAL_FROM_CURVE_FIT_VARIANCE = _si_note(
    "curve_fit parameter variance",
    "with returning the parameter estimate also returns the variance estimate of the "
    "fitted parameter.",
    page=_PAGE_SI_FITTING,
)
INTERVAL_T_QUANTILE = _si_note(
    "t ppf at 0.975",
    "Therefore we evaluate, Percent point function (inverse of cdf) at a value of "
    "0.975 given degrees",
    page=_PAGE_SI_FITTING,
)
INTERVAL_DOF = _si_note(
    "dof = i x 8 - 1",
    "of freedom (dof= Number of data points – number of estimated parameters $= \\dot "
    "{ 1 } \\times 8 - 1$ )",
    page=_PAGE_SI_FITTING,
    note="i is the protein's peptide count and is not released, which is why the "
    "interval is stored as an interval and never inverted to a standard error",
)
DILUTION_IS_THE_CHEMOSTAT_VARIABLE = _paper(
    "D",
    "the known variable $D$ (dilution rate)",
    page=_PAGE_RESULTS,
    note="the dilution rate is the chemostat's controlled variable, not a fitted one, "
    "which is why it belongs to the environment rather than to the phenotype",
)
DILUTION_LIMIT_IS_THE_DOUBLING_TIME = _paper(
    "T_dilution_limit = doubling time",
    "The vertical line marks the dilution limit set by the 6-h doubling time.",
    page="Fig. 2 legend",
    note="a purely diluting protein's total half-life is the doubling time, so with the "
    "Supplementary Note's identity k_total = ln 2 / T_half the dilution rate is ln 2 "
    "over the doubling time; this is the arithmetic Environment.dilution_rate_per_hour "
    "carries",
)
DOUBLING_TIMES_MEASURED = _paper(
    {"batch": 0.7, "chemostat": (3.0, 6.0, 12.0)},
    "rapidly doubling cells with unlimited growth in minimal medium (0.7 h)",
    page=_PAGE_RESULTS,
    note="the four doubling times, the chemostat three of which set a dilution rate; "
    "Table 1 names each condition's own doubling time and is quoted per condition",
)
DOUBLING_TIMES_CHEMOSTAT = _paper(
    (3.0, 6.0, 12.0),
    "slower doubling cells in carbonlimited chemostats (3 h, 6 h, and 12 h)",
    page=_PAGE_RESULTS,
)
REPLICATE_SE_FORMULA = _si_note(
    "sd / sqrt(n)",
    "where $\\bar { \\mathsf X }$ is the sample mean and $\\sigma ^ { 2 }$ is the "
    "sample variance calculated using the",
    page=_PAGE_SI_FITTING,
    note="the paper's own n = 2 dispersion for a condition's mean; "
    "degradation_rate_se applies it to the two replicate RATES, the scale the label "
    "is stored on",
)
DATA1_LEGEND = _si_legend(
    SOURCE_ROWS,
    "Description: Half-lives for 3262 proteins across 13 growth conditions",
    page=_PAGE_SI_LEGENDS,
)
DATA7_LEGEND = _si_legend(
    SOURCE_ROWS,
    "Confidence intervals on half-lives for 3262 proteins across 13 growth conditions",
    page=_PAGE_SI_LEGENDS,
)
WT_IS_THE_KO_COMPARATOR = _paper(
    "N-lim wild type, 6 h doubling",
    "comparing the protein half-lives in protease knockout (KO) with wildtype (WT) "
    "cells",
    page=_PAGE_RESULTS,
    note="why the N-lim 6 h wild-type condition is the phenotype_reference: it is the "
    "comparator the paper reads every knockout against",
)
PRIDE_DEPOSIT = _paper(
    PRIDE_ACCESSION,
    "deposited to the ProteomeXchange Consortium via the PRIDE partner repository "
    "with the dataset identifier PXD042444",
    page="Data availability",
)
CEILING_NOTE = _workbook(
    "ceiling",
    "* Protein total half-life was set to ceiling for this dilution rate",
    relpath=HALF_LIVES_REL,
    sha256=HALF_LIVES_SHA256,
)
UNDETERMINED_NOTE = _workbook(
    "Undetermined",
    "*Undetermined = Model fitting resulted in half-lives much greater than dilution "
    "rate; total half-life was set to ceiling for that condition",
    relpath=CONFIDENCE_REL,
    sha256=CONFIDENCE_SHA256,
)

#: Where we looked for NCM3722's lineage and genotype, for the gaps below.
_LOOKED_IN_PAPER = Provenance(
    source_uri=PAPER_MD,
    citation_key=CITATION_KEY,
    sha256=PAPER_MD_SHA256,
    method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
    page=(
        "Methods 'Chemostat growth and labeling', 'Minimal media batch cultures', "
        "'Strain construction', Table 3 'Information for all the strains and "
        "plasmids'; si1.md Supplementary Note"
    ),
)

#: NCM3722's lesions relative to the pinned MG1655 assembly are not stated anywhere in
#: the paper, so the background declares them absent instead of guessing a lineage.
NCM3722_GENOTYPE_GAP = ProvenanceGap(
    field="genotype_statement",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_LOOKED_IN_PAPER,
    note=(
        "the paper names 'E. coli strain NCM3722' and gives no genotype string for it; "
        "it states no relation to MG1655 or BW25113 and cites no NCM3722 sequence, so "
        "the alleles this background carries beyond the pinned assembly are unknown "
        "rather than empty"
    ),
)
NCM3722_CONSTRUCTION_GAP = ProvenanceGap(
    field="construction",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_LOOKED_IN_PAPER,
    note=(
        "Table 3 lists only the DY378-derived ftsH strains and the plasmid pKD4; it "
        "gives no construction for NCM3722 itself, which the paper treats as a stock"
    ),
)
#: The release carries no synthesis rate: the fit has one free parameter, the total
#: turnover rate, and the paper reports no separate incorporation rate per protein.
SYNTHESIS_RATE_GAP = ProvenanceGap(
    field="synthesis_rate",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=SI1_MD,
        citation_key=CITATION_KEY,
        sha256=SI1_MD_SHA256,
        method="MinerU OCR of the Supplementary Information PDF (mirror)",
        page=_PAGE_SI_FITTING,
    ),
    note=(
        "the model fits one free parameter per protein, the total turnover rate "
        "k_D + D; no per-protein synthesis or incorporation rate is released in "
        "Supplementary Data 1-8"
    ),
)


# --------------------------------------------------------------------------- #
# The 13 released conditions
# --------------------------------------------------------------------------- #
#: MOPS component names this paper's four media override, as ``MOPS_MINIMAL`` spells
#: them. Matched by compound name so a change in the library entry fails loudly.
AMMONIUM_COMPONENT = "ammonium chloride"
PHOSPHATE_COMPONENT = "dipotassium hydrogen phosphate"
#: ``resolved_compound("glucose")`` resolves to this canonical name.
GLUCOSE_COMPONENT = "D-glucose"

#: Doubling time -> the last sampling time of its series, in minutes (Methods Table 2).
#: The whole row is quoted so the series, not only its last entry, is pinned.
SAMPLING_SERIES: dict[str, tuple[str, int]] = {
    "42 min": (
        "<td>42 min</td><td>Batch</td><td>0, 5, 12, 20, 30, 45, 60,175</td>",
        175,
    ),
    "3 h": (
        "<td>3h</td><td>Chemostat</td><td>0, 45, 105, 186, 270, 366, 510, 729</td>",
        729,
    ),
    "6 h": (
        "<td>6h</td><td>Chemostat</td><td>0, 107, 229, 376, 548, 735, 1024, 1612</td>",
        1612,
    ),
    "12 h": (
        "<td>12h</td><td>Chemostat</td><td>0, 173, 302, 444, 649, 873, 1230, 2166</td>",
        2166,
    ),
}

#: Doubling time -> its length in hours. The batch culture's 0.7 h is the paper's own
#: rounding of its 42 min label ("minimal medium (0.7 h)"); the three chemostat values
#: are Table 1's, quoted per condition on ``Condition.table1_quote``.
DOUBLING_HOURS: dict[str, float] = {"42 min": 0.7, "3 h": 3.0, "6 h": 6.0, "12 h": 12.0}

#: What one stored number is, with its unit. The reactor and the doubling time are NOT
#: in here: the dilution regime is the environment's, and it is carried typed, on
#: ``Environment.dilution_rate_per_hour``.
MEASUREMENT_TYPE = "n15_ammonium_tmtproc_total_turnover_rate_per_hour"

#: The level of every stored bound, as the Note and the workbook column headers state.
CONFIDENCE_LEVEL: float = float(CONFIDENCE_LEVEL_SOURCE.value)
#: How the stored bounds were produced, and the one transform applied to them.
INTERVAL_METHOD = (
    "curve_fit_parameter_variance_t_ppf_0.975_dof_8i_minus_1_"
    "published_per_replicate_on_the_total_half_life_inverted_to_the_rate_scale"
)

Limitation = Literal["none", "C", "N", "P"]


class Condition(BaseModel):
    """One released growth condition, with the exact workbook columns it is read from.

    Every column header is pinned verbatim, so a reshaped workbook fails at read time
    rather than silently shifting a condition's values. ``table1_proteins`` is the
    per-condition protein count Table 1 publishes and the L1 oracle this loader
    reproduces; ``ceiling_hours`` is the single ceiling value measured in that
    condition's two replicate columns. ``censored_keys`` and ``interval_keys`` are the
    measured per-condition counts of the two #753 maps, pinned the same way
    ``stored_proteins`` is, so a change in either is a build failure rather than a drift.
    Those four describe the aggregate (the authors' mean, the ``phenotype_reference``).
    ``replicate_proteins``, ``replicate_censored_keys`` and ``replicate_interval_keys``
    pin the same three counts for each of the two replicate RECORDS, in replicate order,
    measured on the pinned workbooks; the two replicates' key sets union to
    ``stored_proteins``.
    """

    model_config = ConfigDict(frozen=True)

    key: str
    short_name: str
    limitation: Limitation
    reactor: Literal["batch", "chemostat"]
    doubling_time: str
    deleted_symbols: tuple[str, ...]
    table1_proteins: int
    stored_proteins: int
    censored_keys: int
    interval_keys: int
    replicate_proteins: tuple[int, int]
    replicate_censored_keys: tuple[int, int]
    replicate_interval_keys: tuple[int, int]
    table1_quote: str
    ceiling_hours: float
    header_replicate_1: str
    header_replicate_2: str
    header_mean: str
    header_ci_1: str
    header_ci_2: str

    @property
    def duration_hours(self) -> float:
        """The labeling window: the last sampling time of this doubling time, in hours."""
        return SAMPLING_SERIES[self.doubling_time][1] / 60.0

    @property
    def doubling_hours(self) -> float:
        """This condition's doubling time, in hours."""
        return DOUBLING_HOURS[self.doubling_time]

    @property
    def dilution_rate_per_hour(self) -> float | None:
        """``ln 2`` over the doubling time for a chemostat; ``None`` for the batch.

        The dilution-limited total half-life IS the doubling time ("the dilution limit
        set by the 6-h doubling time"), and the Supplementary Note's identity
        ``k_total = ln 2 / T_half`` turns that half-life into the rate. A batch culture
        has no dilution rate, so it carries none.
        """
        if self.reactor == "batch":
            return None
        return math.log(2.0) / self.doubling_hours


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        key="wt_minimal_batch_42min",
        short_name="MinimalMedia",
        limitation="none",
        reactor="batch",
        doubling_time="42 min",
        deleted_symbols=(),
        table1_proteins=2555,
        stored_proteins=2553,
        censored_keys=177,
        interval_keys=338,
        replicate_proteins=(2414, 2328),
        replicate_censored_keys=(75, 103),
        replicate_interval_keys=(2324, 2206),
        table1_quote=(
            "<td>1.</td><td>Wild type</td><td>Minimal media</td><td>Batch</td>"
            "<td>42 min</td><td>2</td><td>2555</td>"
        ),
        ceiling_hours=2.0,
        header_replicate_1=(
            "Half-life (hrs) in minimal media batch, doubling time 42 mins, replicate 1"
        ),
        header_replicate_2=(
            "Half-life (hrs) in minimal media batch, doubling time 42 mins, replicate 2"
        ),
        header_mean=(
            "Average of half-lives (hrs) in minimal media batch, doubling time 42 mins"
        ),
        header_ci_1=(
            "Total half-life 95% confidence interval in minimal media batch, doubling "
            "time 42 mins, replicate 1"
        ),
        header_ci_2=(
            "Total half-life 95% confidence interval in minimal media batch, doubling "
            "time 42 mins, replicate 2"
        ),
    ),
    Condition(
        key="wt_clim_3h",
        short_name="C-lim3",
        limitation="C",
        reactor="chemostat",
        doubling_time="3 h",
        deleted_symbols=(),
        table1_proteins=2665,
        stored_proteins=2664,
        censored_keys=240,
        interval_keys=302,
        replicate_proteins=(2458, 2526),
        replicate_censored_keys=(138, 122),
        replicate_interval_keys=(2316, 2398),
        table1_quote=(
            "<td>2.</td><td>Wild type</td><td>C-lim</td><td>Chemostat</td><td>3h</td>"
            "<td>2</td><td>2665</td>"
        ),
        ceiling_hours=4.0,
        header_replicate_1=(
            "Half-life (hrs) in C-lim chemostat, doubling time 3 hrs, replicate 1"
        ),
        header_replicate_2=(
            "Half-life (hrs) in C-lim chemostat, doubling time 3 hrs, replicate 2"
        ),
        header_mean=(
            "Average of half-lives (hrs) in C-lim chemostat, doubling time 3 hrs"
        ),
        header_ci_1=(
            "Total half-life 95% confidence interval in C-lim chemostat, doubling time "
            "3 hrs, replicate 1"
        ),
        header_ci_2=(
            "Total half-life 95% confidence interval in C-lim chemostat, doubling time "
            "3 hrs, replicate 2"
        ),
    ),
    Condition(
        key="wt_clim_6h",
        short_name="C-lim6",
        limitation="C",
        reactor="chemostat",
        doubling_time="6 h",
        deleted_symbols=(),
        table1_proteins=2651,
        stored_proteins=2650,
        censored_keys=166,
        interval_keys=331,
        replicate_proteins=(2531, 2420),
        replicate_censored_keys=(55, 118),
        replicate_interval_keys=(2474, 2300),
        table1_quote=(
            "<td>3.</td><td>Wild type</td><td>C-lim</td><td>Chemostat</td><td>6h</td>"
            "<td>2</td><td>2651</td>"
        ),
        ceiling_hours=8.0,
        header_replicate_1=(
            "Half-life (hrs) in C-lim chemostat, doubling time 6 hrs, replicate 1"
        ),
        header_replicate_2=(
            "Half-life (hrs) in C-lim chemostat, doubling time 6 hrs, replicate 2"
        ),
        header_mean=(
            "Average of half-lives (hrs) in C-lim chemostat, doubling time 6 hrs"
        ),
        header_ci_1=(
            "Total half-life 95% confidence interval in C-lim chemostat, doubling time "
            "6 hrs, replicate 1"
        ),
        header_ci_2=(
            "Total half-life 95% confidence interval in C-lim chemostat, doubling time "
            "6 hrs, replicate 2"
        ),
    ),
    Condition(
        key="wt_clim_12h",
        short_name="C-lim12",
        limitation="C",
        reactor="chemostat",
        doubling_time="12 h",
        deleted_symbols=(),
        table1_proteins=2697,
        stored_proteins=2696,
        censored_keys=156,
        interval_keys=404,
        replicate_proteins=(2400, 2550),
        replicate_censored_keys=(99, 71),
        replicate_interval_keys=(2298, 2470),
        table1_quote=(
            "<td>4.</td><td>Wild type</td><td>C-lim</td><td>Chemostat</td><td>12 h</td>"
            "<td>2</td><td>2697</td>"
        ),
        ceiling_hours=16.0,
        header_replicate_1=(
            "Half-life (hrs) in C-lim chemostat, doubling time 12 hrs, replicate 1"
        ),
        header_replicate_2=(
            "Half-life (hrs) in C-lim chemostat, doubling time 12 hrs, replicate 2"
        ),
        header_mean=(
            "Average of half-lives (hrs) in C-lim chemostat, doubling time 12 hrs"
        ),
        header_ci_1=(
            "Total half-life 95% confidence interval in C-lim chemostat, doubling time "
            "12 hrs, replicate 1"
        ),
        header_ci_2=(
            "Total half-life 95% confidence interval in C-lim chemostat, doubling time "
            "12 hrs, replicate 2"
        ),
    ),
    Condition(
        key="wt_plim_6h",
        short_name="P-lim6",
        limitation="P",
        reactor="chemostat",
        doubling_time="6 h",
        deleted_symbols=(),
        table1_proteins=2469,
        stored_proteins=2468,
        censored_keys=62,
        interval_keys=357,
        replicate_proteins=(2181, 2383),
        replicate_censored_keys=(24, 40),
        replicate_interval_keys=(2157, 2340),
        table1_quote=(
            "<td>5.</td><td>Wild type</td><td>P-lim</td><td>Chemostat</td><td>6h</td>"
            "<td>2</td><td>2469</td>"
        ),
        ceiling_hours=8.0,
        header_replicate_1=(
            "Half-life (hrs) in P-lim chemostat, doubling time 6 hrs, replicate 1"
        ),
        header_replicate_2=(
            "Half-life (hrs) in P-lim chemostat, doubling time 6 hrs, replicate 2"
        ),
        header_mean=(
            "Average of half-lives (hrs) in P-lim chemostat, doubling time 6 hrs"
        ),
        header_ci_1=(
            "Total half-life 95% confidence interval in P-lim chemostat, doubling time "
            "6 hrs, replicate 1"
        ),
        header_ci_2=(
            "Total half-life 95% confidence interval in P-lim chemostat, doubling time "
            "6 hrs, replicate 2"
        ),
    ),
    Condition(
        key="wt_plim_12h",
        short_name="P-lim12",
        limitation="P",
        reactor="chemostat",
        doubling_time="12 h",
        deleted_symbols=(),
        table1_proteins=2619,
        stored_proteins=2618,
        censored_keys=156,
        interval_keys=248,
        replicate_proteins=(2461, 2504),
        replicate_censored_keys=(89, 79),
        replicate_interval_keys=(2371, 2423),
        table1_quote=(
            "<td>6.</td><td>Wild type</td><td>P-lim</td><td>Chemostat</td><td>12h</td>"
            "<td>2</td><td>2619</td>"
        ),
        ceiling_hours=16.0,
        header_replicate_1=(
            "Half-life (hrs) in P-lim chemostat, doubling time 12 hrs, replicate 1"
        ),
        header_replicate_2=(
            "Half-life (hrs) in P-lim chemostat, doubling time 12 hrs, replicate 2"
        ),
        header_mean=(
            "Average of half-lives (hrs) in P-lim chemostat, doubling time 12 hrs"
        ),
        header_ci_1=(
            "Total half-life 95% confidence interval in P-lim chemostat, doubling time "
            "12 hrs, replicate 1"
        ),
        header_ci_2=(
            "Total half-life 95% confidence interval in P-lim chemostat, doubling time "
            "12 hrs, replicate 2"
        ),
    ),
    Condition(
        key="wt_nlim_6h",
        short_name="N-lim6",
        limitation="N",
        reactor="chemostat",
        doubling_time="6 h",
        deleted_symbols=(),
        table1_proteins=2467,
        stored_proteins=2466,
        censored_keys=44,
        interval_keys=395,
        replicate_proteins=(2177, 2347),
        replicate_censored_keys=(15, 30),
        replicate_interval_keys=(2161, 2317),
        table1_quote=(
            "<td>7.</td><td>Wild type</td><td>N-lim</td><td>Chemostat</td><td>6h</td>"
            "<td>2</td><td>2467</td>"
        ),
        ceiling_hours=8.0,
        header_replicate_1=(
            "Half-life (hrs) in N-lim chemostat, doubling time 6 hrs, replicate 1"
        ),
        header_replicate_2=(
            "Half-life (hrs) in N-lim chemostat, doubling time 6 hrs, replicate 2"
        ),
        header_mean=(
            "Average of half-lives (hrs) in N-lim chemostat, doubling time 6 hrs"
        ),
        header_ci_1=(
            "Total half-life 95% confidence interval in N-lim chemostat, doubling time "
            "6 hrs, replicate 1"
        ),
        header_ci_2=(
            "Total half-life 95% confidence interval in N-lim chemostat, doubling time "
            "6 hrs, replicate 2"
        ),
    ),
    Condition(
        key="wt_nlim_12h",
        short_name="N-lim12",
        limitation="N",
        reactor="chemostat",
        doubling_time="12 h",
        deleted_symbols=(),
        table1_proteins=2460,
        stored_proteins=2459,
        censored_keys=98,
        interval_keys=367,
        replicate_proteins=(2265, 2263),
        replicate_censored_keys=(46, 61),
        replicate_interval_keys=(2214, 2202),
        table1_quote=(
            "<td>8.</td><td>Wild type</td><td>N-lim</td><td>Chemostat</td><td>12 h</td>"
            "<td>2</td><td>2460</td>"
        ),
        ceiling_hours=16.0,
        header_replicate_1=(
            "Half-life (hrs) in N-lim chemostat, doubling time 12 hrs, replicate 1"
        ),
        header_replicate_2=(
            "Half-life (hrs) in N-lim chemostat, doubling time 12 hrs, replicate 2"
        ),
        header_mean=(
            "Average of half-lives (hrs) in N-lim chemostat, doubling time 12 hrs"
        ),
        header_ci_1=(
            "Total half-life 95% confidence interval in N-lim chemostat, doubling time "
            "12 hrs, replicate 1"
        ),
        header_ci_2=(
            "Total half-life 95% confidence interval in N-lim chemostat, doubling time "
            "12 hrs, replicate 2"
        ),
    ),
    Condition(
        key="clpp_nlim_6h",
        short_name="clpPN-lim6",
        limitation="N",
        reactor="chemostat",
        doubling_time="6 h",
        deleted_symbols=("clpP",),
        table1_proteins=2491,
        stored_proteins=2490,
        censored_keys=221,
        interval_keys=367,
        replicate_proteins=(2252, 2330),
        replicate_censored_keys=(27, 198),
        replicate_interval_keys=(2225, 2131),
        table1_quote=(
            "<td>11.</td><td>Δclp</td><td>N-lim</td><td>Chemostat</td><td>6h</td>"
            "<td>2</td><td>2491</td>"
        ),
        ceiling_hours=8.0,
        header_replicate_1=(
            "Half-life (hrs) in ∆clpP N-lim chemostat, doubling time 6 hrs, replicate 1"
        ),
        header_replicate_2=(
            "Half-life (hrs) in ∆clpP N-lim chemostat, doubling time 6 hrs, replicate 2"
        ),
        header_mean=(
            "Average of half-lives (hrs) in ∆clpP N-lim chemostat, doubling time 6 hrs"
        ),
        header_ci_1=(
            "Total half-life 95% confidence interval in ∆clpP N-lim chemostat, "
            "doubling time 6 hrs, replicate 1"
        ),
        header_ci_2=(
            "Total half-life 95% confidence interval in ∆clpP N-lim chemostat, "
            "doubling time 6 hrs, replicate 2"
        ),
    ),
    Condition(
        key="lon_nlim_6h",
        short_name="lonN-lim6",
        limitation="N",
        reactor="chemostat",
        doubling_time="6 h",
        deleted_symbols=("lon",),
        table1_proteins=2390,
        stored_proteins=2389,
        censored_keys=194,
        interval_keys=254,
        replicate_proteins=(2266, 2236),
        replicate_censored_keys=(141, 55),
        replicate_interval_keys=(2124, 2181),
        table1_quote=(
            "<td>10.</td><td>Δ lon</td><td>N-lim</td><td>Chemostat</td><td>6h</td>"
            "<td>2</td><td>2390</td>"
        ),
        ceiling_hours=8.0,
        header_replicate_1=(
            "Half-life (hrs) in ∆lon N-lim chemostat, doubling time 6 hrs, replicate 1"
        ),
        header_replicate_2=(
            "Half-life (hrs) in ∆lon N-lim chemostat, doubling time 6 hrs, replicate 2"
        ),
        header_mean=(
            "Average of half-lives (hrs) in ∆lon N-lim chemostat, doubling time 6 hrs"
        ),
        header_ci_1=(
            "Total half-life 95% confidence interval in ∆lon N-lim chemostat, "
            "doubling time 6 hrs, replicate 1"
        ),
        header_ci_2=(
            "Total half-life 95% confidence interval in ∆lon N-lim chemostat, "
            "doubling time 6 hrs, replicate 2"
        ),
    ),
    Condition(
        key="hslv_nlim_6h",
        short_name="hslVN-lim6",
        limitation="N",
        reactor="chemostat",
        doubling_time="6 h",
        deleted_symbols=("hslV",),
        table1_proteins=2393,
        stored_proteins=2392,
        censored_keys=164,
        interval_keys=315,
        replicate_proteins=(2263, 2183),
        replicate_censored_keys=(109, 59),
        replicate_interval_keys=(2153, 2123),
        table1_quote=(
            "<td>9.</td><td>Δ hslV</td><td>N-lim</td><td>Chemostat</td><td>6h</td>"
            "<td>2</td><td>2393</td>"
        ),
        ceiling_hours=8.0,
        header_replicate_1=(
            "Half-life (hrs) in ∆hslV N-lim chemostat, doubling time 6 hrs, replicate 1"
        ),
        header_replicate_2=(
            "Half-life (hrs) in ∆hslV N-lim chemostat, doubling time 6 hrs, replicate 2"
        ),
        header_mean=(
            "Average of half-lives (hrs) in ∆hslV N-lim chemostat, doubling time 6 hrs"
        ),
        header_ci_1=(
            "Total half-life 95% confidence interval in ∆hslV N-lim chemostat, "
            "doubling time 6 hrs, replicate 1"
        ),
        header_ci_2=(
            "Total half-life 95% confidence interval in ∆hslV N-lim chemostat, "
            "doubling time 6 hrs, replicate 2"
        ),
    ),
    Condition(
        key="clpp_lon_hslv_nlim_6h",
        short_name="TripleN-lim6",
        limitation="N",
        reactor="chemostat",
        doubling_time="6 h",
        deleted_symbols=("clpP", "lon", "hslV"),
        table1_proteins=2809,
        stored_proteins=2808,
        censored_keys=254,
        interval_keys=355,
        replicate_proteins=(2520, 2710),
        replicate_censored_keys=(49, 215),
        replicate_interval_keys=(2471, 2492),
        table1_quote=(
            "<td>12.</td><td>Δ hslV Δ lon Δ clpP</td><td>N-lim</td>"
            "<td>Chemostat</td><td>6h</td><td>2</td><td>2809</td>"
        ),
        ceiling_hours=8.0,
        header_replicate_1=(
            "Half-life (hrs) in ∆clpP∆lon∆hslV N-lim chemostat, "
            "doubling time 6 hrs, replicate 1"
        ),
        header_replicate_2=(
            "Half-life (hrs) in ∆clpP∆lon∆hslV N-lim chemostat, "
            "doubling time 6 hrs, replicate 2"
        ),
        header_mean=(
            "Average of half-lives (hrs) in ∆clpP∆lon∆hslV  N-lim "
            "chemostat, doubling time 6 hrs"
        ),
        header_ci_1=(
            "Total half-life 95% confidence interval in ∆clpP∆lon∆hslV "
            "N-lim chemostat, doubling time 6 hrs, replicate 1"
        ),
        header_ci_2=(
            "Total half-life 95% confidence interval in ∆clpP∆lon∆hslV "
            "N-lim chemostat, doubling time 6 hrs, replicate 2"
        ),
    ),
    Condition(
        key="smpb_nlim_6h",
        short_name="smpBN-lim6",
        limitation="N",
        reactor="chemostat",
        doubling_time="6 h",
        deleted_symbols=("smpB",),
        table1_proteins=2535,
        stored_proteins=2534,
        censored_keys=57,
        interval_keys=237,
        replicate_proteins=(2415, 2404),
        replicate_censored_keys=(37, 27),
        replicate_interval_keys=(2375, 2375),
        table1_quote=(
            "<td>13.</td><td>Δ smpB</td><td>N-lim</td><td>Chemostat</td><td>6h</td>"
            "<td>2</td><td>2535</td>"
        ),
        ceiling_hours=8.0,
        header_replicate_1=(
            "Half-life (hrs) in ∆smpB N-lim chemostat, doubling time 6 hrs, replicate 1"
        ),
        header_replicate_2=(
            "Half-life (hrs) in ∆smpB N-lim chemostat, doubling time 6 hrs, replicate 2"
        ),
        header_mean=(
            "Average of half-lives (hrs) in ∆smpB  N-lim chemostat, doubling time 6 hrs"
        ),
        header_ci_1=(
            "Total half-life 95% confidence interval in ∆smpB N-lim chemostat, "
            "doubling time 6 hrs, replicate 1"
        ),
        header_ci_2=(
            "Total half-life 95% confidence interval in ∆smpB N-lim chemostat, "
            "doubling time 6 hrs, replicate 2"
        ),
    ),
)

#: The condition whose phenotype is the reference every record is read against.
REFERENCE_CONDITION_KEY = "wt_nlim_6h"
#: Deleted symbols the paper releases, and whether the allele came from Keio.
KEIO_DERIVED_SYMBOLS = frozenset({"clpP", "lon", "hslV"})
KEIO_COLLECTION = "Keio collection"


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/guptaGlobalProteinTurnover2024``."""
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


def _deposits() -> tuple[tuple[str, str, str], ...]:
    """``(relpath, filename, sha256)`` of each file this loader's records come from."""
    return (
        (HALF_LIVES_REL, HALF_LIVES_FILENAME, HALF_LIVES_SHA256),
        (CONFIDENCE_REL, CONFIDENCE_FILENAME, CONFIDENCE_SHA256),
    )


def deposit_raw_mirror(
    *,
    half_lives_path: str | Path,
    confidence_path: str | Path,
    retrieved_at: str = RAW_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror (Supplementary Data 1 and 7) and its ``manifest.json``.

    Idempotent by sha256: an existing mirror file whose digest matches is left alone and
    a differing one raises rather than being overwritten. Both files are verified BEFORE
    anything is written, so a refusal leaves no partial deposit. Both come from the PMC
    Article Datasets bucket, which is directly scriptable, so the recorded retrieval
    re-runs as-is and reproduced both pinned digests on ``retrieved_at``.
    """
    root = raw_mirror_dir(data_root)
    sources = {HALF_LIVES_REL: half_lives_path, CONFIDENCE_REL: confidence_path}
    for relpath, _, expected in _deposits():
        actual = _sha256(sources[relpath])
        if actual != expected:
            raise RuntimeError(
                f"{sources[relpath]} hashes to {actual}, not the pinned {expected}"
            )
    files: list[ArtifactRecord] = []
    for relpath, filename, expected in _deposits():
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != expected:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(sources[relpath], dest)
        key = pmc_cloud_key(filename)
        url = pmc_cloud_url(key)
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=expected,
                source=url,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.pmc_cloud,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.pmc_cloud_object",
                    params={"key": key},
                    sha256=expected,
                    retrieved_at=retrieved_at,
                ),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=files,
        si_data_sources=[
            pmc_cloud_url(pmc_cloud_key(HALF_LIVES_FILENAME)),
            pmc_cloud_url(pmc_cloud_key(CONFIDENCE_FILENAME)),
            f"https://www.ebi.ac.uk/pride/archive/projects/{PRIDE_ACCESSION}",
            f"https://doi.org/{CODE_DOI}",
        ],
        si_expected=[
            "Supplementary Data 1 (MOESM4) -- deposited; the 'Average of half-lives "
            "(hrs)' column of each of the 13 conditions is what this loader consumes, "
            "with the two replicate columns beside it for the standard error",
            "Supplementary Data 7 (MOESM10) -- deposited; the per-replicate 95% "
            "confidence intervals, consumed as the oracle that an 'Undetermined' "
            "interval is exactly a ceiling-flagged half-life",
            "Supplementary Data 2-6 and 8 (MOESM5-9, MOESM11) -- protease-substrate "
            "assignments, rapidly degrading proteins, N-terminal residues, relative "
            "and absolute protein levels, and the model-variable glossary. NOT "
            "deposited: no loader reads them, and each is an analysis derived from "
            "Supplementary Data 1 rather than an independent measurement",
            "Supplementary Information (MOESM1), the Supplementary Note (MOESM2) and "
            "the Supplementary Data descriptions (MOESM3) are quoted from the "
            "torchcell-library OCR mirror, not the raw mirror",
            f"PRIDE {PRIDE_ACCESSION} -- the raw mass-spectrometry files behind the "
            "fit. NOT deposited: no loader consumes raw spectra, and the released "
            "half-life matrix is the fitted output",
            f"the analysis code at https://doi.org/{CODE_DOI} (github.com/wuhrlab/"
            "ProteinTurnoverEcoli). NOT deposited: this loader reads the released "
            "numbers and refits nothing",
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
# Reading the two workbooks
# --------------------------------------------------------------------------- #
#: Row 4 of each sheet is the long header; row 5 is the short name; data starts at 6.
HEADER_ROW = 4


class HalfLifeCell(BaseModel):
    """One released half-life cell: the value in hours and its ceiling flag."""

    model_config = ConfigDict(frozen=True)

    hours: float
    ceiling: bool


def _sheet(
    path: str | Path, sheet: str
) -> tuple[list[str | None], list[tuple[Any, ...]]]:
    """The long header row and the data rows of one supplementary sheet."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(workbook[sheet].iter_rows(min_row=HEADER_ROW, values_only=True))
    workbook.close()
    header = [None if cell is None else str(cell) for cell in rows[0]]
    data = [row for row in rows[2:] if row[1] is not None]
    return header, data


def _column(header: Sequence[str | None], name: str, path: str) -> int:
    """The index of the one column whose header is exactly ``name``."""
    hits = [index for index, cell in enumerate(header) if cell == name]
    if len(hits) != 1:
        raise RuntimeError(
            f"{path}: {len(hits)} columns are headed {name!r}; the workbook's shape "
            "changed, so no condition can be read from it"
        )
    return hits[0]


def parse_half_life(value: Any) -> HalfLifeCell | None:
    """Parse one half-life cell; ``None`` for a protein not quantified in that column.

    A trailing ``*`` is the workbook's ceiling flag, kept as a flag rather than
    stripped silently.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    ceiling = text.endswith("*")
    return HalfLifeCell(hours=float(text[:-1] if ceiling else text), ceiling=ceiling)


def read_half_lives(
    path: str,
) -> tuple[list[str], list[str], dict[str, list[HalfLifeCell | None]]]:
    """Read Supplementary Data 1: protein ids, gene names, and per-condition cells.

    The returned mapping is keyed by ``Condition.key`` and holds, for each released row
    and in row order, the two replicate cells then the authors' mean cell.
    """
    header, data = _sheet(path, HALF_LIVES_SHEET)
    protein_ids = [str(row[1]) for row in data]
    gene_names = ["" if row[2] is None else str(row[2]).strip() for row in data]
    cells: dict[str, list[HalfLifeCell | None]] = {}
    for condition in CONDITIONS:
        indices = [
            _column(header, name, path)
            for name in (
                condition.header_replicate_1,
                condition.header_replicate_2,
                condition.header_mean,
            )
        ]
        cells[condition.key] = [
            parse_half_life(row[index]) for row in data for index in indices
        ]
    return protein_ids, gene_names, cells


class ConfidenceCell(BaseModel):
    """One released Supplementary Data 7 cell: a half-life interval, or ``Undetermined``.

    The workbook writes a determined cell as ``[lower upper]`` in hours, on the TOTAL
    HALF-LIFE scale its column header names, and an undetermined one as the literal
    ``Undetermined``, which is the censoring flag Supplementary Data 1 writes as ``*``.
    """

    model_config = ConfigDict(frozen=True)

    undetermined: bool
    lower_hours: float | None = None
    upper_hours: float | None = None

    @property
    def bounds_hours(self) -> tuple[float, float]:
        """The two released endpoints; only a determined cell has them."""
        if self.lower_hours is None or self.upper_hours is None:
            raise RuntimeError(
                "an Undetermined confidence cell has no endpoints; it is the ceiling "
                "flag, not an interval"
            )
        return self.lower_hours, self.upper_hours


def parse_confidence_interval(value: Any) -> ConfidenceCell | None:
    """Parse one Supplementary Data 7 cell; ``None`` where the protein is not quantified.

    Any other shape raises: the two endpoints are what the record's bounds are built
    from, so an unparsed cell is never skipped.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text == UNDETERMINED_NOTE.value:
        return ConfidenceCell(undetermined=True)
    if not (text.startswith("[") and text.endswith("]")):
        raise RuntimeError(
            f"{text!r} is neither {UNDETERMINED_NOTE.value!r} nor a '[lower upper]' "
            "interval; Supplementary Data 7's cell shape changed"
        )
    parts = text[1:-1].split()
    if len(parts) != 2:
        raise RuntimeError(
            f"{text!r} has {len(parts)} endpoint(s), not the two a 95% confidence "
            "interval has"
        )
    return ConfidenceCell(
        undetermined=False, lower_hours=float(parts[0]), upper_hours=float(parts[1])
    )


def read_confidence_intervals(
    path: str,
) -> tuple[list[str], dict[str, list[ConfidenceCell | None]]]:
    """Read Supplementary Data 7: the per-replicate 95% intervals on the half-life.

    Returns the protein ids and, per ``Condition.key``, the two replicates' cells in row
    order. An ``Undetermined`` cell is the censoring flag; a determined one carries the
    two released endpoints, in hours.
    """
    header, data = _sheet(path, CONFIDENCE_SHEET)
    protein_ids = [str(row[1]) for row in data]
    cells: dict[str, list[ConfidenceCell | None]] = {}
    for condition in CONDITIONS:
        indices = [
            _column(header, name, path)
            for name in (condition.header_ci_1, condition.header_ci_2)
        ]
        cells[condition.key] = [
            parse_confidence_interval(row[index]) for row in data for index in indices
        ]
    return protein_ids, cells


# --------------------------------------------------------------------------- #
# Identifier route: UniProt db_xref of the pinned assembly, then the gene symbol
# --------------------------------------------------------------------------- #
#: The two ``DerivedIdentifierRoute`` members this release's protein keys reach a locus
#: tag through, named once so the counts, the CSV and the typed mapping cannot drift.
UNIPROT_ROUTE: DerivedIdentifierRoute = "uniprot_db_xref"
SYMBOL_ROUTE: DerivedIdentifierRoute = "gene_symbol"


class IdentifierRoute(BaseModel):
    """How the released protein keys reached locus tags, with every count measured.

    ``rows_per_route`` is keyed on ``DerivedIdentifierRoute`` members, so the ledger
    names the typed route rather than a free string.
    """

    source_rows: int
    unique_accessions: int
    assembly_uniprot_xrefs: int
    resolved_by_uniprot_xref: int
    resolved_by_gene_name: int
    rows_per_route: dict[DerivedIdentifierRoute, int]
    unresolved: tuple[str, ...]
    disagreements: tuple[tuple[str, str, str, str], ...] = ()

    @property
    def resolved(self) -> int:
        """Rows that reached a locus tag through either layer."""
        return self.resolved_by_uniprot_xref + self.resolved_by_gene_name


def base_accession(protein_id: str) -> str:
    """The canonical accession of a released ``sp|ACC|ENTRY`` key, isoform suffix cut."""
    parts = protein_id.split("|")
    if len(parts) != 3:
        raise RuntimeError(
            f"{protein_id!r} is not a 'db|accession|entry' key; the release's protein "
            "column changed shape"
        )
    return parts[1].split("-")[0]


def uniprot_locus_tags(genome: EcoliK12Genome) -> dict[str, str]:
    """UniProt accession -> locus tag, from the pinned assembly's own ``/db_xref``.

    An accession carried by two loci is excluded, so the map is one-to-one by
    construction (measured on the pinned MG1655 assembly: none are).
    """
    claimed: dict[str, set[str]] = {}
    table = genome.locus_table
    for locus_tag, xrefs in zip(table["locus_tag"], table["db_xrefs"], strict=True):
        for xref in xrefs:
            for prefix in _UNIPROT_XREF_PREFIXES:
                if xref.startswith(prefix):
                    claimed.setdefault(xref[len(prefix) :], set()).add(str(locus_tag))
    return {
        accession: next(iter(tags))
        for accession, tags in claimed.items()
        if len(tags) == 1
    }


def _symbol_locus(genome: EcoliK12Genome, name: str) -> str | None:
    """The one locus a released gene name resolves to, or ``None``."""
    if not name:
        return None
    resolution = genome.resolve_gene_name(name)
    if resolution.systematic_name is None:
        return None
    if resolution.status not in (
        GeneNameStatus.CURRENT,
        GeneNameStatus.RENAMED,
        GeneNameStatus.NON_GENE_FEATURE,
    ):
        return None
    return str(resolution.systematic_name)


def identifier_names(
    genome: EcoliK12Genome, protein_ids: Sequence[str], gene_names: Sequence[str]
) -> tuple[list[str], list[DerivedIdentifierMapping | None], IdentifierRoute]:
    """The name handed to ``reconcile_locus_tags`` per released row, typed route and all.

    Layer 1 is the pinned assembly's own UniProt ``/db_xref``, which is
    ``DerivedIdentifierRoute`` member ``uniprot_db_xref``; layer 2 is the released gene
    name, ``gene_symbol``. Each resolved row gets a ``DerivedIdentifierMapping``, so the
    released key is validated as a UniProt accession (through
    ``DerivedIdentifierMapping.uniprot_accession``) at read time rather than trusted. A
    row neither layer resolves keeps its released protein id, which
    ``reconcile_locus_tags`` reports as outside the namespace, and carries no mapping.
    """
    xrefs = uniprot_locus_tags(genome)
    names: list[str] = []
    mappings: list[DerivedIdentifierMapping | None] = []
    by_xref = by_symbol = 0
    unresolved: list[str] = []
    disagreements: list[tuple[str, str, str, str]] = []
    for protein_id, gene_name in zip(protein_ids, gene_names, strict=True):
        from_xref = xrefs.get(base_accession(protein_id))
        from_symbol = _symbol_locus(genome, gene_name)
        if (
            from_xref is not None
            and from_symbol is not None
            and from_xref != from_symbol
        ):
            disagreements.append((protein_id, gene_name, from_xref, from_symbol))
        if from_xref is not None:
            names.append(from_xref)
            mappings.append(
                DerivedIdentifierMapping(
                    source_identifier=protein_id, route=UNIPROT_ROUTE
                )
            )
            by_xref += 1
        elif from_symbol is not None:
            names.append(from_symbol)
            mappings.append(
                DerivedIdentifierMapping(
                    source_identifier=gene_name, route=SYMBOL_ROUTE
                )
            )
            by_symbol += 1
        else:
            names.append(protein_id)
            mappings.append(None)
            unresolved.append(protein_id)
    route = IdentifierRoute(
        source_rows=len(names),
        unique_accessions=len({base_accession(p) for p in protein_ids}),
        assembly_uniprot_xrefs=len(xrefs),
        resolved_by_uniprot_xref=by_xref,
        resolved_by_gene_name=by_symbol,
        rows_per_route={UNIPROT_ROUTE: by_xref, SYMBOL_ROUTE: by_symbol},
        unresolved=tuple(unresolved),
        disagreements=tuple(disagreements),
    )
    log.info(
        "Gupta2024 identifier route: %d rows; %d via the assembly's UniProt db_xref "
        "(%d accessions carried), %d via the released gene name, %d unresolved %s; %d "
        "row(s) where the two layers disagree %s",
        route.source_rows,
        route.resolved_by_uniprot_xref,
        route.assembly_uniprot_xrefs,
        route.resolved_by_gene_name,
        len(route.unresolved),
        list(route.unresolved),
        len(route.disagreements),
        list(route.disagreements),
    )
    return names, mappings, route


# --------------------------------------------------------------------------- #
# Genotype, environment and the phenotype
# --------------------------------------------------------------------------- #
def ncm3722_background() -> StrainBackground:
    """NCM3722 as an edit of the pinned MG1655 assembly, with its two typed gaps."""
    return StrainBackground(
        name=MEASURED_STRAIN,
        reference_strain=MG1655_STRAIN,
        assembly_set=MG1655_ASSEMBLY_SET,
        provenance=[STRAIN],
        provenance_gaps=[NCM3722_GENOTYPE_GAP, NCM3722_CONSTRUCTION_GAP],
    )


def strain_reference(data_root: str | None = None) -> ReferenceGenome:
    """The assembly-pinned reference every record of this paper is written against."""
    return assembly_reference(
        MG1655_STRAIN, background=ncm3722_background(), data_root=data_root
    )


def deletion_perturbation(symbol: str, locus_tag: str) -> DeletionStrainPerturbation:
    """One released protease / SsrA deletion, keyed on the pinned assembly's b-number.

    The route is ``gene_symbol`` and stays so: the paper releases its knockouts only as
    symbols ("The ΔclpP, Δlon, ΔhslV single mutants were generated by P1 transduction",
    "we knocked out the smpB gene"), never as UniProt accessions, and each of the four
    resolves through the genome's name layers. The ``uniprot_db_xref`` route belongs to
    the per-protein LABEL keys, which the release does write as accessions.
    """
    keio = symbol in KEIO_DERIVED_SYMBOLS
    return DeletionStrainPerturbation(
        systematic_gene_name=locus_tag,
        perturbed_gene_name=symbol,
        gene_namespace=MG1655_NAMESPACE,
        identifier_mapping=DerivedIdentifierMapping(
            source_identifier=symbol, route=SYMBOL_ROUTE
        ),
        collection=KEIO_COLLECTION if keio else None,
        construction=None,
    )


def _overridden(
    name: str, value: float, unit: ConcentrationUnit, provenance: list[SourcedValue]
) -> MediaComponent:
    """One MOPS component at the amount this paper states for it."""
    return MediaComponent(
        compound=resolved_compound(name),
        role=(
            MediaComponentRole.nitrogen_source
            if name == AMMONIUM_COMPONENT
            else MediaComponentRole.bulk_salt
        ),
        concentration=Concentration(value=value, unit=unit),
        provenance=provenance,
    )


def medium(limitation: Limitation) -> Media:
    """The MOPS medium of one limitation, as this paper's Methods state it.

    ``MOPS_MINIMAL`` carries the Neidhardt base with no carbon source, 9.5 mM ammonium
    chloride and 1.32 mM dipotassium hydrogen phosphate, which are exactly this paper's
    non-limiting amounts. Each medium is that base with the stated glucose added and,
    for a limitation, the limiting component replaced at the reduced amount.
    """
    glucose_percent = 0.08 if limitation == "C" else 0.4
    glucose_provenance = (
        [REDUCED_C_AND_N, REDUCED_VALUES] if limitation == "C" else [FULL_LEVELS]
    )
    components: list[MediaComponent] = []
    for component in MOPS_MINIMAL.components:
        name = component.compound.name
        if name == AMMONIUM_COMPONENT and limitation == "N":
            components.append(
                _overridden(
                    name,
                    1.9,
                    ConcentrationUnit.millimolar,
                    [REDUCED_C_AND_N, REDUCED_VALUES],
                )
            )
        elif name == PHOSPHATE_COMPONENT and limitation == "P":
            components.append(
                _overridden(
                    name,
                    0.132,
                    ConcentrationUnit.millimolar,
                    [REDUCED_P, REDUCED_P_VALUE],
                )
            )
        else:
            components.append(component)
    components.append(
        MediaComponent(
            compound=resolved_compound("glucose"),
            role=MediaComponentRole.carbon_source,
            concentration=Concentration(
                value=glucose_percent, unit=ConcentrationUnit.percent_w_v
            ),
            provenance=glucose_provenance,
        )
    )
    names = {component.compound.name for component in components}
    for required in (AMMONIUM_COMPONENT, PHOSPHATE_COMPONENT, GLUCOSE_COMPONENT):
        if required not in names:
            raise RuntimeError(
                f"{required!r} is not a component of the built medium; MOPS_MINIMAL's "
                "component names changed, so the limitation override cannot be applied"
            )
    label = {
        "none": "no limitation",
        "C": "carbon-limiting",
        "N": "nitrogen-limiting",
        "P": "phosphorus-limiting",
    }[limitation]
    return Media(
        name=f"40 mM MOPS glucose minimal medium, {label} (Gupta 2024)",
        state="liquid",
        is_synthetic=True,
        base_medium="MOPS_MINIMAL",
        components=components,
        provenance=[BASE_MEDIUM, FULL_LEVELS]
        + ([REDUCED_C_AND_N, REDUCED_VALUES] if limitation in ("C", "N") else [])
        + ([REDUCED_P, REDUCED_P_VALUE] if limitation == "P" else []),
    )


def environment(condition: Condition) -> Environment:
    """The growth environment of one released condition.

    ``duration_hours`` is the labeling window: the last sampling time of this doubling
    time's Table 2 series. ``dilution_rate_per_hour`` is the chemostat's controlled
    variable, ``ln 2`` over the doubling time, and ``None`` for the batch condition; it
    is what the stored rate's dilution term IS, so it is typed on the environment rather
    than spelled into the phenotype's ``measurement_type``.
    """
    perturbations: list[Any] = []
    if condition.reactor == "chemostat":
        perturbations.append(
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.ph,
                magnitude=Concentration(
                    value=float(CHEMOSTAT_PH.value), unit=ConcentrationUnit.ph
                ),
            )
        )
    temperature = (
        TEMPERATURE_CHEMOSTAT_C
        if condition.reactor == "chemostat"
        else TEMPERATURE_BATCH_C
    )
    return Environment(
        media=medium(condition.limitation),
        temperature=Temperature(value=float(temperature.value)),
        perturbations=perturbations,
        aerobicity="aerobic",
        duration_hours=condition.duration_hours,
        dilution_rate_per_hour=condition.dilution_rate_per_hour,
    )


class RefusedInterval(BaseModel):
    """One released interval that has no finite non-negative rate bound.

    The released upper endpoint on the half-life is zero or negative, which is the rate
    interval's lower bound having crossed zero. Inverting it gives a negative rate bound
    and clamping it at zero would manufacture a bound the fit did not give, so the key
    carries no interval and is listed here instead.
    """

    model_config = ConfigDict(frozen=True)

    condition: str
    locus_tag: str
    half_life_hours: float
    released_lower_hours: float
    released_upper_hours: float
    reason: str
    #: The replicate record the refused cell belongs to; ``None`` for the aggregate.
    replicate_id: str | None = None


class ConditionValues(BaseModel):
    """One condition's per-protein label, derived from the released cells.

    ``censoring`` is keyed on every stored key, because the source's oracle resolves
    every cell; ``degradation_rate_lower`` / ``degradation_rate_upper`` are ragged, and
    hold exactly the one-replicate keys whose released interval inverts onto the rate
    scale.
    """

    half_life: dict[str, float]
    degradation_rate: dict[str, float]
    degradation_rate_se: dict[str, float]
    degradation_rate_lower: dict[str, float]
    degradation_rate_upper: dict[str, float]
    censoring: dict[str, Censoring]
    n_replicates: dict[str, int]
    ceiling_cells: tuple[tuple[str, int], ...]
    refused_intervals: tuple[RefusedInterval, ...]
    two_replicate_keys: int


def total_turnover_rate(half_life_hours: float) -> float:
    """``ln 2 / T_half``: the fitted total turnover rate behind a released half-life."""
    if half_life_hours <= 0.0:
        raise RuntimeError(
            f"a released half-life of {half_life_hours} h has no turnover rate; the "
            "release's 61,811 cells are all strictly positive"
        )
    return math.log(2.0) / half_life_hours


def rate_bounds(lower_hours: float, upper_hours: float) -> tuple[float, float]:
    """Invert a released half-life interval onto the rate scale.

    ``r = ln 2 / T`` is strictly decreasing, so the endpoints swap: the lower rate bound
    comes from the UPPER half-life endpoint. Exact arithmetic through the Supplementary
    Note's own identity, the same one ``degradation_rate`` is built with.
    """
    if lower_hours <= 0.0 or upper_hours <= 0.0:
        raise RuntimeError(
            f"the released interval [{lower_hours} {upper_hours}] has a non-positive "
            "endpoint, so it has no finite non-negative rate bound"
        )
    return math.log(2.0) / upper_hours, math.log(2.0) / lower_hours


def condition_values(
    condition: Condition,
    keys: Sequence[str],
    cells: Sequence[HalfLifeCell | None],
    intervals: Sequence[ConfidenceCell | None],
) -> ConditionValues:
    """Build one condition's label maps from its five columns, row by row.

    ``cells`` is the flat ``(replicate 1, replicate 2, mean)`` triple per row that
    :func:`read_half_lives` returns, and ``intervals`` the flat ``(replicate 1,
    replicate 2)`` pair :func:`read_confidence_intervals` returns. A row whose mean cell
    is empty is not a key of this condition: the assay did not quantify that protein
    here, and a missing value is an absent key, never a zero.

    A key is ``Censoring.right`` when any of its replicate cells is ceiling-flagged, and
    ``Censoring.uncensored`` otherwise: the released mean then has a capped value in it,
    so the true mean half-life lies above the stored one. A ONE-REPLICATE key's stored
    value is that replicate's own fitted half-life, so its released interval is the
    record's interval and is stored, inverted onto the rate scale; a two-replicate key
    gets none, because the interval of the mean of two fits is not published.
    """
    half_life: dict[str, float] = {}
    rate: dict[str, float] = {}
    rate_se: dict[str, float] = {}
    rate_lower: dict[str, float] = {}
    rate_upper: dict[str, float] = {}
    censoring: dict[str, Censoring] = {}
    n_replicates: dict[str, int] = {}
    ceiling: list[tuple[str, int]] = []
    refused: list[RefusedInterval] = []
    two_replicate = 0
    for row, key in enumerate(keys):
        replicate_1, replicate_2, mean = cells[3 * row : 3 * row + 3]
        replicates = [cell for cell in (replicate_1, replicate_2) if cell is not None]
        for number, cell in enumerate((replicate_1, replicate_2), start=1):
            if cell is not None and cell.ceiling:
                ceiling.append((key, number))
        if mean is None:
            if replicates:
                raise RuntimeError(
                    f"{condition.key}/{key}: {len(replicates)} replicate value(s) with "
                    "no mean cell; the released mean column is the consumed column"
                )
            continue
        if not replicates:
            raise RuntimeError(
                f"{condition.key}/{key}: a mean cell with no replicate value"
            )
        expected = statistics.fmean(cell.hours for cell in replicates)
        if abs(expected - mean.hours) > 1e-6 * max(1.0, abs(mean.hours)):
            raise RuntimeError(
                f"{condition.key}/{key}: the released mean {mean.hours} is not the "
                f"arithmetic mean {expected} of its replicates"
            )
        half_life[key] = mean.hours
        rate[key] = total_turnover_rate(mean.hours)
        n_replicates[key] = len(replicates)
        if len(replicates) == 2:
            two_replicate += 1
        censored = any(cell.ceiling for cell in replicates)
        censoring[key] = Censoring.right if censored else Censoring.uncensored
        if len(replicates) == 2 and not censored:
            rates = [total_turnover_rate(cell.hours) for cell in replicates]
            rate_se[key] = statistics.stdev(rates) / math.sqrt(2.0)
        else:
            rate_se[key] = float("nan")
        if len(replicates) == 2:
            continue
        number = 0 if replicate_1 is not None else 1
        interval = intervals[2 * row + number]
        if censored:
            if interval is None or not interval.undetermined:
                raise RuntimeError(
                    f"{condition.key}/{key}: replicate {number + 1} is ceiling-flagged "
                    f"but its interval cell is {interval!r}, not "
                    f"{UNDETERMINED_NOTE.value!r}; the two workbooks disagree"
                )
            continue
        if interval is None or interval.undetermined:
            raise RuntimeError(
                f"{condition.key}/{key}: replicate {number + 1} is an estimate but "
                f"its interval cell is {interval!r}; the release publishes an interval "
                "for every determined fit"
            )
        released_lower, released_upper = interval.bounds_hours
        if released_lower <= 0.0 or released_upper <= 0.0:
            refused.append(
                RefusedInterval(
                    condition=condition.key,
                    locus_tag=key,
                    half_life_hours=mean.hours,
                    released_lower_hours=released_lower,
                    released_upper_hours=released_upper,
                    reason="non_positive_released_half_life_endpoint",
                )
            )
            continue
        lower, upper = rate_bounds(released_lower, released_upper)
        if not lower <= rate[key] <= upper:
            raise RuntimeError(
                f"{condition.key}/{key}: the released interval "
                f"[{released_lower} {released_upper}] h inverts to "
                f"[{lower} {upper}] per hour, which does not bracket the stored rate "
                f"{rate[key]}"
            )
        rate_lower[key] = lower
        rate_upper[key] = upper
    return ConditionValues(
        half_life=half_life,
        degradation_rate=rate,
        degradation_rate_se=rate_se,
        degradation_rate_lower=rate_lower,
        degradation_rate_upper=rate_upper,
        censoring=censoring,
        n_replicates=n_replicates,
        ceiling_cells=tuple(ceiling),
        refused_intervals=tuple(refused),
        two_replicate_keys=two_replicate,
    )


class ReplicateValues(BaseModel):
    """One replicate column's per-protein label: one biological replicate's own fits.

    Every map is keyed on the proteins this replicate quantified. ``censoring`` is
    ``right`` exactly for this replicate's ceiling-flagged cells; the bound maps hold
    every determined cell whose released interval inverts onto the rate scale.
    """

    replicate_id: str
    half_life: dict[str, float]
    degradation_rate: dict[str, float]
    degradation_rate_lower: dict[str, float]
    degradation_rate_upper: dict[str, float]
    censoring: dict[str, Censoring]
    refused_intervals: tuple[RefusedInterval, ...]

    def drop(self, keys: set[str]) -> None:
        """Remove the named protein keys from every map (the isoform rows)."""
        for key in keys:
            self.half_life.pop(key, None)
            self.degradation_rate.pop(key, None)
            self.degradation_rate_lower.pop(key, None)
            self.degradation_rate_upper.pop(key, None)
            self.censoring.pop(key, None)
        self.refused_intervals = tuple(
            refused
            for refused in self.refused_intervals
            if refused.locus_tag not in keys
        )


def replicate_values(
    condition: Condition,
    keys: Sequence[str],
    cells: Sequence[HalfLifeCell | None],
    intervals: Sequence[ConfidenceCell | None],
    replicate_id: str,
) -> ReplicateValues:
    """Build one replicate RECORD's label maps from its half-life and interval columns.

    ``cells`` and ``intervals`` are the flat per-row sequences :func:`read_half_lives`
    and :func:`read_confidence_intervals` return; ``replicate_id`` (``"1"`` or ``"2"``)
    picks the column. A row whose replicate cell is empty is not a key of this record:
    the assay did not quantify that protein in this culture. The value is the released
    cell verbatim, and the released interval of the SAME cell is this record's interval,
    because the record's value is that one fit. A ceiling-flagged cell is ``right``
    censored and must carry ``Undetermined``; a determined cell must carry an interval,
    which is stored unless an endpoint is non-positive (then refused, never clamped).
    """
    number = REPLICATE_IDS.index(replicate_id)
    half_life: dict[str, float] = {}
    rate: dict[str, float] = {}
    rate_lower: dict[str, float] = {}
    rate_upper: dict[str, float] = {}
    censoring: dict[str, Censoring] = {}
    refused: list[RefusedInterval] = []
    for row, key in enumerate(keys):
        cell = cells[3 * row + number]
        if cell is None:
            continue
        interval = intervals[2 * row + number]
        half_life[key] = cell.hours
        rate[key] = total_turnover_rate(cell.hours)
        if cell.ceiling:
            if interval is None or not interval.undetermined:
                raise RuntimeError(
                    f"{condition.key}/{key}: replicate {replicate_id} is "
                    f"ceiling-flagged but its interval cell is {interval!r}, not "
                    f"{UNDETERMINED_NOTE.value!r}; the two workbooks disagree"
                )
            censoring[key] = Censoring.right
            continue
        censoring[key] = Censoring.uncensored
        if interval is None or interval.undetermined:
            raise RuntimeError(
                f"{condition.key}/{key}: replicate {replicate_id} is an estimate but "
                f"its interval cell is {interval!r}; the release publishes an interval "
                "for every determined fit"
            )
        released_lower, released_upper = interval.bounds_hours
        if released_lower <= 0.0 or released_upper <= 0.0:
            refused.append(
                RefusedInterval(
                    condition=condition.key,
                    locus_tag=key,
                    half_life_hours=cell.hours,
                    released_lower_hours=released_lower,
                    released_upper_hours=released_upper,
                    reason="non_positive_released_half_life_endpoint",
                    replicate_id=replicate_id,
                )
            )
            continue
        lower, upper = rate_bounds(released_lower, released_upper)
        if not lower <= rate[key] <= upper:
            raise RuntimeError(
                f"{condition.key}/{key} replicate {replicate_id}: the released interval "
                f"[{released_lower} {released_upper}] h inverts to [{lower} {upper}] "
                f"per hour, which does not bracket the stored rate {rate[key]}"
            )
        rate_lower[key] = lower
        rate_upper[key] = upper
    return ReplicateValues(
        replicate_id=replicate_id,
        half_life=half_life,
        degradation_rate=rate,
        degradation_rate_lower=rate_lower,
        degradation_rate_upper=rate_upper,
        censoring=censoring,
        refused_intervals=tuple(refused),
    )


def replicate_phenotype(values: ReplicateValues) -> ProteinTurnoverPhenotype:
    """The protein-turnover phenotype of ONE biological replicate of one condition.

    ``n_replicates`` is 1 for every key and ``degradation_rate_se`` is ``None`` with a
    ``not_applicable_single_measurement`` gap: a single fit has no dispersion over
    replicates. The released per-cell interval is the record's own and is stored.
    """
    return ProteinTurnoverPhenotype(
        degradation_rate=values.degradation_rate,
        degradation_rate_se=None,
        degradation_rate_lower=values.degradation_rate_lower,
        degradation_rate_upper=values.degradation_rate_upper,
        confidence_level=CONFIDENCE_LEVEL,
        interval_method=INTERVAL_METHOD,
        censoring=values.censoring,
        half_life=values.half_life,
        n_replicates=dict.fromkeys(values.degradation_rate, 1),
        measurement_type=MEASUREMENT_TYPE,
        replicate_id=values.replicate_id,
        provenance_gaps=[
            SYNTHESIS_RATE_GAP,
            ProvenanceGap(
                field="degradation_rate_se",
                reason=ProvenanceGapReason.not_applicable_single_measurement,
                note=SINGLE_REPLICATE_SE_NOTE,
            ),
        ],
    )


def phenotype(
    condition: Condition, values: ConditionValues
) -> ProteinTurnoverPhenotype:
    """The AGGREGATE protein-turnover phenotype of one condition: the reference.

    Built from the authors' released mean over the two replicates, with
    ``replicate_id=None``; since the per-replicate conversion only the
    ``phenotype_reference`` is built this way. ``censoring`` is stated for every stored key; the bound maps are ragged and hold the
    one-replicate keys whose published interval is the record's interval.
    ``measurement_type`` names the assay and its unit only, because the dilution regime
    is the environment's ``dilution_rate_per_hour``.
    """
    return ProteinTurnoverPhenotype(
        degradation_rate=values.degradation_rate,
        degradation_rate_se=values.degradation_rate_se,
        degradation_rate_lower=values.degradation_rate_lower,
        degradation_rate_upper=values.degradation_rate_upper,
        confidence_level=CONFIDENCE_LEVEL,
        interval_method=INTERVAL_METHOD,
        censoring=values.censoring,
        half_life=values.half_life,
        n_replicates=values.n_replicates,
        measurement_type=MEASUREMENT_TYPE,
        provenance_gaps=[SYNTHESIS_RATE_GAP],
    )


def publication() -> Publication:
    """This paper, by DOI; the mirror's manifest records no PubMed id."""
    return Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")


# --------------------------------------------------------------------------- #
# Build accounting
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One reason a released protein key is not a key of any record."""

    rule: str
    scope: str
    description: str
    items: tuple[str, ...]


class BuildAccounting(BaseModel):
    """What the build read, what it kept, and the arithmetic that ties the two.

    The per-record maps are keyed ``<condition>/<replicate_id>``. ``censored_keys``,
    ``uncensored_keys``, ``interval_keys`` and ``refused_intervals`` count over the 26
    replicate records; ``per_condition_keys`` is the union of a condition's two records
    and ``one_replicate_keys`` the condition-level proteins only one replicate
    quantified. ``mean_reproduction_max_abs_diff_hours`` is the largest gap between the
    mean of a key's replicate records and the authors' released mean column.
    """

    dataset: str
    source_rows: int
    released_cells: int
    ceiling_cells: int
    censored_keys: int
    uncensored_keys: int
    interval_keys: int
    one_replicate_keys: int
    refused_intervals: list[RefusedInterval]
    confidence_level: float
    interval_method: str
    measurement_type: str
    dilution_rate_per_hour: dict[str, float | None]
    candidate_records: int
    kept_records: int
    dropped_records: int
    kept_protein_keys: int
    dropped_protein_keys: int
    per_condition_keys: dict[str, int]
    per_record_keys: dict[str, int]
    per_record_censored_keys: dict[str, int]
    per_record_interval_keys: dict[str, int]
    mean_reproduction_max_abs_diff_hours: float
    rules: list[DropRule]
    identifier_route: IdentifierRoute
    reconciliation: LocusTagReconciliation
    notes: list[str]

    def check(self) -> None:
        """The ledger must balance, or the build is not what the loader says it is."""
        if self.kept_records + self.dropped_records != self.candidate_records:
            raise RuntimeError(
                f"{self.kept_records} kept + {self.dropped_records} dropped != "
                f"{self.candidate_records} candidates"
            )
        if self.kept_protein_keys + self.dropped_protein_keys != self.source_rows:
            raise RuntimeError(
                f"{self.kept_protein_keys} kept + {self.dropped_protein_keys} dropped "
                f"keys != {self.source_rows} released rows"
            )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class ProteinTurnoverGupta2024Dataset(ExperimentDataset):
    """Gupta 2024 per-protein total turnover: 13 NCM3722 conditions x 2 replicates."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[Literal["MG1655"]] = MG1655_STRAIN

    def __init__(
        self,
        root: str = "data/torchcell/protein_turnover_gupta2024",
        io_workers: int = 0,
        ecoli_genome: EcoliK12MG1655Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the MG1655 genome resolves protein keys and deleted symbols."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProteinTurnoverExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProteinTurnoverExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Supplementary Data 1 and 7, the two released workbooks."""
        return [HALF_LIVES_FILENAME, CONFIDENCE_FILENAME]

    def download(self) -> None:
        """Link the deposited workbooks into ``raw/`` after verifying their pins."""
        manifest = load_manifest()
        root = raw_mirror_dir()
        os.makedirs(self.raw_dir, exist_ok=True)
        for relpath, filename, expected in _deposits():
            check_manifest_pin(relpath, manifest_sha256(manifest, relpath), expected)
            link_verified(root / relpath, osp.join(self.raw_dir, filename), expected)
        log.info("Gupta 2024 workbooks linked into %s", self.raw_dir)

    def _genome(self) -> EcoliK12MG1655Genome:
        """The injected MG1655 genome, or one opened from the genomes tier."""
        if self.ecoli_genome is None:
            genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
            if not isinstance(genome, EcoliK12MG1655Genome):
                raise RuntimeError(
                    f"{self.REFERENCE_STRAIN} resolved to {type(genome).__name__}"
                )
            self.ecoli_genome = genome
        return self.ecoli_genome

    def _genotype(self, condition: Condition, loci: Mapping[str, str]) -> Genotype:
        """The strain one condition was measured in.

        Conditions 1-8 are the unperturbed strain, so the genotype carries no
        perturbation: the axis that varies across them is the environment, and NCM3722
        itself is the reference's background, not an edit of it.
        """
        return Genotype(
            perturbations=[
                deletion_perturbation(symbol, loci[symbol])
                for symbol in condition.deleted_symbols
            ]
        )

    def _deleted_loci(self, genome: EcoliK12Genome) -> dict[str, str]:
        """The pinned assembly's locus tag for each deleted symbol the paper names."""
        symbols = sorted({s for c in CONDITIONS for s in c.deleted_symbols})
        loci: dict[str, str] = {}
        for symbol in symbols:
            locus = _symbol_locus(genome, symbol)
            if locus is None:
                raise RuntimeError(
                    f"the deleted symbol {symbol!r} does not resolve to one locus of "
                    f"{MG1655_ASSEMBLY_SET}; the genotype cannot be keyed"
                )
            loci[symbol] = locus
        if len(set(loci.values())) != len(loci):
            raise RuntimeError(f"two deleted symbols share a locus: {loci}")
        return loci

    @post_process
    def process(self) -> None:
        """Build one protein-turnover record per released biological replicate; write LMDB.

        Each condition is read twice: once as the authors' mean column (the Table 1
        oracle, the mean-equals-replicate-mean check and the reference), and once per
        replicate column, which is what the records store.
        """
        verify_raw_files(
            self.raw_dir,
            {
                HALF_LIVES_FILENAME: HALF_LIVES_SHA256,
                CONFIDENCE_FILENAME: CONFIDENCE_SHA256,
            },
        )
        half_lives_path = osp.join(self.raw_dir, HALF_LIVES_FILENAME)
        confidence_path = osp.join(self.raw_dir, CONFIDENCE_FILENAME)
        protein_ids, gene_names, cells = read_half_lives(half_lives_path)
        ci_ids, intervals = read_confidence_intervals(confidence_path)
        if protein_ids != ci_ids:
            raise RuntimeError(
                "Supplementary Data 1 and 7 do not list the same proteins in the same "
                "order, so their cells cannot be paired"
            )
        if len(protein_ids) != SOURCE_ROWS:
            raise RuntimeError(
                f"{len(protein_ids)} released rows, not the {SOURCE_ROWS} "
                f"Supplementary Data 1's description states ({DATA1_LEGEND.quote!r})"
            )

        genome = self._genome()
        names, mappings, route = identifier_names(genome, protein_ids, gene_names)
        stored, reconciliation = reconcile_locus_tags(
            genome, pd.Series(names), label=self.name
        )
        reconciliation.require_resolved(MIN_RESOLVED_FRACTION)
        keys = list(stored)
        if len(set(keys)) != len(keys):
            duplicated = [k for k, n in Counter(keys).items() if n > 1]
            raise RuntimeError(
                f"{len(duplicated)} locus tag(s) are claimed by two released rows "
                f"{duplicated[:10]}; a gene-level record cannot hold two values"
            )

        dropped = set(reconciliation.outside_namespace)
        if dropped != set(DROPPED_PROTEIN_KEYS):
            raise RuntimeError(
                f"the unresolved keys are {sorted(dropped)}, not the two isoform rows "
                f"{list(DROPPED_PROTEIN_KEYS)} this loader accounts for"
            )
        self._check_undetermined_oracle(keys, cells, intervals)

        loci = self._deleted_loci(genome)
        reference_genome = strain_reference()
        pub = publication()
        by_condition: dict[str, ConditionValues] = {}
        by_replicate: dict[str, tuple[ReplicateValues, ...]] = {}
        worst_mean_gap = 0.0
        for condition in CONDITIONS:
            values = condition_values(
                condition, keys, cells[condition.key], intervals[condition.key]
            )
            if len(values.half_life) != condition.table1_proteins:
                raise RuntimeError(
                    f"{condition.key}: {len(values.half_life)} released proteins, not "
                    f"the {condition.table1_proteins} Table 1 publishes "
                    f"({condition.table1_quote!r})"
                )
            for key in dropped:
                values.half_life.pop(key, None)
                values.degradation_rate.pop(key, None)
                values.degradation_rate_se.pop(key, None)
                values.degradation_rate_lower.pop(key, None)
                values.degradation_rate_upper.pop(key, None)
                values.censoring.pop(key, None)
                values.n_replicates.pop(key, None)
            if len(values.half_life) != condition.stored_proteins:
                raise RuntimeError(
                    f"{condition.key}: {len(values.half_life)} stored proteins, not "
                    f"the {condition.stored_proteins} this loader pins (Table 1's "
                    f"{condition.table1_proteins} less the isoform rows it quantified)"
                )
            censored = sum(
                1 for state in values.censoring.values() if state is Censoring.right
            )
            if censored != condition.censored_keys:
                raise RuntimeError(
                    f"{condition.key}: {censored} right-censored key(s), not the "
                    f"{condition.censored_keys} this loader pins ({CEILING_NOTE.quote!r})"
                )
            if len(values.degradation_rate_lower) != condition.interval_keys:
                raise RuntimeError(
                    f"{condition.key}: {len(values.degradation_rate_lower)} key(s) with "
                    f"a stored interval, not the {condition.interval_keys} this loader "
                    "pins (the one-replicate keys whose released interval inverts onto "
                    "the rate scale)"
                )
            by_condition[condition.key] = values
            replicates = tuple(
                replicate_values(
                    condition,
                    keys,
                    cells[condition.key],
                    intervals[condition.key],
                    replicate_id,
                )
                for replicate_id in REPLICATE_IDS
            )
            for number, replicate in enumerate(replicates):
                replicate.drop(dropped)
                pinned = (
                    condition.replicate_proteins[number],
                    condition.replicate_censored_keys[number],
                    condition.replicate_interval_keys[number],
                )
                observed = (
                    len(replicate.half_life),
                    sum(
                        1
                        for state in replicate.censoring.values()
                        if state is Censoring.right
                    ),
                    len(replicate.degradation_rate_lower),
                )
                if observed != pinned:
                    raise RuntimeError(
                        f"{condition.key} replicate {replicate.replicate_id}: (keys, "
                        f"right-censored keys, interval keys) = {observed}, not the "
                        f"pinned {pinned}"
                    )
            union = set().union(*(set(r.half_life) for r in replicates))
            if union != set(values.half_life):
                raise RuntimeError(
                    f"{condition.key}: the two replicate records key "
                    f"{len(union)} proteins, not the {len(values.half_life)} the "
                    "released mean column keys"
                )
            for key, mean_hours in values.half_life.items():
                own = [r.half_life[key] for r in replicates if key in r.half_life]
                worst_mean_gap = max(
                    worst_mean_gap, abs(statistics.fmean(own) - mean_hours)
                )
            by_replicate[condition.key] = replicates

        reference_values = by_condition[REFERENCE_CONDITION_KEY]
        reference_condition = next(
            c for c in CONDITIONS if c.key == REFERENCE_CONDITION_KEY
        )
        reference = ProteinTurnoverExperimentReference(
            dataset_name=self.name,
            genome_reference=reference_genome,
            environment_reference=environment(reference_condition),
            phenotype_reference=phenotype(reference_condition, reference_values),
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        index = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for condition in tqdm(CONDITIONS, desc="gupta2024-turnover"):
                for replicate in by_replicate[condition.key]:
                    experiment = ProteinTurnoverExperiment(
                        dataset_name=self.name,
                        genotype=self._genotype(condition, loci),
                        environment=environment(condition),
                        phenotype=replicate_phenotype(replicate),
                    )
                    txn.put(
                        f"{index}".encode(),
                        self._intern_record(experiment, reference, pub, itxn),
                    )
                    index += 1
        env.close()
        interned_env.close()

        self._write_reports(
            protein_ids,
            gene_names,
            keys,
            mappings,
            by_condition,
            by_replicate,
            route,
            reconciliation,
            index,
            worst_mean_gap,
        )
        log.info(
            "Gupta2024 turnover: %d records (one per released biological replicate) x "
            "%d-%d protein keys from %d released rows; %d keys dropped; the replicate "
            "records reproduce the authors' mean column to %.3e h",
            index,
            min(len(r.half_life) for rs in by_replicate.values() for r in rs),
            max(len(r.half_life) for rs in by_replicate.values() for r in rs),
            len(protein_ids),
            len(dropped),
            worst_mean_gap,
        )

    @staticmethod
    def _check_undetermined_oracle(
        keys: Sequence[str],
        cells: Mapping[str, Sequence[HalfLifeCell | None]],
        intervals: Mapping[str, Sequence[ConfidenceCell | None]],
    ) -> None:
        """Every ceiling-flagged half-life has an ``Undetermined`` interval, and only it.

        An independent cross-source check on the censoring flag: Supplementary Data 1's
        ``*`` and Supplementary Data 7's ``Undetermined`` are produced by different
        sheets of the authors' pipeline, and they agree on all 61,811 released cells.
        That agreement is what makes ``censoring`` complete per key: the source answers
        the question for every released cell, so the record states it for every key.
        """
        mismatched: list[tuple[str, str, int, bool, bool]] = []
        for condition in CONDITIONS:
            triples = cells[condition.key]
            confidence = intervals[condition.key]
            for row, key in enumerate(keys):
                for number in (0, 1):
                    cell = triples[3 * row + number]
                    interval = confidence[2 * row + number]
                    if cell is None:
                        continue
                    flagged = interval is not None and interval.undetermined
                    ceiling = cell.ceiling
                    if ceiling != flagged:
                        mismatched.append(
                            (condition.key, key, number + 1, ceiling, flagged)
                        )
        if mismatched:
            raise RuntimeError(
                f"{len(mismatched)} released cell(s) disagree between Supplementary "
                f"Data 1's ceiling flag and Supplementary Data 7's Undetermined: "
                f"{mismatched[:10]}"
            )

    def _write_reports(
        self,
        protein_ids: Sequence[str],
        gene_names: Sequence[str],
        keys: Sequence[str],
        mappings: Sequence[DerivedIdentifierMapping | None],
        by_condition: Mapping[str, ConditionValues],
        by_replicate: Mapping[str, Sequence[ReplicateValues]],
        route: IdentifierRoute,
        reconciliation: LocusTagReconciliation,
        kept_records: int,
        mean_gap_hours: float,
    ) -> None:
        """Write the identifier route, the ceiling ledger and the build accounting.

        ``identifier_route.csv`` records the TYPED ``DerivedIdentifierRoute`` member of
        each resolved row and the identifier it was derived from, both read off the row's
        own validated ``DerivedIdentifierMapping``, not a free string.
        """
        dropped = set(reconciliation.outside_namespace)
        with open(
            osp.join(self.preprocess_dir, "identifier_route.csv"), "w", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "protein_id",
                    "accession",
                    "released_gene_name",
                    "route",
                    "source_identifier",
                    "stored_key",
                    "kept",
                ]
            )
            for protein_id, gene_name, mapping, key in zip(
                protein_ids, gene_names, mappings, keys, strict=True
            ):
                writer.writerow(
                    [
                        protein_id,
                        base_accession(protein_id),
                        gene_name,
                        "" if mapping is None else mapping.route,
                        "" if mapping is None else mapping.source_identifier,
                        key,
                        "no" if key in dropped else "yes",
                    ]
                )
        released_cells = 0
        ceiling_cells = 0
        with open(
            osp.join(self.preprocess_dir, "ceiling_cells.csv"), "w", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "condition",
                    "locus_tag",
                    "replicate",
                    "ceiling_hours",
                    "doubling_time",
                ]
            )
            for condition in CONDITIONS:
                values = by_condition[condition.key]
                released_cells += sum(values.n_replicates.values())
                for key, number in values.ceiling_cells:
                    if key in dropped:
                        continue
                    ceiling_cells += 1
                    writer.writerow(
                        [
                            condition.key,
                            key,
                            number,
                            condition.ceiling_hours,
                            condition.doubling_time,
                        ]
                    )
        accounting = BuildAccounting(
            dataset=self.name,
            source_rows=len(protein_ids),
            released_cells=released_cells,
            ceiling_cells=ceiling_cells,
            censored_keys=sum(
                1
                for replicates in by_replicate.values()
                for replicate in replicates
                for state in replicate.censoring.values()
                if state is Censoring.right
            ),
            uncensored_keys=sum(
                1
                for replicates in by_replicate.values()
                for replicate in replicates
                for state in replicate.censoring.values()
                if state is Censoring.uncensored
            ),
            interval_keys=sum(
                len(replicate.degradation_rate_lower)
                for replicates in by_replicate.values()
                for replicate in replicates
            ),
            one_replicate_keys=sum(
                1
                for condition in CONDITIONS
                for n in by_condition[condition.key].n_replicates.values()
                if n == 1
            ),
            refused_intervals=[
                refused
                for condition in CONDITIONS
                for replicate in by_replicate[condition.key]
                for refused in replicate.refused_intervals
            ],
            confidence_level=CONFIDENCE_LEVEL,
            interval_method=INTERVAL_METHOD,
            measurement_type=MEASUREMENT_TYPE,
            dilution_rate_per_hour={
                condition.key: condition.dilution_rate_per_hour
                for condition in CONDITIONS
            },
            candidate_records=len(CONDITIONS) * len(REPLICATE_IDS),
            kept_records=kept_records,
            dropped_records=len(CONDITIONS) * len(REPLICATE_IDS) - kept_records,
            kept_protein_keys=len(protein_ids) - len(dropped),
            dropped_protein_keys=len(dropped),
            per_condition_keys={
                condition.key: len(by_condition[condition.key].half_life)
                for condition in CONDITIONS
            },
            per_record_keys={
                f"{condition.key}/{replicate.replicate_id}": len(replicate.half_life)
                for condition in CONDITIONS
                for replicate in by_replicate[condition.key]
            },
            per_record_censored_keys={
                f"{condition.key}/{replicate.replicate_id}": sum(
                    1
                    for state in replicate.censoring.values()
                    if state is Censoring.right
                )
                for condition in CONDITIONS
                for replicate in by_replicate[condition.key]
            },
            per_record_interval_keys={
                f"{condition.key}/{replicate.replicate_id}": len(
                    replicate.degradation_rate_lower
                )
                for condition in CONDITIONS
                for replicate in by_replicate[condition.key]
            },
            mean_reproduction_max_abs_diff_hours=mean_gap_hours,
            rules=[
                DropRule(
                    rule="protein_key_is_an_alternative_isoform_of_another_row",
                    scope="protein_key",
                    description=(
                        "the key is a UniProt isoform accession whose canonical form is "
                        "already a separate row of the same table, and its gene-name "
                        "cell is empty; a gene-level record cannot carry two values for "
                        "one locus, so the isoform row keys nothing"
                    ),
                    items=tuple(sorted(dropped)),
                )
            ],
            identifier_route=route,
            reconciliation=reconciliation,
            notes=[
                "one record per released BIOLOGICAL replicate: each condition's two "
                "'Half-life (hrs) in <condition>, replicate N' columns are two records "
                f"with replicate_id '1' and '2' ({REPLICATES.quote!r}; "
                f"{REPLICATES_ARE_BIOLOGICAL.quote!r}); no replicate cell is withdrawn "
                f"({NO_DATA_EXCLUDED.quote!r}), so none is dropped",
                "no CONDITION is dropped: all 13 released conditions become two "
                "records each, and the union of each condition's two records reproduces "
                "Table 1's per-condition protein count exactly",
                "half_life stores the replicate cell verbatim and degradation_rate is "
                "ln 2 over it, by the Supplementary Note's identity "
                f"({HALF_LIFE_IDENTITY.quote!r}); every n_replicates value is 1 and "
                "degradation_rate_se is None with a not_applicable_single_measurement "
                "gap",
                "the authors' 'Average of half-lives (hrs) in <condition>' column is "
                "read as an oracle: it equals the mean of the replicate records for "
                "every key (mean_reproduction_max_abs_diff_hours), and it is the "
                "phenotype_reference (N-lim 6 h wild type, replicate_id None, with the "
                "paper's two-replicate sd/sqrt(n) on the rate scale)",
                "every determined replicate cell stores its own released 95% interval, "
                "inverted onto the rate scale; a cell whose released upper endpoint is "
                "non-positive is refused (refused_intervals), and a ceiling cell "
                "carries Undetermined and no bound",
                "degradation_rate is the TOTAL turnover rate (active degradation plus "
                "dilution), per hour; the active rate k_D = k_total - D is not stored "
                "because it is negative wherever the fitted half-life exceeds the "
                "doubling time and the phenotype requires a non-negative rate",
                "on the REFERENCE only, degradation_rate_se is the paper's own "
                "two-replicate sd/sqrt(n) on the rate scale, nan where a key has one "
                "replicate or a ceiling cell, and the reference's bounds are stored for "
                "exactly its one-replicate keys (the interval of the mean of two fits "
                "is not published)",
                "ceiling-flagged cells are KEPT, as the authors keep them in their own "
                "mean and in Table 1's counts, and the flag TRAVELS: in a replicate "
                "record censoring is right exactly where that replicate's cell is "
                f"capped ({CEILING_NOTE.quote!r}) and uncensored otherwise. The side is "
                "stated on the released total half-life, which is what the cap was "
                "applied to; degradation_rate is ln 2 over it, so the same key bounds "
                "the rate from above. ceiling_cells.csv still carries the condition's "
                "ceiling value",
                "the intervals are also the censoring oracle: every Undetermined cell "
                "matches a ceiling-flagged cell and no other, over all 61,811 released "
                "cells, which is what makes the per-key censoring map complete",
                f"each resolved row's route is a typed DerivedIdentifierMapping "
                f"({route.rows_per_route}), validated at read time and recorded as the "
                f"enum value in identifier_route.csv; the per-key mapping cannot sit on "
                f"the record because a dict-keyed phenotype has no identifier_mapping "
                f"field. The one row where the two layers disagree is "
                f"{list(route.disagreements)}",
                "the DELETION perturbations keep route=gene_symbol: the paper releases "
                "its knockouts only as symbols, never as accessions",
                "the chemostat dilution rate is Environment.dilution_rate_per_hour, "
                "ln 2 over the doubling time (the dilution-limited total half-life IS "
                f"the doubling time: {DILUTION_LIMIT_IS_THE_DOUBLING_TIME.quote!r}); "
                "the batch condition carries none, and measurement_type now names the "
                "assay and its unit only",
            ],
        )
        accounting.check()
        with open(
            osp.join(self.preprocess_dir, "build_accounting.json"), "w"
        ) as handle:
            handle.write(accounting.model_dump_json(indent=2))

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "ProteinTurnoverGupta2024Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# Verification: a loader-level L0-L4 gate, because no family verifier exists yet
# --------------------------------------------------------------------------- #
def _records(dataset_root: str) -> list[dict[str, Any]]:
    """The built records of one tree."""
    from torchcell.verification.runners import load_records

    return load_records(dataset_root)


#: What identifies one of the 13 conditions in a built record. ``measurement_type`` is
#: NOT part of it any more: it names the assay and its unit only, which is the same
#: string in all 13. The dilution regime that used to be spelled into it now reads off
#: the environment, as the labeling window already did.
ConditionIdentity = tuple[str, float, float | None, tuple[str, ...]]


def _condition_identity(condition: Condition) -> ConditionIdentity:
    """What identifies one condition in a built record, independent of record order."""
    return (
        medium(condition.limitation).stated_name,
        condition.duration_hours,
        condition.dilution_rate_per_hour,
        tuple(sorted(condition.deleted_symbols)),
    )


def _record_identity(record: Mapping[str, Any]) -> ConditionIdentity:
    """The same identity, read back out of a built record."""
    experiment = record["experiment"]
    env = experiment["environment"]
    dilution = env["dilution_rate_per_hour"]
    return (
        str(env["media"]["name"]),
        float(env["duration_hours"]),
        None if dilution is None else float(dilution),
        tuple(
            sorted(
                str(perturbation["perturbed_gene_name"])
                for perturbation in experiment["genotype"]["perturbations"]
            )
        ),
    )


def _record_replicate(record: Mapping[str, Any]) -> str:
    """The ``replicate_id`` a built record states, or ``""`` when it states none."""
    replicate_id = record["experiment"]["phenotype"].get("replicate_id")
    return "" if replicate_id is None else str(replicate_id)


def _l1_condition_keys(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    """L1: each replicate record keys its pinned count, and each condition's two union
    to Table 1's count less its isoform rows.

    Records are matched to conditions by (medium, labeling window, dilution rate,
    deleted symbols), which is unique across the 13, plus ``replicate_id``, because the
    LMDB returns its keys in lexicographic order and positional matching would compare
    the wrong pairs.
    """
    expected = {
        (_condition_identity(condition), replicate_id): condition.replicate_proteins[
            number
        ]
        for condition in CONDITIONS
        for number, replicate_id in enumerate(REPLICATE_IDS)
    }
    observed: dict[tuple[ConditionIdentity, str], int] = {}
    unions: dict[ConditionIdentity, set[str]] = {}
    for record in records:
        identity = _record_identity(record)
        rates = record["experiment"]["phenotype"]["degradation_rate"]
        observed[(identity, _record_replicate(record))] = len(rates)
        unions.setdefault(identity, set()).update(rates)
    missing = sorted(str(key) for key in set(expected) - set(observed))
    extra = sorted(str(key) for key in set(observed) - set(expected))
    wrong = {
        str(key): {"observed": observed[key], "expected": expected[key]}
        for key in sorted(set(expected) & set(observed), key=str)
        if observed[key] != expected[key]
    }
    union_wrong = {
        str(_condition_identity(condition)): {
            "observed": len(unions.get(_condition_identity(condition), set())),
            "expected": condition.stored_proteins,
        }
        for condition in CONDITIONS
        if len(unions.get(_condition_identity(condition), set()))
        != condition.stored_proteins
    }
    passed = (
        not missing
        and not extra
        and not wrong
        and not union_wrong
        and len(observed) == len(records)
    )
    return LevelResult(
        level=Level.L1,
        name="per_replicate_protein_counts_and_their_union_match_table_1_less_isoforms",
        passed=passed,
        message=(
            f"{len(records)} records matched to {len(expected)} (condition, replicate) "
            f"pairs; {len(wrong)} count mismatch(es), {len(union_wrong)} condition "
            f"union mismatch(es), {len(missing)} missing, {len(extra)} unexpected"
        ),
        details={
            "mismatches": wrong,
            "union_mismatches": union_wrong,
            "missing": missing,
            "unexpected": extra,
            "table1": {
                str(_condition_identity(condition)): condition.table1_proteins
                for condition in CONDITIONS
            },
        },
    )


def _l1_key_alignment(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    """L1: every per-protein map of a record is keyed identically.

    ``censoring`` is in this set, not among the ragged maps: the source's oracle resolves
    every released cell, so a stored key without a censoring state would be a loss.
    """
    misaligned: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        phenotype_dump = record["experiment"]["phenotype"]
        rates = set(phenotype_dump["degradation_rate"])
        for field in ("half_life", "n_replicates", "censoring"):
            if set(phenotype_dump[field] or {}) != rates:
                misaligned.append({"record": index, "field": field})
    return LevelResult(
        level=Level.L1,
        name="every_per_protein_map_is_keyed_on_degradation_rate",
        passed=not misaligned,
        message=f"{len(records)} records checked; {len(misaligned)} misaligned",
        details={"misaligned": misaligned[:20]},
    )


def _l1_one_measurement_per_record(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    """L1: every record is ONE biological replicate, and each condition has both.

    ``replicate_id`` is ``"1"`` or ``"2"``, every ``n_replicates`` value is 1,
    ``degradation_rate_se`` is ``None`` and gapped as
    ``not_applicable_single_measurement``, and every condition holds exactly the two
    replicate records its headers print.
    """
    offenders: list[dict[str, Any]] = []
    replicates: dict[ConditionIdentity, list[str]] = {}
    for index, record in enumerate(records):
        phenotype_dump = record["experiment"]["phenotype"]
        replicate_id = _record_replicate(record)
        replicates.setdefault(_record_identity(record), []).append(replicate_id)
        if replicate_id not in REPLICATE_IDS:
            offenders.append(
                {"record": index, "reason": "replicate_id", "value": replicate_id}
            )
        if any(int(n) != 1 for n in phenotype_dump["n_replicates"].values()):
            offenders.append({"record": index, "reason": "n_replicates is not 1"})
        if phenotype_dump["degradation_rate_se"] is not None:
            offenders.append({"record": index, "reason": "degradation_rate_se is set"})
        gaps = {
            str(gap["field"]): str(gap["reason"])
            for gap in phenotype_dump.get("provenance_gaps") or []
        }
        if gaps.get("degradation_rate_se") != (
            ProvenanceGapReason.not_applicable_single_measurement.value
        ):
            offenders.append(
                {"record": index, "reason": "no single-measurement SE gap"}
            )
    incomplete = {
        str(identity): sorted(ids)
        for identity, ids in replicates.items()
        if sorted(ids) != list(REPLICATE_IDS)
    }
    return LevelResult(
        level=Level.L1,
        name="every_record_is_one_biological_replicate_and_each_condition_has_both",
        passed=not offenders and not incomplete,
        message=(
            f"{len(records)} records over {len(replicates)} conditions; "
            f"{len(offenders)} offender(s), {len(incomplete)} condition(s) without "
            "exactly replicates 1 and 2"
        ),
        details={"offenders": offenders[:20], "incomplete": incomplete},
    )


def _l1_censored_and_interval_counts(
    records: Sequence[Mapping[str, Any]],
) -> LevelResult:
    """L1: each replicate record's censored and interval key counts are the pinned ones."""
    expected = {
        (_condition_identity(condition), replicate_id): (
            condition.replicate_censored_keys[number],
            condition.replicate_interval_keys[number],
        )
        for condition in CONDITIONS
        for number, replicate_id in enumerate(REPLICATE_IDS)
    }
    observed = {
        (_record_identity(record), _record_replicate(record)): (
            sum(
                1
                for state in record["experiment"]["phenotype"]["censoring"].values()
                if str(state) == Censoring.right.value
            ),
            len(record["experiment"]["phenotype"]["degradation_rate_lower"]),
        )
        for record in records
    }
    wrong = {
        str(key): {"observed": observed[key], "expected": expected[key]}
        for key in sorted(set(expected) & set(observed), key=str)
        if observed[key] != expected[key]
    }
    passed = (
        not wrong and set(expected) == set(observed) and len(observed) == len(records)
    )
    censored = sum(count for count, _ in observed.values())
    intervals = sum(count for _, count in observed.values())
    return LevelResult(
        level=Level.L1,
        name="per_replicate_censored_and_interval_key_counts_are_the_pinned_ones",
        passed=passed,
        message=(
            f"{censored} right-censored key(s) and {intervals} key(s) with a stored "
            f"interval over {len(records)} records; {len(wrong)} mismatch(es)"
        ),
        details={"mismatches": wrong},
    )


def _l2_bounds_bracket_the_rate(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    """L2: every stored bound pair brackets its rate, and the two maps share keys."""
    offenders: list[dict[str, Any]] = []
    n = 0
    for index, record in enumerate(records):
        phenotype_dump = record["experiment"]["phenotype"]
        rates = phenotype_dump["degradation_rate"]
        lower = phenotype_dump["degradation_rate_lower"]
        upper = phenotype_dump["degradation_rate_upper"]
        if set(lower) != set(upper):
            offenders.append({"record": index, "reason": "lower and upper keys differ"})
            continue
        for key, low in lower.items():
            n += 1
            high = float(upper[key])
            rate = float(rates[key])
            if not float(low) <= rate <= high:
                offenders.append(
                    {"record": index, "key": key, "bounds": [low, high], "rate": rate}
                )
    return LevelResult(
        level=Level.L2,
        name="stored_interval_brackets_the_stored_rate",
        passed=not offenders,
        message=f"{n} bound pair(s) checked; {len(offenders)} outside",
        details={"offenders": offenders[:20]},
    )


def _l3_dilution_rate_is_ln2_over_the_doubling_time(
    records: Sequence[Mapping[str, Any]],
) -> LevelResult:
    """L3: the chemostat records carry ln 2 over their doubling time, the batch none."""
    expected = {
        _condition_identity(condition): condition.dilution_rate_per_hour
        for condition in CONDITIONS
    }
    wrong: dict[str, dict[str, float | None]] = {}
    continuous = 0
    for record in records:
        identity = _record_identity(record)
        observed = identity[2]
        if observed is not None:
            continuous += 1
        if observed != expected[identity]:
            wrong[str(identity)] = {
                "observed": observed,
                "expected": expected[identity],
            }
    chemostats = len(REPLICATE_IDS) * sum(
        1 for c in CONDITIONS if c.reactor == "chemostat"
    )
    return l3_convention(
        "dilution_rate_per_hour_is_ln2_over_the_doubling_time",
        not wrong and continuous == chemostats,
        detail=(
            f"{continuous} continuous-culture record(s) of {len(records)}, "
            f"{chemostats} expected; {len(wrong)} mismatch(es) "
            f"({DILUTION_LIMIT_IS_THE_DOUBLING_TIME.quote!r})"
        ),
    )


def _l3_rate_is_ln2_over_half_life(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    """L3: the stored rate is exactly ``ln 2`` over the stored half-life."""
    worst = 0.0
    n = 0
    for record in records:
        phenotype_dump = record["experiment"]["phenotype"]
        for key, rate in phenotype_dump["degradation_rate"].items():
            n += 1
            half_life = float(phenotype_dump["half_life"][key])
            worst = max(worst, abs(math.log(2.0) / half_life - float(rate)))
    return l3_convention(
        "degradation_rate_is_ln2_over_half_life",
        worst <= 1e-12,
        detail=(
            f"{n} values; largest |ln2/T - k| is {worst:.3e} (the Supplementary Note's "
            "identity k_total = ln 2 / T_half)"
        ),
    )


def _l3_assembly_pin(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    """L3: every reference pins MG1655 GenBank with the NCM3722 background."""
    pins = {
        (
            str(record["reference"]["genome_reference"].get("assembly_set")),
            str(record["reference"]["genome_reference"].get("strain")),
        )
        for record in records
    }
    expected = {(MG1655_ASSEMBLY_SET, MEASURED_STRAIN)}
    return LevelResult(
        level=Level.L3,
        name="assembly_pin_is_mg1655_genbank_with_the_ncm3722_background",
        passed=pins == expected,
        message=f"{len(pins)} distinct pin(s): {sorted(pins)}",
        details={"pins": sorted(pins)},
    )


def _l4_keys_are_loci(
    records: Sequence[Mapping[str, Any]], genome: EcoliK12Genome
) -> LevelResult:
    """L4: every stored protein key is a locus of the pinned assembly."""
    loci = set(genome.genbank.loci)
    keys = {
        key
        for record in records
        for key in record["experiment"]["phenotype"]["degradation_rate"]
    }
    outside = sorted(keys - loci)
    containment = (len(keys) - len(outside)) / len(keys) if keys else 0.0
    return LevelResult(
        level=Level.L4,
        name="stored_protein_keys_are_loci_of_the_pinned_assembly",
        passed=not outside,
        message=(
            f"{len(keys)} distinct keys; containment {containment:.6f}; "
            f"{len(outside)} outside"
        ),
        details={"n_keys": len(keys), "outside": outside[:20]},
    )


def verify_build(
    dataset_root: str,
    *,
    genome: EcoliK12MG1655Genome | None = None,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run an L0-L4 gate on a built tree and write its report.

    ``torchcell/verification/`` has no protein-turnover family verifier --
    ``verify_protein_dataset`` reads ``phenotype['protein_abundance']`` and cannot see a
    rate -- so the gate is composed here from the shared primitives in
    ``torchcell.verification.levels`` rather than forced through an unrelated family.
    L1 also checks that every record is one biological replicate (``replicate_id`` 1 or
    2, ``n_replicates`` 1, no standard error, the single-measurement gap) and that each
    condition has both. L2 checks the rates and that every stored interval brackets its
    rate; L3 re-derives the half-life identity, the chemostat dilution rate
    and the assembly pin; L4 contains every stored key in the pinned assembly's loci.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    records = _records(dataset_root)
    if genome is None:
        resolved = bacterial_genome("ecoli", MG1655_STRAIN, data_root)
        if not isinstance(resolved, EcoliK12MG1655Genome):
            raise RuntimeError(f"MG1655 resolved to {type(resolved).__name__}")
        genome = resolved
    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    report = VerificationReport(
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{HALF_LIVES_REL}",
            citation_key=CITATION_KEY,
            sha256=HALF_LIVES_SHA256,
            method=(
                "Supplementary Data 1 'Half-life (hrs) in <condition>, replicate N' "
                "per condition; one ProteinTurnoverExperiment per released biological "
                f"replicate ({REPLICATES.value} per condition), degradation_rate = "
                "ln 2 / that replicate's released total half-life (per hour), "
                "n_replicates 1"
            ),
            page="TableS1, columns 'Half-life (hrs) in <condition>, replicate 1/2'",
            retrieved=RAW_RETRIEVED_AT,
        ),
    )
    report.add(l0_structural((record["experiment"] for record in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_condition_keys(records))
    report.add(_l1_key_alignment(records))
    report.add(_l1_one_measurement_per_record(records))
    report.add(_l1_censored_and_interval_counts(records))
    rates = [
        float(value)
        for record in records
        for value in record["experiment"]["phenotype"]["degradation_rate"].values()
    ]
    report.add(l2_value_fidelity(rates, allow_nan=False, minimum=0.0))
    report.add(_l2_bounds_bracket_the_rate(records))
    report.add(_l3_rate_is_ln2_over_half_life(records))
    report.add(_l3_dilution_rate_is_ln2_over_the_doubling_time(records))
    report.add(_l3_assembly_pin(records))
    report.add(_l4_keys_are_loci(records, genome))
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build the dataset under ``DATA_ROOT`` and verify it, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    root = osp.join(data_root, "data/torchcell/protein_turnover_gupta2024")
    dataset = ProteinTurnoverGupta2024Dataset(root=root)
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
                    "released_cells",
                    "ceiling_cells",
                    "kept_records",
                    "kept_protein_keys",
                    "dropped_protein_keys",
                    "per_condition_keys",
                    "per_record_keys",
                    "mean_reproduction_max_abs_diff_hours",
                )
            },
            indent=2,
        )
    )
    print(verify_build(root, data_root=data_root).summary())


if __name__ == "__main__":
    main()
