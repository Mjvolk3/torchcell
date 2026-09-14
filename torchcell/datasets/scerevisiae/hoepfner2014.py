# torchcell/datasets/scerevisiae/hoepfner2014
# [[torchcell.datasets.scerevisiae.hoepfner2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/hoepfner2014
# Test file: tests/torchcell/datasets/scerevisiae/test_hoepfner2014.py
"""Hoepfner 2014 HIP-HOP chemogenomic atlas: env x geno -> sensitivity score.

Hoepfner et al. 2014 (Microbiol Res, doi:10.1016/j.micres.2013.11.004) is the Novartis
genome-wide chemogenomic resource: 2956 HIP + 2923 HOP experiments for 1776 discrete
compounds (>40M data points), each compound profiled in duplicate at (or near) its
S. cerevisiae IC30.

ENCODABLE COMPOUNDS ONLY (build filter): ~92% of the profiled compounds are PROPRIETARY
Novartis ``CMBxxx`` entries with no released structure -- black-box perturbations no
molecular encoder can represent. This loader keeps ONLY the compounds with a released
SMILES in ``Table_S1.xls`` (152 of the 153 CMB rows the table carries; 150 of those CMB
ids appear in a kept HIP column and 149 in a kept HOP column), so every stored
(ORF, compound) record is featurisable by the cell graph transformer. The full atlas is
recoverable by removing the ``smiles is None`` skip in ``_column_meta``. Provenance:
paper.md line 110 ("In addition to 1641 proprietary compounds (named CMBxxx), we included
135 reference compounds ... Table S1"); analysis in
``experiments/017-hoepfner-background-mutations`` (``compound_encodability.json``).

COMPOUND IDENTITY -- the compound is keyed by structure, never by a tagged label. The
resolver is called with the CLEAN common name (or ``CMB<id>`` for a proprietary compound
whose structure WAS released), so the stored ``Compound`` carries the curated table's
canonical name and its InChIKey / ChEBI id / PubChem CID. Where no curated row exists the
InChIKey is DERIVED from the Table S1 SMILES with ``inchikey_from_smiles`` (RDKit); a
curated row always wins, because the two routes measurably disagree (concanamycin A's
curated PubChem key and its SMILES-derived key differ in the stereo block). Measured over
the 150 kept CMB ids: 149 carry a structure identifier, 148 distinct InChIKeys (CMB 244
and CMB 1818 are two fermentation batches of concanamycin A and correctly merge onto one
key).

RECORDS DROPPED -- rule: a column is dropped when its compound carries NO structure
identifier after resolution (no curated identifier, and no RDKit-parseable released
SMILES). Measured: ONE compound, CMB409 "Boromycin", whose released SMILES RDKit
2026.03.6 cannot parse (boron cage). That is 2 columns and 10,232 records (HIP 5,746 +
HOP 4,486), 0.327% of the encodable build, leaving 3,124,319 records over 149 compounds
and 608 sensitivity columns (HIP 1,759,255 + HOP 1,365,064). The ORF rule below removes
30 rows per assay (7,273 HIP + 6,671 HOP cells) on top of that. The rules and the
measured counts are written to ``<root>/dropped_records.json`` at build time.

SCREEN (study) IDENTITY -- ``EnvironmentResponsePhenotype.screen_id`` carries the
deposited study number. It is a real batch covariate, *"<Study number>: an internal id
for the study in which the compound was profiled. All compound profiles of one study use
the same set of control samples and are normalized together."* (paper.md line 50), and it
is what keeps one (strain, condition) L1-unique once the compound name is cleaned:
measured, 47 kept columns collide on (compound, dose) alone (HIP 22, HOP 25) and 0 collide
on (compound, dose, study). The reference is per (assay, study) for the same reason: each
study normalizes against its OWN control samples.

It is the canonical use case for the WS15 schema extension
(``EngineeredCopyNumberPerturbation`` + ``ReferenceGenome.ploidy``), because HIP and HOP
are two DIPLOID deletion collections:

- HIP (haploinsufficiency profiling) uses the HETEROZYGOUS deletion collection (YSC1055):
  in a DIPLOID one of the two autosomal copies is deleted (KanMX), leaving one copy ->
  reduced dosage. This collection INCLUDES ESSENTIAL genes (they are viable as
  heterozygotes), so HIP reaches genes the homozygous collection cannot. A HIP strain is
  ``EngineeredCopyNumberPerturbation(copy_number=1, reference_copy_number=2,
  marker="KanMX")`` in a ``ReferenceGenome(ploidy="diploid")``, exactly as the paper
  constructs it: *"The heterozygous and homozygous deletion strain collections were
  acquired (YSC1055 and YSC1056, OpenBiosystems) and pools generated as published (Pierce
  et al. 2007)."* (line 64); the cassette is KanMX, *"the compound hypersensitivity
  phenotype did not result from the KanMx replacement of the ORF"* (line 176).
- HOP (homozygous deletion profiling) uses the HOMOZYGOUS deletion collection (YSC1056):
  BOTH copies deleted in a diploid -> total absence; only non-essential genes are viable.
  A HOP strain is the existing absence leaf ``KanMxDeletionPerturbation`` in a
  ``ReferenceGenome(ploidy="diploid")``.

A HIP strain is dosage-accurate, not sequence-accurate: one allele is physically REPLACED
by a KanMX cassette, and the CNV leaf records the dosage plus the marker, not the
deletion+insertion edit at that locus.

``ReferenceGenome.strain`` is the bare strain token ``BY4743`` on BOTH arms, so the two
reference genomes join each other and the served BY4743 diploid reference
(``ohnuki2018.py``). The deletion-collection catalogue numbers (YSC1055 for HIP, YSC1056
for HOP) are recorded here and in ``preprocess/sourced_values.json``, NOT in the strain
string: neither ``EngineeredCopyNumberPerturbation`` nor ``KanMxDeletionPerturbation``
carries a collection slot, and both sit inside served dataset closures, so adding one is a
full-rebuild trigger rather than this dataset's call.

TABLE S5 BACKGROUND MUTATIONS -- 157 positional HIP strains carry a documented background
mutation (chromosome XI aneuploidy for clusters 1 and 2, a WHI2/YOR043w nonsense mutation
for cluster 3, a 12 kbp chromosome V amplification for cluster 4). Their records are KEPT,
not dropped: the phenotype is real and measured, and the paper states the hypersensitivity
*"did not track with any one mutation"* (line 176), so inventing a per-gene perturbation
for it would encode an inference as an observation. The flag is a property of the physical
REAGENT and no served-class slot exists for it (``Genotype`` is not a
``ProvenanceGapMixin``, so a gap cannot be asserted on it, and a field on ``Genotype`` or
on either deletion leaf is a full-rebuild trigger), so it is carried as a typed,
sha256-anchored per-strain file beside the build:
``<root>/table_s5_affected_strains.json``, written from
``experiments/017-hoepfner-background-mutations/results/table_s5_affected_strains.csv``
(itself cross-validated against the mirrored ``si/Table_S5.xls``) with the per-strain
cluster, mutation and measured record counts, joinable on ``systematic_gene_name``.

READOUT -- the stored score is the (adjusted) MADL SENSITIVITY score
(``measurement_type=sensitivity_score``), the paper's DEFINED per-experiment quantity:
*"In order to measure the relative abundance of each strain with respect to the averages
of the control samples we compute MAD logarithmic (MADL) scores for each
compound/concentration combination ... the MADL score is given as (r_L - med(r_L)) /
MAD(r_L) where the median and MAD are computed over all strains in one sample."*
(paper.md line 84), then adjusted for replicate variability. The deposited files ALSO
carry a companion gene-wise ``z-score`` column per experiment (a second normalization
across all experiments); this loader stores the sensitivity score (the atomic
per-experiment readout with a clean replicate count) and records the z-score's existence
in ``units``. ``assay_type=pooled_competitive_growth_barcode``, sourced by *"The TAG PCR
amplification and GenFlex Tag16K v2 hybridization protocol was used as described (Pierce
et al. 2006)."* (line 80).

SOURCED PROVENANCE -- every metadata number this loader hardcodes is a ``SourcedValue``
in ``SOURCED_VALUES`` (verbatim quote + the mirrored ``paper.md`` sha256), written to
``preprocess/sourced_values.json`` at build time; what the paper does not report is a
typed ``ProvenanceGap`` on the record:

- Replicate structure -> ``n_samples``: 2 for an ``Ad.`` column, 1 for a ``MADL`` column
  (line 52), ``sample_unit=technical_replicate`` (the two duplicate wells in the same
  plate, line 68). The REFERENCE carries ``n_samples=4``: the paper gives only a range,
  *"the four to eight control replicates"* (line 84), with no per-record column, so the
  conservative lower end is stored (larger implied SE; never the optimistic end).
- Concentration: each column header carries the per-experiment uM value (line 50) and the
  dose-setting rule is IC30 (line 58). The paper qualifies the realized dose, *"tested at
  n = 2 within the same plate at or close to their IC30 concentration"* (line 68), so
  ``basis=IC30`` records how the dose was SET, not that every column is exactly IC30.
- Medium: the shared ``MEDIA_LIBRARY`` object ``YPD_LIQUID`` on BOTH the treated and the
  control arm (the control differs by carrying no compound, not by being a different
  medium). The 2% DMSO vehicle rides on the treated perturbation's ``Solvent`` and is an
  explicit 2% v/v ``SmallMoleculePerturbation`` on the control arm, so the vehicle control
  is chemically stated rather than implied by a medium name.
- Temperature 30 C (lines 68, 70). No pH is reported anywhere in the paper, so nothing is
  owed on the physical-factor axis.
- Duration: HOP is ONE 16 h incubation, ~5 doublings (line 70) ->
  ``duration_hours=16.0, duration_generations=5.0``. HIP is FOUR sequential 16 h
  incubations reaching ~20 generations (line 68) -> ``duration_generations=20.0``, with
  ``duration_hours`` a ``ProvenanceGap(not_reported_by_primary)`` because the paper never
  states WHICH passage's plate was hybridized, so neither 16 nor 64 can be asserted.
- No per-cell uncertainty is released: the replicate t-test p-value is folded into the
  adjusted score (line 84), so ``environment_response_se``,
  ``environment_response_uncertainty`` and ``environment_response_uncertainty_type`` are
  typed ``ProvenanceGap``s on every phenotype, never silent Nones.

DATA SOURCE (scriptable + sha256-pinned): Dryad doi:10.5061/dryad.v5m8v files
``HIP_scores.txt`` (644 MB) and ``HOP_scores.txt`` (505 MB) -- the deposited processed
score matrices (rows = systematic ORF names, columns = per-experiment scores) -- and
``Table_S1.xls`` (compound IC30 + common name + SMILES for the reference/novel MoA
compounds; the ~1641 proprietary CMB compounds have no released name). The canonical copy
is the RAW MIRROR
``$DATA_ROOT/torchcell-raw/hoepfnerHighresolutionChemicalDissection2014/`` and its
``manifest.json`` of ``ArtifactRecord``s; ``download()`` symlinks and sha256-verifies from
the mirror, and runs the recorded Dryad retrieval (whose frontend sits behind an Anubis
SHA-256 proof-of-work that ``_dryad_get`` solves deterministically) only when the mirror is
absent.

BUILD / SOURCE QUIRKS handled deterministically (no fabrication):
- Each experiment column is an atomic measurement (a distinct CMB/concentration/study).
  Identity is structural, never smuggled into a name: the compound is the compound, the
  dose is the ``Concentration``, the assay is the genotype leaf, the study is
  ``screen_id``.
- Systematic ORF names go through the SHARED ``SCerevisiaeGenome.resolve_gene_name``, the
  same policy as Costanzo 2021, Wildenhain 2015 and Hillenmeyer 2008: a CURRENT gene is
  kept as is; a RENAMED alias (an old deletion-collection ORF that SGD merged into a
  neighbour, e.g. ``YAL035C-A`` -> ``YAL034C-B``) is kept under the current systematic
  name with the source ORF as ``perturbed_gene_name``, so it stays a DISTINCT strain of
  that gene; a NON_GENE_FEATURE (pseudogene, blocked reading frame, transposable-element
  gene) or a RETIRED name is DROPPED with its status and feature type in the ledger. The
  R64 FASTA universe alone is NOT the gene set (it lists pseudogenes and blocked reading
  frames as ORFs), which is why a FASTA membership test used to pass 14 non-gene rows
  that the shared L1 canonical-gene-name rule then failed. Empty cells are skipped.
- Constant sub-objects are INTERNED into ``processed/interned``: one environment per
  sensitivity column, one reference per (assay, study), one publication. A record stores
  ``{"$ref": ...}`` pointers for them, which is what keeps a 3.1M-record LMDB small and
  lets a streaming reader memoize per-condition work.
"""

import csv
import hashlib
import json
import logging
import os
import os.path as osp
import pickle
import re
import shutil
import time
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import (
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datamodels.schema import (
    AssayType,
    Compound,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    EngineeredCopyNumberPerturbation,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    Solvent,
    Temperature,
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
from torchcell.sequence.genome.registry import SGD_S288C_R64, resolve
from torchcell.sequence.genome.scerevisiae.s288c import GeneNameStatus
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1016/j.micres.2013.11.004"
CITATION_KEY = "hoepfnerHighresolutionChemicalDissection2014"

# The mirrored OCR markdown every quote below is anchored to, and the mirrored SI table
# the Table S5 strain list was cross-validated against.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "a9877549eff2fe1aaf8aa403d9fea1c381284de030326f4475e869c102af0aeb"
TABLE_S5_XLS = "si/Table_S5.xls"
TABLE_S5_XLS_SHA256 = "b123dc3e87fc10d3b4256f449fcd2eb38c91d1779000278af5a1a788356624a2"

# The 017 cross-validation of Table S5 against the deposited HIP strains: one row per
# affected strain, sha256-pinned so the flag file beside the build stays auditable.
TABLE_S5_STRAINS_CSV = (
    "experiments/017-hoepfner-background-mutations/results/"
    "table_s5_affected_strains.csv"
)
TABLE_S5_STRAINS_CSV_SHA256 = (
    "05bb74330f7118a8bc565fcbc587c2732daf5785db54b9a29a1aedbaca64bdc1"
)

# Dryad doi:10.5061/dryad.v5m8v file-stream ids + pinned sha256 of the deposited files.
DRYAD_DOI = "10.5061/dryad.v5m8v"
_DRYAD_FILES: dict[str, dict[str, str]] = {
    "HIP_scores.txt": {
        "url": "https://datadryad.org/downloads/file_stream/4834608",
        "sha256": "dbc5041defea9c046da0890d5e569f97d5f7afbf50ea0885f539ea8e5980cd24",
    },
    "HOP_scores.txt": {
        "url": "https://datadryad.org/downloads/file_stream/4834609",
        "sha256": "99b386a84384eae847657ed41bf222c9550a87ef961f0ab191833c918771ffd7",
    },
    "Table_S1.xls": {
        "url": "https://datadryad.org/downloads/file_stream/4834600",
        "sha256": "115bb31cc5e696588d1ecb4ffa262475e05025e22347f7e004f77fd635898209",
    },
}

_ASSAYS = (("HIP_scores.txt", "HIP"), ("HOP_scores.txt", "HOP"))

# S288C reference gene universe (systematic ORF + RNA-coding names) for R64 resolution.
_SGD_GENE_FASTAS = (
    "orf_coding_all_R64-4-1_20230830.fasta",
    "rna_coding_R64-4-1_20230830.fasta",
)

# Experiment column header: '(Ad.|MADL) scores for Exp. <CMB>_<conc>_<HIP|HOP>_<study>'
# with an optional trailing ' z-score' (the companion gene-wise z-score column).
_COL_RE = re.compile(
    r"^(?P<prefix>Ad\.|MADL) scores for Exp\. "
    r"(?P<cmb>\d+)_(?P<conc>[\d.]+)_(?P<assay>HIP|HOP)_(?P<study>\S+?)(?P<z> z-score)?$"
)

MEASUREMENT_UNITS = (
    "adjusted MADL sensitivity score = (r_L - med(r_L)) / MAD(r_L) over all pool strains, "
    "r_L = log ratio of treated vs control strain abundance (Hoepfner 2014); negative = "
    "hypersensitive, positive = resistant; 0 = growth equal to control. A companion "
    "gene-wise z-score column is deposited per experiment but not stored here."
)

REFERENCE_UNITS = (
    "the screen's own no-drug DMSO control samples, the denominator of the MADL log "
    "ratio; a strain growing exactly as its controls scores 0"
)

DROP_RULE = (
    "drop every record whose compound carries no structure identifier after resolution: "
    "no curated compound_identity_table row with an InChIKey / ChEBI id / PubChem CID, "
    "and no RDKit-parseable released Table S1 SMILES to derive an InChIKey from"
)

#: Retention rule for a row's ORF: kept only when it resolves to a LIVE R64 gene.
_KEPT_STATUSES = frozenset({GeneNameStatus.CURRENT, GeneNameStatus.RENAMED})

ORF_RULE = (
    "a row is kept only when its 'Systematic Name' resolves through the shared "
    "SCerevisiaeGenome.resolve_gene_name to CURRENT (stored as is) or RENAMED (stored "
    "under the current systematic name, the source ORF kept as perturbed_gene_name so a "
    "merged-ORF strain stays distinct); a NON_GENE_FEATURE (pseudogene, blocked reading "
    "frame, transposable-element gene) or a RETIRED name drops the row and every one of "
    "its cells"
)

TABLE_S5_POLICY = (
    "KEPT and FLAGGED, never dropped: the measurement is real, and the paper reports that "
    "the hypersensitivity did not track with any one mutation, so no per-gene "
    "perturbation is invented for the background mutation. The flag is a property of the "
    "physical strain and no served class carries a slot for it, so it lives here and is "
    "joined on systematic_gene_name."
)

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# --------------------------------------------------------------------------- #
# Sourced metadata. Every number this loader hardcodes is anchored to a verbatim
# quote from the mirrored paper.md plus its sha256; what the paper does NOT report
# is a typed ProvenanceGap on the record instead.
# --------------------------------------------------------------------------- #
def _sv(value: Any, quote: str, note: str | None = None) -> SourcedValue:
    """A ``SourcedValue`` anchored to the mirrored ``paper.md`` by sha256."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (mirrored artifact)",
        ),
        quote=quote,
        note=note,
    )


SOURCED_VALUES: dict[str, SourcedValue] = {
    "concentration_unit": _sv(
        "uM",
        "<Conc.>: The $\\mu \\mathrm { M }$ concentration of a compound",
        note="the per-experiment dose is the column header's numeric field, in micromolar",
    ),
    "dose_basis": _sv(
        "IC30",
        "Testing at $\\mathrm { I C } _ { 3 0 }$ resulted in best expected strain "
        "sensitivity associated with the established biology of these substances.",
        note="records how the dose was SET; the paper qualifies the realized dose as 'at "
        "or close to' the IC30, so this is the rule, not an exactness claim",
    ),
    "dose_basis_qualification": _sv(
        "at or close to IC30",
        "Experimental compounds were tested at $n = 2$ within the same plate at or close "
        "to their $\\mathrm { I C } _ { 3 0 }$ concentration.",
    ),
    "n_samples_treated": _sv(
        2,
        "If a column name starts with “Ad.”, then the MADL score of each strain "
        "is adjusted for the variability between the two measurements of the two samples "
        "which are contained in the study for the given compound. In some cases only one "
        "of the two samples for a compound passed QC; in this case there is only one "
        "measurement and no adjustment for variability is made; this is indicated by the "
        "prefix “MADL”.",
        note="n_samples = 2 for an 'Ad.' column and 1 for a 'MADL' column, read off the "
        "column prefix per experiment",
    ),
    "n_samples_reference": _sv(
        4,
        "We also compute the $t$ -test $p$ -value, $p$ , between the two replicates for a "
        "compound and the four to eight control replicates as a measure of the "
        "variability of the compound and control sample intensities across the "
        "experiment.",
        note="the paper gives a 4-8 range with no per-record column, so the conservative "
        "lower end is stored (larger implied SE); never the optimistic end",
    ),
    "solvent_percent": _sv(
        2.0,
        "In all experiments DMSO was normalized to $2 \\%$ to allow testing compounds up "
        "to $2 0 0 \\mu \\mathrm { M }$ .",
    ),
    "temperature_c": _sv(
        30.0,
        "Plates were incubated for $1 6 \\mathrm { h }$ in a robotic shaking incubator at "
        "$3 0 ^ { \\circ } C / 5 5 0$ RPM allowing for ${ \\sim } 5$ doublings.",
    ),
    "hop_duration_hours": _sv(
        16.0,
        "Plates were incubated for $1 6 \\mathrm { h }$ in a robotic shaking incubator at "
        "$3 0 ^ { \\circ } C / 5 5 0$ RPM allowing for ${ \\sim } 5$ doublings and were "
        "then stored at $4 ^ { \\circ } \\mathsf C$ (Fig. S3).",
    ),
    "hop_duration_generations": _sv(
        5.0,
        "The HOP assay was performed similar to the HIP experiment but the duration was "
        "reduced to ${ \\sim } 5$ doublings and no dilutions were necessary.",
    ),
    "hip_duration_generations": _sv(
        20.0,
        "Once inoculated the new plate was incubated at $3 0 ^ { \\circ } C / 5 5 0$ RPM "
        "to allow the next 5 yeast generations (generation 6–10) and the plate "
        "containing the first 5 doubling cultures was stored at "
        "$4 ^ { \\circ } \\mathrm { C } .$ . This procedure was repeated 2 more times "
        "until the final plate containing the yeast with ${ \\sim } 2 0$ generations were "
        "stored at $4 ^ { \\circ } \\mathsf C$ (Fig. S2).",
        note="HIP is four sequential 16 h incubations reaching ~20 generations; the paper "
        "does not state which passage was hybridized, so duration_hours is a "
        "ProvenanceGap rather than a guessed 16 or 64",
    ),
    "assay_type": _sv(
        "pooled_competitive_growth_barcode",
        "The TAG PCR amplification and GenFlex Tag16K v2 hybridization protocol was used "
        "as described (Pierce et al. 2006).",
    ),
    "screen_id": _sv(
        "the deposited study number",
        "<Study number>: an internal id for the study in which the compound was profiled. "
        "All compound profiles of one study use the same set of control samples and are "
        "normalized together.",
    ),
    "hip_collection": _sv(
        "YSC1055",
        "The heterozygous and homozygous deletion strain collections were acquired "
        "(YSC1055 and YSC1056, OpenBiosystems) and pools generated as published (Pierce "
        "et al. 2007).",
        note="the HIP heterozygous deletion collection; recorded here because no served "
        "perturbation leaf carries a collection slot",
    ),
    "hop_collection": _sv(
        "YSC1056",
        "The heterozygous and homozygous deletion strain collections were acquired "
        "(YSC1055 and YSC1056, OpenBiosystems) and pools generated as published (Pierce "
        "et al. 2007).",
        note="the HOP homozygous deletion collection; recorded here because no served "
        "perturbation leaf carries a collection slot",
    ),
    "hip_marker": _sv(
        "KanMX",
        "Sporulation and analysis of segregants indicated that for all clusters, the "
        "compound hypersensitivity phenotype did not result from the KanMx replacement of "
        "the ORF (data not shown).",
        note="the deletion cassette on the affected HIP allele",
    ),
    "measurement_definition": _sv(
        "adjusted MADL sensitivity score",
        "In order to measure the relative abundance of each strain with respect to the "
        "averages of the control samples we compute MAD logarithmic (MADL) scores for "
        "each compound/concentration combination. If we denote the logarithm of the ratio "
        "of the average intensity of the compound samples over the average intensity of "
        "the control samples as $r _ { L }$ , then the MADL score is given as "
        "$( r _ { L } - \\mathrm { m e d } ( r _ { L } ) ) / \\mathrm { M A D } "
        "( r _ { L } )$ where the median and MAD are computed over all strains in one "
        "sample.",
    ),
    "table_s5_affected_strains": _sv(
        157,
        "The situation was similar for clusters 3 and 4 (Fig. 6B, C and Table S5), "
        "leading to a total of 157 affected HIP strains.",
    ),
    "table_s5_mutations": _sv(
        "chromosome XI aneuploidy (CL1, CL2); WHI2/YOR043w nonsense (CL3); chromosome V "
        "12 kbp amplification (CL4)",
        "For clusters 1 and 2, the hypersensitive phenotype did not track with any one "
        "mutation, but correlated with an increased sequencing coverage of chromosome XI "
        "suggestive of aneuploidy (Fig. S15). In contrast, Cluster 3 strains revealed a "
        "common point mutation in the WHI2/YOR043w gene resulting in a premature stop "
        "codon which truncates the ORF by $6 0 \\%$ and likely results in a "
        "non-functional protein (Fig. S16A). In support of this, the original WHI2/whi2 "
        "HIP strain significantly correlates with all identified cluster 3 strains (Fig. "
        "S16B). Finally, close analysis of Cluster 4 strain sequences revealed a discrete "
        "12 kbps region on chromosome V where the relative coverage was increased by "
        "$5 0 \\%$ (Fig. S17). This region contains 6 annotated chromosomal features, "
        "including 3 genes with defined functions.",
        note="the flag is a reagent property; the paper says the phenotype did not track "
        "with any one mutation, so no per-gene perturbation is invented for it",
    ),
}

# The typed absences. HIP's duration in hours, and the per-cell uncertainty on every
# phenotype, are values the primary never reported: declared, never defaulted.
_HIP_DURATION_HOURS_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="HIP ran four sequential 16 h incubations to ~20 generations; the paper does not "
    "state which passage's plate was hybridized, so neither 16 nor 64 h can be asserted",
)

_SE_GAP_NOTE = (
    "no per-cell uncertainty is released: the replicate t-test p-value is folded into the "
    "adjusted score a_L = min(0.05/p, 1) * s_L"
)

_UNCERTAINTY_FIELDS = (
    "environment_response_se",
    "environment_response_uncertainty",
    "environment_response_uncertainty_type",
)


def _phenotype_gaps() -> list[ProvenanceGap]:
    """The three uncertainty absences every Hoepfner phenotype declares."""
    return [
        ProvenanceGap(
            field=field,
            reason=ProvenanceGapReason.not_reported_by_primary,
            note=_SE_GAP_NOTE,
        )
        for field in _UNCERTAINTY_FIELDS
    ]


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/<citation_key>/`` -- the canonical raw copy."""
    root = data_root if data_root is not None else os.environ["DATA_ROOT"]
    return Path(root) / "torchcell-raw" / CITATION_KEY


def deposit_raw_mirror(
    *, source_dir: str | Path, retrieved_at: str, data_root: str | None = None
) -> Path:
    """Write the raw mirror from already-retrieved files plus its ``manifest.json``.

    Run once by hand. Idempotent by sha256: a mirror file already carrying the pinned
    hash is left alone, one carrying a different hash raises rather than being silently
    overwritten. Every file's sha256 is verified against ``_DRYAD_FILES`` before it is
    recorded, so the manifest can only ever describe the bytes the build consumed.
    """
    root = raw_mirror_dir(data_root)
    root.mkdir(parents=True, exist_ok=True)
    files: list[ArtifactRecord] = []
    for name, spec in _DRYAD_FILES.items():
        dest = root / name
        if not dest.exists():
            shutil.copy2(Path(source_dir) / name, dest)
        digest = sha256_file(dest)
        if digest != spec["sha256"]:
            raise RuntimeError(
                f"{dest} sha256 {digest} != pinned {spec['sha256']}; refusing to record"
            )
        files.append(
            ArtifactRecord(
                path=name,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=digest,
                source=spec["url"],
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=spec["url"],
                    retriever="torchcell.datasets.scerevisiae.hoepfner2014._dryad_get",
                    params={
                        "url": spec["url"],
                        "note": "Dryad's frontend sits behind an Anubis SHA-256 "
                        "proof-of-work; _dryad_get solves it deterministically via "
                        "_solve_anubis and streams the file body",
                    },
                    sha256=digest,
                    retrieved_at=retrieved_at,
                ),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=(
            "High-resolution chemical dissection of a model eukaryote reveals targets, "
            "pathways and gene functions"
        ),
        files=files,
        si_data_sources=[f"https://doi.org/{DRYAD_DOI}"],
        si_expected=sorted(_DRYAD_FILES),
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


# --------------------------------------------------------------------------- #
# Typed build-side records (written beside the LMDB, not into it)
# --------------------------------------------------------------------------- #
class DroppedCompound(BaseModel):
    """One compound the identity rule removed, and what it cost."""

    model_config = ConfigDict(extra="forbid")

    cmb_id: str
    source_name: str
    smiles: str | None
    resolution_status: str
    unresolved_reason: str | None
    n_columns: int
    n_records: int


class DroppedOrf(BaseModel):
    """One source ORF the retention rule removed, with what the resolver said."""

    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(description="the 'Systematic Name' cell, verbatim")
    status: str = Field(description="GeneNameStatus the shared resolver returned")
    resolved_to: str | None = Field(
        default=None, description="what the resolver mapped it to, when anything"
    )
    feature_type: str | None = Field(
        default=None, description="GFF feature type for a NON_GENE_FEATURE"
    )
    n_records: int = Field(
        description="non-empty cells in KEPT sensitivity columns lost with this row"
    )


class DroppedRecordReport(BaseModel):
    """The build's drop ledger: the rules, and exactly what they removed."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    rule: str
    orf_rule: str
    n_kept: int
    n_dropped: int
    kept_by_assay: dict[str, int]
    dropped_by_assay: dict[str, int]
    dropped_compounds: list[DroppedCompound]
    dropped_orfs: dict[str, list[DroppedOrf]] = Field(
        description="per assay, the rows whose ORF resolves to neither a CURRENT nor a "
        "RENAMED R64 gene, never guessed onto a current gene"
    )
    renamed_orfs: dict[str, dict[str, str]] = Field(
        description="per assay, source ORF -> current systematic name for every RENAMED "
        "row kept; the source ORF stays on the record as perturbed_gene_name"
    )
    created_at: str


class TableS5Strain(BaseModel):
    """One Table S5 HIP strain carrying a documented background mutation."""

    model_config = ConfigDict(extra="forbid")

    systematic_gene_name: str
    common_gene_name: str
    cluster: str
    is_positional: bool
    mutation: str
    construction_lab: str
    n_records: int


class TableS5FlagFile(BaseModel):
    """Per-strain reagent-quality flag for the HIP records this build KEPT.

    The flag is a property of the physical strain, not of a measurement, and no served
    class carries a slot for it: ``Genotype`` is not a ``ProvenanceGapMixin`` (so a typed
    gap cannot be asserted on it) and a field on ``Genotype`` or on a deletion leaf would
    force a full rebuild of every served dataset importing it. It is therefore a typed,
    sha256-anchored file beside the build, joinable on ``systematic_gene_name``.
    """

    model_config = ConfigDict(extra="forbid")

    dataset: str
    citation_key: str
    source_csv: str
    source_csv_sha256: str
    table_s5_path: str
    table_s5_sha256: str
    paper_quote: str
    policy: str
    n_strains: int
    n_positional_strains: int
    n_records_flagged: int
    n_positional_records_flagged: int
    strains: list[TableS5Strain]
    created_at: str


# --------------------------------------------------------------------------- #
# Retrieval helpers
# --------------------------------------------------------------------------- #
def _solve_anubis(random_data: str, difficulty: int) -> tuple[str, int]:
    """Solve an Anubis 'fast' proof-of-work: sha256(random_data + nonce) with
    ``difficulty`` leading zero hex nibbles. Deterministic and scriptable.
    """
    p = difficulty // 2
    odd = difficulty % 2
    nonce = 0
    while True:
        digest = hashlib.sha256((random_data + str(nonce)).encode()).digest()
        if all(digest[i] == 0 for i in range(p)) and not (
            odd and (digest[p] >> 4) != 0
        ):
            return digest.hex(), nonce
        nonce += 1


def _dryad_get(session: requests.Session, url: str) -> requests.Response:
    """GET a Dryad file-stream URL, clearing the Anubis PoW challenge if present.

    Returns a streamed response for the file body. Retries on WAF throttling.
    """
    for _ in range(80):
        resp = session.get(url, timeout=180, stream=True)
        if not resp.headers.get("content-type", "").startswith("text/html"):
            return resp
        body = resp.text
        resp.close()
        challenge_match = re.search(
            r'id="anubis_challenge" type="application/json">(.*?)</script>', body, re.S
        )
        if challenge_match is None:
            time.sleep(15)  # throttled: back off and retry
            continue
        challenge = json.loads(challenge_match.group(1))["challenge"]
        base_match = re.search(
            r'id="anubis_base_prefix" type="application/json">(.*?)</script>',
            body,
            re.S,
        )
        base = json.loads(base_match.group(1)) if base_match else ""
        digest, nonce = _solve_anubis(challenge["randomData"], challenge["difficulty"])
        pass_url = (
            f"https://datadryad.org{base}"
            "/.within.website/x/cmd/anubis/api/pass-challenge"
        )
        cleared = session.get(
            pass_url,
            params={
                "id": challenge["id"],
                "response": digest,
                "nonce": nonce,
                "redir": url,
                "elapsedTime": 100,
            },
            stream=True,
            allow_redirects=True,
            timeout=1800,
        )
        if not cleared.headers.get("content-type", "").startswith("text/html"):
            return cleared
        cleared.close()
        time.sleep(15)
    raise RuntimeError(f"could not clear Anubis challenge for {url}")


# --------------------------------------------------------------------------- #
# Source readers
# --------------------------------------------------------------------------- #
def _load_sgd_genes(data_root: str) -> set[str]:
    """S288C R64 systematic-name universe from the ORF + RNA-coding FASTA headers."""
    genes: set[str] = set()
    for name in _SGD_GENE_FASTAS:
        with open(resolve(SGD_S288C_R64, name, data_root=data_root)) as handle:
            for line in handle:
                if line.startswith(">"):
                    genes.add(line[1:].split()[0])
    return genes


def _load_compound_meta(table_s1_path: str) -> dict[str, dict[str, str | None]]:
    """CMB id -> {common_name, smiles} from Table_S1 (reference + novel MoA + structures).

    Only the reference/novel MoA compounds carry a name; the ~1641 proprietary CMB
    compounds are absent (name/SMILES stay None).
    """
    import pandas as pd

    def cmb_ids(value: Any) -> list[str]:
        """Parse a CMB-ID cell; some structure rows list several ids ('244, 1818')."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, (int, float)):
            return [str(int(value))]
        return [tok.strip() for tok in str(value).split(",") if tok.strip().isdigit()]

    meta: dict[str, dict[str, str | None]] = {}
    xl = pd.ExcelFile(table_s1_path)
    for sheet in ("Reference Substances known MoA", "Substances novel MoA"):
        frame = xl.parse(sheet, header=0)
        for _, row in frame.iterrows():
            name = row.get("Common Name")
            name = None if (name is None or pd.isna(name)) else str(name).strip()
            for cmb_id in cmb_ids(row.get("CMB ID")):
                meta.setdefault(cmb_id, {"common_name": None, "smiles": None})
                if name:
                    meta[cmb_id]["common_name"] = name
    structures = xl.parse("All Structures", header=0)
    for _, row in structures.iterrows():
        smiles = row.get("SMILE string")
        smiles = None if (smiles is None or pd.isna(smiles)) else str(smiles).strip()
        for cmb_id in cmb_ids(row.get("CMB ID")):
            meta.setdefault(cmb_id, {"common_name": None, "smiles": None})
            if smiles:
                meta[cmb_id]["smiles"] = smiles
    return meta


def load_table_s5_strains(repo_root: str | Path) -> dict[str, dict[str, str]]:
    """Table S5 affected HIP strains from the 017 cross-validation CSV (sha256-pinned).

    Keyed by systematic ORF name. The CSV is the committed result of
    ``experiments/017-hoepfner-background-mutations``, which reconciled the mirrored
    ``si/Table_S5.xls`` against the deposited HIP strain list.
    """
    path = Path(repo_root) / TABLE_S5_STRAINS_CSV
    digest = sha256_file(path)
    if digest != TABLE_S5_STRAINS_CSV_SHA256:
        raise RuntimeError(
            f"{path} sha256 {digest} != pinned {TABLE_S5_STRAINS_CSV_SHA256}"
        )
    with path.open() as handle:
        return {row["orf"]: row for row in csv.DictReader(handle)}


def _has_identifier(compound: Compound) -> bool:
    """Whether a compound carries ANY structure identifier (the retention rule)."""
    return (
        compound.inchikey is not None
        or compound.chebi_id is not None
        or compound.pubchem_cid is not None
    )


class _ColumnMeta:
    """One kept sensitivity column: its record templates and its identity fields."""

    __slots__ = ("index", "cmb", "study", "exp_base", "pheno_base", "env_dump")

    def __init__(
        self,
        index: int,
        cmb: str,
        study: str,
        exp_base: dict[str, Any],
        pheno_base: dict[str, Any],
        env_dump: dict[str, Any],
    ) -> None:
        self.index = index
        self.cmb = cmb
        self.study = study
        self.exp_base = exp_base
        self.pheno_base = pheno_base
        # The environment INLINE, for the per-column schema validation: what the LMDB
        # stores is the interned pointer, and the schema only accepts the resolved form.
        self.env_dump = env_dump


class _BuildCounts:
    """Everything the build measures while it streams, for the two ledger files.

    Kept as plain counters rather than a pydantic model because it is a mutable
    accumulator, not a stored record; what it produces IS modelled
    (``DroppedRecordReport``, ``TableS5FlagFile``).
    """

    __slots__ = ("kept", "dropped", "flagged", "dropped_orfs", "renamed_orfs")

    def __init__(self) -> None:
        self.kept: Counter[str] = Counter()  # assay -> kept records
        self.dropped: Counter[tuple[str, str]] = Counter()  # (assay, CMB) -> removed
        self.flagged: Counter[str] = Counter()  # ORF -> kept HIP records, Table S5 only
        self.dropped_orfs: dict[str, list[DroppedOrf]] = {}  # assay -> dropped rows
        self.renamed_orfs: dict[str, dict[str, str]] = {}  # assay -> source -> current


@register_dataset
class EnvChemgenHoepfner2014Dataset(ExperimentDataset):
    """Hoepfner 2014 HIP-HOP atlas: env x (het-CNV | hom-deletion) -> sensitivity score."""

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_hoepfner2014",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        genome: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset. ``genome`` supplies ``resolve_gene_name``; when it is
        None a read-only S288C genome is opened at build time (never at import).
        """
        self.genome = genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    def _resolver(self) -> Callable[[str], Any]:
        """``resolve_gene_name`` from the supplied genome, or a read-only S288C genome."""
        if self.genome is None:
            from dotenv import load_dotenv

            from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

            load_dotenv()
            data_root = os.environ["DATA_ROOT"]
            # overwrite=False is mandatory: a rebuild here would race any other process
            # holding the same gffutils database.
            self.genome = SCerevisiaeGenome(
                genome_root=osp.join(data_root, "data/sgd/genome"),
                go_root=osp.join(data_root, "data/go"),
                overwrite=False,
            )
        resolver: Callable[[str], Any] = self.genome.resolve_gene_name
        return resolver

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return EnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return EnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Dryad score matrices + compound table required before processing."""
        return ["HIP_scores.txt", "HOP_scores.txt", "Table_S1.xls"]

    # ---- retrieval ------------------------------------------------------------ #
    def download(self) -> None:
        """Link the raw files from the sha256-verified mirror; fetch Dryad if absent.

        The mirror is the canonical copy (the stored artifact plus its sha256 is
        canonical, the URL is historical retrieval metadata), so the rebuild path reads it
        first and verifies every byte against the pinned hash. The Dryad fetch is the
        recorded retrieval command, run only where the mirror has not been deposited.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        mirror = raw_mirror_dir()
        for name, spec in _DRYAD_FILES.items():
            dest = osp.join(self.raw_dir, name)
            if osp.exists(dest):
                continue
            source = mirror / name
            if not source.exists():
                self._fetch_from_dryad(name, spec, dest)
                continue
            digest = sha256_file(source)
            if digest != spec["sha256"]:
                raise RuntimeError(
                    f"raw mirror {source} sha256 mismatch: got {digest}, expected "
                    f"{spec['sha256']}"
                )
            os.symlink(source, dest)
            log.info("Linked %s from the raw mirror (sha256 verified)", name)

    def _fetch_from_dryad(self, name: str, spec: dict[str, str], dest: str) -> None:
        """Stream one Dryad file (solving the Anubis PoW) and verify its pinned sha256."""
        session = requests.Session()
        session.headers.update({"User-Agent": _UA})
        log.info("Downloading Hoepfner2014 %s from %s", name, spec["url"])
        resp = _dryad_get(session, spec["url"])
        digest = hashlib.sha256()
        with open(dest, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
        resp.close()
        got = digest.hexdigest()
        if got != spec["sha256"]:
            raise RuntimeError(
                f"{name} sha256 mismatch: got {got}, expected {spec['sha256']}"
            )
        log.info("Wrote %s (sha256 verified)", dest)

    # ---- record builders ------------------------------------------------------ #
    @staticmethod
    def _compound_name(cmb: str, meta: dict[str, dict[str, str | None]]) -> str:
        """The compound's own name: its released common name, or its ``CMB`` id.

        The resolver key is this CLEAN name. The experiment tag (dose, assay, study) is
        NOT part of the chemical identity: the dose is the ``Concentration``, the assay is
        the genotype leaf, and the study is the phenotype's ``screen_id``.
        """
        return (meta.get(cmb) or {}).get("common_name") or f"CMB{cmb}"

    @staticmethod
    def _vehicle() -> Compound:
        """The DMSO vehicle, resolved to its curated structure identity."""
        return resolved_compound("DMSO")

    def _environment(
        self, assay: str, perturbations: list[EnvironmentPerturbationType]
    ) -> Environment:
        """The assay's environment: YPD liquid at 30 C for its own exposure duration."""
        if assay == "HIP":
            return Environment(
                media=YPD_LIQUID,
                temperature=Temperature(value=30.0),
                perturbations=perturbations,
                aerobicity="aerobic",
                duration_generations=20.0,
                provenance_gaps=[_HIP_DURATION_HOURS_GAP],
            )
        return Environment(
            media=YPD_LIQUID,
            temperature=Temperature(value=30.0),
            perturbations=perturbations,
            aerobicity="aerobic",
            duration_hours=16.0,
            duration_generations=5.0,
        )

    def _reference(
        self, assay: str, study: str
    ) -> EnvironmentResponseExperimentReference:
        """The screen's own no-drug DMSO control of the diploid collection, score 0.

        Compound-INDEPENDENT but study-SPECIFIC: *"All compound profiles of one study use
        the same set of control samples and are normalized together."* (paper.md line 50),
        so a study's control set is its own object rather than one control standing in for
        every screen in the atlas.
        """
        control_env = self._environment(
            assay,
            [
                SmallMoleculePerturbation(
                    compound=self._vehicle(),
                    concentration=Concentration(
                        value=2.0,
                        unit=ConcentrationUnit.percent_v_v,
                        basis=DoseBasis.fixed,
                    ),
                )
            ],
        )
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4743", ploidy="diploid"
            ),
            environment_reference=control_env,
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.sensitivity_score,
                assay_type=AssayType.pooled_competitive_growth_barcode,
                environment_response=0.0,
                n_samples=4,
                sample_unit=SampleUnit.technical_replicate,
                units=REFERENCE_UNITS,
                screen_id=study,
                provenance_gaps=_phenotype_gaps(),
            ),
        )

    def _genotype(
        self, assay: str, orf: str, source_orf: str | None = None
    ) -> Genotype:
        """HIP -> heterozygous engineered-CNV (copy 1 of 2); HOP -> homozygous deletion.

        ``orf`` is the CURRENT systematic name; ``source_orf`` is the deposited row name,
        which differs only for a RENAMED (merged) ORF and then stays on the record as
        ``perturbed_gene_name`` so that strain is not conflated with the gene's own row.
        """
        perturbed = orf if source_orf is None else source_orf
        if assay == "HIP":
            return Genotype(
                perturbations=[
                    EngineeredCopyNumberPerturbation(
                        systematic_gene_name=orf,
                        perturbed_gene_name=perturbed,
                        copy_number=1,
                        reference_copy_number=2,
                        marker="KanMX",
                    )
                ]
            )
        return Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=perturbed
                )
            ]
        )

    def _intern_ref(self, obj: Any, hint: str, itxn: Any) -> Any:
        """Intern one constant sub-object and return the ``{"$ref": ...}`` pointer.

        Small objects stay inline (``_maybe_intern`` leaves the dump in place), so the
        return value is always exactly what belongs in the record.
        """
        container: dict[str, Any] = {"value": obj.model_dump()}
        self._maybe_intern(container, "value", obj, hint, itxn)
        return container["value"]

    def _column_meta(
        self,
        header: list[str],
        assay: str,
        meta: dict[str, dict[str, str | None]],
        itxn: Any,
    ) -> tuple[list[_ColumnMeta], list[tuple[int, str]]]:
        """Parse the sensitivity columns into kept templates and dropped columns.

        Returns ``(kept, dropped)``, where ``dropped`` is ``(column index, CMB id)`` for a
        column whose compound carries NO structure identifier after resolution. A CMB id
        with no released SMILES at all is the encodable-only build filter, not a drop.
        """
        kept: list[_ColumnMeta] = []
        dropped: list[tuple[int, str]] = []
        for index, raw in enumerate(header):
            name = raw.strip().strip('"')
            match = _COL_RE.match(name)
            if match is None or match.group("z") is not None:
                continue  # column 0 (Systematic Name) or a z-score companion column
            if match.group("assay") != assay:
                continue
            cmb = match.group("cmb")
            smiles = (meta.get(cmb) or {}).get("smiles")
            if smiles is None:
                # ENCODABLE-COMPOUNDS-ONLY build: ~92% of the atlas is proprietary Novartis
                # CMBxxx with no released structure (plus the lone named-but-structureless
                # CMB222 "Enniatin derivative"); these are black-box perturbations a
                # molecular encoder cannot represent, so they are dropped at build time.
                continue
            compound = resolved_compound(
                self._compound_name(cmb, meta),
                smiles=smiles,
                known_proprietary=(meta.get(cmb) or {}).get("common_name") is None,
                derive_from_smiles=True,
            )
            if not _has_identifier(compound):
                dropped.append((index, cmb))
                continue
            conc = float(match.group("conc"))
            study = match.group("study")
            environment = self._environment(
                assay,
                [
                    SmallMoleculePerturbation(
                        compound=compound,
                        concentration=Concentration(
                            value=conc,
                            unit=ConcentrationUnit.micromolar,
                            basis=DoseBasis.IC30,
                        ),
                        solvent=Solvent(
                            name="DMSO", percent=2.0, compound=self._vehicle()
                        ),
                    )
                ],
            )
            phenotype = EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.sensitivity_score,
                assay_type=AssayType.pooled_competitive_growth_barcode,
                environment_response=0.0,
                n_samples=2 if match.group("prefix") == "Ad." else 1,
                sample_unit=SampleUnit.technical_replicate,
                units=MEASUREMENT_UNITS,
                screen_id=study,
                provenance_gaps=_phenotype_gaps(),
            )
            # The template experiment is a real model dump; only the genotype, the score
            # and the interned environment pointer differ per record.
            exp_base = EnvironmentResponseExperiment(
                dataset_name=self.name,
                genotype=self._genotype(assay, "YAL001C"),
                environment=environment,
                phenotype=phenotype,
            ).model_dump()
            env_dump = exp_base["environment"]
            exp_base = {
                **exp_base,
                "environment": self._intern_ref(environment, compound.name, itxn),
            }
            kept.append(
                _ColumnMeta(
                    index, cmb, study, exp_base, phenotype.model_dump(), env_dump
                )
            )
        return kept, dropped

    # ---- build ---------------------------------------------------------------- #
    def _read_header(self, filename: str) -> list[str]:
        """The first (header) line of one score matrix, split on tabs."""
        with open(osp.join(self.raw_dir, filename)) as handle:
            return handle.readline().rstrip("\n").split("\t")

    def _iter_records(
        self,
        path: str,
        assay: str,
        sgd_genes: set[str],
        columns: list[_ColumnMeta],
        dropped_columns: list[tuple[int, str]],
        reference_refs: dict[tuple[str, str], Any],
        pub_ref: Any,
        counts: _BuildCounts,
        flag_orfs: frozenset[str],
        validate: Callable[[Any], object],
        resolve: Callable[[str], Any],
    ) -> Iterator[bytes]:
        """Yield one pickled record per (kept row, kept sensitivity column).

        A row's ORF goes through ``resolve`` under ``ORF_RULE``; the cells a dropped
        row loses in KEPT columns are counted, and the dropped columns' non-empty cells
        are counted in the SAME pass, so the drop ledger is measured rather than
        estimated. The first record of each column is validated against the experiment
        schema, which is what makes the per-column template safe.
        """
        dropped_orfs: list[DroppedOrf] = []
        renamed: dict[str, str] = {}
        validated: set[int] = set()
        genotypes: dict[str, dict[str, Any]] = {}
        with open(path) as handle:
            handle.readline()
            for line in tqdm(handle, desc=f"Hoepfner2014 {assay}"):
                parts = line.rstrip("\n").split("\t")
                source_orf = parts[0].strip().strip('"')
                resolution = resolve(source_orf)
                orf = resolution.systematic_name
                if resolution.status not in _KEPT_STATUSES or orf not in sgd_genes:
                    dropped_orfs.append(
                        DroppedOrf(
                            source_name=source_orf,
                            status=str(resolution.status.value),
                            resolved_to=orf,
                            feature_type=resolution.feature_type,
                            n_records=sum(
                                1
                                for col in columns
                                if col.index < len(parts)
                                and parts[col.index].strip().strip('"') != ""
                            ),
                        )
                    )
                    continue
                if resolution.status == GeneNameStatus.RENAMED:
                    renamed[source_orf] = orf
                genotype = genotypes.get(source_orf)
                if genotype is None:
                    genotype = self._genotype(assay, orf, source_orf).model_dump()
                    genotypes[source_orf] = genotype
                for index, cmb in dropped_columns:
                    if index < len(parts) and parts[index].strip().strip('"') != "":
                        counts.dropped[(assay, cmb)] += 1
                # Table S5 names PHYSICAL strains by their 2014 name, so the flag matches
                # on the deposited (source) name only: a RENAMED merged-ORF strain is not
                # the listed strain of the gene it now maps to, and is never flagged
                # through the current name.
                flagged = assay == "HIP" and source_orf in flag_orfs
                for col in columns:
                    if col.index >= len(parts):
                        continue
                    cell = parts[col.index].strip().strip('"')
                    if cell == "":
                        continue
                    experiment = {
                        **col.exp_base,
                        "genotype": genotype,
                        "phenotype": {
                            **col.pheno_base,
                            "environment_response": float(cell),
                        },
                    }
                    if col.index not in validated:
                        # Validate the RESOLVED record (the interned pointer spliced back
                        # to the real environment): the per-column template is only safe
                        # if what a reader reconstructs still satisfies the schema.
                        validate({**experiment, "environment": col.env_dump})
                        validated.add(col.index)
                    counts.kept[assay] += 1
                    if flagged:
                        counts.flagged[source_orf] += 1
                    yield pickle.dumps(
                        {
                            "experiment": experiment,
                            "reference": reference_refs[(assay, col.study)],
                            "publication": pub_ref,
                        }
                    )
        counts.dropped_orfs[assay] = sorted(dropped_orfs, key=lambda d: d.source_name)
        counts.renamed_orfs[assay] = dict(sorted(renamed.items()))
        log.info(
            "Hoepfner2014 %s: wrote %d records; kept %d RENAMED rows under their current "
            "name; dropped %d rows (%d cells) whose ORF is no current gene: %s",
            assay,
            counts.kept[assay],
            len(renamed),
            len(dropped_orfs),
            sum(d.n_records for d in dropped_orfs),
            [f"{d.source_name}:{d.status}" for d in counts.dropped_orfs[assay]],
        )

    @post_process
    def process(self) -> None:
        """Stream both score matrices into the LMDB (one record per (ORF, experiment)).

        The constant sub-objects are interned FIRST and their transaction committed before
        any record is written, so a crash can only ever leave orphan interned rows, never
        a record with a dangling ``$ref``. Records then commit in batches so a multi-
        million-record build never accumulates one giant dirty-page write transaction.
        """
        from dotenv import load_dotenv
        from pydantic import TypeAdapter

        from torchcell.datamodels.schema import ExperimentType

        load_dotenv()
        data_root = os.environ["DATA_ROOT"]
        repo_root = Path(__file__).resolve().parents[3]
        sgd_genes = _load_sgd_genes(data_root)
        meta = _load_compound_meta(osp.join(self.raw_dir, "Table_S1.xls"))
        table_s5 = load_table_s5_strains(repo_root)
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")
        validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        columns: dict[str, list[_ColumnMeta]] = {}
        dropped_columns: dict[str, list[tuple[int, str]]] = {}
        reference_refs: dict[tuple[str, str], Any] = {}
        with interned_env.begin(write=True) as itxn:
            for filename, assay in _ASSAYS:
                kept, dropped = self._column_meta(
                    self._read_header(filename), assay, meta, itxn
                )
                columns[assay] = kept
                dropped_columns[assay] = dropped
                log.info(
                    "Hoepfner2014 %s: %d kept sensitivity experiments, %d dropped "
                    "(compound without a structure identifier)",
                    assay,
                    len(kept),
                    len(dropped),
                )
                for study in sorted({col.study for col in kept}):
                    reference_refs[(assay, study)] = self._intern_ref(
                        self._reference(assay, study), f"{assay}_{study}", itxn
                    )
            pub_ref = self._intern_ref(publication, DOI, itxn)

        counts = _BuildCounts()
        flag_orfs = frozenset(table_s5)
        resolve = self._resolver()
        idx = 0
        batch_size = 500_000
        txn = env.begin(write=True)
        for filename, assay in _ASSAYS:
            for value in self._iter_records(
                osp.join(self.raw_dir, filename),
                assay,
                sgd_genes,
                columns[assay],
                dropped_columns[assay],
                reference_refs,
                pub_ref,
                counts,
                flag_orfs,
                validate,
                resolve,
            ):
                txn.put(f"{idx}".encode(), value)
                idx += 1
                if idx % batch_size == 0:
                    txn.commit()
                    txn = env.begin(write=True)
        txn.commit()
        env.close()
        interned_env.close()
        log.info("Wrote %d Hoepfner2014 environment-response experiments to LMDB", idx)

        self._write_drop_ledger(meta, dropped_columns, counts)
        self._write_table_s5_flags(table_s5, counts)
        self._write_sourced_values()

    def _write_drop_ledger(
        self,
        meta: dict[str, dict[str, str | None]],
        dropped_columns: dict[str, list[tuple[int, str]]],
        counts: _BuildCounts,
    ) -> None:
        """Write ``<root>/dropped_records.json``: the rule and what it measured."""
        cmb_columns: Counter[str] = Counter()
        for dropped in dropped_columns.values():
            for _, cmb in dropped:
                cmb_columns[cmb] += 1
        compounds: list[DroppedCompound] = []
        for cmb in sorted(cmb_columns):
            source_name = self._compound_name(cmb, meta)
            resolution = resolve_compound_identity(
                name=source_name,
                known_proprietary=(meta.get(cmb) or {}).get("common_name") is None,
            )
            compounds.append(
                DroppedCompound(
                    cmb_id=cmb,
                    source_name=source_name,
                    smiles=(meta.get(cmb) or {}).get("smiles"),
                    resolution_status=resolution.status.value,
                    unresolved_reason=resolution.unresolved_reason,
                    n_columns=cmb_columns[cmb],
                    n_records=sum(
                        n
                        for (_, dropped_cmb), n in counts.dropped.items()
                        if dropped_cmb == cmb
                    ),
                )
            )
        report = DroppedRecordReport(
            dataset=self.name,
            rule=DROP_RULE,
            orf_rule=ORF_RULE,
            n_kept=sum(counts.kept.values()),
            n_dropped=sum(counts.dropped.values()),
            kept_by_assay=dict(counts.kept),
            dropped_by_assay={
                assay: sum(counts.dropped[(assay, cmb)] for _, cmb in dropped)
                for assay, dropped in dropped_columns.items()
            },
            dropped_compounds=compounds,
            dropped_orfs=counts.dropped_orfs,
            renamed_orfs=counts.renamed_orfs,
            created_at=datetime.now(UTC).isoformat(),
        )
        out = osp.join(self.root, "dropped_records.json")
        with open(out, "w") as handle:
            handle.write(report.model_dump_json(indent=2))
        log.info(
            "Hoepfner2014 drop ledger: kept %d, dropped %d -> %s",
            report.n_kept,
            report.n_dropped,
            out,
        )

    def _write_table_s5_flags(
        self, table_s5: dict[str, dict[str, str]], counts: _BuildCounts
    ) -> None:
        """Write ``<root>/table_s5_affected_strains.json``: the reagent-quality flag."""
        flagged = counts.flagged
        strains = [
            TableS5Strain(
                systematic_gene_name=orf,
                common_gene_name=row["gene"],
                cluster=row["Cluster"],
                is_positional=row["is_positional"] == "True",
                mutation=row["mutation"],
                construction_lab=row["Lab"],
                n_records=flagged[orf],
            )
            for orf, row in sorted(table_s5.items())
            if flagged[orf] > 0
        ]
        positional = [strain for strain in strains if strain.is_positional]
        flag_file = TableS5FlagFile(
            dataset=self.name,
            citation_key=CITATION_KEY,
            source_csv=TABLE_S5_STRAINS_CSV,
            source_csv_sha256=TABLE_S5_STRAINS_CSV_SHA256,
            table_s5_path=TABLE_S5_XLS,
            table_s5_sha256=TABLE_S5_XLS_SHA256,
            paper_quote=str(SOURCED_VALUES["table_s5_mutations"].quote),
            policy=TABLE_S5_POLICY,
            n_strains=len(strains),
            n_positional_strains=len(positional),
            n_records_flagged=sum(strain.n_records for strain in strains),
            n_positional_records_flagged=sum(strain.n_records for strain in positional),
            strains=strains,
            created_at=datetime.now(UTC).isoformat(),
        )
        out = osp.join(self.root, "table_s5_affected_strains.json")
        with open(out, "w") as handle:
            handle.write(flag_file.model_dump_json(indent=2))
        log.info(
            "Hoepfner2014 Table S5 flags: %d strains (%d positional), %d records -> %s",
            flag_file.n_strains,
            flag_file.n_positional_strains,
            flag_file.n_records_flagged,
            out,
        )

    def _write_sourced_values(self) -> None:
        """Write ``preprocess/sourced_values.json``: every hardcoded number's quote."""
        out = osp.join(self.preprocess_dir, "sourced_values.json")
        payload = {
            key: value.model_dump(mode="json")
            for key, value in sorted(SOURCED_VALUES.items())
        }
        with open(out, "w") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        log.info("Hoepfner2014 sourced values -> %s", out)

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
    root = osp.join(data_root, "data/torchcell/env_chemgen_hoepfner2014")
    dataset = EnvChemgenHoepfner2014Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
