# torchcell/datasets/scerevisiae/hillenmeyer2008
# [[torchcell.datasets.scerevisiae.hillenmeyer2008]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/hillenmeyer2008
# Test file: tests/torchcell/datasets/scerevisiae/test_hillenmeyer2008.py
"""Hillenmeyer 2008 "chemical genomic portrait of yeast" (FitDb): env x geno -> fitness.

Hillenmeyer et al. 2008 (Science 320:362; doi:10.1126/science.1150021; citation key
``hillenmeyerChemicalGenomicPortrait2008``) is the genome-scale HIP/HOP fitness
compendium: the heterozygous (HIP) and homozygous (HOP) barcoded deletion collections
competitively grown under ~400 chemical and environmental conditions, read out as a
per-strain fitness defect. It is the earlier, broader companion to the Hoepfner 2014
HIP-HOP atlas.

TWO DATASETS (the readouts are incomparable -- a log-ratio must never be mixed with a
z-score in one ``measurement_type`` -- so HET and HOM are SEPARATE dataset classes,
``HetHillenmeyer2008Dataset`` and ``HomHillenmeyer2008Dataset``, cf. the Smf/Dmf split):

- ``het.ratio_result_nm.pub`` -- HET (HIP) fitness-defect LOG-RATIO, 5984 strain rows x
  726 arrays, ``measurement_type=log2_ratio``. The SOM defines it as
  ``log-ratio_FD = log2(mu_i^c / x_i^t)``, mu_i^c the mean intensity of tag i across the
  matched control arrays and x_i^t its intensity in the treatment, averaged over the up
  and down tags. Positive = fitness defect (the strain is depleted under treatment).
- ``hom.z_result_nm.pub`` -- HOM (HOP) fitness-defect Z-SCORE, 4769 strain rows x 418
  arrays, ``measurement_type=z_score``, ``z_FD = (mu_i^c - x_i^t) / sigma_i^c`` with
  sigma_i^c the SD of tag i across the SAME matched control arrays. No HOM log-ratio
  matrix was archived, which is why the two classes carry different statistics.

REFERENCE. Both scores are defined against a MATCHED CONTROL SET, not against a single
global control, so the reference is emitted per control set rather than once per matrix:
"Each treatment array was compared to an associated control set of no-drug arrays that
matched (a) the deletion pool used in the experiment, (b) the number of growth
generations of the experiment, and (c) the scanner for the experiment (one of two
scanners)." ``het.txt`` / ``hom.txt`` release the per-array control-set assignment and
``het_controls.txt`` / ``hom_controls.txt`` list that set's control arrays, which is the
sample size of the z-score's denominator. The control-set id is stored on
``EnvironmentResponsePhenotype.screen_id``, so two arrays of one condition normalized
against different control sets stay distinct measurements instead of merging.

STRAIN BACKGROUND (#505 G1, schema #507). Records are the strain-resolved family
(``StrainEnvironmentResponseExperiment`` / ``...Reference``). The reference genome is a
``StrainReferenceGenome(strain="BY4743")`` whose ``StrainBackground`` is sourced by the
only in-release statement of the background, the ``hom.txt`` key-file label "synthetic
complete for BY4743", plus the SOM's deferral of the collections to its ref (1), Giaever
2002, and of the growth protocol to its ref (2), Pierce 2006 (neither mirrored). The BY4743
alleles (his3d1/his3d1 leu2d0/leu2d0 LYS2/lys2d0 met15d0/MET15 ura3d0/ura3d0), the mating
type and the parents are stated by no mirrored source, so each is a
``deferred_pending_source_review`` gap naming Brachmann 1998.

GENOTYPE. One record per CONSTRUCTED STRAIN, i.e. per matrix row ``ORF:batch``, never per
gene: "some gene deletions were constructed more than once, in different batches", and
each row is a different physical strain.

- HET: ``HeterozygousDeletionPerturbation`` (one allele replaced by kanMX4, Giaever 2014),
  replacing the old ``EngineeredCopyNumberPerturbation(1 of 2)`` encoding, which asserted
  a working copy at loci where BY4743 is already null. The functional dose is derived
  from the background (``heterozygous_deletion_functional_copies``): HIS3 0, LYS2 and
  MET15 undetermined (which allele the cassette replaced is a typed gap), every other
  gene 1.
- HOM: ``BarcodedKanMxDeletionPerturbation`` with the same cassette and construction
  fields; at the BY marker loci what was physically deleted is a typed gap on
  ``constructed_orf``.
- Both: ``construction=StrainConstruction(batch=<batch>)`` from the row id verbatim,
  ``collection`` = the deletion pool named in the control-set id (``het_04_01_2`` ...),
  barcodes ``None`` with a gap (the tag sequences are in the SGDP tables, not in the
  release).

ORF. ``systematic_gene_name`` and ``perturbed_gene_name`` are the row id's ORF exactly as
the release spells it (the release carries no gene symbol). The loader consults no
genome: no name resolution, no gene universe, no ``ConstructedOrf``. An ORF that a later
SGD release renamed, merged or retired (24 het and 22 hom row ORFs under R64-4-1, among
them non-gene features) is stored under its released spelling like every other row;
relating a released ORF to the current annotation is a modeling-time join, not a loader
decision.

``YDL227C:ctrl_<n>`` (11 rows per arm). The SOM defines no ``ctrl_`` batch; these rows are
stored as the release labels them, a deletion of YDL227C with
``StrainConstruction(batch="ctrl_<n>")`` verbatim, so they stay distinct strains from the
``YDL227C:chr4_3`` row. Hypothesis (untested): they are HO-locus control strains with
distinct tags (Pierce 2006/2007, not mirrored).

ENVIRONMENT -- parsed from each column header
``filename:cond1:conc1:unit1:cond2:conc2:unit2:generations:pool:scanner`` and CROSS-CHECKED
against the key file's ``condition`` column for every array (#505 E1); see
``KEY_HEADER_CONFLICTS`` for the written rule per disagreement. cond1 is classified per
SOM Table S1 and cond2, when present, ALWAYS adds a second small molecule:

- ``<N> degrees C`` -> ``Environment.temperature`` (M2, no perturbation).
- ``pH<x>`` -> ``EnvironmentPhysicalPerturbation(factor=ph)``.
- a named-nutrient drop-out -> the DERIVED MEDIUM ``HILLENMEYER_DROPOUT_MEDIA[label]``;
  a partial drop-out's released level must equal the level that medium records.
- ``no drug irradiated`` -> ``EnvironmentPhysicalPerturbation(factor=radiation)``;
  ``angelicin irradiated`` / ``psoralen irradiated`` -> the compound AND radiation.
- ``minimal media`` -> ``SD`` with ``auxotroph_supplements=None`` and a typed gap (BY4743
  is a His/Leu/Ura auxotroph, so a supplement is implied and never named); the 18 hom
  arrays whose key file reads "synthetic complete for BY4743" are ``SC`` (key wins, see
  the conflict table); ``synthetic complete`` -> ``SC``; ``YP glycerol`` ->
  ``YP_GLYCEROL_LIQUID``.
- everything else -> ``SmallMoleculePerturbation(resolved_compound(label), dose)`` with
  ``solvent=None`` and a typed gap (no vehicle is stated per compound). An IDENTIFIED
  label the curated compound table gives no InChIKey (a public name not yet resolved,
  such as ``AG 1387 (Biomol)`` or ``amphotericin``, or a mixture) keeps the released
  label as ``Compound.name`` and carries the typed ``inchikey`` gap
  ``resolved_compound`` attaches; it is stored, and listed with its record count in
  ``preprocess/compound_identity_gaps.json``. An UNIDENTIFIED label is held out (below).

UNIDENTIFIED PERTURBING AGENTS (database constitution principle 2, owner ruling
2026-10-15: "if you cannot tell me what goes into the flask so to speak, then we have
no hope of modeling!"). The criterion is mechanical: a label is unidentified iff its row
in the pinned compound-identity table has status ``PROPRIETARY`` and the loader's
admission table ``RELEASED_IDENTITIES`` (empty today) has no entry for it. Twelve labels
meet it, each checked against the SOM: the ten ChemDiv codes ``chemical diversity labs
14a``, ``14f``, ``1c10``, ``218``, ``232``, ``246``, ``275``, ``3f2``, ``41``, ``6842``,
plus ``RIL1 (Biomol)`` and ``telomerase inhibitor ix (Biomol)``. SOM Table S1 lists them
only by these codes, and none is among the structures of Table S2, whose caption reads
"Table S2. 316 compounds tested in this study for which the chemical structures were
available. Not included are non-compound experiments such as environmental stresses and
compounds for which the chemical structure was not known or unavailable." (AG 1478
(Biomol) and Tyrphostin AG 957 are drawn there.) Every array dosing one of them is
dropped under ``UNIDENTIFIED_COMPOUND_RULE`` and listed per label, with its headers, key
file conditions, records not written and value cells, in
``preprocess/dropped_records.json`` (``unidentified_compounds_held_out``), beside the
quotes (``unidentified_compound_sources``). Measured on the raw mirror through the
loader's own ``parse_columns``, ``group_columns`` and ``count_records`` (2026-10-15):
HET 12 labels, 19 environments, 19 arrays, 110,449 records and value cells; HOM 10
labels (no RIL1 or telomerase array), 11 environments, 11 arrays, 45,760 records and
value cells. Stored after the hold-out: HET 2,963,683 records, HOM 1,182,916. The
``chemical diversity labs 14a`` array of each arm is the header/key conflict whose key
file says ``Tyrphostin``; the header is what the values are released under, so it is
held out with the key string kept beside it. A label is
admitted by an entry in ``RELEASED_IDENTITIES`` (a ``ReleasedIdentity`` with a name,
SMILES or InChIKey and the ``SourcedValue`` of a public label-to-structure release);
only such a release fills it, never an identity inferred from profile similarity, a
filled entry rebuilds the store, and the ledger's held-out list is the request list.

HEADER VS KEY FILE COMPOUND. On five header/key pairs the key file names a different
compound from the matrix header and the SOM gives no tie-breaker. The array is served
under the HEADER's compound, the column the values are released under
(``KeyHeaderRule.header_served_key_disagrees``); both strings stay in
``preprocess/key_header_checks.json``.

The environment is a ``CultureEnvironment``: the SIGN of the generation count becomes
``pre_culture`` (negative: ``frozen_stock``; positive: a YPD log-phase pre-culture to
OD600 2.0, ~10 generations, both quoted from the SOM), ``duration_generations`` keeps the
magnitude, and ``culture_format`` is gapped (vessel and aeration are deferred to Pierce
2006). A ``0gen`` array is stored with ``duration_generations=0.0`` and
``pre_culture=None`` with a typed ``not_reported_by_primary`` gap: a count of 0 carries no
sign, so it states no pre-culture. TEMPERATURE IS NOT DEFAULTED: a typed gap naming
Pierce 2006.

DOSES are canonicalized within the molar family before they become part of the
environment identity (``1.5 m`` and ``1.5e+06 um`` are one dose).

NOT STORED (counted per rule in ``preprocess/dropped_records.json`` and
``preprocess/dropped_strains.json``, with a per-cell accounting that must sum):

- STRAIN, author withdrawal: the homozygous PDR5 row (``YOR153W``), which the SOM says
  "did not have the correct gene deleted".
- ARRAYS, no typed carrier in the schema (needs schema): the ``37c, 45c`` heat-shock cycle
  (``PhysicalFactor`` has no heat-shock member, and a steady 45 C would be false), and a
  dose on a media swap whose agent is named nowhere (a ``SmallMoleculePerturbation``
  needs a compound name).
- ARRAYS, not representable: an unidentified perturbing agent
  (``UNIDENTIFIED_COMPOUND_RULE``, above).

REPLICATE AGGREGATION. Within one (strain row, environment, control set) group each array
contributes ONE value and the record's value is the mean over arrays with the sample SD
across arrays as the uncertainty (``n_samples`` = arrays).

SUSPICIOUS BATCHES. The SOM names ten construction batches whose strains share a
secondary genotype; the ORFs with a row in them are written to
``preprocess/suspicious_batch_strains.json``, and the batch is now also on every record's
``construction``.

DATA SOURCE / RAW MIRROR. Live FitDb portals are DNS-dead and the Science SI 403s; the
Stanford static supplement survives only in the Internet Archive. The seven files this
loader and its provenance consume are deposited at
``$DATA_ROOT/torchcell-raw/hillenmeyerChemicalGenomicPortrait2008/data/`` with a
``manifest.json`` recording, per file, the exact ``web.archive.org/web/<TS>id_/`` URL,
the ``direct_url`` retriever, the sha256 and the retrieval date; ``download()`` links them
into ``raw/`` and verifies every sha256 against that manifest. See
``[[fitdb-hillenmeyer2008-wayback-data-source]]``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import os.path as osp
import re
import shutil
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import (
    CompoundResolutionStatus,
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.media import (
    HILLENMEYER_DROPOUT_MEDIA,
    HILLENMEYER_PARTIAL_DROPOUT_LEVELS,
    SC,
    SD,
    YP_GLYCEROL_LIQUID,
    YPD_LIQUID,
)
from torchcell.datamodels.schema import (
    AssayType,
    BarcodedKanMxDeletionPerturbation,
    Compound,
    Concentration,
    ConcentrationUnit,
    ConstructedOrf,
    CultureEnvironment,
    DoseBasis,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    EukaryoticStrainBackground,
    Experiment,
    ExperimentReference,
    Genotype,
    HeterozygousDeletionPerturbation,
    MeasurementType,
    Media,
    OrfHistoryRelation,
    PhysicalFactor,
    PreCulture,
    PreCultureSource,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainConstruction,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    StrainReferenceGenome,
    Temperature,
    UncertaintyType,
)
from torchcell.datamodels.strain_background import (
    BRACHMANN_1998,
    GIAEVER_2002,
    KANMX4_CASSETTE,
    pending_source_review,
    standard_background,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.hoepfner2014 import ReleasedIdentity
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    SourceCheck,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "hillenmeyerChemicalGenomicPortrait2008"
DOI = "10.1126/science.1150021"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
#: sha256 of the mirrored OCR supplement every quote below is a literal substring of.
SOM_SHA256 = "cf4759f00083de78dd953b12dd66d4360a2f645321305f768403f26b451c1df0"
#: ``si/si1.md`` of the same library key: OCR of the print pages, which carry the MAIN
#: Science article (lines 31-113) between two unrelated articles.
MAIN_ARTICLE_URI = "si/si1.md"
MAIN_ARTICLE_SHA256 = "d2c544882fdd09b49aa7555bb5fb75b738263540eeb920b944fcd216597eaa34"
#: The raw-mirror files a quote is read from (``data/<name>`` under ``RAW_DIR_REL``);
#: the build re-checks each against the mirror manifest, these pin the quotes.
RAW_SHA256: dict[str, str] = {
    "hom.txt": "312f76309547ae2e2870029a2ca1f3580176ba0b97a24308eeb5afe17de9131b",
    "het.txt": "65bd59fcc87e89ebfbeca90658f4a9111997a461ed0757ebb3a3e08e2901f21a",
}
#: The SOM's declared deferral target for the growth protocol; NOT mirrored.
PIERCE2006 = "pierceGenomewideAnalysisBarcoded2006"

# Wayback-archived Stanford supplement. The timestamps are per file (the crawler hit
# them minutes apart) and come from the CDX index; re-running each URL today reproduces
# the deposited bytes exactly, which is what SOURCE_CHECKED_AT records.
WAYBACK_BASE = "http://chemogenomics.stanford.edu/supplements/global/download/data"
WAYBACK_TIMESTAMPS: dict[str, str] = {
    "het.ratio_result_nm.pub": "20151207003548",
    "hom.z_result_nm.pub": "20151207003024",
    "het.txt": "20151207063659",
    "hom.txt": "20151207011015",
    "het_controls.txt": "20151207003758",
    "hom_controls.txt": "20151207004537",
    "README_CEL_files.txt": "20151207004902",
}
#: Date the bytes were first retrieved into the dev tree (memory note
#: ``fitdb-hillenmeyer2008-wayback-data-source``; the staged files carry that mtime).
RAW_RETRIEVED_AT = "2026-07-11"

_SOM_PROVENANCE = Provenance(
    source_uri="paper.md", citation_key=CITATION_KEY, sha256=SOM_SHA256
)
_PIERCE_PROVENANCE = Provenance(
    source_uri="paper.pdf",
    citation_key=PIERCE2006,
    sha256="not-mirrored",
    method="S. E. Pierce et al., Nat Methods 3, 601 (Aug, 2006) -- ref (2) of the "
    "Hillenmeyer SOM, the declared source of the growth protocol",
)

# Verbatim SOM sentences (literal substrings of the pinned paper.md).
QUOTE_CONTROL_SET = (
    "Each treatment array was compared to an associated control set of no-drug arrays "
    "that matched (a) the deletion pool used in the experiment, (b) the number of growth "
    "generations of the experiment"
)
QUOTE_CONTROL_COUNT = (
    "combinations yielded 34 control sets, each composed of between 3 and 48 control "
    "arrays"
)
QUOTE_GENERATIONS_SIGN = (
    "A negative sign preceding the number indicates that the pool was taken directly "
    "from the freezer, thawed, diluted and grown in the condition."
)
QUOTE_PRE_CULTURE = (
    "Absence of a negative sign indicates that the pool was thawed, inoculated into YPD "
    "and grown overnight until log phase (OD600= 2.0)(~10 generations of recovery) "
    "before drug addition."
)
QUOTE_GENERATIONS_DEFINITION = (
    "The number of generations for which the pool was grown in drug."
)
QUOTE_PRESCREEN = (
    "those compounds/treatments that produced a measurable $1 0 { - } 1 5 \\%$ inhibition "
    "of wildtype growth (IC-15) were chosen for further full-genome screening"
)
QUOTE_PROTOCOL_DEFERRAL = (
    "The protocol for pooled, competitive growth of the deletion strains, genomic DNA "
    "purification and PCR, and tag hybridization follows Ref. (2)."
)
QUOTE_COLLECTIONS_DEFERRAL = "The deletion collections have been described (1)."
QUOTE_REF1 = "1. G. Giaever et al., Nature 418, 387 (Jul 25, 2002)."
QUOTE_REF2 = "2. S. E. Pierce et al., Nat Methods 3, 601 (Aug, 2006)."
QUOTE_POOLS = (
    "We used five pools over the course of this study, named according to date:"
)
QUOTE_TAGS = "Each gene deletion cassette has four oligonucleotide barcodes"
QUOTE_BATCH_IDS = (
    "All strains associated with batch $\\mathrm { c h r l } _ { - 1 }$ were made by a "
    "single laboratory or group. The gene names in our downloadable data include the "
    "strain-batch number."
)
QUOTE_RECONSTRUCTED = (
    "The number of strains exceeds the number of genes because some gene deletions were "
    "constructed more than once, in different batches."
)
QUOTE_SUSPICIOUS = (
    "we noted a few abnormally strong clusters whose genes were functionally unrelated "
    "but the strains in each cluster originated from a common deletion collection "
    "generation batch. We believe this is due to unrelated artifacts, e.g. a secondary "
    "site mutation in the parent strain used to make each batch."
)
QUOTE_SUSPICIOUS_COUNT = (
    "647 strains belonged to these suspicious batches, and we exclude these strains from "
    "most analyses."
)
QUOTE_SUSPICIOUS_HOM = "The homozygous strains did not show this pattern."
QUOTE_ARRAY_TYPE = (
    "The array type for all experiments was TAG3 (Affymetrix GenFlex Tag Array, Part No. "
    "510389)."
)
QUOTE_WRONG_GENE = (
    "We identified strains that, upon PCR confirmation experiments, did not have the "
    "correct gene deleted."
)
QUOTE_PDR5 = (
    "Unfortunately, one of the strains in this set was the homozygous deletion strain of "
    "PDR5, the known multidrug resistance efflux transporter."
)
QUOTE_TABLE_S1_PH = "high pH pH7.5, pH8"
QUOTE_TABLE_S1_MEDIA = (
    "media change YP glycerol, minimal media, sorbitol, synthetic complete"
)
#: Main article (si/si1.md), the only sentence on the collections' ploidy.
QUOTE_DIPLOID = "The diploid yeast deletion collections comprise"
#: hom.txt, the only in-release statement of the strain background (27 arrays).
QUOTE_KEY_BY4743 = "synthetic complete for BY4743"
#: SOM Table S2 caption: the structures the SOM releases, and what it leaves out.
QUOTE_TABLE_S2 = (
    "Table S2. 316 compounds tested in this study for which the chemical structures "
    "were available. Not included are non-compound experiments such as environmental "
    "stresses and compounds for which the chemical structure was not known or "
    "unavailable."
)
#: SOM Table S1 ("other"), the ChemDiv library codes as the OCR carries them.
QUOTE_TABLE_S1_CHEMDIV = (
    "chemical diversity labs 14a, chemical diversity labs 14f, chemical diversity labs "
    "1c10, chemical diversity labs 218, chemical diversity labs 232, chemical diversity "
    "labs 246, chemical diversity labs 275, chemical diversity labs 3f2,"
)
#: SOM Table S1, the Biomol telomerase inhibitor as the OCR carries it.
QUOTE_TABLE_S1_TELOMERASE = "telomerase inhibitor ix (Biomol)"

#: The ten construction batches the SOM names as carrying a shared secondary genotype.
SUSPICIOUS_BATCHES: tuple[str, ...] = (
    "chr00_1",
    "chr3_1",
    "chr00_5",
    "chr13_4",
    "chr12_4",
    "chr13_1b",
    "chr4_4",
    "chr13_5",
    "chr13_2",
    "chr7_5",
)


def _som_sv(value: Any, quote: str, note: str | None = None) -> SourcedValue:
    """A value sourced to the pinned SOM OCR (``paper.md``)."""
    return SourcedValue(value=value, provenance=_SOM_PROVENANCE, quote=quote, note=note)


def _main_sv(value: Any, quote: str, note: str | None = None) -> SourcedValue:
    """A value sourced to the main Science article OCR (``si/si1.md``)."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=MAIN_ARTICLE_URI,
            citation_key=CITATION_KEY,
            sha256=MAIN_ARTICLE_SHA256,
        ),
        quote=quote,
        note=note,
    )


def _raw_sv(value: Any, name: str, quote: str, note: str | None = None) -> SourcedValue:
    """A value sourced to a raw-mirror release file (``data/<name>`` of the raw key)."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=f"data/{name}",
            citation_key=CITATION_KEY,
            sha256=RAW_SHA256[name],
            method=f"raw mirror $DATA_ROOT/{RAW_DIR_REL}/data/{name}",
        ),
        quote=quote,
        note=note,
    )


#: Every value this loader hardcodes from the SOM, the main article or the release.
SOURCED_VALUES: dict[str, SourcedValue] = {
    "background_strain": _raw_sv(
        "BY4743",
        "hom.txt",
        QUOTE_KEY_BY4743,
        note="the only place in the SOM, the main article or the release that names the "
        "background: 27 hom.txt arrays carry this condition label; het.txt names no "
        "strain. The heterozygous pool is the same YKO diploid background by the SOM's "
        "deferral to its ref (1), Giaever 2002 (not mirrored)",
    ),
    "diploid": _main_sv(
        "diploid",
        QUOTE_DIPLOID,
        note="both collections are diploid (main article, refs 2-3 = Winzeler 1999, "
        "Giaever 2002)",
    ),
    "collections_deferral": _som_sv(
        "Giaever 2002",
        QUOTE_COLLECTIONS_DEFERRAL,
        note="ref (1) of the SOM, the construction of the collections; not mirrored",
    ),
    "ref1": _som_sv("doi:10.1038/nature00935", QUOTE_REF1),
    "protocol_deferral": _som_sv(
        "Pierce 2006",
        QUOTE_PROTOCOL_DEFERRAL,
        note="ref (2) of the SOM, the pooled-growth protocol; not mirrored",
    ),
    "ref2": _som_sv("Pierce 2006 Nat Methods 3:601", QUOTE_REF2),
    "pools": _som_sv(
        "het_04_01_02, het_09_02, het_06_03, hom_05_01, hom_09_02",
        QUOTE_POOLS,
        note="the record's collection is the pool token verbatim from the control-set "
        "id (which spells the first pool 'het_04_01_2')",
    ),
    "construction_batch": _som_sv("chrX_Y", QUOTE_BATCH_IDS),
    "tags": _som_sv(
        "uptag + downtag",
        QUOTE_TAGS,
        note="the tags exist; their sequences are not in the release, so the record's "
        "barcode fields are gapped",
    ),
    "pre_culture_frozen": _som_sv(
        PreCultureSource.frozen_stock, QUOTE_GENERATIONS_SIGN
    ),
    "pre_culture_log_phase": _som_sv(
        {"medium": "YPD", "od600_at_transfer": 2.0, "generations": 10.0},
        QUOTE_PRE_CULTURE,
        note="'~10 generations' is approximate as written; stored as 10.0",
    ),
    "generations_definition": _som_sv(
        "generations in the condition",
        QUOTE_GENERATIONS_DEFINITION,
        note="a 0gen array is stored with duration_generations 0.0; its count carries "
        "no sign, so its pre-culture is a typed gap",
    ),
    "pdr5_wrong_strain": _som_sv(
        "YOR153W (hom)", QUOTE_PDR5, note=f"preceded by: {QUOTE_WRONG_GENE!r}"
    ),
    "table_s1_ph": _som_sv(
        [7.5, 8.0],
        QUOTE_TABLE_S1_PH,
        note="Table S1 lists only the high-pH conditions; no low pH appears anywhere in "
        "the SOM",
    ),
    "table_s1_media": _som_sv(
        ["YP glycerol", "minimal media", "sorbitol", "synthetic complete"],
        QUOTE_TABLE_S1_MEDIA,
    ),
    "sc_for_by4743": _raw_sv(
        "SC",
        "hom.txt",
        QUOTE_KEY_BY4743,
        note="the key file names the medium of 18 hom arrays whose header reads "
        "'minimal media' (key wins, KEY_HEADER_CONFLICTS)",
    ),
    "table_s2_structures": _som_sv(
        "316 released structures; a compound outside Table S2 has none",
        QUOTE_TABLE_S2,
        note="Table S2 is the SOM's only release of structures (page images, read "
        "against the mirrored images/ of this key). None of the twelve held-out labels "
        "is among its 316 drawings, while AG 1478 (Biomol) and Tyrphostin AG 957 are, so "
        "the SOM's own vendor-labelled compounds are drawn when their structure was "
        "known",
    ),
    "table_s1_chemdiv_codes": _som_sv(
        "chemical diversity labs <code>",
        QUOTE_TABLE_S1_CHEMDIV,
        note="Table S1 lists the ChemDiv ('chemical diversity labs') compounds only by "
        "library code, under 'other'. The OCR stops after 3f2; the PDF (paper.pdf, "
        "sha256 eb052a3fd164af497a6365f84a57a452a321545c02340a9f528c1f97d2ae63f0) "
        "continues 'chemical diversity labs 41, chemical diversity labs 6842' and lists "
        "'antiinflammatory RIL1 (Biomol)', which the OCR drops",
    ),
    "table_s1_telomerase_inhibitor": _som_sv(
        "telomerase inhibitor ix (Biomol)",
        QUOTE_TABLE_S1_TELOMERASE,
        note="a vendor product designation (Biomol, inhibitor number ix) with no "
        "released name or structure; absent from Table S2",
    ),
}
_PRE_CULTURE_FROZEN = SOURCED_VALUES["pre_culture_frozen"]
_PRE_CULTURE_LOG_PHASE = SOURCED_VALUES["pre_culture_log_phase"]

# --------------------------------------------------------------------------- #
# Matrices
# --------------------------------------------------------------------------- #
UNITS_HET = (
    "HIP fitness-defect log-ratio, log2(mean control intensity / treatment intensity) "
    "averaged over the up and down tags (Hillenmeyer 2008 scoring method 1); positive = "
    "fitness defect. Control = the matched no-drug control set named in screen_id "
    "(same deletion pool, generation count and scanner). Mean over the replicate arrays "
    "of that control set for ONE constructed strain (one matrix row ORF:batch)"
)
UNITS_HOM = (
    "HOP fitness-defect z-score, (mean control intensity - treatment intensity) / SD of "
    "the control intensities (Hillenmeyer 2008 scoring method 2); positive = fitness "
    "defect. Control = the matched no-drug control set named in screen_id (same deletion "
    "pool, generation count and scanner). Mean over the replicate arrays of that control "
    "set for ONE constructed strain (one matrix row ORF:batch)"
)


class _MatrixSpec(BaseModel):
    """One released score matrix and everything that is constant across it."""

    key: str
    filename: str
    keyfile: str
    controls_file: str
    measurement_type: MeasurementType
    collection: str
    units: str


MATRICES: dict[str, _MatrixSpec] = {
    "het": _MatrixSpec(
        key="het",
        filename="het.ratio_result_nm.pub",
        keyfile="het.txt",
        controls_file="het_controls.txt",
        measurement_type=MeasurementType.log2_ratio,
        collection="heterozygous",
        units=UNITS_HET,
    ),
    "hom": _MatrixSpec(
        key="hom",
        filename="hom.z_result_nm.pub",
        keyfile="hom.txt",
        controls_file="hom_controls.txt",
        measurement_type=MeasurementType.z_score,
        collection="homozygous",
        units=UNITS_HOM,
    ),
}

#: The strain name every record's reference genome states (``SOURCED_VALUES``).
BACKGROUND_STRAIN = "BY4743"

_MOLAR_FAMILY: tuple[tuple[str, ConcentrationUnit, Decimal], ...] = (
    ("m", ConcentrationUnit.molar, Decimal(1)),
    ("mm", ConcentrationUnit.millimolar, Decimal("1e-3")),
    ("um", ConcentrationUnit.micromolar, Decimal("1e-6")),
    ("nm", ConcentrationUnit.nanomolar, Decimal("1e-9")),
)
_MOLAR_BY_TOKEN = {token: (unit, factor) for token, unit, factor in _MOLAR_FAMILY}
_OTHER_UNITS = {
    "ug/ml": ConcentrationUnit.ug_per_ml,
    "%": ConcentrationUnit.percent_v_v,
}

_TEMP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*degrees?\s*c", re.I)
_TEMP_CYCLE_RE = re.compile(r"^\s*(\d+)\s*c\s*,\s*(\d+)\s*c", re.I)
_PH_RE = re.compile(r"^\s*ph\s*([\d.]+)\s*$", re.I)
_IRRADIATED_SUFFIX = " irradiated"
_RADIATION_ONLY = "no drug irradiated"
_MINIMAL_MEDIA = "minimal media"
#: condition label -> the shared medium it denotes (Table S1 "media change").
_MEDIA_SWAPS: dict[str, Media] = {
    _MINIMAL_MEDIA: SD,
    "synthetic complete": SC,
    "yp glycerol": YP_GLYCEROL_LIQUID,
}
# Missing-value tokens in the score matrices (upper-cased) -- skipped when aggregating.
_MISSING = {"", "NA", "NAN", "NULL"}

#: Array (column) rules: a released array with no typed carrier in the schema. Both
#: need a schema change to be stored (no new member or leaf is added here).
DROP_HEAT_SHOCK_CYCLE = "heat_shock_cycle_not_representable"
DROP_UNNAMED_MEDIA_AGENT = "unnamed_agent_dosed_into_a_media_swap"

#: Strain (row) rule, written to ``preprocess/dropped_strains.json``: the one author
#: withdrawal in the release.
DROP_WRONG_GENE_DELETED = "som_wrong_gene_deleted"

#: Array rule for an unidentified perturbing agent (database constitution principle 2,
#: owner ruling 2026-10-15): the same rule string Hoepfner 2014 uses.
UNIDENTIFIED_COMPOUND_RULE = (
    "compound_has_no_released_name_or_structure_not_representable"
)

UNIDENTIFIED_COMPOUND_POLICY = (
    "a compound label whose pinned compound-identity row has status PROPRIETARY (a "
    "vendor or library catalog code: the ChemDiv 'chemical diversity labs' codes and "
    "two Biomol products) and no RELEASED_IDENTITIES entry is held out: the SOM lists it "
    "only by its code in Table S1 and releases no structure for it in Table S2, so what "
    "went into the flask cannot be stated and the experiment cannot be represented. "
    "Every array dosing it is HELD OUT, never stored under the bare label with an "
    "inchikey gap; the ledger lists each label with its arrays and record counts, and "
    "the label is admitted the day a public label-to-structure release fills "
    "RELEASED_IDENTITIES. Any other label (a public name the table has not resolved, "
    "e.g. AG 1387 (Biomol), amphotericin, or a mixture) is identified and stays stored "
    "with its typed inchikey gap"
)

#: The ADMISSION TABLE, keyed by the released label: EMPTY today. A label is stored iff
#: its identity-table status is not PROPRIETARY or it has an entry here, and an entry's
#: name, SMILES and InChIKey build the stored ``Compound``. Only a public
#: label-to-structure release (ChemDiv or Biomol, or a later paper), mirrored and
#: sha256-pinned like any other artifact, fills it; an identity inferred from profile
#: similarity is a hypothesis and never goes here. A filled entry changes the stored
#: records, so it rebuilds the store. The ledger's ``unidentified_compounds_held_out``
#: (every held-out label with its arrays and records) is the request list.
RELEASED_IDENTITIES: dict[str, ReleasedIdentity] = {}

DROP_RULES: dict[str, str] = {
    DROP_HEAT_SHOCK_CYCLE: "needs schema: '37c, 45c' is a heat-shock cycle; "
    "PhysicalFactor has no heat_shock member (adding one changes a served class) and a "
    "steady 45 C would be false",
    DROP_UNNAMED_MEDIA_AGENT: "needs schema: a dose is given but the agent is named "
    "nowhere in the SOM or the release, and a SmallMoleculePerturbation needs a "
    "compound name",
    UNIDENTIFIED_COMPOUND_RULE: "not representable: the dosed agent is a vendor or "
    "library code with no released name or structure (UNIDENTIFIED_COMPOUND_POLICY)",
}
STRAIN_DROP_RULES: dict[str, str] = {
    DROP_WRONG_GENE_DELETED: f"author withdrawal, the SOM (paper.md:332): "
    f"{QUOTE_WRONG_GENE!r} {QUOTE_PDR5!r}; the homozygous PDR5 row would assert a "
    "pdr5/pdr5 genotype the authors say the strain does not have"
}
#: arm -> released ORF -> strain drop rule (whole-ORF exclusions named by the SOM).
EXCLUDED_SOURCE_ORFS: dict[str, dict[str, str]] = {
    "het": {},
    "hom": {"YOR153W": DROP_WRONG_GENE_DELETED},
}


# --------------------------------------------------------------------------- #
# Header vs key file (#505 E1)
# --------------------------------------------------------------------------- #
class KeyHeaderOutcome(StrEnum):
    """How an array's matrix-header condition relates to its key-file condition."""

    agree = "agree"
    spelling_variant = "spelling_variant"
    conflict = "conflict"


class KeyHeaderRule(StrEnum):
    """The written rule that resolves one class of header/key conflict.

    - ``header_wins_table_s1``: pH. The key file says ``ph4``, the header ``pH7.5`` /
      ``pH8``; SOM Table S1 lists only "high pH pH7.5, pH8", so the header is served.
    - ``key_wins_strain_medium``: the header says ``minimal media``, the key file
      "synthetic complete for BY4743"; the key names a medium specific to the strain
      (the only in-release mention of it), so the array is served on ``SC``.
    - ``header_served_key_disagrees``: the two name different compounds and the SOM
      gives no tie-breaker; the array is served under the HEADER's compound, the column
      the values are released under, and the key file's string is kept beside it in
      ``preprocess/key_header_checks.json``.
    """

    header_wins_table_s1 = "header_wins_table_s1"
    key_wins_strain_medium = "key_wins_strain_medium"
    header_served_key_disagrees = "header_served_key_disagrees"


#: (key-file condition, header cond1), both lower-cased: one compound or condition
#: spelled two ways. Each pair was read off the release; a new disagreement raises.
KEY_HEADER_SPELLING_VARIANTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("ipraflavone", "ipriflavone"),
        ("ansamitosin", "ansamitocin"),
        ("p-aminobenzoic acid", "paba"),
        ("sodium fluoride", "naf"),
        ("parkinsons peptide", "parkinson-inducing peptide"),
        ("mit", "mitomycin c"),
        ("cadmium chloride", "cdcl2"),
        ("olic acid drop-out", "folic acid drop-out"),
        ("no drug minimal media", "minimal media"),
        ("synthetic complete for by4743", "synthetic complete"),
        # the key names the congener, the header the compound family; the header's
        # label is served (with the inchikey gap if the compound table lacks it)
        ("dihydromotuporamine c", "motuporamine"),
    }
)
#: (key-file condition, header cond1), lower-cased -> the rule that resolves it.
KEY_HEADER_CONFLICTS: dict[tuple[str, str], KeyHeaderRule] = {
    ("ph4", "ph8"): KeyHeaderRule.header_wins_table_s1,
    ("ph4", "ph7.5"): KeyHeaderRule.header_wins_table_s1,
    ("synthetic complete for by4743", "minimal media"): (
        KeyHeaderRule.key_wins_strain_medium
    ),
    ("bisphenol s", "bathophenanthroline disulfonate"): (
        KeyHeaderRule.header_served_key_disagrees
    ),
    ("sodium sulfate", "sodium arsenite"): KeyHeaderRule.header_served_key_disagrees,
    ("colchiceine", "colchicine"): KeyHeaderRule.header_served_key_disagrees,
    ("tyrphostin", "chemical diversity labs 14a"): (
        KeyHeaderRule.header_served_key_disagrees
    ),
    ("phosphatase inhibitor", "ptp2"): KeyHeaderRule.header_served_key_disagrees,
}
_VOLUME_SUFFIX_RE = re.compile(r",\s*\d+ul total up/dn$")
_CHEMDIV_RE = re.compile(r"^chemdiv (\S+)$")


class KeyHeaderCheck(BaseModel):
    """The cross-check of one array: both strings, the outcome and the rule applied."""

    filename: str
    header_condition: str
    key_condition: str
    outcome: KeyHeaderOutcome
    rule: KeyHeaderRule | None = None


def _normalized_header(cond1: str) -> str:
    """Header cond1 without the parts the key file never spells (lower-cased)."""
    low = cond1.strip().lower()
    low = _VOLUME_SUFFIX_RE.sub("", low)
    if low.endswith(_IRRADIATED_SUFFIX) and low != _RADIATION_ONLY:
        low = low[: -len(_IRRADIATED_SUFFIX)].strip()
    return low


def _normalized_key(condition: str) -> str:
    """Key-file condition, ``chemdiv N`` expanded, volume suffix cut (lower-cased)."""
    low = _VOLUME_SUFFIX_RE.sub("", condition.strip().lower())
    chemdiv = _CHEMDIV_RE.match(low)
    return f"chemical diversity labs {chemdiv.group(1)}" if chemdiv else low


def check_key_header(filename: str, cond1: str, key_condition: str) -> KeyHeaderCheck:
    """Cross-check one array's header cond1 against its key-file condition.

    The key file lists only the FIRST agent of a double-drug array, so cond1 is the
    comparison. Exact (case-insensitive) equality after the normalizations above is
    agreement; a listed spelling pair is a variant; a listed conflict carries its rule.
    Any other disagreement raises: an unexamined contradiction in the release must not
    be served silently.
    """
    header, key = _normalized_header(cond1), _normalized_key(key_condition)
    pair = (key, header)
    if header == key:
        outcome, rule = KeyHeaderOutcome.agree, None
    elif pair in KEY_HEADER_SPELLING_VARIANTS:
        outcome, rule = KeyHeaderOutcome.spelling_variant, None
    elif pair in KEY_HEADER_CONFLICTS:
        outcome, rule = KeyHeaderOutcome.conflict, KEY_HEADER_CONFLICTS[pair]
    else:
        raise ValueError(
            f"array {filename!r}: header condition {cond1!r} and key-file condition "
            f"{key_condition!r} disagree and no written rule covers the pair"
        )
    return KeyHeaderCheck(
        filename=filename,
        header_condition=cond1.strip(),
        key_condition=key_condition.strip(),
        outcome=outcome,
        rule=rule,
    )


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_root() -> str:
    """``DATA_ROOT`` from the repo-root ``.env``."""
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/hillenmeyerChemicalGenomicPortrait2008``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def raw_relpaths() -> dict[str, str]:
    """Mirror-relative path of every deposited file, keyed by its released name."""
    return {name: f"data/{name}" for name in WAYBACK_TIMESTAMPS}


def wayback_url(name: str) -> str:
    """The exact archived URL whose bytes the mirror holds."""
    return f"http://web.archive.org/web/{WAYBACK_TIMESTAMPS[name]}id_/{WAYBACK_BASE}/{name}"


def deposit_raw_mirror(
    source_dir: str | Path,
    *,
    data_root: str | None = None,
    verify_source: bool = False,
    retrieved_at: str = RAW_RETRIEVED_AT,
) -> Path:
    """Deposit the Wayback-recovered supplement into the raw mirror with a manifest.

    ``source_dir`` holds the already-retrieved bytes (the dev staging directory). Copies
    are idempotent by sha256: an existing mirror file with a different hash raises rather
    than being overwritten. With ``verify_source`` the recorded retrieval is RE-RUN and
    each file's ``last_check`` records whether the archive still yields the same bytes;
    a mismatch raises, because a silently different archive copy is exactly what the
    sha256 anchor exists to catch.
    """
    root = raw_mirror_dir(data_root)
    rel = raw_relpaths()
    files: list[ArtifactRecord] = []
    for name, relpath in rel.items():
        src = Path(source_dir) / name
        if not src.exists():
            raise RuntimeError(f"required raw file missing from {source_dir}: {name}")
        sha = _sha256(src)
        check: SourceCheck | None = None
        if verify_source:
            from torchcell.literature.retrieve import direct_url

            produced = hashlib.sha256(direct_url(wayback_url(name))).hexdigest()
            if produced != sha:
                raise RuntimeError(
                    f"{name}: the archived URL now yields sha256 {produced}, the "
                    f"deposited bytes are {sha}"
                )
            check = SourceCheck(
                checked_at=date.today().isoformat(),
                produced_sha256=produced,
                matches=True,
            )
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != sha:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=sha,
                source=wayback_url(name),
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=wayback_url(name),
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"url": wayback_url(name)},
                    sha256=sha,
                    retrieved_at=retrieved_at,
                    last_check=check,
                ),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title="The chemical genomic portrait of yeast: uncovering a phenotype for all genes",
        files=files,
        si_data_sources=[
            "http://web.archive.org/cdx/search/cdx?url=chemogenomics.stanford.edu/"
            "supplements/global*&output=text&fl=original,timestamp,statuscode,length",
            f"{WAYBACK_BASE}/ (Internet Archive; the live host is DNS-dead)",
        ],
        si_expected=[
            "het.ratio_result_nm.pub (HIP log-ratio matrix)",
            "hom.z_result_nm.pub (HOP z-score matrix)",
            "het.txt / hom.txt (array -> condition -> control set)",
            "het_controls.txt / hom_controls.txt (control-set membership)",
            "README_CEL_files.txt (key-file semantics)",
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


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the Hillenmeyer raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #
def canonical_concentration(conc: str, unit: str) -> Concentration:
    """Parse one released dose, canonicalizing the molar family exactly.

    The release spells one molar dose two ways in five places (``1.5 m`` vs
    ``1.5e+06 um``), which would otherwise become two environments for one condition.
    Conversion runs in ``Decimal`` (so ``1.5e+06 uM`` and ``1.5 M`` land on the identical
    float) and picks the largest unit leaving the value >= 1. A dose with no unit is a
    ``DoseBasis.fixed`` dose, which is what the release means by an empty unit field.
    """
    value, token = conc.strip(), unit.strip().lower()
    if not value or not token:
        return Concentration(basis=DoseBasis.fixed)
    if token in _OTHER_UNITS:
        return Concentration(value=float(value), unit=_OTHER_UNITS[token])
    if token not in _MOLAR_BY_TOKEN:
        raise ValueError(f"unrecognized concentration unit {unit!r} (value {conc!r})")
    molar = Decimal(value) * _MOLAR_BY_TOKEN[token][1]
    for _, enum_unit, factor in _MOLAR_FAMILY:
        scaled = molar / factor
        if scaled >= 1:
            return Concentration(value=float(scaled), unit=enum_unit)
    return Concentration(
        value=float(molar / Decimal("1e-9")), unit=ConcentrationUnit.nanomolar
    )


class ConditionParse(BaseModel):
    """One parsed column header: its typed environment, or the rule that drops it.

    ``unsupplemented_minimal`` marks the ``minimal media`` arrays, whose environment
    carries a typed gap on ``auxotroph_supplements``.
    """

    model_config = {"arbitrary_types_allowed": True}

    media: Media
    temperature_c: float | None
    perturbations: list[SmallMoleculePerturbation | EnvironmentPhysicalPerturbation]
    unsupplemented_minimal: bool = False
    drop_reason: str | None = None
    drop_detail: str | None = None


def _solvent_gap() -> ProvenanceGap:
    """The typed absence of the vehicle every dosed compound shares."""
    return ProvenanceGap(
        field="solvent",
        reason=ProvenanceGapReason.deferred_pending_source_review,
        looked_in=_SOM_PROVENANCE,
        resolve_with=_PIERCE_PROVENANCE,
        note="the SOM names no vehicle per compound and no vehicle fraction; the "
        "release's control-set ids end '::YPD::dmso::0' (the matched no-drug arrays), "
        "which suggests DMSO as the compound vehicle, but that is not stated per "
        "compound (inorganic salts such as NaCl at 600 mM are unlikely to be in DMSO), "
        "so no Solvent is asserted",
    )


def compound_is_identified(label: str) -> bool:
    """Whether a released compound label names identified material.

    ``UNIDENTIFIED_COMPOUND_RULE``: a label is unidentified iff its pinned
    compound-identity row has status ``PROPRIETARY`` (a vendor or library code the SOM
    releases no structure for) and ``RELEASED_IDENTITIES`` has no entry for it. Every
    other label, a gapped public name or a mixture included, is identified.
    """
    if label in RELEASED_IDENTITIES:
        return True
    status = resolve_compound_identity(name=label).status
    return status is not CompoundResolutionStatus.PROPRIETARY


def _compound(label: str) -> Compound:
    """The compound of a released label: the shared table, or an admission entry.

    A ``RELEASED_IDENTITIES`` entry supplies the name and SMILES the label lacks (its
    SMILES yields an RDKit InChIKey), and its released InChIKey fills a key the resolver
    could not, retiring that gap. Without an entry the label resolves as released.
    """
    released = RELEASED_IDENTITIES.get(label)
    if released is None:
        return resolved_compound(label)
    compound = resolved_compound(
        released.name if released.name is not None else label,
        smiles=released.smiles,
        derive_from_smiles=released.smiles is not None,
    )
    if released.inchikey is not None and compound.inchikey is None:
        compound = compound.model_copy(
            update={
                "inchikey": released.inchikey,
                "provenance_gaps": [
                    gap for gap in compound.provenance_gaps if gap.field != "inchikey"
                ],
            }
        )
    return compound


def _small_molecule(label: str, conc: str, unit: str) -> SmallMoleculePerturbation:
    """One dosed compound, resolved through the shared table by its SOURCE label."""
    return SmallMoleculePerturbation(
        compound=_compound(label),
        concentration=canonical_concentration(conc, unit),
        solvent=None,
        provenance_gaps=[_solvent_gap()],
    )


def _check_dropout_level(label: str, conc: str, unit: str) -> None:
    """A partial drop-out's released level must be the one its medium records.

    Every other drop-out carries no dose (or the ``0`` of the vitamin control); a
    nonzero dose on one would be an edit the derived medium does not represent.
    """
    released = (conc.strip(), unit.strip())
    if label in HILLENMEYER_PARTIAL_DROPOUT_LEVELS:
        if released != HILLENMEYER_PARTIAL_DROPOUT_LEVELS[label]:
            raise ValueError(
                f"{label!r}: released level {released} != the level its medium records "
                f"{HILLENMEYER_PARTIAL_DROPOUT_LEVELS[label]}"
            )
    elif released[0] not in {"", "0"}:
        raise ValueError(
            f"{label!r}: a drop-out carries an unexplained dose {released}"
        )


def parse_condition(
    cond1: str, conc1: str, unit1: str, cond2: str, conc2: str, unit2: str
) -> ConditionParse:
    """Classify one column header into a typed environment (SOM Table S1).

    cond1 selects the branch; cond2, when present, ALWAYS appends a second small molecule
    (the double-drug arms). Splitting the two is what keeps ``pH7.5 + FK506`` from being
    stored as a plain pH stress and ``<compound> irradiated`` from being stored as bare
    radiation. A compound the curated table does not identify is kept under its released
    label with the typed ``inchikey`` gap ``resolved_compound`` attaches, unless the
    label is unidentified (``compound_is_identified``): then the array is dropped under
    ``UNIDENTIFIED_COMPOUND_RULE`` with the unidentified labels (joined by ``"; "``) as
    its detail.
    """
    label = cond1.strip()
    low = label.lower()
    media: Media = YPD_LIQUID
    temperature_c: float | None = None
    perturbations: list[
        SmallMoleculePerturbation | EnvironmentPhysicalPerturbation
    ] = []
    drop_reason: str | None = None
    drop_detail: str | None = None
    compound_labels: list[str] = []

    if _TEMP_CYCLE_RE.match(label):
        drop_reason, drop_detail = DROP_HEAT_SHOCK_CYCLE, label
    elif (temp_match := _TEMP_RE.search(label)) is not None:
        temperature_c = float(temp_match.group(1))
    elif (ph_match := _PH_RE.match(label)) is not None:
        perturbations.append(
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.ph,
                magnitude=Concentration(
                    value=float(ph_match.group(1)), unit=ConcentrationUnit.ph
                ),
            )
        )
    elif label in HILLENMEYER_DROPOUT_MEDIA:
        _check_dropout_level(label, conc1, unit1)
        media = HILLENMEYER_DROPOUT_MEDIA[label]
    elif low == _RADIATION_ONLY:
        perturbations.append(
            EnvironmentPhysicalPerturbation(factor=PhysicalFactor.radiation)
        )
    elif low.endswith(_IRRADIATED_SUFFIX):
        compound_labels.append(label[: -len(_IRRADIATED_SUFFIX)].strip())
        perturbations.append(_small_molecule(compound_labels[-1], conc1, unit1))
        perturbations.append(
            EnvironmentPhysicalPerturbation(factor=PhysicalFactor.radiation)
        )
    elif low in _MEDIA_SWAPS:
        media = _MEDIA_SWAPS[low]
        if conc1.strip():
            drop_reason = DROP_UNNAMED_MEDIA_AGENT
            drop_detail = f"{label}: {conc1.strip()} {unit1.strip()}".strip()
    else:
        compound_labels.append(label)
        perturbations.append(_small_molecule(label, conc1, unit1))

    if cond2.strip():
        compound_labels.append(cond2.strip())
        perturbations.append(_small_molecule(cond2.strip(), conc2, unit2))

    unidentified = [
        name for name in compound_labels if not compound_is_identified(name)
    ]
    if drop_reason is None and unidentified:
        drop_reason, drop_detail = UNIDENTIFIED_COMPOUND_RULE, "; ".join(unidentified)

    return ConditionParse(
        media=media,
        temperature_c=temperature_c,
        perturbations=perturbations,
        unsupplemented_minimal=low == _MINIMAL_MEDIA,
        drop_reason=drop_reason,
        drop_detail=drop_detail,
    )


class ColumnSpec(BaseModel):
    """One matrix data column: its index, its environment and its grouping key."""

    model_config = {"arbitrary_types_allowed": True}

    index: int
    filename: str
    header: str
    control_set: str
    pool: str
    environment: CultureEnvironment
    group_key: str
    key_check: KeyHeaderCheck
    drop_reason: str | None = None
    drop_detail: str | None = None


def _temperature_gap() -> ProvenanceGap:
    """The typed absence behind an unstated standard growth temperature."""
    return ProvenanceGap(
        field="temperature",
        reason=ProvenanceGapReason.deferred_pending_source_review,
        looked_in=_SOM_PROVENANCE,
        resolve_with=_PIERCE_PROVENANCE,
        note="the SOM states no growth temperature for the non-temperature conditions "
        "and defers the whole pooled-growth protocol to Pierce 2006, which is not "
        "mirrored; 30 C is the community default but this paper never states it",
    )


def _culture_format_gap() -> ProvenanceGap:
    """The typed absence of the culture vessel, volume, aeration and inoculum."""
    return ProvenanceGap(
        field="culture_format",
        reason=ProvenanceGapReason.deferred_pending_source_review,
        looked_in=_SOM_PROVENANCE,
        resolve_with=_PIERCE_PROVENANCE,
        note="the SOM states no vessel, volume, shaking (aeration) or inoculum and "
        "defers the pooled-growth protocol to Pierce 2006; liquid culture is implied by "
        "'OD600= 2.0'. Environment.aerobicity keeps its schema default 'aerobic', which "
        "is NOT sourced (the field is not optional, so it cannot carry a gap)",
    )


def _supplement_gap() -> ProvenanceGap:
    """The typed absence of the auxotroph supplement on the minimal medium."""
    return ProvenanceGap(
        field="auxotroph_supplements",
        reason=ProvenanceGapReason.deferred_pending_source_review,
        looked_in=_SOM_PROVENANCE,
        resolve_with=_PIERCE_PROVENANCE,
        note="the SOM (Table S1 'minimal media') and the release name a minimal medium "
        "with no supplement; the BY4743 background (his3d1/his3d1 leu2d0/leu2d0 "
        "ura3d0/ura3d0, pending Brachmann 1998) cannot grow on unsupplemented SD, so a "
        "His/Leu/Ura supplement is implied but never named. Base medium stays SD",
    )


def signed_generations(generations: str) -> float:
    """The release's signed generation count (``'-5gen'`` -> -5.0); empty raises."""
    token = generations.strip()
    if not token:
        raise ValueError("a column header has no generations field")
    return float(token.replace("gen", ""))


def _zero_generation_pre_culture_gap(label: str) -> ProvenanceGap:
    """The typed absence of the pre-culture of a ``0gen`` array or control set."""
    return ProvenanceGap(
        field="pre_culture",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_SOM_PROVENANCE,
        note=f"the generation count is {label!r}; the SOM states the pre-culture only "
        "through the count's sign (negative: from the freezer; no sign: a YPD log-phase "
        "pre-culture), and a count of 0 carries no sign, so it states no pre-culture",
    )


def pre_culture(signed: float, label: str) -> PreCulture | None:
    """What the SIGN of a generation count says about the culture before treatment.

    Negative: straight from the freezer. Positive: a YPD log-phase pre-culture to OD600
    2.0, about 10 generations. Zero carries no sign and so states nothing: ``None``, and
    the caller attaches ``_zero_generation_pre_culture_gap``.
    """
    if signed < 0:
        return PreCulture(
            source=PreCultureSource.frozen_stock,
            source_label=label,
            provenance=[_PRE_CULTURE_FROZEN],
        )
    if signed > 0:
        return PreCulture(
            source=PreCultureSource.log_phase_culture,
            medium=YPD_LIQUID,
            generations=10.0,
            od600_at_transfer=2.0,
            source_label=label,
            provenance=[_PRE_CULTURE_LOG_PHASE],
        )
    return None


def build_environment(parse: ConditionParse, generations: str) -> CultureEnvironment:
    """The typed environment of one column: generation magnitude, the pre-culture its
    sign implies, and typed gaps for what the SOM does not state.
    """
    signed = signed_generations(generations)
    gaps = [_culture_format_gap()]
    if parse.temperature_c is None:
        gaps.append(_temperature_gap())
    if parse.unsupplemented_minimal:
        gaps.append(_supplement_gap())
    if signed == 0:
        gaps.append(_zero_generation_pre_culture_gap(generations.strip()))
    return CultureEnvironment(
        media=parse.media,
        temperature=(
            Temperature(value=parse.temperature_c)
            if parse.temperature_c is not None
            else None
        ),
        perturbations=list(parse.perturbations),
        aerobicity="aerobic",
        duration_generations=abs(signed),
        pre_culture=pre_culture(signed, generations.strip()),
        culture_format=None,
        auxotroph_supplements=None,
        provenance_gaps=gaps,
    )


def read_control_set_map(path: str | Path) -> dict[str, str]:
    """``het.txt`` / ``hom.txt``: array filename -> the control set it was scored against."""
    mapping: dict[str, str] = {}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:3] != ["filename", "condition", "control_set"]:
            raise ValueError(f"unexpected key-file header in {path}: {header}")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            mapping[fields[0]] = fields[2]
    return mapping


def read_key_conditions(path: str | Path) -> dict[str, str]:
    """``het.txt`` / ``hom.txt``: array filename -> the key file's condition string."""
    mapping: dict[str, str] = {}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:3] != ["filename", "condition", "control_set"]:
            raise ValueError(f"unexpected key-file header in {path}: {header}")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            mapping[fields[0]] = fields[1]
    return mapping


def read_control_set_sizes(path: str | Path) -> dict[str, int]:
    """``*_controls.txt``: control set -> the number of control arrays it contains."""
    sizes: dict[str, int] = {}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:2] != ["control_set", "filename"]:
            raise ValueError(f"unexpected controls-file header in {path}: {header}")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            sizes[fields[0]] = sizes.get(fields[0], 0) + 1
    return sizes


def control_set_pool(control_set: str) -> str:
    """The deletion pool a control-set id names (its first ``::`` field)."""
    return control_set.split("::")[0]


def control_set_generations(control_set: str) -> str:
    """The SIGNED generation field of a control-set id, as a ``<n>gen`` token."""
    return f"{control_set.split('::')[2]}gen"


def parse_columns(
    header: list[str], control_sets: dict[str, str], key_conditions: dict[str, str]
) -> list[ColumnSpec]:
    """Parse every data column into its environment, control set and grouping key.

    Each array's header is cross-checked against its key-file condition first
    (``check_key_header``): a compound conflict and a pH conflict serve the header, a
    strain-medium conflict serves the key's medium. A 0gen array is stored with
    duration 0 and a typed pre-culture gap.
    """
    specs: list[ColumnSpec] = []
    for index, raw in enumerate(header):
        if index == 0:
            continue  # the 'Orf' row-id column
        fields = raw.strip().split(":")
        while len(fields) < 10:
            fields.append("")
        filename, c1, co1, u1, c2, co2, u2, generations = fields[:8]
        if filename not in control_sets:
            raise ValueError(f"array {filename!r} has no control set in the key file")
        key_check = check_key_header(filename, c1, key_conditions[filename])
        served_c1 = (
            "synthetic complete"
            if key_check.rule is KeyHeaderRule.key_wins_strain_medium
            else c1
        )
        parse = parse_condition(served_c1, co1, u1, c2, co2, u2)
        control_set = control_sets[filename]
        drop_reason, drop_detail = parse.drop_reason, parse.drop_detail
        if (
            key_check.rule is KeyHeaderRule.key_wins_strain_medium
            and drop_reason is not None
        ):
            drop_detail = (
                f"{drop_detail} (header {key_check.header_condition!r}, key file "
                f"{key_check.key_condition!r})"
            )
        environment = build_environment(parse, generations)
        # Group replicate arrays by the CANONICAL environment plus the control set the
        # score was computed against: two arrays of one condition normalized against
        # different control sets are different measurements.
        group_key = json.dumps(
            [environment.model_dump(), control_set], sort_keys=True, default=str
        )
        specs.append(
            ColumnSpec(
                index=index,
                filename=filename,
                header=raw.strip(),
                control_set=control_set,
                pool=control_set_pool(control_set),
                environment=environment,
                group_key=group_key,
                key_check=key_check,
                drop_reason=drop_reason,
                drop_detail=drop_detail,
            )
        )
    return specs


# --------------------------------------------------------------------------- #
# Strain rows (one physical construction each)
# --------------------------------------------------------------------------- #
class StrainRow(BaseModel):
    """One matrix row: a constructed strain ``ORF:batch`` and its per-array values.

    ``orf`` and ``batch`` are the two halves of the row id, verbatim.
    """

    row_id: str
    orf: str
    batch: str
    values: list[float | None]


class DroppedStrain(BaseModel):
    """A row kept out of the records under a named strain rule."""

    row_id: str
    rule: str
    values: list[float | None]


class MatrixRows(BaseModel):
    """Every row of one matrix: the stored strains and the author-withdrawn ones."""

    rows: list[StrainRow]
    dropped_strains: list[DroppedStrain]


# Kept only for parsons2006.py's import until #1053 lands; Hillenmeyer stores every ORF
# as released and calls neither function below.
def _sgd_history_gap(field: str, note: str) -> ProvenanceGap:
    """A gap on a ``ConstructedOrf`` field that only the SGD ORF history would close."""
    return pending_source_review(
        field,
        Provenance(
            source_uri="https://www.yeastgenome.org (locus history)",
            method="not mirrored; the SGD locus-history record of the source ORF",
        ),
        note,
    )


def constructed_orf(source_orf: str, current: str, n_sources: int) -> ConstructedOrf:
    """What a strain built against a since-renamed ORF physically deleted.

    Kept only for parsons2006.py's import until #1053 lands; Hillenmeyer stores every
    ORF as released. ``n_sources`` >= 2 is ``merged``; a lone rename gaps the relation.
    The deleted interval is always a gap.
    """
    relation = OrfHistoryRelation.merged if n_sources >= 2 else None
    gaps = [
        _sgd_history_gap(
            "deleted_span",
            f"the strain deleted {source_orf} as annotated when it was made; that "
            "interval is not released",
        )
    ]
    if relation is None:
        gaps.append(
            _sgd_history_gap(
                "relation",
                f"resolve_gene_name maps {source_orf} onto {current} through the R64 "
                "GFF alias layer, which does not say whether it is an alias, a "
                "reannotation or a merge",
            )
        )
    return ConstructedOrf(
        source_systematic_name=source_orf,
        relation=relation,
        deleted_span=None,
        provenance_gaps=gaps,
    )


def read_matrix_rows(path: str | Path, n_columns: int, arm: str) -> MatrixRows:
    """Stream the matrix into one ``StrainRow`` per row id ``ORF:batch``, verbatim.

    No genome is consulted: the ORF is stored as the release spells it, whatever a
    later annotation calls it. Only the arm's SOM-named withdrawal
    (``EXCLUDED_SOURCE_ORFS``) is kept out, under its strain rule.
    """
    rows: list[StrainRow] = []
    dropped_strains: list[DroppedStrain] = []
    excluded = EXCLUDED_SOURCE_ORFS[arm]
    with open(path) as handle:
        handle.readline()
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            row_id = cells[0].strip().strip('"')
            orf, _, batch = row_id.partition(":")
            values: list[float | None] = [None] * n_columns
            for index in range(1, min(n_columns, len(cells))):
                cell = cells[index].strip()
                if cell.upper() not in _MISSING:
                    values[index] = float(cell)
            if orf in excluded:
                dropped_strains.append(
                    DroppedStrain(row_id=row_id, rule=excluded[orf], values=values)
                )
                continue
            rows.append(StrainRow(row_id=row_id, orf=orf, batch=batch, values=values))
    return MatrixRows(rows=rows, dropped_strains=dropped_strains)


# --------------------------------------------------------------------------- #
# Record builders
# --------------------------------------------------------------------------- #
def hillenmeyer_background() -> EukaryoticStrainBackground:
    """BY4743, named by the release's key file; every allele pending Brachmann 1998.

    The name and ploidy are sourced (``hom.txt``; the main article), and the
    background's provenance carries the SOM's two deferrals (ref 1 Giaever 2002 for the
    collections, ref 2 Pierce 2006 for the protocol). Mating type, parents and the
    construction of the diploid are stated by no mirrored source and are gapped.
    """
    alleles = standard_background(
        BACKGROUND_STRAIN,
        resolve_with=BRACHMANN_1998,
        note="BY4741 x BY4742 genotype is not in a mirrored source; the background "
        "name BY4743 is sourced to the release's hom.txt key file",
    ).alleles
    return EukaryoticStrainBackground(
        reference_strain="S288C",
        assembly_set=SGD_S288C_R64,
        name=BACKGROUND_STRAIN,
        parents=None,
        construction=None,
        mating_type=None,
        ploidy="diploid",
        alleles=alleles,
        provenance=[
            SOURCED_VALUES["background_strain"],
            SOURCED_VALUES["diploid"],
            SOURCED_VALUES["collections_deferral"],
            SOURCED_VALUES["ref1"],
            SOURCED_VALUES["protocol_deferral"],
            SOURCED_VALUES["ref2"],
        ],
        provenance_gaps=[
            pending_source_review(
                "mating_type",
                BRACHMANN_1998,
                "BY4743 is MATa/MATalpha in the literature-standard genotype; no "
                "mirrored source states it",
            ),
            pending_source_review(
                "parents", BRACHMANN_1998, "BY4741 x BY4742, not in a mirrored source"
            ),
            pending_source_review(
                "construction",
                GIAEVER_2002,
                "how the YKO diploids were made (hom: MATa x MATalpha deletants mated) "
                "is in the SOM's ref (1), not mirrored",
            ),
        ],
    )


def marker_loci(background: EukaryoticStrainBackground) -> frozenset[str]:
    """Loci where the background carries a non-R64 allele (the BY marker loci)."""
    return frozenset(a.systematic_gene_name for a in background.alleles)


def _barcode_gaps() -> list[ProvenanceGap]:
    """Typed absences of the UPTAG / DNTAG sequences (not in the release)."""
    note = (
        "the SOM states each cassette carries an uptag and a downtag; the release "
        "identifies strains by ORF:batch only, the tag sequences are in the SGDP strain "
        "tables (SOM ref 1), not mirrored"
    )
    return [
        pending_source_review("barcode", GIAEVER_2002, note),
        pending_source_review("downtag_barcode", GIAEVER_2002, note),
    ]


def strain_perturbation(
    arm: str, row: StrainRow, pool: str, markers: frozenset[str]
) -> HeterozygousDeletionPerturbation | BarcodedKanMxDeletionPerturbation:
    """The screened edit of one constructed strain in one deletion pool.

    The gene is the released ORF verbatim and ``constructed_orf`` is ``None``. HET: a
    heterozygous kanMX4 allele replacement; at a BY marker locus which allele the
    cassette replaced is a typed gap (it decides the functional dose at LYS2 / MET15).
    HOM: a barcoded kanMX4 deletion; at a marker locus what was physically deleted is a
    typed gap on ``constructed_orf`` (keyed on the released ORF).
    """
    construction = StrainConstruction(batch=row.batch)
    gaps = _barcode_gaps()
    marker = row.orf in markers
    if arm == "het":
        if marker:
            gaps.append(
                pending_source_review(
                    "replaced_allele",
                    GIAEVER_2002,
                    f"{row.orf} is a BY4743 marker locus; which background allele the "
                    "kanMX4 cassette replaced is not stated in a mirrored source",
                )
            )
        return HeterozygousDeletionPerturbation(
            systematic_gene_name=row.orf,
            perturbed_gene_name=row.orf,
            cassette=KANMX4_CASSETTE.value,
            collection=pool,
            construction=construction,
            constructed_orf=None,
            replaced_allele=None,
            barcode=None,
            downtag_barcode=None,
            provenance_gaps=gaps,
        )
    if marker:
        gaps.append(
            pending_source_review(
                "constructed_orf",
                GIAEVER_2002,
                f"{row.orf} is a BY4743 marker locus already edited in a parent "
                "haploid; what the homozygous strain physically lacks there is not "
                "stated in a mirrored source",
            )
        )
    return BarcodedKanMxDeletionPerturbation(
        systematic_gene_name=row.orf,
        perturbed_gene_name=row.orf,
        cassette=KANMX4_CASSETTE.value,
        collection=pool,
        construction=construction,
        constructed_orf=None,
        barcode=None,
        downtag_barcode=None,
        provenance_gaps=gaps,
    )


def build_reference(
    dataset_name: str,
    spec: _MatrixSpec,
    control_set: str,
    n_control_arrays: int,
    background: EukaryoticStrainBackground,
) -> StrainEnvironmentResponseExperimentReference:
    """The matched no-drug control set this control set's scores were computed against.

    ``control_set`` is the release's own id, ``<pool>::<scanner>::<generations>::tag3::
    YPD::dmso::0``. Its signed generation field gives the reference's duration and
    pre-culture, so the reference differs from the treatment in the perturbation and in
    nothing else.
    """
    generations = control_set_generations(control_set)
    signed = signed_generations(generations)
    gaps = [_culture_format_gap(), _temperature_gap()]
    if signed == 0:
        gaps.append(_zero_generation_pre_culture_gap(generations))
    environment = CultureEnvironment(
        media=YPD_LIQUID,
        temperature=None,
        perturbations=[],
        aerobicity="aerobic",
        duration_generations=abs(signed),
        pre_culture=pre_culture(signed, generations),
        culture_format=None,
        auxotroph_supplements=None,
        provenance_gaps=gaps,
    )
    return StrainEnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=StrainReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain=BACKGROUND_STRAIN,
            ploidy="diploid",
            background=background,
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        ),
        environment_reference=environment,
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=spec.measurement_type,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            n_samples=n_control_arrays,
            sample_unit=SampleUnit.biological_replicate,
            screen_id=control_set,
            units=(
                f"no-drug control set {control_set!r}: {n_control_arrays} control "
                "arrays on the same deletion pool, generation count and scanner, "
                "run in YPD with a DMSO vehicle at concentration 0 as the release "
                "spells it. The score is 0 by construction -- it is the control "
                "mean this set's treatment arrays are scored against"
            ),
        ),
    )


def build_phenotype(
    spec: _MatrixSpec, values: list[float], control_set: str
) -> EnvironmentResponsePhenotype:
    """Mean over the replicate arrays of one group, with the across-array sample SD.

    A sample SD of exactly 0 is NOT reported as a dispersion. The matrices print four
    to six decimals, so two arrays agreeing to the last printed digit means the
    dispersion is below the release's resolution, not that it was measured to be zero.
    Storing 0 would give those records a standard error of 0 and claim infinite
    precision, so the uncertainty is a typed absence instead.
    """
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n >= 2 else None
    gaps: list[ProvenanceGap] = []
    if sd == 0.0:
        sd = None
        gaps.append(
            ProvenanceGap(
                field="environment_response_uncertainty",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=_SOM_PROVENANCE,
                note=f"the {n} arrays of this group print the identical score to the "
                "release's last decimal, so the replicate dispersion is below the "
                "released precision rather than measured to be zero",
            )
        )
    return EnvironmentResponsePhenotype(
        measurement_type=spec.measurement_type,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=mean,
        n_samples=n,
        sample_unit=SampleUnit.biological_replicate,
        environment_response_uncertainty=sd,
        environment_response_uncertainty_type=(
            UncertaintyType.sample_sd if sd is not None else None
        ),
        screen_id=control_set,
        units=spec.units,
        provenance_gaps=gaps,
    )


def array_values(
    row_values: list[float | None], members: list[ColumnSpec]
) -> list[float]:
    """One value per replicate ARRAY of a group for one strain row (missing skipped)."""
    values: list[float] = []
    for column in members:
        value = row_values[column.index]
        if value is not None:
            values.append(value)
    return values


def group_columns(
    columns: list[ColumnSpec],
) -> tuple[dict[str, list[ColumnSpec]], dict[str, list[ColumnSpec]]]:
    """Kept and dropped columns grouped by environment + control set."""
    kept: dict[str, list[ColumnSpec]] = {}
    dropped: dict[str, list[ColumnSpec]] = {}
    for column in columns:
        target = kept if column.drop_reason is None else dropped
        target.setdefault(column.group_key, []).append(column)
    return kept, dropped


def iter_records(
    dataset_name: str,
    spec: _MatrixSpec,
    matrix: MatrixRows,
    groups: dict[str, list[ColumnSpec]],
    background: EukaryoticStrainBackground,
) -> Iterator[tuple[StrainEnvironmentResponseExperiment, str]]:
    """Every record of a matrix: (experiment, control set), strain row by strain row.

    Shared by ``process()`` and the measurement script, so a sliced run goes through
    the same builders as the full build.
    """
    markers = marker_loci(background)
    group_list = [(members[0], members) for members in groups.values()]
    for row in matrix.rows:
        genotypes: dict[str, Genotype] = {}
        for head, members in group_list:
            values = array_values(row.values, members)
            if not values:
                continue
            if head.pool not in genotypes:
                genotypes[head.pool] = Genotype(
                    perturbations=[
                        strain_perturbation(spec.key, row, head.pool, markers)
                    ]
                )
            yield (
                StrainEnvironmentResponseExperiment(
                    dataset_name=dataset_name,
                    genotype=genotypes[head.pool],
                    environment=head.environment,
                    phenotype=build_phenotype(spec, values, head.control_set),
                ),
                head.control_set,
            )


def count_records(
    rows: list[StrainRow] | list[DroppedStrain], groups: dict[str, list[ColumnSpec]]
) -> int:
    """How many (row, group) pairs carry at least one value: the record count."""
    return sum(
        1
        for row in rows
        for members in groups.values()
        if array_values(row.values, members)
    )


class CellAccounting(BaseModel):
    """Every released value cell of one matrix, by where it went.

    ``n_value_cells`` is counted independently of the split (every non-missing cell of
    every row in the matrix's data columns); the split must sum to it.
    """

    n_value_cells: int
    n_in_stored_records: int
    n_in_dropped_arrays: dict[str, int]
    n_in_withdrawn_strains: dict[str, int]


def cell_accounting(
    matrix: MatrixRows,
    kept: dict[str, list[ColumnSpec]],
    dropped: dict[str, list[ColumnSpec]],
) -> CellAccounting:
    """Split every value cell into stored, dropped-array (per rule) and withdrawn."""
    cells: list[list[float | None]] = [row.values for row in matrix.rows]
    cells += [strain.values for strain in matrix.dropped_strains]
    total = sum(1 for values in cells for value in values if value is not None)
    stored = sum(
        len(array_values(row.values, members))
        for row in matrix.rows
        for members in kept.values()
    )
    by_rule: dict[str, int] = {}
    for members in dropped.values():
        rule = str(members[0].drop_reason)
        by_rule[rule] = by_rule.get(rule, 0) + sum(
            len(array_values(row.values, members)) for row in matrix.rows
        )
    withdrawn: dict[str, int] = {}
    for strain in matrix.dropped_strains:
        withdrawn[strain.rule] = withdrawn.get(strain.rule, 0) + sum(
            1 for value in strain.values if value is not None
        )
    accounting = CellAccounting(
        n_value_cells=total,
        n_in_stored_records=stored,
        n_in_dropped_arrays=by_rule,
        n_in_withdrawn_strains=withdrawn,
    )
    split = stored + sum(by_rule.values()) + sum(withdrawn.values())
    if split != total:
        raise ValueError(
            f"cell accounting does not sum: {split} split vs {total} value cells"
        )
    return accounting


def compound_identity_gaps(
    rows: list[StrainRow], kept: dict[str, list[ColumnSpec]]
) -> dict[str, dict[str, int]]:
    """Stored compound labels carrying the typed ``inchikey`` gap, with their counts.

    Keyed by ``Compound.name`` (the released label, since no curated row matched);
    ``n_records`` counts the records whose environment doses that compound.
    """
    gaps: dict[str, dict[str, int]] = {}
    for key, members in kept.items():
        names = sorted(
            {
                perturbation.compound.name
                for perturbation in members[0].environment.perturbations
                if isinstance(perturbation, SmallMoleculePerturbation)
                and "inchikey" in perturbation.compound.gapped_fields()
            }
        )
        if not names:
            continue
        n_records = count_records(rows, {key: members})
        for name in names:
            bucket = gaps.setdefault(
                name, {"n_environments": 0, "n_arrays": 0, "n_records": 0}
            )
            bucket["n_environments"] += 1
            bucket["n_arrays"] += len(members)
            bucket["n_records"] += n_records
    return dict(sorted(gaps.items()))


class UnidentifiedCompoundHeldOut(BaseModel):
    """One label held out under ``UNIDENTIFIED_COMPOUND_RULE``, with what it carried.

    ``n_records_not_written`` counts the (stored strain row, environment, control set)
    records its arrays would have written; ``n_value_cells`` the released value cells of
    those arrays over every row, the figure the cell accounting sums.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="the released compound label, verbatim")
    vendor: str | None = Field(
        description="the vendor the label names ('Biomol' from a trailing '(Biomol)', "
        "'chemical diversity labs' from that prefix), verbatim; None when it names none"
    )
    vendor_product_name: str | None = Field(
        description="the vendor's product name or catalog code with the vendor removed "
        "(e.g. 'telomerase inhibitor ix'), the key for a catalog lookup that could admit "
        "the label through RELEASED_IDENTITIES; None when the label names no vendor"
    )
    n_environments: int
    n_arrays: int
    n_records_not_written: int
    n_value_cells: int
    headers: list[str] = Field(description="the deposited column headers, verbatim")
    key_file_conditions: list[str] = Field(
        description="the key file's condition string for each array, verbatim"
    )


_VENDOR_SUFFIX = re.compile(r"^(?P<product>.+) \((?P<vendor>Biomol)\)$")
_VENDOR_PREFIX = re.compile(r"^(?P<vendor>chemical diversity labs) (?P<product>\S+)$")


def vendor_catalog_entry(label: str) -> tuple[str | None, str | None]:
    """``(vendor, product name)`` a held-out label names, verbatim, or ``(None, None)``.

    The two vendor spellings the release uses: a trailing ``(Biomol)`` and the
    ``chemical diversity labs`` prefix. Nothing is normalized: the product name is the
    released text with the vendor removed, so a catalog lookup sees what the source wrote.
    """
    for pattern in (_VENDOR_SUFFIX, _VENDOR_PREFIX):
        match = pattern.match(label)
        if match is not None:
            return match.group("vendor"), match.group("product")
    return None, None


def unidentified_compounds_held_out(
    matrix: MatrixRows, dropped: dict[str, list[ColumnSpec]]
) -> list[UnidentifiedCompoundHeldOut]:
    """Every label held out under ``UNIDENTIFIED_COMPOUND_RULE``, sorted by label.

    An array dosing two unidentified labels is listed under each (none does in the
    release; the per-label counts then overlap and the rule total stays the cell
    accounting's).
    """
    by_label: dict[str, list[list[ColumnSpec]]] = {}
    for members in dropped.values():
        head = members[0]
        if head.drop_reason != UNIDENTIFIED_COMPOUND_RULE:
            continue
        for label in str(head.drop_detail).split("; "):
            by_label.setdefault(label, []).append(members)
    held: list[UnidentifiedCompoundHeldOut] = []
    for label, groups in sorted(by_label.items()):
        columns = [column for members in groups for column in members]
        vendor, product = vendor_catalog_entry(label)
        held.append(
            UnidentifiedCompoundHeldOut(
                label=label,
                vendor=vendor,
                vendor_product_name=product,
                n_environments=len(groups),
                n_arrays=len(columns),
                n_records_not_written=sum(
                    count_records(matrix.rows, {members[0].group_key: members})
                    for members in groups
                ),
                n_value_cells=sum(
                    len(array_values(strain.values, members))
                    for strain in matrix.rows
                    for members in groups
                ),
                headers=[column.header for column in columns],
                key_file_conditions=[
                    column.key_check.key_condition for column in columns
                ],
            )
        )
    return held


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class _Hillenmeyer2008Base(ExperimentDataset):
    """Shared FitDb HIP/HOP loader. HET (log2_ratio) and HOM (z_score) are SEPARATE
    datasets (incomparable readouts, distinct collections), one concrete subclass each.
    """

    _matrix_key: str = ""  # 'het' | 'hom' -- set by concrete subclasses

    def __init__(
        self,
        root: str,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (no genome: the loader stores what was released)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def spec(self) -> _MatrixSpec:
        """This subclass's released matrix and its constants."""
        return MATRICES[self._matrix_key]

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return StrainEnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return StrainEnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The mirrored files this dataset reads: its matrix and its control-set tables."""
        spec = self.spec
        return [spec.filename, spec.keyfile, spec.controls_file]

    def download(self) -> None:
        """Link the manifest-listed mirror files into ``raw/`` and verify each sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        rel = raw_relpaths()
        os.makedirs(self.raw_dir, exist_ok=True)
        for name in self.raw_file_names:
            src = raw_mirror_dir(data_root) / rel[name]
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(
                src, osp.join(self.raw_dir, name), manifest_sha256(manifest, rel[name])
            )
        log.info(
            "Hillenmeyer 2008 %s raw files linked into %s (sha256 verified against the "
            "raw-mirror manifest)",
            self._matrix_key,
            self.raw_dir,
        )

    # ---- build ----------------------------------------------------------------
    @post_process
    def process(self) -> None:
        """Build one record per (strain row, environment, control set)."""
        spec = self.spec
        data_root = _data_root()
        manifest = load_manifest(data_root)
        rel = raw_relpaths()
        verify_raw_files(
            self.raw_dir,
            {
                name: manifest_sha256(manifest, rel[name])
                for name in self.raw_file_names
            },
        )
        matrix_path = osp.join(self.raw_dir, spec.filename)
        keyfile = osp.join(self.raw_dir, spec.keyfile)
        control_sizes = read_control_set_sizes(
            osp.join(self.raw_dir, spec.controls_file)
        )
        with open(matrix_path) as handle:
            header = handle.readline().rstrip("\n").split("\t")
        columns = parse_columns(
            header, read_control_set_map(keyfile), read_key_conditions(keyfile)
        )
        groups, dropped_groups = group_columns(columns)
        log.info(
            "Hillenmeyer2008 %s: %d arrays -> %d kept environments (%d arrays), "
            "%d dropped environments (%d arrays)",
            spec.key,
            len(columns),
            len(groups),
            sum(len(m) for m in groups.values()),
            len(dropped_groups),
            sum(len(m) for m in dropped_groups.values()),
        )
        matrix = read_matrix_rows(matrix_path, len(header), spec.key)
        background = hillenmeyer_background()
        references = {
            members[0].control_set: build_reference(
                self.name,
                spec,
                members[0].control_set,
                control_sizes[members[0].control_set],
                background,
            )
            for members in groups.values()
        }
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        batch_size = 250_000
        txn = env.begin(write=True)
        itxn = interned_env.begin(write=True)
        for experiment, control_set in tqdm(
            iter_records(self.name, spec, matrix, groups, background),
            desc=f"Hillenmeyer2008 {spec.key}",
        ):
            txn.put(
                f"{idx}".encode(),
                self._intern_record(
                    experiment, references[control_set], publication, itxn
                ),
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

        self._write_drop_report(dropped_groups, groups, matrix, idx)
        self._write_compound_gap_report(groups, matrix)
        self._write_strain_reports(matrix, groups)
        self._write_key_header_report(columns)
        self._write_batch_report(matrix)
        log.info(
            "Wrote %d Hillenmeyer2008 %s environment-response records to LMDB",
            idx,
            spec.key,
        )

    def _write_drop_report(
        self,
        dropped_groups: dict[str, list[ColumnSpec]],
        groups: dict[str, list[ColumnSpec]],
        matrix: MatrixRows,
        kept_records: int,
    ) -> None:
        """Write ``dropped_records.json``: the array rules, counts and cell accounting.

        A dropped environment's records-not-written count is over the stored strain
        rows, keyed by the GROUP, so one condition dropped at several control sets is
        counted once per group. ``cell_accounting`` splits every released value cell into
        stored, dropped-array and withdrawn-strain cells and raises if they do not sum.
        The ``UNIDENTIFIED_COMPOUND_RULE`` arrays are also listed per label, with the
        source quotes that the labels are codes with no released structure.
        """
        environments: list[dict[str, Any]] = []
        by_rule: dict[str, dict[str, int]] = {}
        for members in dropped_groups.values():
            head = members[0]
            n_records = count_records(matrix.rows, {head.group_key: members})
            environments.append(
                {
                    "rule": head.drop_reason,
                    "detail": head.drop_detail,
                    "n_arrays": len(members),
                    "headers": [c.header for c in members],
                    "n_records_not_written": n_records,
                }
            )
            bucket = by_rule.setdefault(
                str(head.drop_reason),
                {"n_environments": 0, "n_arrays": 0, "n_records_not_written": 0},
            )
            bucket["n_environments"] += 1
            bucket["n_arrays"] += len(members)
            bucket["n_records_not_written"] += n_records
        payload = {
            "dataset": self.name,
            "matrix": self.spec.filename,
            "kept_records": kept_records,
            "rules": DROP_RULES,
            "by_rule": by_rule,
            "unidentified_compound_rule": UNIDENTIFIED_COMPOUND_RULE,
            "unidentified_compound_policy": UNIDENTIFIED_COMPOUND_POLICY,
            "unidentified_compound_sources": [
                SOURCED_VALUES[key].model_dump(mode="json")
                for key in (
                    "table_s2_structures",
                    "table_s1_chemdiv_codes",
                    "table_s1_telomerase_inhibitor",
                )
            ],
            "unidentified_compounds_held_out": [
                held.model_dump()
                for held in unidentified_compounds_held_out(matrix, dropped_groups)
            ],
            "cell_accounting": cell_accounting(
                matrix, groups, dropped_groups
            ).model_dump(),
            "environments": sorted(
                environments, key=lambda e: (str(e["rule"]), str(e["detail"]))
            ),
        }
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            json.dump(payload, handle, indent=2)

    def _write_compound_gap_report(
        self, groups: dict[str, list[ColumnSpec]], matrix: MatrixRows
    ) -> None:
        """Write ``compound_identity_gaps.json``: stored compounds with no InChIKey."""
        gaps = compound_identity_gaps(matrix.rows, groups)
        with open(
            osp.join(self.preprocess_dir, "compound_identity_gaps.json"), "w"
        ) as handle:
            json.dump(
                {
                    "dataset": self.name,
                    "rule": "an IDENTIFIED compound label the curated compound table "
                    "gives no InChIKey is stored under the released label with the "
                    "typed inchikey gap resolved_compound attaches; an unidentified "
                    "label (status PROPRIETARY) is held out under "
                    f"{UNIDENTIFIED_COMPOUND_RULE} and listed in dropped_records.json",
                    "n_compounds": len(gaps),
                    "n_records": sum(g["n_records"] for g in gaps.values()),
                    "compounds": gaps,
                },
                handle,
                indent=2,
            )

    def _write_strain_reports(
        self, matrix: MatrixRows, groups: dict[str, list[ColumnSpec]]
    ) -> None:
        """Write ``dropped_strains.json``: the author-withdrawn rows and their counts."""
        by_rule: dict[str, dict[str, Any]] = {}
        for strain in matrix.dropped_strains:
            bucket = by_rule.setdefault(
                strain.rule, {"row_ids": [], "n_records_not_written": 0}
            )
            bucket["row_ids"].append(strain.row_id)
            bucket["n_records_not_written"] += count_records([strain], groups)
        with open(osp.join(self.preprocess_dir, "dropped_strains.json"), "w") as handle:
            json.dump(
                {"dataset": self.name, "rules": STRAIN_DROP_RULES, "by_rule": by_rule},
                handle,
                indent=2,
            )

    def _write_key_header_report(self, columns: list[ColumnSpec]) -> None:
        """Write ``key_header_checks.json``: every non-agreeing array, both strings."""
        checks = [c.key_check for c in columns]
        with open(
            osp.join(self.preprocess_dir, "key_header_checks.json"), "w"
        ) as handle:
            json.dump(
                {
                    "keyfile": self.spec.keyfile,
                    "n_arrays": len(checks),
                    "n_by_outcome": {
                        outcome.value: sum(1 for c in checks if c.outcome is outcome)
                        for outcome in KeyHeaderOutcome
                    },
                    "rules": {
                        KeyHeaderRule.header_wins_table_s1.value: "Table S1 lists only "
                        f"{QUOTE_TABLE_S1_PH!r}; the header is served",
                        KeyHeaderRule.key_wins_strain_medium.value: "the key names the "
                        f"strain's medium {QUOTE_KEY_BY4743!r}; served on SC",
                        KeyHeaderRule.header_served_key_disagrees.value: "the "
                        "header and the key file name different compounds and the SOM "
                        "gives no tie-breaker; the array is served under the header's "
                        "compound, the column the values are released under",
                    },
                    "disagreements": [
                        c.model_dump(mode="json")
                        for c in checks
                        if c.outcome is not KeyHeaderOutcome.agree
                    ],
                },
                handle,
                indent=2,
            )

    def _write_batch_report(self, matrix: MatrixRows) -> None:
        """Write the construction-batch artifacts, including the SOM's suspicious set.

        The batch is on every record's ``construction`` now; these files keep the
        per-ORF view and the SOM's suspicious-batch flag beside the build.
        """
        batches: dict[str, list[str]] = {}
        for row in matrix.rows:
            batches.setdefault(row.orf, []).append(row.batch)
        suspicious = set(SUSPICIOUS_BATCHES)
        flagged = {
            orf: sorted(set(orf_batches) & suspicious)
            for orf, orf_batches in batches.items()
            if set(orf_batches) & suspicious
        }
        multi = {
            orf: sorted(orf_batches)
            for orf, orf_batches in batches.items()
            if len(orf_batches) > 1
        }
        with open(osp.join(self.preprocess_dir, "strain_batches.json"), "w") as handle:
            json.dump(
                {
                    "quote_batch_ids": QUOTE_BATCH_IDS,
                    "quote_reconstructed": QUOTE_RECONSTRUCTED,
                    "citation_key": CITATION_KEY,
                    "sha256": SOM_SHA256,
                    "n_orfs": len(batches),
                    "n_rows": sum(len(v) for v in batches.values()),
                    "n_orfs_with_multiple_batches": len(multi),
                    "batches_by_orf": {
                        k: sorted(v) for k, v in sorted(batches.items())
                    },
                },
                handle,
                indent=2,
            )
        with open(
            osp.join(self.preprocess_dir, "suspicious_batch_strains.json"), "w"
        ) as handle:
            json.dump(
                {
                    "quote": QUOTE_SUSPICIOUS,
                    "quote_count": QUOTE_SUSPICIOUS_COUNT,
                    "quote_homozygous": QUOTE_SUSPICIOUS_HOM,
                    "citation_key": CITATION_KEY,
                    "sha256": SOM_SHA256,
                    "suspicious_batches": list(SUSPICIOUS_BATCHES),
                    "collection": self.spec.collection,
                    "n_flagged_orfs": len(flagged),
                    "note": "these ORFs have at least one construction row in a batch "
                    "the SOM names as carrying a shared secondary genotype; the authors "
                    "excluded 647 heterozygous strains on this basis and state the "
                    "homozygous collection did not show the pattern. Records are SERVED "
                    "with the batch on their construction rather than dropped, because "
                    "the exclusion is an analysis choice, not a defect of the measurement",
                    "flagged_orfs": dict(sorted(flagged.items())),
                },
                handle,
                indent=2,
            )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "Hillenmeyer2008 builds its records in process(); see iter_records"
        )


@register_dataset
class HetHillenmeyer2008Dataset(_Hillenmeyer2008Base):
    """FitDb HIP (heterozygous kanMX4 allele replacement) fitness-defect log2-ratio."""

    organism = Organism.scerevisiae

    _matrix_key = "het"

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_hillenmeyer2008_het",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HET (HIP) dataset."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)


@register_dataset
class HomHillenmeyer2008Dataset(_Hillenmeyer2008Base):
    """FitDb HOP (homozygous barcoded kanMX4 deletion) fitness-defect z-score dataset."""

    organism = Organism.scerevisiae

    _matrix_key = "hom"

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_hillenmeyer2008_hom",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HOM (HOP) dataset."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)


def main() -> None:
    """Build/load both datasets for interactive debugging."""
    data_root = _data_root()
    for cls, sub in (
        (HetHillenmeyer2008Dataset, "het"),
        (HomHillenmeyer2008Dataset, "hom"),
    ):
        root = osp.join(data_root, f"data/torchcell/env_chemgen_hillenmeyer2008_{sub}")
        dataset = cls(root=root)
        print(f"{sub}: len = {len(dataset)}")
        print(dataset[0])


if __name__ == "__main__":
    main()
