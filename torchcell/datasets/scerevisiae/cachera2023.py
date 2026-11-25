# torchcell/datasets/scerevisiae/cachera2023
# [[torchcell.datasets.scerevisiae.cachera2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/cachera2023
# Test file: tests/torchcell/datasets/scerevisiae/test_cachera2023.py
"""Cachera 2023 CRI-SPA betaxanthin dataset (genome-wide product-proxy screen).

Cachera et al. 2023 (Nucleic Acids Research, doi:10.1093/nar/gkad656) used CRI-SPA to
transfer four betaxanthin-biosynthesis genes into each strain of the ~4800-strain yeast
knockout (YKO) collection, then read out per-colony **betaxanthin** (a yellow plant
metabolite) by image analysis. The "CRI-SPA score" is a corrected/normalized colony
YELLOWNESS score, the geometric mean of Value and Saturation of the colony's pixels in
HSV color space (paper line 58), not a fluorescence measurement -- a quantitative proxy
for betaxanthin level (it can be negative because it is population-centered).

Environment (issue #509): the final screen plate is solid YPD with 200 mg/l G418
(``CACHERA_YPD_G418``, every value quoted from the mirror OCR), and the paper states no
growth temperature, so ``temperature=None`` with a typed ``ProvenanceGap``.

Source: the paper's Data Availability (``paper.md`` line 175) points to the CRI-SPA
GitHub repo (github.com/pc2912/CRI-SPA_repo) for the per-screen raw datasets and the
individual and corrected, filtered datasets (``SOURCED_VALUES["data_availability"]``).
Two of its files are read, both sha256-pinned and both deposited in the raw
mirror ``$DATA_ROOT/torchcell-raw/cacheraCRISPAHighthroughputMethod2023/data/`` with a
``manifest.json`` retrieval record (commit ``2c800ab7``):

- ``QD1_2_4_6.csv`` (the repository README: "(Quadruplicate) Size and Yellowness raw and
  corrected data for colony quadruplicates"; for the ``1_2_4_6`` files "data extracted
  from several screen repeats, pooled and processed after pooling"; repeats 1, 2, 4 and
  6 are each a "Screen repeat with final antibiotic selection"): one row per COLONY
  POSITION of each of the four screen repeats (79,872 rows = 4 x 13 plates x 1,536),
  keyed by ``repeat``, ``plate``, ``row`` and ``column``. The stored value is the 24 h
  ``corrected_mean_intensity`` of one colony.
- ``GA1_2_4_6.csv``: the authors' per-gene mean, std and count over those colonies. Only
  its ``WT`` row is read, as the reference summary.

ONE RECORD PER COLONY (#1177, tracking #1178). A colony is one biological replicate: one
CRI-SPA transfer event ("using CRI-SPA in quadruplicate amounting to a total volume of
19148 transfer events", line 126; the four colonies of a strain are four "trials", line
117), and the gene score the authors report is "the mean of the colony replicate" (line
64), "a median of 15 replicates (16 is maximum) per gene" over the four pooled screens
(line 145). Each colony is a ``MetaboliteExperiment`` whose ``metabolite_level`` is that
colony's value verbatim, ``n_replicates`` 1, ``metabolite_level_se`` ``None`` with a typed
``not_applicable_single_measurement`` gap, and ``replicate_id`` the colony's position in
the release, ``repeat<r>/plate<p>/row<i>/column<j>`` (the file prints no other colony
label). The position also tells apart the two or three arrayed library strains that 61
gene tokens occupy (e.g. ACO2 at plate 0 row 5 column 17 and plate 1 row 5 column 16):
the per-gene file merges them into one mean, the colony records keep every one. Grouping
the colony records of one gene and taking their mean reproduces ``GA1_2_4_6.csv`` to
1.3e-15 (``experiments/036-dataset-fixes-before-kg-build/scripts/
cachera2023_per_colony_mean_check.py``). Before 2026.11.24 the loader stored one record
per gene from ``GA1_2_4_6.csv`` (mean, std / sqrt(count), count): 4,759 records.

THE CONTROL COLONIES ARE RECORDS. The library positions the release labels ``WT`` (8 per
repeat; 12 colonies carry a value, all in repeats 4 and 6) are stored as records with the
Btx cassette and no deletion: every colony on the final plate of these four repeats grew
under the cassette's own selection ("we employed NTC selection in the final step of
CRI-SPA as a safety measure to ensure that all cells contain the Btx-cassette", line
131), and the paper calls that point "the BY-Btx strain" (Figure 5 legend, line 148).
``phenotype_reference`` is the authors' own summary of those colonies, the ``WT`` row of
``GA1_2_4_6.csv`` (mean 0.2224, std / sqrt(12), n 12, ``replicate_id`` ``None``); before
2026.11.24 it was a constant 0.0 asserted from the centering of the score.

Maps to `MetabolitePhenotype` (WS4): `metabolite_level = {"betaxanthin": score}` with
`measurement_type = "cri_spa_corrected_hsv_yellowness_24h"`.

Every colony whose position names a gene is stored under the name the ``gene`` column
releases, verbatim (#1045): a systematic name goes on ``KanMxDeletionPerturbation``
(both names the ORF), any other name on ``ReleasedIdentifierPerturbation`` (the released
symbol, kind ``deletion``, marker KanMX). The token is upper-cased (two systematic names
are released in mixed case) and otherwise verbatim; no two released tokens upper-case to
one name. No genome is consulted and nothing is renamed or merged.

The colony positions without a record, each listed in ``preprocess/dropped_records.json``
under its rule: positions where no colony was measured (both the raw and the corrected
24 h value blank, 10,825), colonies the AUTHORS withdrew (a raw value but a blank
corrected one, 126: line 64 deletes Grubbs outliers within quadruplicates and genes with
two colonies or less, ``SOURCED_VALUES["authors_withdrawal"]``), and colonies at a position the library map labels ``0`` (an
empty library well, so no strain is named; 87). The 48 h columns are released and not
stored.
"""

import logging
import math
import os
import os.path as osp
import pickle
import re
import urllib.request
from collections.abc import Callable
from typing import Any, Literal

import lmdb
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    post_process,
    verify_raw_files,
    write_verified,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import YPD
from torchcell.datamodels.schema import (
    ArtifactRef,
    Concentration,
    ConcentrationUnit,
    ConstructPlacement,
    ConstructSpan,
    ConstructVariantCall,
    Environment,
    Experiment,
    ExperimentReference,
    FeatureKey,
    GeneAdditionPerturbation,
    Genotype,
    HostPlacement,
    KanMxDeletionPerturbation,
    Localization,
    Media,
    MediaComponent,
    MediaComponentRole,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    VariantType,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.construct.registry import CACHERA2023_CRI_SPA_PLASMIDS
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

TARGET_METABOLITE = "betaxanthin"

# --------------------------------------------------------------------------- #
# Paper anchors (issue #509). Every quote below is a verbatim substring of the mirror
# OCR (MinerU output, so LaTeX markup is kept as written); the line is that file's.
# --------------------------------------------------------------------------- #
CITATION_KEY = "cacheraCRISPAHighthroughputMethod2023"
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "5fb7310dacb44bc04fc121a990e4ccc70020c9598877c4b074656e91fc4f32d5"
_PAPER = Provenance(
    source_uri=PAPER_MD, citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
)

# line 58: the readout is a color score from image pixels, not a fluorescence intensity.
READOUT_QUOTE = (
    "filter taking the geometric mean of Value and Saturation of image pixels in the "
    "HSV color space"
)
# line 58: the color score was calibrated on YPD.
READOUT_MEDIUM_QUOTE = "which was our observation on YPD media"
# line 117: the plate the screen colonies grew on.
SCREEN_PLATE_QUOTE = (
    "we picked cells from each of the four colonies obtained for each strain on the "
    "final CRI-SPA screen plate (YPD-G418)"
)
# line 31: YPD recipe deferral, agar and G418 doses on solid medium.
YPD_RECIPE_QUOTE = (
    "YPD and synthetic complete (SC) media, and SC drop out media were prepared as "
    "described by Sherman et al. (28)"
)
AGAR_QUOTE = "For growth on solid medium, $2 0 \\ \\mathrm { g / l }$ agar was added."
G418_QUOTE = (
    "For selection on solid medium, plates were supplemented with "
    "$2 0 0 ~ \\mathrm { { m g / l } }$ geneticin (G418)"
)
# line 126: a colony is one transfer event (the biological replicate, #1177).
TRANSFER_EVENTS_QUOTE = "using CRI-SPA in quadruplicate amounting to a total volume of 19148 transfer events"
# line 145: the four pooled screens give up to 16 colonies per gene.
REPLICATES_PER_GENE_QUOTE = "with a median of 15 replicates (16 is maximum) per gene"
# line 64: the authors' own deletions, and the gene score is the colony mean. The
# apostrophe is the OCR's U+2019, kept verbatim.
WITHDRAWAL_QUOTE = (
    "outliers within quadruplicates were detected with Grubbs’ test for outliers "
    "and deleted. Genes represented by two colonies or less were deleted from the "
    "analysis. Finally, the mean of the colony replicate was used as the final score "
    "for each gene."
)
# line 68: the same deletion after the screens are pooled.
POOLED_WITHDRAWAL_QUOTE = (
    "the individual datasets were pooled and outliers removed with Grubbs’ test"
)
# line 131: the final plate selects the cassette, so every measured colony carries it.
CASSETTE_SELECTION_QUOTE = (
    "we employed NTC selection in the final step of CRI-SPA as a safety measure to "
    "ensure that all cells contain the Btx-cassette"
)
# line 148: the WT position of the pooled screen is the BY-Btx strain.
BY_BTX_QUOTE = "The position of the BY-Btx strain is marked with a black asterisk."
# line 175: where the per-colony and per-gene files are released.
DATA_AVAILABILITY_QUOTE = (
    "the individual and the corrected $^ +$ filtered datasets are all available on our "
    "CRI-SPA github repository"
)

#: What the stored level IS: the 24 h ``corrected_mean_intensity`` column of
#: ``GA1_2_4_6.csv``, an HSV yellowness score (``READOUT_QUOTE``), corrected for plate
#: position. The word "intensity" in the column name is the image Value/Saturation
#: intensity, not fluorescence.
MEASUREMENT_TYPE = "cri_spa_corrected_hsv_yellowness_24h"


def _paper_sv(value: object, quote: str, note: str | None = None) -> SourcedValue:
    """A SourcedValue pinned to the Cachera mirror OCR (quote + sha256)."""
    return SourcedValue(value=value, provenance=_PAPER, quote=quote, note=note)


SOURCED_VALUES: dict[str, SourcedValue] = {
    "measurement_type": _paper_sv(
        MEASUREMENT_TYPE,
        READOUT_QUOTE,
        note="line 58; a colony-color score, so the name says HSV yellowness",
    ),
    "screen_medium": _paper_sv("solid YPD + G418", SCREEN_PLATE_QUOTE, note="line 117"),
    "readout_medium": _paper_sv("YPD", READOUT_MEDIUM_QUOTE, note="line 58"),
    "ypd_recipe": _paper_sv(
        "YPD per Sherman et al. (ref 28)",
        YPD_RECIPE_QUOTE,
        note="line 31; Sherman et al. is not mirrored, so the YPD ingredient amounts "
        "are the library YPD root's (Tong and Boone 2006 YEPD), which this sentence "
        "does not restate",
    ),
    "agar": _paper_sv("20 g/l", AGAR_QUOTE, note="line 31"),
    "g418": _paper_sv("200 mg/l", G418_QUOTE, note="line 31"),
    # --- #1177: what a stored record is (one colony), and what the authors withdrew ---
    "replicate_unit": _paper_sv(
        "one colony = one CRI-SPA transfer event",
        TRANSFER_EVENTS_QUOTE,
        note="line 126; 4,787 strains x 4 colonies per screen repeat",
    ),
    "replicates_per_gene": _paper_sv(
        "up to 16 colonies per gene over the four pooled screens",
        REPLICATES_PER_GENE_QUOTE,
        note="line 145; the released per-gene count is the number of colonies",
    ),
    "authors_withdrawal": _paper_sv(
        "Grubbs outliers and genes with <= 2 colonies deleted; gene score = colony mean",
        WITHDRAWAL_QUOTE,
        note="line 64 (and line 68 for the pooled screens); a colony with a raw value "
        "and a blank corrected value in QD1_2_4_6.csv is one the authors deleted",
    ),
    "pooled_withdrawal": _paper_sv(
        "pooled screens: Grubbs outliers removed after pooling",
        POOLED_WITHDRAWAL_QUOTE,
        note="line 68",
    ),
    "control_cassette": _paper_sv(
        "the final plate selects the Btx-cassette (NTC)",
        CASSETTE_SELECTION_QUOTE,
        note="line 131; so a colony at a WT position carries the cassette",
    ),
    "control_is_by_btx": _paper_sv(
        "the WT position of the screen is the BY-Btx strain",
        BY_BTX_QUOTE,
        note="line 148, Figure 5 legend",
    ),
    "data_availability": _paper_sv(
        "per-colony and per-gene files on the CRI-SPA GitHub repository",
        DATA_AVAILABILITY_QUOTE,
        note="line 175",
    ),
}

CACHERA_YPD_G418: Media = Media(
    name="YPD + G418 (solid, 2% agar; Cachera 2023 final CRI-SPA screen plate)",
    state="solid",
    is_synthetic=False,
    base_medium="YPD",
    components=[
        *YPD.components,
        MediaComponent(
            compound=resolved_compound("agar"),
            role=MediaComponentRole.gelling_agent,
            concentration=Concentration(value=2.0, unit=ConcentrationUnit.percent_w_v),
            provenance=[SOURCED_VALUES["agar"]],
        ),
        MediaComponent(
            compound=resolved_compound("G418 (geneticin)"),
            role=MediaComponentRole.selection_agent,
            concentration=Concentration(value=200.0, unit=ConcentrationUnit.ug_per_ml),
            provenance=[SOURCED_VALUES["g418"]],
            note="selects the kanMX marker of the YKO deletion (CRI-SPA step 5)",
        ),
    ],
    provenance=[
        SOURCED_VALUES["screen_medium"],
        SOURCED_VALUES["ypd_recipe"],
        SOURCED_VALUES["readout_medium"],
    ],
)
"""The final CRI-SPA screen plate: library ``YPD`` root components plus the paper's own
agar and G418 doses. Loader-local on purpose (as ``COOPER_SC`` is): ``media.py`` is a
value surface of the served graph, so editing it would touch every YPD dataset."""

#: No growth temperature in the paper. Measured by
#: ``experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_environment_readout.py``
#: (2026.10.02, issue #509): the mirror ``paper.md`` has 0 matches for a degree sign,
#: ``\circ``, "temperature" or a two-digit number followed by C, and the SI archive's
#: four documents (Supplementary Material.docx, Supp Methods S1/S2, Figure S8) have 3,
#: all SGD gene descriptions ("required for growth at low temperature") in a hit table.
TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER,
    note="no growth or incubation temperature is stated in the paper or its SI "
    "(gkad656_supplemental_files.zip); the loader formerly stored an unsourced 30 C",
)


# --------------------------------------------------------------------------- #
# The Btx-cassette's MATERIAL: the pBTX1 contig, its four gene features, and the two
# designed alleles as contig coordinates (P4, 2026-10-15).
#
# Every value below is measured by
# ``experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_btx_cassette_placement.py``
# (results JSON beside it) from the deposited construct set
# ``cachera2023_cri_spa_plasmids``, and nothing here is read off a design note.
#
# WHY pBTX1 and not pBTX2. The paper, line 46: "CD-Btx was constructed from UC+XII-5a by
# transforming with the integrative plasmid pBTX1 harboring the betaxanthin-synthesizing
# genes CYP76AD1 and DOD, an NTC selectable marker, negative feedback resistant ARO4
# K229L, ARO7 G141S and an sgRNA expression cassette for targeting the XII-5 site. The
# BY-Btx strain was made by transforming the NotI excised insert of pBTX2". The screen
# transfers the cassette FROM CD-Btx into the YKO library ("we then used CD-Btx to
# transfer the Btx-cassette to the 4787 unique deletion strains"), so the cassette in
# every screened strain is pBTX1's. pBTX2 is the BY-Btx vector; since 2026-10-15 a span
# holds the parts of an origin-crossing location, so pBTX2 indexes too (handle
# 63f89c1eaa13), but it is a different replicon on different coordinates and no record
# here points at it.
#
# WHY the host span is a typed gap. The paper names the landing pad "XII-5" and never
# gives its S288C coordinates, which is the common case for a named pad; the genome
# never gates a record, so the pad is stated verbatim and the coordinate is a gap.
#
# WHY the insert span is a typed gap, re-measured 2026-10-15 with pBTX2 indexed. pBTX1
# annotates both homology arms ("XII-5 up B" at 10..220, "XII-5 dw A" at 9598..9859),
# but the three readings of "the transferred interval" they admit disagree: arm to arm
# inclusive is 9,850 bp, between the arms exclusive is 9,377 bp, and the paper's own
# figure is "the considerable size, 9.3 kb, of the Btx-cassette". pBTX1 carries NO NotI
# site, so there is no cut to resolve the arms against. The contrast is what settles it:
# on pBTX2 the same question HAS an exact answer, because the paper says "the NotI
# excised insert of pBTX2" and the bytes carry exactly two NotI sites (GCGGCCGC at 1527
# and 10797), whose top-strand cuts bound 1529..10798 = 9,270 bp, which is the paper's
# 9.3 kb. That interval is on pBTX2's coordinates and pBTX2 is not this record's contig,
# so it is recorded in the construct index (pBTX2's own quoted name statement is that
# NotI sentence) and in
# ``experiments/036-dataset-fixes-before-kg-build/results/cachera2023_construct_span_parts.json``,
# and it is NOT transferred onto a pBTX1 placement. The source states no interval of
# pBTX1 as transferred, so neither does the record.
#
# Two heterologous plant genes (CYP76AD1, DOD) plus two feedback-resistant mutant
# alleles of NATIVE yeast genes (ARO4^K229L / YBR249C, ARO7^G141S / YPR060C) integrated
# ectopically (hence GeneAddition with a variant, NOT AllelePerturbation at the native
# locus). The natMX marker is not a betaxanthin gene and is omitted. source_organism for
# DOD needs confirmation against DeLoache 2015 (ref 25) at review. See
# [[torchcell.datamodels.gene-addition-perturbation-design]].
# --------------------------------------------------------------------------- #
#: The construct set the placement reads, and the contig's 12-hex handle in it.
CONSTRUCT_SET = CACHERA2023_CRI_SPA_PLASMIDS
PBTX1_HANDLE = "bbaca60509e0"
#: The sha256 of ``si/plasmids/pBTX001(ante113).gb`` IS the contig's identity; the
#: GenBank LOCUS line of that file reads ``ANTE113_with_NatMX``, so the name the paper
#: uses is a quoted statement in the construct index and never a minted id.
PBTX1_SHA256 = "bbaca60509e079222f7cdd1bb54b70d93445af3a699c65109d94b553fc0087f7"
PBTX1_PATH = "si/plasmids/pBTX001(ante113).gb"
PBTX1_BYTES = 31895
PBTX1_REF = ArtifactRef(
    tier="library",
    key=CITATION_KEY,
    path=PBTX1_PATH,
    sha256=PBTX1_SHA256,
    bytes=PBTX1_BYTES,
)

#: The landing pad, verbatim, and the assembly set its (unpublished) coordinate would be
#: written against.
INTEGRATION_SITE = "XII-5"
ASSEMBLY_SET = SGD_S288C_R64

#: The gene features of pBTX1, keyed by the loader's gene key: label, SO id
#: (``SO:0000704`` gene), 1-based end-inclusive span and orientation. Each was checked
#: to be a CDS in frame against the contig's own bases (length a multiple of three, ATG
#: start, one terminal stop, no internal stop).
_FEATURE_SPANS: dict[
    str, tuple[str, int, int, Literal["inline", "reverse_complement"]]
] = {
    "CYP76AD1": ("CYP76AD/P450 fusion", 4388, 5887, "reverse_complement"),
    "DOD": ("DOPA 4,5 dioxygenase", 7190, 7999, "inline"),
    "YBR249C": ("ARO4-K229L", 2684, 3796, "inline"),
    "YPR060C": ("ARO7-G141S", 472, 1242, "reverse_complement"),
}

#: The two designed alleles as contig coordinates, derived from the feature span and the
#: codon number and CONFIRMED against the bytes: codon 229 of the ARO4 feature reads
#: ``TTA`` (Leu) and codon 141 of the ARO7 feature reads ``AGT`` (Ser), so the map
#: carries the designed residue at the coordinate the record states. The codon is three
#: bases, so the call is a ``substitution`` over the codon's span rather than an SNV.
_ALLELE_CALLS: dict[str, tuple[str, int, int, int]] = {
    "YBR249C": ("K229L", 3368, 3370, 229),
    "YPR060C": ("G141S", 820, 822, 141),
}

#: ``HostPlacement.span`` and ``.insert_span``: typed absences, with what was read.
_SPAN_GAP_NOTE = (
    "the paper names the landing pad 'XII-5' and gives no S288C coordinate for it; the "
    "sgRNA target feature on pBTX1 ('target sequence XII-5', 14691..14710) is a "
    "coordinate on the VECTOR, not on chromosome XII"
)
_INSERT_SPAN_GAP_NOTE = (
    "pBTX1 annotates both homology arms ('XII-5 up B' 10..220, 'XII-5 dw A' "
    "9598..9859), but the readings they admit disagree with each other and with the "
    "paper's own figure: 9,850 bp arm to arm inclusive, 9,377 bp between the arms, "
    "against 'the considerable size, 9.3 kb, of the Btx-cassette'. pBTX1 carries no "
    "NotI site, so no cut resolves the arms; the sister vector pBTX2 does admit an "
    "exact interval (the paper's 'NotI excised insert of pBTX2', two GCGGCCGC sites at "
    "1527 and 10797, cuts bounding 1529..10798 = 9,270 bp = the paper's 9.3 kb), but "
    "that is pBTX2's coordinate space and this record's contig is pBTX1. The source "
    "states no interval of pBTX1 as transferred, so the record states none"
)


def _host_placement() -> HostPlacement:
    """The XII-5 landing pad: named by the paper, with both spans typed absent."""
    return HostPlacement(
        assembly_set=ASSEMBLY_SET,
        site_label=INTEGRATION_SITE,
        span=None,
        insert_span=None,
        provenance_gaps=[
            ProvenanceGap(
                field="span",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=_PAPER,
                note=_SPAN_GAP_NOTE,
            ),
            ProvenanceGap(
                field="insert_span",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=_PAPER,
                note=_INSERT_SPAN_GAP_NOTE,
            ),
        ],
    )


def _placement(gene_key: str) -> ConstructPlacement:
    """The pBTX1 placement of one Btx-cassette gene, with its designed allele."""
    label, start, end, orientation = _FEATURE_SPANS[gene_key]
    call = _ALLELE_CALLS.get(gene_key)
    variants = []
    if call is not None:
        statement, position_start, position_end, codon_number = call
        variants.append(
            ConstructVariantCall(
                variant_type=VariantType.substitution,
                variant_statement=statement,
                reference_contig_sha256=PBTX1_SHA256,
                position_start=position_start,
                position_end=position_end,
                amino_acid_change=statement,
                codon_number=codon_number,
            )
        )
    return ConstructPlacement(
        contig_sha256=PBTX1_SHA256,
        contig_ref=PBTX1_REF,
        feature_key=FeatureKey(
            label=label,
            so_id="SO:0000704",
            span=ConstructSpan.one(start, end, orientation),
        ),
        host_placement=_host_placement(),
        variants=variants,
    )


def _betaxanthin_cassette() -> list[GeneAdditionPerturbation]:
    """Fresh Btx-cassette perturbations (new objects per record)."""
    return [
        GeneAdditionPerturbation(
            systematic_gene_name="CYP76AD1",
            gene_namespace="scerevisiae_s288c_sgd_systematic",
            perturbed_gene_name="CYP76AD1",
            source_organism="Beta vulgaris",
            is_heterologous=True,
            localization=Localization.chromosomal_integration,
            integration_locus=INTEGRATION_SITE,
            construct_name="Btx-cassette",
            placement=_placement("CYP76AD1"),
        ),
        GeneAdditionPerturbation(
            systematic_gene_name="DOD",
            gene_namespace="scerevisiae_s288c_sgd_systematic",
            perturbed_gene_name="DOD",
            source_organism="Mirabilis jalapa",
            is_heterologous=True,
            localization=Localization.chromosomal_integration,
            integration_locus=INTEGRATION_SITE,
            construct_name="Btx-cassette",
            placement=_placement("DOD"),
        ),
        GeneAdditionPerturbation(
            systematic_gene_name="YBR249C",
            gene_namespace="scerevisiae_s288c_sgd_systematic",
            perturbed_gene_name="ARO4",
            source_organism="Saccharomyces cerevisiae",
            is_heterologous=False,
            localization=Localization.chromosomal_integration,
            integration_locus=INTEGRATION_SITE,
            construct_name="Btx-cassette",
            variant="K229L",
            placement=_placement("YBR249C"),
        ),
        GeneAdditionPerturbation(
            systematic_gene_name="YPR060C",
            gene_namespace="scerevisiae_s288c_sgd_systematic",
            perturbed_gene_name="ARO7",
            source_organism="Saccharomyces cerevisiae",
            is_heterologous=False,
            localization=Localization.chromosomal_integration,
            integration_locus=INTEGRATION_SITE,
            construct_name="Btx-cassette",
            variant="G141S",
            placement=_placement("YPR060C"),
        ),
    ]


# Gene-level corrected+filtered dataset (replicates 1/2/4/6) from the CRI-SPA repo.
DATA_URL = "https://raw.githubusercontent.com/pc2912/CRI-SPA_repo/main/GA1_2_4_6.csv"
DATA_FILENAME = "GA1_2_4_6.csv"
# Pinned sha256 of GA1_2_4_6.csv (role si_data in the library manifest:
# torchcell-library/cacheraCRISPAHighthroughputMethod2023/manifest.json). The stored
# artifact + this hash is canonical, NOT the live GitHub URL; verified on download so
# upstream drift is detected rather than silently followed.
DATA_SHA256 = "71f55609067301e1430a1ab3226618c46428488753c6412603e8f0c21bbcac8a"
_LEVEL = "corrected_mean_intensity.24_mean"
_STD = "corrected_mean_intensity.24_std"
_COUNT = "corrected_mean_intensity.24_count"

#: The commit of github.com/pc2912/CRI-SPA_repo the per-colony file was taken from (the
#: repository tip on 2026-11-25, last pushed 2023-08-02); the GA file is byte-identical
#: there and on ``main``.
REPO_COMMIT = "2c800ab7b9811456f353d325be75b24e938440c6"
#: The per-colony release (#1177): one row per colony position of each screen repeat.
QD_FILENAME = "QD1_2_4_6.csv"
QD_URL = (
    f"https://raw.githubusercontent.com/pc2912/CRI-SPA_repo/{REPO_COMMIT}/{QD_FILENAME}"
)
QD_SHA256 = "58fc1cba88ce54fc0af073657b285a6d0a09cb0b703eb3178ee0f958659cf4ff"
QD_BYTES = 10645057
#: The raw mirror holding both files and their retrieval records.
RAW_MIRROR_REL = f"torchcell-raw/{CITATION_KEY}"
#: One colony's 24 h value, corrected (edge effects, screen centering and scaling) and
#: after the authors' outlier deletion; ``_COLONY_RAW`` is the same colony before any
#: correction, read only to tell an unmeasured position from a deleted colony.
_COLONY_LEVEL = "corrected_mean_intensity.24"
_COLONY_RAW = "mean_intensity.24"
#: The released key of one colony: screen repeat, then plate, row and column.
_POSITION = ("repeat", "plate", "row", "column")

#: An S288C systematic name, the form that goes on the placed deletion leaf.
_SYSTEMATIC_RE = re.compile(
    r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$"
)

#: The three kinds of released colony position that are not a record, by name.
RULE_NO_COLONY = "no_colony_measured"
RULE_WITHDRAWN = "withdrawn_by_the_authors"
RULE_NO_GENE = "gene_cell_is_0_or_blank"
_RULE_DESCRIPTIONS = {
    RULE_NO_COLONY: (
        "both the raw and the corrected 24 h value of the position are blank: no "
        "colony was measured there (none grew, or the image gave none)"
    ),
    RULE_WITHDRAWN: (
        "the position has a raw 24 h value but a blank corrected one: the authors "
        "deleted the colony as a Grubbs outlier or with a gene of two colonies or less "
        "(paper.md line 64 and line 68)"
    ),
    RULE_NO_GENE: (
        "the gene cell is '0' or blank, an empty library well, so the colony names no "
        "strain"
    ),
}
#: The library positions the release labels as the wild type: the BY-Btx control.
_CONTROL_TOKENS = frozenset({"WT"})

#: Why a colony record has no standard error.
_SINGLE_COLONY_SE_NOTE = (
    "a record is ONE colony (one CRI-SPA transfer event) and one measurement has no "
    "dispersion of its own; the per-gene mean and SE over the colony records of one "
    "gene are a modeling-time derivation (the authors' own are in GA1_2_4_6.csv)"
)


def replicate_id(position: tuple[int, int, int, int]) -> str:
    """The colony's position in the release, ``repeat<r>/plate<p>/row<i>/column<j>``."""
    repeat, plate, row, column = position
    return f"repeat{repeat}/plate{plate}/row{row}/column{column}"


def _released_deletion(
    gene: str,
) -> KanMxDeletionPerturbation | ReleasedIdentifierPerturbation:
    """The varying deletion under the name the ``gene`` column releases (#1045).

    A systematic name goes on the placed KanMX leaf with both names the ORF; any other
    name goes on the released-identifier leaf as the symbol the source wrote.
    """
    if _SYSTEMATIC_RE.match(gene):
        return KanMxDeletionPerturbation(
            systematic_gene_name=gene, perturbed_gene_name=gene
        )
    return ReleasedIdentifierPerturbation(
        systematic_gene_name=gene,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=gene,
        identifier_kind="gene column entry, as released",
        released_kind=ReleasedKind.deletion,
        cassette="KanMX",
    )


class RepeatedGeneError(ValueError):
    """Two distinct released gene tokens upper-case to one name; nothing is merged."""


class RepeatedPositionError(ValueError):
    """Two released rows carry one colony position; one colony is one record."""


class MissingControlSummaryError(ValueError):
    """``GA1_2_4_6.csv`` has no usable ``WT`` row to summarize the control colonies."""


class DropRule(BaseModel):
    """One rule and the released colony positions it left without a record."""

    rule: str
    description: str
    n_records: int
    items: list[str]


class DropLog(BaseModel):
    """The build's accounting: every released colony position is a record or listed."""

    dataset: str
    source_records: int
    kept_records: int
    kept_control_records: int
    kept_under_systematic_name: int
    kept_under_released_symbol: int
    dropped_records: int
    rules: list[DropRule]


@register_dataset
class BetaxanthinCachera2023Dataset(ExperimentDataset):
    """Genome-wide betaxanthin CRI-SPA product-proxy screen of the YKO collection."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/betaxanthin_cachera2023",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; every colony is stored under its released gene name."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return MetaboliteExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return MetaboliteExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The per-gene and the per-colony CRI-SPA files required before processing."""
        return [DATA_FILENAME, QD_FILENAME]

    def download(self) -> None:
        """Download the per-gene and per-colony files from the authors' GitHub repo.

        The stored artifact's sha256 is canonical: a freshly downloaded payload is
        verified before it is written, and each file in ``raw/`` (present or fresh) is
        verified against its pin at the start of ``process``. A mismatch means upstream
        drift or corruption and raises (never silently followed). A file already in
        ``raw/`` is not fetched again.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        for url, filename, sha256 in (
            (DATA_URL, DATA_FILENAME, DATA_SHA256),
            (QD_URL, QD_FILENAME, QD_SHA256),
        ):
            dest = osp.join(self.raw_dir, filename)
            if osp.exists(dest):
                continue
            log.info("Downloading CRI-SPA data from %s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) < 10000:
                raise RuntimeError(f"CRI-SPA download too small: {len(data)} bytes")
            write_verified(data, dest, sha256, url)
            log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(data))

    def _control_summary(self) -> MetabolitePhenotype:
        """The authors' own summary of the control colonies: the ``WT`` row of GA.

        Mean, std / sqrt(count) and count of the 24 h corrected level, ``replicate_id``
        ``None``. It is the source's statistic, not one this loader computes.
        """
        ga = pd.read_csv(osp.join(self.raw_dir, DATA_FILENAME))
        rows = ga[ga["gene"].astype(str).str.strip().isin(_CONTROL_TOKENS)]
        if len(rows) != 1:
            raise MissingControlSummaryError(
                f"{DATA_FILENAME} has {len(rows)} WT rows; the reference needs one"
            )
        row = rows.iloc[0]
        if pd.isna(row[_LEVEL]) or pd.isna(row[_STD]) or pd.isna(row[_COUNT]):
            raise MissingControlSummaryError(
                f"{DATA_FILENAME} WT row lacks a level, std or count"
            )
        count = int(row[_COUNT])
        if count < 2:
            raise MissingControlSummaryError(
                f"{DATA_FILENAME} WT row summarizes {count} colony; no SE"
            )
        return MetabolitePhenotype(
            metabolite_level={TARGET_METABOLITE: float(row[_LEVEL])},
            metabolite_level_se={
                TARGET_METABOLITE: float(row[_STD]) / math.sqrt(count)
            },
            n_replicates={TARGET_METABOLITE: count},
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=None,
        )

    @post_process
    def process(self) -> None:
        """Store every measured colony as a Metabolite experiment; write LMDB."""
        verify_raw_files(
            self.raw_dir, {DATA_FILENAME: DATA_SHA256, QD_FILENAME: QD_SHA256}
        )
        self._reference_phenotype = self._control_summary()
        qd = pd.read_csv(osp.join(self.raw_dir, QD_FILENAME))

        os.makedirs(self.preprocess_dir, exist_ok=True)
        no_colony: list[str] = []
        withdrawn: list[str] = []
        no_gene: list[str] = []
        positions: set[tuple[int, int, int, int]] = set()
        released_token: dict[str, str] = {}
        rows: list[dict[str, Any]] = []
        for token_cell, level, raw, *key in zip(
            qd["gene"].tolist(),
            qd[_COLONY_LEVEL].tolist(),
            qd[_COLONY_RAW].tolist(),
            *(qd[name].tolist() for name in _POSITION),
            strict=True,
        ):
            position = (int(key[0]), int(key[1]), int(key[2]), int(key[3]))
            if position in positions:
                raise RepeatedPositionError(
                    f"{QD_FILENAME} names colony {replicate_id(position)} on two rows; "
                    "one colony is one record"
                )
            positions.add(position)
            rid = replicate_id(position)
            token = str(token_cell).strip()
            if pd.isna(level):
                (no_colony if pd.isna(raw) else withdrawn).append(rid)
                continue
            if token in ("0", "nan", ""):
                no_gene.append(rid)
                continue
            if token in _CONTROL_TOKENS:
                gene = None
            else:
                # two systematic names are released in mixed case (YAL043C-a,
                # YML009c); a systematic name is case-insensitive, so the token is
                # upper-cased, and two distinct tokens may not meet on one name
                gene = token.upper()
                first = released_token.setdefault(gene, token)
                if first != token:
                    raise RepeatedGeneError(
                        f"{QD_FILENAME} releases {first!r} and {token!r}, both "
                        f"{gene!r} upper-cased; one name is one strain"
                    )
            rows.append(
                {
                    "gene": gene,
                    "replicate_id": rid,
                    "repeat": position[0],
                    "plate": position[1],
                    "row": position[2],
                    "column": position[3],
                    "level": float(level),
                }
            )
        deletions = [r for r in rows if r["gene"] is not None]
        n_systematic = sum(1 for r in deletions if _SYSTEMATIC_RE.match(r["gene"]))
        drop_log = DropLog(
            dataset=self.name,
            source_records=int(len(qd)),
            kept_records=len(rows),
            kept_control_records=len(rows) - len(deletions),
            kept_under_systematic_name=n_systematic,
            kept_under_released_symbol=len(deletions) - n_systematic,
            dropped_records=len(no_colony) + len(withdrawn) + len(no_gene),
            rules=[
                DropRule(
                    rule=rule,
                    description=_RULE_DESCRIPTIONS[rule],
                    n_records=len(items),
                    items=items,
                )
                for rule, items in (
                    (RULE_NO_COLONY, no_colony),
                    (RULE_WITHDRAWN, withdrawn),
                    (RULE_NO_GENE, no_gene),
                )
            ],
        )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as fh:
            fh.write(drop_log.model_dump_json(indent=2))
        log.info(
            "Cachera: %d colony records (%d control, %d under a systematic name, %d "
            "under a released symbol); %d positions without a colony, %d colonies "
            "withdrawn by the authors, %d colonies on a position naming no strain",
            len(rows),
            drop_log.kept_control_records,
            n_systematic,
            len(deletions) - n_systematic,
            len(no_colony),
            len(withdrawn),
            len(no_gene),
        )
        pd.DataFrame(rows).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        with env.begin(write=True) as txn:
            for record_row in tqdm(rows):
                experiment, reference, publication = self.create_experiment(record_row)
                txn.put(
                    f"{idx}".encode(),
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": reference.model_dump(),
                            "publication": publication.model_dump(),
                        }
                    ),
                )
                idx += 1
        env.close()
        log.info("Wrote %d Cachera betaxanthin experiments to LMDB", idx)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[MetaboliteExperiment, MetaboliteExperimentReference, Publication]:
        """Build the Metabolite experiment/reference/publication for one colony.

        ``row["gene"]`` is the upper-cased released token, or ``None`` for a control
        colony (the Btx cassette and no deletion).
        """
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4741",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )
        deletion = [] if row["gene"] is None else [_released_deletion(row["gene"])]
        genotype = Genotype(perturbations=[*deletion, *_betaxanthin_cassette()])
        # Issue #509: YPD + G418 screen plate (line 117), no stated temperature.
        environment = Environment(
            media=CACHERA_YPD_G418, temperature=None, provenance_gaps=[TEMPERATURE_GAP]
        )
        phenotype = MetabolitePhenotype(
            metabolite_level={TARGET_METABOLITE: row["level"]},
            metabolite_level_se=None,
            n_replicates={TARGET_METABOLITE: 1},
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=None,
            replicate_id=row["replicate_id"],
            provenance_gaps=[
                ProvenanceGap(
                    field="metabolite_level_se",
                    reason=ProvenanceGapReason.not_applicable_single_measurement,
                    note=_SINGLE_COLONY_SE_NOTE,
                )
            ],
        )
        experiment = MetaboliteExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        # Reference = the Btx background (4 betaxanthin genes, no deletion), summarized
        # by the authors' own WT row: mean, std / sqrt(n) and n over its colonies.
        reference = MetaboliteExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=self._reference_phenotype.model_copy(),
        )
        publication = Publication(
            pubmed_id="37572348",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/37572348/",
            doi="10.1093/nar/gkad656",
            doi_url="https://doi.org/10.1093/nar/gkad656",
        )
        return experiment, reference, publication


def main() -> None:
    """Build/load the dataset for interactive debugging.

    Loads the existing LMDB if already built; to step through ``process()`` /
    ``create_experiment`` under a debugger, delete ``<root>/processed`` first so the
    build re-runs.
    """
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/torchcell/betaxanthin_cachera2023")
    dataset = BetaxanthinCachera2023Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
