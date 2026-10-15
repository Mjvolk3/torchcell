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

Source: the paper's Data Availability points to the CRI-SPA GitHub repo
(github.com/pc2912/CRI-SPA_repo). We ingest `GA1_2_4_6.csv` -- the gene-level
corrected+filtered dataset combining screen replicates 1/2/4/6 -- using the 24 h
`corrected_mean_intensity` (mean/std/count) as the betaxanthin level + SE + n.

Maps to `MetabolitePhenotype` (WS4): `metabolite_level = {"betaxanthin": score}` with
`measurement_type = "cri_spa_corrected_hsv_yellowness_24h"`.

Every row that names a gene and carries a 24 h level is one record under the name the
``gene`` column releases, verbatim (#1045): a systematic name goes on
``KanMxDeletionPerturbation`` (both names the ORF), any other name on
``ReleasedGeneNamePerturbation`` (the released symbol, kind ``deletion``, marker KanMX).
The token is upper-cased (two systematic names are released in mixed case) and
otherwise verbatim. No genome is consulted and nothing is renamed or merged; the pinned
file (4,788 rows) repeats no gene token. The rows without a record are the 28 whose gene cell is ``0`` or
blank or whose level is blank and the one ``WT`` control, listed in
``preprocess/dropped_records.json``. Before #1045 the loader resolved every symbol
through a genome (4,719 records), dropping 11 names and merging 30 collisions.
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
    BacterialVariantType,
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
    ReleasedGeneKind,
    ReleasedGeneNamePerturbation,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.construct.registry import CACHERA2023_CRI_SPA_PLASMIDS
from torchcell.sequence.genome.registry import SGD_S288C_R64
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
# every screened strain is pBTX1's. pBTX2 is the BY-Btx vector and is one of the 11 maps
# the parser blocks for an origin-wrapping location, which does not reach these records.
#
# WHY the host span is a typed gap. The paper names the landing pad "XII-5" and never
# gives its S288C coordinates, which is the common case for a named pad; the genome
# never gates a record, so the pad is stated verbatim and the coordinate is a gap.
#
# WHY the insert span is a typed gap. pBTX1 annotates both homology arms ("XII-5 up B"
# at 10..220, "XII-5 dw A" at 9598..9859), but the three readings of "the transferred
# interval" they admit disagree: arm to arm inclusive is 9,850 bp, between the arms
# exclusive is 9,377 bp, and the paper's own figure is "the considerable size, 9.3 kb,
# of the Btx-cassette". The source states no interval, so neither does the record.
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
    "against 'the considerable size, 9.3 kb, of the Btx-cassette'. The source states no "
    "interval of the vector as transferred, so the record states none"
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
                variant_type=BacterialVariantType.substitution,
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
            span=ConstructSpan(start=start, end=end, orientation=orientation),
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

#: An S288C systematic name, the form that goes on the placed deletion leaf.
_SYSTEMATIC_RE = re.compile(
    r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$"
)

#: The two kinds of released row that are not a record (#1045), by name.
RULE_NO_GENE_OR_LEVEL = "gene_cell_is_0_or_blank_or_level_is_blank"
RULE_CONTROL_ROW = "control_row"
_RULE_DESCRIPTIONS = {
    RULE_NO_GENE_OR_LEVEL: (
        "the gene cell is '0' or blank (an empty plate position) or the 24 h corrected "
        "level is blank, so the row names no deletion strain with a measurement"
    ),
    RULE_CONTROL_ROW: (
        "the gene cell is 'WT', the unperturbed CRI-SPA Donor background, which is the "
        "reference and not a deletion strain"
    ),
}
_CONTROL_TOKENS = frozenset({"WT"})


def _released_deletion(
    gene: str,
) -> KanMxDeletionPerturbation | ReleasedGeneNamePerturbation:
    """The varying deletion under the name the ``gene`` column releases (#1045).

    A systematic name goes on the placed KanMX leaf with both names the ORF; any other
    name goes on the released-gene-name leaf as the symbol the source wrote.
    """
    if _SYSTEMATIC_RE.match(gene):
        return KanMxDeletionPerturbation(
            systematic_gene_name=gene, perturbed_gene_name=gene
        )
    return ReleasedGeneNamePerturbation(
        systematic_gene_name=gene,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=gene,
        released_gene_name=gene,
        released_kind=ReleasedGeneKind.deletion,
        marker="KanMX",
    )


class RepeatedGeneError(ValueError):
    """Two released rows carry one gene token; one strain is one record."""


class DropRule(BaseModel):
    """One rule and the released rows it left without a record."""

    rule: str
    description: str
    n_records: int
    items: list[str]


class DropLog(BaseModel):
    """The build's accounting: every released row is a record or listed here."""

    dataset: str
    source_records: int
    kept_records: int
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
        """Initialize the dataset; every row is stored under its released gene name."""
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
        """The gene-level CRI-SPA dataset required before processing."""
        return [DATA_FILENAME]

    def download(self) -> None:
        """Download the gene-level CRI-SPA dataset from the authors' GitHub repo.

        The stored artifact's sha256 is canonical: a freshly downloaded payload is
        verified before it is written, and the file in ``raw/`` (present or fresh) is
        verified against ``DATA_SHA256`` at the start of ``process``. A mismatch means
        upstream drift or corruption and raises (never silently followed).
        """
        dest = osp.join(self.raw_dir, DATA_FILENAME)
        if osp.exists(dest):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        log.info("Downloading CRI-SPA data from %s", DATA_URL)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        if len(data) < 10000:
            raise RuntimeError(f"CRI-SPA download too small: {len(data)} bytes")
        write_verified(data, dest, DATA_SHA256, DATA_URL)
        log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(data))

    @post_process
    def process(self) -> None:
        """Store every released deletion row as a Metabolite experiment; write LMDB."""
        verify_raw_files(self.raw_dir, {DATA_FILENAME: DATA_SHA256})
        df = pd.read_csv(osp.join(self.raw_dir, DATA_FILENAME))

        os.makedirs(self.preprocess_dir, exist_ok=True)
        no_gene_or_level: list[str] = []
        control: list[str] = []
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            gene = str(row["gene"]).strip()
            level = row[_LEVEL]
            if gene in ("0", "nan", "") or pd.isna(level):
                no_gene_or_level.append(gene)
                continue
            if gene in _CONTROL_TOKENS:
                control.append(gene)
                continue
            # two systematic names are released in mixed case (YAL043C-a, YML009c);
            # a systematic name is case-insensitive, so the token is upper-cased
            gene = gene.upper()
            if gene in seen:
                raise RepeatedGeneError(
                    f"GA1_2_4_6.csv names {gene!r} on two rows; one strain is one record"
                )
            seen.add(gene)
            count = int(row[_COUNT]) if pd.notna(row[_COUNT]) else 1
            std = row[_STD]
            se = (
                float(std) / math.sqrt(count)
                if pd.notna(std) and count > 1
                else float("nan")
            )
            rows.append(
                {"gene": gene, "level": float(level), "se": se, "n": max(count, 1)}
            )
        n_systematic = sum(1 for r in rows if _SYSTEMATIC_RE.match(r["gene"]))
        drop_log = DropLog(
            dataset=self.name,
            source_records=int(len(df)),
            kept_records=len(rows),
            kept_under_systematic_name=n_systematic,
            kept_under_released_symbol=len(rows) - n_systematic,
            dropped_records=len(no_gene_or_level) + len(control),
            rules=[
                DropRule(
                    rule=RULE_NO_GENE_OR_LEVEL,
                    description=_RULE_DESCRIPTIONS[RULE_NO_GENE_OR_LEVEL],
                    n_records=len(no_gene_or_level),
                    items=no_gene_or_level,
                ),
                DropRule(
                    rule=RULE_CONTROL_ROW,
                    description=_RULE_DESCRIPTIONS[RULE_CONTROL_ROW],
                    n_records=len(control),
                    items=control,
                ),
            ],
        )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as fh:
            fh.write(drop_log.model_dump_json(indent=2))
        log.info(
            "Cachera: %d records (%d under a systematic name, %d under a released "
            "symbol), %d rows without a gene or a level, %d control rows %s",
            len(rows),
            n_systematic,
            len(rows) - n_systematic,
            len(no_gene_or_level),
            len(control),
            sorted(control),
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
        """Build the Metabolite experiment/reference/publication for one released row."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        )
        genotype = Genotype(
            perturbations=[_released_deletion(row["gene"]), *_betaxanthin_cassette()]
        )
        # Issue #509: YPD + G418 screen plate (line 117), no stated temperature.
        environment = Environment(
            media=CACHERA_YPD_G418, temperature=None, provenance_gaps=[TEMPERATURE_GAP]
        )
        phenotype = MetabolitePhenotype(
            metabolite_level={TARGET_METABOLITE: row["level"]},
            metabolite_level_se={TARGET_METABOLITE: row["se"]},
            n_replicates={TARGET_METABOLITE: int(row["n"])},
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=None,
        )
        # Reference = the CRI-SPA Donor background (4 betaxanthin genes, no extra
        # deletion). Scores are population-centered, so the control level is 0.
        phenotype_reference = MetabolitePhenotype(
            metabolite_level={TARGET_METABOLITE: 0.0},
            metabolite_level_se=None,
            n_replicates={TARGET_METABOLITE: 1},
            measurement_type=MEASUREMENT_TYPE,
        )
        experiment = MetaboliteExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        reference = MetaboliteExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
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
