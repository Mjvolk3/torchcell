# torchcell/datasets/scerevisiae/cooper2010
# [[torchcell.datasets.scerevisiae.cooper2010]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/cooper2010
# Test file: tests/torchcell/datasets/scerevisiae/test_cooper2010.py
"""Cooper 2010 amino-acid metabolome of the yeast deletion collection (CE-LIF).

Cooper et al. 2010 (Genome Res 20:1288-1296, doi:10.1101/gr.105825.110, PMID 20610602)
profiled the free amine-containing metabolite pools of the haploid MATa deletion
collection (Open Biosystems YSC1053) after overnight growth in synthetic complete medium:
cold-methanol extracts were derivatized with NBD-F, separated by capillary electrophoresis
and detected by laser-induced fluorescence (CE-LIF), and every trace was aligned to a
curated template so that 18 numbered peaks could be assigned by spike-in standards. The
released per-strain table is Genome Research Supplemental Table 4, and the deposited copy
of it is the ONLY file this loader consumes.

WHAT SUPPLEMENTAL TABLE 4 HOLDS (measured on the deposited file, sha256
3c56cd492b4cab51249358e90c7e9731982960eb619e45b8850093e616a471fb): 4,382 rows x 19
tab-separated columns, ``Sysname``, ``NAME``, then 17 peak columns named by the authors'
template rather than by Fig. 1: ``RPS19``, ``arg1``, ``gshbiotin``, ``NacOrn``,
``LeuIleCit``, ``GlnVal``, ``MetPro``, ``Thr``, ``Ala``, ``Ser``, ``Asn-Tyr``, ``Gly``,
``LysA``, ``Orn``, ``LysB``, ``Glu``, ``Asp``. Missing cells are ``-``. The 17 headers are
the 17 column labels of Fig. 4B (Biotin, Threonine, N-acetylOrnithine, Leu,Ile,Cit,
Asn,Tyr, Lysine-related, LysineA, LysineB, Gln,Val, Arginine, Met,Pro, Ornithine, Alanine,
Glycine, Serine, Glu, Asp; the Fig. 4 caption says *"Columns represent each amino acid as
labeled in B."*), matched one to one by name, which is how ``RPS19`` is read as the
"lysine-related" peak 1 of Fig. 1 (the metabolite that accumulates in rps19a/rps19b
strains, paper.md line 88), ``arg1`` as the arginine peak, ``gshbiotin`` as Fig. 4B's
"Biotin" column, and ``LeuIleCit`` as a single released column for Fig. 1's separate
peaks 3 (citrulline) and 4 (leucine+isoleucine). ``PEAK_KEYS`` fixes the header -> key
map and ``process()`` refuses any header set that is not exactly this one.

LEGEND VS VALUES: the deposited legend reads *"Supplemental Table 4. Log2 transformed
ratios representing fold change compared to average for each identified amino acid in
each of nearly 4500 samples."* and the Methods say a log2-transformed ratio to the plate
average was calculated. The released numbers are NOT log2 values: every value is >= 0,
the minima are exactly 0 (201 zeros in ``RPS19``, 236 in ``NacOrn``, 74 in ``Ser``), the
maxima reach 83, and the column means of the nine dense peaks (``arg1``, ``LeuIleCit``,
``GlnVal``, ``MetPro``, ``Ala``, ``Asn-Tyr``, ``Gly``, ``LysA``, ``LysB``) sit within 0.03
of 1.0, which is what a LINEAR ratio to the plate mean gives. The values are therefore
stored exactly as released with ``measurement_type`` naming the linear ratio, the
reference is 1.0 per key (a strain equal to its plate average), and a released 0 is a
released value (a peak of zero area), never a missing cell. Five columns have means
above 1 with no stated reason (``RPS19`` 2.23, ``Glu`` 2.15, ``Asp`` 1.99,
``gshbiotin`` 1.53, ``NacOrn`` 1.30); that observation is recorded here and in the loader
note, not corrected.

RECORD SHAPE: one ``MetaboliteExperiment`` per kept Table 4 row. Missingness is ragged
(``RPS19`` is missing in 2,348 rows, ``gshbiotin`` 1,502, ``NacOrn`` 1,016, ``Asp`` 361,
``Thr`` 338, ``Ser`` 322, ``Orn`` 285; the dense peaks miss fewer than 40), so a record
carries exactly the keys its row has (``MetabolitePhenotype`` forbids NaN and requires
matching ``n_replicates`` keys), and its reference carries the same keys at 1.0.
``n_replicates`` is the conservative lower end 1 for every key: strains were screened in
duplicate and the two replicates averaged, but *"In the case where only one quality
trace was collected, those data were used alone."* and no per-row replicate count is
released. Supplemental Tables 3A/3B list the integrated peaks of the first and second
collection, but they are not a per-row join (two Table 4 rows are in neither after
identifier normalization, 46 duplicated identifiers cannot be paired row to row, and
membership counts collected traces rather than traces that passed the 0.35 template
correlation cut), so they are not consumed. ``metabolite_level_se`` is a typed
``ProvenanceGap`` on every phenotype.

RETENTION RULES (every count written to ``preprocess/dropped_records.json``):

- Identifier normalization: five cells are malformed by whitespace, case or a misplaced
  suffix (``YJL038C `` , ``YLR228 C``, ``YML009c``, ``YMR062 C``, ``YML048WA-``); they
  are normalized (whitespace removed, upper-cased, trailing ``WA-`` -> ``W-A``) BEFORE
  the resolver, and every normalization is recorded with the verbatim cell.
- Resolver retention (the shared ``SCerevisiaeGenome.resolve_gene_name``): CURRENT kept
  as is; RENAMED kept under the current systematic name with the normalized source name
  as ``perturbed_gene_name`` (a merged-ORF strain stays a distinct strain of that gene);
  NON_GENE_FEATURE, RETIRED and AMBIGUOUS go to the ``dropped_orfs`` ledger.
- Duplicate identifiers: 46 identifiers occur twice (92 rows), replicate strains of the
  collection with different values. They are NOT averaged. The metabolite verifier's L1
  ``genotype_uniqueness`` keys a record on ``(systematic_gene_name, perturbation_type,
  perturbed_gene_name)`` and ``KanMxDeletionPerturbation`` carries no strain
  discriminator, so a second row cannot be a distinct served record without a schema or
  verifier change; the FIRST row in file order is kept and the second is written, with
  all of its values, to the ``duplicate_strain_rows`` ledger. Re-admitting those rows is a
  follow-up that needs a strain_id-aware L1 (as ``verification/fitness.py`` has).
- Non-deletion rows: a parent-strain (BY4742) row would go to ``excluded_non_deletion``
  (a wild type has no perturbation and cannot be a deletion record); Table 4 carries
  none. Tet-promoter (YSC1182) strains are essential-gene knockdowns under doxycycline,
  not deletions, and would be excluded the same way; Table 4 carries no canonical
  essential gene (CDC28, ACT1, TUB2, RPB1, FAS1, FAS2, GLC7, CMD1 and CDC42 are all
  absent), so no tet-promoter row is present and the exclusion count is 0.
- SGD essentiality flag: the local ``gene_essentiality_sgd`` store is consulted and every
  kept ORF it lists is written to the ``essentiality_flagged`` ledger and KEPT. The store
  holds SGD "inviable" null-phenotype annotations, which include conditionally inviable
  genes (the 21 hits are ATG1, ATG2, ATG3, ATG5, ATG7, ATG8, ATG9, ATG10, ATG12, ATG15,
  ATG16, ATG17, ATG18, ERG24, HHT2, PLC1, RPS28A, SAC1, SPC72, VPS30, YMR185W, all viable
  deletion-collection strains; the deletion-only Mulleder 2016 set carries 28 of the same
  kind), so the flag is diagnostic, not an exclusion.

ENVIRONMENT: ``COOPER_SC``, a loader-local ``dropout(SC, ...)`` medium (never a
``media.py`` edit, which would move the value surface and block incremental admission):
the paper's recipe sentence lists 15 supplements and no uracil, and alanine, asparagine,
cysteine, glutamine, glycine and proline are absent, so those seven are typed dropouts of
the library ``SC`` and the medium joins Mulleder's and Hillenmeyer's at ``base_medium ==
"SC"``. The 15 stated concentrations stay in the quote only: serine 3.64, threonine 1.6
and valine 1.19 uM are two to three orders below the rest and were never corrected.
Growth is ~16 h in deep-well plates; no growth temperature is stated anywhere in the
paper, so ``temperature=None`` with a typed gap. The parent is stored as ``BY4741``: the
paper names ``BY4742`` twice but calls the YSC1053 strains MATa, and YSC1053 is the MATa
(BY4741-background) collection; ``PARENT_STRAIN_QUOTE`` carries the discrepancy.

DATA SOURCE (manual, sha256-pinned): the Genome Research supplement is behind an
institutional login (every scripted attempt returned 429/403/404 on 2026-09-15), so the
retrieval record is a genuine ``manual_browser`` recipe and the canonical copy is the RAW
MIRROR ``$DATA_ROOT/torchcell-raw/cooperHighthroughputProfilingAmino2010/`` with its
``manifest.json``. ``download()`` symlinks and sha256-verifies from the mirror and has
NO network path: when the mirror is absent it raises with the recipe.
"""

import csv
import json
import logging
import os
import os.path as osp
import pickle
import re
import shutil
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lmdb
from pydantic import BaseModel, ConfigDict, Field

from torchcell.data import ExperimentDataset, post_process
from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datamodels.media import SC, dropout
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.sequence.genome.scerevisiae.s288c import GeneNameStatus
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1101/gr.105825.110"
PMID = "20610602"
CITATION_KEY = "cooperHighthroughputProfilingAmino2010"

# The mirrored OCR markdown every paper quote below is anchored to.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "b7739b1eeaf68b609a3e13b870f28649c331fe36b3c930bbbfd8a4c4fd19e52a"

# The deposited supplement files this loader consumes, under ``data/`` in the raw mirror.
TABLE4_NAME = "SupplementalTable4.txt"
TABLE4_SHA256 = "3c56cd492b4cab51249358e90c7e9731982960eb619e45b8850093e616a471fb"
LEGENDS_NAME = "Supplemental_Table_Legends.doc"
LEGENDS_SHA256 = "ca4983cda4c34318dbc3a05d10edf8e05df11f1560ffc99442df83a6a4399184"
_RAW_FILES: dict[str, dict[str, str]] = {
    TABLE4_NAME: {"relpath": f"data/{TABLE4_NAME}", "sha256": TABLE4_SHA256},
    LEGENDS_NAME: {"relpath": f"data/{LEGENDS_NAME}", "sha256": LEGENDS_SHA256},
}
SHA256SUMS_NAME = "SHA256SUMS.txt"

ARTICLE_URL = "https://genome.cshlp.org/content/20/9/1288"
DC1_URL = "http://genome.cshlp.org/content/suppl/2010/07/07/gr.105825.110.DC1"
RETRIEVED_AT = "2026-09-15"

#: The manual recipe that produced the deposited bytes, recorded as the
#: ``retrieval_command`` of a typed ``manual_browser`` record: un-scriptable, but not
#: unknown, so the manifest is ``provenance_complete=True`` and a rebuild re-runs it by
#: hand and verifies the sha256.
MANUAL_RECIPE = (
    "manual browser download -- the Genome Research supplement is behind an institutional "
    "login wall and every scripted request to the DC1 page returned HTTP 429/403/404 or a "
    f"login redirect (measured 2026-09-15), so: open {ARTICLE_URL} in a browser signed in "
    f"through an institutional subscription, follow 'Supplemental Material' ({DC1_URL}), "
    f"download '{TABLE4_NAME}' and '{LEGENDS_NAME}' unchanged, and verify sha256 "
    f"{TABLE4_SHA256} and {LEGENDS_SHA256}"
)

MEASUREMENT_TYPE = "ce_lif_peak_area_ratio_to_plate_mean"

MEASUREMENT_UNITS = (
    "linear ratio of the strain's NBD-F peak area to the average peak area of its 96-well "
    "plate (Cooper 2010 Supplemental Table 4, replicate traces averaged); 1 = equal to the "
    "plate average, 0 = a peak of zero area; the deposited legend says log2 but the "
    "released values are the linear ratio"
)

#: Table 4 header -> stored metabolite key, in the file's column order. Composite keys
#: are peaks the platform did not separate; ``gshbiotin``, ``lysine_a`` and ``lysine_b``
#: keep the released label because the paper does not resolve them further.
PEAK_KEYS: dict[str, str] = {
    "RPS19": "lysine_related_peak1",
    "arg1": "arginine",
    "gshbiotin": "gshbiotin",
    "NacOrn": "n_acetylornithine",
    "LeuIleCit": "leucine+isoleucine+citrulline",
    "GlnVal": "glutamine+valine",
    "MetPro": "methionine+proline",
    "Thr": "threonine",
    "Ala": "alanine",
    "Ser": "serine",
    "Asn-Tyr": "asparagine+tyrosine",
    "Gly": "glycine",
    "LysA": "lysine_a",
    "Orn": "ornithine",
    "LysB": "lysine_b",
    "Glu": "glutamate",
    "Asp": "aspartate",
}
PEAK_ORDER: tuple[str, ...] = tuple(PEAK_KEYS.values())
IDENTIFIER_COLUMN = "Sysname"
NAME_COLUMN = "NAME"
EXPECTED_HEADER: tuple[str, ...] = (IDENTIFIER_COLUMN, NAME_COLUMN, *PEAK_KEYS)
MISSING_CELL = "-"

#: How each Table 4 header was read (the Fig. 4B column label it matches, and the Fig. 1
#: peak where one is stated). Recorded, never used to alter a value.
PEAK_NOTES: dict[str, str] = {
    "RPS19": "Fig. 4B column 'Lysine-related' = Fig. 1 peak 1 'Lysine-related'; the "
    "unidentified metabolite that accumulates in rps19a/rps19b strains; the header names "
    "the strain, not a compound",
    "arg1": "Fig. 4B column 'Arginine' = Fig. 1 peak 2; the header names the arg1 mutant "
    "that lacks arginine",
    "gshbiotin": "Fig. 4B column 'Biotin'; glutathione and biotin were both spike-in "
    "standards; which compound(s) the peak carries is not stated, so the released label "
    "is kept and the absence is a typed gap on target_metabolite_ids",
    "NacOrn": "Fig. 4B column 'N-acetylOrnithine'; N-acetyl ornithine was a spike-in "
    "standard and is named as accumulating in slow growers",
    "LeuIleCit": "Fig. 4B column 'Leu,Ile,Cit'; Fig. 1 draws citrulline (peak 3) and "
    "leucine+isoleucine (peak 4) separately but the released table has one column",
    "GlnVal": "Fig. 4B 'Gln,Val' = Fig. 1 peak 5; glutamine does not separate from valine",
    "MetPro": "Fig. 4B 'Met,Pro' = Fig. 1 peak 6",
    "Thr": "Fig. 4B 'Threonine' = Fig. 1 peak 8",
    "Ala": "Fig. 4B 'Alanine' = Fig. 1 peak 7",
    "Ser": "Fig. 4B 'Serine' = Fig. 1 peak 10",
    "Asn-Tyr": "Fig. 4B 'Asn,Tyr' = Fig. 1 peak 9",
    "Gly": "Fig. 4B 'Glycine' = Fig. 1 peak 11",
    "LysA": "Fig. 4B 'LysineA'; lysine gives two peaks (Fig. 1 peaks 12 and 14, once- and "
    "twice-labeled); which of A/B is which peak is not stated",
    "Orn": "Fig. 4B 'Ornithine' = Fig. 1 peak 13",
    "LysB": "Fig. 4B 'LysineB'; see LysA",
    "Glu": "Fig. 4B 'Glu' = Fig. 1 peak 17",
    "Asp": "Fig. 4B 'Asp' = Fig. 1 peak 18",
}

#: Retention rule for a row's ORF: kept only when it resolves to a LIVE R64 gene.
_KEPT_STATUSES = frozenset({GeneNameStatus.CURRENT, GeneNameStatus.RENAMED})

ORF_RULE = (
    "a row is kept only when its normalized 'Sysname' resolves through the shared "
    "SCerevisiaeGenome.resolve_gene_name to CURRENT (stored as is) or RENAMED (stored "
    "under the current systematic name, the normalized source name kept as "
    "perturbed_gene_name so a merged-ORF strain stays distinct); a NON_GENE_FEATURE, a "
    "RETIRED name or an AMBIGUOUS name drops the row with its status in the ledger"
)
NORMALIZATION_RULE = (
    "the 'Sysname' cell has all whitespace removed and is upper-cased, and a trailing "
    "'WA-' is rewritten 'W-A' (YML048WA- -> YML048W-A), before the resolver; every "
    "changed cell is recorded verbatim"
)
DUPLICATE_RULE = (
    "an identifier occurring on more than one row (replicate strains of the collection, "
    "different values) keeps its FIRST row in file order as the served record; later "
    "rows are written to the duplicate_strain_rows ledger with all their values, never "
    "averaged, because the metabolite verifier keys L1 uniqueness on the deletion set "
    "and KanMxDeletionPerturbation carries no strain discriminator"
)
EXCLUSION_RULE = (
    "a row is excluded when its identifier or NAME names the parent strain (BY4742, WT, "
    "wild type) rather than a deletion, or when it is a tet-promoter (YSC1182) "
    "essential-gene knockdown, which is not a deletion; Table 4 carries neither (no "
    "canonical essential gene is present), measured 0"
)
ESSENTIALITY_RULE = (
    "every kept ORF listed in the local gene_essentiality_sgd store (SGD inviable null "
    "phenotypes, which include conditionally inviable genes such as ATG1) is flagged and "
    "KEPT; the flag is diagnostic because the store lists viable deletion-collection "
    "strains and Table 4 holds no essential-gene knockdown"
)
REPLICATE_RULE = (
    "n_replicates = 1 for every key (conservative lower end): strains were screened in "
    "duplicate and averaged, but a strain with only one quality trace was used alone and "
    "no per-row replicate count is released; Supplemental Tables 3A/3B do not join row "
    "to row and count collected rather than quality traces"
)
LEGEND_CONTRADICTION = (
    "the deposited legend and the Methods call Table 4 log2-transformed ratios; the "
    "released values are non-negative with exact zeros, maxima up to 83 and dense-column "
    "means within 0.03 of 1.0, i.e. the linear ratio to the plate mean; stored as "
    "released with the measurement type naming the linear ratio and reference 1.0"
)


# --------------------------------------------------------------------------- #
# Sourced metadata. Every number this loader hardcodes is anchored to a verbatim
# quote from the mirrored paper.md (or the deposited legend) plus its sha256.
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


_RECIPE_QUOTE = (
    "Yeast growth was in synthetic complete media (adenine $1 4 0 ~ \\mu \\mathrm { M }$ "
    ", arginine $1 0 9 \\ \\mu \\mathrm { M } ,$ aspartic acid $7 2 0 ~ { \\mu \\mathrm "
    "{ M } } ,$ glutamic acid $6 5 1 ~ \\mu \\mathrm { M } ,$ histidine $1 2 2 \\ \\mu "
    "\\mathrm { M } _ { i }$ isoleucine $5 7 9 ~ \\mu \\mathrm { M } _ { i }$ leucine "
    "$5 7 9 \\ \\mu \\mathrm { M } ,$ , lysine $3 9 0 ~ \\mu \\mathrm { M } _ { \\mathrm "
    "{ { \\ell } } }$ , methionine $1 2 7 \\ \\mu \\mathrm { M } ,$ phenylalanine "
    "$2 8 9 ~ \\mu \\mathrm { M } ,$ , serine $3 . 6 4 ~ \\mu \\mathrm { M }$ , threonine "
    "1.6 $\\mu \\mathrm { M } ,$ , tryptophan $3 7 2 \\mu \\mathrm { M } _ { i }$ , "
    "tyrosine $3 1 4 ~ { \\mu \\mathrm { M } } _ { \\mathrm { \\Omega } }$ , and valine "
    "$1 . 1 9 ~ \\mu \\mathrm { M } )$ ."
)

PARENT_STRAIN_QUOTE = _sv(
    "BY4742 (as written) / BY4741 (as stored)",
    "Compared with the parent strain, BY4742, arginine mutants had only one-quarter the "
    "arginine in the extract (Fig. 3).",
    note="the paper names the parent BY4742 here and in the spike-in methods, but calls "
    "the YSC1053 deletion strains MATa, and YSC1053 is the MATa (BY4741-background) "
    "collection; ReferenceGenome.strain is stored as BY4741 and this quote flags the "
    "discrepancy; a one-line change flips it",
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "collection": _sv(
        "YSC1053",
        "Haploid yeast deletion strains of the MATa mating type were obtained from Open "
        "Biosystems (YSC1053), originally constructed as part of the Saccharomyces Genome "
        "Deletion Project (Winzeler et al. 1999).",
        note="the KanMX deletion collection of the Saccharomyces Genome Deletion Project; "
        "the marker is not named in this paper and follows from the Winzeler 1999 "
        "citation",
    ),
    "tet_promoter_collection": _sv(
        "YSC1182",
        "We obtained the tet-promoter collection allowing regulation of essential genes "
        "by doxycycline from Open Biosystems (YSC1182) (Mnaimneh et al. 2004).",
        note="essential-gene knockdowns, not deletions; out of scope, and Table 4 carries "
        "no essential-gene row",
    ),
    "parent_strain": PARENT_STRAIN_QUOTE,
    "medium_recipe": _sv(
        "synthetic complete: 15 listed supplements, no uracil, no Ala/Asn/Cys/Gln/Gly/Pro",
        _RECIPE_QUOTE,
        note="the seven absent SC components are typed dropouts of the library SC; the "
        "15 concentrations are not typed because serine 3.64, threonine 1.6 and valine "
        "1.19 uM are two to three orders below the rest and were never corrected",
    ),
    "duration_hours": _sv(
        16.0,
        "In brief, cells were grown for ${ \\sim } 1 6 \\mathrm { h }$ in synthetic "
        "complete media in deepwell 96 well plates (VWR).",
        note="the tilde is the source's; overnight growth in deep-well 96-well plates",
    ),
    "replicates": _sv(
        2,
        "Strains were screened in duplicate, starting from fresh yeast colonies.",
        note="the screen design; the stored n_replicates is the conservative 1 because a "
        "strain with one quality trace was used alone and no per-row count is released",
    ),
    "replicate_averaging": _sv(
        1,
        "An average of the two replicates was calculated (Supplemental Table 4). In the "
        "case where only one quality trace was collected, those data were used alone.",
        note="n_replicates = 1 per key, the conservative lower end of one or two traces",
    ),
    "normalization": _sv(
        "ratio to the plate average",
        "For each sample, a $\\log _ { 2 }$ -transformed ratio of the amino acid quantity "
        "in the sample to the average for the plate was calculated. This ratio corrects "
        "for plate effects.",
        note=LEGEND_CONTRADICTION,
    ),
    "quantification": _sv(
        "peak area",
        "The resulting output is a quantification of peak area correlating to relative "
        "amino acid concentration in each profile.",
    ),
    "detection": _sv(
        "CE-LIF",
        "Capillary electrophoresis was used to separate the derivatized samples and "
        "detection was achieved by laser-induced fluorescence.",
    ),
    "derivatization": _sv(
        "NBD-F",
        "We performed a cold methanol extraction of small molecules from the yeast cells "
        "and derivatized the extracts with the amine-reactive fluorophore, "
        "4-nitro-7-benzofurazan (NBD-F) (Villas-Boas et al. 2005; Zhu et al. 2005).",
    ),
    "qc_threshold": _sv(
        0.35,
        "Following alignment of all the traces, we eliminated data for which correlation "
        "with the template was below 0.35.",
        note="applied by the authors before Table 4 was released; no record is dropped on "
        "it here",
    ),
    "n_samples": _sv(
        4382,
        "We analyzed 4382 samples with data meeting quality standards for at least one "
        "replicate.",
        note="equals the Table 4 row count before resolver drops and duplicate rows",
    ),
    "n_clustered_strains": _sv(
        4337, "we ordered 4337 yeast deletion strains using hierarchical clustering"
    ),
    "median_od600": _sv(
        0.370,
        "The median optical density measurement was 0.370",
        note="note material; no record is filtered on growth",
    ),
    "peak_assignment": _sv(
        "18 numbered peaks by spike-in",
        "Peaks were assigned by spike-in experiments and are labeled "
        "$\\scriptstyle 1 - 1 8 ,$ corresponding to the compounds shown.",
        note="the 1-18 list is inside the Fig. 1 image; Table 4 releases 17 columns "
        "matching the Fig. 4B labels",
    ),
    "lysine_two_peaks": _sv(
        "lysine_a, lysine_b",
        "(Lysine has two peaks as it has two reactive amine groups and therefore can be "
        "labeled once or twice.)",
    ),
    "lysine_peaks_reported_separately": _sv(
        "lysine peak 1 and peak 2 correlate with OD600 independently",
        "The greatest correlations we identified were between growth rate (measured "
        "optical densities) and lysine levels (lysine peak 1 versus "
        "$\\mathrm { O D } _ { 6 0 0 }$ , $\\mathrm { R } = - 0 . 3 2$ and peak "
        "$2 , \\mathrm { R } = - 0 . 3 7 )$ .",
        note="why the two lysine peaks are two keys rather than one",
    ),
    "glutamine_valine": _sv(
        "glutamine+valine",
        "The one amino acid that did not show a high correlation was glutamine, which "
        "does not separate from valine by capillary electrophoresis under these "
        "conditions.",
    ),
    "asparagine_tyrosine": _sv("asparagine+tyrosine", "<td>Asparagine + tyrosine</td>"),
    "fig4b_columns": _sv(
        "the 17 Table 4 headers match the 17 Fig. 4B column labels",
        "Columns represent each amino acid as labeled in B.",
        note="the bridge from the template headers (RPS19, arg1, gshbiotin, NacOrn, ...) "
        "to the peak identities",
    ),
    "lysine_related_metabolite": _sv(
        "lysine_related_peak1",
        "A striking example of the accumulation of the lysine-related metabolite occurs "
        "in strains lacking the ribosomal protein gene RPS19A or RPS19B (Fig. 5A).",
        note="why the header RPS19 is read as the lysine-related peak",
    ),
    "n_acetylornithine": _sv(
        "n_acetylornithine",
        "we confirmed our initial finding and additionally identified significant "
        "accumulations of ornithine, lysine, leucine, and $N \\cdot$ -acetyl ornithine, "
        "as well as depletion of glutamine in slow growers (Supplemental Table 1).",
    ),
    "spike_in_standards": _sv(
        "glutathione and biotin were spike-in standards",
        "The following amine-containing molecules were tested: all 20 coding amino acids, "
        "spermidine, spermine, ornithine, citrulline, glutathione, biotin, creatine, $N$ "
        "-acetyl lysine, $N \\cdot$ -acetyl ornithine, $N \\cdot$ -acetyl aspartate, "
        "diaminobutane, carnitine, and carnosine (all from Sigma-Aldrich).",
        note="the only text behind the header gshbiotin",
    ),
    "log_ratios_clustered": _sv(
        "log-transformed ratios (as described)",
        "hierarchical clustering of the log-transformed ratios",
        note="the paper's own description of Table 4's transform, contradicted by the "
        "released values",
    ),
}

#: The one quote taken from the deposited legend rather than from paper.md.
LEGEND_QUOTE = (
    "Supplemental Table 4. Log2 transformed ratios representing fold change compared to "
    "average for each identified amino acid in each of nearly 4500 samples."
)
LEGEND_SOURCED_VALUES: dict[str, SourcedValue] = {
    "table4_legend": SourcedValue(
        value="log2 (as written) / linear ratio (as released)",
        provenance=Provenance(
            source_uri=_RAW_FILES[LEGENDS_NAME]["relpath"],
            citation_key=CITATION_KEY,
            sha256=LEGENDS_SHA256,
            method="manual browser deposit of the Genome Research supplement; the legend "
            "text is 8-bit in the Word binary",
        ),
        quote=LEGEND_QUOTE,
        note=LEGEND_CONTRADICTION,
    )
}

_TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="growth temperature is not stated; the only degree values in the paper are GC-MS "
    "derivatization and oven settings",
)
_SE_GAP = ProvenanceGap(
    field="metabolite_level_se",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="duplicates were averaged; no per-strain spread is released",
)
_TARGET_IDS_NOTE = (
    "amino-acid keys -> Yeast9 s_NNNN ids are to be sourced from YeastGEM in a follow-up, "
    "never guessed (Mulleder 2016 defers the same mapping)"
)
_TARGET_IDS_GSHBIOTIN_NOTE = (
    f"{_TARGET_IDS_NOTE}; the key gshbiotin has no metabolite identity to map: "
    f"{PEAK_NOTES['gshbiotin']}"
)


def _phenotype_gaps(keys: Sequence[str]) -> list[ProvenanceGap]:
    """The typed absences a phenotype with these keys declares.

    ``target_metabolite_ids`` is the field where a key's compound identity would live, so
    the unexplained ``gshbiotin`` header is recorded there (a gap must name a real field
    that is ``None``; ``metabolite_level`` holds a value and cannot be gapped).
    """
    return [
        _SE_GAP,
        ProvenanceGap(
            field="target_metabolite_ids",
            reason=ProvenanceGapReason.deferred_pending_source_review,
            note=(
                _TARGET_IDS_GSHBIOTIN_NOTE if "gshbiotin" in keys else _TARGET_IDS_NOTE
            ),
        ),
    ]


# --------------------------------------------------------------------------- #
# The medium: the paper's synthetic complete recipe as a typed edit of library SC
# --------------------------------------------------------------------------- #
COOPER_SC_DROPOUTS: tuple[str, ...] = (
    "L-alanine",
    "L-asparagine",
    "L-cysteine",
    "L-glutamine",
    "glycine",
    "L-proline",
    "uracil",
)

COOPER_SC: Media = dropout(
    SC,
    *COOPER_SC_DROPOUTS,
    name="SC minus Ala/Asn/Cys/Gln/Gly/Pro/Ura (Cooper 2010 synthetic complete recipe)",
    provenance=[SOURCED_VALUES["medium_recipe"]],
)
"""Cooper 2010's synthetic complete medium: library ``SC`` minus the seven components the
recipe sentence does not list. Loader-local on purpose: ``media.py`` is a value surface of
the served graph, and editing it blocks incremental admission."""


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/<citation_key>/`` -- the canonical raw copy."""
    root = data_root if data_root is not None else os.environ["DATA_ROOT"]
    return Path(root) / "torchcell-raw" / CITATION_KEY


def _read_sha256sums(path: Path) -> dict[str, str]:
    """``{file name: sha256}`` from a ``sha256sum``-format listing."""
    sums: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        sums[name.strip()] = digest
    return sums


def deposit_raw_mirror(
    *,
    source_dir: str | Path,
    retrieved_at: str = RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from the user's browser download plus its ``manifest.json``.

    Run once by hand. ``source_dir`` is the staged deposit holding the supplement files
    unchanged and a ``SHA256SUMS.txt``; only the two files the loader consumes are
    deposited, under ``data/``. Idempotent by sha256: a mirror file already carrying the
    pinned hash is left alone, one carrying a different hash raises. Every file's sha256
    is checked against the pinned constant AND against ``SHA256SUMS.txt`` before it is
    recorded, so the manifest can only describe the bytes the build consumed.
    """
    source = Path(source_dir)
    sums = _read_sha256sums(source / SHA256SUMS_NAME)
    root = raw_mirror_dir(data_root)
    files: list[ArtifactRecord] = []
    for name, spec in _RAW_FILES.items():
        if sums[name] != spec["sha256"]:
            raise RuntimeError(
                f"{SHA256SUMS_NAME} lists {name} as {sums[name]}, pinned {spec['sha256']}"
            )
        staged = source / name
        digest = sha256_file(staged)
        if digest != spec["sha256"]:
            raise RuntimeError(
                f"{staged} sha256 {digest} != pinned {spec['sha256']}; refusing to deposit"
            )
        dest = root / spec["relpath"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            existing = sha256_file(dest)
            if existing != spec["sha256"]:
                raise RuntimeError(
                    f"{dest} exists with sha256 {existing} != pinned {spec['sha256']}; "
                    "refusing to overwrite"
                )
        else:
            shutil.copy2(staged, dest)
        files.append(
            ArtifactRecord(
                path=spec["relpath"],
                role=ROLE_RAW_DATA if name == TABLE4_NAME else ROLE_SI_DATA,
                bytes=dest.stat().st_size,
                sha256=spec["sha256"],
                source=DC1_URL,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.manual_browser,
                    source_url=ARTICLE_URL,
                    retriever="manual",
                    params={
                        "retrieval_command": MANUAL_RECIPE,
                        "supplement_page": DC1_URL,
                        "retrieved_by": "the user, browser download with institutional "
                        "login, copied to the staging directory unchanged (cp -p) and "
                        "listed in SHA256SUMS.txt",
                    },
                    sha256=spec["sha256"],
                    retrieved_at=retrieved_at,
                ),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=(
            "High-throughput profiling of amino acids in strains of the Saccharomyces "
            "cerevisiae deletion collection"
        ),
        files=files,
        si_data_sources=[DC1_URL],
        si_expected=[
            "Supplemental Table 1 (SupplementalTable1.xls; growth-rate effects)",
            "Supplemental Tables 2A,B (raw electropherograms; not deposited)",
            "Supplemental Tables 3A,B (integrated peaks per replicate; not deposited)",
            f"Supplemental Table 4 ({TABLE4_NAME}; consumed)",
            f"Supplemental Table Legends ({LEGENDS_NAME}; consumed for the Table 4 legend)",
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


# --------------------------------------------------------------------------- #
# Typed build-side records (written beside the LMDB, not into it)
# --------------------------------------------------------------------------- #
class TableRow(BaseModel):
    """One Table 4 row after parsing and identifier normalization."""

    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(description="0-based data row in the deposited file")
    source_name: str = Field(description="the 'Sysname' cell, verbatim")
    normalized_name: str = Field(description="the identifier after NORMALIZATION_RULE")
    name_cell: str = Field(description="the 'NAME' cell, verbatim")
    levels: dict[str, float] = Field(
        description="metabolite key -> released value, present keys only"
    )


class KeptRow(TableRow):
    """A retained row with what the resolver made of its identifier."""

    systematic_gene_name: str
    perturbed_gene_name: str
    status: str


class IdentifierNormalization(BaseModel):
    """One 'Sysname' cell the normalization rule changed."""

    model_config = ConfigDict(extra="forbid")

    row_index: int
    source_name: str
    normalized_name: str


class DroppedOrf(BaseModel):
    """One source identifier the retention rule removed, with what the resolver said."""

    model_config = ConfigDict(extra="forbid")

    row_index: int
    source_name: str = Field(description="the 'Sysname' cell, verbatim")
    normalized_name: str
    status: str = Field(description="GeneNameStatus the shared resolver returned")
    resolved_to: str | None = Field(
        default=None, description="what the resolver mapped it to, when anything"
    )
    feature_type: str | None = Field(
        default=None, description="GFF feature type for a NON_GENE_FEATURE"
    )
    n_values: int = Field(description="present peak values lost with this row")


class DuplicateStrainRow(BaseModel):
    """A row whose identifier already has a served record, with all of its values."""

    model_config = ConfigDict(extra="forbid")

    row_index: int
    source_name: str
    normalized_name: str
    systematic_gene_name: str
    name_cell: str
    kept_row_index: int = Field(description="the served row of the same identifier")
    levels: dict[str, float]


class ExcludedRow(BaseModel):
    """A row that is not a deletion-collection strain, with its values."""

    model_config = ConfigDict(extra="forbid")

    row_index: int
    source_name: str
    name_cell: str
    reason: str
    levels: dict[str, float]


class EssentialityFlag(BaseModel):
    """A kept ORF the SGD essentiality store lists (diagnostic, not an exclusion)."""

    model_config = ConfigDict(extra="forbid")

    row_index: int
    systematic_gene_name: str
    name_cell: str


class BuildLedger(BaseModel):
    """The build's retention accounting: the rules, and exactly what they did."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    orf_rule: str
    normalization_rule: str
    duplicate_rule: str
    exclusion_rule: str
    essentiality_rule: str
    replicate_rule: str
    legend_contradiction: str
    n_source_rows: int
    n_kept: int
    n_dropped_orfs: int
    n_renamed: int
    n_normalized: int
    n_duplicate_rows: int
    n_excluded_non_deletion: int
    n_essentiality_flagged: int
    n_values_kept: int
    normalizations: list[IdentifierNormalization]
    dropped_orfs: list[DroppedOrf]
    renamed_orfs: dict[str, str] = Field(
        description="normalized source name -> current systematic name for every RENAMED "
        "row kept; the source name stays on the record as perturbed_gene_name"
    )
    duplicate_strain_rows: list[DuplicateStrainRow]
    excluded_non_deletion: list[ExcludedRow]
    essentiality_flagged: list[EssentialityFlag]
    created_at: str


# --------------------------------------------------------------------------- #
# Source readers
# --------------------------------------------------------------------------- #
_WA_SUFFIX_RE = re.compile(r"^(Y[A-P][LR]\d{3}[CW])A-$")
_PARENT_RE = re.compile(r"BY47|^WT$|WILD.?TYPE", re.IGNORECASE)


def normalize_identifier(cell: str) -> str:
    """Apply ``NORMALIZATION_RULE`` to one 'Sysname' cell."""
    name = re.sub(r"\s+", "", cell).upper()
    match = _WA_SUFFIX_RE.match(name)
    if match is not None:
        name = f"{match.group(1)}-A"
    return name


def read_table4(path: str | Path) -> list[TableRow]:
    """Parse the deposited Table 4; refuse any header that is not exactly the known one."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = tuple(next(reader))
        if header != EXPECTED_HEADER:
            unmapped = [h for h in header if h not in EXPECTED_HEADER]
            missing = [h for h in EXPECTED_HEADER if h not in header]
            raise RuntimeError(
                f"{path} header is not the expected Table 4 header; unmapped columns "
                f"{unmapped}, missing columns {missing}, order {list(header)}"
            )
        rows: list[TableRow] = []
        for row_index, parts in enumerate(reader):
            if len(parts) != len(EXPECTED_HEADER):
                raise RuntimeError(
                    f"{path} row {row_index} has {len(parts)} cells, expected "
                    f"{len(EXPECTED_HEADER)}"
                )
            levels = {
                PEAK_KEYS[column]: float(cell)
                for column, cell in zip(header[2:], parts[2:], strict=True)
                if cell != MISSING_CELL
            }
            rows.append(
                TableRow(
                    row_index=row_index,
                    source_name=parts[0],
                    normalized_name=normalize_identifier(parts[0]),
                    name_cell=parts[1],
                    levels=levels,
                )
            )
    return rows


def load_sgd_essential_genes(data_root: str) -> frozenset[str]:
    """Systematic names whose stored SGD essentiality phenotype is ``is_essential=True``.

    Read from the dev-tree ``gene_essentiality_sgd`` LMDB (SGD inviable null phenotypes).
    The store must exist: the flag is measured against it, never assumed.
    """
    root = osp.join(data_root, "data/torchcell/gene_essentiality_sgd")
    lmdb_dir = osp.join(root, "processed", "lmdb")
    if not osp.isdir(lmdb_dir):
        raise FileNotFoundError(
            f"{lmdb_dir} is absent; build GeneEssentialitySgdDataset first, the "
            "essentiality flag is measured against it"
        )
    interned: dict[str, Any] = {}
    interned_dir = osp.join(root, "processed", "interned")
    if osp.isdir(interned_dir):
        ienv = lmdb.open(interned_dir, readonly=True, lock=False)
        with ienv.begin() as itxn:
            for key, value in itxn.cursor():
                interned[key.decode()] = pickle.loads(value)
        ienv.close()
    essential: set[str] = set()
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    with env.begin() as txn:
        for _, value in txn.cursor():
            record = resolve_interned(pickle.loads(value), interned)
            if record["experiment"]["phenotype"]["is_essential"] is not True:
                continue
            for perturbation in record["experiment"]["genotype"]["perturbations"]:
                essential.add(perturbation["systematic_gene_name"])
    env.close()
    return frozenset(essential)


def retain_rows(
    rows: Sequence[TableRow],
    resolve: Callable[[str], Any],
    essential_genes: frozenset[str],
    *,
    dataset_name: str,
) -> tuple[list[KeptRow], BuildLedger]:
    """Apply every retention rule in order and return the kept rows plus the ledger.

    Order per row: parent-strain exclusion, resolver retention (``ORF_RULE``), duplicate
    identifier (``DUPLICATE_RULE``), essentiality flag (``ESSENTIALITY_RULE``, kept).
    """
    kept: list[KeptRow] = []
    normalizations: list[IdentifierNormalization] = []
    dropped: list[DroppedOrf] = []
    renamed: dict[str, str] = {}
    duplicates: list[DuplicateStrainRow] = []
    excluded: list[ExcludedRow] = []
    flagged: list[EssentialityFlag] = []
    served: dict[tuple[str, str], int] = {}
    for row in rows:
        if row.source_name != row.normalized_name:
            normalizations.append(
                IdentifierNormalization(
                    row_index=row.row_index,
                    source_name=row.source_name,
                    normalized_name=row.normalized_name,
                )
            )
        if _PARENT_RE.search(row.normalized_name) or _PARENT_RE.search(row.name_cell):
            excluded.append(
                ExcludedRow(
                    row_index=row.row_index,
                    source_name=row.source_name,
                    name_cell=row.name_cell,
                    reason="parent-strain row, not a deletion",
                    levels=row.levels,
                )
            )
            continue
        resolution = resolve(row.normalized_name)
        orf = resolution.systematic_name
        if resolution.status not in _KEPT_STATUSES or orf is None:
            dropped.append(
                DroppedOrf(
                    row_index=row.row_index,
                    source_name=row.source_name,
                    normalized_name=row.normalized_name,
                    status=str(resolution.status.value),
                    resolved_to=orf,
                    feature_type=resolution.feature_type,
                    n_values=len(row.levels),
                )
            )
            continue
        if resolution.status == GeneNameStatus.RENAMED:
            renamed[row.normalized_name] = orf
        key = (orf, row.normalized_name)
        if key in served:
            duplicates.append(
                DuplicateStrainRow(
                    row_index=row.row_index,
                    source_name=row.source_name,
                    normalized_name=row.normalized_name,
                    systematic_gene_name=orf,
                    name_cell=row.name_cell,
                    kept_row_index=served[key],
                    levels=row.levels,
                )
            )
            continue
        served[key] = row.row_index
        if orf in essential_genes:
            flagged.append(
                EssentialityFlag(
                    row_index=row.row_index,
                    systematic_gene_name=orf,
                    name_cell=row.name_cell,
                )
            )
        kept.append(
            KeptRow(
                **row.model_dump(),
                systematic_gene_name=orf,
                perturbed_gene_name=row.normalized_name,
                status=str(resolution.status.value),
            )
        )
    ledger = BuildLedger(
        dataset=dataset_name,
        orf_rule=ORF_RULE,
        normalization_rule=NORMALIZATION_RULE,
        duplicate_rule=DUPLICATE_RULE,
        exclusion_rule=EXCLUSION_RULE,
        essentiality_rule=ESSENTIALITY_RULE,
        replicate_rule=REPLICATE_RULE,
        legend_contradiction=LEGEND_CONTRADICTION,
        n_source_rows=len(rows),
        n_kept=len(kept),
        n_dropped_orfs=len(dropped),
        n_renamed=len(renamed),
        n_normalized=len(normalizations),
        n_duplicate_rows=len(duplicates),
        n_excluded_non_deletion=len(excluded),
        n_essentiality_flagged=len(flagged),
        n_values_kept=sum(len(row.levels) for row in kept),
        normalizations=normalizations,
        dropped_orfs=dropped,
        renamed_orfs=dict(sorted(renamed.items())),
        duplicate_strain_rows=duplicates,
        excluded_non_deletion=excluded,
        essentiality_flagged=flagged,
        created_at=datetime.now(UTC).isoformat(),
    )
    return kept, ledger


@register_dataset
class AminoAcidCooper2010Dataset(ExperimentDataset):
    """Cooper 2010 CE-LIF amino-acid pools of the deletion collection (ratio to plate)."""

    def __init__(
        self,
        root: str = "data/torchcell/amino_acid_cooper2010",
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
        return MetaboliteExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return MetaboliteExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The deposited Supplemental Table 4 required before processing."""
        return [TABLE4_NAME]

    # ---- retrieval ------------------------------------------------------------ #
    def download(self) -> None:
        """Link Table 4 from the sha256-verified raw mirror. There is no network path.

        The supplement is behind a login wall, so the mirror (the stored artifact plus
        its sha256) is the only source; an absent mirror raises with the manual recipe.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        dest = osp.join(self.raw_dir, TABLE4_NAME)
        if osp.exists(dest):
            return
        source = raw_mirror_dir() / _RAW_FILES[TABLE4_NAME]["relpath"]
        if not source.exists():
            raise FileNotFoundError(
                f"raw mirror file {source} is absent and the source is not scriptable; "
                f"deposit it with deposit_raw_mirror(). Recipe: {MANUAL_RECIPE}"
            )
        digest = sha256_file(source)
        if digest != TABLE4_SHA256:
            raise RuntimeError(
                f"raw mirror {source} sha256 mismatch: got {digest}, expected "
                f"{TABLE4_SHA256}"
            )
        os.symlink(source, dest)
        log.info("Linked %s from the raw mirror (sha256 verified)", TABLE4_NAME)

    # ---- record builders ------------------------------------------------------ #
    @staticmethod
    def _environment() -> Environment:
        """Cooper's SC recipe as a typed SC dropout, ~16 h, temperature a typed gap."""
        return Environment(
            media=COOPER_SC,
            temperature=None,
            duration_hours=16.0,
            provenance_gaps=[_TEMPERATURE_GAP],
        )

    def create_experiment(  # type: ignore[override]
        self, row: KeptRow
    ) -> tuple[MetaboliteExperiment, MetaboliteExperimentReference, Publication]:
        """Build the Metabolite experiment/reference/publication for one kept row."""
        keys = [key for key in PEAK_ORDER if key in row.levels]
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=row.systematic_gene_name,
                    perturbed_gene_name=row.perturbed_gene_name,
                )
            ]
        )
        environment = self._environment()
        phenotype = MetabolitePhenotype(
            metabolite_level={key: row.levels[key] for key in keys},
            metabolite_level_se=None,
            n_replicates=dict.fromkeys(keys, 1),
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=None,  # a typed gap: see _phenotype_gaps
            provenance_gaps=_phenotype_gaps(keys),
        )
        # The reference is the plate average by construction: a ratio of 1.0 per key.
        phenotype_reference = MetabolitePhenotype(
            metabolite_level=dict.fromkeys(keys, 1.0),
            metabolite_level_se=None,
            n_replicates=dict.fromkeys(keys, 1),
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=None,
        )
        experiment = MetaboliteExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        reference = MetaboliteExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4741"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )
        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )
        return experiment, reference, publication

    # ---- build ---------------------------------------------------------------- #
    @post_process
    def process(self) -> None:
        """Parse Table 4, apply the retention rules, write the LMDB and the ledgers."""
        from dotenv import load_dotenv

        load_dotenv()
        data_root = os.environ["DATA_ROOT"]
        rows = read_table4(osp.join(self.raw_dir, TABLE4_NAME))
        kept, ledger = retain_rows(
            rows,
            self._resolver(),
            load_sgd_essential_genes(data_root),
            dataset_name=self.name,
        )
        log.info(
            "Cooper2010: %d source rows -> %d kept (%d values); dropped %d ORFs "
            "(%s); %d RENAMED kept under the current name; %d normalized identifiers; "
            "%d duplicate rows to the ledger; %d excluded non-deletion rows; "
            "%d essentiality-flagged rows kept",
            ledger.n_source_rows,
            ledger.n_kept,
            ledger.n_values_kept,
            ledger.n_dropped_orfs,
            [f"{d.source_name}:{d.status}" for d in ledger.dropped_orfs],
            ledger.n_renamed,
            ledger.n_normalized,
            ledger.n_duplicate_rows,
            ledger.n_excluded_non_deletion,
            ledger.n_essentiality_flagged,
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_data_csv(kept)

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        # LIFO exit: the interned txn commits before the records txn (crash safety).
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, row in enumerate(kept):
                experiment, reference, publication = self.create_experiment(row)
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )
        env.close()
        interned_env.close()
        log.info("Wrote %d Cooper2010 amino-acid experiments to LMDB", len(kept))

        ledger_path = osp.join(self.preprocess_dir, "dropped_records.json")
        with open(ledger_path, "w") as handle:
            handle.write(ledger.model_dump_json(indent=2))
        log.info("Cooper2010 ledger -> %s", ledger_path)
        self._write_sourced_values()

    def _write_data_csv(self, kept: Sequence[KeptRow]) -> None:
        """``preprocess/data.csv``: one kept row per line, missing peaks left empty."""
        out = osp.join(self.preprocess_dir, "data.csv")
        with open(out, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "systematic_gene_name",
                    "perturbed_gene_name",
                    "source_name",
                    "name_cell",
                    "row_index",
                    *PEAK_ORDER,
                ]
            )
            for row in kept:
                writer.writerow(
                    [
                        row.systematic_gene_name,
                        row.perturbed_gene_name,
                        row.source_name,
                        row.name_cell,
                        row.row_index,
                        *[row.levels.get(key, "") for key in PEAK_ORDER],
                    ]
                )

    def _write_sourced_values(self) -> None:
        """``preprocess/sourced_values.json``: every hardcoded number's quote."""
        out = osp.join(self.preprocess_dir, "sourced_values.json")
        payload = {
            key: value.model_dump(mode="json")
            for key, value in sorted(
                {**SOURCED_VALUES, **LEGEND_SOURCED_VALUES}.items()
            )
        }
        payload["peak_notes"] = dict(PEAK_NOTES)
        with open(out, "w") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        log.info("Cooper2010 sourced values -> %s", out)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    root = osp.join(os.environ["DATA_ROOT"], "data/torchcell/amino_acid_cooper2010")
    dataset = AminoAcidCooper2010Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
