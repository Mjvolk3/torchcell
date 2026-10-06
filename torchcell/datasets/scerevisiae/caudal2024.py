# torchcell/datasets/scerevisiae/caudal2024
# [[torchcell.datasets.scerevisiae.caudal2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/caudal2024
# Test file: tests/torchcell/datasets/scerevisiae/test_caudal2024.py
"""Caudal 2024 S. cerevisiae natural-isolate pan-transcriptome (RNA-seq, WS10).

Caudal et al. 2024 (Nat. Genet. 56:1278, doi:10.1038/s41588-024-01769-9) measured the
whole-transcriptome expression of ~1000 natural S. cerevisiae isolates (the 1002-genomes
panel), each mapped to its OWN genome. We model every isolate as a PERTURBATION SET off
the S288C reference: its genotype is the full set of sequence/copy-number differences from
S288C, and its phenotype is the absolute per-gene expression (TPM + raw counts).

GENOTYPE (three natural-variation perturbation families, all off-graph pointers):
  - ``SequenceVariantPerturbation`` -- a native S288C gene whose isolate allele differs
    from the S288C reference sequence. The reference allele is reconstructed EXACTLY in
    Peter's representation: the coordinate in a gene's first FASTA header
    (``chromosomeN:start-end +/-``) slices the SGD R64 chromosome (reverse-complemented on
    the minus strand); an isolate has a variant iff its sequence != that slice. Source =
    Peter et al. 2018 ``allReferenceGenesWithSNPsAndIndelsInferred`` (gene-keyed store).
  - ``NaturalGenePresencePerturbation`` for an ACCESSORY (non-reference) pangenome ORF that
    is PRESENT in the isolate (AXIS-1 state=present, natural insertion; ``copy_number`` from
    the copy-number matrix, default 1.0).
  - ``NaturalGeneAbsencePerturbation`` for a CORE (reference) ORF that is ABSENT in the
    isolate (AXIS-1 state=absent, natural deletion). EVERY absence is recorded -- an ORF
    with no S288C systematic name is kept by its pangenome id, never dropped (a dropped
    absence would wrongly reconstruct the gene as present).
These are NATURAL genome-content edits off S288C, distinct from engineered CNV (true dosage
of a PRESENT gene stays on the copy-number axis). Core vs accessory is Peter's
presence/absence matrix over the 1011 isolates: an ORF present in >= 99% of isolates is core.

PHENOTYPE (``RNASeqExpressionPhenotype``): per-isolate ``expression_tpm`` +
``expression_count`` for the genes that isolate carries (a gene absent from an isolate is
KEY-ABSENT, never 0). ``measurement_type = "rnaseq_tpm"``. Phenotype keys are Datafile 1's
``systematic_name``; a row with a BLANK ``systematic_name`` (459,790 rows in the built
isolates) is classified by its ``pan_absence`` into a ``BlankRowClass`` and served or
dropped by ``BLANK_ROW_RULES`` (issue #598): ``present`` rows are served (under the S288C
name when the row is an accessory feature merged with its S288C homolog, else under the
pangenome id ``X<n>-<name>``), ``absent`` rows are dropped as the paper does, and
``bad annotation`` / ``unannotated`` rows are dropped under a typed ``ProvenanceGap``
because no mirrored source defines those classes. Every row is counted in
``preprocess/blank_systematic_name_ledger.json``. The shared
``phenotype_reference`` is the POPULATION MEAN over the 943 built isolates (mean TPM /
rounded mean count per gene) -- an absolute WT-equivalent baseline, NOT a centered 0
(reference is not the record itself; ``reference_centered = False`` for verification).

STRAIN SET: the 943 isolates that are the intersection of Caudal's ``Strain`` codes (969)
with Peter's genome panel. The 26 Caudal-only strains (25 ``XTRA_*`` + ``FY4-6``) have no
Peter genome and are EXCLUDED -- only isolates with a matched genome are built. An isolate
code is whatever both sources spell it, including the ``SACE_`` form (78 of the 943, e.g.
``SACE_YAU``); see ``_isolate_and_symbol``.

Sources (hash-pinned, local library mirror):
  - Caudal expression: ``caudalPantranscriptomeRevealsLarge2024/data/
    final_data_annotated_merged_04052022.tab.zip`` (comma-delimited, latin-1;
    sha256 8b55ccd76e1d19476d8f5f718e9e061cb9e4693e343965114dd4cd65d5f8d26b).
  - Peter genome: the genomes tier set ``peter2018_1011_assemblies`` (resolved and
    sha256-verified through ``torchcell.sequence.genome.registry``) --
    ``allReferenceGenesWithSNPsAndIndelsInferred.tar.gz`` (sha256 b5400b89...),
    ``genesMatrix_PresenceAbsence.tab.gz``, ``genesMatrix_CopyNumber.tab.gz``.
  - S288C reference: SGD R64-4-1 ``S288C_reference_sequence_R64-4-1_20230830.fsa``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import os.path as osp
import pickle
import re
import tarfile
from enum import StrEnum
from typing import Any, Literal

import lmdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    post_process,
    verify_raw_files,
    verify_sha256,
)
from torchcell.datamodels.media import SC, restated
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    Media,
    NaturalGeneAbsencePerturbation,
    NaturalGenePresencePerturbation,
    Publication,
    ReferenceGenome,
    RNASeqExpressionExperiment,
    RNASeqExpressionExperimentReference,
    RNASeqExpressionPhenotype,
    SequenceVariantPerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.registry import (
    PETER2018_1011,
    SGD_S288C_R64,
    load_genome_manifest,
    resolve,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MEASUREMENT_TYPE = "rnaseq_tpm"
SEQUENCE_SOURCE = "peterGenomeEvolution10112018"
CORE_PRESENCE_THRESHOLD = 0.99  # ORF present in >= 99% of the 1011 isolates -> core

# Library-mirror-relative paths (under $DATA_ROOT). The stored artifact + sha256 is
# canonical; these are referenced (symlinked) in-place, never re-downloaded/copied.
CAUDAL_ZIP_REL = (
    "torchcell-library/caudalPantranscriptomeRevealsLarge2024/data/"
    "final_data_annotated_merged_04052022.tab.zip"
)
CAUDAL_ZIP_SHA256 = "8b55ccd76e1d19476d8f5f718e9e061cb9e4693e343965114dd4cd65d5f8d26b"
REFGENE_TAR_NAME = "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz"
REFGENE_TAR_SHA256 = "b5400b89499fe84b1feada51abd7742c29838ae1f28c0cbd208b6622ca533f25"
PRESENCE_NAME = "genesMatrix_PresenceAbsence.tab.gz"
COPYNUMBER_NAME = "genesMatrix_CopyNumber.tab.gz"
SGD_FSA_NAME = "S288C_reference_sequence_R64-4-1_20230830.fsa"

CAUDAL_ZIP_BASENAME = "final_data_annotated_merged_04052022.tab.zip"
RAW_FILES = [CAUDAL_ZIP_BASENAME, REFGENE_TAR_NAME, PRESENCE_NAME, COPYNUMBER_NAME]

_ROMAN = [
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
]
_S288C_RE = re.compile(r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$")
_EXCLUDED_STRAIN_RE = re.compile(r"^XTRA_")
# A Datafile 1 ``ORF`` value naming a non-reference pangenome ORF (R ``make.names`` form).
_PANGENOME_ORF_RE = re.compile(r"^X\d+\.")
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


class MissingTabMemberError(ValueError):
    """The Caudal expression zip does not hold exactly one ``.tab`` member."""


class UnextractableMemberError(RuntimeError):
    """A regular file in the reference-gene tarball could not be opened for reading."""


def _tab_member(zip_path: str, names: list[str]) -> str:
    """The one ``.tab`` member of the Caudal zip; any other count is refused.

    The released archive (sha256 ``8b55ccd7...``) holds exactly one member,
    ``final_data_annotated_merged_04052022.tab`` (issue #541).
    """
    tabs = [n for n in names if n.endswith(".tab")]
    if len(tabs) != 1:
        raise MissingTabMemberError(
            f"{zip_path} must hold exactly one '.tab' member, found {tabs}; "
            f"members: {names}"
        )
    return tabs[0]


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA string (A/C/G/T/N, case-preserving)."""
    return seq.translate(_COMPLEMENT)[::-1]


def _sgd_chromosomes(fsa_path: str) -> dict[str, str]:
    """Map each SGD chromosome id (roman I..XVI, ``MT``) to its uppercase sequence."""
    chrom: dict[str, str] = {}
    key: str | None = None
    buf: list[str] = []
    with open(fsa_path) as handle:
        for line in handle:
            if line.startswith(">"):
                if key is not None:
                    chrom[key] = "".join(buf).upper()
                match = re.search(r"\[chromosome=([IVX]+)\]", line)
                if match:
                    key = match.group(1)
                elif "[location=mitochondrion]" in line:
                    key = "MT"
                else:
                    key = None
                buf = []
            else:
                buf.append(line.strip())
    if key is not None:
        chrom[key] = "".join(buf).upper()
    return chrom


def _reference_slice(header: str, chrom: dict[str, str]) -> str:
    """Reconstruct the S288C reference allele from a Peter gene-file header coordinate."""
    match = re.search(r"chromosome(\d+):(\d+)-(\d+)\s+([+-])", header)
    if match is None:
        raise ValueError(f"unparseable coordinate header: {header!r}")
    chrom_n = int(match.group(1))
    start = int(match.group(2))
    end = int(match.group(3))
    strand = match.group(4)
    key = "MT" if chrom_n == 17 else _ROMAN[chrom_n - 1]
    seq = chrom[key][start - 1 : end]
    if strand == "-":
        seq = _reverse_complement(seq)
    return seq.upper()


def _isolate_and_symbol(token: str, sys_name: str) -> tuple[str, str]:
    """Split a Peter gene-FASTA header token into ``(isolate_code, gene_symbol)``.

    A token is ``<isolate>_<systematic_name>_<symbol>`` and the isolate code appears
    VERBATIM, in either of two forms: ``AEE_YAL001C_TFC3`` and
    ``SACE_YAU_YAL001C_TFC3``. ``SACE_YAU`` is that isolate's own code, NOT a species
    prefix on ``YAU``: 93 of the 1011 codes indexing
    ``genesMatrix_PresenceAbsence.tab.gz`` carry it (``SACE_GAL`` ... ``SACE_YDO``), and
    so do 78 of the 943 Caudal ``Strain`` codes we build. Stripping the prefix (the fix
    proposed in issue 73) leaves those 78 isolates unmatchable and silently discards
    469,170 of the 5,672,145 gene records, so the prefix is KEPT. Verified by an
    exhaustive scan of all 6015 gene files / 6,081,165 headers.

    Raises on a token that does not carry ``_<sys_name>_`` or that names an empty
    isolate; no header is ever skipped silently (zero such headers exist in the pinned
    tarball).
    """
    split_key = f"_{sys_name}_"
    if split_key not in token:
        raise ValueError(f"header token {token!r} carries no {split_key!r} gene key")
    iso, symbol = token.split(split_key, 1)
    if not iso:
        raise ValueError(f"header token {token!r} names an empty isolate")
    return iso, symbol


def _assert_all_isolates_seen(matched: set[str], seen: set[str]) -> None:
    """Fail loudly when a built isolate never appeared in a Peter gene-FASTA header.

    An isolate absent from every header would be assembled with zero sequence variants
    and so reconstruct as a perfect S288C match, which is the maximally misleading
    failure: the strain and its phenotype are present, its genotype is silently empty.
    """
    missing = sorted(matched - seen)
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(matched)} matched isolates never appeared in a "
            f"Peter gene-FASTA header, so their genotype would be silently empty: "
            f"{missing}"
        )


def _demangle_orf(col: str) -> str:
    """Reverse R ``make.names`` on a pangenome-ORF matrix column -> the raw ORF id.

    ``X1834.YAL063C`` -> ``1834-YAL063C``; ``X1.EC1118_1F14_0012g`` ->
    ``1-EC1118_1F14_0012g`` (leading ``X`` dropped; the first ``.`` was the ``-`` after the
    ORF number).
    """
    stem = col[1:] if col.startswith("X") else col
    return stem.replace(".", "-", 1)


def _orf_to_s288c(orf_id: str) -> str | None:
    """Return the S288C systematic name a pangenome ORF id encodes, else None.

    A pangenome ORF id is ``<number>-<name>``; ``<name>`` is an S288C systematic name for
    reference ORFs (``1834-YAL063C`` -> ``YAL063C``) and an assembly id otherwise
    (``1-EC1118_1F14_0012g`` -> None). Residual ``.`` (from mangled ``-B`` suffixes) is
    restored before the pattern test.

    A ``_NumOfGenes_N`` suffix marks a pangenome ORF that collapses N paralogous copies
    of a gene into one cluster (``1771-YAL005C_NumOfGenes_3`` -> ``YAL005C``). These ARE
    reference genes -- 793 of them -- so we strip the suffix and map the cluster to its
    S288C name. This conflates paralogs for the presence/absence question (the cluster is
    present iff any copy is), which is the right granularity here; every reference name
    maps from exactly one pangenome column, so no de-duplication is needed.
    """
    suffix = re.sub(r"^\d+-", "", orf_id).replace(".", "-")
    suffix = re.sub(r"_NumOfGenes_\d+$", "", suffix)
    return suffix if _S288C_RE.match(suffix) else None


# --------------------------------------------------------------------------- #
# Rows with a blank ``systematic_name`` (issue #598)
# --------------------------------------------------------------------------- #
# Datafile 1 carries one row per (Strain, ORF): 6,517 ORF rows per strain, and in the
# 943 built isolates 459,790 of them have a blank ``systematic_name``. Each such row has
# a ``pan_absence`` class. Before #598 a pandas groupby dropped all of them silently;
# now every one is classified, counted and either served or dropped by a named rule.
#
# Columns the loader consumes from Datafile 1.
CAUDAL_COLUMNS = [
    "Strain",
    "systematic_name",
    "ORF",
    "Ortholog_in_SGD_2010",
    "pan_absence",
    "count",
    "tpm",
]

CITATION_KEY = "caudalPantranscriptomeRevealsLarge2024"
# Online Methods, derived from the Europe PMC full-text XML of PMC11176082 (mirror
# ``si/PMC11176082_fulltext.xml``, sha256 d8b8db20...) by
# ``experiments/036-dataset-fixes-before-kg-build/scripts/caudal2024_retrieve_methods_si.py``.
METHODS_MD = "methods.md"
METHODS_MD_SHA256 = "10ccc3d0267ca337e4c3941ef3893ce39dfe393ebadbab83b6e9ff07a7e4f1e0"
PAPER_MD_SHA256 = "652b3497bc7799fef9972db233a7046f2815742e17280270f5071c623f0ac8aa"
SI_PDF = "si/41588_2024_1769_MOESM1_ESM.pdf"
SI_PDF_SHA256 = "826a0c7c34a3ab7b4a30eff1b89ef4fcf7d081a4e00885b97f552e49ee66f389"
# Supplementary Tables 1-9; sheet "Table S2" ("Description of genes included in this
# study") is the paper's 6,445-ORF gene table.
SI_TABLES_XLSX = "si/41588_2024_1769_MOESM3_ESM.xlsx"
SI_TABLES_XLSX_SHA256 = (
    "753e17d6ef9540206d3960c7f0c778fadfe43c9bee887ce4434885b81444bd7a"
)


def _methods_sv(value: object, quote: str, line: int, note: str) -> SourcedValue:
    """A SourcedValue pinned to the mirrored online Methods (``methods.md``)."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=METHODS_MD,
            citation_key=CITATION_KEY,
            sha256=METHODS_MD_SHA256,
            page=f"methods.md line {line}",
        ),
        quote=quote,
        note=note,
    )


#: methods.md line 9: the medium the RNA-seq cultures grew in, and its carbon source.
#: The Methods name "standard synthetic complete medium" and give no recipe beyond the
#: glucose, so the ingredient amounts are the library ``SC`` object's (issue #622).
SC_CULTURE_QUOTE = (
    "then grown in 1 ml of liquid standard synthetic complete medium using deep well "
    "blocks until the mid-log phase was reached (OD ~0.3)"
)
SC_CARBON_QUOTE = (
    "We measured growth in all strains using 96-well liquid growth in standard "
    "synthetic complete medium with 2% glucose as the carbon source."
)

MEDIUM_SOURCED_VALUES: dict[str, SourcedValue] = {
    "culture_medium": _methods_sv(
        "liquid synthetic complete (SC), harvested at mid-log",
        SC_CULTURE_QUOTE,
        9,
        note="the paper names the medium and prints no recipe; every ingredient "
        "amount on the record is the library SC object's (Mormino 2022 / Wildenhain "
        "2015 quotes), a deferral, not a Caudal statement",
    ),
    "carbon_source": _methods_sv(
        "2% glucose",
        SC_CARBON_QUOTE,
        9,
        note="the same amount as the library SC glucose row (20 g/L)",
    ),
}

CAUDAL_SC: Media = restated(SC, *MEDIUM_SOURCED_VALUES.values())
"""The RNA-seq culture medium: the library ``SC`` composition plus Caudal's own two
Methods sentences. Same ``media_identity`` as ``SC`` (provenance is not identity)."""


class BlankRowClass(StrEnum):
    """The ledger class of a Datafile 1 row whose ``systematic_name`` is blank."""

    present_s288c_homolog = "present_s288c_homolog"
    present_pangenome_orf = "present_pangenome_orf"
    absent = "absent"
    bad_annotation = "bad_annotation"
    unannotated = "unannotated"


class BlankRowRule(BaseModel):
    """What the loader does with one blank-name class, and the source that says why.

    Exactly one of ``definition`` (the sourced basis) and ``gap`` (a typed absence of
    one, ``gap.field == "definition"``) is set.
    """

    model_config = ConfigDict(extra="forbid")

    row_class: BlankRowClass
    pan_absence: str
    action: Literal["served", "dropped"]
    id_rule: str | None
    definition: SourcedValue | None
    gap: ProvenanceGap | None

    @model_validator(mode="after")
    def _one_basis(self) -> BlankRowRule:
        if (self.definition is None) == (self.gap is None):
            raise ValueError(
                f"{self.row_class}: exactly one of definition / gap must be set"
            )
        if self.gap is not None and self.gap.field != "definition":
            raise ValueError(f"{self.row_class}: gap must name the 'definition' field")
        if (self.action == "served") != (self.id_rule is not None):
            raise ValueError(f"{self.row_class}: a served class needs an id_rule")
        return self


_PRESENT_QUOTE = (
    "Abundance corresponds to the mean expression levels of all isolates where the "
    "gene is annotated as being present."
)
_MERGE_QUOTE = (
    "The read counts for 39 accessory features with a known homolog in S. cerevisiae "
    "according to the pangenome annotations were merged with the corresponding homolog."
)
_UNDEFINED_CLASS_GAP_NOTE = (
    "The pan_absence value {value!r} is not defined anywhere in the mirrored sources: "
    "searched methods.md (online Methods + Data/Code availability), paper.md (main "
    "text), the Supplementary Information PDF ({si_pdf}, sha256 {si_pdf_sha}) and every "
    "sheet of Supplementary Tables 1-9 ({si_xlsx}, sha256 {si_xlsx_sha}) for "
    "'bad annotation', 'unannotated' and 'pan_absence' with zero hits; the authors' "
    "code (github.com/HaploTeam/1011yeastsRNAseq: tpm_calc.R and GWAS scripts) does not "
    "assign the column either (read 2026-10-02, not mirrored). Rows stay dropped and "
    "counted until a source defines the class."
)


def _undefined_class_gap(value: str) -> ProvenanceGap:
    """Typed gap for a ``pan_absence`` class no mirrored source defines."""
    return ProvenanceGap(
        field="definition",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=Provenance(
            source_uri=METHODS_MD, citation_key=CITATION_KEY, sha256=METHODS_MD_SHA256
        ),
        note=_UNDEFINED_CLASS_GAP_NOTE.format(
            value=value,
            si_pdf=SI_PDF,
            si_pdf_sha=SI_PDF_SHA256,
            si_xlsx=SI_TABLES_XLSX,
            si_xlsx_sha=SI_TABLES_XLSX_SHA256,
        ),
    )


BLANK_ROW_RULES: dict[BlankRowClass, BlankRowRule] = {
    BlankRowClass.present_s288c_homolog: BlankRowRule(
        row_class=BlankRowClass.present_s288c_homolog,
        pan_absence="present",
        action="served",
        id_rule=(
            "serve under the S288C systematic name in ORF when ORF matches the S288C "
            "pattern, equals the row's Ortholog_in_SGD_2010, is a systematic_name "
            "served by named rows, and the isolate has no named row of that name"
        ),
        definition=_methods_sv(
            "present_s288c_homolog",
            _MERGE_QUOTE,
            35,
            "Datafile 1 relabels an accessory feature merged with its S. cerevisiae "
            "homolog by putting the S288C name in ORF and Ortholog_in_SGD_2010 while "
            "leaving systematic_name blank. Supplementary Table 2 (si/"
            "41588_2024_1769_MOESM3_ESM.xlsx sheet 'Table S2', sha256 753e17d6..., "
            "spreadsheet rows 7085-7100) assigns each of the 16 Annotation_Name values "
            "these rows carry to that same S288C systematic_name (for example "
            "1060-augustus_masked-ASN_8-20595 -> YBR020W). In these isolates Peter's "
            "presence matrix marks the reference ORF absent and the accessory ORF "
            "present, so the record holds both a NaturalGeneAbsencePerturbation for "
            "the S288C name and the expression keyed by it, as the paper's merge does.",
        ),
        gap=None,
    ),
    BlankRowClass.present_pangenome_orf: BlankRowRule(
        row_class=BlankRowClass.present_pangenome_orf,
        pan_absence="present",
        action="served",
        id_rule=(
            "serve under 'X' + _demangle_orf(ORF) (ORF 'X37.augustus_masked.2."
            "CGIPLA_MA' -> 'X37-augustus_masked.2.CGIPLA_MA'), the form Datafile 1 "
            "gives every named accessory ORF as its systematic_name; refused unless "
            "ORF is a pangenome id 'X<number>.<name>' and the id is no served "
            "systematic_name"
        ),
        definition=_methods_sv(
            "present_pangenome_orf",
            _PRESENT_QUOTE,
            43,
            "pan_absence is the per-isolate pangenome presence annotation ('All "
            "annotations can be found in datafile 1.', same line), and the paper "
            "computes abundance over the isolates annotated present. These ORFs (12 "
            "plasmid ORFs in the released table) never carry a systematic_name and are "
            "not in Supplementary Table 2, so they are outside the paper's 6,445-ORF "
            "analysis set; they are quantified in Datafile 1 (per-strain TPM reaches "
            "~1e6 only with the blank rows included) and Peter's presence matrix marks "
            "each present in every isolate that carries such a row, so they are served "
            "under the pangenome id. Measured by experiments/036-dataset-fixes-before-"
            "kg-build/scripts/caudal2024_blank_systematic_name.py.",
        ),
        gap=None,
    ),
    BlankRowClass.absent: BlankRowRule(
        row_class=BlankRowClass.absent,
        pan_absence="absent",
        action="dropped",
        id_rule=None,
        definition=_methods_sv(
            "absent",
            _PRESENT_QUOTE,
            43,
            "The paper's abundance excludes isolates where the gene is not annotated "
            "present; the Fig. 2 caption agrees (paper.md line 55, sha256 652b3497...: "
            "'For accessory genes, isolates that did not carry the given gene were "
            "excluded from the calculations.'). A gene absent from an isolate is "
            "key-absent in its phenotype.",
        ),
        gap=None,
    ),
    BlankRowClass.bad_annotation: BlankRowRule(
        row_class=BlankRowClass.bad_annotation,
        pan_absence="bad annotation",
        action="dropped",
        id_rule=None,
        definition=None,
        gap=_undefined_class_gap("bad annotation"),
    ),
    BlankRowClass.unannotated: BlankRowRule(
        row_class=BlankRowClass.unannotated,
        pan_absence="unannotated",
        action="dropped",
        id_rule=None,
        definition=None,
        gap=_undefined_class_gap("unannotated"),
    ),
}


class UnclassifiedBlankRowError(ValueError):
    """A blank-``systematic_name`` row whose ``pan_absence`` no ledger class covers."""


class GeneIdCollisionError(ValueError):
    """A served blank-name row would key a gene already served, or no rule fits its id."""


class BlankNameLedger(BaseModel):
    """Accounting for every Datafile 1 row of the built isolates (issue #598).

    ``counts`` holds the rows of each blank-name class, ``served_ids`` the rows each
    served blank-name id received. Every blank row is in exactly one class.
    """

    model_config = ConfigDict(extra="forbid")

    rules: list[BlankRowRule]
    n_rows: int
    n_rows_named: int
    n_rows_blank: int
    counts: dict[BlankRowClass, int]
    tpm_by_class: dict[BlankRowClass, float]
    served_ids: dict[str, int]
    named_pan_absence_counts: dict[str, int]

    @model_validator(mode="after")
    def _accounted(self) -> BlankNameLedger:
        if self.n_rows_named + self.n_rows_blank != self.n_rows:
            raise ValueError("named + blank rows must equal all rows")
        if sum(self.counts.values()) != self.n_rows_blank:
            raise ValueError("every blank row must be in exactly one class")
        served = sum(
            self.counts[c] for c, r in BLANK_ROW_RULES.items() if r.action == "served"
        )
        if sum(self.served_ids.values()) != served:
            raise ValueError("served_ids must account for every served row")
        return self


def read_caudal_table(zip_path: str) -> pd.DataFrame:
    """Read ``CAUDAL_COLUMNS`` of Datafile 1 from its zip (comma-delimited, latin-1).

    ``Strain`` is cast to ``str``; a missing column raises pandas' ``Usecols do not
    match columns`` error.
    """
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        name = _tab_member(zip_path, zf.namelist())
        with zf.open(name) as handle:
            df = pd.read_csv(
                handle, encoding="latin-1", low_memory=False, usecols=CAUDAL_COLUMNS
            )
    df["Strain"] = df["Strain"].astype(str)
    return df


def restrict_to_built_isolates(
    df: pd.DataFrame, peter_isolates: set[str]
) -> pd.DataFrame:
    """Keep the rows of built isolates: not ``XTRA_*``, not ``FY4-6``, in Peter's panel."""
    df = df[~df["Strain"].str.match(_EXCLUDED_STRAIN_RE)]
    df = df[df["Strain"] != "FY4-6"]
    return df[df["Strain"].isin(peter_isolates)]


def resolve_gene_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, BlankNameLedger]:
    """Give every kept Datafile 1 row a ``gene_id`` and account for every blank row.

    ``df`` holds ``CAUDAL_COLUMNS`` for the built isolates only. A named row keeps its
    ``systematic_name``. A blank-name row is classified by ``pan_absence`` into a
    ``BlankRowClass``; the two ``present`` classes are served under the id their
    ``BLANK_ROW_RULES`` entry names and the others are dropped. A blank row with any
    other ``pan_absence`` raises ``UnclassifiedBlankRowError``; a served id that would
    collide with a named gene of the same isolate raises ``GeneIdCollisionError``.
    Returns the kept rows (with ``gene_id``) and the ledger.
    """
    blank_mask = df["systematic_name"].isna()
    named = df[~blank_mask].assign(gene_id=lambda f: f["systematic_name"].astype(str))
    blank = df[blank_mask]
    served_names = set(named["gene_id"])

    by_value = {r.pan_absence: c for c, r in BLANK_ROW_RULES.items()}
    unknown = sorted(set(blank["pan_absence"].astype(str)) - set(by_value), key=str)
    if unknown:
        raise UnclassifiedBlankRowError(
            f"{int((~blank['pan_absence'].astype(str).isin(set(by_value))).sum())} "
            f"blank-systematic_name rows carry a pan_absence no ledger class covers: "
            f"{unknown}"
        )

    orf = blank["ORF"].astype(str)
    homolog = (
        (blank["pan_absence"] == "present")
        & orf.str.match(_S288C_RE)
        & (orf == blank["Ortholog_in_SGD_2010"].astype(str))
        & orf.isin(served_names)
    )
    pangenome = (blank["pan_absence"] == "present") & ~homolog
    row_class = pd.Series(
        blank["pan_absence"].map(by_value), index=blank.index, dtype=object
    )
    row_class[homolog] = BlankRowClass.present_s288c_homolog
    row_class[pangenome] = BlankRowClass.present_pangenome_orf

    homolog_rows = blank[homolog].assign(gene_id=orf[homolog])
    not_pangenome = sorted(set(orf[pangenome & ~orf.str.match(_PANGENOME_ORF_RE)]))
    if not_pangenome:
        raise GeneIdCollisionError(
            f"pangenome-id rule refuses ORFs that are no pangenome id: {not_pangenome}"
        )
    pangenome_ids = "X" + orf[pangenome].map(_demangle_orf)
    named_ids = sorted(set(pangenome_ids) & served_names)
    if named_ids:
        raise GeneIdCollisionError(
            f"pangenome-id rule would serve ids named rows already serve: {named_ids}"
        )
    pangenome_rows = blank[pangenome].assign(gene_id=pangenome_ids)
    served = pd.concat([homolog_rows, pangenome_rows])
    named_keys = set(zip(named["Strain"], named["gene_id"]))
    clashes = sorted(
        {k for k in zip(served["Strain"], served["gene_id"]) if k in named_keys}
    )
    duplicated = served.duplicated(["Strain", "gene_id"])
    if clashes or bool(duplicated.any()):
        raise GeneIdCollisionError(
            f"served blank-name rows collide with named rows {clashes[:10]} "
            f"(of {len(clashes)}) or repeat a (Strain, gene_id) "
            f"{int(duplicated.sum())} times"
        )

    ledger = BlankNameLedger(
        rules=list(BLANK_ROW_RULES.values()),
        n_rows=len(df),
        n_rows_named=len(named),
        n_rows_blank=len(blank),
        counts={c: int((row_class == c).sum()) for c in BlankRowClass},
        tpm_by_class={
            c: float(blank.loc[row_class == c, "tpm"].sum()) for c in BlankRowClass
        },
        served_ids={
            str(g): int(n) for g, n in served["gene_id"].value_counts().items()
        },
        named_pan_absence_counts={
            str(k): int(v)
            for k, v in named["pan_absence"].value_counts(dropna=False).items()
        },
    )
    return pd.concat([named, served]), ledger


@register_dataset
class CaudalPanTranscriptome2024Dataset(ExperimentDataset):
    """Natural-isolate pan-transcriptome: each isolate a perturbation set off S288C."""

    def __init__(
        self,
        root: str = "data/torchcell/caudal_pantranscriptome2024",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (isolates keyed by their 3-letter Caudal strain code)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return RNASeqExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return RNASeqExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Caudal + Peter raw files (symlinked from the library mirror)."""
        return RAW_FILES

    def _data_root(self) -> str:
        """Return $DATA_ROOT (the library-mirror parent)."""
        from dotenv import load_dotenv

        load_dotenv()
        return os.environ["DATA_ROOT"]

    def download(self) -> None:
        """Symlink the hash-pinned mirror files into ``raw_dir`` and verify sha256.

        Large artifacts are referenced in place (never copied). The two files with a
        module pin (Caudal zip, Peter reference-gene tarball) are hash-verified here; the
        presence/copy-number matrices are verified by the genomes tier on ``resolve``.
        ``process`` re-verifies all four in ``raw/`` before reading them.
        """
        data_root = self._data_root()
        os.makedirs(self.raw_dir, exist_ok=True)
        sources = {
            CAUDAL_ZIP_BASENAME: osp.join(data_root, CAUDAL_ZIP_REL),
            # Peter files come from the genomes tier, sha256-verified on resolve.
            REFGENE_TAR_NAME: resolve(PETER2018_1011, REFGENE_TAR_NAME),
            PRESENCE_NAME: resolve(PETER2018_1011, PRESENCE_NAME),
            COPYNUMBER_NAME: resolve(PETER2018_1011, COPYNUMBER_NAME),
        }
        sha256_expected = {
            CAUDAL_ZIP_BASENAME: CAUDAL_ZIP_SHA256,
            REFGENE_TAR_NAME: REFGENE_TAR_SHA256,
        }
        for name, src in sources.items():
            if not osp.exists(src):
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            expected = sha256_expected.get(name)
            if expected is not None:
                verify_sha256(src, expected)
            dest = osp.join(self.raw_dir, name)
            if not osp.exists(dest):
                os.symlink(src, dest)
        log.info(
            "Caudal/Peter raw files linked into %s (sha256 verified)", self.raw_dir
        )

    @post_process
    def process(self) -> None:
        """Build the 943 per-isolate pan-transcriptome experiments and write LMDB.

        Every file in ``raw/`` is verified before a row is read: the Caudal zip and the
        reference-gene tarball against this module's pins, the presence and copy-number
        matrices against the genomes tier's manifest (the pin ``resolve`` checks in
        ``download``). A mismatch raises ``RawSha256MismatchError``.
        """
        data_root = self._data_root()
        peter = load_genome_manifest(PETER2018_1011, data_root)
        verify_raw_files(
            self.raw_dir,
            {
                CAUDAL_ZIP_BASENAME: CAUDAL_ZIP_SHA256,
                REFGENE_TAR_NAME: REFGENE_TAR_SHA256,
                PRESENCE_NAME: peter.record(PRESENCE_NAME).sha256,
                COPYNUMBER_NAME: peter.record(COPYNUMBER_NAME).sha256,
            },
        )
        os.makedirs(self.preprocess_dir, exist_ok=True)

        # 1. Peter presence/absence + copy-number matrices -> core/accessory + ORF maps.
        presence = pd.read_csv(
            osp.join(self.raw_dir, PRESENCE_NAME), sep="\t", index_col=0
        )
        copynumber = pd.read_csv(
            osp.join(self.raw_dir, COPYNUMBER_NAME), sep="\t", index_col=0
        ).reindex(columns=presence.columns)
        cols = list(presence.columns)
        orf_ids = [_demangle_orf(c) for c in cols]
        s288c_names = [_orf_to_s288c(o) for o in orf_ids]
        s288c_mask = np.array([n is not None for n in s288c_names], dtype=bool)
        presence_vals = presence.to_numpy(dtype=float)
        copynumber_vals = copynumber.to_numpy(dtype=float)
        core_mask = presence_vals.mean(axis=0) >= CORE_PRESENCE_THRESHOLD
        peter_isolates = set(presence.index.astype(str))
        log.info(
            "Peter pangenome: %d ORFs (%d core, %d accessory), %d isolates",
            len(cols),
            int(core_mask.sum()),
            int((~core_mask).sum()),
            len(peter_isolates),
        )

        # 2. Caudal expression -> per-(strain, gene) tpm/count, restricted to matched
        #    strains; population-mean reference over those strains.
        strains, per_strain, ref_tpm, ref_count = self._load_caudal(
            data_root, peter_isolates
        )
        log.info("Matched isolates (Caudal ∩ Peter): %d", len(strains))

        # 3. Sequence variants vs the S288C reference slice (heavy; resumable parquet).
        variants_by_strain = self._sequence_variants(data_root, set(strains))

        # 4. Shared population-mean phenotype reference (absolute TPM baseline).
        self._reference_phenotype = RNASeqExpressionPhenotype(
            expression_tpm=ref_tpm,
            expression_count=ref_count,
            measurement_type=MEASUREMENT_TYPE,
            n_mapped_reads=None,
        )

        # Row index per strain into the numpy presence/copy-number matrices.
        strain_row = {str(s): i for i, s in enumerate(presence.index.astype(str))}
        pd.DataFrame({"strain_id": strains}).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(2e11))
        n_seq_total = 0
        n_cnv_acc_total = 0
        n_cnv_core_total = 0
        idx = 0
        with env.begin(write=True) as txn:
            for strain in tqdm(strains, desc="assembling isolates"):
                row = strain_row[strain]
                presence_perts, absence_perts = self._content_perturbations(
                    strain,
                    presence_vals[row],
                    copynumber_vals[row],
                    s288c_mask,
                    orf_ids,
                    s288c_names,
                )
                seq_perts = self._sequence_perturbations(
                    strain, variants_by_strain.get(strain, [])
                )
                n_seq_total += len(seq_perts)
                n_cnv_acc_total += len(presence_perts)
                n_cnv_core_total += len(absence_perts)
                experiment, reference, publication = self.create_experiment(
                    {
                        "strain": strain,
                        "perturbations": seq_perts + presence_perts + absence_perts,
                        "expression_tpm": per_strain[strain]["tpm"],
                        "expression_count": per_strain[strain]["count"],
                    }
                )
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
        n = max(idx, 1)
        log.info(
            "Wrote %d isolates. Per-isolate means: sequence_variants=%.1f, "
            "accessory-present=%.1f, core-absent=%.2f",
            idx,
            n_seq_total / n,
            n_cnv_acc_total / n,
            n_cnv_core_total / n,
        )

    def _load_caudal(
        self, data_root: str, peter_isolates: set[str]
    ) -> tuple[
        list[str],
        dict[str, dict[str, dict[str, Any]]],
        dict[str, float],
        dict[str, int],
    ]:
        """Aggregate Caudal expression per (strain, gene); return matched strains + ref.

        Every row of the built isolates is accounted for by ``resolve_gene_ids``: named
        rows keep their ``systematic_name``, blank-name rows are served or dropped by
        their ledger class, and the ledger is written to
        ``preprocess/blank_systematic_name_ledger.json`` (issue #598).
        """
        df = read_caudal_table(osp.join(self.raw_dir, CAUDAL_ZIP_BASENAME))
        df = restrict_to_built_isolates(df, peter_isolates)
        df, ledger = resolve_gene_ids(df)
        ledger_path = osp.join(self.preprocess_dir, "blank_systematic_name_ledger.json")
        with open(ledger_path, "w") as handle:
            handle.write(ledger.model_dump_json(indent=2))
        log.info(
            "Caudal rows in built isolates: %d (%d named, %d blank systematic_name); "
            "blank rows by class %s; served %d blank rows under %d ids -> %s",
            ledger.n_rows,
            ledger.n_rows_named,
            ledger.n_rows_blank,
            {str(c): n for c, n in ledger.counts.items()},
            sum(ledger.served_ids.values()),
            len(ledger.served_ids),
            ledger_path,
        )
        # Aggregate multi-allele rows per (strain, gene): tpm sum, count sum (defensive;
        # the merged file already carries one row per pair). ``gene_id`` is never null.
        agg = (
            df.groupby(["Strain", "gene_id"], sort=False, dropna=False)
            .agg(tpm=("tpm", "sum"), count=("count", "sum"))
            .reset_index()
        )
        per_strain: dict[str, dict[str, dict[str, Any]]] = {}
        for strain, sub in agg.groupby("Strain", sort=False):
            genes = sub["gene_id"].tolist()
            tpm = {g: float(t) for g, t in zip(genes, sub["tpm"].tolist())}
            count = {
                g: int(round(float(c))) for g, c in zip(genes, sub["count"].tolist())
            }
            per_strain[str(strain)] = {"tpm": tpm, "count": count}
        # Population-mean reference over the matched isolates (mean per gene).
        ref = agg.groupby("gene_id", sort=False).agg(
            tpm=("tpm", "mean"), count=("count", "mean")
        )
        ref_tpm = {str(g): float(t) for g, t in ref["tpm"].items()}
        ref_count = {str(g): int(round(float(c))) for g, c in ref["count"].items()}
        strains = sorted(per_strain)
        return strains, per_strain, ref_tpm, ref_count

    def _sequence_variants(
        self, data_root: str, matched: set[str]
    ) -> dict[str, list[tuple[str, str, str]]]:
        """Diff every isolate's allele vs the S288C reference slice, per gene (resumable).

        Returns ``strain_id -> [(systematic_gene_name, symbol, header_token), ...]``. The
        result is cached to ``preprocess/sequence_variants.parquet`` so re-assembly need not
        re-diff the ~6015-gene x 1011-isolate store.

        Headers naming a Peter isolate outside ``matched`` (68 isolates Caudal did not
        sequence) are the only skipped records, and ``_assert_all_isolates_seen`` proves
        no isolate we DO build was skipped along with them.
        """
        cache = osp.join(self.preprocess_dir, "sequence_variants.parquet")
        if osp.exists(cache):
            log.info("Loading cached sequence variants from %s", cache)
            vdf = pd.read_parquet(cache)
            out: dict[str, list[tuple[str, str, str]]] = {}
            for strain, sub in vdf.groupby("strain_id", sort=False):
                out[str(strain)] = list(
                    zip(sub["systematic_gene_name"], sub["symbol"], sub["header_token"])
                )
            return out

        chrom = _sgd_chromosomes(resolve(SGD_S288C_R64, SGD_FSA_NAME))
        tar_path = osp.join(self.raw_dir, REFGENE_TAR_NAME)
        strain_col: list[str] = []
        sys_col: list[str] = []
        sym_col: list[str] = []
        tok_col: list[str] = []
        seen_isolates: set[str] = set()
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tqdm(tf, desc="diffing reference genes"):
                if not member.isfile():
                    continue
                sys_name = member.name.replace(".fasta", "")
                extracted = tf.extractfile(member)
                if extracted is None:
                    # Unreachable for a released archive: all 6,015 members of the
                    # pinned tarball are regular files and extract (issue #541).
                    raise UnextractableMemberError(
                        f"{tar_path}: member {member.name!r} is a regular file but "
                        "tarfile returned no file object for it; refusing to drop "
                        "its variants"
                    )
                records = _parse_fasta(extracted.read().decode("latin-1"))
                if not records:
                    continue
                ref = _reference_slice(records[0][0], chrom)
                for header, seq in records:
                    token = header.split()[0].split("\t")[0]
                    iso, symbol = _isolate_and_symbol(token, sys_name)
                    if iso not in matched:
                        continue
                    seen_isolates.add(iso)
                    if seq.upper() != ref:
                        strain_col.append(iso)
                        sys_col.append(sys_name)
                        sym_col.append(symbol)
                        tok_col.append(token)
        # Every built isolate must have been read from a header; a missing one would be
        # assembled with an empty genotype rather than failing.
        _assert_all_isolates_seen(matched, seen_isolates)
        vdf = pd.DataFrame(
            {
                "strain_id": strain_col,
                "systematic_gene_name": sys_col,
                "symbol": sym_col,
                "header_token": tok_col,
            }
        )
        vdf.to_parquet(cache, index=False)
        log.info(
            "Diffed %d sequence variants across %d isolates -> %s",
            len(vdf),
            vdf["strain_id"].nunique(),
            cache,
        )
        out = {}
        for strain, sub in vdf.groupby("strain_id", sort=False):
            out[str(strain)] = list(
                zip(sub["systematic_gene_name"], sub["symbol"], sub["header_token"])
            )
        return out

    @staticmethod
    def _sequence_perturbations(
        strain: str, variants: list[tuple[str, str, str]]
    ) -> list[SequenceVariantPerturbation]:
        """Build the SequenceVariantPerturbations for one isolate."""
        perts: list[SequenceVariantPerturbation] = []
        for sys_name, symbol, token in variants:
            perts.append(
                SequenceVariantPerturbation(
                    systematic_gene_name=sys_name,
                    perturbed_gene_name=symbol or sys_name,
                    strain_id=strain,
                    sequence_source=SEQUENCE_SOURCE,
                    sequence_uri=f"{sys_name}.fasta#{token}",
                    sequence_sha256=REFGENE_TAR_SHA256,
                )
            )
        return perts

    @staticmethod
    def _content_perturbations(
        strain: str,
        presence_row: np.ndarray[Any, Any],
        copynumber_row: np.ndarray[Any, Any],
        s288c_mask: np.ndarray[Any, Any],
        orf_ids: list[str],
        s288c_names: list[str | None],
    ) -> tuple[
        list[NaturalGenePresencePerturbation], list[NaturalGeneAbsencePerturbation]
    ]:
        """AXIS-1 presence/absence edits for one isolate, RELATIVE TO S288C.

        Presence/absence vs S288C is a question of **reference membership** (is this ORF
        in S288C?), NOT population frequency (is it in >=99% of isolates?). The two sets
        differ -- a reference ORF can be variable, an accessory ORF can be near-ubiquitous
        -- so both loops gate on ``s288c_mask`` (does the pangenome column map to an S288C
        systematic name, paralog clusters included), never on a frequency ``core_mask``:

        * non-reference ORF PRESENT -> a natural PRESENCE (S288C lacks it, the isolate has
          it), with observed copy number;
        * reference ORF ABSENT -> a natural ABSENCE (S288C has it, the isolate lost it).

        A reference ORF present and an accessory ORF absent are BOTH no-ops vs S288C. The
        earlier core/accessory gating silently dropped every *variable* reference ORF that
        was absent (~133 per isolate), so those isolates reconstructed as if they still
        carried the gene -- the exact failure the "never dropped" invariant forbids.
        copy_number != 1 on a present gene is a separate dosage axis, not modelled here.
        """
        presence: list[NaturalGenePresencePerturbation] = []
        for j in np.nonzero((~s288c_mask) & (presence_row == 1))[0]:
            cn = copynumber_row[j]
            copy_number = float(cn) if np.isfinite(cn) and cn > 0 else 1.0
            orf_id = orf_ids[j]
            presence.append(
                NaturalGenePresencePerturbation(
                    systematic_gene_name=orf_id,
                    perturbed_gene_name=orf_id,
                    copy_number=copy_number,
                    strain_id=strain,
                    pangenome_orf_id=orf_id,
                    origin=None,
                    sequence_source=SEQUENCE_SOURCE,
                )
            )
        absence: list[NaturalGeneAbsencePerturbation] = []
        for j in np.nonzero(s288c_mask & (presence_row == 0))[0]:
            # s288c_mask guarantees a systematic name here; keep the pangenome-id fallback
            # defensively so an absence is never dropped.
            name = s288c_names[j] or orf_ids[j]
            absence.append(
                NaturalGeneAbsencePerturbation(
                    systematic_gene_name=name,
                    perturbed_gene_name=name,
                    strain_id=strain,
                    pangenome_orf_id=orf_ids[j],
                    sequence_source=SEQUENCE_SOURCE,
                )
            )
        return presence, absence

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[
        RNASeqExpressionExperiment, RNASeqExpressionExperimentReference, Publication
    ]:
        """Build the RNA-seq experiment/reference/publication for one isolate."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )
        genotype = Genotype(perturbations=row["perturbations"])
        # SC liquid medium, 30 C, harvested at mid-log (OD ~0.3) -- Caudal Methods.
        environment = Environment(media=CAUDAL_SC, temperature=Temperature(value=30))
        phenotype = RNASeqExpressionPhenotype(
            expression_tpm=row["expression_tpm"],
            expression_count=row["expression_count"],
            measurement_type=MEASUREMENT_TYPE,
            n_mapped_reads=None,
        )
        experiment = RNASeqExpressionExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        reference = RNASeqExpressionExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=self._reference_phenotype,
        )
        publication = Publication(
            pubmed_id="38778243",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/38862621/",
            doi="10.1038/s41588-024-01769-9",
            doi_url="https://doi.org/10.1038/s41588-024-01769-9",
        )
        return experiment, reference, publication


def _parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into ``[(header_without_gt, sequence), ...]``."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
            header = line[1:]
            seq = []
        else:
            seq.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def _sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Build/load the dataset for interactive debugging.

    Loads the existing LMDB if already built. To rebuild, delete ``<root>/processed``
    (and ``<root>/preprocess/sequence_variants.parquet`` to re-diff) first.
    """
    from dotenv import load_dotenv

    load_dotenv()
    root = osp.join(
        os.environ["DATA_ROOT"], "data/torchcell/caudal_pantranscriptome2024"
    )
    dataset = CaudalPanTranscriptome2024Dataset(root=root)
    print(f"len = {len(dataset)}")
    record = dataset[0]
    exp = record["experiment"]
    ptypes: dict[str, int] = {}
    for pert in exp["genotype"]["perturbations"]:
        ptypes[pert["perturbation_type"]] = ptypes.get(pert["perturbation_type"], 0) + 1
    print("record[0] perturbation-type counts:", ptypes)
    print("record[0] phenotype gene count:", len(exp["phenotype"]["expression_tpm"]))
    print("record[0] genome_reference:", record["reference"]["genome_reference"])


if __name__ == "__main__":
    main()
