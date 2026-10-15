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

GENOTYPE. Each isolate is a perturbation set off S288C built by
``torchcell.sequence.natural_isolate`` (``NaturalIsolateGenotypes``), the one module every
natural-isolate loader on the Peter 2018 panel shares, so an isolate is the same
``Genotype`` here as in the Peter 2018, Muenzner 2024 and De Chiara 2022 stores:
``SequenceVariantPerturbation`` per reference gene whose inferred allele differs from the
S288C R64 slice, ``NaturalGenePresencePerturbation`` per non-reference pangenome ORF the
isolate carries (with its copy number), ``NaturalGeneAbsencePerturbation`` per reference
ORF it lacks, ``DeletionPerturbation`` of HO where Peter's Supplementary Table 1 marks an
HO deletion (#988), ``NaturalLossOfFunctionPerturbation`` per call of Peter's two
loss-of-function matrices (#987), and ``Genotype.natural_isolate`` (ploidy, zygosity,
aneuploidy and manipulation from Table S1). Presence vs absence is reference MEMBERSHIP
(does the pangenome column map to an S288C name), not population frequency.

PHENOTYPE (``RNASeqExpressionPhenotype``): per-isolate ``expression_tpm`` +
``expression_count`` for the genes that isolate carries (a gene absent from an isolate is
KEY-ABSENT, never 0). ``measurement_type = "rnaseq_tpm"``. Phenotype keys are Datafile 1's
``systematic_name``; a row with a BLANK ``systematic_name`` (459,790 rows in the built
isolates) is classified by its ``pan_absence`` into a ``BlankRowClass`` and served or
dropped by ``BLANK_ROW_RULES`` (issue #598): ``present`` rows are served (under the S288C
name when the row is an accessory feature merged with its S288C homolog, else under the
pangenome id ``X<n>-<name>``), ``absent`` rows are dropped as the paper does, and
``bad annotation`` / ``unannotated`` rows are served under their pangenome id with a
typed ``ProvenanceGap`` on the class definition, because no mirrored source defines
those classes and a released value is stored as released (owner rule 2026-10-12; the
16,031 ``bad annotation`` rows of the built isolates carry 17 pangenome ids). Every row
is counted in ``preprocess/blank_systematic_name_ledger.json``. The shared
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
  - Peter genome: the six files of the genomes tier set ``peter2018_1011_assemblies``
    that ``natural_isolate.GENOTYPE_FILES`` names (the reference-gene tarball, the
    presence and copy-number matrices, the two loss-of-function matrices and the
    Supplementary Tables workbook), each sha256-verified against the tier manifest.
  - S288C reference: SGD R64-4-1 chromosomes from the SGD genomes tier (the slices
    the inferred alleles are diffed against).
"""

from __future__ import annotations

import logging
import os
import os.path as osp
import pickle
import re
from enum import StrEnum
from typing import Any, Literal

import lmdb
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    post_process,
    verify_raw_files,
    verify_sha256,
)
from torchcell.datamodels.media import SC, restated
from torchcell.datamodels.schema import (
    DeletionPerturbation,
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    Media,
    NaturalGeneAbsencePerturbation,
    NaturalGenePresencePerturbation,
    NaturalIsolateDescriptor,
    NaturalLossOfFunctionPerturbation,
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
    SGD_S288C_R64_GENBANK_ACCESSION,
    load_genome_manifest,
)
from torchcell.sequence.natural_isolate import (
    GENOTYPE_FILES,
    S288C_RE,
    NaturalIsolateGenotypes,
    demangle_orf,
    link_genotype_files,
    panel_isolates,
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

# Library-mirror-relative path (under $DATA_ROOT). The stored artifact + sha256 is
# canonical; it is referenced (symlinked) in place, never re-downloaded/copied.
CAUDAL_ZIP_REL = (
    "torchcell-library/caudalPantranscriptomeRevealsLarge2024/data/"
    "final_data_annotated_merged_04052022.tab.zip"
)
CAUDAL_ZIP_SHA256 = "8b55ccd76e1d19476d8f5f718e9e061cb9e4693e343965114dd4cd65d5f8d26b"
CAUDAL_ZIP_BASENAME = "final_data_annotated_merged_04052022.tab.zip"
#: The Caudal release plus the six Peter-tier files an isolate genotype is built from.
RAW_FILES = [CAUDAL_ZIP_BASENAME, *GENOTYPE_FILES]
#: What the isolate genotypes of this build carry, counted (``natural_isolate``).
GENOTYPE_LEDGER = "natural_isolate_genotype_ledger.json"
#: The classes an isolate genotype holds, built in ``torchcell.sequence.natural_isolate``.
#: Named here because the schema-impact closure of this loader is read from this file's
#: own imports: a change to any of them must re-stale this store.
GENOTYPE_CLASSES = (
    SequenceVariantPerturbation,
    NaturalGenePresencePerturbation,
    NaturalGeneAbsencePerturbation,
    DeletionPerturbation,
    NaturalLossOfFunctionPerturbation,
    NaturalIsolateDescriptor,
)

_EXCLUDED_STRAIN_RE = re.compile(r"^XTRA_")
# A Datafile 1 ``ORF`` value naming a non-reference pangenome ORF (R ``make.names`` form).
_PANGENOME_ORF_RE = re.compile(r"^X\d+\.")


class MissingTabMemberError(ValueError):
    """The Caudal expression zip does not hold exactly one ``.tab`` member."""


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
    "assign the column either (read 2026-10-02, not mirrored). The rows are served "
    "under their pangenome id as released, with this gap on the class definition, "
    "until a source defines the class."
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


_PANGENOME_ID_RULE = (
    "serve under 'X' + demangle_orf(ORF) (ORF 'X37.augustus_masked.2.CGIPLA_MA' -> "
    "'X37-augustus_masked.2.CGIPLA_MA'), the form Datafile 1 gives every named accessory "
    "ORF as its systematic_name; refused unless ORF is a pangenome id 'X<number>.<name>' "
    "and the id is no served systematic_name"
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
        id_rule=_PANGENOME_ID_RULE,
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
        action="served",
        id_rule=_PANGENOME_ID_RULE,
        definition=None,
        gap=_undefined_class_gap("bad annotation"),
    ),
    BlankRowClass.unannotated: BlankRowRule(
        row_class=BlankRowClass.unannotated,
        pan_absence="unannotated",
        action="served",
        id_rule=_PANGENOME_ID_RULE,
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
    ``BlankRowClass``; every class whose rule says ``served`` is served under the id
    the rule names (the S288C homolog name, else the pangenome id) and ``absent`` is
    dropped as the paper does. A blank row with any
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
        & orf.str.match(S288C_RE)
        & (orf == blank["Ortholog_in_SGD_2010"].astype(str))
        & orf.isin(served_names)
    )
    served_by_pangenome_id = {
        r.pan_absence
        for r in BLANK_ROW_RULES.values()
        if r.action == "served" and r.id_rule == _PANGENOME_ID_RULE
    }
    pangenome = blank["pan_absence"].isin(served_by_pangenome_id) & ~homolog
    row_class = pd.Series(
        blank["pan_absence"].map(by_value), index=blank.index, dtype=object
    )
    row_class[homolog] = BlankRowClass.present_s288c_homolog
    row_class[pangenome & (blank["pan_absence"] == "present")] = (
        BlankRowClass.present_pangenome_orf
    )

    homolog_rows = blank[homolog].assign(gene_id=orf[homolog])
    not_pangenome = sorted(set(orf[pangenome & ~orf.str.match(_PANGENOME_ORF_RE)]))
    if not_pangenome:
        raise GeneIdCollisionError(
            f"pangenome-id rule refuses ORFs that are no pangenome id: {not_pangenome}"
        )
    pangenome_ids = "X" + orf[pangenome].map(demangle_orf)
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

    organism = Organism.scerevisiae

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
        """Symlink the Caudal zip and the six Peter-tier genotype files into ``raw_dir``.

        Large artifacts are referenced in place (never copied). The Caudal zip is
        hash-verified against its module pin; each Peter file against the genomes tier's
        manifest (``natural_isolate.link_genotype_files``). ``process`` re-verifies all
        seven in ``raw/`` before reading them.
        """
        data_root = self._data_root()
        os.makedirs(self.raw_dir, exist_ok=True)
        source = osp.join(data_root, CAUDAL_ZIP_REL)
        if not osp.exists(source):
            raise RuntimeError(f"required raw artifact missing from mirror: {source}")
        verify_sha256(source, CAUDAL_ZIP_SHA256)
        dest = osp.join(self.raw_dir, CAUDAL_ZIP_BASENAME)
        if not osp.exists(dest):
            os.symlink(source, dest)
        link_genotype_files(self.raw_dir, data_root)
        log.info(
            "Caudal/Peter raw files linked into %s (sha256 verified)", self.raw_dir
        )

    @post_process
    def process(self) -> None:
        """Build the 943 per-isolate pan-transcriptome experiments and write LMDB.

        Every file in ``raw/`` is verified before a row is read: the Caudal zip against
        this module's pin, the six Peter files against the genomes tier's manifest. A
        mismatch raises ``RawSha256MismatchError``. Each isolate's genotype comes from
        ``NaturalIsolateGenotypes``; what the genotypes carry is counted in
        ``preprocess/natural_isolate_genotype_ledger.json``.
        """
        data_root = self._data_root()
        peter = load_genome_manifest(PETER2018_1011, data_root)
        verify_raw_files(
            self.raw_dir,
            {
                CAUDAL_ZIP_BASENAME: CAUDAL_ZIP_SHA256,
                **{name: peter.record(name).sha256 for name in GENOTYPE_FILES},
            },
        )
        os.makedirs(self.preprocess_dir, exist_ok=True)

        # 1. Caudal expression -> per-(strain, gene) tpm/count, restricted to the
        #    isolates with a Peter genome; population-mean reference over those strains.
        panel = panel_isolates(self.raw_dir)
        strains, per_strain, ref_tpm, ref_count = self._load_caudal(data_root, panel)
        log.info("Matched isolates (Caudal ∩ Peter): %d", len(strains))

        # 2. The isolate genotypes (allele diff cached in preprocess/).
        genotypes = NaturalIsolateGenotypes(
            self.raw_dir, self.preprocess_dir, strains, data_root
        )
        genotypes.write_ledger(osp.join(self.preprocess_dir, GENOTYPE_LEDGER))

        # 3. Shared population-mean phenotype reference (absolute TPM baseline).
        self._reference_phenotype = RNASeqExpressionPhenotype(
            expression_tpm=ref_tpm,
            expression_count=ref_count,
            measurement_type=MEASUREMENT_TYPE,
            n_mapped_reads=None,
        )
        pd.DataFrame({"strain_id": strains}).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(2e11))
        kinds: dict[str, int] = {}
        idx = 0
        with env.begin(write=True) as txn:
            for strain in tqdm(strains, desc="assembling isolates"):
                genotype = genotypes.genotype(strain)
                for kind in genotype.perturbation_types:
                    kinds[kind] = kinds.get(kind, 0) + 1
                experiment, reference, publication = self.create_experiment(
                    {
                        "genotype": genotype,
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
            "Wrote %d isolates. Per-isolate means: %s",
            idx,
            {kind: round(count / n, 2) for kind, count in sorted(kinds.items())},
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
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )
        genotype: Genotype = row["genotype"]
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


def verify_build(dataset_root: str, data_root: str | None = None) -> Any:
    """L0-L4 report for a built store (``build_dataset_lmdb --verify`` calls this).

    The RNA-seq family rules plus the four isolate-genotype rules every natural-isolate
    store shares (``runners.verify_rnaseq_store``); the CLI writes the report to
    ``<dataset_root>/preprocess/verification_report.json``.
    """
    from dotenv import load_dotenv

    from torchcell.verification import runners

    load_dotenv()
    return runners.verify_rnaseq_store(
        "caudal_pantranscriptome2024",
        data_root or os.environ["DATA_ROOT"],
        abs_root=dataset_root,
    )


def main() -> None:
    """Build/load the dataset for interactive debugging.

    Loads the existing LMDB if already built. To rebuild, retire ``<root>/processed``
    and ``<root>/preprocess`` first (``build_dataset_lmdb --retire-existing``).
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
