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

ONE RECORD PER CULTURE (2026-11-24, tracking issue #1178). The paper profiled 29 isolates
in independent culture replicates (``REPLICATE_SOURCED_VALUES``), released as Datafile 2
(``replicate_data_tpm_22042023.tab``: one row per (``SampleID``, ``ORF``), TPM only, and
the sample's ``Final.Group``). ``Final_set`` is the culture Datafile 1 carries (one per
isolate; its ``Number.of.clean.mapped.reads``, ``Library`` and ``Sequencing.run`` equal the
isolate's Supplementary Table 1 row), and ``good_duplicate`` / ``good_duplicateBis`` are
the other cultures. So every isolate keeps its Datafile 1 record unchanged, and each
other culture of a built isolate is one more record: its Datafile 2 TPM verbatim, keyed by
the Datafile 1 gene id of the same ORF in the same isolate (``r_make_names`` recovers
Datafile 1's spelling), a blank-name ``absent`` ORF left out as for the Datafile 1 record,
``expression_count = None`` with ``REPLICATE_COUNT_GAP`` (Datafile 2 releases no counts),
``n_mapped_reads`` the sample's released clean mapped reads, and ``replicate_id`` the
released ``SampleID``. A replicated isolate's Datafile 1 record carries its ``Final_set``
``SampleID`` as ``replicate_id``; an isolate profiled once keeps ``replicate_id = None``.
26 of the 29 replicated isolates are built, so the store is 943 + 28 = 971 records. Every
Datafile 2 sample is accounted for in ``preprocess/replicate_culture_ledger.json``.
The reference stays the Datafile 1 population mean over the 943 isolates.

Sources (hash-pinned, local library mirror):
  - Caudal expression: ``caudalPantranscriptomeRevealsLarge2024/data/
    final_data_annotated_merged_04052022.tab.zip`` (comma-delimited, latin-1;
    sha256 8b55ccd76e1d19476d8f5f718e9e061cb9e4693e343965114dd4cd65d5f8d26b).
  - Caudal replicate cultures: ``caudalPantranscriptomeRevealsLarge2024/data/
    replicate_data_tpm_22042023.tab`` (comma-delimited; sha256
    6860fa5f7d42e7ae1d733f3db60b84d4cb45fb0d9757f7fb5c500a4b640cf900).
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
#: Datafile 2, the per-sample TPM of the 29 isolates profiled in culture replicates
#: (methods.md line 95). Library mirror, sha256 from its ``manifest.json``.
REPLICATE_TAB_REL = (
    "torchcell-library/caudalPantranscriptomeRevealsLarge2024/data/"
    "replicate_data_tpm_22042023.tab"
)
REPLICATE_TAB_SHA256 = (
    "6860fa5f7d42e7ae1d733f3db60b84d4cb45fb0d9757f7fb5c500a4b640cf900"
)
REPLICATE_TAB_BASENAME = "replicate_data_tpm_22042023.tab"
#: The two Caudal releases plus the six Peter-tier files an isolate genotype is built from.
RAW_FILES = [CAUDAL_ZIP_BASENAME, REPLICATE_TAB_BASENAME, *GENOTYPE_FILES]
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


# --------------------------------------------------------------------------- #
# Datafile 2: the replicate cultures (issue #1178)
# --------------------------------------------------------------------------- #
#: Columns the loader consumes from Datafile 2.
REPLICATE_COLUMNS = [
    "ORF",
    "SampleID",
    "tpm",
    "Standardized.name",
    "Final.Group",
    "Number.of.clean.mapped.reads",
]


def _paper_sv(value: object, quote: str, line: int, note: str) -> SourcedValue:
    """A SourcedValue pinned to the mirrored main text (``paper.md``)."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri="paper.md",
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            page=f"paper.md line {line}",
        ),
        quote=quote,
        note=note,
    )


#: What a Datafile 2 sample is, from the paper's own sentences.
REPLICATE_SOURCED_VALUES: dict[str, SourcedValue] = {
    "replicate_kind": _paper_sv(
        "independent culture (biological replicate)",
        "We performed independent culture replicates for 29 samples (Supplementary "
        "Fig. 1b).",
        25,
        note="each Datafile 2 sample of an isolate is its own culture; Supplementary "
        "Fig. 1c compares 66 replicate pairs, the ordered pairs of the 60 samples "
        "(27 isolates x 2, 2 x 3), so the Final_set culture is one of the replicates",
    ),
    "replicates_sequenced": _methods_sv(
        "29 isolates sequenced in duplicate",
        "1,046 samples were sequenced, including duplicates for 29 of the isolates "
        "(datafile 2).",
        19,
        note="Datafile 2 holds 60 samples of 29 isolates: 27 with 2, 2 (AFD, AKL) with 3",
    ),
    "replicate_release": _methods_sv(
        "Datafile 2: TPM and sample information, no counts",
        "datafile 2: normalized gene expression levels (TPM) and sample information "
        "for 29 replicates (file name: replicate_data_tpm_22042023.tab)",
        95,
        note="the file's columns are ORF, SampleID, tpm and per-sample information; "
        "it carries no read counts",
    ),
}

REPLICATE_COUNT_GAP = ProvenanceGap(
    field="expression_count",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=REPLICATE_TAB_REL,
        citation_key=CITATION_KEY,
        sha256=REPLICATE_TAB_SHA256,
    ),
    note="Datafile 2 releases TPM and sample information only (methods.md line 95); "
    "the featureCounts read counts of the replicate cultures are not in it, and "
    "Datafile 1 carries the counts of the Final_set culture alone",
)


class ReplicateGroup(StrEnum):
    """Datafile 2's ``Final.Group``: which culture of an isolate a sample is."""

    final_set = "Final_set"
    good_duplicate = "good_duplicate"
    good_duplicate_bis = "good_duplicateBis"


class ReplicateTableError(ValueError):
    """Datafile 2 does not have the structure the loader reads it by."""


def r_make_names(orf: str) -> str:
    """R ``make.names`` of a Datafile 2 ``ORF``: the spelling Datafile 1 gives it.

    Every character outside ``[A-Za-z0-9._]`` becomes ``.`` and an ``X`` is prepended
    unless the name starts with a letter or with a dot not followed by a digit:
    ``1003-augustus_masked-YCM-7680`` -> ``X1003.augustus_masked.YCM.7680``; an S288C
    name is unchanged.
    """
    name = re.sub(r"[^A-Za-z0-9._]", ".", orf)
    if not re.match(r"^([A-Za-z]|\.(?![0-9]))", name):
        name = "X" + name
    return name


class ReplicateCulture(BaseModel):
    """One Datafile 2 sample of a built isolate that Datafile 1 does not carry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    strain: str
    group: ReplicateGroup
    n_mapped_reads: int
    tpm: dict[str, float]


class ReplicateLedger(BaseModel):
    """Accounting for every Datafile 2 sample and row.

    ``final_set_sample_ids`` names, per built replicated isolate, the sample that is its
    Datafile 1 culture; ``samples`` lists every sample with its isolate, group and fate.
    """

    model_config = ConfigDict(extra="forbid")

    n_rows: int
    n_samples: int
    n_strains: int
    samples_by_group: dict[ReplicateGroup, int]
    samples: list[dict[str, str | int]]
    strains_not_built: list[str]
    final_set_sample_ids: dict[str, str]
    n_culture_records: int
    n_rows_culture_records: int
    n_rows_absent_dropped: int


def read_replicate_table(path: str) -> pd.DataFrame:
    """Read ``REPLICATE_COLUMNS`` of Datafile 2 (comma-delimited, a header row).

    ``tpm`` is parsed round-trip exact (pandas' default parser misreads the last digit
    of some released values, e.g. ``0.00832811287968815`` of ``Run4.7.CCH``),
    ``SampleID`` and ``Standardized.name`` are text, the mapped-read count is read
    from its thousands-separated text (``1,542,282``) to an integer, and every
    ``Final.Group`` must be a ``ReplicateGroup``.
    """
    df = pd.read_csv(
        path,
        usecols=REPLICATE_COLUMNS,
        float_precision="round_trip",
        dtype={
            "SampleID": str,
            "Standardized.name": str,
            "Final.Group": str,
            "Number.of.clean.mapped.reads": str,
            "ORF": str,
        },
    )
    df["Number.of.clean.mapped.reads"] = (
        df["Number.of.clean.mapped.reads"].str.replace(",", "", regex=False).astype(int)
    )
    unknown = sorted(set(df["Final.Group"]) - {g.value for g in ReplicateGroup})
    if unknown:
        raise ReplicateTableError(f"Final.Group values no rule covers: {unknown}")
    return df


def replicate_cultures(
    replicates: pd.DataFrame, built: pd.DataFrame, kept: pd.DataFrame
) -> tuple[list[ReplicateCulture], ReplicateLedger]:
    """The non-``Final_set`` cultures of built isolates, with every sample accounted for.

    ``built`` is Datafile 1 restricted to the built isolates (``CAUDAL_COLUMNS``) and
    ``kept`` its rows ``resolve_gene_ids`` kept, with ``gene_id``. A sample belongs to
    one isolate, group and mapped-read count; a built isolate has exactly one
    ``Final_set`` sample (its Datafile 1 culture). Each row of another sample takes the
    ``gene_id`` of the Datafile 1 row of the same isolate and ``r_make_names(ORF)``; a
    row whose Datafile 1 row was dropped (a blank-name ``absent`` ORF) is dropped and
    counted, and a row with no Datafile 1 row, or a repeated (sample, gene), is refused.
    """
    per_sample = replicates.groupby("SampleID", sort=True).agg(
        strain=("Standardized.name", "unique"),
        group=("Final.Group", "unique"),
        reads=("Number.of.clean.mapped.reads", "unique"),
    )
    ambiguous = [
        str(s)
        for s, row in per_sample.iterrows()
        if len(row["strain"]) != 1 or len(row["group"]) != 1 or len(row["reads"]) != 1
    ]
    if ambiguous:
        raise ReplicateTableError(
            f"samples with more than one isolate, group or mapped-read count: "
            f"{ambiguous}"
        )
    if bool(replicates.duplicated(["SampleID", "ORF"]).any()):
        raise ReplicateTableError("Datafile 2 repeats a (SampleID, ORF) row")
    sample_strain = {str(s): str(r["strain"][0]) for s, r in per_sample.iterrows()}
    sample_group = {
        str(s): ReplicateGroup(r["group"][0]) for s, r in per_sample.iterrows()
    }
    sample_reads = {str(s): int(r["reads"][0]) for s, r in per_sample.iterrows()}
    built_strains = set(built["Strain"])

    final_set: dict[str, list[str]] = {}
    for sample, strain in sample_strain.items():
        if sample_group[sample] is ReplicateGroup.final_set:
            final_set.setdefault(strain, []).append(sample)
    replicated = sorted(set(sample_strain.values()))
    off = [
        s for s in replicated if s in built_strains and len(final_set.get(s, [])) != 1
    ]
    if off:
        raise ReplicateTableError(
            f"built isolates without exactly one Final_set sample: {off}"
        )

    rows = replicates[
        replicates["Standardized.name"].isin(built_strains)
        & (replicates["Final.Group"] != ReplicateGroup.final_set.value)
    ].assign(ORF=lambda f: f["ORF"].map(r_make_names))
    released = rows.merge(
        built[["Strain", "ORF"]],
        left_on=["Standardized.name", "ORF"],
        right_on=["Strain", "ORF"],
        how="left",
        indicator=True,
    )
    missing = released[released["_merge"] == "left_only"]
    if len(missing):
        raise ReplicateTableError(
            f"{len(missing)} Datafile 2 rows name an ORF the isolate has no Datafile 1 "
            f"row for: {sorted(set(zip(missing['SampleID'], missing['ORF'])))[:10]}"
        )
    keyed = rows.merge(
        kept[["Strain", "ORF", "gene_id"]],
        left_on=["Standardized.name", "ORF"],
        right_on=["Strain", "ORF"],
        how="inner",
    )
    if bool(keyed.duplicated(["SampleID", "gene_id"]).any()):
        raise ReplicateTableError("two Datafile 2 rows of one sample share a gene id")

    cultures = [
        ReplicateCulture(
            sample_id=str(sample),
            strain=sample_strain[str(sample)],
            group=sample_group[str(sample)],
            n_mapped_reads=sample_reads[str(sample)],
            tpm={str(g): float(t) for g, t in zip(sub["gene_id"], sub["tpm"])},
        )
        for sample, sub in keyed.groupby("SampleID", sort=True)
    ]
    cultures.sort(
        key=lambda c: (c.strain, list(ReplicateGroup).index(c.group), c.sample_id)
    )

    def fate(sample: str) -> str:
        if sample_strain[sample] not in built_strains:
            return "isolate_not_built"
        if sample_group[sample] is ReplicateGroup.final_set:
            return "datafile_1_record"
        return "culture_record"

    ledger = ReplicateLedger(
        n_rows=len(replicates),
        n_samples=len(sample_strain),
        n_strains=len(replicated),
        samples_by_group={
            g: sum(1 for v in sample_group.values() if v is g) for g in ReplicateGroup
        },
        samples=[
            {
                "sample_id": s,
                "strain": sample_strain[s],
                "group": sample_group[s].value,
                "n_mapped_reads": sample_reads[s],
                "fate": fate(s),
            }
            for s in sorted(sample_strain)
        ],
        strains_not_built=[s for s in replicated if s not in built_strains],
        final_set_sample_ids={
            s: final_set[s][0] for s in replicated if s in built_strains
        },
        n_culture_records=len(cultures),
        n_rows_culture_records=len(keyed),
        n_rows_absent_dropped=len(rows) - len(keyed),
    )
    return cultures, ledger


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
        """Symlink the two Caudal files and the six Peter-tier files into ``raw_dir``.

        Large artifacts are referenced in place (never copied). The Caudal zip and
        Datafile 2 are hash-verified against their module pins; each Peter file against
        the genomes tier's manifest (``natural_isolate.link_genotype_files``).
        ``process`` re-verifies all eight in ``raw/`` before reading them.
        """
        data_root = self._data_root()
        os.makedirs(self.raw_dir, exist_ok=True)
        for rel, sha256, basename in (
            (CAUDAL_ZIP_REL, CAUDAL_ZIP_SHA256, CAUDAL_ZIP_BASENAME),
            (REPLICATE_TAB_REL, REPLICATE_TAB_SHA256, REPLICATE_TAB_BASENAME),
        ):
            source = osp.join(data_root, rel)
            if not osp.exists(source):
                raise RuntimeError(
                    f"required raw artifact missing from mirror: {source}"
                )
            verify_sha256(source, sha256)
            dest = osp.join(self.raw_dir, basename)
            if not osp.exists(dest):
                os.symlink(source, dest)
        link_genotype_files(self.raw_dir, data_root)
        log.info(
            "Caudal/Peter raw files linked into %s (sha256 verified)", self.raw_dir
        )

    @post_process
    def process(self) -> None:
        """Build one experiment per culture (943 isolates, 971 cultures) and write LMDB.

        Every file in ``raw/`` is verified before a row is read: the two Caudal files
        against this module's pins, the six Peter files against the genomes tier's
        manifest. A mismatch raises ``RawSha256MismatchError``. Each isolate's genotype
        comes from ``NaturalIsolateGenotypes``; what the genotypes carry is counted in
        ``preprocess/natural_isolate_genotype_ledger.json``. An isolate's records are
        its Datafile 1 culture, then its other Datafile 2 cultures in ``Final.Group``
        and ``SampleID`` order.
        """
        data_root = self._data_root()
        peter = load_genome_manifest(PETER2018_1011, data_root)
        verify_raw_files(
            self.raw_dir,
            {
                CAUDAL_ZIP_BASENAME: CAUDAL_ZIP_SHA256,
                REPLICATE_TAB_BASENAME: REPLICATE_TAB_SHA256,
                **{name: peter.record(name).sha256 for name in GENOTYPE_FILES},
            },
        )
        os.makedirs(self.preprocess_dir, exist_ok=True)

        # 1. Caudal expression -> per-(strain, gene) tpm/count, restricted to the
        #    isolates with a Peter genome; population-mean reference over those strains;
        #    the other cultures of the replicated isolates from Datafile 2.
        panel = panel_isolates(self.raw_dir)
        strains, per_strain, ref_tpm, ref_count, cultures, final_set = (
            self._load_caudal(data_root, panel)
        )
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

        by_strain: dict[str, list[ReplicateCulture]] = {}
        for culture in cultures:
            by_strain.setdefault(culture.strain, []).append(culture)

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(2e11))
        kinds: dict[str, int] = {}
        idx = 0
        with env.begin(write=True) as txn:
            for strain in tqdm(strains, desc="assembling isolates"):
                genotype = genotypes.genotype(strain)
                for kind in genotype.perturbation_types:
                    kinds[kind] = kinds.get(kind, 0) + 1
                rows: list[dict[str, Any]] = [
                    {
                        "genotype": genotype,
                        "expression_tpm": per_strain[strain]["tpm"],
                        "expression_count": per_strain[strain]["count"],
                        "n_mapped_reads": None,
                        "replicate_id": final_set.get(strain),
                    }
                ]
                rows += [
                    {
                        "genotype": genotype,
                        "expression_tpm": culture.tpm,
                        "expression_count": None,
                        "n_mapped_reads": culture.n_mapped_reads,
                        "replicate_id": culture.sample_id,
                    }
                    for culture in by_strain.get(strain, [])
                ]
                for row in rows:
                    experiment, reference, publication = self.create_experiment(row)
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
        n = max(len(strains), 1)
        log.info(
            "Wrote %d records (%d isolates, %d replicate cultures). Per-isolate "
            "means: %s",
            idx,
            len(strains),
            len(cultures),
            {kind: round(count / n, 2) for kind, count in sorted(kinds.items())},
        )

    def _load_caudal(
        self, data_root: str, peter_isolates: set[str]
    ) -> tuple[
        list[str],
        dict[str, dict[str, dict[str, Any]]],
        dict[str, float],
        dict[str, int],
        list[ReplicateCulture],
        dict[str, str],
    ]:
        """Aggregate Caudal expression per (strain, gene); return matched strains + ref.

        Every row of the built isolates is accounted for by ``resolve_gene_ids``: named
        rows keep their ``systematic_name``, blank-name rows are served or dropped by
        their ledger class, and the ledger is written to
        ``preprocess/blank_systematic_name_ledger.json`` (issue #598). Datafile 2's
        other cultures of the built isolates and each replicated isolate's ``Final_set``
        sample id are returned too, accounted for in
        ``preprocess/replicate_culture_ledger.json``.
        """
        built = restrict_to_built_isolates(
            read_caudal_table(osp.join(self.raw_dir, CAUDAL_ZIP_BASENAME)),
            peter_isolates,
        )
        df, ledger = resolve_gene_ids(built)
        cultures, replicate_ledger = replicate_cultures(
            read_replicate_table(osp.join(self.raw_dir, REPLICATE_TAB_BASENAME)),
            built,
            df,
        )
        replicate_path = osp.join(self.preprocess_dir, "replicate_culture_ledger.json")
        with open(replicate_path, "w") as handle:
            handle.write(replicate_ledger.model_dump_json(indent=2))
        log.info(
            "Datafile 2: %d samples of %d isolates; %d culture records for built "
            "isolates (%d rows, %d absent rows dropped); not built %s -> %s",
            replicate_ledger.n_samples,
            replicate_ledger.n_strains,
            replicate_ledger.n_culture_records,
            replicate_ledger.n_rows_culture_records,
            replicate_ledger.n_rows_absent_dropped,
            replicate_ledger.strains_not_built,
            replicate_path,
        )
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
        return (
            strains,
            per_strain,
            ref_tpm,
            ref_count,
            cultures,
            replicate_ledger.final_set_sample_ids,
        )

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
        """Build the RNA-seq experiment/reference/publication for one culture.

        ``row["expression_count"]`` None (a Datafile 2 culture) carries
        ``REPLICATE_COUNT_GAP``.
        """
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
            n_mapped_reads=row["n_mapped_reads"],
            replicate_id=row["replicate_id"],
            provenance_gaps=(
                [REPLICATE_COUNT_GAP] if row["expression_count"] is None else []
            ),
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
