# torchcell/datasets/ecoli/li2014
# [[torchcell.datasets.ecoli.li2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/li2014
# Test file: tests/torchcell/datasets/ecoli/test_li2014.py
r"""Li 2014 absolute protein synthesis rates in E. coli MG1655, three MOPS media.

Li GW, Burkhardt D, Gross C, Weissman JS (2014) "Quantifying absolute protein synthesis
rates reveals principles underlying allocation of cellular resources", Cell
157:624-635, doi:10.1016/j.cell.2014.02.033 (PMC4006352). Ribosome profiling of
MG1655 growing in MOPS glucose medium with the full Neidhardt supplement, the same
supplement without L-methionine, and no supplement. The release is ONE statistic per
gene and medium, Table S1's absolute synthesis rate, "where ki has the unit of molecules
per generation". This module is the first consumer of
``ProteinSynthesisRatePhenotype`` (issue #857): one ``ProteinSynthesisRateExperiment``
per medium, wild type, keyed on MG1655 b-numbers.

WHAT IS STORED. Only Table S1's PLAIN integer cells. A cell written ``[n]`` is refused
with the typed reason ``below_read_count_gate``: the bracket is defined nowhere in the
paper or the SI, and its meaning is back-solved from a count. The main text says "we
evaluated 3,041 genes" in rich defined medium and "All of these genes have >128
ribosome footprint fragments sequenced", and exactly 3,041 MOPS-complete cells are
plain (:data:`EVALUATED_GENES`, asserted at build time). A bracketed value is therefore
a number the authors did not evaluate, and it is not a censored bound either: the
gate is on footprint COUNT, not on the rate, so 683 plain MOPS-complete values sit at
or below the largest bracketed value of that column.

A plain cell that cannot carry one locus is refused too, each with its own typed
reason (:class:`RefusalReason`): a merged-pair row (``tufA+tufB``, one value for two
loci), a name the pinned assembly retires, a name that resolves to several loci, and
the two pairs of rows whose names resolve to ONE locus (``ecpD``/``yagW``,
``mscM``/``yjeP``), where a gene-level record cannot hold two values. Every refused
cell is written to ``preprocess/refused_cells.csv`` with the cell text verbatim.

WHAT IS NOT STORED, AND WHY.

- No degradation rate, half-life or turnover: Li releases none, which is why the
  record is a synthesis-rate phenotype rather than a ``ProteinTurnoverPhenotype``.
- No abundance: "For stable proteins, ki is also the copy number" is an assumption
  about the protein, and the main text calls the rate "an upper bound for the protein
  levels for the small subset of proteins that are rapidly degraded".
- No per-gene uncertainty or replicate count: the paper states "an error of less than
  1.3-fold across biological replicates" as a bound over all genes, and GEO GSE53767
  holds pooled tracks (one rich-medium and two minimal-medium ribosome-profiling
  samples, none for the methionine dropout). ``n_replicates`` and ``synthesis_rate_se``
  are therefore None, each with a typed ``ProvenanceGap``.

The time basis is the generation, so every record carries its medium's doubling time
from the SI (21.5, 26.5 and 56.3 min) on ``generation_time_minutes``. The reference of
every record is the MOPS-complete record, the condition "The results presented in this
work are based on".

Finding and L0-L4 table: [[torchcell.datasets.ecoli.li2014]]. Release inventory:
``experiments/036-dataset-fixes-before-kg-build/scripts/li2014_release_inventory.py``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import os.path as osp
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal

import openpyxl
import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    file_sha256,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import MOPS_MINIMAL
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialAssemblySet,
    ComponentDefinition,
    Compound,
    Concentration,
    ConcentrationUnit,
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    ProteinSynthesisRateExperiment,
    ProteinSynthesisRateExperimentReference,
    ProteinSynthesisRatePhenotype,
    Publication,
    SynthesisRateUnit,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import Manifest
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
    audit_sourced_value,
)

log = logging.getLogger(__name__)

__all__ = [
    "CITATION_KEY",
    "MEDIA",
    "ProteinSynthesisRateLi2014Dataset",
    "RefusalReason",
    "load_manifest",
    "raw_mirror_dir",
    "verify_build",
]

# --------------------------------------------------------------------------- #
# Paper, mirror and raw-file pins
# --------------------------------------------------------------------------- #
CITATION_KEY = "liQuantifyingAbsoluteProtein2014"
DOI = "10.1016/j.cell.2014.02.033"
PUBMED_ID = "24766808"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
RAW_RETRIEVED_AT = "2026-10-10"

#: Table S1 (mmc1): the one workbook every record is built from.
TABLE_S1_FILENAME = "1-s2.0-S0092867414002323-mmc1.xlsx"
TABLE_S1_REL = f"data/{TABLE_S1_FILENAME}"
TABLE_S1_SHA256 = "f493023e9b4bf7a154fd12b4c1fd617047af46153fffde7f1215932d35d5a661"
TABLE_S1_SHEET = "TableS1"
#: The ``pdftotext -layout`` rendering of the published article + Extended
#: Experimental Procedures (mmc6), and PMC's text of the author manuscript.
SI_TEXT_REL = "si/1-s2.0-S0092867414002323-mmc6.pdftotext.txt"
SI_TEXT_SHA256 = "053b55b6327afc1e3f6ac90bf7af231e081620f9992ad3aee54f97d4df59e3db"
PAPER_TEXT_REL = "paper/PMC4006352.1.txt"
PAPER_TEXT_SHA256 = "b3cc01fb5111bef6ae40d758d5e89e69cd198c77bcf33419da7578d27c605a72"

MG1655_STRAIN: Literal["MG1655"] = "MG1655"
MG1655_ASSEMBLY_SET: BacterialAssemblySet = "ecoli_K12_MG1655_ASM584v2"

SOURCE_ROWS = 4095
#: Measured on the pinned workbook; asserted at build time.
EXPECTED_RECORDS = 3

MEASUREMENT_TYPE = "ribosome_profiling_footprint_density_absolute_synthesis_rate"


# --------------------------------------------------------------------------- #
# Sourced values: each quote is a verbatim substring of the pinned mirror bytes
# --------------------------------------------------------------------------- #
def _si(value: Any, quote: str, *, page: str, note: str | None = None) -> SourcedValue:
    """Bind a value to the pinned ``pdftotext`` rendering of the published SI."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI_TEXT_REL,
            citation_key=CITATION_KEY,
            sha256=SI_TEXT_SHA256,
            method="pdftotext -layout of the published article + Extended "
            "Experimental Procedures (raw mirror)",
            page=page,
        ),
    )


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to PMC's plain text of the author manuscript (raw mirror)."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_TEXT_REL,
            citation_key=CITATION_KEY,
            sha256=PAPER_TEXT_SHA256,
            method="PMC Article Datasets plain text of the NIH author manuscript",
            page=page,
        ),
    )


def _table_s1(value: Any, quote: str, *, cell: str) -> SourcedValue:
    """Bind a value to one verbatim header cell of Table S1."""
    return SourcedValue(
        value=value,
        quote=quote,
        provenance=Provenance(
            source_uri=TABLE_S1_REL,
            citation_key=CITATION_KEY,
            sha256=TABLE_S1_SHA256,
            method="published Table S1 workbook (raw mirror)",
            page=f"sheet {TABLE_S1_SHEET}, cell {cell}",
        ),
    )


_PAGE_GROWTH = "Extended Experimental Procedures, 'Strain and Growth Conditions'"
_PAGE_RATE = "Extended Experimental Procedures, absolute synthesis rate derivation"
_PAGE_RESULTS = "Results, 'Absolute protein synthesis rates'"

STRAIN = _si(
    MG1655_STRAIN,
    "E. coli K-12 strain MG1655 was used for this study.",
    page=_PAGE_GROWTH,
)
BASE_MEDIUM = _si(
    "MOPS glucose 0.2%",
    "All cultures were based on MOPS media with 0.2% glucose (Teknova), with either",
    page=_PAGE_GROWTH,
)
SUPPLEMENTS = _si(
    ("full supplement", "full supplement without L-methionine", "no supplement"),
    "full supplement (Neidhardt et al., 1974), full supplement without L-methionine, "
    "or no supplement.",
    page=_PAGE_GROWTH,
)
TEMPERATURE_C = _si(
    37.0,
    "The culture was kept in a 2.8-l flask at 37\x01 C with aeration (180 rpm)",
    page=_PAGE_GROWTH,
    note="pdftotext renders the degree sign as the control byte 0x01; kept verbatim. "
    "The aeration is why every record is aerobic; Environment has no culture-format "
    "field (that is CultureEnvironment's), so the flask and shaking are not stored",
)
DOUBLING_COMPLETE_AND_DROPOUT = _si(
    {"complete": 21.5, "complete_without_methionine": 26.5},
    "The doubling time at 37\x01 C is 21.5 ± 0.4 min in fully supplemented MOPS media, "
    "26.5 ± 1.1 min in the methionine",
    page=_PAGE_GROWTH,
)
DOUBLING_MINIMAL = _si(
    56.3, "dropout medium, and 56.3 ± 0.5 min in minimal medium.", page=_PAGE_GROWTH
)
REFERENCE_CONDITION = _si(
    "complete",
    "The results presented in this work are based on MOPS complete media",
    page=_PAGE_GROWTH,
)
UNIT = _si(
    SynthesisRateUnit.molecules_per_generation,
    "where ki has the unit of molecules per generation.",
    page=_PAGE_RATE,
)
STABLE_EQUALS_COPY_NUMBER = _si(
    "assumption_not_measurement",
    "proteins, ki is also the copy number. The results are listed in Table S1.",
    page=_PAGE_RATE,
    note="the conversion to a copy number holds only for a stable protein, so the rate "
    "is not stored as an abundance",
)
EVALUATED_GENES = _paper(
    3041,
    "For growth in a rich defined medium (Neidhardt et al., 1974), we evaluated 3,041 "
    "genes which account for >96% of total proteins synthesized.",
    page=_PAGE_RESULTS,
    note="the count that back-solves the bracket: exactly 3,041 MOPS-complete cells "
    "of Table S1 are plain integers",
)
READ_COUNT_GATE = _paper(
    128,
    "All of these genes have >128 ribosome footprint fragments sequenced, with an "
    "error of less than 1.3-fold across biological replicates.",
    page=_PAGE_RESULTS,
)
UPPER_BOUND_FOR_DEGRADED = _paper(
    "upper_bound_for_rapidly_degraded",
    "Our measures based on synthesis rates thus provide an upper bound for the protein "
    "levels for the small subset of proteins that are rapidly degraded.",
    page="Results",
)
GEO_DEPOSIT = _paper(
    "GSE53767",
    "Data are available at Gene Expression Omnibus with accession number GSE53767.",
    page="Accession numbers",
)


class MediumSpec(BaseModel):
    """One released medium: its Table S1 column, supplement and doubling time."""

    model_config = ConfigDict(frozen=True)

    key: Literal["complete", "minimal", "complete_without_methionine"]
    column: str
    column_cell: str
    supplement: str | None
    generation_time_minutes: float
    generation_time_source: SourcedValue
    #: Plain integer cells of the column, measured on the pinned workbook.
    plain_cells: int
    #: Records' keys after the typed refusals, measured on the pinned workbook.
    stored_keys: int

    @property
    def header(self) -> SourcedValue:
        """The verbatim Table S1 header naming this medium."""
        return _table_s1(self.key, self.column, cell=self.column_cell)


MEDIA: tuple[MediumSpec, ...] = (
    MediumSpec(
        key="complete",
        column="MOPS complete",
        column_cell="B1",
        supplement="full supplement",
        generation_time_minutes=21.5,
        generation_time_source=DOUBLING_COMPLETE_AND_DROPOUT,
        plain_cells=3041,
        stored_keys=3025,
    ),
    MediumSpec(
        key="minimal",
        column="MOPS minimal",
        column_cell="C1",
        supplement=None,
        generation_time_minutes=56.3,
        generation_time_source=DOUBLING_MINIMAL,
        plain_cells=3362,
        stored_keys=3346,
    ),
    MediumSpec(
        key="complete_without_methionine",
        column="MOPS complete without methionine",
        column_cell="D1",
        supplement="full supplement without L-methionine",
        generation_time_minutes=26.5,
        generation_time_source=DOUBLING_COMPLETE_AND_DROPOUT,
        plain_cells=2241,
        stored_keys=2226,
    ),
)
REFERENCE_MEDIUM = "complete"

_LOOKED_IN = Provenance(
    source_uri=PAPER_TEXT_REL,
    citation_key=CITATION_KEY,
    sha256=PAPER_TEXT_SHA256,
    method="PMC Article Datasets plain text of the NIH author manuscript",
    page=(
        "Results, Methods; si/...-mmc6.pdftotext.txt Extended Experimental "
        "Procedures; Table S1 (mmc1) columns; GEO GSE53767 sample list"
    ),
)
#: Neither a per-gene replicate count nor a per-gene uncertainty is released.
N_REPLICATES_GAP = ProvenanceGap(
    field="n_replicates",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_LOOKED_IN,
    note=(
        "the paper states 'an error of less than 1.3-fold across biological "
        "replicates' over all genes and gives no per-gene replicate count; GEO "
        "GSE53767 holds pooled tracks of one rich-medium and two minimal-medium "
        "ribosome-profiling samples and none for the methionine dropout"
    ),
)
SYNTHESIS_RATE_SE_GAP = ProvenanceGap(
    field="synthesis_rate_se",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_LOOKED_IN,
    note=(
        "Table S1 carries one integer per gene and medium; no standard error, "
        "interval or replicate column is released"
    ),
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/liQuantifyingAbsoluteProtein2014``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json`` (written by the release inventory)."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Reading Table S1
# --------------------------------------------------------------------------- #
_BRACKET = re.compile(r"^\[(\d+)\]$")


class RefusalReason(StrEnum):
    """Why one released cell of Table S1 is not a key of any record."""

    #: ``[n]``: a gene the authors did not evaluate (<=128 footprints), back-solved.
    below_read_count_gate = "below_read_count_gate"
    #: ``tufA+tufB``: one value released for two loci.
    merged_pair_one_value_two_loci = "merged_pair_one_value_two_loci"
    #: the pinned assembly retires the name (IS elements, the dnaX frameshift split).
    name_retired_in_assembly = "name_retired_in_assembly"
    #: the name resolves to several loci of the pinned assembly.
    name_ambiguous_in_assembly = "name_ambiguous_in_assembly"
    #: two released rows resolve to ONE locus, so a gene-level key cannot hold both.
    two_rows_one_locus = "two_rows_one_locus"


class TableS1Row(BaseModel):
    """One gene row of Table S1 with its three cells as released."""

    model_config = ConfigDict(frozen=True)

    gene: str
    cells: tuple[int | str, int | str, int | str]


def read_table_s1(path: str | Path) -> tuple[tuple[str, ...], list[TableS1Row]]:
    """Header and gene rows of Table S1, every cell as openpyxl reads it."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(book[TABLE_S1_SHEET].iter_rows(values_only=True))
    book.close()
    header = tuple(str(cell) for cell in rows[0])
    body = [
        TableS1Row(gene=str(row[0]), cells=(row[1], row[2], row[3])) for row in rows[1:]
    ]
    return header, body


def plain_value(cell: int | str) -> int | None:
    """The released integer of a plain cell; None for a ``[n]`` cell; raise otherwise."""
    if isinstance(cell, int) and not isinstance(cell, bool):
        return cell
    if isinstance(cell, str) and _BRACKET.match(cell):
        return None
    raise ValueError(f"Table S1 cell {cell!r} is neither an integer nor '[n]'")


class RefusedCell(BaseModel):
    """One released cell that is not stored, with its typed reason."""

    model_config = ConfigDict(frozen=True)

    medium: str
    gene: str
    cell: str
    reason: RefusalReason


class MediumValues(BaseModel):
    """What one medium's column stores and refuses."""

    synthesis_rate: dict[str, float]
    refused: list[RefusedCell]


def name_refusals(reconciliation: LocusTagReconciliation) -> dict[str, RefusalReason]:
    """Typed refusal of every single gene name that does not reach one locus."""
    out: dict[str, RefusalReason] = {}
    for name in reconciliation.retired_kept:
        out[name] = RefusalReason.name_retired_in_assembly
    for name in reconciliation.ambiguous_kept:
        out[name] = RefusalReason.name_ambiguous_in_assembly
    for name in reconciliation.kept_on_collision:
        out[name] = RefusalReason.two_rows_one_locus
    unaccounted = set(reconciliation.outside_namespace) - set(out)
    if unaccounted:
        raise RuntimeError(
            f"{sorted(unaccounted)} fall outside the namespace with no typed reason"
        )
    return out


def medium_values(
    column: int,
    medium: MediumSpec,
    rows: Sequence[TableS1Row],
    locus_of: Mapping[str, str],
    refusals: Mapping[str, RefusalReason],
) -> MediumValues:
    """Plain cells keyed by locus tag, and every other cell with its typed reason."""
    rates: dict[str, float] = {}
    refused: list[RefusedCell] = []
    for row in rows:
        cell = row.cells[column]
        value = plain_value(cell)
        if value is None:
            refused.append(
                RefusedCell(
                    medium=medium.key,
                    gene=row.gene,
                    cell=str(cell),
                    reason=RefusalReason.below_read_count_gate,
                )
            )
            continue
        reason = (
            RefusalReason.merged_pair_one_value_two_loci
            if "+" in row.gene
            else refusals.get(row.gene)
        )
        if reason is not None:
            refused.append(
                RefusedCell(
                    medium=medium.key, gene=row.gene, cell=str(cell), reason=reason
                )
            )
            continue
        locus = locus_of[row.gene]
        if locus in rates:
            raise RuntimeError(f"{medium.key}: locus {locus} is keyed twice")
        rates[locus] = float(value)
    return MediumValues(synthesis_rate=rates, refused=refused)


# --------------------------------------------------------------------------- #
# Environment, reference and the phenotype
# --------------------------------------------------------------------------- #
def _supplement(name: str) -> MediaComponent:
    """The Neidhardt 1974 supplement, a defined sub-mix not expanded here.

    Li 2014 names the supplement by citation only. Its amino acid, vitamin and base
    composition lives in Neidhardt 1974, which is not mirrored, so the component is
    ``composition_deferred`` with ``defers_to`` naming it. The methionine dropout is a
    different named sub-mix rather than a ``dropout`` edit, because an edit of an
    unexpanded mixture would claim a composition this record does not hold.
    """
    return MediaComponent(
        compound=Compound(name=f"{name} (Neidhardt et al., 1974)"),
        role=MediaComponentRole.other,
        definition=ComponentDefinition.composition_deferred,
        provenance=[SUPPLEMENTS],
        defers_to=["neidhardtCultureMediumEnterobacteria1974"],
    )


def medium(spec: MediumSpec) -> Media:
    """MOPS glucose 0.2% with the supplement this column names, or none."""
    components = [
        *MOPS_MINIMAL.components,
        MediaComponent(
            compound=resolved_compound("glucose"),
            role=MediaComponentRole.carbon_source,
            concentration=Concentration(value=0.2, unit=ConcentrationUnit.percent_w_v),
            provenance=[BASE_MEDIUM],
        ),
    ]
    if spec.supplement is not None:
        components.append(_supplement(spec.supplement))
    return Media(
        name=f"{spec.column}, 0.2% glucose (Li 2014)",
        state="liquid",
        is_synthetic=True,
        base_medium="MOPS_MINIMAL",
        components=components,
        provenance=[BASE_MEDIUM, SUPPLEMENTS, spec.header],
    )


def environment(spec: MediumSpec) -> Environment:
    """Aerobic 37 C batch culture in one of the three media."""
    return Environment(
        media=medium(spec),
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        aerobicity="aerobic",
    )


def strain_reference(data_root: str | None = None) -> AssemblyReferenceGenome:
    """The assembly-pinned MG1655 reference every record is written against."""
    return assembly_reference(MG1655_STRAIN, data_root=data_root)


def phenotype(spec: MediumSpec, values: MediumValues) -> ProteinSynthesisRatePhenotype:
    """The synthesis-rate phenotype of one medium."""
    return ProteinSynthesisRatePhenotype(
        synthesis_rate=values.synthesis_rate,
        rate_unit=SynthesisRateUnit(UNIT.value),
        generation_time_minutes=spec.generation_time_minutes,
        measurement_type=MEASUREMENT_TYPE,
        provenance_gaps=[N_REPLICATES_GAP, SYNTHESIS_RATE_SE_GAP],
    )


def publication() -> Publication:
    """This paper, by DOI and PubMed id (PMC JATS ``article-id pub-id-type=pmid``)."""
    return Publication(
        pubmed_id=PUBMED_ID,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
        doi=DOI,
        doi_url=f"https://doi.org/{DOI}",
    )


class BuildAccounting(BaseModel):
    """What the build read, what it kept, and the arithmetic that ties the two."""

    dataset: str
    source_rows: int
    kept_records: int
    per_medium_plain_cells: dict[str, int]
    per_medium_keys: dict[str, int]
    per_medium_refusals: dict[str, dict[str, int]]
    per_medium_zero_rates: dict[str, int]
    reconciliation: LocusTagReconciliation

    def check(self) -> None:
        """Kept + refused cells of every medium add up to the released rows."""
        for key, kept in self.per_medium_keys.items():
            refused = sum(self.per_medium_refusals[key].values())
            if kept + refused != self.source_rows:
                raise RuntimeError(
                    f"{key}: {kept} kept + {refused} refused != {self.source_rows}"
                )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class ProteinSynthesisRateLi2014Dataset(ExperimentDataset):
    """Li 2014 absolute protein synthesis rates of MG1655 in three MOPS media."""

    REFERENCE_STRAIN: ClassVar[Literal["MG1655"]] = MG1655_STRAIN
    #: Every record is the wild type: the three records differ by medium only.
    has_gene_perturbations = False

    def __init__(
        self,
        root: str = "data/torchcell/protein_synthesis_rate_li2014",
        io_workers: int = 0,
        ecoli_genome: EcoliK12MG1655Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the MG1655 genome resolves Table S1's gene names."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProteinSynthesisRateExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProteinSynthesisRateExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Table S1, the one released workbook."""
        return [TABLE_S1_FILENAME]

    def download(self) -> None:
        """Link the deposited Table S1 into ``raw/`` after verifying its pins."""
        manifest = load_manifest()
        check_manifest_pin(
            TABLE_S1_REL, manifest_sha256(manifest, TABLE_S1_REL), TABLE_S1_SHA256
        )
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(
            raw_mirror_dir() / TABLE_S1_REL,
            osp.join(self.raw_dir, TABLE_S1_FILENAME),
            TABLE_S1_SHA256,
        )

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

    @post_process
    def process(self) -> None:
        """Build one synthesis-rate record per released medium; write LMDB."""
        verify_raw_files(self.raw_dir, {TABLE_S1_FILENAME: TABLE_S1_SHA256})
        header, rows = read_table_s1(osp.join(self.raw_dir, TABLE_S1_FILENAME))
        expected_header = ("Gene", *(spec.column for spec in MEDIA))
        if header != expected_header:
            raise RuntimeError(f"Table S1 header {header} != {expected_header}")
        if len(rows) != SOURCE_ROWS:
            raise RuntimeError(f"{len(rows)} Table S1 rows, not {SOURCE_ROWS}")

        single = [row.gene for row in rows if "+" not in row.gene]
        stored, reconciliation = reconcile_locus_tags(
            self._genome(), pd.Series(single), label=self.name
        )
        refusals = name_refusals(reconciliation)
        locus_of = dict(zip(single, (str(tag) for tag in stored), strict=True))

        by_medium: dict[str, MediumValues] = {}
        for column, spec in enumerate(MEDIA):
            plain = sum(1 for row in rows if plain_value(row.cells[column]) is not None)
            if plain != spec.plain_cells:
                raise RuntimeError(
                    f"{spec.key}: {plain} plain cells, not the pinned {spec.plain_cells}"
                )
            values = medium_values(column, spec, rows, locus_of, refusals)
            if len(values.synthesis_rate) != spec.stored_keys:
                raise RuntimeError(
                    f"{spec.key}: {len(values.synthesis_rate)} stored keys, not the "
                    f"pinned {spec.stored_keys}"
                )
            by_medium[spec.key] = values
        if MEDIA[0].plain_cells != EVALUATED_GENES.value:
            raise RuntimeError(
                "the MOPS-complete plain-cell count no longer back-solves the bracket "
                f"({EVALUATED_GENES.quote!r})"
            )

        reference_spec = next(s for s in MEDIA if s.key == REFERENCE_MEDIUM)
        reference = ProteinSynthesisRateExperimentReference(
            dataset_name=self.name,
            genome_reference=strain_reference(),
            environment_reference=environment(reference_spec),
            phenotype_reference=phenotype(reference_spec, by_medium[REFERENCE_MEDIUM]),
        )
        pub = publication()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        index = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for spec in tqdm(MEDIA, desc="li2014-synthesis-rate"):
                experiment = ProteinSynthesisRateExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(perturbations=[]),
                    environment=environment(spec),
                    phenotype=phenotype(spec, by_medium[spec.key]),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                index += 1
        env.close()
        interned_env.close()
        if index != EXPECTED_RECORDS:
            raise RuntimeError(f"{index} records, not {EXPECTED_RECORDS}")
        self._write_reports(by_medium, reconciliation, index)
        log.info(
            "Li2014 synthesis rates: %d records, keys %s",
            index,
            {k: len(v.synthesis_rate) for k, v in by_medium.items()},
        )

    def _write_reports(
        self,
        by_medium: Mapping[str, MediumValues],
        reconciliation: LocusTagReconciliation,
        kept_records: int,
    ) -> None:
        """Write the per-cell refusal ledger and the build accounting."""
        with open(
            osp.join(self.preprocess_dir, "refused_cells.csv"), "w", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["medium", "gene", "cell", "reason"])
            for values in by_medium.values():
                for refused in values.refused:
                    writer.writerow(
                        [refused.medium, refused.gene, refused.cell, refused.reason]
                    )
        accounting = BuildAccounting(
            dataset=self.name,
            source_rows=SOURCE_ROWS,
            kept_records=kept_records,
            per_medium_plain_cells={spec.key: spec.plain_cells for spec in MEDIA},
            per_medium_keys={k: len(v.synthesis_rate) for k, v in by_medium.items()},
            per_medium_refusals={
                k: dict(Counter(str(r.reason) for r in v.refused))
                for k, v in by_medium.items()
            },
            per_medium_zero_rates={
                k: sum(1 for rate in v.synthesis_rate.values() if rate == 0.0)
                for k, v in by_medium.items()
            },
            reconciliation=reconciliation,
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
            "ProteinSynthesisRateLi2014Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# Verification: a loader-level L0-L4 gate (no synthesis-rate family verifier exists)
# --------------------------------------------------------------------------- #
def _records(dataset_root: str) -> list[dict[str, Any]]:
    """The built records of one tree."""
    from torchcell.verification.runners import load_records

    return load_records(dataset_root)


def _medium_of(record: Mapping[str, Any]) -> str:
    """The medium key of a built record, read off its environment's media name."""
    name = str(record["experiment"]["environment"]["media"]["name"])
    for spec in MEDIA:
        if name == medium(spec).name:
            return spec.key
    raise KeyError(f"no released medium is named {name!r}")


def _l1_per_medium_keys(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    """L1: each medium's record keys the pinned count (plain cells less refusals)."""
    expected = {spec.key: spec.stored_keys for spec in MEDIA}
    observed = {
        _medium_of(record): len(record["experiment"]["phenotype"]["synthesis_rate"])
        for record in records
    }
    return LevelResult(
        level=Level.L1,
        name="per_medium_key_counts_are_the_pinned_ones",
        passed=observed == expected and len(observed) == len(records),
        message=f"observed {observed}; expected {expected}",
        details={"observed": observed, "expected": expected},
    )


def _l2_generation_time(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    """L2: each record's generation time is its medium's SI doubling time."""
    expected = {spec.key: spec.generation_time_minutes for spec in MEDIA}
    observed = {
        _medium_of(record): record["experiment"]["phenotype"]["generation_time_minutes"]
        for record in records
    }
    return LevelResult(
        level=Level.L2,
        name="generation_time_is_the_si_doubling_time",
        passed=observed == expected,
        message=f"observed {observed}; expected {expected}",
        details={"observed": observed},
    )


def _l3_assembly_pin(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    """L3: every reference pins the MG1655 GenBank assembly."""
    pins = {
        (
            str(record["reference"]["genome_reference"].get("assembly_set")),
            str(record["reference"]["genome_reference"].get("strain")),
        )
        for record in records
    }
    expected = {(MG1655_ASSEMBLY_SET, MG1655_STRAIN)}
    return l3_convention(
        "assembly_pin_is_mg1655_genbank",
        pins == expected,
        detail=f"{len(pins)} distinct pin(s): {sorted(pins)}",
    )


#: Every text-quoted value; each is re-read against the raw-mirror bytes at L3.
TEXT_SOURCED_VALUES: tuple[SourcedValue, ...] = (
    STRAIN,
    BASE_MEDIUM,
    SUPPLEMENTS,
    TEMPERATURE_C,
    DOUBLING_COMPLETE_AND_DROPOUT,
    DOUBLING_MINIMAL,
    REFERENCE_CONDITION,
    UNIT,
    STABLE_EQUALS_COPY_NUMBER,
    EVALUATED_GENES,
    READ_COUNT_GATE,
    UPPER_BOUND_FOR_DEGRADED,
    GEO_DEPOSIT,
)


def _l3_quotes_are_verbatim(data_root: str | None) -> LevelResult:
    """L3: every quote is a verbatim substring of its sha256-pinned mirror file.

    Text quotes are re-read with ``audit_sourced_value`` (hash first, then substring);
    the Table S1 header quotes are read back out of the pinned workbook's cells.
    """
    raw_root = Path(data_root or _data_root()) / "torchcell-raw"
    failed = [
        sv.quote[:60]
        for sv in TEXT_SOURCED_VALUES
        if not audit_sourced_value(sv, raw_root).passed
    ]
    workbook = raw_mirror_dir(data_root) / TABLE_S1_REL
    if file_sha256(workbook) != TABLE_S1_SHA256:
        failed.append(TABLE_S1_REL)
    header, _ = read_table_s1(workbook)
    failed.extend(
        spec.column
        for column, spec in enumerate(MEDIA, start=1)
        if header[column] != spec.header.quote
    )
    n = len(TEXT_SOURCED_VALUES) + len(MEDIA)
    return l3_convention(
        "sourced_quotes_are_verbatim_in_the_pinned_mirror",
        not failed,
        detail=f"{n} quotes re-read; {len(failed)} failed {failed}",
    )


def _l4_keys_are_loci(
    records: Sequence[Mapping[str, Any]], genome: EcoliK12Genome
) -> LevelResult:
    """L4: every stored protein key is a locus of the pinned assembly."""
    loci = set(genome.genbank.loci)
    keys = {
        key
        for record in records
        for key in record["experiment"]["phenotype"]["synthesis_rate"]
    }
    outside = sorted(keys - loci)
    return LevelResult(
        level=Level.L4,
        name="stored_protein_keys_are_loci_of_the_pinned_assembly",
        passed=bool(keys) and not outside,
        message=f"{len(keys)} distinct keys; {len(outside)} outside the assembly",
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

    L0 validates every record against the schema union; L1 checks the record count and
    each medium's pinned key count; L2 checks the rates are finite and non-negative and
    the generation times are the SI's; L3 checks the assembly pin; L4 contains every
    stored key in the pinned assembly's loci.
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
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{TABLE_S1_REL}",
            citation_key=CITATION_KEY,
            sha256=TABLE_S1_SHA256,
            method=(
                "Table S1 plain integer cells, one ProteinSynthesisRateExperiment per "
                "medium, molecules per generation; [n] cells refused"
            ),
            page=f"sheet {TABLE_S1_SHEET}",
            retrieved=RAW_RETRIEVED_AT,
        ),
    )
    report.add(l0_structural((record["experiment"] for record in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_per_medium_keys(records))
    rates = [
        float(value)
        for record in records
        for value in record["experiment"]["phenotype"]["synthesis_rate"].values()
    ]
    report.add(l2_value_fidelity(rates, allow_nan=False, minimum=0.0))
    report.add(_l2_generation_time(records))
    report.add(_l3_assembly_pin(records))
    report.add(_l3_quotes_are_verbatim(data_root))
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
    root = osp.join(data_root, "data/torchcell/protein_synthesis_rate_li2014")
    dataset = ProteinSynthesisRateLi2014Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(Path(osp.join(root, "preprocess/build_accounting.json")).read_text()[:2000])
    print(json.dumps(verify_build(root, data_root=data_root).summary()))


if __name__ == "__main__":
    main()
