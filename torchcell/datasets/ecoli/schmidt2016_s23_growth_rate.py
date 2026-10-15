# torchcell/datasets/ecoli/schmidt2016_s23_growth_rate
# [[torchcell.datasets.ecoli.schmidt2016_s23_growth_rate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/schmidt2016_s23_growth_rate
# Test file: tests/torchcell/datasets/ecoli/test_schmidt2016_s23_growth_rate.py
r"""Schmidt 2016 Table S23: the per-condition growth rates, as an absolute readout.

Schmidt et al. 2016 (Nat Biotechnol 34:104, doi:10.1038/nbt.3418) Supplementary Table
23 releases one ``Growth rate (h-1)`` with a ``Stdev`` per (growth condition, strain),
26 rows over three strains. This module serves 24 of them as
:class:`GrowthRateS23Schmidt2016Dataset`, one ``EnvironmentResponseExperiment``
per (condition, strain) carrying the ABSOLUTE rate, with each strain's Glucose row as its
reference.

Table S23 was the E. coli SI audit's rank 13 and was blocked. Three blockers were
stated; re-measured on 2026-10-09 after PR #836 (#776) only the third survives, and it
is not a blocker:

1. "No absolute growth member, so L3 ``reference_zero`` is unsatisfiable" is LIFTED.
   ``MeasurementType.growth_rate`` is in ``ABSOLUTE_MEASUREMENT_TYPES``, so
   ``reference_centered=False`` is admissible and the reference states its own finite
   rate on the record's own scale instead of a 0 that would assert instant growth.
2. "Two rows are negative, which a log2 ratio cannot hold" is MOOT on the absolute
   route: ``EnvironmentResponsePhenotype`` never clamps, and in any case both negative
   rows are the two stationary-phase rows, which drop for an unrelated reason below.
3. "The ``Stdev``'s replicate design is not back-solvable" is CONFIRMED and is handled
   under THE STDEV below rather than by dropping the statistic.

A SEPARATE MODULE FROM ``schmidt2016.py``, on purpose, and from
``schmidt2016_growth_rate.py`` too. ``build_manifest`` keys a built store's staleness on
the schema closure of the loader MODULE's own ``torchcell.datamodels`` imports
(``provenance/schema_deps.py:loader_closure``), so importing
``EnvironmentResponsePhenotype`` into either of those modules would mark the served
``proteome_schmidt2016`` or ``growth_rate_schmidt2016`` store stale for a change that
touches none of their records. The pinned artifact, the media objects, the condition
table and the environment helpers are imported FROM ``schmidt2016`` instead, so there is
one copy of each.

NO DROPPED ROW: 26 released rows, 26 records. Since #1057 (the owner rule of 2026-10-12:
"the data is the data", and the genome does not gate a released row) every row is
stored as released, and since #1045 part 3 the two stationary-phase rows are too: they
take ``schmidt2016``'s condition table, whose starved arms carry
``Environment.growth_phase = stationary`` and ``duration_hours`` 24 and 72 (they were
``growth_phase_not_representable``, which would have collapsed them onto the Glucose
arm). Their released growth rate is -0.01 h-1 and is stored as released.

* The 4 rows of the two non-BW25113 strains are stored, each on a
  ``BacterialStrainBackground`` named as Table S23 writes it (``MG1665`` verbatim, the
  paper's text saying MG1655; ``NCM3722``) with the genotype the paper's text states
  (``F-, λ, rph-1``; ``F+``), pinned to the K-12 MG1655 reference assembly, which no
  gene of these wild-type records uses. Each strain's reference is its own Glucose row.
  Every Table S23 row is wild type, so if the shared L1 ``pair_uniqueness`` signature
  carries no strain field it reads the three strains' LB and Glucose rows as
  duplicates; that is a property of the verifier's key, reported rather than resolved
  by dropping rows.
* The 4 chemostat rows are stored. Their environments carry the dilution rate (#1045 row
  21); the growth rate Table S23 releases for them is that set point, and its ``Stdev``
  is printed as ``0``, which is no measured spread, so the uncertainty is a typed gap
  quoting the released 0 rather than a sample SD of 0.

``Glycerol + AA`` was dropped on ``medium_has_no_media_library_entry`` until #1045 row 17
gave it ``M9_AA_SCHMIDT2016``; it is a record now (1.27 h-1).

THE STDEV, AND THE ONE INFERENTIAL STEP. The released ``Stdev`` is stored as
``environment_response_uncertainty`` with ``UncertaintyType.sample_sd``, and
``EnvironmentResponsePhenotype`` then requires an ``n_samples`` and a ``sample_unit``.
The paper never states the statistic's denominator, and a back-solve over all 30 sheets
of the pinned workbook is NEGATIVE: Table S24 is a different cultivation (its glucose
wild-type average is 0.5921 +/- 0.0120 against this table's 0.58 +/- 0.01, and its
acetate 0.3103 +/- 0.0057 against 0.30 +/- 0.04, so neither the means nor the SDs
match); Table S28's three replicate columns are protein mass, not rates, and its
``Experimental Growth rate`` column restates this table's single value; and Table S23's
own three unlabelled replicate cells are ``OD @ harvesting``, optical densities rather
than rates. Two candidate denominators are released and neither is tied to the
statistic: the cultivation was triplicate (``_Q_TRIPLICATES_22_CONDITIONS``, Table S25's
three sample files per condition and this table's own three OD-at-harvesting cells all
state 3), while the Methods fit each rate "from at least four consecutive measurements"
WITHIN one curve. ``n_samples = 3`` is the conservative resolution of that pair -- the
lower count, so the larger standard error -- and it is the only one with a sourced
number; the note on ``SOURCED_VALUES["n_samples"]`` names the inferential step in the
record itself. Measured both ways on 2026-10-09: the then 15-record build passed all 16
L0-L4 rows with ``n_samples=3``, and also passed with the ``Stdev`` left unstored behind
two typed gaps. The four chemostat rows carry no sample SD (their released 0 is a typed
gap), so the inferential step touches only the 20 batch rows.

WHAT IS NOT STORED, with its count. Table S23's other released columns have no axis in
the schema and are written verbatim to ``preprocess/not_stored.json`` instead of being
coerced onto one: ``Single cell volume [fl]1`` (26 values), ``Doubling time (h-1)``
(22 numeric, 4 of them the string ``after >6 volume changes``), ``Time exp before
harvest (h)``, ``# of doublings at exponential growth before harvesting``, the three
``OD @ harvesting. replicates`` cells and ``Number of Proteins Identified (FDR 1%)2``.
No cell of the 26 x 12 block is blank; what reads blank is typed text (``-`` in the two
stationary rows, ``after >6 volume changes`` in the four chemostat rows).

RECORDED, NOT ACTED ON. Table S28 restates the two stationary-phase conditions'
``Experimental Growth rate`` as ``0`` where Table S23 releases ``-0.01``: an internal
inconsistency of the release, on the two rows this module drops for an unrelated reason.

DATA. The one consumed file is ``si2.xlsx``, the same pinned Supplementary-tables
workbook ``schmidt2016.py`` consumes, read from the same raw mirror under the same
sha256 pin. Nothing is retrieved here that that module does not already record.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import openpyxl
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.schema import (
    AssayType,
    BacterialStrainBackground,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    ReferenceGenome,
    SampleUnit,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import assembly_reference
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.ecoli import schmidt2016 as sm
from torchcell.sequence.genome.ecoli.k12 import EcoliK12StrainName
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

DATASET_ROOT_REL = "data/torchcell/growth_rate_s23_schmidt2016"

SHEET_S23 = "Table S23"
#: Table S23's released headers, in file order. Columns 10 and 11 are ``None``: they are
#: the two unlabelled continuation cells of ``OD @ harvesting. replicates``.
S23_HEADERS: tuple[str | None, ...] = (
    "Growth condition",
    "Strain",
    "Growth rate (h-1)",
    "Stdev",
    "Single cell volume [fl]1",
    "Doubling time (h-1)",
    "Time exp before harvest (h)",
    "# of doublings at exponential growth before harvesting",
    "OD @ harvesting. replicates",
    None,
    None,
    "Number of Proteins Identified (FDR 1%)2",
)
COL_CONDITION = 0
COL_STRAIN = 1
COL_RATE = 2
COL_STDEV = 3
#: The columns with no axis in the schema, by position, for ``not_stored.json``.
UNSTORED_COLUMNS: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 11)

#: The strain spelling of each released row, and what it is.
STRAIN_BW25113 = "BW25113"
STRAIN_MG1665 = "MG1665"
STRAIN_NCM3722 = "NCM3722"

#: Build oracles, every one measured on the pinned bytes before being written here.
EXPECTED_SOURCE_ROWS = 26
EXPECTED_STRAIN_ROWS: dict[str, int] = {
    STRAIN_BW25113: 22,
    STRAIN_MG1665: 2,
    STRAIN_NCM3722: 2,
}
EXPECTED_RECORDS = 26
#: What L3 ``environment_perturbed`` must observe: 0 records with no environmental edit
#: at all. An edit is a perturbation, a non-baseline temperature OR a non-baseline
#: medium, which is what covers the one record below.
EXPECTED_UNPERTURBED = 0
#: The ONE kept condition whose only edit is its medium: ``LB`` is a complex medium with
#: no carbon-source factor on top, so its ``Environment.perturbations`` is empty while
#: its ``media`` differs from the M9 records' baseline. Its three rows (BW25113, MG1665,
#: NCM3722, in sheet order) are records since #1057. Declared so that a condition
#: silently losing its perturbation cannot pass as this one.
EXPECTED_MEDIA_ONLY_EDIT: tuple[str, ...] = ("LB", "LB", "LB")

#: The released condition label that IS the reference, and ``schmidt2016``'s own name
#: for it. The Glucose row is also a record, as Caglar 2017's base condition is: for an
#: absolute readout a reference condition is a measured condition too.
REFERENCE_CONDITION = sm.REFERENCE_CONDITION

#: The units of the readout, the released header plus the Methods' fit.
UNITS = "growth rate in h^-1, from flow-cytometry cell counts over time"
#: The denominator of the released ``Stdev``, resolved conservatively; see THE STDEV.
N_SAMPLES = 3

_Q_TABLE_S23 = (
    "Table S23 | Experimental details for all E. coli samples analzed in "
    "this study including growth rate, harvesting conditions, OD-values "
    "and number of identified proteins"
)
_Q_GROWTH_RATE_FIT = (
    "The growth rate of the cultures was determined from the cell counts "
    "over time at cell concentrations from $1 0 ^ { 5 } c e l l s / "
    "\\mathrm { m l }$ to $1 0 ^ { 9 } \\mathrm { c e l l s / m l }$ The "
    "growth rate was calculated from at least four consecutive "
    "measurements."
)
_Q_TRIPLICATES_22_CONDITIONS = (
    "We grew E. coli BW25113 (ref. 19) under 22 different growth conditions in "
    "biological triplicates."
)
_Q_CELL_COUNTS_BY_FLOW = (
    "Third, the numbers of cells taken for LC-MS/MS analyses were determined for each "
    "sample by flow cytometry."
)

PAPER = Provenance(
    source_uri=sm.PAPER_MD, citation_key=sm.CITATION_KEY, sha256=sm.PAPER_MD_SHA256
)
SI2_SOURCE = Provenance(
    source_uri=sm.SI2_MIRROR_RELPATH, citation_key=sm.CITATION_KEY, sha256=sm.SI2_SHA256
)


def _paper(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    """A ``SourcedValue`` quoting the pinned ``paper.md``."""
    return SourcedValue(value=value, provenance=PAPER, quote=quote, note=note)


def _si2(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    """A ``SourcedValue`` quoting the pinned ``si2.xlsx``."""
    return SourcedValue(value=value, provenance=SI2_SOURCE, quote=quote, note=note)


SOURCED_VALUES: dict[str, SourcedValue] = {
    "growth_table": _si2("Table S23", _Q_TABLE_S23),
    "reference_strain": _paper(STRAIN_BW25113, sm._Q_STRAIN),
    "other_strains": _paper(
        [STRAIN_MG1665, STRAIN_NCM3722],
        sm._Q_OTHER_STRAINS,
        note="the two rows per strain are stored since #1057, each strain named as Table "
        "S23 writes it on a background pinned to the K-12 MG1655 reference assembly",
    ),
    "growth_rate_estimator": _paper(
        "exponential fit to flow-cytometry cell counts over at least four consecutive "
        "measurements, between 1e5 and 1e9 cells/ml",
        _Q_GROWTH_RATE_FIT,
    ),
    "assay_type": _paper(
        AssayType.other.value,
        _Q_CELL_COUNTS_BY_FLOW,
        note="the paper states the assay and the schema has no member for it, so the "
        "record carries AssayType.other rather than a typed gap: the rate is fit to "
        "flow-cytometry cell counts, and liquid_od_growth would be a false type, "
        "because the released OD is the harvest point and not the readout the rate is "
        "fit to",
    ),
    "n_samples": _paper(
        N_SAMPLES,
        _Q_TRIPLICATES_22_CONDITIONS,
        note="the paper does not state what the released Stdev is computed over. Two "
        "denominators are released and neither is tied to the statistic: the triplicate "
        "cultivation this quote states (corroborated by Table S25's three sample files "
        "per condition and by this table's own three OD-at-harvesting cells), and the "
        "Methods' 'at least four consecutive measurements' within one growth curve. 3 "
        "is the conservative resolution, the lower count and so the larger standard "
        "error, and the only one with a sourced number; a back-solve over all 30 sheets "
        "of the pinned workbook is negative",
    ),
    "temperature_c": _paper(37.0, sm._Q_BATCH_37C),
    "aerobicity": sm.SOURCED_VALUES["aerobicity"],
}

_GAP_REPLICATE_ID = ProvenanceGap(
    field="replicate_id",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="Table S23 releases one aggregate rate per (condition, strain) and no "
    "per-replicate rate, so there is no released replicate to identify",
)
RECORD_GAPS: tuple[ProvenanceGap, ...] = (_GAP_REPLICATE_ID,)


# --------------------------------------------------------------------------- #
# Reading Table S23
# --------------------------------------------------------------------------- #
class S23Row(BaseModel):
    """One released Table S23 row, typed, with every unstored cell kept verbatim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_number: int = Field(description="1-based row in the sheet, for the ledger.")
    condition: str = Field(description="``Growth condition``, verbatim.")
    strain: str = Field(description="``Strain``, verbatim (``MG1665`` included).")
    growth_rate: float
    stdev: float
    unstored: dict[str, Any] = Field(
        description="the released cells with no axis in the schema, by header position"
    )


def _condition_spec(label: str) -> sm.ConditionSpec:
    """The ``schmidt2016`` condition one Table S23 label names.

    The released label is normalized exactly three ways, each of which a measured
    spelling of this table needs: verbatim (most rows), stripped of padding space
    (``'Galactose '``) and stripped of a trailing footnote digit
    (``'Osmotic-stress glucose3'``). The four chemostat rows match the lowercase
    ``s25_label``; every other row matches the ``s6_column``.
    """
    by_column = {spec.s6_column: spec for spec in sm.CONDITIONS}
    by_label = {spec.s25_label: spec for spec in sm.CONDITIONS}
    for key in (label, label.strip(), label.strip().rstrip("0123456789")):
        if key in by_column:
            return by_column[key]
        if key in by_label:
            return by_label[key]
    raise RuntimeError(
        f"Table S23's growth condition {label!r} names no schmidt2016 condition; the "
        "released condition set has changed"
    )


def read_table_s23(path: str | Path) -> list[S23Row]:
    """Table S23's 26 released rows, typed, with the sheet's own header asserted.

    The header must be the module's declared 12 cells, so a re-export that renames or
    reorders a column stops the build rather than reading a rate out of the wrong one.
    Every row must name a ``schmidt2016`` condition and one of the three released strain
    spellings, and the per-strain row counts must be the measured ones.
    """
    book = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    try:
        sheet = list(book[SHEET_S23].iter_rows(values_only=True))
    finally:
        book.close()
    header = tuple(
        cell if cell is None else str(cell).strip()
        for cell in sheet[sm.HEADER_ROW - 1][: len(S23_HEADERS)]
    )
    if header != S23_HEADERS:
        raise RuntimeError(
            f"Table S23 is headed {header}, the module declares {S23_HEADERS}"
        )
    rows: list[S23Row] = []
    for offset, cells in enumerate(sheet[sm.HEADER_ROW :], start=sm.HEADER_ROW + 1):
        if cells[COL_CONDITION] is None:
            break
        condition = str(cells[COL_CONDITION])
        _condition_spec(condition)
        strain = str(cells[COL_STRAIN]).strip()
        if strain not in EXPECTED_STRAIN_ROWS:
            raise RuntimeError(
                f"Table S23 row {offset} names the strain {strain!r}, which is none of "
                f"the released {sorted(EXPECTED_STRAIN_ROWS)}"
            )
        rows.append(
            S23Row(
                row_number=offset,
                condition=condition,
                strain=strain,
                growth_rate=float(cells[COL_RATE]),
                stdev=float(cells[COL_STDEV]),
                unstored={
                    str(S23_HEADERS[position] or f"unlabelled_{position}"): cells[
                        position
                    ]
                    for position in UNSTORED_COLUMNS
                },
            )
        )
    if len(rows) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(
            f"Table S23 holds {len(rows)} rows, the module declares "
            f"{EXPECTED_SOURCE_ROWS}"
        )
    counts = {
        strain: sum(1 for row in rows if row.strain == strain)
        for strain in EXPECTED_STRAIN_ROWS
    }
    if counts != EXPECTED_STRAIN_ROWS:
        raise RuntimeError(
            f"Table S23's per-strain row counts are {counts}, the module declares "
            f"{EXPECTED_STRAIN_ROWS}"
        )
    return rows


class DropLedger(BaseModel):
    """Which released rows became records, and under which rule the rest did not."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_rows: int
    kept_records: int
    dropped: dict[str, list[str]] = Field(
        description="``{rule: ['<condition>/<strain>', ...]}`` over the dropped rows"
    )

    @property
    def dropped_rows(self) -> int:
        """Released rows that are not records."""
        return sum(len(rows) for rows in self.dropped.values())


def partition(rows: Sequence[S23Row]) -> tuple[list[S23Row], DropLedger]:
    """Split the released rows into the records and the ledger of what dropped.

    One rule: a row whose ``schmidt2016`` condition carries a ``DropReason`` goes under
    that reason's own ``rule`` name. A row of another strain and a chemostat row are
    records since #1057.
    """
    kept: list[S23Row] = []
    dropped: dict[str, list[str]] = {}
    for row in rows:
        key = f"{row.condition}/{row.strain}"
        spec = _condition_spec(row.condition)
        if spec.drop is not None:
            dropped.setdefault(spec.drop.rule, []).append(key)
            continue
        kept.append(row)
    ledger = DropLedger(source_rows=len(rows), kept_records=len(kept), dropped=dropped)
    if ledger.kept_records + ledger.dropped_rows != ledger.source_rows:
        raise RuntimeError("the drop ledger does not account for every released row")
    return kept, ledger


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def build_environments(rows: Sequence[S23Row]) -> dict[str, Environment]:
    """``{released condition label: environment}`` for the kept rows.

    The media objects, the carbon sources and their g/L are ``schmidt2016``'s, sourced
    there from the Methods. ``check_environments_distinct`` then refuses two kept
    conditions that serialize to one environment, which is what makes the
    ``culture_not_batch`` and ``growth_phase_not_representable`` drops checked rather
    than asserted.
    """
    environments = {
        row.condition: sm.build_environment(_condition_spec(row.condition))
        for row in rows
    }
    sm.check_environments_distinct(environments)
    return environments


#: The typed gap on a chemostat row's uncertainty: Table S23 prints its ``Stdev`` as 0.
SET_POINT_STDEV_GAP = ProvenanceGap(
    field="environment_response_uncertainty",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=SI2_SOURCE,
    note="a chemostat row: Table S23 releases Stdev 0, because the growth rate it prints "
    "is the dilution rate the experimenter set; 0 is no measured spread, so no sample SD "
    "is stored (#1057)",
)


def phenotype(row: S23Row) -> EnvironmentResponsePhenotype:
    """One condition's absolute growth rate, with the released ``Stdev``.

    A chemostat row's rate is its dilution-rate set point and its released ``Stdev`` is
    0, so its uncertainty is the typed gap :data:`SET_POINT_STDEV_GAP` (#1057).
    """
    if _condition_spec(row.condition).dilution_rate_per_hour is not None:
        if row.stdev != 0.0:
            raise ValueError(
                f"chemostat row {row.condition!r} releases Stdev {row.stdev}, not the "
                "set point's 0"
            )
        return EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.growth_rate,
            assay_type=AssayType.other,
            environment_response=row.growth_rate,
            units=UNITS,
            provenance_gaps=[*RECORD_GAPS, SET_POINT_STDEV_GAP],
        )
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.growth_rate,
        assay_type=AssayType.other,
        environment_response=row.growth_rate,
        environment_response_uncertainty=row.stdev,
        environment_response_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=N_SAMPLES,
        sample_unit=SampleUnit.biological_replicate,
        units=UNITS,
        provenance_gaps=list(RECORD_GAPS),
    )


#: The genotype the paper's text writes for each non-BW25113 strain of Table S23, keyed
#: by the strain as Table S23 names it (``MG1665``; the text says MG1655). The value is
#: the quoted string with the OCR's citation superscripts (20, 21) removed;
#: ``sm._Q_OTHER_STRAINS`` is the verbatim quote.
STRAIN_GENOTYPES: dict[str, str] = {STRAIN_MG1665: "F-, λ, rph-1", STRAIN_NCM3722: "F+"}


def strain_genome(strain: str, data_root: str | None = None) -> ReferenceGenome:
    """BW25113's assembly pin, or another strain on a background named as released.

    A Table S23 strain that is not BW25113 is stored under the name Table S23 writes
    (``MG1665``, ``NCM3722``) on the K-12 MG1655 reference assembly, with the genotype
    the paper's text states for it. The records are wild type, so no gene uses the pin
    (#1057).
    """
    if strain == STRAIN_BW25113:
        return assembly_reference(
            GrowthRateS23Schmidt2016Dataset.REFERENCE_STRAIN, data_root=data_root
        )
    background = BacterialStrainBackground(
        name=strain,
        reference_strain="MG1655",
        assembly_set="ecoli_K12_MG1655_ASM584v2",
        genotype_statement=STRAIN_GENOTYPES[strain],
        provenance=[
            _si2(strain, strain, note="the strain name as Table S23 writes it"),
            _paper(STRAIN_GENOTYPES[strain], sm._Q_OTHER_STRAINS),
        ],
    )
    return assembly_reference("MG1655", background=background, data_root=data_root)


def reference_row(rows: Sequence[S23Row], strain: str = STRAIN_BW25113) -> S23Row:
    """A strain's Glucose row, its reference; BW25113's is the dataset's."""
    matches = [
        row
        for row in rows
        if row.condition == REFERENCE_CONDITION and row.strain == strain
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{len(matches)} kept rows carry the reference condition "
            f"{REFERENCE_CONDITION!r}, expected exactly one"
        )
    return matches[0]


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class GrowthRateS23Schmidt2016Dataset(ExperimentDataset):
    """Schmidt 2016 Table S23: 24 absolute per-condition growth rates, three strains."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = sm.SCHMIDT_REFERENCE_STRAIN
    #: Every released row is wild type, so the gene set is legitimately empty.
    has_gene_perturbations = False

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

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
        """The one consumed file, linked from the raw mirror."""
        return [sm.SI2]

    def download(self) -> None:
        """Link the mirrored workbook into ``raw/`` after checking manifest and sha256."""
        data_root = os.environ["DATA_ROOT"]
        manifest = sm.load_manifest(data_root)
        check_manifest_pin(
            sm.SI2_MIRROR_RELPATH,
            sm.manifest_sha256(manifest, sm.SI2_MIRROR_RELPATH),
            sm.SI2_SHA256,
        )
        src = sm.raw_mirror_dir(data_root) / sm.SI2_MIRROR_RELPATH
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(src, osp.join(self.raw_dir, sm.SI2), sm.SI2_SHA256)
        log.info(
            "Schmidt 2016 Table S23 raw file linked into %s (sha256 verified)",
            self.raw_dir,
        )

    @post_process
    def process(self) -> None:
        """Write one record per kept Table S23 (condition, strain), 24 in all."""
        verify_raw_files(self.raw_dir, sm.DATA_SHA256)
        rows = read_table_s23(osp.join(self.raw_dir, sm.SI2))
        kept, ledger = partition(rows)
        if ledger.kept_records != EXPECTED_RECORDS:
            raise RuntimeError(
                f"{ledger.kept_records} records survive the drop rules, the module "
                f"declares {EXPECTED_RECORDS}"
            )
        environments = build_environments(kept)
        media_only = tuple(
            row.condition
            for row in kept
            if not environments[row.condition].perturbations
        )
        if media_only != EXPECTED_MEDIA_ONLY_EDIT:
            raise RuntimeError(
                f"the conditions whose only environmental edit is their medium are "
                f"{media_only}, the module declares {EXPECTED_MEDIA_ONLY_EDIT}"
            )
        base = reference_row(kept)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        references = {
            strain: EnvironmentResponseExperimentReference(
                dataset_name=self.name,
                genome_reference=strain_genome(strain),
                environment_reference=environments[REFERENCE_CONDITION],
                phenotype_reference=phenotype(reference_row(kept, strain)),
            )
            for strain in sorted({row.strain for row in kept})
        }
        pub = sm.publication()
        genotype = Genotype(perturbations=[])
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in enumerate(tqdm(kept, desc="schmidt2016 Table S23")):
                experiment = EnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=genotype,
                    environment=environments[row.condition],
                    phenotype=phenotype(row),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, references[row.strain], pub, itxn),
                )
        env.close()
        interned_env.close()

        self._write_ledgers(rows, kept, ledger, base)
        log.info(
            "Schmidt 2016 Table S23: %d records of %d released rows; growth rate "
            "%.6f to %.6f h-1; reference %s = %.6f h-1",
            ledger.kept_records,
            ledger.source_rows,
            min(row.growth_rate for row in kept),
            max(row.growth_rate for row in kept),
            base.condition,
            base.growth_rate,
        )

    def _write_ledgers(
        self,
        rows: Sequence[S23Row],
        kept: Sequence[S23Row],
        ledger: DropLedger,
        base: S23Row,
    ) -> None:
        """The retention ledger, the sourcing table and the unstored columns."""
        out = Path(self.preprocess_dir)
        rules = {
            spec.drop.rule: spec.drop.description
            for spec in sm.CONDITIONS
            if spec.drop is not None
        }
        (out / "dropped_records.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    **ledger.model_dump(mode="json"),
                    "dropped_records": ledger.dropped_rows,
                    "rules": [
                        {"rule": rule, "description": rules[rule], "rows": members}
                        for rule, members in sorted(ledger.dropped.items())
                    ],
                    "notes": [
                        f"each strain's reference is its own {base.condition} row "
                        f"(BW25113's is {base.growth_rate} h-1), which is ALSO a "
                        "record: for an absolute readout a reference condition is a measured condition "
                        "too, as Caglar 2017's base condition is",
                        "the four chemostat rows release a Stdev of exactly 0, the "
                        "dilution-rate set-point rather than a measured spread; since "
                        "#1057 they are records whose uncertainty is a typed gap quoting "
                        "that 0",
                        "the MG1665 and NCM3722 rows are records since #1057, each on a "
                        "background named as released and referenced to its own Glucose "
                        "row",
                        "Table S28 restates the two stationary-phase conditions' growth "
                        "rate as 0 where Table S23 releases -0.01; since #1045 part 3 "
                        "both rows are records storing Table S23's -0.01 as released, "
                        "and the inconsistency is recorded, not acted on",
                    ],
                },
                indent=2,
            )
        )
        (out / "not_stored.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "issue": 826,
                    "why_not_stored": "no phenotype class in the schema carries a cell "
                    "volume, a harvest OD, a doubling count or an identified-protein "
                    "count, and deriving an axis for one would be inventing a field "
                    "rather than sourcing it",
                    "columns": [
                        str(S23_HEADERS[position] or f"unlabelled_{position}")
                        for position in UNSTORED_COLUMNS
                    ],
                    "values": {
                        f"{row.condition}/{row.strain}": row.unstored for row in rows
                    },
                },
                indent=2,
            )
        )
        (out / "sourced_values.json").write_text(
            json.dumps(
                {
                    name: value.model_dump(mode="json")
                    for name, value in SOURCED_VALUES.items()
                },
                indent=2,
            )
        )
        pd.DataFrame(
            [
                {
                    "row_number": row.row_number,
                    "condition": row.condition,
                    "strain": row.strain,
                    "growth_rate_h1": row.growth_rate,
                    "stdev": row.stdev,
                    "n_samples": (
                        None
                        if _condition_spec(row.condition).dilution_rate_per_hour
                        is not None
                        else N_SAMPLES
                    ),
                    "is_reference": row.condition == base.condition,
                }
                for row in kept
            ]
        ).to_csv(out / "growth_rates.csv", index=False)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "GrowthRateS23Schmidt2016Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# L0-L4 verification
# --------------------------------------------------------------------------- #
def l4_assembly_pin(
    records: Sequence[Mapping[str, Any]], data_root: str | None = None
) -> LevelResult:
    """L4: the record's pinned assembly IS the one the genomes tier deposited.

    Every record is wild type, so no record names a gene and the L4 gene rules have
    nothing to look at. What the records DO assert about the outside world is their
    genome pin, so the rule re-derives each strain's pin from the deposited assembly
    report (BW25113's own set; MG1665 and NCM3722 on the K-12 MG1655 set, #1057) and
    requires every record to carry its strain's pair. This is the Caglar 2017 L4.
    """
    by_strain: dict[str, set[tuple[str, str]]] = {}
    for rec in records:
        genome = rec["reference"]["genome_reference"]
        by_strain.setdefault(str(genome["strain"]), set()).add(
            (str(genome["assembly_set"]), str(genome["assembly_accession"]))
        )
    expected = {
        strain: (str(pin.assembly_set), str(pin.assembly_accession))
        for strain in sorted(by_strain)
        for pin in [strain_genome(strain, data_root)]
    }
    wrong = {
        strain: sorted(pins)
        for strain, pins in sorted(by_strain.items())
        if pins != {expected[strain]}
    }
    return LevelResult(
        level=Level.L4,
        name="assembly_pin_resolves",
        passed=not wrong,
        message=(
            f"all {len(records)} records pin their strain's deposited assembly: "
            + "; ".join(f"{k} {v[0]} / {v[1]}" for k, v in expected.items())
            if not wrong
            else f"records pin {wrong}; the deposited assembly reports name {expected}"
        ),
        details={
            "pins": {k: sorted(v) for k, v in sorted(by_strain.items())},
            "expected": expected,
            "n_records": len(records),
        },
    )


def verify_build(
    dataset_root: str,
    data_root: str | None = None,
    *,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the environment-response L0-L4 gate over a built tree and write the report.

    ``reference_centered=False`` selects the absolute branch PR #836 landed: the
    reference states its own finite rate on the record's own scale, and the branch
    refuses any record whose ``measurement_type`` is not in
    ``ABSOLUTE_MEASUREMENT_TYPES``. ``sgd_genes`` is not supplied: every record is wild
    type, so the shared L4 gene rules would run vacuously against a universe no record
    names, and :func:`l4_assembly_pin` is this dataset's L4 instead.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset,
    )
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    report = verify_environment_response_dataset(
        records,
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{sm.RAW_DIR_REL}/{sm.SI2_MIRROR_RELPATH}",
            citation_key=sm.CITATION_KEY,
            sha256=sm.SI2_SHA256,
            method=(
                "Supplementary Table 23 'Growth rate (h-1)' and 'Stdev': one "
                "EnvironmentResponseExperiment per kept (condition, strain) "
                "row, MeasurementType.growth_rate carrying the ABSOLUTE rate in h^-1; "
                "each strain's reference is its released Glucose row's own rate"
            ),
            page=f"si2.xlsx sheet '{SHEET_S23}'",
            retrieved=sm.SI2_RETRIEVED_AT,
        ),
        expected_count=expected_count,
        reference_centered=False,
        expected_unperturbed=EXPECTED_UNPERTURBED,
    )
    report.add(l4_assembly_pin(records, data_root))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``build`` the dev-tree LMDB, or ``verify`` an already built one."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.schmidt2016_s23_growth_rate"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, DATASET_ROOT_REL)
    if args.command == "build":
        dataset = GrowthRateS23Schmidt2016Dataset(root=root)
        print(f"len = {len(dataset)}")
        print(Path(root, "preprocess", "dropped_records.json").read_text())
        return 0
    report = verify_build(root, data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
