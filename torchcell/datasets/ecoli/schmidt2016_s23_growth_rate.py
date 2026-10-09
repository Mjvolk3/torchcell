# torchcell/datasets/ecoli/schmidt2016_s23_growth_rate
# [[torchcell.datasets.ecoli.schmidt2016_s23_growth_rate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/schmidt2016_s23_growth_rate
# Test file: tests/torchcell/datasets/ecoli/test_schmidt2016_s23_growth_rate.py
r"""Schmidt 2016 Table S23: the per-condition growth rates, as an absolute readout.

Schmidt et al. 2016 (Nat Biotechnol 34:104, doi:10.1038/nbt.3418) Supplementary Table
23 releases one ``Growth rate (h-1)`` with a ``Stdev`` per (growth condition, strain),
26 rows over three strains. This module serves the 15 of them that the schema can hold
as :class:`GrowthRateS23Schmidt2016Dataset`, one
``BacterialEnvironmentResponseExperiment`` per condition carrying the ABSOLUTE rate,
plus the Glucose row as the reference.

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

THE 11 DROPPED ROWS, each by a measured rule and not a preference. 26 released rows
minus 15 records:

* 4 rows are the two non-BW25113 strains. ``BacterialReferenceStrain`` is
  ``MG1655 | BW25113 | KT2440 | REL606`` and the genomes tier deposits an assembly set
  for each; ``NCM3722`` is in neither, so its 2 rows cannot be pinned to a genome at
  all. The other 2 rows spell the strain ``MG1665``, a transposition of ``MG1655``;
  even correcting it does not rescue them, because every Table S23 row is wild type, so
  L1 ``pair_uniqueness``'s genotype signature is the empty tuple for all three strains
  and its condition signature carries no strain field. Measured on the 17 BW25113 +
  typo-corrected MG1655 records: "2 records duplicate an earlier (study, strain,
  condition) triple; 15 unique triples" -- LB and Glucose collide across strains. The
  multi-strain rows are blocked on the uniqueness key, independently of the vocabulary.
* 7 rows carry the proteome loader's own structural drop rules, imported from
  ``schmidt2016`` rather than restated: ``Glycerol + AA`` on
  ``medium_has_no_media_library_entry``, the two stationary-phase rows on
  ``growth_phase_not_representable``, and the four chemostat rows on
  ``culture_not_batch``. The collapse those last two rules describe was re-measured here
  on records rather than on environments: a 19-record build with the chemostat rows in
  fails L1 with "3 records duplicate an earlier (study, strain, condition) triple; 16
  unique triples", and also fails L2 ``uncertainty_sanity`` with "4/19 labeled
  uncertainties are a sample dispersion of exactly 0", because the chemostat ``Stdev``
  is literally ``0`` in all four rows: it is the dilution-rate set-point, not a measured
  spread.

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
record itself. Measured both ways: the 15-record build passes all 16 L0-L4 rows with
``n_samples=3``, and also passes with the ``Stdev`` left unstored behind two typed gaps.

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
inconsistency of the release, on two rows this module drops for an unrelated reason.

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
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.schema import (
    AssayType,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Environment,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
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
EXPECTED_RECORDS = 15
#: What L3 ``environment_perturbed`` must observe: 0 records with no environmental edit
#: at all. An edit is a perturbation, a non-baseline temperature OR a non-baseline
#: medium, which is what covers the one record below.
EXPECTED_UNPERTURBED = 0
#: The ONE kept condition whose only edit is its medium: ``LB`` is a complex medium with
#: no carbon-source factor on top, so its ``Environment.perturbations`` is empty while
#: its ``media`` differs from the 14 M9 records' baseline. Declared so that a condition
#: silently losing its perturbation cannot pass as this one.
EXPECTED_MEDIA_ONLY_EDIT: tuple[str, ...] = ("LB",)

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
        note="the two rows per strain are not stored: NCM3722 has no deposited assembly "
        "set and no BacterialReferenceStrain member, and every Table S23 row is wild "
        "type, so L1 pair_uniqueness cannot tell two strains apart on one condition",
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


#: The rule that drops the two non-BW25113 strains. The four rows fail on the uniqueness
#: key whatever the vocabulary says, which is why one rule covers both spellings.
DROP_STRAIN_RULE = "strain_not_distinguishable_from_bw25113"


def partition(rows: Sequence[S23Row]) -> tuple[list[S23Row], DropLedger]:
    """Split the released rows into the records and the ledger of what dropped.

    Two rules, in this order: a row of either non-BW25113 strain goes under
    ``DROP_STRAIN_RULE``, and a BW25113 row whose ``schmidt2016`` condition carries a
    ``DropReason`` goes under that reason's own ``rule`` name. Both rules are the
    proteome loader's or the verifier's, never a preference of this module.
    """
    kept: list[S23Row] = []
    dropped: dict[str, list[str]] = {}
    for row in rows:
        key = f"{row.condition}/{row.strain}"
        if row.strain != STRAIN_BW25113:
            dropped.setdefault(DROP_STRAIN_RULE, []).append(key)
            continue
        drop = _condition_spec(row.condition).drop
        if drop is not None:
            dropped.setdefault(drop.rule, []).append(key)
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


def phenotype(row: S23Row) -> EnvironmentResponsePhenotype:
    """One condition's absolute growth rate, with the released ``Stdev``."""
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


def reference_row(rows: Sequence[S23Row]) -> S23Row:
    """The BW25113 Glucose row, which is this dataset's reference."""
    matches = [row for row in rows if row.condition == REFERENCE_CONDITION]
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
    """Schmidt 2016 Table S23: 15 absolute per-condition growth rates of BW25113."""

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
        return BacterialEnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialEnvironmentResponseExperimentReference

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
        """Write one record per kept Table S23 condition, 15 in all."""
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
        reference = BacterialEnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=assembly_reference(self.REFERENCE_STRAIN),
            environment_reference=environments[base.condition],
            phenotype_reference=phenotype(base),
        )
        pub = sm.publication()
        genotype = Genotype(perturbations=[])
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in enumerate(tqdm(kept, desc="schmidt2016 Table S23")):
                experiment = BacterialEnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=genotype,
                    environment=environments[row.condition],
                    phenotype=phenotype(row),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
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
            DROP_STRAIN_RULE: (
                "the released strain is MG1665 or NCM3722: NCM3722 is in neither "
                "BacterialReferenceStrain nor the deposited assembly sets, and because "
                "every Table S23 row is wild type, L1 pair_uniqueness's genotype "
                "signature is empty for all three strains, so a second strain's LB and "
                "Glucose rows duplicate BW25113's (measured: 2 duplicate triples among "
                "17 records)"
            ),
            **{
                spec.drop.rule: spec.drop.description
                for spec in sm.CONDITIONS
                if spec.drop is not None
            },
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
                        f"the reference is the BW25113 {base.condition} row "
                        f"({base.growth_rate} h-1), which is ALSO a record: for an "
                        "absolute readout a reference condition is a measured condition "
                        "too, as Caglar 2017's base condition is",
                        "the four chemostat rows release a Stdev of exactly 0, the "
                        "dilution-rate set-point rather than a measured spread, which L2 "
                        "uncertainty_sanity refuses on its own; they are dropped on "
                        "culture_not_batch before that rule is reached",
                        "Table S28 restates the two stationary-phase conditions' growth "
                        "rate as 0 where Table S23 releases -0.01; both rows are dropped "
                        "on growth_phase_not_representable, so the inconsistency is "
                        "recorded and not acted on",
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
                    "n_samples": N_SAMPLES,
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

    Every record is wild-type BW25113, so no record names a gene and the L4 gene rules
    have nothing to look at. What the records DO assert about the outside world is their
    genome pin, so the rule re-derives it from the deposited assembly report and
    requires every record to carry exactly that pair. This is the Caglar 2017 L4.
    """
    expected = assembly_reference(
        GrowthRateS23Schmidt2016Dataset.REFERENCE_STRAIN, data_root=data_root
    )
    want = (str(expected.assembly_set), str(expected.assembly_accession))
    pins = sorted(
        {
            (
                str(rec["reference"]["genome_reference"]["assembly_set"]),
                str(rec["reference"]["genome_reference"]["assembly_accession"]),
            )
            for rec in records
        }
    )
    return LevelResult(
        level=Level.L4,
        name="assembly_pin_resolves",
        passed=pins == [want],
        message=(
            f"all {len(records)} records pin {want[0]} / {want[1]}, the accession the "
            "deposited assembly report names"
            if pins == [want]
            else f"records pin {pins}; the deposited assembly report names {want}"
        ),
        details={"pins": pins, "expected": want, "n_records": len(records)},
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
                "BacterialEnvironmentResponseExperiment per kept (condition, BW25113) "
                "row, MeasurementType.growth_rate carrying the ABSOLUTE rate in h^-1; "
                "the reference is the released Glucose row's own rate"
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
