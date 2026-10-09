# torchcell/datasets/ecoli/caglar2017_doubling_time
# [[torchcell.datasets.ecoli.caglar2017_doubling_time]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/caglar2017_doubling_time
# Test file: tests/torchcell/datasets/ecoli/test_caglar2017_doubling_time.py
r"""Caglar 2017 Table S5: the 55 per-replicate doubling times, as an absolute readout.

Caglar et al. 2017 (Scientific Reports, doi:10.1038/srep45303) Supplementary Table S5
releases the doubling time of wild-type *E. coli* B REL606 in 19 environments, ONE ROW
PER BIOLOGICAL REPLICATE GROWTH CURVE, each with its own 95% confidence interval and the
r^2 of that curve's linear fit to OD600. Caption, ``si/si1.md`` line 174, verbatim:

> Supplementary Table S5: Doubling time measurements in exponential phase. Includes the
> mean, $\pm 9 5 \%$ confidence interval, and $r ^ { 2 }$ from the linear fit to OD600
> values.

Methods, Cell Growth, verbatim: "Doubling times were calculated as $\log _ { \mathrm {
e } } 2$ divided by the fit slope for each biological replicate separately. Means and
confidence intervals were calculated from three replicate growth curves for all
conditions except for gluconate and lactate, which had measurements for only two
replicates." The file agrees: 3 replicates for 17 conditions, 2 for exactly
``Gluconate.tab`` and ``Lactate.tab``, 55 rows in all.

A SEPARATE MODULE FROM ``caglar2017.py``, on purpose. ``build_manifest`` keys a built
store's staleness on the schema closure of the loader MODULE's own
``torchcell.datamodels`` imports (``provenance/schema_deps.py:loader_closure``), so
importing ``EnvironmentResponsePhenotype`` into ``caglar2017.py`` would mark the served
``rnaseq_caglar2017`` and ``proteome_caglar2017`` stores stale for a change that touches
none of their records. The pinned artifacts, the media objects, the environment helpers
and the source studies are imported FROM ``caglar2017`` instead, so there is one copy
of each. Same reason ``schmidt2016_growth_rate.py`` is separate from ``schmidt2016.py``.

WHY THIS IS AN ABSOLUTE READOUT AND NOT A RATIO. The record is
``BacterialEnvironmentResponseExperiment`` with ``MeasurementType.growth_rate``, whose
enum description is "absolute or normalized growth rate / doubling time", and the stored
number is the released doubling time in minutes. The log2-ratio alternative was MEASURED
and rejected (``experiments/036-dataset-fixes-before-kg-build/scripts/
caglar2017_doubling_time_loadability.py``, results in that experiment's ``results/``):
it stores 11 of 19 conditions, because the ``MgSO4_stress_low`` series releases no
0.8 mM Mg2+ curve of its own, so 5 conditions and 15 of the 55 rows have no in-series
reference; the three released base measurements span 0.2174 log2 (53.2538, 61.9140 and
58.3515 min), so borrowing one is not neutral; and a doubling-time ratio is a quantity
the paper never released. ``FitnessPhenotype`` is also wrong twice over: it is
``ko_growth_rate/wt_growth_rate``, a GENOTYPE ratio, and every record here is wild type.

WHAT THE SCHEMA NEEDED, AND WHY (#776). Three things, all measured before being asked
for:

1. An ASYMMETRIC interval carrier. ``UncertaintyType.ci95`` is a single half-width, and
   the released interval is asymmetric in **55 of 55 rows** (upper-to-lower half-width
   ratio min -26.3920, median 1.3187, max 10.1066; median absolute asymmetry 2.5783
   min). ``Glycerol.tab`` replicate 1 releases ``95p = -1027.769034`` against a value of
   80.95212424 -- what the image of a slope interval straddling zero becomes under
   ``DT = log_e 2 / slope`` -- and feeding that side to ``ci95`` derives
   ``environment_response_se = -565.6845``, a negative standard error. So the record
   stores ``environment_response_lower`` / ``_upper`` / ``confidence_level``, the shape
   ``FluxPhenotype`` already uses, and ``environment_response_se`` carries a typed
   ``not_reported_by_primary`` gap: an asymmetric interval does not reduce to an SE.
2. Relief from L3 ``reference_zero``. The reference condition's doubling time is 53.68
   min; a stored 0 would assert instant growth. ``reference_centered=False`` switches the
   rule to the absolute branch (the reference states its own finite value on the record's
   own scale) and that branch REFUSES any record whose ``measurement_type`` is not in
   ``ABSOLUTE_MEASUREMENT_TYPES``, so the relief cannot leak to a log2-ratio dataset.
3. A per-replicate identifier. 55 per-replicate rows collapse to 16 unique (study,
   strain, condition) triples without one, so L1 ``pair_uniqueness`` would demand an
   aggregate the paper never released. ``replicate_id`` is Table S5's own ``replicate``
   column and joins the L1 key.

THE THREE DECLARED COUNTS, each an oracle the verifier checks rather than a waiver:
``EXPECTED_RECORDS`` 55; ``EXPECTED_UNPERTURBED`` 9 (the base condition was run in three
separate experiments -- ``Glucose.tab``, ``MgSO4_000.800_mM.tab``, ``NaCl_005_mM.tab`` --
and for an absolute readout each is a measured condition, not a defect);
``EXPECTED_NON_BRACKETING`` 1 (the ``Glycerol.tab`` replicate-1 row above).

``screen_id`` IS Table S1's ``experiment`` column. Four of the 19 conditions collide on
the environment alone: the three base-condition rows carry no environmental edit at all,
and ``MgSO4_000.080_mM`` sits at the same 0.080 mM as ``MgSO4-2_000.080_mM``. The run is
what distinguishes them and Table S1 names it, so the screen id is the source's own
label, never a synthesized one. The 1:1 join of Table S5's 19 condition names onto Table
S1's ``(experiment, carbonSource, Mg_mM, Na_mM)`` keys is asserted at build time
(``CONDITIONS``), so a re-released table cannot silently re-map a condition.

THE REFERENCE. "The reference conditions always had glucose as carbon source and base
$\mathrm{Na^+}$ and $\mathbf{Mg}^{2+}$ concentrations" (``REFERENCE_CONDITIONS``), and
Table S5 is entirely exponential phase, so there is ONE reference for the dataset: wild
type in the base condition. Its stored value is Table S1's RELEASED condition-level fit
for ``glucose_time_course`` -- 53.679955 min, 95m 49.100298, 95p 59.201792 -- because a
reference needs one number per context and Table S5 releases only per-replicate ones;
computing a mean of the three glucose replicates would be a number the paper never
released (measured: Table S1's value equals none of the arithmetic, geometric or
harmonic means of them). Table S1 is read for THAT ONE ROW only and is never loaded as a
record, which is what keeps the two fits of the same OD600 experiment from being stored
twice. Which of the three base-condition rows is the reference changes NOT ONE stored
number, because the readout is absolute; the 0.2174 log2 spread between them is recorded
in ``preprocess/base_condition_spread.json`` as a dataset-level limitation.

WHAT IS NOT STORED. Table S5's ``r.squared`` column, 55 values, plus the reference row's
own ``rSquared``: there is no fit-quality axis anywhere in the schema (``FluxPhenotype``
carries an interval and no r^2), and deriving one would be inventing a field rather than
sourcing it. The column is recorded verbatim in ``preprocess/not_stored.json`` with its
count, the same discipline Borchert 2023 uses for its significance triple. Tables S6 to
S14 are untouched by this module.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process, verify_raw_files
from torchcell.datamodels.media import DAVIS_MINIMAL, DM500
from torchcell.datamodels.schema import (
    AssayType,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    SampleUnit,
    Temperature,
)
from torchcell.datasets.bacteria_common import assembly_reference
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.ecoli.caglar2017 import (
    CITATION_KEY,
    CULTURE_CONDITIONS,
    RAW_DIR_REL,
    REFERENCE_CONDITIONS,
    SI_TABLES,
    SOURCE_STUDIES,
    SourceStudyKey,
    _link_mirror_files,
    _paper,
    _raw_pins,
    _si,
    _table_pin,
    carbon_source_perturbation,
    magnesium_perturbation,
    require_pinnable_strain,
    sodium_perturbation,
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
)

log = logging.getLogger(__name__)

DATASET_ROOT_REL = "data/torchcell/doubling_time_caglar2017"

#: Table S5's released columns, in file order.
COL_CONDITION = "name"
COL_REPLICATE = "replicate"
COL_MINUTES = "doubling.time.minutes"
COL_LOWER = "doubling.time.minutes.95m"
COL_UPPER = "doubling.time.minutes.95p"
COL_R_SQUARED = "r.squared"

#: Table S1's doubling-time columns (the reference row's). The upper bound's header uses
#: an underscore where the lower bound's uses a dot; that is the release's own spelling.
S1_COL_EXPERIMENT = "experiment"
S1_COL_CARBON = "carbonSource"
S1_COL_MG = "Mg_mM"
S1_COL_NA = "Na_mM"
S1_COL_MINUTES = "doublingTimeMinutes"
S1_COL_LOWER = "doublingTimeMinutes.95m"
S1_COL_UPPER = "doublingTimeMinutes_95p"
S1_COL_R_SQUARED = "rSquared"

#: Davis Minimal's base magnesium and sodium levels, Table S1's own ``baseMg`` /
#: ``baseNa`` numbers (identical to ``caglar2017.BASE_MG_SHEET_MM`` / ``BASE_NA_MM``;
#: restated here because this module reads Table S5's condition names, not Table S1's
#: level labels).
BASE_MG_MM = 0.8
BASE_NA_MM = 5.0
BASE_CARBON = "glucose"

#: The confidence level of both tables' intervals: the caption says 95%.
CONFIDENCE_LEVEL = 0.95

#: The released units of the readout.
UNITS = "doubling time in minutes (log_e 2 / slope of the linear fit to OD600)"

#: Table S5 condition -> (Table S1 ``experiment`` prefix, carbon source, Mg2+ mM, Na+ mM).
#: The ``-2`` suffix of the second magnesium series is Table S1's ``MgSO4_stress_low``,
#: the unsuffixed one ``MgSO4_stress_high``. The 1:1 join is asserted by
#: :func:`read_reference_row` for the base condition and by :func:`read_table_s5` for the
#: full condition set, so a re-released table cannot re-map a condition silently.
CONDITIONS: dict[str, tuple[str, str, float, float]] = {
    "Gluconate.tab": ("gluconate_growth", "gluconate", BASE_MG_MM, BASE_NA_MM),
    "Glucose.tab": ("glucose_time_course", "glucose", BASE_MG_MM, BASE_NA_MM),
    "Glycerol.tab": ("glycerol_time_course", "glycerol", BASE_MG_MM, BASE_NA_MM),
    "Lactate.tab": ("lactate_growth", "lactate", BASE_MG_MM, BASE_NA_MM),
    **{
        f"MgSO4_{mg}_mM.tab": ("MgSO4_stress_high", "glucose", float(mg), BASE_NA_MM)
        for mg in ("000.080", "000.800", "008.000", "050.000", "200.000", "400.000")
    },
    **{
        f"MgSO4-2_{mg}_mM.tab": ("MgSO4_stress_low", "glucose", float(mg), BASE_NA_MM)
        for mg in ("000.005", "000.010", "000.020", "000.040", "000.080")
    },
    **{
        f"NaCl_{na}_mM.tab": ("NaCl_stress", "glucose", BASE_MG_MM, float(na))
        for na in ("005", "100", "200", "300")
    },
}

#: The Table S5 condition that IS the paper's reference condition, and the Table S1
#: experiment its released condition-level fit sits under.
REFERENCE_CONDITION = "Glucose.tab"

#: The study every Table S5 record is attributed to. Caglar 2017 attributes 54 of its
#: 257 mRNA and protein records to Houser 2015 (#771, ``caglar2017.SOURCE_STUDIES`` +
#: ``attribute_sample``), and that split's evidence is scoped to SAMPLES: the note on
#: ``caglar2017.HOUSER2015_DEFERRAL`` names "the 27 samples that appear as columns of
#: BOTH Table S2 (mRNA) and Table S3 (protein)", and ``HOUSER2015_DEPOSITS`` splits two
#: molecular accessions. Table S5 releases no sample columns: its grain is a
#: growth-curve fit per biological replicate, and all 55 values plus both released
#: limits plus the Table S1 reference row are fits this paper released. So the split
#: rule does not reach this table and every record names Caglar 2017. Measured overlap,
#: recorded rather than acted on: 3 of 55 rows (``Glucose.tab``) and the reference row
#: join onto Table S1's ``glucose_time_course``, the experiment Houser 2015 presented;
#: whether Houser 2015 released a doubling time for it is unknown here, because that
#: paper is unmirrored and unread (``caglar2017.HOUSER2015_IS_MIRRORED``).
SOURCE_STUDY: SourceStudyKey = "caglar2017"

#: The three Table S5 conditions that ARE the base condition (glucose, base Mg2+, base
#: Na+), run in three separate experiments. They carry no environmental edit, which for
#: an absolute readout is a measured condition rather than a defect.
BASE_CONDITIONS: tuple[str, ...] = (
    "Glucose.tab",
    "MgSO4_000.800_mM.tab",
    "NaCl_005_mM.tab",
)

#: Build oracles, every one of them measured on the pinned bytes before being written
#: here (see the module docstring's "THE THREE DECLARED COUNTS").
EXPECTED_RECORDS = 55
EXPECTED_CONDITIONS = 19
EXPECTED_UNPERTURBED = 9
EXPECTED_NON_BRACKETING = 1
#: Per-condition replicate counts the Methods state: 2 for exactly these two.
TWO_REPLICATE_CONDITIONS: tuple[str, ...] = ("Gluconate.tab", "Lactate.tab")

# Every quote below is a literal substring of the artifact its provenance names, and
# both helpers come from ``caglar2017`` so the pins are the module's own, never a second
# copy: ``_si`` binds to the OCR of the SI PDF (``si/si1.md``), ``_paper`` to the OCR of
# the article (``paper.md``).
TABLE_S5_CAPTION = _si(
    "per-replicate doubling time, 95% CI and r^2",
    "Supplementary Table S5: Doubling time measurements in exponential phase. Includes "
    "the mean, $\\pm 9 5 \\%$ confidence interval, and $r ^ { 2 }$ from the linear fit "
    "to OD600 values.",
    note="the caption's 'mean' is the fit's estimate for ONE replicate curve: the file "
    "is one row per replicate, and the condition mean is what Fig. 2 draws rather than "
    "a released number",
)
DOUBLING_TIME_FIT = _paper(
    "log_e 2 / fit slope, per biological replicate",
    "Doubling times were calculated as $\\log _ { \\mathrm { e } } 2$ divided by the "
    "fit slope for each biological replicate separately. Means and confidence intervals "
    "were calculated from three replicate growth curves for all conditions except for "
    "gluconate and lactate, which had measurements for only two replicates.",
    page="Methods, Cell Growth",
    note="the grain of the release (per replicate) and the replicate counts the file is "
    "checked against (3, except 2 for gluconate and lactate)",
)
BASE_CONDITION_LINE = _paper(
    ("glucose", BASE_NA_MM, BASE_MG_MM),
    "The red points and dashed orange lines represent the doubling time at the base "
    "condition (glucose, $5 \\mathrm { m M N a ^ { + } }$ , $0 . 8 \\mathrm { m M } "
    "\\mathrm { M } \\mathrm { g } ^ { 2 + } ,$ ). Doubling times were measured in "
    "triplicates and error bars represents $9 5 \\%$ confidence intervals",
    page="Fig. 2 legend",
    note="Fig. 2's own definition of the base condition, which is the condition the "
    "reference record carries. The legend says 'triplicates' of the whole figure; the "
    "Methods and the file both say gluconate and lactate released two curves, which is "
    "what the build checks",
)

#: Every sourced value this module writes to ``preprocess/sourced_values.json``.
SOURCED_VALUES: dict[str, SourcedValue] = {
    "table_s5_caption": TABLE_S5_CAPTION,
    "doubling_time_fit": DOUBLING_TIME_FIT,
    "base_condition": BASE_CONDITION_LINE,
    "culture_temperature": CULTURE_CONDITIONS,
    "reference_condition": REFERENCE_CONDITIONS,
}

SE_GAP = ProvenanceGap(
    field="environment_response_se",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="the release carries a two-sided 95% confidence interval of the OD600 slope "
    "fit, asymmetric in 55 of 55 rows, which does not reduce to a standard error; both "
    "limits are stored in environment_response_lower / _upper at confidence_level 0.95",
)
UNCERTAINTY_GAP = ProvenanceGap(
    field="environment_response_uncertainty",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="no single dispersion number is released; the interval is the uncertainty and "
    "is stored as an interval",
)
REFERENCE_N_SAMPLES_GAP = ProvenanceGap(
    field="n_samples",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="Table S1's condition-level doubling time is a second fit of the same OD600 "
    "experiment and the paper never says what it is computed over; measured, it equals "
    "none of the arithmetic, geometric or harmonic means of Table S5's three glucose "
    "replicates, so asserting n_samples = 3 for it would be a guess",
)


class FitRow(BaseModel):
    """One released Table S5 growth-curve fit: a value, its 95% limits and its r^2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: str
    replicate: int
    minutes: float
    lower_95: float
    upper_95: float
    r_squared: float

    @property
    def brackets_value(self) -> bool:
        """Whether the released limits actually bracket the released value."""
        return self.lower_95 <= self.minutes <= self.upper_95


class ReferenceRow(BaseModel):
    """Table S1's condition-level fit for the base condition, the reference's value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment: str
    minutes: float
    lower_95: float
    upper_95: float
    r_squared: float
    sample_rows: int


def read_table_s5(path: str | Path) -> list[FitRow]:
    """Table S5's per-replicate fits, typed, with the release's own shape asserted.

    Asserts the row count, the condition set (every name must be in ``CONDITIONS`` and
    every ``CONDITIONS`` name must appear) and the per-condition replicate counts the
    Methods state. A released file that drifts raises rather than building fewer records.
    """
    frame = pd.read_csv(path, float_precision="round_trip")
    rows = [
        FitRow(
            condition=str(cells[COL_CONDITION]),
            replicate=int(cells[COL_REPLICATE]),
            minutes=float(cells[COL_MINUTES]),
            lower_95=float(cells[COL_LOWER]),
            upper_95=float(cells[COL_UPPER]),
            r_squared=float(cells[COL_R_SQUARED]),
        )
        for cells in frame.to_dict(orient="records")
    ]
    if len(rows) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Table S5 has {len(rows)} rows, expected {EXPECTED_RECORDS}"
        )
    names = {row.condition for row in rows}
    if names != set(CONDITIONS):
        raise RuntimeError(
            f"Table S5 conditions differ from CONDITIONS: "
            f"{sorted(names ^ set(CONDITIONS))}"
        )
    counts = {
        name: sum(1 for row in rows if row.condition == name) for name in sorted(names)
    }
    for name, count in counts.items():
        expected = 2 if name in TWO_REPLICATE_CONDITIONS else 3
        if count != expected:
            raise RuntimeError(
                f"Table S5 {name}: {count} replicates, the Methods state {expected}"
            )
    if len({(row.condition, row.replicate) for row in rows}) != len(rows):
        raise RuntimeError("Table S5 repeats a (condition, replicate) key")
    return rows


def read_reference_row(path: str | Path) -> ReferenceRow:
    """Table S1's condition-level fit for the base condition, joined 1:1 by condition.

    Selects the Table S1 rows of ``REFERENCE_CONDITION``'s ``(experiment, carbonSource,
    Mg_mM, Na_mM)`` key and requires them to carry exactly ONE distinct doubling-time
    tuple. That is the measurement behind "Table S1 is one condition-level fit repeated
    per sample"; more than one tuple means the release changed and the join is no longer
    1:1.
    """
    experiment, carbon, mg_mm, na_mm = CONDITIONS[REFERENCE_CONDITION]
    frame = pd.read_csv(path, float_precision="round_trip").dropna(
        subset=[S1_COL_MINUTES]
    )
    members = frame[
        frame[S1_COL_EXPERIMENT].astype(str).str.startswith(experiment)
        & (frame[S1_COL_CARBON] == carbon)
        & (frame[S1_COL_MG] == mg_mm)
        & (frame[S1_COL_NA] == na_mm)
    ]
    tuples = {
        (
            float(row[S1_COL_MINUTES]),
            float(row[S1_COL_LOWER]),
            float(row[S1_COL_UPPER]),
            float(row[S1_COL_R_SQUARED]),
        )
        for row in members.to_dict(orient="records")
    }
    if len(tuples) != 1:
        raise RuntimeError(
            f"Table S1's {REFERENCE_CONDITION} key selects {len(tuples)} distinct "
            "doubling-time tuples, not the one condition-level fit"
        )
    minutes, lower, upper, r_squared = next(iter(tuples))
    if not lower <= minutes <= upper:
        raise RuntimeError(
            f"Table S1's reference interval does not bracket its value: "
            f"{lower} / {minutes} / {upper}"
        )
    return ReferenceRow(
        experiment=experiment,
        minutes=minutes,
        lower_95=lower,
        upper_95=upper,
        r_squared=r_squared,
        sample_rows=len(members),
    )


def build_environment(condition: str) -> Environment:
    """The environment of one Table S5 condition, from ``caglar2017``'s own helpers.

    DM500 for a glucose condition; Davis Minimal plus a ``carbon_source`` factor at
    0.5 g/L when the carbon source is swapped; a magnesium-sulfate perturbation at any
    level other than the base 0.8 mM; NaCl added for any Na+ level above the ~5 mM base.
    37 C, aerobic (``CULTURE_CONDITIONS``). The base condition therefore carries NO
    perturbation, which is what ``EXPECTED_UNPERTURBED`` declares.
    """
    _, carbon, mg_mm, na_mm = CONDITIONS[condition]
    perturbations: list[EnvironmentPerturbationType] = []
    media = DM500
    if carbon != BASE_CARBON:
        media = DAVIS_MINIMAL
        perturbations.append(carbon_source_perturbation(carbon))
    if mg_mm != BASE_MG_MM:
        perturbations.append(magnesium_perturbation(mg_mm))
    if na_mm != BASE_NA_MM:
        perturbations.append(sodium_perturbation(na_mm))
    return Environment(
        media=media,
        temperature=Temperature(value=float(CULTURE_CONDITIONS.value)),
        perturbations=perturbations,
        aerobicity="aerobic",
    )


def phenotype(row: FitRow) -> EnvironmentResponsePhenotype:
    """One replicate curve's absolute doubling time, with both released 95% limits."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.growth_rate,
        assay_type=AssayType.liquid_od_growth,
        environment_response=row.minutes,
        environment_response_lower=row.lower_95,
        environment_response_upper=row.upper_95,
        confidence_level=CONFIDENCE_LEVEL,
        n_samples=1,
        sample_unit=SampleUnit.biological_replicate,
        replicate_id=str(row.replicate),
        screen_id=CONDITIONS[row.condition][0],
        units=UNITS,
        provenance_gaps=[SE_GAP, UNCERTAINTY_GAP],
    )


def reference_phenotype(row: ReferenceRow) -> EnvironmentResponsePhenotype:
    """The base condition's released condition-level doubling time and its limits."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.growth_rate,
        assay_type=AssayType.liquid_od_growth,
        environment_response=row.minutes,
        environment_response_lower=row.lower_95,
        environment_response_upper=row.upper_95,
        confidence_level=CONFIDENCE_LEVEL,
        sample_unit=SampleUnit.biological_replicate,
        screen_id=row.experiment,
        units=UNITS,
        provenance_gaps=[SE_GAP, UNCERTAINTY_GAP, REFERENCE_N_SAMPLES_GAP],
    )


class BuildAccounting(BaseModel):
    """What the build measured on the released bytes, written beside the store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_rows: int
    kept_records: int
    conditions: int
    replicate_counts: dict[str, int]
    unperturbed_records: int
    unperturbed_conditions: list[str]
    non_bracketing_records: int
    non_bracketing_rows: list[dict[str, Any]]
    reference_condition: str
    reference_minutes: float
    screen_ids: dict[str, int]


def _accounting(
    rows: Sequence[FitRow], reference: ReferenceRow, unperturbed: Sequence[FitRow]
) -> BuildAccounting:
    """The build's own census of the 55 rows, so every declared count is auditable."""
    return BuildAccounting(
        source_rows=len(rows),
        kept_records=len(rows),
        conditions=len({row.condition for row in rows}),
        replicate_counts={
            name: sum(1 for row in rows if row.condition == name)
            for name in sorted({row.condition for row in rows})
        },
        unperturbed_records=len(unperturbed),
        unperturbed_conditions=sorted({row.condition for row in unperturbed}),
        non_bracketing_records=sum(1 for row in rows if not row.brackets_value),
        non_bracketing_rows=[
            {
                "condition": row.condition,
                "replicate": row.replicate,
                "minutes": row.minutes,
                "lower_95": row.lower_95,
                "upper_95": row.upper_95,
            }
            for row in rows
            if not row.brackets_value
        ],
        reference_condition=REFERENCE_CONDITION,
        reference_minutes=reference.minutes,
        screen_ids={
            screen: sum(1 for row in rows if CONDITIONS[row.condition][0] == screen)
            for screen in sorted({CONDITIONS[row.condition][0] for row in rows})
        },
    )


@register_dataset
class DoublingTimeCaglar2017Dataset(ExperimentDataset):
    """Caglar 2017 Table S5: 55 absolute doubling times, one per replicate curve."""

    REFERENCE_STRAIN: ClassVar[Literal["REL606"]] = "REL606"
    #: Every record is wild-type REL606, so the gene set is legitimately empty.
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

    @staticmethod
    def pins() -> list[tuple[str, str]]:
        """The mirror files this dataset reads: Table S5, and Table S1's reference row."""
        return [_table_pin("S5"), _table_pin("S1")]

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
        """Tables S5 and S1, required before processing."""
        return list(_raw_pins(self.pins()))

    def download(self) -> None:
        """Link Tables S5 and S1 from the raw mirror after verifying their pins."""
        _link_mirror_files(self.raw_dir, self.pins())

    @post_process
    def process(self) -> None:
        """Write one record per released replicate curve, 55 in all."""
        require_pinnable_strain()
        verify_raw_files(self.raw_dir, _raw_pins(self.pins()))
        rows = read_table_s5(osp.join(self.raw_dir, SI_TABLES["S5"][0]))
        reference_row = read_reference_row(osp.join(self.raw_dir, SI_TABLES["S1"][0]))

        environments = {name: build_environment(name) for name in CONDITIONS}
        unperturbed = [
            row for row in rows if not environments[row.condition].perturbations
        ]
        if len(unperturbed) != EXPECTED_UNPERTURBED:
            raise RuntimeError(
                f"{len(unperturbed)} records carry no environmental edit, the module "
                f"declares {EXPECTED_UNPERTURBED}"
            )
        if {row.condition for row in unperturbed} != set(BASE_CONDITIONS):
            raise RuntimeError(
                "the unperturbed records are not exactly the base conditions: "
                f"{sorted({row.condition for row in unperturbed})}"
            )
        non_bracketing = [row for row in rows if not row.brackets_value]
        if len(non_bracketing) != EXPECTED_NON_BRACKETING:
            raise RuntimeError(
                f"{len(non_bracketing)} released intervals do not bracket their value, "
                f"the module declares {EXPECTED_NON_BRACKETING}"
            )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        reference = BacterialEnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=assembly_reference(self.REFERENCE_STRAIN),
            environment_reference=environments[REFERENCE_CONDITION],
            phenotype_reference=reference_phenotype(reference_row),
        )
        pub = SOURCE_STUDIES[SOURCE_STUDY].publication
        genotype = Genotype(perturbations=[])
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in enumerate(tqdm(rows, desc="caglar2017 doubling time")):
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

        self._write_ledgers(rows, reference_row, unperturbed)
        log.info(
            "Caglar 2017 Table S5: %d records over %d conditions; doubling time "
            "%.6f to %.6f min; reference %s = %.6f min",
            len(rows),
            len({row.condition for row in rows}),
            min(row.minutes for row in rows),
            max(row.minutes for row in rows),
            REFERENCE_CONDITION,
            reference_row.minutes,
        )

    def _write_ledgers(
        self,
        rows: Sequence[FitRow],
        reference: ReferenceRow,
        unperturbed: Sequence[FitRow],
    ) -> None:
        """The retention ledger, the sourcing table, the unstored column and the spread."""
        out = Path(self.preprocess_dir)
        accounting = _accounting(rows, reference, unperturbed)
        (out / "build_accounting.json").write_text(accounting.model_dump_json(indent=2))
        (out / "dropped_records.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "source_rows": len(rows),
                    "kept_records": len(rows),
                    "dropped_records": 0,
                    "rules": [],
                    "notes": [
                        "every released Table S5 row is a record: the release is one "
                        "row per biological replicate growth curve, and replicate_id "
                        "keeps the replicates of one condition L1-distinct instead of "
                        "forcing a condition mean the paper never released",
                        f"MgSO4-2_000.020_mM replicate 3 and MgSO4-2_000.040_mM "
                        f"replicate 3 repeat each other in all four numeric columns, so "
                        f"{len(rows)} rows hold 54 distinct fits; they are two "
                        "conditions and stay two records",
                        "Table S1's condition-level doubling time is NOT loaded as a "
                        "record: its 165 stated rows hold 19 distinct tuples that join "
                        "1:1 onto these 19 conditions, so loading it would store the "
                        "same OD600 experiment twice at a coarser grain. One row of it, "
                        "the base condition's, is the reference's value",
                        "Tables S6 to S14 are not touched by this module",
                    ],
                },
                indent=2,
            )
        )
        (out / "not_stored.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "issue": 776,
                    "items": [
                        {
                            "released": f"Table S5 '{COL_R_SQUARED}'",
                            "n_values": len(rows),
                            "why_not_stored": "no phenotype class in the schema has a "
                            "fit-quality axis (FluxPhenotype carries a confidence "
                            "interval and no r^2), and deriving one would be inventing "
                            "a field rather than sourcing it",
                            "values": {
                                f"{row.condition}/{row.replicate}": row.r_squared
                                for row in rows
                            },
                        },
                        {
                            "released": f"Table S1 '{S1_COL_R_SQUARED}' of the "
                            f"reference condition",
                            "n_values": 1,
                            "why_not_stored": "same absent axis",
                            "values": {reference.experiment: reference.r_squared},
                        },
                    ],
                },
                indent=2,
            )
        )
        (out / "base_condition_spread.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "note": "the base condition (glucose, 0.8 mM Mg2+, 5 mM Na+) was "
                    "run in three separate experiments, so three Table S5 conditions "
                    "measure it. The reference record carries one of them; because the "
                    "readout is ABSOLUTE, the choice changes no stored number, and the "
                    "spread below is what it would change if a ratio were stored",
                    "base_conditions": list(BASE_CONDITIONS),
                    "mean_minutes_per_base_condition": {
                        name: sum(row.minutes for row in rows if row.condition == name)
                        / sum(1 for row in rows if row.condition == name)
                        for name in BASE_CONDITIONS
                    },
                    "reference_condition": REFERENCE_CONDITION,
                    "reference_minutes_table_s1": reference.minutes,
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
                    "condition": row.condition,
                    "replicate": row.replicate,
                    "screen_id": CONDITIONS[row.condition][0],
                    "minutes": row.minutes,
                    "lower_95": row.lower_95,
                    "upper_95": row.upper_95,
                    "brackets_value": row.brackets_value,
                    "r_squared": row.r_squared,
                }
                for row in rows
            ]
        ).to_csv(out / "doubling_times.csv", index=False)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "DoublingTimeCaglar2017Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# L0-L4 verification
# --------------------------------------------------------------------------- #
def l4_assembly_pin(
    records: Sequence[Mapping[str, Any]], data_root: str | None = None
) -> LevelResult:
    """L4: the record's pinned assembly IS the one the genomes tier deposited for REL606.

    This is the dataset's only cross-source rule, and it is the one that fits: every
    record is wild-type REL606, so no record names a gene and the L4 gene-containment
    rules have nothing to look at. What the records DO assert about the outside world is
    their genome pin, so the rule re-derives it from the deposited assembly report
    (``assembly_reference``, which reads the genomes tier, not this loader's constants)
    and requires every record to carry exactly that pair.
    """
    expected = assembly_reference(
        DoublingTimeCaglar2017Dataset.REFERENCE_STRAIN, data_root=data_root
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

    The three ABSOLUTE declarations are passed here and nowhere else: this is the only
    dataset that holds an absolute growth readout, so it is the only one that asks for
    ``reference_centered=False``, and the absolute branch refuses the request for any
    record whose ``measurement_type`` is not absolute. ``sgd_genes`` is not supplied:
    every record is wild type, so the shared L4 GENE rules have nothing to look at and
    are left off rather than run vacuously against a universe no record names;
    :func:`l4_assembly_pin` is this dataset's L4 instead.
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
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/data/{SI_TABLES['S5'][0]}",
            citation_key=CITATION_KEY,
            sha256=SI_TABLES["S5"][1],
            method=(
                "Supplementary Table S5: one BacterialEnvironmentResponseExperiment "
                "per released biological-replicate growth curve, "
                "MeasurementType.growth_rate carrying the doubling time in minutes "
                "with both released 95% limits; the reference is the base condition's "
                "condition-level fit from Table S1"
            ),
            page="si6.csv (PMC object srep45303-s6.csv)",
            retrieved="2026-10-09",
        ),
        expected_count=expected_count,
        reference_centered=False,
        expected_unperturbed=EXPECTED_UNPERTURBED,
        expected_non_bracketing=EXPECTED_NON_BRACKETING,
    )
    report.add(l4_assembly_pin(records, data_root))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def released_summary(rows: Sequence[FitRow]) -> Mapping[str, Any]:
    """The three declared counts recomputed from ``rows``, for a test or a CLI print."""
    return {
        "records": len(rows),
        "conditions": len({row.condition for row in rows}),
        "non_bracketing": sum(1 for row in rows if not row.brackets_value),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: ``build`` the dev-tree LMDB, or ``verify`` an already built one."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.caglar2017_doubling_time"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, DATASET_ROOT_REL)
    if args.command == "build":
        dataset = DoublingTimeCaglar2017Dataset(root=root)
        print(f"len = {len(dataset)}")
        print(Path(root, "preprocess", "build_accounting.json").read_text())
        return 0
    report = verify_build(root, data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
