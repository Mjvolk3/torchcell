# experiments/036-dataset-fixes-before-kg-build/scripts/caglar2017_doubling_time_loadability.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.caglar2017_doubling_time_loadability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/caglar2017_doubling_time_loadability
"""Settle whether Caglar 2017's Table S5 doubling times are loadable, by measurement.

Two of our own audit notes disagree. ``notes/plan.bacteria-si-phenotype-audit-pputida.md``
calls the quantity loadable now as ``BacterialEnvironmentResponseExperiment`` /
``EnvironmentResponsePhenotype`` with ``measurement_type = growth_rate``,
``assay_type = liquid_od_growth`` and ``environment_response_uncertainty_type = ci95``,
counting 55 records. ``notes/plan.bacteria-si-phenotype-audit-ecoli.md`` rank 15 calls the
same quantity blocked by its gap 1 (absolute growth readout) and gap 9 (asymmetric
interval), counting 19. This script measures everything the disagreement turns on:

1. The shape of ``si/si6.csv`` (Table S5): row count, condition count, per-condition
   replicate counts against the loader's sourced ``DOUBLING_TIME_REPLICATES`` quote, and
   any row whose four numeric columns repeat another row's.
2. Whether the 95% confidence interval is symmetric, per row, in both released tables.
3. Whether ``si/si2.csv`` (Table S1) ``doublingTimeMinutes`` is Table S5's condition mean
   repeated per sample: the 1:1 join of the 19 Table S5 conditions onto Table S1's
   ``(experiment, carbonSource, Mg_mM, Na_mM)`` keys, and the difference between Table
   S1's value and the arithmetic, geometric and harmonic means of Table S5's replicates.
4. Four candidate record forms, each built as real pydantic records with the Caglar
   loader's own environment helpers and each scored by the environment-response family's
   own L1 ``pair_uniqueness``, L3 ``environment_perturbed`` and L3 ``reference_zero``
   rules: absolute per replicate, absolute per condition, log2 ratio over all 19
   conditions, and log2 ratio restricted to the conditions whose own experiment released a
   base-condition row.

Both SI files are sha256-verified against the library mirror's ``manifest.json`` before
anything is read.

Writes ``results/caglar2017_doubling_time_loadability.json`` and
``results/caglar2017_doubling_time_conditions.csv``.

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/caglar2017_doubling_time_loadability.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import os.path as osp
import statistics
from collections import Counter
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.datamodels.media import DAVIS_MINIMAL, DM500
from torchcell.datamodels.schema import (
    AssayType,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentResponsePhenotype,
    Genotype,
    MeasurementType,
    SampleUnit,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import assembly_reference
from torchcell.datasets.ecoli.caglar2017 import (
    CITATION_KEY,
    CULTURE_CONDITIONS,
    carbon_source_perturbation,
    magnesium_perturbation,
    publication,
    sodium_perturbation,
)
from torchcell.verification.environment_response import (
    _l1_pair_uniqueness,
    _l3_environment_perturbed,
    _l3_reference_zero,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

RESULTS_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
BASE_MG_MM = 0.8
BASE_NA_MM = 5.0
MEAN_KINDS = ("arithmetic", "geometric", "harmonic")

#: Table S5 condition -> (Table S1 ``experiment``, carbon source, Mg2+ mM, Na+ mM).
#: The ``-2`` suffix of the second magnesium series is Table S1's ``MgSO4_stress_low``;
#: the unsuffixed series is ``MgSO4_stress_high``. Asserted by the 1:1 join below.
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

#: The Table S5 row that IS each series' base condition, as Fig. 2's three panels draw
#: the reference line: "The red points and dashed orange lines represent the doubling
#: time at the base condition (glucose, 5 mM Na+, 0.8 mM Mg2+)". One row per experiment,
#: so ``MgSO4_stress_low`` has none -- that series released no 0.8 mM Mg2+ curve.
BASE_ROWS: dict[str, str] = {
    "gluconate_growth": "Glucose.tab",
    "glycerol_time_course": "Glucose.tab",
    "lactate_growth": "Glucose.tab",
    "glucose_time_course": "Glucose.tab",
    "MgSO4_stress_high": "MgSO4_000.800_mM.tab",
    "NaCl_stress": "NaCl_005_mM.tab",
}


class FitRow(BaseModel):
    """One released growth-curve fit: a value and its 95% bounds, in minutes."""

    model_config = ConfigDict(extra="forbid")

    condition: str
    replicate: int
    minutes: float
    lower_95: float
    upper_95: float
    r_squared: float

    def numeric_key(self) -> tuple[float, float, float, float]:
        """The four released numbers, the key an exactly repeated fit collides on."""
        return (self.minutes, self.lower_95, self.upper_95, self.r_squared)


class ConditionMeans(BaseModel):
    """A Table S5 condition's replicate count and the three means of its values."""

    model_config = ConfigDict(extra="forbid")

    condition: str
    n_replicates: int
    arithmetic: float
    geometric: float
    harmonic: float

    def mean(self, kind: str) -> float:
        """The named mean, so the three can be compared in one loop."""
        return float(getattr(self, kind))


class IntervalCensus(BaseModel):
    """How many of a table's released 95% intervals are symmetric about the value."""

    model_config = ConfigDict(extra="forbid")

    n_rows: int
    n_symmetric: int
    n_upper_wider: int
    n_lower_wider: int
    n_upper_bound_not_above_value: int
    half_width_ratio_min: float
    half_width_ratio_median: float
    half_width_ratio_max: float
    abs_asymmetry_minutes_median: float
    abs_asymmetry_minutes_max: float


class JoinedCondition(BaseModel):
    """One Table S5 condition beside the Table S1 value of the same condition."""

    model_config = ConfigDict(extra="forbid")

    table_s5_condition: str
    table_s1_experiment: str
    carbon_source: str
    mg_mm: float
    na_mm: float
    n_replicates: int
    table_s1_sample_rows: int
    table_s1_doubling_time: float
    table_s1_95m: float
    table_s1_95p: float
    table_s5_arithmetic_mean: float
    table_s5_geometric_mean: float
    table_s5_harmonic_mean: float
    s1_minus_s5_arithmetic: float
    s1_minus_s5_geometric: float
    s1_minus_s5_harmonic: float

    def difference(self, kind: str) -> float:
        """Table S1's value minus Table S5's named mean of the same condition."""
        return float(getattr(self, f"s1_minus_s5_{kind}"))


class RuleOutcome(BaseModel):
    """One verifier rule's verdict on one candidate record form."""

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    message: str


class FormOutcome(BaseModel):
    """A candidate record form and the three rules that decide whether it can be built."""

    model_config = ConfigDict(extra="forbid")

    form: str
    description: str
    n_records: int
    rules: list[RuleOutcome]


def sha256_of(path: str) -> str:
    """The file's sha256, read whole (both tables are under 40 KB)."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def verified_si_path(library_key_dir: str, relative: str) -> str:
    """The mirrored SI file's path, with its bytes checked against ``manifest.json``."""
    with open(osp.join(library_key_dir, "manifest.json")) as handle:
        manifest = json.load(handle)
    entries = [f for f in manifest["files"] if f["path"] == relative]
    if len(entries) != 1:
        raise RuntimeError(f"{relative} appears {len(entries)} times in the manifest")
    path = osp.join(library_key_dir, relative)
    digest = sha256_of(path)
    expected = str(entries[0]["sha256"])
    if digest != expected:
        raise RuntimeError(f"{relative} sha256 {digest} != manifest {expected}")
    return path


def read_table_s5(path: str) -> list[FitRow]:
    """Table S5's per-replicate growth-curve fits, one typed row each."""
    frame = pd.read_csv(path)
    return [
        FitRow(
            condition=str(row["name"]),
            replicate=int(row["replicate"]),
            minutes=float(row["doubling.time.minutes"]),
            lower_95=float(row["doubling.time.minutes.95m"]),
            upper_95=float(row["doubling.time.minutes.95p"]),
            r_squared=float(row["r.squared"]),
        )
        for _, row in frame.iterrows()
    ]


def interval_census(triples: list[tuple[float, float, float]]) -> IntervalCensus:
    """Symmetry census of released ``(value, 95m, 95p)`` triples, row by row."""
    lower_half = [value - lower for value, lower, _ in triples]
    upper_half = [upper - value for value, _, upper in triples]
    pairs = list(zip(upper_half, lower_half, strict=True))
    ratios = [up / low for up, low in pairs]
    asymmetry = [abs(up - low) for up, low in pairs]
    return IntervalCensus(
        n_rows=len(triples),
        n_symmetric=sum(1 for a in asymmetry if a < 1e-9),
        n_upper_wider=sum(1 for up, low in pairs if up > low),
        n_lower_wider=sum(1 for up, low in pairs if up < low),
        n_upper_bound_not_above_value=sum(
            1 for value, _, upper in triples if upper <= value
        ),
        half_width_ratio_min=min(ratios),
        half_width_ratio_median=statistics.median(ratios),
        half_width_ratio_max=max(ratios),
        abs_asymmetry_minutes_median=statistics.median(asymmetry),
        abs_asymmetry_minutes_max=max(asymmetry),
    )


def condition_means(fits: list[FitRow]) -> dict[str, ConditionMeans]:
    """Each condition's replicate count and the three means of its replicate values."""
    grouped: dict[str, list[float]] = {}
    for fit in fits:
        grouped.setdefault(fit.condition, []).append(fit.minutes)
    return {
        condition: ConditionMeans(
            condition=condition,
            n_replicates=len(values),
            arithmetic=statistics.fmean(values),
            geometric=statistics.geometric_mean(values),
            harmonic=statistics.harmonic_mean(values),
        )
        for condition, values in sorted(grouped.items())
    }


def join_table_s1(path: str, means: dict[str, ConditionMeans]) -> list[JoinedCondition]:
    """Each Table S5 condition beside the Table S1 sample-level value of that condition.

    Raises when a Table S5 condition does not select exactly one Table S1 doubling time,
    which is what makes the 1:1 correspondence a measurement rather than an assumption.
    """
    frame = pd.read_csv(path)
    stated = frame.dropna(subset=["doublingTimeMinutes"])
    joined = []
    for condition, mean in means.items():
        experiment, carbon, mg_mm, na_mm = CONDITIONS[condition]
        members = stated[
            stated["experiment"].str.startswith(experiment)
            & (stated["carbonSource"] == carbon)
            & (stated["Mg_mM"] == mg_mm)
            & (stated["Na_mM"] == na_mm)
        ]
        values = sorted({float(v) for v in members["doublingTimeMinutes"]})
        if len(values) != 1:
            raise RuntimeError(f"{condition} selects {len(values)} Table S1 values")
        joined.append(
            JoinedCondition(
                table_s5_condition=condition,
                table_s1_experiment=experiment,
                carbon_source=carbon,
                mg_mm=mg_mm,
                na_mm=na_mm,
                n_replicates=mean.n_replicates,
                table_s1_sample_rows=len(members),
                table_s1_doubling_time=values[0],
                table_s1_95m=float(members["doublingTimeMinutes.95m"].iloc[0]),
                table_s1_95p=float(members["doublingTimeMinutes_95p"].iloc[0]),
                table_s5_arithmetic_mean=mean.arithmetic,
                table_s5_geometric_mean=mean.geometric,
                table_s5_harmonic_mean=mean.harmonic,
                s1_minus_s5_arithmetic=values[0] - mean.arithmetic,
                s1_minus_s5_geometric=values[0] - mean.geometric,
                s1_minus_s5_harmonic=values[0] - mean.harmonic,
            )
        )
    return joined


def table_s1_intervals(path: str) -> list[tuple[float, float, float]]:
    """Table S1's sample-level ``(doubling time, 95m, 95p)`` triples."""
    frame = pd.read_csv(path).dropna(subset=["doublingTimeMinutes"])
    return [
        (
            float(row["doublingTimeMinutes"]),
            float(row["doublingTimeMinutes.95m"]),
            float(row["doublingTimeMinutes_95p"]),
        )
        for _, row in frame.iterrows()
    ]


def table_s1_distinct_tuples(path: str) -> int:
    """How many distinct ``(value, 95m, 95p, r2)`` tuples Table S1's 165 rows hold."""
    frame = pd.read_csv(path).dropna(subset=["doublingTimeMinutes"])
    return len(
        {
            (
                float(row["doublingTimeMinutes"]),
                float(row["doublingTimeMinutes.95m"]),
                float(row["doublingTimeMinutes_95p"]),
                float(row["rSquared"]),
            )
            for _, row in frame.iterrows()
        }
    )


def environment_of(condition: str) -> Environment:
    """The Caglar loader's own environment for one Table S5 condition."""
    _, carbon, mg_mm, na_mm = CONDITIONS[condition]
    perturbations: list[EnvironmentPerturbationType] = []
    media = DM500
    if carbon != "glucose":
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


RATIO_UNCERTAINTY_GAP = ProvenanceGap(
    field="environment_response_uncertainty",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="Table S5 reports a per-replicate asymmetric 95% confidence interval of the "
    "OD600 slope fit, not an uncertainty of a ratio of doubling times; the paper "
    "releases no ratio and so no uncertainty of one",
)


def absolute_phenotype(
    minutes: float, n_replicates: int | None
) -> EnvironmentResponsePhenotype:
    """The absolute doubling time as the environment-response readout."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.growth_rate,
        assay_type=AssayType.liquid_od_growth,
        environment_response=minutes,
        n_samples=n_replicates,
        sample_unit=None if n_replicates is None else SampleUnit.biological_replicate,
        units="doubling time in minutes",
    )


def ratio_phenotype(
    log2_ratio: float, n_replicates: int, screen_id: str
) -> EnvironmentResponsePhenotype:
    """The doubling time as a log2 ratio against its series' base condition."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.liquid_od_growth,
        environment_response=log2_ratio,
        n_samples=n_replicates,
        sample_unit=SampleUnit.biological_replicate,
        units="log2(doubling time / base-condition doubling time)",
        screen_id=screen_id,
        provenance_gaps=[RATIO_UNCERTAINTY_GAP],
    )


def half_width_as_ci95(minutes: float, half_width: float) -> float | None:
    """The SE the schema derives if ONE side of the asymmetric interval is passed as ci95.

    ``UncertaintyType.ci95`` is defined as a single half-width, so feeding it either bound
    of an asymmetric interval is accepted silently. Returned so the result file states
    what storing the P. putida audit's ``ci95`` would actually record.
    """
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.growth_rate,
        assay_type=AssayType.liquid_od_growth,
        environment_response=minutes,
        environment_response_uncertainty=half_width,
        environment_response_uncertainty_type=UncertaintyType.ci95,
        n_samples=1,
        sample_unit=SampleUnit.biological_replicate,
        units="doubling time in minutes",
    ).environment_response_se


def record(
    condition: str,
    phenotype: EnvironmentResponsePhenotype,
    reference_condition: str,
    reference_phenotype: EnvironmentResponsePhenotype,
) -> dict[str, Any]:
    """One verifier-shaped record dict around a real pydantic experiment + reference."""
    experiment = BacterialEnvironmentResponseExperiment(
        dataset_name="caglar2017_doubling_time_probe",
        genotype=Genotype(perturbations=[]),
        environment=environment_of(condition),
        phenotype=phenotype,
    )
    reference = BacterialEnvironmentResponseExperimentReference(
        dataset_name="caglar2017_doubling_time_probe",
        genome_reference=assembly_reference("REL606"),
        environment_reference=environment_of(reference_condition),
        phenotype_reference=reference_phenotype,
    )
    return {
        "experiment": experiment.model_dump(),
        "reference": reference.model_dump(),
        "publication": publication().model_dump(),
    }


def score(form: str, description: str, records: list[dict[str, Any]]) -> FormOutcome:
    """Run the three environment-response rules the disagreement turns on."""
    results = [
        _l1_pair_uniqueness(records, frozenset()),
        _l3_environment_perturbed(records),
        _l3_reference_zero(records),
    ]
    return FormOutcome(
        form=form,
        description=description,
        n_records=len(records),
        rules=[
            RuleOutcome(name=r.name, passed=r.passed, message=r.message)
            for r in results
        ],
    )


def candidate_forms(
    fits: list[FitRow], means: dict[str, ConditionMeans]
) -> list[FormOutcome]:
    """Build and score every record form the two audit notes propose or imply."""
    base = means["Glucose.tab"].arithmetic
    base_reference = absolute_phenotype(base, None)

    per_replicate = [
        record(
            fit.condition,
            absolute_phenotype(fit.minutes, None),
            "Glucose.tab",
            base_reference,
        )
        for fit in fits
    ]
    per_condition = [
        record(
            mean.condition,
            absolute_phenotype(mean.arithmetic, mean.n_replicates),
            "Glucose.tab",
            base_reference,
        )
        for mean in means.values()
    ]
    glucose_reference = ratio_phenotype(
        0.0, means["Glucose.tab"].n_replicates, "glucose_time_course"
    )
    ratio_all = [
        record(
            mean.condition,
            ratio_phenotype(
                math.log2(mean.arithmetic / base),
                mean.n_replicates,
                CONDITIONS[mean.condition][0],
            ),
            "Glucose.tab",
            glucose_reference,
        )
        for mean in means.values()
    ]

    base_conditions = set(BASE_ROWS.values())
    ratio_in_series = []
    for mean in means.values():
        if mean.condition in base_conditions:
            continue
        experiment = CONDITIONS[mean.condition][0]
        if experiment not in BASE_ROWS:
            continue
        series_base = means[BASE_ROWS[experiment]]
        ratio_in_series.append(
            record(
                mean.condition,
                ratio_phenotype(
                    math.log2(mean.arithmetic / series_base.arithmetic),
                    mean.n_replicates,
                    experiment,
                ),
                series_base.condition,
                ratio_phenotype(0.0, series_base.n_replicates, experiment),
            )
        )

    return [
        score(
            "A_absolute_per_replicate",
            "the P. putida audit's 55 records: Table S5's per-replicate doubling time "
            "in minutes as the environment response, reference = the base condition's "
            "absolute doubling time",
            per_replicate,
        ),
        score(
            "B_absolute_per_condition",
            "the E. coli audit's 19 records: the per-condition mean doubling time in "
            "minutes, reference = the base condition's absolute doubling time",
            per_condition,
        ),
        score(
            "C_log2_ratio_all_conditions",
            "all 19 conditions as log2(doubling time / Glucose.tab doubling time), "
            "screen_id = the Table S1 experiment",
            ratio_all,
        ),
        score(
            "D_log2_ratio_in_series_base_only",
            "log2 ratio for only the conditions whose own Table S1 experiment released "
            "a base-condition Table S5 row, with screen_id = that experiment",
            ratio_in_series,
        ),
    ]


def write_conditions_csv(joined: list[JoinedCondition]) -> str:
    """The per-condition Table S1 against Table S5 table, as a CSV."""
    path = osp.join(RESULTS_DIR, "caglar2017_doubling_time_conditions.csv")
    fields = list(JoinedCondition.model_fields)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in joined:
            writer.writerow(row.model_dump())
    return path


def main() -> None:
    """Measure both tables, score every candidate form, and write the results."""
    load_dotenv()
    key_dir = osp.join(os.environ["DATA_ROOT"], "torchcell-library", CITATION_KEY)
    s5_path = verified_si_path(key_dir, "si/si6.csv")
    s1_path = verified_si_path(key_dir, "si/si2.csv")

    fits = read_table_s5(s5_path)
    means = condition_means(fits)
    joined = join_table_s1(s1_path, means)

    repeated_keys = {
        key for key, n in Counter(f.numeric_key() for f in fits).items() if n > 1
    }
    repeated = [f for f in fits if f.numeric_key() in repeated_keys]
    worst = max(
        fits, key=lambda f: abs((f.upper_95 - f.minutes) - (f.minutes - f.lower_95))
    )
    base_minutes = {c: means[c].arithmetic for c in sorted(set(BASE_ROWS.values()))}

    out = {
        "citation_key": CITATION_KEY,
        "table_s5": {
            "path": "si/si6.csv",
            "sha256": sha256_of(s5_path),
            "n_data_rows": len(fits),
            "n_conditions": len(means),
            "replicates_per_condition": {c: m.n_replicates for c, m in means.items()},
            "replicate_count_distribution": dict(
                Counter(m.n_replicates for m in means.values())
            ),
            "conditions_not_in_triplicate": {
                c: m.n_replicates for c, m in means.items() if m.n_replicates != 3
            },
            "n_rows_repeating_another_rows_numbers": len(repeated),
            "repeated_rows": [row.model_dump() for row in repeated],
            "interval_census": interval_census(
                [(f.minutes, f.lower_95, f.upper_95) for f in fits]
            ).model_dump(),
            "most_asymmetric_row": worst.model_dump(),
            "se_if_lower_half_width_passed_as_ci95": half_width_as_ci95(
                worst.minutes, worst.minutes - worst.lower_95
            ),
            "se_if_upper_half_width_passed_as_ci95": half_width_as_ci95(
                worst.minutes, worst.upper_95 - worst.minutes
            ),
        },
        "table_s1": {
            "path": "si/si2.csv",
            "sha256": sha256_of(s1_path),
            "n_rows_with_a_doubling_time": len(table_s1_intervals(s1_path)),
            "n_distinct_value_interval_r2_tuples": table_s1_distinct_tuples(s1_path),
            "interval_census": interval_census(
                table_s1_intervals(s1_path)
            ).model_dump(),
        },
        "table_s1_versus_table_s5": {
            "join": "1:1 on (experiment, carbonSource, Mg_mM, Na_mM); raises otherwise",
            "n_conditions_joined": len(joined),
            "n_table_s1_rows_covered": sum(j.table_s1_sample_rows for j in joined),
            "n_exact_matches": {
                kind: sum(1 for j in joined if abs(j.difference(kind)) < 1e-6)
                for kind in MEAN_KINDS
            },
            "abs_difference_minutes": {
                kind: {
                    "median": statistics.median(
                        [abs(j.difference(kind)) for j in joined]
                    ),
                    "max": max(abs(j.difference(kind)) for j in joined),
                }
                for kind in MEAN_KINDS
            },
        },
        "base_condition_choice": {
            "table_s5_rows_that_are_a_base_condition": sorted(base_minutes),
            "arithmetic_mean_minutes": base_minutes,
            "log2_spread_of_the_choice": math.log2(
                max(base_minutes.values()) / min(base_minutes.values())
            ),
            "experiments_with_no_base_row": sorted(
                {CONDITIONS[c][0] for c in means if CONDITIONS[c][0] not in BASE_ROWS}
            ),
            "conditions_with_no_base_row": sorted(
                c for c in means if CONDITIONS[c][0] not in BASE_ROWS
            ),
        },
        "candidate_forms": [form.model_dump() for form in candidate_forms(fits, means)],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = osp.join(RESULTS_DIR, "caglar2017_doubling_time_loadability.json")
    with open(json_path, "w") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    write_conditions_csv(joined)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
