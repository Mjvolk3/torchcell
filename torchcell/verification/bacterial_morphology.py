# torchcell/verification/bacterial_morphology
# [[torchcell.verification.bacterial_morphology]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/bacterial_morphology
"""L0-L4 record-level verifier for a ``BacterialMorphologyPhenotype`` dataset.

:mod:`torchcell.verification.morphology` verifies the yeast CalMorph family and cannot
serve this one: it imports ``CALMORPH_LABELS`` / ``CALMORPH_STATISTICS`` at module scope,
reads ``phenotype["calmorph"]`` by literal key, and asserts the literal counts 281 / 220
/ 501. A bacterial record names its own assay, so the vocabulary is read from
``MORPHOLOGY_ASSAYS[phenotype["assay"]]`` per record and a dataset that mixes assays is
caught rather than averaged.

What the schema already guarantees at L0, so this verifier does not re-check it: every
key is a feature of the named assay, a key whose declared statistic is a coefficient of
variation is in ``morphology_coefficient_of_variation`` and no other key is, and no value
is NaN (``BacterialMorphologyPhenotype.validate_against_the_assay``). What it adds:

1. L1 ``assay_coverage`` -- the schema accepts any non-empty subset, so a build that
   silently dropped a column would pass it. The check is deliberately NOT full coverage:
   a feature the source did not determine for a strain is absent by design (Campos 2018
   writes a non-determined field as NaN, and 278 of its 4,227 imaged strains have no
   nucleoid channel). It asserts instead that every record carries the features
   ``required`` names, that no record carries a feature outside the assay, and it
   censuses the coverage strata so an unexpected one is visible rather than averaged
   away.
2. L2 ``value_fidelity`` -- PER FEATURE, with the bound its declared statistic implies
   (a CV and a mean are non-negative, a Pearson correlation lies in [-1, 1], a fraction
   of cells and an inferred relative timing in [0, 1], a fitted intercept is bounded only
   by finiteness). Pooling 26 columns into one list, which is what the CalMorph verifier
   does, cannot express those and reports an index no reader can trace back to a column.
3. L2 ``cv_nonnegative`` -- the pooled CV check the CalMorph verifier also runs, kept
   because a negative CV is a computation bug worth naming on its own.
4. L3 ``vocabulary_parity`` -- the named assay is internally consistent: its symbols are
   distinct, its value and CV symbol sets are disjoint and together are all of it.
5. L3 ``reference_populated`` -- the reference profile carries at least the features
   ``required`` names, so a record is always comparable to a parent.

The shared family-agnostic rules (the gap census, canonical gene names, uncertainty
sanity, compound identity, media membership, and the L4 gene universe) come from
:class:`~torchcell.verification.common.SharedRecordRules`, so they read identically here
and in the fitness verifier.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from typing import Any

from torchcell.datamodels.bacterial_morphology_features import (
    MorphologyAssay,
    MorphologyStatistic,
    morphology_assay,
)
from torchcell.verification.common import GeneNameResolver, SharedRecordRules
from torchcell.verification.levels import l0_structural, l1_count, l3_convention
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

# One LMDB entry: {"experiment": {...}, "reference": {...}, "publication": {...}}.
Record = dict[str, Any]

#: The closed interval each statistic's values must lie in, or ``None`` when the
#: statistic implies no bound beyond finiteness. Every bound is a property of the
#: statistic's definition, not of any one release: a coefficient of variation is a
#: standard deviation over a mean of a positive quantity, a Pearson correlation is
#: bounded by Cauchy-Schwarz, a fraction of cells is a proportion, and an inferred
#: relative timing is a cell age in units of the cycle. A fitted intercept has no such
#: bound, so none is asserted for it.
STATISTIC_BOUNDS: dict[MorphologyStatistic, tuple[float, float] | None] = {
    MorphologyStatistic.mean: (0.0, math.inf),
    MorphologyStatistic.coefficient_of_variation: (0.0, math.inf),
    MorphologyStatistic.pearson_correlation: (-1.0, 1.0),
    MorphologyStatistic.regression_intercept: None,
    MorphologyStatistic.fraction_of_cells: (0.0, 1.0),
    MorphologyStatistic.inferred_relative_timing: (0.0, 1.0),
}


def _feature_values(phenotype: dict[str, Any]) -> dict[str, float]:
    """Every stored feature of one record, both dicts flattened into one mapping."""
    values = dict(phenotype["morphology"])
    values.update(phenotype.get("morphology_coefficient_of_variation") or {})
    return values


def _assay_of(records: Sequence[Record]) -> MorphologyAssay:
    """The one assay every record names, or a ValueError naming the ones found."""
    names = {rec["experiment"]["phenotype"]["assay"] for rec in records}
    if len(names) != 1:
        raise ValueError(
            f"a morphology dataset carries one assay vocabulary; found {sorted(names)}"
        )
    return morphology_assay(names.pop())


def _l1_assay_coverage(
    records: Sequence[Record], assay: MorphologyAssay, required: frozenset[str]
) -> LevelResult:
    """L1: no feature outside the assay, and every record carries ``required``."""
    symbols = frozenset(assay.by_symbol)
    unknown: list[dict[str, Any]] = []
    short: list[dict[str, Any]] = []
    strata: dict[int, int] = {}
    for i, rec in enumerate(records):
        stored = frozenset(_feature_values(rec["experiment"]["phenotype"]))
        strata[len(stored)] = strata.get(len(stored), 0) + 1
        outside = stored - symbols
        if outside:
            unknown.append({"index": i, "outside_the_assay": sorted(outside)})
        missing = required - stored
        if missing:
            short.append({"index": i, "missing_required": sorted(missing)})
    passed = not unknown and not short
    return LevelResult(
        level=Level.L1,
        name="assay_coverage",
        passed=passed,
        message=(
            f"all {len(records)} records carry the {len(required)} required features of "
            f"assay {assay.name} and nothing outside its {len(symbols)}"
            if passed
            else f"{len(unknown)} records carry a feature outside assay {assay.name} "
            f"and {len(short)} are missing a required feature"
        ),
        details={
            "assay": assay.name,
            "n_records": len(records),
            "n_assay_features": len(symbols),
            "n_required_features": len(required),
            "n_records_outside_the_assay": len(unknown),
            "n_records_missing_required": len(short),
            "coverage_strata": {str(k): v for k, v in sorted(strata.items())},
            "outside_the_assay": unknown[:20],
            "missing_required": short[:20],
        },
    )


def _l2_value_fidelity(
    records: Sequence[Record], assay: MorphologyAssay
) -> LevelResult:
    """L2: every stored value is finite and inside the bound its statistic implies."""
    checked = 0
    by_feature: dict[str, int] = {}
    bad: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        for symbol, value in _feature_values(rec["experiment"]["phenotype"]).items():
            feature = assay.by_symbol[symbol]
            number = float(value)
            checked += 1
            by_feature[symbol] = by_feature.get(symbol, 0) + 1
            bounds = STATISTIC_BOUNDS[feature.statistic]
            if not math.isfinite(number) or (
                bounds is not None and not bounds[0] <= number <= bounds[1]
            ):
                bad.append(
                    {
                        "index": i,
                        "feature": symbol,
                        "statistic": feature.statistic.value,
                        "value": number,
                        "bounds": None if bounds is None else list(bounds),
                    }
                )
    passed = not bad
    return LevelResult(
        level=Level.L2,
        name="value_fidelity",
        passed=passed,
        message=(
            f"{checked} values over {len(by_feature)} features are finite and inside "
            "the bound of their declared statistic"
            if passed
            else f"{len(bad)}/{checked} values are non-finite or outside their "
            "statistic's bound"
        ),
        details={
            "n_values": checked,
            "n_features": len(by_feature),
            "n_bad": len(bad),
            "values_by_feature": dict(sorted(by_feature.items())),
            "bad": bad[:20],
        },
    )


def _l2_cv_nonnegative(records: Sequence[Record]) -> LevelResult:
    """L2: a coefficient of variation is non-negative by definition."""
    checked = 0
    negative: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        coefficients = (
            rec["experiment"]["phenotype"].get("morphology_coefficient_of_variation")
            or {}
        )
        for symbol, value in coefficients.items():
            checked += 1
            if float(value) < 0.0:
                negative.append({"index": i, "feature": symbol, "value": float(value)})
    passed = not negative
    return LevelResult(
        level=Level.L2,
        name="cv_nonnegative",
        passed=passed,
        message=(
            f"all {checked} coefficients of variation are non-negative"
            if passed
            else f"{len(negative)}/{checked} coefficients of variation are negative"
        ),
        details={
            "n_values": checked,
            "n_negative": len(negative),
            "negative": negative[:20],
        },
    )


def _l1_pair_uniqueness(records: Sequence[Record]) -> LevelResult:
    """L1: one record per (strain, environment).

    The strain is the sorted set of its perturbations, each keyed by ``(systematic gene
    name, perturbation type, perturbed gene name)``; the environment is its whole stored
    object, so two conditions can never be merged by a key that forgot a field.
    """
    seen: dict[tuple[Any, ...], int] = {}
    for rec in records:
        experiment = rec["experiment"]
        strain = tuple(
            sorted(
                (
                    p.get("systematic_gene_name"),
                    p.get("perturbation_type"),
                    p.get("perturbed_gene_name"),
                )
                for p in experiment["genotype"]["perturbations"]
            )
        )
        key = (strain, json.dumps(experiment["environment"], sort_keys=True))
        seen[key] = seen.get(key, 0) + 1
    duplicated = {key: n for key, n in seen.items() if n > 1}
    return LevelResult(
        level=Level.L1,
        name="pair_uniqueness",
        passed=not duplicated,
        message=(
            f"{len(seen)} unique (strain, environment) records, one each"
            if not duplicated
            else f"{len(duplicated)} (strain, environment) pairs appear in more than "
            "one record"
        ),
        details={
            "n_pairs": len(seen),
            "n_duplicated": len(duplicated),
            "n_strains": len({strain for strain, _ in seen}),
            "n_environments": len({environment for _, environment in seen}),
        },
    )


def _l3_vocabulary_parity(assay: MorphologyAssay) -> LevelResult:
    """L3: the named assay's symbol sets are distinct, disjoint and exhaustive."""
    symbols = [feature.symbol for feature in assay.features]
    values = assay.value_symbols
    coefficients = assay.coefficient_of_variation_symbols
    distinct = len(set(symbols)) == len(symbols)
    disjoint = not (values & coefficients)
    exhaustive = values | coefficients == frozenset(symbols)
    holds = distinct and disjoint and exhaustive
    return l3_convention(
        "vocabulary_parity",
        holds,
        detail=(
            f"assay {assay.name}: {len(values)} value symbols + {len(coefficients)} CV "
            f"symbols == {len(symbols)} features, disjoint={disjoint}, "
            f"exhaustive={exhaustive}, distinct={distinct}"
        ),
    )


def _l3_reference_populated(
    records: Sequence[Record], required: frozenset[str]
) -> LevelResult:
    """L3: every record's reference profile carries at least the required features."""
    short: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        stored = frozenset(_feature_values(rec["reference"]["phenotype_reference"]))
        missing = required - stored
        if missing:
            short.append({"index": i, "missing_required": sorted(missing)})
    passed = not short
    return LevelResult(
        level=Level.L3,
        name="reference_populated",
        passed=passed,
        message=(
            f"the reference profile carries the {len(required)} required features in "
            f"all {len(records)} records"
            if passed
            else f"{len(short)}/{len(records)} records have an under-populated reference"
        ),
        details={
            "n_required_features": len(required),
            "n_short": len(short),
            "short": short[:20],
        },
    )


def verify_bacterial_morphology_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    required_features: frozenset[str],
    sgd_genes: set[str] | None = None,
    gene_universe_label: str = "reference",
    resolve_gene_name: GeneNameResolver | None = None,
    min_containment: float = 0.90,
) -> VerificationReport:
    """Run the L0-L4 gate for a ``BacterialMorphologyPhenotype`` dataset.

    Args:
        records: the per-LMDB-entry dicts (experiment / reference / publication).
        dataset_name: dataset identity for the report.
        provenance: where the records came from.
        expected_count: the record-count oracle.
        required_features: the assay symbols every record must carry. The complement is
            the set a source may leave non-determined for a strain, which is data rather
            than a defect, so it is the caller that decides and states which is which.
        sgd_genes: the L4 gene universe. When given, the shared gene rules are emitted.
        gene_universe_label: what that universe is, for the L4 message.
        resolve_gene_name: the host's canonical-name resolver, for the L1 name rule.
        min_containment: the L4 containment floor.

    Returns:
        A :class:`VerificationReport` carrying the L0-L4 results.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    assay = _assay_of(records)
    outside = required_features - frozenset(assay.by_symbol)
    if outside:
        raise ValueError(
            f"required_features {sorted(outside)} are not features of assay {assay.name}"
        )

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_pair_uniqueness(records))
    report.add(_l1_assay_coverage(records, assay, required_features))
    report.add(_l2_value_fidelity(records, assay))
    report.add(_l2_cv_nonnegative(records))
    report.add(_l3_vocabulary_parity(assay))
    report.add(_l3_reference_populated(records, required_features))

    shared = SharedRecordRules(
        resolve_gene_name=resolve_gene_name,
        sgd_genes=sgd_genes,
        gene_universe_label=gene_universe_label,
        min_containment=min_containment,
    )
    for record in records:
        shared.add(record)
    for result in shared.results():
        report.add(result)
    return report


def morphology_gene_set(records: Sequence[Record]) -> set[str]:
    """The systematic gene names a morphology dataset perturbs, for L4 cross-source."""
    return {
        p["systematic_gene_name"]
        for rec in records
        for p in rec["experiment"]["genotype"]["perturbations"]
        if p.get("systematic_gene_name")
    }
