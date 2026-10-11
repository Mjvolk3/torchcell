# torchcell/verification/expression_score
# [[torchcell.verification.expression_score]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/expression_score
# Test file: tests/torchcell/verification/test_expression_score.py
"""L0-L4 record-level verifier for error-model microarray SCORE datasets (#1022).

The family is ``MicroarrayExpressionScorePhenotype``: a per-gene signed significance
score of a strain against a reference, with its p-value, and no log ratio (Hu 2007,
GEO GSE4654). The checks reuse :mod:`torchcell.verification.levels` and assert the
family's own conventions:

1. ``reference_score_zero`` -- a reference is not tested against itself, so its score
   is the neutral 0 for every gene and it carries no p-value.
2. ``deletion_lowers_score`` -- the score is signed like ``log(sample / reference)``;
   a strain with gene *G* deleted must show *G* lowered, so the MEDIAN score at the
   deleted genes is < 0. A sign inversion would flip it positive.
3. ``p_value_is_two_sided_normal`` (optional) -- when the release's p-value is the
   two-sided normal tail of the score, ``p == 2 * (1 - Phi(|score|))`` holds within the
   release's rounding; the caller states that tolerance.

L4 (:func:`l4_matched_profile_agreement`) compares each strain's profile with the
profile a second source measured for the SAME deleted gene, against the profiles the
second source measured for the OTHER shared genes: matched pairs must correlate more
than mismatched ones. Two measurements of one strain are two records, so the check is
an agreement test, not a duplicate test, and its threshold is a one-sided rank test
rather than a fixed correlation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from statistics import median
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import mannwhitneyu, norm

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

# One LMDB entry: {"experiment": {...}, "reference": {...}, "publication": {...}}.
Record = Mapping[str, Any]


def deleted_gene(experiment: Mapping[str, Any]) -> str | None:
    """The one deleted gene of a single-perturbation experiment, else None."""
    perturbations = experiment["genotype"]["perturbations"]
    if len(perturbations) != 1:
        return None
    name = perturbations[0].get("systematic_gene_name")
    return None if name is None else str(name)


def _l2_p_value_keys(records: Sequence[Record]) -> LevelResult:
    """L2: every record's p-value map tests exactly the genes its score map carries."""
    bad = [
        i
        for i, rec in enumerate(records)
        if set(rec["experiment"]["phenotype"]["expression_score_p_value"] or {})
        != set(rec["experiment"]["phenotype"]["expression_score"])
    ]
    return LevelResult(
        level=Level.L2,
        name="p_value_keys_match_scores",
        passed=not bad,
        message=(
            f"all {len(records)} records carry a p-value for every scored gene"
            if not bad
            else f"{len(bad)}/{len(records)} records key p-values unlike scores"
        ),
        details={"n_records": len(records), "bad_indices": bad[:20]},
    )


def _l3_reference_zero(records: Sequence[Record]) -> LevelResult:
    """L3: every reference score is exactly 0 and no reference carries a p-value."""
    worst = 0.0
    n = 0
    with_p = 0
    for rec in records:
        reference = rec["reference"]["phenotype_reference"]
        if reference["expression_score_p_value"] is not None:
            with_p += 1
        for value in reference["expression_score"].values():
            n += 1
            worst = max(worst, abs(float(value)))
    holds = worst == 0.0 and with_p == 0
    return l3_convention(
        "reference_score_zero",
        holds,
        detail=(
            f"reference score == 0 for all {n} values and no reference p-value"
            if holds
            else f"max|reference score|={worst:.3g}; {with_p} references carry p-values"
        ),
    )


def _l3_deletion_orientation(records: Sequence[Record]) -> LevelResult:
    """L3: deletions lower the deleted gene's score -> the median is < 0."""
    scores: list[float] = []
    absent = 0
    for rec in records:
        experiment = rec["experiment"]
        gene = deleted_gene(experiment)
        if gene is None:
            continue
        values = experiment["phenotype"]["expression_score"]
        if gene in values:
            scores.append(float(values[gene]))
        else:
            absent += 1
    if not scores:
        return l3_convention(
            "deletion_lowers_score",
            False,
            detail="no deleted gene carries a score in its own record",
        )
    med = median(scores)
    frac_neg = sum(1 for v in scores if v < 0) / len(scores)
    return l3_convention(
        "deletion_lowers_score",
        med < 0.0,
        detail=(
            f"median deleted-gene score={med:.3f} (<0 => correct orientation); "
            f"frac_neg={frac_neg:.3f} over {len(scores)} deleted genes "
            f"({absent} deleted genes carry no score in their own record)"
        ),
    )


def _l3_two_sided_normal(records: Sequence[Record], tol: float) -> LevelResult:
    """L3: ``p == 2 * (1 - Phi(|score|))`` within ``tol`` for every stored pair."""
    worst = 0.0
    n = 0
    for rec in records:
        phenotype = rec["experiment"]["phenotype"]
        p_values = phenotype["expression_score_p_value"] or {}
        for gene, score in phenotype["expression_score"].items():
            n += 1
            expected = 2.0 * float(norm.sf(abs(float(score))))
            worst = max(worst, abs(expected - float(p_values[gene])))
    return l3_convention(
        "p_value_is_two_sided_normal",
        worst <= tol,
        detail=(
            f"max|p - 2*(1 - Phi(|score|))| = {worst:.3g} over {n} pairs "
            f"(tolerance {tol:g}, the release's rounding)"
        ),
    )


def verify_expression_score_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    two_sided_normal_tol: float | None = None,
) -> VerificationReport:
    """Run the family's L0-L3 gate for one error-model microarray score dataset.

    Args:
        records: the per-LMDB-entry dicts (experiment/reference/publication).
        dataset_name: dataset identity for the report.
        provenance: where the records came from.
        expected_count: the record-count oracle.
        two_sided_normal_tol: when set, assert the release's p-value is the two-sided
            normal tail of the score within this tolerance.

    Returns:
        A :class:`VerificationReport` with the L0-L3 results; the caller adds L4.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    scores = [
        float(v)
        for rec in records
        for v in rec["experiment"]["phenotype"]["expression_score"].values()
    ]
    report.add(
        l2_value_fidelity(scores, allow_nan=False).model_copy(
            update={"name": "score_finite"}
        )
    )
    p_values = [
        float(v)
        for rec in records
        for v in (
            rec["experiment"]["phenotype"]["expression_score_p_value"] or {}
        ).values()
    ]
    report.add(
        l2_value_fidelity(
            p_values, allow_nan=False, minimum=0.0, maximum=1.0
        ).model_copy(update={"name": "p_value_in_unit_interval"})
    )
    report.add(_l2_p_value_keys(records))
    report.add(_l3_reference_zero(records))
    report.add(_l3_deletion_orientation(records))
    if two_sided_normal_tol is not None:
        report.add(_l3_two_sided_normal(records, two_sided_normal_tol))
    return report


def deletion_profiles(
    records: Sequence[Record], field: str
) -> dict[str, Mapping[str, float]]:
    """``deleted gene -> its per-gene profile`` for the single-deletion records."""
    out: dict[str, Mapping[str, float]] = {}
    for rec in records:
        gene = deleted_gene(rec["experiment"])
        if gene is not None:
            out[gene] = rec["experiment"]["phenotype"][field]
    return out


def matched_profile_correlations(
    profiles: Mapping[str, Mapping[str, float]],
    other: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, float], list[float], int]:
    """Pearson r of each shared strain's profile with the other source's profiles.

    Genes are the union of both sources' keys minus every shared strain's deleted gene
    (so no pair is driven by the trivially lowered deleted gene). A pair is scored over
    the genes BOTH profiles carry. Returns ``(matched r per strain, every mismatched r,
    the size of the gene universe)``.
    """
    shared = sorted(set(profiles) & set(other))
    excluded = set(shared)
    genes = sorted(
        (
            set().union(*(set(p) for p in profiles.values()))
            & set().union(*(set(p) for p in other.values()))
        )
        - excluded
    )
    column = {gene: j for j, gene in enumerate(genes)}

    def matrix(source: Mapping[str, Mapping[str, float]]) -> npt.NDArray[np.float64]:
        values = np.full((len(shared), len(genes)), np.nan)
        for i, strain in enumerate(shared):
            for gene, value in source[strain].items():
                j = column.get(gene)
                if j is not None:
                    values[i, j] = float(value)
        return values

    a, b = matrix(profiles), matrix(other)
    matched: dict[str, float] = {}
    mismatched: list[float] = []
    for i, strain in enumerate(shared):
        for k in range(len(shared)):
            mask = ~np.isnan(a[i]) & ~np.isnan(b[k])
            x, y = a[i, mask], b[k, mask]
            r = float(np.corrcoef(x, y)[0, 1]) if x.size > 2 else math.nan
            if k == i:
                matched[strain] = r
            else:
                mismatched.append(r)
    return matched, mismatched, len(genes)


def l4_matched_profile_agreement(
    records: Sequence[Record],
    other_records: Sequence[Record],
    *,
    other_name: str,
    other_field: str,
    alpha: float,
) -> LevelResult:
    """L4: a strain correlates more with the SAME deletion in another source than with
    other deletions there.

    Passes when the matched Pearson r exceed the mismatched ones by a one-sided
    Mann-Whitney U test at ``alpha`` and the matched median is the larger. A NaN r
    (fewer than three shared genes) fails the check rather than being dropped.
    """
    matched, mismatched, n_genes = matched_profile_correlations(
        deletion_profiles(records, "expression_score"),
        deletion_profiles(other_records, other_field),
    )
    name = f"matched_profile_agreement_{other_name}"
    if len(matched) < 2 or any(math.isnan(r) for r in [*matched.values(), *mismatched]):
        return LevelResult(
            level=Level.L4,
            name=name,
            passed=False,
            message=f"{len(matched)} shared strains; too few or undefined correlations",
            details={"n_shared": len(matched), "n_genes": n_genes},
        )
    matched_r = list(matched.values())
    test = mannwhitneyu(matched_r, mismatched, alternative="greater")
    p_value = float(test.pvalue)
    med_matched, med_mismatched = median(matched_r), median(mismatched)
    passed = p_value < alpha and med_matched > med_mismatched
    q25, q75 = (float(q) for q in np.percentile(matched_r, [25, 75]))
    top = sorted(matched.items(), key=lambda item: -item[1])[:10]
    return LevelResult(
        level=Level.L4,
        name=name,
        passed=passed,
        message=(
            f"{len(matched)} shared deletion strains: median matched r={med_matched:.3f} "
            f"vs mismatched {med_mismatched:.3f} (one-sided Mann-Whitney p={p_value:.2g}, "
            f"alpha {alpha:g})"
        ),
        details={
            "n_shared": len(matched),
            "n_genes": n_genes,
            "median_matched_r": med_matched,
            "matched_r_q25": q25,
            "matched_r_q75": q75,
            "median_mismatched_r": med_mismatched,
            "n_mismatched": len(mismatched),
            "mann_whitney_u": float(test.statistic),
            "mann_whitney_p": p_value,
            "alpha": alpha,
            "top_matched": [{"strain": s, "r": r} for s, r in top],
        },
    )
