# torchcell/verification/expression
# [[torchcell.verification.expression]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/expression
"""L0-L4 record-level verifier for microarray expression datasets (roadmap WS5).

Assembles the reusable checks in :mod:`torchcell.verification.levels` into a
:class:`VerificationReport` for the Sameith2015 / Kemmeren2014 microarray
expression datasets. Two expression-specific L3 conventions are asserted:

1. ``reference_log2_zero`` -- the reference phenotype is self-referential, so every
   ``expression_log2_ratio`` in it must be exactly 0.
2. ``deletion_downregulates`` -- the canonical orientation is
   ``log2(sample / reference)``; a strain with gene *G* deleted must show *G*
   down-regulated, so the MEDIAN log2 ratio at deleted genes must be < 0. A sign
   inversion (storing ``log2(reference / sample)``) would flip this positive. The
   median (not a fixed pass-fraction) is used because transcription-factor
   deletions (Sameith) are legitimately noisier than a clean deletion library
   (Kemmeren): both are correctly oriented, but their per-gene negative fractions
   differ (~0.72 vs ~0.97), while both medians are clearly negative.

The verifier operates purely on the pydantic/LMDB records -- no graph (Phase A).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from statistics import median
from typing import Any

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
Record = dict[str, Any]


def _deleted_genes(experiment: dict[str, Any]) -> list[str]:
    """Systematic names of the genes deleted in this experiment's genotype."""
    out: list[str] = []
    for pert in experiment["genotype"]["perturbations"]:
        name = pert.get("systematic_gene_name")
        if name is not None:
            out.append(name)
    return out


def _l1_gene_completeness(records: Sequence[Record]) -> LevelResult:
    """L1: no genes silently dropped -- every record measures the same gene set.

    The oracle is the union of measured genes across all records; a correct build
    measures the full platform gene set in every record, so any record short of the
    union has silently dropped genes.
    """
    per_record = [
        set(rec["experiment"]["phenotype"]["expression_log2_ratio"].keys())
        for rec in records
    ]
    universe: set[str] = set().union(*per_record) if per_record else set()
    short = [
        {"index": i, "n_missing": len(universe - genes)}
        for i, genes in enumerate(per_record)
        if genes != universe
    ]
    passed = not short
    return LevelResult(
        level=Level.L1,
        name="gene_completeness",
        passed=passed,
        message=(
            f"all {len(records)} records measure the full {len(universe)}-gene universe"
            if passed
            else f"{len(short)}/{len(records)} records missing genes vs the universe"
        ),
        details={
            "n_records": len(records),
            "gene_universe_size": len(universe),
            "n_short_records": len(short),
            "short": short[:20],
        },
    )


def _l3_reference_zero(records: Sequence[Record]) -> LevelResult:
    """L3: the reference phenotype is self-referential -> all log2 ratios == 0."""
    worst = 0.0
    n = 0
    for rec in records:
        ref_log2 = rec["reference"]["phenotype_reference"]["expression_log2_ratio"]
        for value in ref_log2.values():
            n += 1
            worst = max(worst, abs(float(value)))
    holds = worst == 0.0
    return l3_convention(
        "reference_log2_zero",
        holds,
        detail=(
            f"reference log2(sample/ref) == 0 for all {n} values"
            if holds
            else f"reference log2 not identically zero: max|value|={worst:.3g}"
        ),
    )


def _l3_deletion_orientation(records: Sequence[Record]) -> LevelResult:
    """L3: deletions down-regulate the deleted gene -> median deleted-gene log2 < 0."""
    deleted_log2: list[float] = []
    missing = 0
    for rec in records:
        exp = rec["experiment"]
        log2 = exp["phenotype"]["expression_log2_ratio"]
        for gene in _deleted_genes(exp):
            if gene in log2:
                deleted_log2.append(float(log2[gene]))
            else:
                missing += 1
    if not deleted_log2:
        return l3_convention(
            "deletion_downregulates",
            False,
            detail="no deleted genes were present in any expression map",
        )
    med = median(deleted_log2)
    n_neg = sum(1 for v in deleted_log2 if v < 0)
    frac_neg = n_neg / len(deleted_log2)
    holds = med < 0.0
    detail = (
        f"median deleted-gene log2={med:.3f} (<0 => correct orientation); "
        f"frac_neg={frac_neg:.3f} over {len(deleted_log2)} deleted genes "
        f"({missing} deleted genes absent from the platform map)"
    )
    return l3_convention("deletion_downregulates", holds, detail=detail)


def verify_expression_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
) -> VerificationReport:
    """Run the L0-L4 record-level gate for one microarray expression dataset.

    Args:
        records: the per-LMDB-entry dicts (experiment/reference/publication).
        dataset_name: dataset identity for the report.
        provenance: where the records came from.
        expected_count: the record-count oracle (e.g. Sameith DM=72, SM=82,
            Kemmeren=1450).

    Returns:
        A :class:`VerificationReport` carrying the L0-L3 results. L4 (cross-source)
        is asserted across datasets by the caller, not here.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)

    # L0: every experiment record validates against the schema union.
    report.add(l0_structural((rec["experiment"] for rec in records), validate))

    # L1: exact record count + no genes silently dropped per record.
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_gene_completeness(records))

    # L2: numeric fidelity of the stored statistics.
    log2_values = [
        float(v)
        for rec in records
        for v in rec["experiment"]["phenotype"]["expression_log2_ratio"].values()
    ]
    report.add(l2_value_fidelity(log2_values, allow_nan=False))

    se_values = [
        float(v)
        for rec in records
        for v in (
            rec["experiment"]["phenotype"].get("expression_log2_ratio_se") or {}
        ).values()
        if v is not None
    ]
    # SE may be NaN where a gene has a single replicate (SE undefined); never negative.
    se_result = l2_value_fidelity(se_values, allow_nan=True, minimum=0.0)
    report.add(
        LevelResult(
            level=Level.L2,
            name="se_nonnegative",
            passed=se_result.passed,
            message=se_result.message,
            details=se_result.details,
        )
    )

    nrep_values = [
        float(v)
        for rec in records
        for v in (rec["experiment"]["phenotype"].get("n_replicates") or {}).values()
    ]
    nrep_result = l2_value_fidelity(nrep_values, allow_nan=False, minimum=1.0)
    report.add(
        LevelResult(
            level=Level.L2,
            name="n_replicates_ge_1",
            passed=nrep_result.passed,
            message=nrep_result.message,
            details=nrep_result.details,
        )
    )

    # L3: expression-specific conventions.
    report.add(_l3_reference_zero(records))
    report.add(_l3_deletion_orientation(records))

    return report


def measured_gene_universe(records: Sequence[Record]) -> set[str]:
    """Union of all measured genes across records (the L4 cross-source key)."""
    universe: set[str] = set()
    for rec in records:
        universe.update(rec["experiment"]["phenotype"]["expression_log2_ratio"].keys())
    return universe


__all__ = ["verify_expression_dataset", "measured_gene_universe", "Record"]
