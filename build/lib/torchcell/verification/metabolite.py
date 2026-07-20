# torchcell/verification/metabolite
# [[torchcell.verification.metabolite]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/metabolite
"""L0-L4 record-level verifier for metabolite datasets (roadmap WS8; Cachera2023).

For `MetabolitePhenotype` datasets (e.g. the Cachera CRI-SPA betaxanthin screen). The
schema validator already enforces non-empty levels, matching level/replicate keys, and
non-negative SE, so L0 subsumes those. This verifier adds:

1. L1 `genotype_uniqueness` -- one record per strain (deletion-set signature; a single-
   deletion collection reduces to one gene per record, a combinatorial one keys on the set).
2. L2 `level_finiteness` -- measured levels are finite (guards inf/NaN).
3. L3 `reference_zero` -- the control level is 0 (population-centered baseline).
4. L3 `measurement_type_consistent` -- every record shares one measurement_type (levels
   from different assays must not be silently mixed).
5. L4 `gene_containment` (caller) -- the screened deletions overlap the yeast deletion
   collection used by the other datasets.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from torchcell.verification.levels import l0_structural, l1_count, l2_value_fidelity
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

Record = dict[str, Any]


def _deleted_genes(experiment: dict[str, Any]) -> list[str]:
    """Systematic names deleted in this experiment's genotype.

    Excludes ``gene_addition`` perturbations: the engineered cassette background is
    constant across every strain (and heterologous names are not real ORFs), so it is
    not a screened deletion and must not enter the L1/L4 deleted-ORF key.
    """
    return [
        p["systematic_gene_name"]
        for p in experiment["genotype"]["perturbations"]
        if p.get("systematic_gene_name") is not None
        and p.get("perturbation_type") != "gene_addition"
    ]


def _genotype_signature(
    experiment: dict[str, Any],
) -> tuple[tuple[str | None, ...], ...]:
    """Canonical STRAIN identity: the sorted set of deleted perturbations, each keyed by
    (systematic_gene_name, perturbation_type, perturbed_gene_name), excluding gene additions.

    A single-deletion collection reduces to one gene per record (so this equals per-ORF
    uniqueness); a COMBINATORIAL collection (e.g. Xue FFA: a POX1-FAA1-FAA4 baseline plus a
    set of TF deletions) has one record per unique deletion SET -- an individual ORF legitimately
    recurs across many strains, so uniqueness must key on the whole set, not the single ORF.
    """
    return tuple(
        sorted(
            (
                p.get("systematic_gene_name"),
                p.get("perturbation_type"),
                p.get("perturbed_gene_name"),
            )
            for p in experiment["genotype"]["perturbations"]
            if p.get("perturbation_type") != "gene_addition"
        )
    )


def _l1_orf_uniqueness(records: Sequence[Record]) -> LevelResult:
    """L1: exactly one record per STRAIN (genotype signature of deleted ORFs)."""
    seen: dict[tuple[tuple[str | None, ...], ...], int] = {}
    for rec in records:
        sig = _genotype_signature(rec["experiment"])
        seen[sig] = seen.get(sig, 0) + 1
    dups = {s: n for s, n in seen.items() if n > 1}
    return LevelResult(
        level=Level.L1,
        name="genotype_uniqueness",
        passed=not dups,
        message=(
            f"{len(seen)} unique strains (deletion sets), one record each"
            if not dups
            else f"{len(dups)} deletion sets appear in multiple records"
        ),
        details={"n_strains": len(seen), "n_duplicated": len(dups)},
    )


def _l3_reference_zero(records: Sequence[Record]) -> LevelResult:
    """L3: the reference (control) metabolite level is 0 (centered baseline)."""
    worst = 0.0
    n = 0
    for rec in records:
        levels = rec["reference"]["phenotype_reference"]["metabolite_level"]
        for v in levels.values():
            n += 1
            worst = max(worst, abs(float(v)))
    holds = worst == 0.0
    return LevelResult(
        level=Level.L3,
        name="reference_zero",
        passed=holds,
        message=(
            f"reference metabolite level == 0 for all {n} values"
            if holds
            else f"reference level not identically 0: max|v|={worst:.3g}"
        ),
        details={"n_values": n, "worst_abs": worst},
    )


def _l3_reference_finite(records: Sequence[Record]) -> LevelResult:
    """L3: the reference level is finite and defined for every measured metabolite.

    For ABSOLUTE-quantity datasets the reference is a WT-equivalent baseline, not a
    centered 0, so we check the reference is well-defined: non-empty, every value finite,
    and its metabolite keys are a SUBSET of the experiment's measured metabolites. Dense
    screens (e.g. Mulleder amino acids) key-match exactly (subset + equal cardinality);
    sparse targeted metabolomics (e.g. Zelezniak) reference each strain against the WT
    baseline only for the metabolites the WT measured, so the reference is a proper subset
    (a strain may measure metabolites the WT lacks, which then have no WT baseline).
    """
    n = 0
    bad = 0
    for rec in records:
        exp_keys = set(rec["experiment"]["phenotype"]["metabolite_level"])
        levels = rec["reference"]["phenotype_reference"]["metabolite_level"]
        if not levels or not set(levels) <= exp_keys:
            bad += 1
            continue
        for v in levels.values():
            n += 1
            if not math.isfinite(float(v)):
                bad += 1
    holds = bad == 0
    return LevelResult(
        level=Level.L3,
        name="reference_finite",
        passed=holds,
        message=(
            f"reference level finite + key-subset for all {n} values"
            if holds
            else f"{bad} reference levels non-finite, empty, or not a key-subset"
        ),
        details={"n_values": n, "n_bad": bad},
    )


def _l3_measurement_type_consistent(records: Sequence[Record]) -> LevelResult:
    """L3: all records share a single measurement_type (no silent cross-assay mixing)."""
    types = {rec["experiment"]["phenotype"]["measurement_type"] for rec in records}
    return LevelResult(
        level=Level.L3,
        name="measurement_type_consistent",
        passed=len(types) <= 1,
        message=(
            f"single measurement_type: {next(iter(types), None)!r}"
            if len(types) <= 1
            else f"{len(types)} distinct measurement_types mixed: {sorted(types)}"
        ),
        details={"measurement_types": sorted(types)},
    )


def verify_metabolite_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    reference_centered: bool = True,
) -> VerificationReport:
    """Run the L0-L3 record-level gate for a metabolite dataset.

    ``reference_centered`` (default True): assert the reference level is identically 0,
    for population-CENTERED scores (Cachera CRI-SPA). Set False for ABSOLUTE-quantity
    datasets (Mulleder amino-acid concentrations), where the reference is a WT-equivalent
    baseline and is instead checked for finiteness + key-consistency.

    L4 (cross-source gene overlap with the deletion collection) is asserted by the
    caller across datasets.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_orf_uniqueness(records))

    levels = [
        float(v)
        for rec in records
        for v in rec["experiment"]["phenotype"]["metabolite_level"].values()
    ]
    report.add(l2_value_fidelity(levels, allow_nan=False))

    # SE may be NaN where a metabolite has a single replicate; never negative.
    se_values = [
        float(v)
        for rec in records
        for v in (
            rec["experiment"]["phenotype"].get("metabolite_level_se") or {}
        ).values()
        if not (isinstance(v, float) and math.isnan(v))
    ]
    se_result = l2_value_fidelity(se_values, allow_nan=False, minimum=0.0)
    report.add(
        LevelResult(
            level=Level.L2,
            name="se_nonnegative",
            passed=se_result.passed,
            message=se_result.message,
            details=se_result.details,
        )
    )

    if reference_centered:
        report.add(_l3_reference_zero(records))
    else:
        report.add(_l3_reference_finite(records))
    report.add(_l3_measurement_type_consistent(records))
    return report


def metabolite_gene_set(records: Sequence[Record]) -> set[str]:
    """Union of deleted systematic gene names across records (the L4 overlap key)."""
    genes: set[str] = set()
    for rec in records:
        genes.update(_deleted_genes(rec["experiment"]))
    return genes


__all__ = ["verify_metabolite_dataset", "metabolite_gene_set", "Record"]
