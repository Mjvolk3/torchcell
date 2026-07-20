# torchcell/verification/visual_score
# [[torchcell.verification.visual_score]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/visual_score
"""L0-L4 record-level verifier for visual-score datasets (roadmap WS7; Ozaydin2013).

For `VisualScorePhenotype` datasets (e.g. the Ozaydin carotenoid colony-color screen).
The schema's `validate_visual_score` validator already enforces, at instantiation, that
`visual_score` lies within `[score_scale_min, score_scale_max]` and `n_replicates >= 1`,
so L0 subsumes those. This verifier adds what the schema does NOT encode:

1. L1 `orf_uniqueness` -- one record per deleted ORF (no duplicate strains).
2. L2 `score_finiteness` -- scores finite (schema bounds them but this guards inf/NaN
   defensively across serialization).
3. L3 `reference_zero` -- the control (WT carrying the reporter background) is scored 0.
4. L3 `target_product_set` -- every record names the metabolite the score proxies for.
5. L4 `gene_containment` (caller) -- the screened deletions overlap the yeast deletion
   collection used by the other datasets (same library, consistent naming).
"""

from __future__ import annotations

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


def _l1_orf_uniqueness(records: Sequence[Record]) -> LevelResult:
    """L1: exactly one record per deleted ORF (no duplicate strains)."""
    seen: dict[str, int] = {}
    for rec in records:
        for g in _deleted_genes(rec["experiment"]):
            seen[g] = seen.get(g, 0) + 1
    dups = {g: n for g, n in seen.items() if n > 1}
    return LevelResult(
        level=Level.L1,
        name="orf_uniqueness",
        passed=not dups,
        message=(
            f"{len(seen)} unique deleted ORFs, one record each"
            if not dups
            else f"{len(dups)} ORFs appear in multiple records"
        ),
        details={"n_orfs": len(seen), "n_duplicated": len(dups)},
    )


def _l3_reference_zero(records: Sequence[Record]) -> LevelResult:
    """L3: the reference (control) visual score is 0 by construction."""
    worst = 0.0
    n = 0
    for rec in records:
        n += 1
        worst = max(
            worst, abs(float(rec["reference"]["phenotype_reference"]["visual_score"]))
        )
    holds = worst == 0.0
    return LevelResult(
        level=Level.L3,
        name="reference_zero",
        passed=holds,
        message=(
            f"reference visual_score == 0 for all {n} records"
            if holds
            else f"reference visual_score not identically 0: max|v|={worst:.3g}"
        ),
        details={"n_records": n, "worst_abs": worst},
    )


def _l3_target_product_set(records: Sequence[Record]) -> LevelResult:
    """L3: every record names the target product the score is a proxy for."""
    missing = sum(
        1 for rec in records if not rec["experiment"]["phenotype"].get("target_product")
    )
    return LevelResult(
        level=Level.L3,
        name="target_product_set",
        passed=missing == 0,
        message=(
            "every record names a target_product"
            if missing == 0
            else f"{missing} records missing target_product"
        ),
        details={"n_missing": missing},
    )


def verify_visual_score_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
) -> VerificationReport:
    """Run the L0-L3 record-level gate for a visual-score dataset.

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

    scores = [float(rec["experiment"]["phenotype"]["visual_score"]) for rec in records]
    report.add(l2_value_fidelity(scores, allow_nan=False))

    report.add(_l3_reference_zero(records))
    report.add(_l3_target_product_set(records))
    return report


def visual_score_gene_set(records: Sequence[Record]) -> set[str]:
    """Union of deleted systematic gene names across records (the L4 overlap key)."""
    genes: set[str] = set()
    for rec in records:
        genes.update(_deleted_genes(rec["experiment"]))
    return genes


__all__ = ["verify_visual_score_dataset", "visual_score_gene_set", "Record"]
