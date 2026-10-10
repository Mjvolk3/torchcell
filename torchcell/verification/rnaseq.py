# torchcell/verification/rnaseq
# [[torchcell.verification.rnaseq]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/rnaseq
"""L0-L4 record-level verifier for RNA-seq expression datasets (roadmap WS10; Caudal2024).

For ``RNASeqExpressionPhenotype`` datasets (the Caudal natural-isolate pan-transcriptome),
where each isolate stores ABSOLUTE per-gene ``expression_tpm`` + ``expression_count`` on its
own genome. The schema validator already enforces non-empty, non-negative, key-matched
maps, so L0 subsumes those. This verifier adds:

1. L1 ``strain_uniqueness`` -- one record per isolate (each isolate's perturbations all
   carry one ``strain_id``; that id is unique across records). A compendium whose rows are
   sequenced LIBRARIES rather than isolates passes ``replicate_aware=True`` and gets
   ``replicate_groups`` instead: replicates of one condition share a genotype and an
   environment and carry no strain id, so the strain rule cannot be satisfied there and
   the group structure is what is checked.
2. L2 ``tpm_value_fidelity`` -- every TPM is finite and >= 0.
3. L2 ``count_value_fidelity`` -- every raw count is a non-negative integer.
4. L3 ``measurement_type_consistent`` -- one shared measurement_type (no cross-assay mix).
5. L3 ``reference_finite`` -- the shared population-mean reference TPMs are all finite.
6. L4 ``gene_containment`` (caller) -- the measured gene universe is contained in the S288C
   reference gene set.

The verifier operates purely on the pydantic/LMDB records -- no graph (Phase A).
"""

from __future__ import annotations

import json
import math
from collections import Counter
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


def _expr_map(phenotype: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return (per-gene expression dict, family kind) for either expression phenotype.

    ``RNASeqExpressionPhenotype`` (Caudal) stores absolute ``expression_tpm``;
    ``PseudobulkExpressionPhenotype`` (Nadal-Ribelles) stores per-gene log2 fold-change vs
    WT in ``expression_log2_ratio``; ``MrnaNumberFractionPhenotype`` (Balakrishnan 2022,
    issue #854) stores a count-less per-gene ``mrna_number_fraction``. One verifier
    serves all three families.
    """
    if "expression_log2_ratio" in phenotype:
        return phenotype["expression_log2_ratio"], "log2_ratio"
    if "mrna_number_fraction" in phenotype:
        return phenotype["mrna_number_fraction"], "number_fraction"
    return phenotype["expression_tpm"], "tpm"


def _env_identity(environment: dict[str, Any]) -> str:
    """Stable condition identity of an environment (media/temp/perturbations/duration)."""
    return json.dumps(environment, sort_keys=True, default=str)


def _record_strain(experiment: dict[str, Any]) -> str | None:
    """Return the strain id of a record (the shared ``strain_id`` of its perturbations)."""
    for pert in experiment["genotype"]["perturbations"]:
        strain = pert.get("strain_id")
        if strain is not None:
            return str(strain)
    return None


def _l1_strain_uniqueness(records: Sequence[Record]) -> LevelResult:
    """L1: one record per (strain, environment) and every record carries a strain id.

    Keyed on (strain_id, environment) so a strain profiled in two conditions (Nadal-Ribelles
    control vs NaCl) is two legitimate records, while a genome-scale single-condition survey
    (Caudal, one environment) still reduces to one record per strain.
    """
    seen: dict[tuple[str, str], int] = {}
    n_missing = 0
    for rec in records:
        exp = rec["experiment"]
        strain = _record_strain(exp)
        if strain is None:
            n_missing += 1
            continue
        key = (strain, _env_identity(exp["environment"]))
        seen[key] = seen.get(key, 0) + 1
    dups = {k: n for k, n in seen.items() if n > 1}
    n_strains = len({k[0] for k in seen})
    passed = not dups and n_missing == 0
    return LevelResult(
        level=Level.L1,
        name="strain_uniqueness",
        passed=passed,
        message=(
            f"{len(seen)} unique (strain, condition) records over {n_strains} strains"
            if passed
            else f"{len(dups)} (strain, condition) duplicated, "
            f"{n_missing} records without a strain"
        ),
        details={
            "n_records": len(seen),
            "n_strains": n_strains,
            "n_duplicated": len(dups),
            "n_missing": n_missing,
        },
    )


def _l1_replicate_groups(records: Sequence[Record]) -> LevelResult:
    """L1 for a REPLICATE-level compendium: one record per sequenced library.

    ``strain_uniqueness`` is the wrong rule for these datasets and cannot be satisfied by
    construction, so this is its replacement rather than a relaxation of it. Two reasons,
    both structural: a compendium releases one row per library, so the replicates of a
    condition share BOTH their genotype and their environment; and a wild-type record
    carries no perturbation at all, while the bacterial perturbation leaves carry no
    ``strain_id``, so there is no strain id to key on.

    What is checked instead is what replicate-level records can get wrong:

    1. **No library is counted twice.** Two records with the same expression profile are
       the same library read twice (the per-sample rule Lim 2022's own verifier uses).
       A genuine replicate differs in every gene's value, so an exact tie is a defect.
    2. **A replicate group measures one gene set.** Records sharing a (genotype,
       environment) are claimed to be replicates of one condition, so they must report
       the same genes; a group whose members measure different genes is two conditions
       pooled onto one environment identity.

    The group sizes are reported, since that is the structure a reader wants to see.
    """
    profiles: Counter[str] = Counter()
    groups: dict[str, set[str]] = {}
    for rec in records:
        exp = rec["experiment"]
        expr, _ = _expr_map(exp["phenotype"])
        profiles[json.dumps(expr, sort_keys=True)] += 1
        key = json.dumps(
            [exp["genotype"], exp["environment"]], sort_keys=True, default=str
        )
        groups.setdefault(key, set()).add(json.dumps(sorted(expr), sort_keys=True))
    n_repeated = sum(count for count in profiles.values() if count > 1)
    mixed = {key for key, gene_sets in groups.items() if len(gene_sets) > 1}
    sizes = Counter(
        sum(
            1
            for rec in records
            if json.dumps(
                [rec["experiment"]["genotype"], rec["experiment"]["environment"]],
                sort_keys=True,
                default=str,
            )
            == key
        )
        for key in groups
    )
    passed = n_repeated == 0 and not mixed
    return LevelResult(
        level=Level.L1,
        name="replicate_groups",
        passed=passed,
        message=(
            f"{len(records)} records with distinct profiles over {len(groups)} "
            "(genotype, environment) groups, each measuring one gene set"
            if passed
            else f"{n_repeated} records share an expression profile; {len(mixed)} "
            "groups pool records measuring different genes"
        ),
        details={
            "n_records": len(records),
            "n_groups": len(groups),
            "n_in_repeated_profiles": n_repeated,
            "n_groups_with_mixed_gene_sets": len(mixed),
            "group_size_histogram": {str(k): v for k, v in sorted(sizes.items())},
        },
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


def _l3_reference_finite(records: Sequence[Record]) -> LevelResult:
    """L3: every reference expression value is finite (baseline is well-defined)."""
    n = 0
    bad = 0
    for rec in records:
        levels, _ = _expr_map(rec["reference"]["phenotype_reference"])
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
            f"reference expression finite for all {n} values"
            if holds
            else f"{bad}/{n} reference expression values non-finite"
        ),
        details={"n_values": n, "n_bad": bad},
    )


def _l3_fraction_sum_at_most_one(records: Sequence[Record]) -> LevelResult:
    """L3: a record's number fractions, and its reference's, sum to at most 1.

    A record holds a subset of one transcriptome's genes, so its fractions cannot add to
    more than the whole; a sum above 1 means two libraries were pooled or a value was
    rescaled. The smallest sum is reported as the share of the transcriptome stored.
    """
    from torchcell.datamodels.schema import MRNA_NUMBER_FRACTION_SUM_ATOL

    sums = [
        sum(float(v) for v in phenotype["mrna_number_fraction"].values())
        for rec in records
        for phenotype in (
            rec["experiment"]["phenotype"],
            rec["reference"]["phenotype_reference"],
        )
    ]
    over = [s for s in sums if s > 1.0 + MRNA_NUMBER_FRACTION_SUM_ATOL]
    return LevelResult(
        level=Level.L3,
        name="fraction_sum_at_most_one",
        passed=bool(sums) and not over,
        message=(
            f"{len(sums)} stored profiles sum to {min(sums):.6f} .. {max(sums):.6f}"
            if sums
            else "no stored profiles"
        ),
        details={"n_profiles": len(sums), "n_over": len(over)},
    )


def verify_rnaseq_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    replicate_aware: bool = False,
) -> VerificationReport:
    """Run the L0-L3 record-level gate for an RNA-seq expression dataset.

    ``replicate_aware`` (default False) selects the L1 rule. False keys one record per
    (strain, condition) -- a one-record-per-isolate survey (Caudal) or a pseudobulk
    genotype x condition grid (Nadal-Ribelles). True is for a compendium that releases
    one row per sequenced LIBRARY (PRECISE-1K, putidaPRECISE321), where the replicates of
    a condition legitimately share a genotype and an environment and carry no strain id;
    the group rule (:func:`_l1_replicate_groups`) is what those records can get wrong.

    L4 (containment in the gene universe of the genome a record is written against) is
    asserted by the caller.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(
        _l1_replicate_groups(records)
        if replicate_aware
        else _l1_strain_uniqueness(records)
    )

    kind = _expr_map(records[0]["experiment"]["phenotype"])[1] if records else "tpm"
    expr_values = [
        float(v)
        for rec in records
        for v in _expr_map(rec["experiment"]["phenotype"])[0].values()
    ]
    if kind == "tpm":
        # Absolute TPM: finite and non-negative.
        fidelity = l2_value_fidelity(expr_values, allow_nan=False, minimum=0.0)
        fidelity_name = "tpm_value_fidelity"
    elif kind == "number_fraction":
        # A fraction of one transcriptome's mRNA molecules: finite and in [0, 1].
        fidelity = l2_value_fidelity(
            expr_values, allow_nan=False, minimum=0.0, maximum=1.0
        )
        fidelity_name = "number_fraction_value_fidelity"
    else:
        # Pseudobulk log2 fold-change vs WT: finite, negatives allowed (down-regulation).
        fidelity = l2_value_fidelity(expr_values, allow_nan=False)
        fidelity_name = "log2_ratio_value_fidelity"
    report.add(
        LevelResult(
            level=Level.L2,
            name=fidelity_name,
            passed=fidelity.passed,
            message=fidelity.message,
            details=fidelity.details,
        )
    )

    # Raw integer counts exist only for the absolute-TPM family.
    if kind == "tpm":
        count_bad = 0
        count_n = 0
        for rec in records:
            for v in rec["experiment"]["phenotype"]["expression_count"].values():
                count_n += 1
                if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                    count_bad += 1
        report.add(
            LevelResult(
                level=Level.L2,
                name="count_value_fidelity",
                passed=count_bad == 0,
                message=(
                    f"{count_n} counts are non-negative integers"
                    if count_bad == 0
                    else f"{count_bad}/{count_n} counts are not non-negative integers"
                ),
                details={"n_values": count_n, "n_bad": count_bad},
            )
        )

    if kind == "number_fraction":
        report.add(_l3_fraction_sum_at_most_one(records))

    report.add(_l3_measurement_type_consistent(records))
    report.add(_l3_reference_finite(records))
    return report


def rnaseq_gene_set(records: Sequence[Record]) -> set[str]:
    """Union of measured expression genes across records (the L4 containment key)."""
    genes: set[str] = set()
    for rec in records:
        expr, _ = _expr_map(rec["experiment"]["phenotype"])
        genes.update(expr.keys())
    return genes


__all__ = ["verify_rnaseq_dataset", "rnaseq_gene_set", "Record"]
