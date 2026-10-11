# torchcell/verification/gene_interaction
# [[torchcell.verification.gene_interaction]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/gene_interaction
"""L0-L4 record-level verifier for the yeast SGA gene-interaction stores (#889).

A ``GeneInteractionExperiment`` record holds one interaction score (Costanzo 2016's
digenic epsilon, Kuzmin 2018/2020's digenic epsilon or trigenic tau) and its p-value for
one (strain, environment) pair, against a reference whose score is 0. The battery is
single-pass, because ``dmi_costanzo2016`` holds 20,705,612 records:

1. L0 ``structural`` / ``reference_structural`` -- every experiment and reference
   validates as the ``GeneInteractionExperiment`` / ``...Reference`` the registry
   declares.
2. L1 ``count`` -- the registry's record-count oracle.
3. L1 ``pair_uniqueness`` -- one record per (screened STRAIN, environment), with the
   fitness family's signatures, so a screen label (``screen_id``) or an allele's strain
   id keeps two measurements apart exactly as it does for fitness.
4. L1 ``interaction_order`` -- every record perturbs ``order`` distinct genes (2 for a
   digenic store, 3 for a trigenic one) and its phenotype sits at the graph level that
   order names (``edge`` / ``hyperedge``): the digenic/trigenic distinction.
5. L2 ``score_finite`` / ``p_value_in_unit_interval`` -- every score is a finite
   number and every stated p-value lies in [0, 1].
6. L2 ``released_values`` -- the stored (score, p-value) multiset IS the released
   table's, read from the sha256-pinned files by
   :func:`torchcell.verification.released.interaction_values_from_table`: same count,
   and the same values position by position once both are sorted.
7. L3 ``reference_zero`` -- every reference score is exactly 0 with no p-value, at the
   experiment's graph level; L3 ``reference_environment`` -- every reference's
   environment is the experiment's own; L3 ``signed_unclamped`` -- the store holds both
   aggravating and alleviating scores (the releases are unfiltered).
8. L3 ``provenance_audit`` (one per quote) -- the sentences that define the table and
   its score column are still verbatim in the pinned library bytes.
9. L4 ``fitness_companion_containment`` -- every interaction record's (strain,
   environment) key is a record of the companion fitness store built from the same
   table (DMI in DMF, TMI in TMF).

It also runs :class:`torchcell.verification.common.SharedRecordRules` (gap census,
canonical gene names, uncertainty sanity, compound identity, media membership and, with
a gene universe, the two L4 gene rules).
"""

from __future__ import annotations

import math
from array import array
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from torchcell.verification.common import (
    GeneNameResolver,
    SharedRecordRules,
    declared_member_validator,
    key_digest,
    l0_validated_row,
)
from torchcell.verification.fitness import _environment_signature, _genotype_signature
from torchcell.verification.levels import l1_count
from torchcell.verification.released import InteractionValues, sorted_value_mismatches
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

__all__ = [
    "GRAPH_LEVEL_OF_ORDER",
    "Record",
    "ReleasedComparison",
    "pair_key",
    "released_values_result",
    "verify_gene_interaction_dataset",
]

Record = Mapping[str, Any]

#: The phenotype graph level each interaction order is stored at.
GRAPH_LEVEL_OF_ORDER: dict[int, str] = {2: "edge", 3: "hyperedge"}


def pair_key(experiment: Mapping[str, Any]) -> bytes:
    """The digest of a record's (strain, environment) key, shared with the fitness family.

    The interaction store and its companion fitness store are keyed identically, so this
    is the key :func:`verify_gene_interaction_dataset` checks for uniqueness and the key
    the companion containment rule looks up.
    """
    return key_digest(
        (
            key_digest(_genotype_signature(dict(experiment))),
            key_digest(_environment_signature(dict(experiment))),
        )
    )


class ReleasedComparison:
    """What the caller read from the pinned release: its values, or why it could not.

    ``drift`` names every pinned file whose sha256 changed; when it is non-empty the
    values are not read and the rule fails on the drift alone.
    """

    def __init__(
        self,
        *,
        files: Sequence[str],
        drift: Mapping[str, str],
        values: InteractionValues | None,
        description: str,
    ) -> None:
        """Hold the pinned file names, their drift, and the values read when none drifted."""
        if not drift and values is None:
            raise ValueError("a release with no drift must carry its values")
        self.files = list(files)
        self.drift = dict(drift)
        self.values = values
        self.description = description


def released_values_result(
    stored: InteractionValues, released: ReleasedComparison
) -> LevelResult:
    """L2 ``released_values``: the stored (score, p-value) multiset is the release's."""
    details: dict[str, Any] = {
        "files": released.files,
        "selection": released.description,
        "n_stored": stored.n_rows,
    }
    if released.drift:
        return LevelResult(
            level=Level.L2,
            name="released_values",
            passed=False,
            message=f"sha256 drift in {sorted(released.drift)}: the release was not read",
            details={**details, "drift": released.drift},
        )
    values = released.values
    assert values is not None  # guaranteed by ReleasedComparison
    details["n_released"] = values.n_rows
    if values.n_rows != stored.n_rows:
        return LevelResult(
            level=Level.L2,
            name="released_values",
            passed=False,
            message=f"{stored.n_rows} stored scores against {values.n_rows} released "
            f"rows ({released.description})",
            details=details,
        )
    n_differ, examples = sorted_value_mismatches(stored, values)
    return LevelResult(
        level=Level.L2,
        name="released_values",
        passed=n_differ == 0,
        message=(
            f"the {stored.n_rows} stored (score, p-value) pairs are the released "
            f"multiset ({released.description})"
            if n_differ == 0
            else f"{n_differ} of {stored.n_rows} sorted (score, p-value) positions "
            f"differ from the release ({released.description})"
        ),
        details={**details, "n_differing_positions": n_differ, "examples": examples},
    )


def verify_gene_interaction_dataset(
    records: Iterable[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    order: int,
    released: ReleasedComparison,
    companion_keys: set[bytes] | None,
    companion_name: str | None,
    extra_results: Sequence[LevelResult] = (),
    resolve_gene_name: GeneNameResolver | None = None,
    sgd_genes: set[str] | None = None,
    gene_universe_label: str = "reference",
    min_containment: float = 0.90,
) -> VerificationReport:
    """Run the L0-L4 gate over one gene-interaction store in a single pass.

    ``companion_keys`` is the set of :func:`pair_key` digests of the companion fitness
    store; None means the store has no companion and the L4 row is omitted (the caller
    says why in its registry). ``extra_results`` are rows the caller computed outside
    the pass, the provenance audits of the quotes that define the table.
    """
    if order not in GRAPH_LEVEL_OF_ORDER:
        raise ValueError(f"interaction order {order} is not 2 or 3")
    graph_level = GRAPH_LEVEL_OF_ORDER[order]
    validate = declared_member_validator("GeneInteractionExperiment")
    validate_reference = declared_member_validator(
        "GeneInteractionExperimentReference", union="ExperimentReferenceType"
    )
    shared = SharedRecordRules(
        resolve_gene_name=resolve_gene_name,
        sgd_genes=sgd_genes,
        gene_universe_label=gene_universe_label,
        min_containment=min_containment,
    )
    n_records = 0
    l0_failures: list[dict[str, Any]] = []
    ref_failures: list[dict[str, Any]] = []
    pair_counts: Counter[bytes] = Counter()
    wrong_order: Counter[str] = Counter()
    wrong_order_examples: list[dict[str, Any]] = []
    scores = array("d")
    p_values = array("d")
    bad_scores: list[dict[str, Any]] = []
    bad_p: list[dict[str, Any]] = []
    n_p = 0
    n_negative = n_positive = n_zero = 0
    reference_scores: Counter[float] = Counter()
    n_reference_with_p = 0
    n_reference_wrong_level = 0
    n_reference_env_differs = 0
    reference_env_examples: list[int] = []

    for i, rec in enumerate(records):
        n_records += 1
        exp = rec["experiment"]
        ref = rec["reference"]
        shared.add(rec)
        try:
            validate(exp)
        except (ValueError, TypeError) as err:
            l0_failures.append({"index": i, "error": str(err)[:500]})
        try:
            validate_reference(ref)
        except (ValueError, TypeError) as err:
            ref_failures.append({"index": i, "error": str(err)[:500]})

        pair_counts[pair_key(exp)] += 1

        perturbations = exp["genotype"]["perturbations"]
        genes = {p.get("systematic_gene_name") for p in perturbations}
        phenotype = exp["phenotype"]
        problem = None
        if len(perturbations) != order:
            problem = f"{len(perturbations)} perturbations"
        elif len(genes) != order or None in genes:
            problem = f"{len(genes - {None})} distinct genes"
        elif phenotype.get("graph_level") != graph_level:
            problem = f"graph_level {phenotype.get('graph_level')!r}"
        if problem is not None:
            wrong_order[problem] += 1
            if len(wrong_order_examples) < 20:
                wrong_order_examples.append(
                    {
                        "index": i,
                        "problem": problem,
                        "genes": sorted(str(g) for g in genes),
                    }
                )

        score = phenotype["gene_interaction"]
        p_value = phenotype.get("gene_interaction_p_value")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            bad_scores.append({"index": i, "value": repr(score)})
            scores.append(math.nan)
        else:
            scores.append(float(score))
            if not math.isfinite(score):
                bad_scores.append({"index": i, "value": repr(score)})
            elif score < 0:
                n_negative += 1
            elif score > 0:
                n_positive += 1
            else:
                n_zero += 1
        if p_value is None:
            p_values.append(math.nan)
        else:
            n_p += 1
            p_values.append(float(p_value))
            if not (0.0 <= float(p_value) <= 1.0):
                bad_p.append({"index": i, "value": repr(p_value)})

        ref_phenotype = ref["phenotype_reference"]
        reference_scores[float(ref_phenotype["gene_interaction"])] += 1
        if ref_phenotype.get("gene_interaction_p_value") is not None:
            n_reference_with_p += 1
        if ref_phenotype.get("graph_level") != phenotype.get("graph_level"):
            n_reference_wrong_level += 1
        if ref["environment_reference"] != exp["environment"]:
            n_reference_env_differs += 1
            if len(reference_env_examples) < 20:
                reference_env_examples.append(i)

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(
        l0_validated_row(
            "structural", n_records, l0_failures, "GeneInteractionExperiment"
        )
    )
    report.add(
        l0_validated_row(
            "reference_structural",
            n_records,
            ref_failures,
            "GeneInteractionExperimentReference",
        )
    )
    report.add(l1_count(n_records, expected_count))
    n_duplicated = sum(1 for n in pair_counts.values() if n > 1)
    report.add(
        LevelResult(
            level=Level.L1,
            name="pair_uniqueness",
            passed=n_records > 0 and not n_duplicated,
            message=(
                f"{len(pair_counts)} unique (strain, environment) records, one each"
                if not n_duplicated
                else f"{n_duplicated} (strain, environment) pairs appear in multiple "
                "records"
            ),
            details={
                "n_pairs": len(pair_counts),
                "n_duplicated": n_duplicated,
                "n_extra_records": sum(n - 1 for n in pair_counts.values() if n > 1),
            },
        )
    )
    n_wrong_order = sum(wrong_order.values())
    report.add(
        LevelResult(
            level=Level.L1,
            name="interaction_order",
            passed=n_records > 0 and not n_wrong_order,
            message=(
                f"all {n_records} records perturb {order} distinct genes at graph level "
                f"{graph_level!r}"
                if not n_wrong_order
                else f"{n_wrong_order} of {n_records} records are not a {order}-gene "
                f"{graph_level}: {dict(wrong_order.most_common(5))}"
            ),
            details={
                "order": order,
                "graph_level": graph_level,
                "n_wrong": n_wrong_order,
                "by_problem": dict(wrong_order),
                "examples": wrong_order_examples,
            },
        )
    )
    report.add(
        LevelResult(
            level=Level.L2,
            name="score_finite",
            passed=not bad_scores,
            message=(
                f"{n_records} interaction scores are finite numbers"
                if not bad_scores
                else f"{len(bad_scores)}/{n_records} interaction scores are not finite"
            ),
            details={
                "n_values": n_records,
                "n_bad": len(bad_scores),
                "bad": bad_scores[:20],
            },
        )
    )
    report.add(
        LevelResult(
            level=Level.L2,
            name="p_value_in_unit_interval",
            passed=not bad_p,
            message=(
                f"{n_p} stated p-values lie in [0, 1] ({n_records - n_p} records state "
                "none)"
                if not bad_p
                else f"{len(bad_p)}/{n_p} stated p-values lie outside [0, 1]"
            ),
            details={
                "n_p_values": n_p,
                "n_without_p_value": n_records - n_p,
                "n_bad": len(bad_p),
                "bad": bad_p[:20],
            },
        )
    )
    stored = InteractionValues(
        scores=np.frombuffer(scores, dtype=np.float64),
        p_values=np.frombuffer(p_values, dtype=np.float64),
    )
    report.add(released_values_result(stored, released))
    zero_reference = (
        set(reference_scores) == {0.0}
        and n_reference_with_p == 0
        and n_reference_wrong_level == 0
    )
    report.add(
        LevelResult(
            level=Level.L3,
            name="reference_zero",
            passed=n_records > 0 and zero_reference,
            message=(
                f"every one of the {n_records} reference scores is 0 with no p-value, at "
                "the experiment's graph level"
                if zero_reference
                else f"reference scores {dict(reference_scores.most_common(5))}; "
                f"{n_reference_with_p} carry a p-value; {n_reference_wrong_level} sit at "
                "another graph level"
            ),
            details={
                "reference_scores": {
                    str(k): v for k, v in reference_scores.most_common(10)
                },
                "n_reference_with_p_value": n_reference_with_p,
                "n_reference_wrong_graph_level": n_reference_wrong_level,
            },
        )
    )
    report.add(
        LevelResult(
            level=Level.L3,
            name="reference_environment",
            passed=n_records > 0 and n_reference_env_differs == 0,
            message=(
                f"every one of the {n_records} references is measured in its "
                "experiment's own environment"
                if n_reference_env_differs == 0
                else f"{n_reference_env_differs} of {n_records} references sit in "
                "another environment than their experiment"
            ),
            details={
                "n_differ": n_reference_env_differs,
                "example_indices": reference_env_examples,
            },
        )
    )
    report.add(
        LevelResult(
            level=Level.L3,
            name="signed_unclamped",
            passed=n_negative > 0 and n_positive > 0,
            message=f"{n_negative} aggravating and {n_positive} alleviating scores, "
            f"{n_zero} at exactly zero",
            details={
                "n_negative": n_negative,
                "n_positive": n_positive,
                "n_zero": n_zero,
            },
        )
    )
    for result in extra_results:
        report.add(result)
    for result in shared.results():
        report.add(result)
    if companion_keys is not None:
        missing = sum(1 for key in pair_counts if key not in companion_keys)
        report.add(
            LevelResult(
                level=Level.L4,
                name="fitness_companion_containment",
                passed=n_records > 0 and missing == 0,
                message=(
                    f"every one of the {len(pair_counts)} (strain, environment) keys is a "
                    f"record of {companion_name}"
                    if missing == 0
                    else f"{missing} of {len(pair_counts)} (strain, environment) keys "
                    f"have no record in {companion_name}"
                ),
                details={
                    "companion": companion_name,
                    "n_keys": len(pair_counts),
                    "n_companion_keys": len(companion_keys),
                    "n_missing": missing,
                },
            )
        )
    return report
