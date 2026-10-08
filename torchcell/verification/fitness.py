# torchcell/verification/fitness
# [[torchcell.verification.fitness]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/fitness
"""L0-L4 record-level verifier for single-mutant fitness datasets.

For ``FitnessPhenotype`` datasets: a deletion/allele strain's growth relative to wild-type
(``fitness``, with WT == 1.0). The schema validator already enforces the uncertainty
invariant (subsumed by L0); this verifier adds:

1. L1 ``count`` -- exact record-count oracle.
2. L1 ``pair_uniqueness`` -- one record per (screened STRAIN x environment) pair (the strain
   is the full genotype signature, so an allelic series is distinct, not duplicate; the
   environment includes its typed EDITS, so one strain grown on thirty carbon sources is
   thirty records, not one repeated thirty times).
3. L2 ``value_fidelity`` -- fitness values are finite and non-negative (WT == 1, sick < 1).
4. L2 ``se_nonnegative`` -- reported fitness SEs are non-negative.
5. L3 ``reference_one`` -- the reference (wild-type) fitness is 1.0 (the convention baseline).
6. L4 ``gene_containment`` -- screened deletions overlap the S288C gene universe.

It also runs :class:`torchcell.verification.common.SharedRecordRules`, the rules that are
not specific to a readout: the gap + silent-None census over every carrier, canonical gene
names, uncertainty sanity, compound identity, media membership, and (when the caller
supplies the gene universe) the two L4 gene rules.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from torchcell.verification.common import GeneNameResolver, SharedRecordRules
from torchcell.verification.levels import l0_structural, l1_count, l2_value_fidelity
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

Record = dict[str, Any]


def _genotype_signature(
    experiment: dict[str, Any],
) -> tuple[tuple[str | None, ...], ...]:
    """Canonical STRAIN identity: the sorted set of perturbations, each keyed by
    ``(systematic_gene_name, perturbation_type, perturbed_gene_name, strain_id)`` plus
    the guide-library discriminators of a CRISPR construct.

    ``strain_id`` (present on SGA perturbation variants, else None) distinguishes an allelic
    SERIES that shares gene + type -- e.g. Baryshnikova 2010 has 58 genes with more than one
    temperature-sensitive allele (YAL041W x4); without it they collide as duplicates. It is
    None for non-SGA perturbations, so adding it only ever REFINES the key -- pre-existing
    fitness datasets keep their (already-unique) signatures.

    The CRISPR fields are here for the same reason and on the same terms as in
    :func:`environment_response._genotype_signature`: a guide library has MANY strains per
    (gene, mode), so the spacer (``crispr.guide_sequence``) is part of the strain, and the
    SAME spacer screened in two library pools is two pool-relative measurements, so the
    pool (``crispr.library_pool``) is too. Both are ``None`` on a non-CRISPR leaf and on a
    construct that names neither, which leaves every pre-existing key unchanged; a field
    added to this tuple can only SPLIT a group, never merge two.
    """

    def _identity(p: dict[str, Any]) -> tuple[str | None, ...]:
        payload = p.get("crispr")
        crispr: dict[str, Any] = payload if isinstance(payload, dict) else {}
        return (
            p.get("systematic_gene_name"),
            p.get("perturbation_type"),
            p.get("perturbed_gene_name"),
            p.get("strain_id"),
            crispr.get("guide_sequence"),
            crispr.get("library_pool"),
        )

    return tuple(sorted(_identity(p) for p in experiment["genotype"]["perturbations"]))


def _environment_signature(experiment: dict[str, Any]) -> tuple[Any, ...]:
    """Canonical environment identity: the environment EDITS plus the scalars.

    The edits come first because they are what distinguishes most conditions of a
    fitness screen that varies its environment: Tong 2020 grows every deletion strain on
    ONE medium (solid MOPS minimal) at one temperature for one duration and varies only
    the carbon source, which rides on ``environment.perturbations``. Keyed on the
    scalars alone, its 3,644 Keio strains each read as thirty duplicates of one record,
    and L1 fails on a dataset that holds exactly one record per (strain, carbon source).

    Each edit is keyed by ``(perturbation_type, compound or agent name, factor, dose
    value, unit, basis)``, the same tuple :func:`environment_response._condition_signature`
    uses, so the two verifiers agree on what makes two conditions different. Elements are
    stringified for a total order (mixed None / str / float never breaks the sort).

    A study that screened the SAME strains in the SAME medium at the SAME dose twice
    measured two conditions, not one: the screens are normalized independently, so the
    phenotype's ``screen_id`` joins the signature, exactly as it does in
    :func:`environment_response._condition_signature`. Rachwalski 2024 releases its
    whole CRISPRi collection on MOPS minimal at six doses in BOTH Table S2A and Table
    S3, and none of the 2,262 overlapping cells agree. ``.get`` keeps the signature
    identical for every dataset that carries no screen label.

    A yeast fitness dataset carries no environment perturbations, so its key gains an
    empty tuple and its (already unique) signatures are unchanged.
    """
    env = experiment["environment"]
    perturbations: list[tuple[str, ...]] = []
    for perturbation in env.get("perturbations") or []:
        compound = (
            perturbation.get("compound")
            if isinstance(perturbation.get("compound"), dict)
            else {}
        )
        agent = (
            perturbation.get("agent")
            if isinstance(perturbation.get("agent"), dict)
            else {}
        )
        dose = perturbation.get("concentration") or perturbation.get("magnitude") or {}
        dose = dose if isinstance(dose, dict) else {}
        fields = (
            perturbation.get("perturbation_type"),
            compound.get("name") or agent.get("name"),
            perturbation.get("factor"),
            dose.get("value"),
            dose.get("unit"),
            dose.get("basis"),
        )
        perturbations.append(tuple("" if v is None else str(v) for v in fields))
    temp = (env.get("temperature") or {}).get("value")
    media = (env.get("media") or {}).get("name")
    return (
        tuple(sorted(perturbations)),
        temp,
        media,
        env.get("duration_hours"),
        env.get("duration_generations"),
        experiment["phenotype"].get("screen_id"),
    )


def _l1_pair_uniqueness(records: Sequence[Record]) -> LevelResult:
    """L1: exactly one record per (screened STRAIN, environment) pair."""
    seen: dict[tuple[Any, ...], int] = {}
    for rec in records:
        exp = rec["experiment"]
        key = (_genotype_signature(exp), _environment_signature(exp))
        seen[key] = seen.get(key, 0) + 1
    dups = {k: n for k, n in seen.items() if n > 1}
    return LevelResult(
        level=Level.L1,
        name="pair_uniqueness",
        passed=not dups,
        message=(
            f"{len(seen)} unique (strain, environment) records, one each"
            if not dups
            else f"{len(dups)} (strain, environment) pairs appear in multiple records"
        ),
        details={
            "n_pairs": len(seen),
            "n_duplicated": len(dups),
            "n_strains": len({genotype for genotype, _ in seen}),
            "n_environments": len({environment for _, environment in seen}),
        },
    )


def _l3_reference_one(records: Sequence[Record]) -> LevelResult:
    """L3: the reference (wild-type) fitness is 1.0 -- the WT-normalized convention."""
    worst = 0.0
    n = 0
    for rec in records:
        v = rec["reference"]["phenotype_reference"]["fitness"]
        if v is None:
            continue
        n += 1
        worst = max(worst, abs(float(v) - 1.0))
    holds = worst == 0.0
    return LevelResult(
        level=Level.L3,
        name="reference_one",
        passed=holds,
        message=(
            f"reference fitness == 1.0 for all {n} records"
            if holds
            else f"reference fitness not identically 1.0: max|v-1|={worst:.3g}"
        ),
        details={"n_values": n, "worst_abs_dev": worst},
    )


def verify_fitness_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    resolve_gene_name: GeneNameResolver | None = None,
    sgd_genes: set[str] | None = None,
    gene_universe_label: str = "reference",
    min_containment: float = 0.90,
) -> VerificationReport:
    """Run the L0-L4 record-level gate for a single-mutant fitness dataset.

    ``sgd_genes`` turns on the L4 gene rules (aggregate containment + per-record genome
    membership), ``gene_universe_label`` is what the containment row calls that universe
    (the host's, not always S288C's), and ``resolve_gene_name`` the annotation half of the
    canonical-name rule; without them the caller owns L4 (:func:`fitness_gene_set` is the
    overlap key).
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_pair_uniqueness(records))

    fitness_values = [
        float(rec["experiment"]["phenotype"]["fitness"])
        for rec in records
        if rec["experiment"]["phenotype"]["fitness"] is not None
    ]
    report.add(l2_value_fidelity(fitness_values, allow_nan=False, minimum=0.0))

    se_values = [
        float(v)
        for rec in records
        if (v := rec["experiment"]["phenotype"].get("fitness_se")) is not None
        and not (isinstance(v, float) and math.isnan(v))
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

    report.add(_l3_reference_one(records))

    shared = SharedRecordRules(
        resolve_gene_name=resolve_gene_name,
        sgd_genes=sgd_genes,
        gene_universe_label=gene_universe_label,
        min_containment=min_containment,
    )
    shared.add_all(records)
    for result in shared.results():
        report.add(result)
    return report


def fitness_gene_set(records: Sequence[Record]) -> set[str]:
    """Union of screened deleted gene names -- the L4 gene-containment key."""
    genes: set[str] = set()
    for rec in records:
        for p in rec["experiment"]["genotype"]["perturbations"]:
            name = p.get("systematic_gene_name")
            if name is not None:
                genes.add(name)
    return genes


__all__ = ["verify_fitness_dataset", "fitness_gene_set", "Record"]
