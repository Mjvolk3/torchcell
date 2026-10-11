# torchcell/verification/gene_essentiality
# [[torchcell.verification.gene_essentiality]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/gene_essentiality
"""L0-L4 record-level verifier for the yeast gene-essentiality store (#889).

``gene_essentiality_sgd`` stores one ``GeneEssentialityExperiment`` per SGD phenotype
annotation that reads "null" mutant, strain S288C, phenotype "inviable": a single-gene
deletion, ``is_essential`` True, against a viable (``is_essential`` False) reference.
The phenotype is a boolean with no uncertainty, so the battery is mostly structural,
counting and containment:

1. L0 ``structural`` / ``reference_structural`` -- every experiment and reference
   validates as the declared ``GeneEssentialityExperiment`` / ``...Reference``.
2. L1 ``count`` -- the registry's record-count oracle.
3. L1 ``single_gene`` -- every record perturbs exactly one named gene.
4. L1 ``one_record_per_gene_and_publication`` -- no two records share a gene, an
   environment AND a citing publication: two annotations SGD carries for one gene from
   two papers are two records, while two records identical down to the publication
   describe one fact twice.
5. L2 ``essential_label`` -- every experiment says ``is_essential`` True.
6. L2 ``released_annotations`` -- the stored (gene, PubMed id) multiset is the
   multiset of matching annotations in the pinned SGD per-gene JSON release
   (:func:`torchcell.verification.released.sgd_inviable_null_annotations`).
7. L3 ``reference_viable`` -- every reference says ``is_essential`` False, in the
   experiment's own environment.
8. Shared rules, and the two L4 gene rules against the S288C universe.

The caller adds any cross-dataset L4 rows (``extra_results``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from torchcell.verification.common import (
    GeneNameResolver,
    SharedRecordRules,
    declared_member_validator,
    key_digest,
    l0_validated_row,
)
from torchcell.verification.fitness import _environment_signature, _genotype_signature
from torchcell.verification.levels import l1_count
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

__all__ = [
    "Record",
    "essential_gene_set",
    "essential_without_viable_deletion_result",
    "released_annotations_result",
    "verify_gene_essentiality_dataset",
]

Record = Mapping[str, Any]


def released_annotations_result(
    stored: Counter[tuple[str, str]],
    released: Counter[tuple[str, str]] | None,
    *,
    pinned_digest: str,
    observed_digest: str,
) -> LevelResult:
    """L2 ``released_annotations``: the stored (gene, PubMed) multiset is SGD's.

    ``released`` is None when the per-gene JSON digest drifted from its pin; the rule
    then fails on the drift alone and the annotations are not compared.
    """
    details: dict[str, Any] = {
        "pinned_digest": pinned_digest,
        "observed_digest": observed_digest,
        "n_stored": sum(stored.values()),
    }
    if released is None:
        return LevelResult(
            level=Level.L2,
            name="released_annotations",
            passed=False,
            message="the SGD per-gene JSON release drifted from its pin: not compared",
            details=details,
        )
    only_stored = stored - released
    only_released = released - stored
    passed = not only_stored and not only_released
    return LevelResult(
        level=Level.L2,
        name="released_annotations",
        passed=passed,
        message=(
            f"the {sum(stored.values())} stored (gene, PubMed id) records are the "
            f"{sum(released.values())} inviable null S288C annotations SGD releases"
            if passed
            else f"{sum(only_stored.values())} stored records have no SGD annotation and "
            f"{sum(only_released.values())} SGD annotations have no record"
        ),
        details={
            **details,
            "n_released": sum(released.values()),
            "only_stored": sorted(map(str, only_stored))[:20],
            "only_released": sorted(map(str, only_released))[:20],
        },
    )


def verify_gene_essentiality_dataset(
    records: Iterable[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    released: Counter[tuple[str, str]] | None,
    pinned_digest: str,
    observed_digest: str,
    extra_results: Sequence[LevelResult] = (),
    resolve_gene_name: GeneNameResolver | None = None,
    sgd_genes: set[str] | None = None,
    gene_universe_label: str = "reference",
    min_containment: float = 0.90,
) -> VerificationReport:
    """Run the L0-L4 gate over one gene-essentiality store in a single pass."""
    validate = declared_member_validator("GeneEssentialityExperiment")
    validate_reference = declared_member_validator(
        "GeneEssentialityExperimentReference", union="ExperimentReferenceType"
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
    not_single: list[dict[str, Any]] = []
    keys: Counter[bytes] = Counter()
    duplicate_examples: dict[bytes, str] = {}
    stored: Counter[tuple[str, str]] = Counter()
    not_essential: list[int] = []
    reference_problems: Counter[str] = Counter()

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

        perturbations = exp["genotype"]["perturbations"]
        names = [p.get("systematic_gene_name") for p in perturbations]
        if len(perturbations) != 1 or names[0] is None:
            not_single.append({"index": i, "genes": [str(n) for n in names]})
        pubmed_id = str(rec["publication"]["pubmed_id"])
        key = key_digest(
            (
                key_digest(_genotype_signature(dict(exp))),
                key_digest(_environment_signature(dict(exp))),
                pubmed_id,
            )
        )
        keys[key] += 1
        if keys[key] == 2 and len(duplicate_examples) < 20:
            duplicate_examples[key] = f"{'+'.join(map(str, names))} PMID {pubmed_id}"
        for name in names:
            stored[(str(name), pubmed_id)] += 1
        if exp["phenotype"]["is_essential"] is not True:
            not_essential.append(i)
        if ref["phenotype_reference"]["is_essential"] is not False:
            reference_problems["reference is_essential is not False"] += 1
        if ref["environment_reference"] != exp["environment"]:
            reference_problems["reference environment differs"] += 1

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(
        l0_validated_row(
            "structural", n_records, l0_failures, "GeneEssentialityExperiment"
        )
    )
    report.add(
        l0_validated_row(
            "reference_structural",
            n_records,
            ref_failures,
            "GeneEssentialityExperimentReference",
        )
    )
    report.add(l1_count(n_records, expected_count))
    report.add(
        LevelResult(
            level=Level.L1,
            name="single_gene",
            passed=n_records > 0 and not not_single,
            message=(
                f"all {n_records} records perturb exactly one named gene"
                if not not_single
                else f"{len(not_single)} of {n_records} records are not a single named gene"
            ),
            details={"n_bad": len(not_single), "examples": not_single[:20]},
        )
    )
    n_duplicated = sum(1 for n in keys.values() if n > 1)
    n_extra = sum(n - 1 for n in keys.values() if n > 1)
    report.add(
        LevelResult(
            level=Level.L1,
            name="one_record_per_gene_and_publication",
            passed=n_records > 0 and not n_duplicated,
            message=(
                f"{len(keys)} (gene, environment, publication) keys, one record each"
                if not n_duplicated
                else f"{n_duplicated} (gene, environment, publication) keys hold more "
                f"than one record ({n_extra} extra records, identical down to the "
                "publication)"
            ),
            details={
                "n_keys": len(keys),
                "n_duplicated": n_duplicated,
                "n_extra_records": n_extra,
                "examples": sorted(duplicate_examples.values()),
            },
        )
    )
    report.add(
        LevelResult(
            level=Level.L2,
            name="essential_label",
            passed=n_records > 0 and not not_essential,
            message=(
                f"all {n_records} experiments say is_essential True"
                if not not_essential
                else f"{len(not_essential)} of {n_records} experiments do not say "
                "is_essential True"
            ),
            details={
                "n_bad": len(not_essential),
                "example_indices": not_essential[:20],
            },
        )
    )
    report.add(
        released_annotations_result(
            stored,
            released,
            pinned_digest=pinned_digest,
            observed_digest=observed_digest,
        )
    )
    report.add(
        LevelResult(
            level=Level.L3,
            name="reference_viable",
            passed=n_records > 0 and not reference_problems,
            message=(
                f"every one of the {n_records} references is viable (is_essential "
                "False) in its experiment's own environment"
                if not reference_problems
                else f"reference problems: {dict(reference_problems)}"
            ),
            details={"problems": dict(reference_problems)},
        )
    )
    for result in extra_results:
        report.add(result)
    for result in shared.results():
        report.add(result)
    return report


def essential_gene_set(records: Iterable[Record]) -> set[str]:
    """The systematic names a gene-essentiality store calls essential."""
    return {
        str(p["systematic_gene_name"])
        for rec in records
        for p in rec["experiment"]["genotype"]["perturbations"]
        if rec["experiment"]["phenotype"]["is_essential"] is True
    }


def essential_without_viable_deletion_result(
    essential: set[str], viable_deletions: Mapping[str, str], *, viable_store: str
) -> LevelResult:
    """L4: no gene the store calls essential was grown as a viable full deletion.

    ``viable_deletions`` maps each gene a fitness store holds as a single full-deletion
    strain with a positive fitness to one such strain id. A gene in both sets is a
    contradiction between the two stores on the same S288C background: one says its
    deletion is inviable, the other grew it.
    """
    contradicted = sorted(essential & set(viable_deletions))
    return LevelResult(
        level=Level.L4,
        name=f"no_viable_deletion_in_{viable_store}",
        passed=bool(essential) and not contradicted,
        message=(
            f"none of the {len(essential)} essential genes is a viable full-deletion "
            f"strain of {viable_store}"
            if not contradicted
            else f"{len(contradicted)} of {len(essential)} essential genes are viable "
            f"full-deletion strains of {viable_store}: "
            + ", ".join(
                f"{gene} ({viable_deletions[gene]})" for gene in contradicted[:10]
            )
        ),
        details={
            "n_essential": len(essential),
            "n_viable_deletions": len(viable_deletions),
            "n_contradicted": len(contradicted),
            "contradicted": {gene: viable_deletions[gene] for gene in contradicted},
        },
    )
