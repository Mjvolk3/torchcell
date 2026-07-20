# torchcell/verification/environment_response
# [[torchcell.verification.environment_response]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/environment_response
"""L0-L4 record-level verifier for environment-response datasets (WS15).

For ``EnvironmentResponsePhenotype`` datasets: chemical-genomic / stress screens read
out as ``(deletion genotype x EnvironmentPerturbation -> response)`` records (e.g. the
Vanacloig 2022 anaerobic hydrolysate-toxin bar-seq screen). The schema validator already
enforces numeric-vs-categorical coherence and the uncertainty invariant (subsumed by L0);
this verifier adds:

1. L1 ``count`` -- exact record-count oracle.
2. L1 ``pair_uniqueness`` -- one record per (screened STRAIN x condition) pair, where the
   strain is the full genotype signature (so an allelic series is distinct, not duplicate).
3. L2 ``response_finiteness`` -- numeric responses are finite (SIGNED; negatives allowed).
4. L2 ``se_nonnegative`` -- reported SEs are non-negative.
5. L3 ``measurement_type_consistent`` -- one measurement_type across the dataset.
6. L3 ``reference_zero`` -- the reference (parent-strain) response is 0.
7. L3 ``environment_perturbed`` -- every experiment carries a genuine environmental edit
   (>= 1 perturbation, or a temperature shift off the dataset baseline, e.g. heat).
8. L4 ``gene_containment`` (caller) -- screened deletions overlap the deletion collection.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from torchcell.verification.levels import l0_structural, l1_count, l2_value_fidelity
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import (
    ProvenanceGapCensus,
    ProvenanceGapReason,
    l1_provenance_gaps,
    provenance_gap_level_result,
)

Record = dict[str, Any]


def _screened_genes(
    experiment: dict[str, Any], background: frozenset[str]
) -> list[str]:
    """Systematic names deleted in this experiment's genotype, minus the constant
    drug-sensitized background (which is identical in every strain and is not a
    screened deletion, so it must not enter the L1/L4 deleted-ORF key).
    """
    return [
        p["systematic_gene_name"]
        for p in experiment["genotype"]["perturbations"]
        if p.get("systematic_gene_name") is not None
        and p["systematic_gene_name"] not in background
    ]


def _condition_signature(experiment: dict[str, Any]) -> tuple[Any, ...]:
    """Canonical ENVIRONMENT (condition) identity -- the join key on the environment axis.

    A record's condition is its full environment, not just the compound NAME: the same
    compound at two concentrations, or two exposure durations, are DIFFERENT conditions.
    The signature is the sorted set of perturbations -- each keyed by
    ``(perturbation_type, compound/agent name, factor, concentration value, unit, basis)``
    -- plus the scalar environment fields (temperature, media, duration in hours and in
    generations). Temperature-only records (heat on ``Environment.temperature``, no
    perturbation) are distinguished by the temperature scalar.

    Mirrors :func:`_genotype_signature` on the environment axis: identity is derived from
    the environment's own content, so nothing has to be smuggled into a name. Elements are
    stringified for a total order (mixed None/str/float never breaks the sort).
    """
    env = experiment["environment"]
    perts: list[tuple[str, ...]] = []
    for p in env.get("perturbations") or []:
        compound = p.get("compound") if isinstance(p.get("compound"), dict) else {}
        agent = p.get("agent") if isinstance(p.get("agent"), dict) else {}
        dose = p.get("concentration") or p.get("magnitude") or {}
        dose = dose if isinstance(dose, dict) else {}
        fields = (
            p.get("perturbation_type"),
            compound.get("name") or agent.get("name"),
            p.get("factor"),
            dose.get("value"),
            dose.get("unit"),
            dose.get("basis"),
        )
        perts.append(tuple("" if v is None else str(v) for v in fields))
    temp = (env.get("temperature") or {}).get("value")
    media = (env.get("media") or {}).get("name")
    return (
        tuple(sorted(perts)),
        temp,
        media,
        env.get("duration_hours"),
        env.get("duration_generations"),
    )


def _genotype_signature(
    experiment: dict[str, Any], background: frozenset[str]
) -> tuple[tuple[str | None, ...], ...]:
    """Canonical STRAIN identity: the sorted set of screened perturbations, each keyed by
    ``(systematic_gene_name, perturbation_type, perturbed_gene_name)``, excluding the constant
    drug-sensitized background.

    This is the entity a record measures and the join key across datasets. A deletion
    collection has one strain per gene, so the signature reduces to the gene. A TS-allele
    collection has MANY strains per essential gene: ``act1-101`` and ``act1-3`` differ in
    ``perturbed_gene_name`` and so get distinct signatures -- an allelic series is not a set
    of duplicates. A CRISPR guide library likewise has MANY strains per (gene, mode): six
    guides targeting the same gene under the same effector are distinct strains, so the
    guide spacer (``crispr.guide_sequence``) joins the key when present (None for a
    background/unspecified guide leaves the key unchanged). When a study screens several
    guide LIBRARY POOLS, the SAME spacer measured in two pools is two independent pooled
    measurements (Smith 2016: pool-relative median-centred fitness differs by up to ~8 log2
    units between pools), so the pool (``crispr.library_pool``) also joins the key when
    present (None leaves the key unchanged -> single-pool studies keep their signatures). L4
    gene-containment still keys on the bare systematic name (a gene-level question); L1
    uniqueness keys on this strain identity.
    """

    def _identity(p: dict[str, Any]) -> tuple[str | None, ...]:
        ident: tuple[str | None, ...] = (
            p.get("systematic_gene_name"),
            p.get("perturbation_type"),
            p.get("perturbed_gene_name"),
        )
        crispr = p.get("crispr")
        if isinstance(crispr, dict):
            if crispr.get("guide_sequence") is not None:
                ident = ident + (crispr["guide_sequence"],)
            if crispr.get("library_pool") is not None:
                ident = ident + (crispr["library_pool"],)
        return ident

    return tuple(
        sorted(
            _identity(p)
            for p in experiment["genotype"]["perturbations"]
            if p.get("systematic_gene_name") not in background
        )
    )


def _study_key(record: Record) -> tuple[str, str]:
    """Measurement-context discriminator: (source publication, readout ``units``).

    A single-study, single-assay dataset has a constant context, so this does not affect
    uniqueness. A MULTI-study / multi-assay aggregation (e.g. YeastPhenome) legitimately
    measures the SAME (strain, condition) in DIFFERENT screens -- a different study, or the
    same study by a different assay (microarray vs barseq, recorded in ``units``). Each NPV
    is normalized within its own screen, so those are independent measurements, NOT
    duplicates -- the context joins the uniqueness key. A true duplicate (same context, same
    strain, same condition) is still caught.
    """
    pub = record.get("publication") or {}
    units = record["experiment"]["phenotype"].get("units") or ""
    return (str(pub.get("pubmed_id") or pub.get("doi") or ""), str(units))


def _l1_pair_uniqueness(
    records: Sequence[Record], background: frozenset[str]
) -> LevelResult:
    """L1: exactly one record per (study, screened STRAIN, condition) triple.

    The screened unit is the STRAIN (the full genotype signature), not the bare gene: an
    essential gene screened as an allelic series (18 ACT1 ts alleles) contributes 18 distinct
    strains, not 18 duplicate ACT1 records. Genuine replicate measurements of an identical
    (strain, condition) WITHIN one study must aggregate into one record (n_samples); a repeat
    within a study is a real duplicate. The study (source publication) joins the key so that
    independent near-replicate SCREENS across studies (a curated multi-study aggregation) are
    not flagged as duplicates -- for a single-study dataset the study is constant and the key
    reduces to (strain, condition).
    """
    seen: dict[tuple[Any, ...], int] = {}
    for rec in records:
        exp = rec["experiment"]
        key = (
            _study_key(rec),
            _genotype_signature(exp, background),
            _condition_signature(exp),
        )
        seen[key] = seen.get(key, 0) + 1
    dups = {k: n for k, n in seen.items() if n > 1}
    return LevelResult(
        level=Level.L1,
        name="pair_uniqueness",
        passed=not dups,
        message=(
            f"{len(seen)} unique (study, strain, condition) records, one each"
            if not dups
            else f"{len(dups)} (study, strain, condition) triples appear in multiple records"
        ),
        details={"n_pairs": len(seen), "n_duplicated": len(dups)},
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


def _l3_reference_zero(records: Sequence[Record]) -> LevelResult:
    """L3: the reference (parent-strain) response is 0 -- log2(1)=0, the control baseline."""
    worst = 0.0
    n = 0
    for rec in records:
        v = rec["reference"]["phenotype_reference"]["environment_response"]
        if v is None:
            continue
        n += 1
        worst = max(worst, abs(float(v)))
    holds = worst == 0.0
    return LevelResult(
        level=Level.L3,
        name="reference_zero",
        passed=holds,
        message=(
            f"reference response == 0 for all {n} records"
            if holds
            else f"reference response not identically 0: max|v|={worst:.3g}"
        ),
        details={"n_values": n, "worst_abs": worst},
    )


def _modal_scalar(
    records: Sequence[Record], getter: Callable[[dict[str, Any]], Any]
) -> Any:
    """The dataset's baseline (most common) value of an environment scalar, if any."""
    from collections import Counter

    values = Counter(
        v
        for rec in records
        if (v := getter(rec["experiment"]["environment"])) is not None
    )
    return values.most_common(1)[0][0] if values else None


def _l3_environment_perturbed(records: Sequence[Record]) -> LevelResult:
    """L3: every experiment carries a genuine environmental edit.

    The edit is >= 1 environment perturbation (an added small molecule / physical factor)
    OR a base-environment scalar that differs from the dataset's baseline (modal) value --
    a temperature shift (heat) or a base-medium swap lives canonically on
    ``Environment.temperature`` / ``Environment.media`` with NO perturbation object (M2),
    and is a valid edit. A record is flagged only when it has NO perturbation AND sits at
    the baseline temperature AND the baseline media (a genuinely unperturbed record).
    """
    baseline_temp = _modal_scalar(
        records, lambda e: (e.get("temperature") or {}).get("value")
    )
    baseline_media = _modal_scalar(
        records, lambda e: (e.get("media") or {}).get("name")
    )
    n_missing = 0
    for rec in records:
        env = rec["experiment"]["environment"]
        if env.get("perturbations"):
            continue
        temp = (env.get("temperature") or {}).get("value")
        if temp is not None and temp != baseline_temp:
            continue  # temperature-only edit (e.g. heat) -- genuine environmental edit
        media = (env.get("media") or {}).get("name")
        if media is not None and media != baseline_media:
            continue  # base-medium swap (minimal / synthetic complete) -- genuine edit
        n_missing += 1
    return LevelResult(
        level=Level.L3,
        name="environment_perturbed",
        passed=n_missing == 0,
        message=(
            f"all {len(records)} experiments carry an environmental edit "
            f"(perturbation, non-baseline temperature, or non-baseline media; "
            f"baseline temp={baseline_temp}, media={baseline_media!r})"
            if n_missing == 0
            else f"{n_missing} experiments have no environmental edit "
            f"(no perturbation, baseline temperature {baseline_temp}, baseline media)"
        ),
        details={
            "n_records": len(records),
            "n_missing": n_missing,
            "baseline_temperature": baseline_temp,
            "baseline_media": baseline_media,
        },
    )


def verify_environment_response_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    background_genes: frozenset[str] = frozenset(),
) -> VerificationReport:
    """Run the L0-L3 record-level gate for an environment-response dataset.

    ``background_genes`` are the systematic names of the constant drug-sensitized
    background (e.g. Vanacloig 3DeltaAlpha = PDR1/PDR3/SNQ2), excluded from the
    (ORF, compound) uniqueness and gene-set keys. L4 (cross-source gene overlap with the
    deletion collection) is asserted by the caller.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_pair_uniqueness(records, background_genes))
    # Census gaps from BOTH gap-capable carriers (phenotype + environment) per record.
    report.add(
        l1_provenance_gaps(
            {
                "provenance_gaps": (
                    rec["experiment"]["phenotype"].get("provenance_gaps") or []
                )
                + (rec["experiment"]["environment"].get("provenance_gaps") or [])
            }
            for rec in records
        )
    )

    responses = [
        float(rec["experiment"]["phenotype"]["environment_response"])
        for rec in records
        if rec["experiment"]["phenotype"]["environment_response"] is not None
    ]
    report.add(l2_value_fidelity(responses, allow_nan=False))

    se_values = [
        float(v)
        for rec in records
        if (v := rec["experiment"]["phenotype"].get("environment_response_se"))
        is not None
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

    report.add(_l3_measurement_type_consistent(records))
    report.add(_l3_reference_zero(records))
    report.add(_l3_environment_perturbed(records))
    return report


def environment_response_gene_set(
    records: Sequence[Record], background_genes: frozenset[str] = frozenset()
) -> set[str]:
    """Union of screened (non-background) deleted gene names -- the L4 overlap key."""
    genes: set[str] = set()
    for rec in records:
        genes.update(_screened_genes(rec["experiment"], background_genes))
    return genes


def verify_environment_response_dataset_streaming(
    records: Iterable[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    sgd_genes: set[str],
    background_genes: frozenset[str] = frozenset(),
    min_containment: float = 0.90,
) -> VerificationReport:
    """Single-pass, memory-bounded L0-L4 gate for LARGE environment-response datasets.

    Semantically identical to ``verify_environment_response_dataset`` (+ the caller's L4
    gene-containment), but consumes ``records`` as a stream so a 30M-record dataset (e.g.
    the Hoepfner HIP-HOP atlas) never has to be materialized in RAM. The dominant
    accumulator is the (ORF, compound) pair set; interning keeps it a few GB, not the
    ~450 GB a full materialization would cost.
    """
    from collections import Counter

    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    n_records = 0
    l0_failures: list[dict[str, Any]] = []
    pair_seen: set[tuple[Any, ...]] = set()
    n_pairs = 0
    n_pair_dups = 0
    n_responses = 0
    bad_responses: list[dict[str, Any]] = []
    n_se = 0
    bad_se: list[dict[str, Any]] = []
    measurement_types: set[str] = set()
    ref_worst = 0.0
    n_ref = 0
    # Environment-edit accounting: a record with no perturbation is a valid edit iff its
    # temperature or media differs from the dataset baseline (modal). Baselines need all
    # records, so accumulate scalar counts + the (temp, media) of no-perturbation records
    # and resolve after the pass (the no-perturbation set is small).
    temp_counts: Counter[Any] = Counter()
    media_counts: Counter[Any] = Counter()
    no_pert_env: list[tuple[Any, Any]] = []
    screened: set[str] = set()
    # Provenance-gap census (single-pass; a documented gap is informational, never a fail).
    gap_records_with = 0
    gap_total = 0
    gap_by_reason: Counter[str] = Counter()
    gap_by_field: Counter[str] = Counter()
    gap_worklist: set[str] = set()

    for i, rec in enumerate(records):
        exp = rec["experiment"]
        n_records += 1
        gaps = (exp["phenotype"].get("provenance_gaps") or []) + (
            exp["environment"].get("provenance_gaps") or []
        )
        if gaps:
            gap_records_with += 1
        for gap in gaps:
            gap_total += 1
            reason = str(gap["reason"])
            field = str(gap["field"])
            gap_by_reason[reason] += 1
            gap_by_field[field] += 1
            if reason == ProvenanceGapReason.deferred_pending_source_review:
                gap_worklist.add(field)
        try:
            validate(exp)
        except (ValueError, TypeError) as err:
            l0_failures.append({"index": i, "error": str(err)[:500]})

        # L1 uniqueness keys on the STUDY x the STRAIN (genotype signature) x the full
        # CONDITION signature (environment identity); L4 gene-containment accumulates the
        # bare screened systematic names.
        pkey = (
            _study_key(rec),
            _genotype_signature(exp, background_genes),
            _condition_signature(exp),
        )
        if pkey in pair_seen:
            n_pair_dups += 1
        else:
            pair_seen.add(pkey)
            n_pairs += 1
        for gene in _screened_genes(exp, background_genes):
            screened.add(sys.intern(gene))

        response = exp["phenotype"]["environment_response"]
        if response is not None:
            n_responses += 1
            if math.isnan(response) or math.isinf(response):
                bad_responses.append({"index": i, "value": repr(response)})
        se = exp["phenotype"].get("environment_response_se")
        if se is not None and not (isinstance(se, float) and math.isnan(se)):
            n_se += 1
            if se < 0.0:
                bad_se.append({"index": i, "value": se})
        measurement_types.add(exp["phenotype"]["measurement_type"])

        ref_val = rec["reference"]["phenotype_reference"]["environment_response"]
        if ref_val is not None:
            n_ref += 1
            ref_worst = max(ref_worst, abs(float(ref_val)))
        temp = (exp["environment"].get("temperature") or {}).get("value")
        media = (exp["environment"].get("media") or {}).get("name")
        if temp is not None:
            temp_counts[temp] += 1
        if media is not None:
            media_counts[media] += 1
        if not (exp["environment"].get("perturbations") or []):
            no_pert_env.append((temp, media))

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(
        LevelResult(
            level=Level.L0,
            name="structural",
            passed=not l0_failures,
            message=(
                f"{n_records} records validated"
                if not l0_failures
                else f"{len(l0_failures)}/{n_records} records failed schema validation"
            ),
            details={
                "n_records": n_records,
                "n_failures": len(l0_failures),
                "failures": l0_failures[:10],
            },
        )
    )
    report.add(
        LevelResult(
            level=Level.L1,
            name="count",
            passed=n_records == expected_count,
            message=f"observed {n_records}, expected {expected_count}",
            details={"observed": n_records, "expected": expected_count},
        )
    )
    report.add(
        LevelResult(
            level=Level.L1,
            name="pair_uniqueness",
            passed=n_pair_dups == 0,
            message=(
                f"{n_pairs} unique (strain, condition) records, one each"
                if n_pair_dups == 0
                else f"{n_pair_dups} (strain, condition) records duplicate an existing pair"
            ),
            details={"n_pairs": n_pairs, "n_duplicated": n_pair_dups},
        )
    )
    report.add(
        provenance_gap_level_result(
            ProvenanceGapCensus(
                n_records=n_records,
                n_records_with_gaps=gap_records_with,
                n_gaps=gap_total,
                by_reason=dict(gap_by_reason),
                by_field=dict(gap_by_field),
                worklist_fields=sorted(gap_worklist),
            )
        )
    )
    report.add(
        LevelResult(
            level=Level.L2,
            name="value_fidelity",
            passed=not bad_responses,
            message=(
                f"{n_responses} values checked"
                if not bad_responses
                else f"{len(bad_responses)}/{n_responses} values invalid"
            ),
            details={
                "n_values": n_responses,
                "n_bad": len(bad_responses),
                "bad": bad_responses[:20],
            },
        )
    )
    report.add(
        LevelResult(
            level=Level.L2,
            name="se_nonnegative",
            passed=not bad_se,
            message=(
                f"{n_se} values checked"
                if not bad_se
                else f"{len(bad_se)}/{n_se} values invalid"
            ),
            details={"n_values": n_se, "n_bad": len(bad_se), "bad": bad_se[:20]},
        )
    )
    report.add(
        LevelResult(
            level=Level.L3,
            name="measurement_type_consistent",
            passed=len(measurement_types) <= 1,
            message=(
                f"single measurement_type: {next(iter(measurement_types), None)!r}"
                if len(measurement_types) <= 1
                else f"{len(measurement_types)} distinct measurement_types mixed: "
                f"{sorted(measurement_types)}"
            ),
            details={"measurement_types": sorted(measurement_types)},
        )
    )
    report.add(
        LevelResult(
            level=Level.L3,
            name="reference_zero",
            passed=ref_worst == 0.0,
            message=(
                f"reference response == 0 for all {n_ref} records"
                if ref_worst == 0.0
                else f"reference response not identically 0: max|v|={ref_worst:.3g}"
            ),
            details={"n_values": n_ref, "worst_abs": ref_worst},
        )
    )
    baseline_temp = temp_counts.most_common(1)[0][0] if temp_counts else None
    baseline_media = media_counts.most_common(1)[0][0] if media_counts else None
    n_env_missing = sum(
        1
        for temp, media in no_pert_env
        if not (temp is not None and temp != baseline_temp)
        and not (media is not None and media != baseline_media)
    )
    report.add(
        LevelResult(
            level=Level.L3,
            name="environment_perturbed",
            passed=n_env_missing == 0,
            message=(
                f"all {n_records} experiments carry an environmental edit (perturbation, "
                f"non-baseline temperature, or non-baseline media; baseline temp="
                f"{baseline_temp}, media={baseline_media!r})"
                if n_env_missing == 0
                else f"{n_env_missing} experiments have no environmental edit "
                f"(no perturbation, baseline temperature {baseline_temp}, baseline media)"
            ),
            details={
                "n_records": n_records,
                "n_missing": n_env_missing,
                "baseline_temperature": baseline_temp,
                "baseline_media": baseline_media,
            },
        )
    )
    overlap = len(screened & sgd_genes) / len(screened) if screened else 0.0
    report.add(
        LevelResult(
            level=Level.L4,
            name="gene_containment_sgd",
            passed=overlap >= min_containment,
            message=(
                f"{overlap:.3f} of {len(screened)} measured genes are S288C reference "
                f"genes (>= {min_containment})"
            ),
            details={
                "n_measured": len(screened),
                "n_in_sgd": len(screened & sgd_genes),
                "overlap": overlap,
                "missing_examples": sorted(screened - sgd_genes)[:20],
            },
        )
    )
    return report


__all__ = [
    "verify_environment_response_dataset",
    "verify_environment_response_dataset_streaming",
    "environment_response_gene_set",
    "Record",
]
