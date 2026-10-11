# torchcell/verification/genome_architecture
# [[torchcell.verification.genome_architecture]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/genome_architecture
"""L0-L4 record-level verifier for a ``GenomeArchitecturePhenotype`` dataset.

No existing family verifier fits. :mod:`torchcell.verification.morphology` and
:mod:`torchcell.verification.bacterial_morphology` are the two multi-feature verifiers
and both are imaging vocabularies; neither can express a copy number per haploid genome,
a chromosome copy number, or the aggregate identity that makes this release
self-checking. What this verifier adds over what the schema already enforces at L0
(keys belong to the named assay, every value is finite and non-negative, a rearrangement
count is integral, and a stored ploidy-deviation aggregate equals the chromosome
deviations it sums over):

1. L1 ``assay_coverage`` -- every record carries the features ``required`` names and
   nothing outside the assay. The schema accepts any non-empty subset, so a build that
   dropped a released column would pass it.
2. L1 ``sample_uniqueness`` -- one record per sequenced sample, keyed on the source's own
   sample id. Two colonies of one strain are two samples and must both survive; two
   records of one colony mean the build duplicated a row.
3. L2 ``feature_fidelity`` -- per feature, a census and the bound its declared basis
   implies, so a defect names the column rather than an index into a pooled list.
4. L2 ``release_fidelity`` -- every stored number is re-read from the released workbook
   and compared cell by cell against the record that carries that sample id. This is the
   check that makes the store a faithful copy of the release rather than of the loader.
5. L3 ``vocabulary_parity`` -- the named assay's symbols are distinct and every one
   declares a basis this verifier knows how to bound.
6. L3 ``provenance_audit`` -- every paper-OCR quote the loader pins is still present in
   the sha256-pinned mirror bytes, and every workbook legend cell it pins is still a cell
   of the sheet it names.
7. L3 ``reference_profile`` -- the reference carries the published wild-type mean with a
   named uncertainty type and its sample count, and states a per-key gap for every
   feature it does not cover.
8. L4 ``paper_statistics`` -- the store reproduces the paper's own summary numbers: the
   count of rho0 colonies (mtDNA below one copy) and the fraction of colonies deviating
   from 2n.

The family-agnostic rules (gap census, canonical gene names, uncertainty sanity,
compound identity, media membership, the L4 gene universe) come from
:class:`~torchcell.verification.common.SharedRecordRules`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from torchcell.datamodels.genome_architecture_features import (
    INTEGRAL_BASES,
    ArchitectureAssay,
    ArchitectureBasis,
    architecture_assay,
)
from torchcell.verification.common import GeneNameResolver, SharedRecordRules
from torchcell.verification.levels import l0_structural, l1_count, l3_convention
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

# One LMDB entry: {"experiment": {...}, "reference": {...}, "publication": {...}}.
Record = dict[str, Any]

#: Lower bound every basis shares (a copy number, a read share, an absolute deviation
#: and a count are all non-negative) and the upper bound none of them has.
BASIS_BOUNDS: dict[ArchitectureBasis, tuple[float, float]] = {
    basis: (0.0, math.inf) for basis in ArchitectureBasis
}


def _assay_of(records: Sequence[Record]) -> ArchitectureAssay:
    """The one assay every record names, or a ValueError naming the ones found."""
    names = {rec["experiment"]["phenotype"]["assay"] for rec in records}
    if len(names) != 1:
        raise ValueError(
            "a genome-architecture dataset carries one assay vocabulary; found "
            f"{sorted(names)}"
        )
    return architecture_assay(names.pop())


def _profile(record: Record) -> dict[str, float]:
    """One record's stored feature values."""
    return dict(record["experiment"]["phenotype"]["genome_architecture"])


def _l1_assay_coverage(
    records: Sequence[Record], assay: ArchitectureAssay, required: frozenset[str]
) -> LevelResult:
    """L1: no feature outside the assay, and every record carries ``required``."""
    symbols = frozenset(assay.by_symbol)
    unknown: list[dict[str, Any]] = []
    short: list[dict[str, Any]] = []
    strata: dict[int, int] = {}
    for i, rec in enumerate(records):
        stored = frozenset(_profile(rec))
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
            "coverage_strata": {str(k): v for k, v in sorted(strata.items())},
            "outside_the_assay": unknown[:20],
            "missing_required": short[:20],
        },
    )


def _l1_sample_uniqueness(records: Sequence[Record]) -> LevelResult:
    """L1: one record per sequenced sample id, and every record states one."""
    seen: dict[str, int] = {}
    missing = 0
    for rec in records:
        sample = rec["experiment"]["phenotype"].get("sample_id")
        if sample is None:
            missing += 1
            continue
        seen[str(sample)] = seen.get(str(sample), 0) + 1
    duplicated = {sample: n for sample, n in seen.items() if n > 1}
    passed = not duplicated and not missing
    return LevelResult(
        level=Level.L1,
        name="sample_uniqueness",
        passed=passed,
        message=(
            f"{len(seen)} sequenced samples, one record each"
            if passed
            else f"{len(duplicated)} sample ids appear in more than one record and "
            f"{missing} records state none"
        ),
        details={
            "n_samples": len(seen),
            "n_duplicated": len(duplicated),
            "n_without_sample_id": missing,
            "duplicated": sorted(duplicated)[:20],
        },
    )


def _l2_feature_fidelity(
    records: Sequence[Record], assay: ArchitectureAssay
) -> LevelResult:
    """L2: per feature, every value is finite, inside its basis bound, and integral
    where the basis is a count.
    """
    checked = 0
    by_feature: dict[str, int] = {}
    bad: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        for symbol, value in _profile(rec).items():
            feature = assay.by_symbol[symbol]
            number = float(value)
            checked += 1
            by_feature[symbol] = by_feature.get(symbol, 0) + 1
            low, high = BASIS_BOUNDS[feature.basis]
            integral = feature.basis in INTEGRAL_BASES
            if (
                not math.isfinite(number)
                or not low <= number <= high
                or (integral and number != int(number))
            ):
                bad.append(
                    {
                        "index": i,
                        "feature": symbol,
                        "basis": feature.basis.value,
                        "value": number,
                        "bounds": [low, high],
                        "must_be_integral": integral,
                    }
                )
    passed = not bad
    return LevelResult(
        level=Level.L2,
        name="feature_fidelity",
        passed=passed,
        message=(
            f"{checked} values over {len(by_feature)} features are finite and inside "
            "the bound of their declared basis"
            if passed
            else f"{len(bad)}/{checked} values are non-finite, out of bound, or a "
            "non-integral count"
        ),
        details={
            "n_values": checked,
            "n_features": len(by_feature),
            "n_bad": len(bad),
            "values_by_feature": dict(sorted(by_feature.items())),
            "bad": bad[:20],
        },
    )


def _l2_release_fidelity(
    records: Sequence[Record],
    released: dict[str, dict[str, float]],
    *,
    tol: float = 0.0,
) -> LevelResult:
    """L2: every stored number equals the released cell for that sample and feature."""
    disagreements: list[dict[str, Any]] = []
    unknown_samples: list[str] = []
    compared = 0
    for i, rec in enumerate(records):
        phenotype = rec["experiment"]["phenotype"]
        sample = str(phenotype.get("sample_id"))
        row = released.get(sample)
        if row is None:
            unknown_samples.append(sample)
            continue
        for symbol, value in _profile(rec).items():
            compared += 1
            expected = row.get(symbol)
            if expected is None or abs(float(value) - float(expected)) > tol:
                disagreements.append(
                    {
                        "index": i,
                        "sample_id": sample,
                        "feature": symbol,
                        "stored": float(value),
                        "released": None if expected is None else float(expected),
                    }
                )
    passed = not disagreements and not unknown_samples
    return LevelResult(
        level=Level.L2,
        name="release_fidelity",
        passed=passed,
        message=(
            f"{compared} stored values equal the released cell for their sample"
            if passed
            else f"{len(disagreements)}/{compared} values differ from the release and "
            f"{len(unknown_samples)} samples are not in it"
        ),
        details={
            "n_compared": compared,
            "n_disagreements": len(disagreements),
            "n_samples_not_in_release": len(unknown_samples),
            "tolerance": tol,
            "disagreements": disagreements[:20],
            "samples_not_in_release": unknown_samples[:20],
        },
    )


def _l3_vocabulary_parity(assay: ArchitectureAssay) -> LevelResult:
    """L3: the assay's symbols are distinct and every basis is one this gate bounds."""
    symbols = [feature.symbol for feature in assay.features]
    distinct = len(set(symbols)) == len(symbols)
    bounded = {feature.basis for feature in assay.features} <= set(BASIS_BOUNDS)
    return l3_convention(
        "vocabulary_parity",
        distinct and bounded,
        detail=(
            f"assay {assay.name}: {len(symbols)} features, distinct={distinct}, "
            f"every basis bounded={bounded}, reference assembly "
            f"{assay.reference_assembly}"
        ),
    )


def _l3_provenance_audit(
    paper_values: Iterable[SourcedValue],
    workbook_values: Iterable[SourcedValue],
    *,
    library_root: str | Path,
    workbook_cells: frozenset[str],
) -> LevelResult:
    """L3: every pinned quote is still in the bytes it names."""
    failed: list[str] = []
    audited = 0
    for sv in paper_values:
        audit = audit_sourced_value(sv, library_root)
        audited += 1
        if not audit.passed:
            failed.append(f"{sv.quote[:60]!r}: {audit.message}")
    for sv in workbook_values:
        audited += 1
        if sv.quote not in workbook_cells:
            failed.append(f"workbook legend cell {sv.quote[:60]!r} is not in the sheet")
    return LevelResult(
        level=Level.L3,
        name="provenance_audit",
        passed=not failed,
        message=(
            f"{audited} pinned quotes are present in the sha256-pinned sources"
            if not failed
            else f"{len(failed)}/{audited} pinned quotes could not be re-found"
        ),
        details={"n_audited": audited, "n_failed": len(failed), "failed": failed[:20]},
    )


def _l3_reference_profile(
    records: Sequence[Record], assay: ArchitectureAssay
) -> LevelResult:
    """L3: the reference states a mean, a named uncertainty kind, its n, and its gaps."""
    problems: list[str] = []
    profiles = 0
    seen: set[str] = set()
    for rec in records:
        reference = rec["reference"]["phenotype_reference"]
        key = str(sorted(reference["genome_architecture"]))
        if key in seen:
            continue
        seen.add(key)
        profiles += 1
        values = reference["genome_architecture"]
        dispersion = reference.get("genome_architecture_sd")
        if not values:
            problems.append("the reference profile carries no value")
        if not dispersion:
            problems.append("the reference profile carries no dispersion")
        if reference.get("uncertainty_type") is None:
            problems.append("the reference dispersion has no uncertainty type")
        if not reference.get("n_samples"):
            problems.append("the reference does not state how many samples it covers")
        covered = set(values) | set(dispersion or {})
        uncovered = frozenset(assay.by_symbol) - covered
        gapped = {
            key
            for gap in reference.get("provenance_gaps", [])
            for key in gap.get("keys", [])
        }
        unexplained = sorted(uncovered - gapped)
        if unexplained:
            problems.append(
                f"the reference omits {unexplained} with no per-key gap saying why"
            )
    return LevelResult(
        level=Level.L3,
        name="reference_profile",
        passed=not problems,
        message=(
            f"{profiles} reference profile(s) state a published mean, its uncertainty "
            "kind, its sample count, and a typed gap for every feature not covered"
            if not problems
            else "; ".join(sorted(set(problems))[:5])
        ),
        details={"n_profiles": profiles, "problems": sorted(set(problems))[:20]},
    )


def _l4_paper_statistics(
    records: Sequence[Record],
    *,
    assay: ArchitectureAssay,
    rho0_feature: str,
    rho0_threshold: float,
    expected_rho0: int,
    expected_aneuploid_fraction: float,
    aneuploid_fraction_tol: float,
    wild_type_rho0: int,
) -> LevelResult:
    """L4: the store reproduces the paper's own rho0 count and aneuploid fraction.

    The paper's numbers are over all released colonies; the store serves the knockout
    colonies, so the wild-type contribution is stated by the caller and added back
    rather than hidden in a tolerance.
    """
    chromosomes = sorted(
        assay.symbols_with_basis(ArchitectureBasis.chromosome_copies_per_cell)
    )
    rho0 = 0
    aneuploid = 0
    for rec in records:
        profile = _profile(rec)
        if float(profile[rho0_feature]) < rho0_threshold:
            rho0 += 1
        if any(float(profile[symbol]) != 2.0 for symbol in chromosomes):
            aneuploid += 1
    fraction = aneuploid / len(records) if records else 0.0
    rho0_total = rho0 + wild_type_rho0
    rho0_ok = rho0_total == expected_rho0
    fraction_ok = abs(fraction - expected_aneuploid_fraction) <= aneuploid_fraction_tol
    return LevelResult(
        level=Level.L4,
        name="paper_statistics",
        passed=rho0_ok and fraction_ok,
        message=(
            f"{rho0_total} rho0 colonies (paper: {expected_rho0}) and "
            f"{fraction:.4f} of served colonies away from 2n (paper: "
            f"{expected_aneuploid_fraction})"
        ),
        details={
            "n_rho0_served": rho0,
            "n_rho0_wild_type": wild_type_rho0,
            "n_rho0_total": rho0_total,
            "expected_rho0": expected_rho0,
            "rho0_feature": rho0_feature,
            "rho0_threshold": rho0_threshold,
            "n_aneuploid_served": aneuploid,
            "aneuploid_fraction": fraction,
            "expected_aneuploid_fraction": expected_aneuploid_fraction,
            "aneuploid_fraction_tol": aneuploid_fraction_tol,
        },
    )


def verify_genome_architecture_records(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    required_features: frozenset[str],
    released: dict[str, dict[str, float]],
    paper_values: Iterable[SourcedValue],
    workbook_values: Iterable[SourcedValue],
    library_root: str | Path,
    workbook_cells: frozenset[str],
    rho0_feature: str,
    rho0_threshold: float,
    expected_rho0: int,
    expected_aneuploid_fraction: float,
    aneuploid_fraction_tol: float = 0.01,
    wild_type_rho0: int = 0,
    sgd_genes: set[str] | None = None,
    gene_universe_label: str = "reference",
    resolve_gene_name: GeneNameResolver | None = None,
    min_containment: float = 0.90,
) -> VerificationReport:
    """Run the L0-L4 gate for a ``GenomeArchitecturePhenotype`` dataset.

    Args:
        records: the per-LMDB-entry dicts (experiment / reference / publication).
        dataset_name: dataset identity for the report.
        provenance: where the records came from.
        expected_count: the record-count oracle.
        required_features: assay symbols every record must carry.
        released: ``{sample id: {feature: released value}}`` re-read from the release.
        paper_values: the paper-OCR quotes the loader pins.
        workbook_values: the workbook legend cells the loader pins.
        library_root: ``$DATA_ROOT/torchcell-library``, for the quote audit.
        workbook_cells: every cell string of the released workbook, as strings.
        rho0_feature: the assay symbol holding mitochondrial copy number.
        rho0_threshold: the copy number below which the paper calls a colony rho0.
        expected_rho0: the paper's rho0 colony count.
        expected_aneuploid_fraction: the paper's fraction of colonies away from 2n.
        aneuploid_fraction_tol: how far from it the store may be ("about 52%").
        wild_type_rho0: rho0 colonies among the reference colonies the store does not
            serve, so the L4 count is comparable to the paper's.
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
    report.add(_l1_sample_uniqueness(records))
    report.add(_l1_assay_coverage(records, assay, required_features))
    report.add(_l2_feature_fidelity(records, assay))
    report.add(_l2_release_fidelity(records, released))
    report.add(_l3_vocabulary_parity(assay))
    report.add(
        _l3_provenance_audit(
            paper_values,
            workbook_values,
            library_root=library_root,
            workbook_cells=workbook_cells,
        )
    )
    report.add(_l3_reference_profile(records, assay))
    report.add(
        _l4_paper_statistics(
            records,
            assay=assay,
            rho0_feature=rho0_feature,
            rho0_threshold=rho0_threshold,
            expected_rho0=expected_rho0,
            expected_aneuploid_fraction=expected_aneuploid_fraction,
            aneuploid_fraction_tol=aneuploid_fraction_tol,
            wild_type_rho0=wild_type_rho0,
        )
    )

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


def genome_architecture_gene_set(records: Sequence[Record]) -> set[str]:
    """The systematic gene names a genome-architecture dataset perturbs."""
    return {
        p["systematic_gene_name"]
        for rec in records
        for p in rec["experiment"]["genotype"]["perturbations"]
        if p.get("systematic_gene_name")
    }
