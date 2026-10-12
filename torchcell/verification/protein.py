# torchcell/verification/protein
# [[torchcell.verification.protein]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/protein
"""L0-L4 record-level verifier for protein-abundance datasets (WS9; Zelezniak2018).

For `ProteinAbundancePhenotype` datasets (e.g. Zelezniak SWATH-MS of kinase knockouts).
The schema validator already enforces non-empty abundances, matching abundance/replicate
keys, and non-negative SE, so L0 subsumes those. This verifier adds:

1. L1 `orf_uniqueness` -- one record per deleted (knocked-out) ORF.
2. L2 `value_fidelity` -- measured abundances are finite (guards inf/NaN).
3. L2 `se_nonnegative` -- released SE values are finite + non-negative.
4. L3 `reference_finite` -- the WT reference abundance is finite + key-matched to the
   experiment (absolute abundances, not a centered 0).
5. L3 `measurement_type_consistent` -- every record shares one measurement_type.
6. L4 `gene_containment` (caller) -- the screened knockouts overlap the deletion
   collection used by the other datasets.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from torchcell.verification.levels import (
    l0_structural,
    l1_count,
    l2_value_fidelity,
    validate_by_type_tag,
)
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

if TYPE_CHECKING:
    from torchcell.verification.natural_isolate_genotype import IsolateGenotypeAudit

Record = dict[str, Any]


def _deleted_genes(experiment: dict[str, Any]) -> list[str]:
    """Systematic names knocked out in this experiment's genotype."""
    return [
        p["systematic_gene_name"]
        for p in experiment["genotype"]["perturbations"]
        if p.get("systematic_gene_name") is not None
    ]


def _l1_orf_uniqueness(
    records: Sequence[Record], allow_duplicate_orfs: bool = False
) -> LevelResult:
    """L1: one record per knocked-out ORF.

    Some datasets intentionally hold multiple deletion strains for the same ORF (e.g.
    Messner 2023 has 145 ORFs with 2-3 strains "of different origins", each analysed
    independently). For those, ``allow_duplicate_orfs=True`` turns this into an
    informational check (duplicates reported, not failed) -- the per-instance
    uniqueness is guaranteed by the record count (L1 ``expected_count``).
    """
    seen: dict[str, int] = {}
    for rec in records:
        for g in _deleted_genes(rec["experiment"]):
            seen[g] = seen.get(g, 0) + 1
    dups = {g: n for g, n in seen.items() if n > 1}
    passed = allow_duplicate_orfs or not dups
    if not dups:
        message = f"{len(seen)} unique knocked-out ORFs, one record each"
    elif allow_duplicate_orfs:
        message = f"{len(seen)} ORFs, {len(dups)} with multiple strains (expected)"
    else:
        message = f"{len(dups)} ORFs appear in multiple records"
    return LevelResult(
        level=Level.L1,
        name="orf_uniqueness",
        passed=passed,
        message=message,
        details={"n_orfs": len(seen), "n_duplicated": len(dups)},
    )


def _l3_reference_finite(
    records: Sequence[Record], label_key: str = "protein_abundance"
) -> LevelResult:
    """L3: the WT reference abundance is finite + key-matched for every protein."""
    n = 0
    bad = 0
    for rec in records:
        exp_keys = set(rec["experiment"]["phenotype"][label_key])
        levels = rec["reference"]["phenotype_reference"][label_key]
        if set(levels) != exp_keys:
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
            f"reference abundance finite + key-matched for all {n} values"
            if holds
            else f"{bad} reference abundances non-finite or key-mismatched"
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


def verify_protein_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    allow_duplicate_orfs: bool = False,
    label_key: str = "protein_abundance",
    se_key: str = "protein_abundance_se",
) -> VerificationReport:
    """Run the L0-L3 record-level gate for a protein-abundance dataset.

    L4 (cross-source gene overlap with the deletion collection) is asserted by the
    caller across datasets. ``allow_duplicate_orfs`` relaxes L1 uniqueness for
    datasets that intentionally hold multiple strains per ORF (e.g. Messner 2023).

    ``label_key`` / ``se_key`` name the phenotype's value and SE maps, so the same gate
    runs over the relative sibling of this family: a ``ProteinFoldChangePhenotype``
    dataset passes ``label_key="protein_fold_change"``. Every rule here is about the
    SHAPE of a per-protein map (finite values, non-negative SE, a key-matched finite
    reference, one measurement_type), which both classes share; what the number MEANS is
    the caller's own L3/L4 business, which is why nothing here reads the scale.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_orf_uniqueness(records, allow_duplicate_orfs=allow_duplicate_orfs))

    levels = [
        float(v)
        for rec in records
        for v in rec["experiment"]["phenotype"][label_key].values()
    ]
    report.add(l2_value_fidelity(levels, allow_nan=False))

    se_values = [
        float(v)
        for rec in records
        for v in (rec["experiment"]["phenotype"].get(se_key) or {}).values()
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

    report.add(_l3_reference_finite(records, label_key))
    report.add(_l3_measurement_type_consistent(records))
    return report


def protein_gene_set(records: Sequence[Record]) -> set[str]:
    """Union of knocked-out systematic gene names across records (the L4 overlap key)."""
    genes: set[str] = set()
    for rec in records:
        genes.update(_deleted_genes(rec["experiment"]))
    return genes


# --------------------------------------------------------------------------- #
# Natural isolates (Muenzner 2024): one record per isolate, a shared population reference
# --------------------------------------------------------------------------- #
#: Relative tolerance of the half-life identity HL = ln 2 / kdp (exact in the release to
#: 5.7e-15; anything near this bound is a transcription, not a rounding).
HALF_LIFE_IDENTITY_RTOL = 1e-9


def natural_isolate_of(rec: Record) -> str | None:
    """The isolate a natural-isolate record measures.

    The isolate its genotype's ``natural_isolate`` descriptor names; for an S288C-derived
    laboratory strain stored with no perturbation off S288C and no descriptor (Muenzner
    2024's FY4, #1045), the background its own reference genome names. None when neither
    says.
    """
    from torchcell.sequence.natural_isolate import genotype_isolate

    genotype = rec["experiment"]["genotype"]
    isolate = genotype_isolate(genotype)
    if isolate is not None:
        return isolate
    if genotype["perturbations"]:
        return None
    background = str(rec["reference"]["genome_reference"]["strain"])
    return None if background == "S288C" else background


def _l1_genotype_strain_consistency(records: Sequence[Record]) -> LevelResult:
    """L1: every record's genotype names ONE isolate.

    A natural isolate is ONE perturbation set off S288C, named by its
    ``natural_isolate`` descriptor; every perturbation that carries a ``strain_id``
    must name that isolate (the engineered HO deletion carries none). A record mixing
    two isolates' perturbations is a join error the one-strain-per-record rule cannot
    see, since that rule reads only the first perturbation that names a strain. A record
    with no perturbation and no descriptor passes only when its reference genome names
    an S288C-derived background (FY4), and is counted apart.
    """
    from torchcell.sequence.natural_isolate import genotype_isolate

    mixed = []
    n_background = 0
    for i, rec in enumerate(records):
        genotype = rec["experiment"]["genotype"]
        if not genotype["perturbations"] and genotype_isolate(genotype) is None:
            if natural_isolate_of(rec) is None:
                mixed.append(i)
            else:
                n_background += 1
            continue
        strains = {
            p.get("strain_id")
            for p in genotype["perturbations"]
            if p.get("strain_id") is not None
        } | {genotype_isolate(genotype)}
        if len(strains) != 1 or None in strains:
            mixed.append(i)
    return LevelResult(
        level=Level.L1,
        name="genotype_strain_consistency",
        passed=not mixed,
        message=(
            f"all {len(records)} genotypes name one isolate each"
            + (
                f" ({n_background} S288C-derived records carry no perturbation off "
                "S288C and name their background on the reference)"
                if n_background
                else ""
            )
            if not mixed
            else f"{len(mixed)} records mix isolates or carry none"
        ),
        details={
            "n_records": len(records),
            "n_mixed": len(mixed),
            "first": mixed[:5],
            "n_s288c_derived": n_background,
        },
    )


def _l3_shared_reference(
    records: Sequence[Record], label_key: str
) -> tuple[LevelResult, dict[str, Any]]:
    """L3: one reference profile serves every record, with finite positive values.

    Returns the result and the shared reference phenotype (empty when there is none).
    """
    dumps = {
        json.dumps(rec["reference"]["phenotype_reference"], sort_keys=True)
        for rec in records
    }
    reference: dict[str, Any] = (
        records[0]["reference"]["phenotype_reference"] if records else {}
    )
    values = [float(v) for v in reference.get(label_key, {}).values()]
    bad = [v for v in values if not (math.isfinite(v) and v > 0)]
    passed = len(dumps) == 1 and bool(values) and not bad
    return (
        LevelResult(
            level=Level.L3,
            name="shared_reference",
            passed=passed,
            message=(
                f"one shared reference over {len(values)} proteins, all finite and > 0"
                if passed
                else f"{len(dumps)} distinct references, {len(bad)} bad values"
            ),
            details={
                "n_references": len(dumps),
                "n_keys": len(values),
                "n_bad": len(bad),
            },
        ),
        reference,
    )


def _uncovered_keys(
    records: Sequence[Record], label_key: str, reference: dict[str, Any]
) -> int:
    """How many (record, protein) cells have no key in the shared reference."""
    ref_keys = set(reference.get(label_key, {}))
    return sum(
        len(set(rec["experiment"]["phenotype"][label_key]) - ref_keys)
        for rec in records
    )


def _l4_containment(
    measured: set[str], universe: set[str], minimum: float, name: str
) -> LevelResult:
    """L4: the measured protein keys are S288C systematic names (fraction >= minimum)."""
    inside = measured & universe
    fraction = len(inside) / len(measured) if measured else 0.0
    return LevelResult(
        level=Level.L4,
        name=name,
        passed=fraction >= minimum,
        message=(
            f"{fraction:.4f} of {len(measured)} measured proteins are S288C systematic "
            f"names (>= {minimum}); outside: {sorted(measured - universe)[:10]}"
        ),
        details={
            "n_measured": len(measured),
            "n_inside": len(inside),
            "outside": sorted(measured - universe),
        },
    )


def _measured_keys(records: Sequence[Record], label_key: str) -> set[str]:
    keys: set[str] = set()
    for rec in records:
        keys.update(rec["experiment"]["phenotype"][label_key])
    return keys


def _strains(records: Sequence[Record]) -> set[str]:
    return {
        strain for rec in records if (strain := natural_isolate_of(rec)) is not None
    }


def _l4_caudal_genotype_agreement(
    records: Sequence[Record], caudal_genotypes: Mapping[str, Mapping[str, Any]]
) -> LevelResult:
    """L4: an isolate's genotype here equals its genotype in the Caudal 2024 store."""
    from torchcell.sequence.natural_isolate import genotype_isolate

    by_strain: dict[str, Mapping[str, Any]] = {}
    for rec in records:
        genotype = rec["experiment"]["genotype"]
        isolate = genotype_isolate(genotype)
        if isolate is not None:
            by_strain[isolate] = genotype
    absent = sorted(set(caudal_genotypes) - set(by_strain))
    differ = sorted(
        s
        for s, genotype in caudal_genotypes.items()
        if s in by_strain and dict(by_strain[s]) != dict(genotype)
    )
    compared = len(caudal_genotypes) - len(absent)
    passed = bool(caudal_genotypes) and not absent and not differ
    return LevelResult(
        level=Level.L4,
        name="caudal_genotype_agreement",
        passed=passed,
        message=(
            f"{compared} isolates compared with the Caudal 2024 store: "
            f"{compared - len(differ)} identical genotypes"
            + (f"; differ {differ}" if differ else "")
            + (f"; not built here {absent}" if absent else "")
        ),
        details={
            "compared": sorted(caudal_genotypes),
            "differ": differ,
            "absent": absent,
            "n_perturbations": {
                s: len(g["perturbations"]) for s, g in caudal_genotypes.items()
            },
        },
    )


def verify_natural_isolate_proteome(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    sgd_genes: set[str],
    caudal_isolates: set[str],
    expected_caudal_overlap: int,
    caudal_genotypes: Mapping[str, Mapping[str, Any]],
    genotype_audit: IsolateGenotypeAudit,
    min_containment: float = 0.99,
) -> VerificationReport:
    """L0-L4 for a natural-isolate proteome (Muenzner 2024).

    L1 is one record per isolate (``strain_uniqueness``, the Caudal rule) and one isolate
    per genotype; the ORF-uniqueness rule of the deletion proteomes does not apply, since
    every isolate carries thousands of variants. L2: quantities finite and > 0 and one
    acquisition per isolate (``n_replicates`` = 1). L3: one shared euploid-median
    reference that covers every measured key, one measurement_type. L4: protein keys are
    S288C names, and the isolates overlap the built Caudal 2024 transcriptome store in
    exactly ``expected_caudal_overlap`` isolates (the pairing the dataset is built for),
    and each isolate of ``caudal_genotypes`` (genotype dumps read from that store)
    carries the identical genotype here, since both are built by one construction.
    ``genotype_audit`` adds the four isolate-genotype rules every natural-isolate store
    shares (``natural_isolate_genotype``).
    """
    from torchcell.verification.levels import l3_convention
    from torchcell.verification.rnaseq import _l1_strain_uniqueness

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(
        l0_structural((rec["experiment"] for rec in records), validate_by_type_tag)
    )
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_strain_uniqueness(records, strain_of=natural_isolate_of))
    report.add(_l1_genotype_strain_consistency(records))
    levels = [
        float(v)
        for rec in records
        for v in rec["experiment"]["phenotype"]["protein_abundance"].values()
    ]
    report.add(l2_value_fidelity(levels, allow_nan=False, minimum=0.0))
    n_reps = {
        n
        for rec in records
        for n in rec["experiment"]["phenotype"]["n_replicates"].values()
    }
    report.add(
        LevelResult(
            level=Level.L2,
            name="single_acquisition",
            passed=n_reps == {1},
            message=f"n_replicates values {sorted(n_reps)} (the release: acquired once)",
            details={"n_replicates": sorted(n_reps)},
        )
    )
    shared, reference = _l3_shared_reference(records, "protein_abundance")
    report.add(shared)
    uncovered = _uncovered_keys(records, "protein_abundance", reference)
    report.add(
        l3_convention(
            "reference_covers_every_key",
            uncovered == 0,
            detail=f"{uncovered} measured cells have no euploid reference value",
        )
    )
    report.add(_l3_measurement_type_consistent(records))
    report.add(
        _l4_containment(
            _measured_keys(records, "protein_abundance"),
            sgd_genes,
            min_containment,
            "gene_containment_sgd",
        )
    )
    overlap = _strains(records) & caudal_isolates
    report.add(
        LevelResult(
            level=Level.L4,
            name="caudal_isolate_overlap",
            passed=len(overlap) == expected_caudal_overlap,
            message=(
                f"{len(overlap)} isolates also in the built Caudal 2024 store "
                f"(expected {expected_caudal_overlap})"
            ),
            details={"n_overlap": len(overlap), "expected": expected_caudal_overlap},
        )
    )
    report.add(_l4_caudal_genotype_agreement(records, caudal_genotypes))
    _add_genotype_audit(report, records, genotype_audit)
    return report


def verify_natural_isolate_turnover(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    sgd_genes: set[str],
    expected_rows_without_reference: int,
    genotype_audit: IsolateGenotypeAudit,
    min_containment: float = 0.99,
) -> VerificationReport:
    """L0-L4 for a natural-isolate turnover store (Muenzner 2024 SI Table 17).

    L2: rates and half-lives finite and > 0. L3: ``half_life`` equals ln 2 over the rate
    within ``HALF_LIFE_IDENTITY_RTOL`` (the Methods' definition), one shared reference,
    and the cells without a reference key number exactly what the build ledger counted.
    L4: protein keys are S288C names. ``genotype_audit`` adds the four isolate-genotype
    rules every natural-isolate store shares.
    """
    from torchcell.verification.levels import l3_convention
    from torchcell.verification.rnaseq import _l1_strain_uniqueness

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(
        l0_structural((rec["experiment"] for rec in records), validate_by_type_tag)
    )
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_strain_uniqueness(records, strain_of=natural_isolate_of))
    report.add(_l1_genotype_strain_consistency(records))
    rates = [
        float(v)
        for rec in records
        for v in rec["experiment"]["phenotype"]["degradation_rate"].values()
    ]
    report.add(l2_value_fidelity(rates, allow_nan=False, minimum=0.0))
    worst = 0.0
    n_pairs = 0
    for rec in records:
        phenotype = rec["experiment"]["phenotype"]
        for key, rate in phenotype["degradation_rate"].items():
            half_life = float(phenotype["half_life"][key])
            worst = max(worst, abs(math.log(2) / float(rate) - half_life) / half_life)
            n_pairs += 1
    report.add(
        l3_convention(
            "half_life_is_ln2_over_rate",
            worst <= HALF_LIFE_IDENTITY_RTOL,
            detail=f"max relative error {worst:.3g} over {n_pairs} (rate, HL) pairs",
        )
    )
    shared, reference = _l3_shared_reference(records, "degradation_rate")
    report.add(shared)
    uncovered = _uncovered_keys(records, "degradation_rate", reference)
    report.add(
        l3_convention(
            "uncovered_cells_match_ledger",
            uncovered == expected_rows_without_reference,
            detail=(
                f"{uncovered} measured cells without a euploid reference value "
                f"(ledger: {expected_rows_without_reference})"
            ),
        )
    )
    report.add(_l3_measurement_type_consistent(records))
    report.add(
        _l4_containment(
            _measured_keys(records, "degradation_rate"),
            sgd_genes,
            min_containment,
            "gene_containment_sgd",
        )
    )
    _add_genotype_audit(report, records, genotype_audit)
    return report


def _add_genotype_audit(
    report: VerificationReport, records: Sequence[Record], audit: IsolateGenotypeAudit
) -> None:
    """Feed each natural-isolate genotype (one per isolate here) to ``audit``; add its
    rules. A record with no perturbation and no descriptor whose reference names an
    S288C-derived background (FY4) is not an isolate of the Peter panel and is not fed;
    ``genotype_strain_consistency`` counts it apart.
    """
    for rec in records:
        genotype = rec["experiment"]["genotype"]
        if (
            not genotype["perturbations"]
            and genotype.get("natural_isolate") is None
            and natural_isolate_of(rec) is not None
        ):
            continue
        audit.add(genotype)
    for result in audit.results():
        report.add(result)


__all__ = [
    "verify_protein_dataset",
    "verify_natural_isolate_proteome",
    "verify_natural_isolate_turnover",
    "protein_gene_set",
    "Record",
]
