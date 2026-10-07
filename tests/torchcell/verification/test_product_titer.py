# tests/torchcell/verification/test_product_titer.py
# [[tests.torchcell.verification.test_product_titer]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_product_titer.py
"""The shared product-titer family verifier, on synthetic records.

Every record here is a real ``ProductTiterExperiment`` dump: a two-gene heterologous
pathway plus one host deletion on a KT2440 assembly pin, one culture of a product
measured in ``ug/mL``. The baseline pair is the Carruthers design (a released sample SD
with a derived ``titer_se``) and the gapped pair is the Kang design (the replicate
design sourced or typed, no uncertainty number anywhere), so one battery covers both.

Derived expectations: with SD 3.0 over n=4 the derived SE is exactly 1.5, so the L2
identity holds at any tolerance and fails by 0.2 when the stored SE is nudged to 1.7.
A report carries eleven levels, one per rule, and ``passed`` is the conjunction.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from torchcell.datamodels.media import M9_NREL_CARRUTHERS2025
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialDeletionPerturbation,
    Compound,
    ConcentrationUnit,
    CultureEnvironment,
    Genotype,
    HeterologousPathwayPerturbation,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
    SampleUnit,
    Temperature,
    UncertaintyType,
)
from torchcell.verification.product_titer import verify_product_titer_dataset
from torchcell.verification.report import Level, Provenance
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

Record = dict[str, Any]
PROV = Provenance(source_uri="test://synthetic", citation_key="productTiterTest2026")
PRODUCT = "isoprenol"
PATHWAY_GENES = ("MvaSEf", "MvaEEf")
HOST_DELETION = "PP_5003"
NAMESPACE: Literal["pputida_kt2440_locus_tag"] = "pputida_kt2440_locus_tag"
REFERENCE = AssemblyReferenceGenome(
    species="Pseudomonas putida",
    strain="IY1449b",
    assembly_set="pputida_KT2440_ASM756v2",
    assembly_accession="GCA_000007565.2",
)


def _environment() -> CultureEnvironment:
    return CultureEnvironment(
        media=M9_NREL_CARRUTHERS2025,
        temperature=Temperature(value=24.0),
        duration_hours=48.0,
    )


def _genotype() -> Genotype:
    return Genotype(
        perturbations=[
            *(
                HeterologousPathwayPerturbation(
                    systematic_gene_name=token,
                    perturbed_gene_name=token,
                    source_organism="Enterococcus faecalis",
                    is_heterologous=True,
                    localization="plasmid",
                    gene_namespace=NAMESPACE,
                    pathway_name="isoprenol via mevalonate",
                )
                for token in PATHWAY_GENES
            ),
            BacterialDeletionPerturbation(
                systematic_gene_name=HOST_DELETION,
                perturbed_gene_name=HOST_DELETION,
                gene_namespace=NAMESPACE,
            ),
        ]
    )


def _phenotype(
    titer: float, *, uncertainty: float | None, n_samples: int | None
) -> ProductTiterPhenotype:
    """A released-SD phenotype, or the all-gapped one a design-only source gives."""
    if uncertainty is None:
        return ProductTiterPhenotype(
            product=Compound(name=PRODUCT),
            titer=titer,
            titer_unit=ConcentrationUnit.ug_per_ml,
            n_samples=None,
            sample_unit=None,
            provenance_gaps=[
                ProvenanceGap(
                    field=field, reason=ProvenanceGapReason.not_reported_by_primary
                )
                for field in (
                    "titer_uncertainty",
                    "titer_uncertainty_type",
                    "n_samples",
                    "sample_unit",
                )
            ],
        )
    assert n_samples is not None
    return ProductTiterPhenotype(
        product=Compound(name=PRODUCT),
        titer=titer,
        titer_unit=ConcentrationUnit.ug_per_ml,
        titer_uncertainty=uncertainty,
        titer_uncertainty_type=UncertaintyType.sample_sd,
        titer_se=uncertainty / n_samples**0.5,
        n_samples=n_samples,
        sample_unit=SampleUnit.biological_replicate,
    )


def _record(
    titer: float, *, uncertainty: float | None = 3.0, n_samples: int | None = 4
) -> Record:
    phenotype = _phenotype(titer, uncertainty=uncertainty, n_samples=n_samples)
    return {
        "experiment": ProductTiterExperiment(
            dataset_name="synthetic_titer",
            genotype=_genotype(),
            environment=_environment(),
            phenotype=phenotype,
        ).model_dump(),
        "reference": ProductTiterExperimentReference(
            dataset_name="synthetic_titer",
            genome_reference=REFERENCE,
            environment_reference=_environment(),
            phenotype_reference=_phenotype(100.0, uncertainty=1.0, n_samples=4),
        ).model_dump(),
    }


def _verify(records: list[Record], **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "dataset_name": "synthetic_titer",
        "provenance": PROV,
        "expected_count": len(records),
        "titer_unit": ConcentrationUnit.ug_per_ml.value,
        "titer_unit_detail": "1 mg/L == 1 ug/mL exactly",
        "se_tol": 1e-9,
        "pathway_gene_counts": (len(PATHWAY_GENES),),
        "product_names": (PRODUCT,),
    }
    kwargs.update(overrides)
    return verify_product_titer_dataset(records, **kwargs)


RULE_NAMES = [
    "structural",
    "count",
    "value_fidelity",
    "uncertainty_nonnegative",
    "se_is_the_uncertainty_over_sqrt_n",
    "titer_unit_is_the_pinned_unit",
    "uncertainty_is_typed_or_gapped",
    "replicate_design_is_sourced_or_gapped",
    "heterologous_pathway_gene_counts",
    "product_is_the_declared_one",
]


def test_the_gate_names_every_rule_once_and_covers_l0_to_l3() -> None:
    """Eleven rules in a fixed order; L4 is the caller's."""
    report = _verify([_record(200.0), _record(150.0, uncertainty=None, n_samples=None)])
    assert [result.name for result in report.results] == RULE_NAMES
    assert report.levels_covered == {Level.L0, Level.L1, Level.L2, Level.L3}
    assert report.passed


def test_the_se_identity_counts_released_pairs_and_gapped_records_apart() -> None:
    """Three released SDs are checked value by value; one gapped record derives none."""
    records = [
        _record(200.0),
        _record(210.0),
        _record(220.0),
        _record(150.0, uncertainty=None, n_samples=None),
    ]
    result = next(
        r
        for r in _verify(records).results
        if r.name == "se_is_the_uncertainty_over_sqrt_n"
    )
    assert result.passed
    assert result.details["n_pairs"] == 3
    assert result.details["n_without_uncertainty"] == 1
    assert result.details["worst_abs_diff"] == 0.0
    assert "3 pairs agree within 1e-09" in result.message


def test_the_se_identity_fails_a_stored_se_that_is_not_sd_over_sqrt_n() -> None:
    """1.7 against the 1.5 that SD 3.0 over n=4 implies: a 0.2 disagreement, named."""
    records = [_record(200.0), _record(210.0)]
    records[1]["experiment"]["phenotype"]["titer_se"] = 1.7
    result = next(
        r
        for r in _verify(records).results
        if r.name == "se_is_the_uncertainty_over_sqrt_n"
    )
    assert not result.passed
    assert result.message == "1 of 2 records break the derivation"
    assert result.details["worst"] == [
        {"index": 1, "titer_se": 1.7, "expected": 1.5, "diff": pytest.approx(0.2)}
    ]
    assert result.details["worst_abs_diff"] == pytest.approx(0.2)


def test_the_se_identity_fails_an_se_derived_from_no_uncertainty() -> None:
    """A gapped record that still carries a titer_se is a fabricated precision."""
    records = [_record(150.0, uncertainty=None, n_samples=None)]
    records[0]["experiment"]["phenotype"]["titer_se"] = 0.9
    result = next(
        r
        for r in _verify(records).results
        if r.name == "se_is_the_uncertainty_over_sqrt_n"
    )
    assert not result.passed
    assert result.details["worst"] == [
        {"index": 0, "reason": "titer_se without an uncertainty"}
    ]


def test_a_titer_below_zero_and_a_count_off_the_oracle_both_fail() -> None:
    """L2 fidelity is floored at zero and L1 is the loader's own oracle.

    The schema forbids a negative titer too, so L0 catches the same cell: three rules
    report it, which is what a gate is for.
    """
    records = [_record(200.0)]
    records[0]["experiment"]["phenotype"]["titer"] = -1.0
    report = _verify(records, expected_count=2)
    failed = {result.name for result in report.results if not result.passed}
    assert failed == {"structural", "count", "value_fidelity"}


def test_a_negative_released_uncertainty_fails_its_own_rule() -> None:
    """The uncertainty rule is separate so a gapped record never dilutes it."""
    records = [_record(200.0), _record(150.0, uncertainty=None, n_samples=None)]
    records[0]["experiment"]["phenotype"]["titer_uncertainty"] = -3.0
    report = _verify(records)
    uncertainty = next(r for r in report.results if r.name == "uncertainty_nonnegative")
    assert not uncertainty.passed
    assert uncertainty.details["n_values"] == 1


def test_the_unit_rule_pins_the_dataset_s_own_decision() -> None:
    """A store in mM against a ug/mL pin fails, and the detail names both."""
    records = [_record(200.0)]
    records[0]["experiment"]["phenotype"]["titer_unit"] = (
        ConcentrationUnit.millimolar.value
    )
    result = next(
        r for r in _verify(records).results if r.name == "titer_unit_is_the_pinned_unit"
    )
    assert not result.passed
    assert result.message == "stored units ['mM']; 1 mg/L == 1 ug/mL exactly"


def test_an_uncertainty_number_without_its_type_is_refused() -> None:
    """Both halves are stored or both are gapped; a bare number is unreadable."""
    records = [_record(200.0)]
    records[0]["experiment"]["phenotype"]["titer_uncertainty_type"] = None
    report = _verify(records)
    assert not next(
        r for r in report.results if r.name == "uncertainty_is_typed_or_gapped"
    ).passed


def test_an_unsourced_replicate_count_without_a_gap_is_refused() -> None:
    """A silent None is indistinguishable from 'not applicable', so it fails."""
    records = [_record(150.0, uncertainty=None, n_samples=None)]
    records[0]["experiment"]["phenotype"]["provenance_gaps"] = [
        gap
        for gap in records[0]["experiment"]["phenotype"]["provenance_gaps"]
        if gap["field"] not in {"n_samples", "sample_unit"}
    ]
    report = _verify(records)
    failed = {result.name for result in report.results if not result.passed}
    assert failed == {"replicate_design_is_sourced_or_gapped"}


def test_a_lost_pathway_gene_and_a_second_product_both_fail() -> None:
    """A production strain missing its pathway, and two products under one label."""
    records = [_record(200.0), _record(210.0)]
    perturbations = records[0]["experiment"]["genotype"]["perturbations"]
    records[0]["experiment"]["genotype"]["perturbations"] = [
        p for p in perturbations if p["perturbation_type"] != "heterologous_pathway"
    ]
    records[1]["experiment"]["phenotype"]["product"]["name"] = "isoprenyl acetate"
    report = _verify(records)
    failed = {result.name for result in report.results if not result.passed}
    assert failed == {"heterologous_pathway_gene_counts", "product_is_the_declared_one"}


def test_several_declared_pathway_counts_are_all_admissible() -> None:
    """Kang's integration variants carry 6, 11 or 13 pathway genes, not one count."""
    records = [_record(200.0), _record(210.0)]
    records[0]["experiment"]["genotype"]["perturbations"].append(
        records[0]["experiment"]["genotype"]["perturbations"][0]
    )
    report = _verify(records, pathway_gene_counts=(2, 3))
    assert report.passed
    result = next(
        r for r in report.results if r.name == "heterologous_pathway_gene_counts"
    )
    assert result.message == (
        "per-record heterologous pathway gene counts [2, 3]; the dataset declares [2, 3]"
    )


def test_a_record_that_does_not_validate_fails_l0_only() -> None:
    """L0 reports the schema failure as data; the value rules still read the dump."""
    records = [_record(200.0)]
    records[0]["experiment"]["phenotype"]["titer_unit"] = "furlongs"
    report = _verify(records)
    structural = next(r for r in report.results if r.name == "structural")
    assert not structural.passed
    assert structural.details["n_failures"] == 1
