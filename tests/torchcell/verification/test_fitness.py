# tests/torchcell/verification/test_fitness.py
# [[tests.torchcell.verification.test_fitness]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_fitness.py
"""The single-mutant fitness verifier on hand-built, schema-valid records.

Fixture: three KanMX deletion strains (YAL001C, YBR085W, YJR155W) on the shared
``SC`` medium at 30 C, fitness 0.8 / 1.05 / 0.3 with ``fitness_se`` 0.02 and
``n_samples`` 2; every reference carries ``fitness`` 1.0. A correct dataset emits exactly
twelve results in this order: structural, count, pair_uniqueness, value_fidelity,
se_nonnegative, reference_one, then the six shared rules provenance_gaps,
canonical_gene_names, uncertainty_sanity, compound_identity, media_compound_identity,
media_membership; passing ``sgd_genes`` appends gene_containment_sgd and
current_genome_genes.

Derived expectations: with ``sgd_genes`` covering two of the three genes the aggregate
containment is 2/3 = 0.667, below the default floor 0.90 and above 0.5; a reference
fitness of 1.02 gives ``worst_abs_dev`` 0.02 and the message ``max|v-1|=0.02``
(``:.3g``); the L2 ``bad`` entries index into the FILTERED value list (None values are
dropped before indexing), so a negative fitness in the second record after a None in the
first sits at index 0.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from torchcell.datamodels.media import SC
from torchcell.datamodels.schema import (
    Compound,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    PhysicalFactor,
    ReferenceGenome,
    SampleUnit,
    SgaKanMxDeletionPerturbation,
    Temperature,
)
from torchcell.verification.fitness import fitness_gene_set, verify_fitness_dataset
from torchcell.verification.report import Level, LevelResult, Provenance

PROV = Provenance(source_uri="test://synthetic", citation_key="oduibhirTest2014")
GENES = ["YAL001C", "YBR085W", "YJR155W"]
FITNESS = [0.8, 1.05, 0.3]

SHARED_NAMES = [
    "provenance_gaps",
    "canonical_gene_names",
    "uncertainty_sanity",
    "compound_identity",
    "media_compound_identity",
    "media_membership",
]
FAMILY_NAMES = [
    "structural",
    "count",
    "pair_uniqueness",
    "value_fidelity",
    "se_nonnegative",
    "reference_one",
]


def _record(
    genes: list[str],
    fitness: float,
    *,
    se: float | None = 0.02,
    strain_id: str | None = None,
    temperature: float = 30.0,
    ref_fitness: float = 1.0,
    carbon_source: str | None = None,
    carbon_g_per_l: float = 2.0,
) -> dict[str, Any]:
    """One ``{experiment, reference}`` record for a (multi-)deletion strain."""
    perturbations: list[Any] = [
        (
            SgaKanMxDeletionPerturbation(
                systematic_gene_name=g, perturbed_gene_name=g, strain_id=strain_id
            )
            if strain_id is not None
            else KanMxDeletionPerturbation(
                systematic_gene_name=g, perturbed_gene_name=g
            )
        )
        for g in genes
    ]
    perturbed_environment: list[EnvironmentPerturbationType] = (
        []
        if carbon_source is None
        else [
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.carbon_source,
                agent=Compound(name=carbon_source),
                magnitude=Concentration(
                    value=carbon_g_per_l, unit=ConcentrationUnit.g_per_l
                ),
            )
        ]
    )
    environment = Environment(
        media=SC,
        temperature=Temperature(value=temperature),
        perturbations=perturbed_environment,
    )
    experiment = FitnessExperiment(
        dataset_name="test",
        genotype=Genotype(perturbations=perturbations),
        environment=environment,
        phenotype=FitnessPhenotype(
            fitness=fitness,
            fitness_se=se,
            n_samples=2,
            sample_unit=SampleUnit.biological_replicate,
        ),
    )
    reference = FitnessExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=environment.model_copy(),
        phenotype_reference=FitnessPhenotype(
            fitness=ref_fitness,
            n_samples=2,
            sample_unit=SampleUnit.biological_replicate,
        ),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _good_records() -> list[dict[str, Any]]:
    return [_record([g], f) for g, f in zip(GENES, FITNESS)]


def _verify(records: list[dict[str, Any]], **kwargs: Any) -> Any:
    return verify_fitness_dataset(
        records,
        dataset_name="fit",
        provenance=PROV,
        expected_count=len(records),
        **kwargs,
    )


def _result(report: Any, name: str) -> LevelResult:
    matches = [r for r in report.results if r.name == name]
    assert len(matches) == 1, [r.name for r in report.results]
    result: LevelResult = matches[0]
    return result


class _Resolution:
    def __init__(self, status: str, systematic_name: str | None) -> None:
        self.status = status
        self.systematic_name = systematic_name


def _retiring_resolver(name: str) -> _Resolution:
    """Every name is current except YJR155W, which the genome has retired."""
    if name == "YJR155W":
        return _Resolution("retired", "YJR155W")
    return _Resolution("current", name)


def test_good_dataset_emits_twelve_results_in_order_and_passes() -> None:
    report = _verify(_good_records())
    assert [r.name for r in report.results] == FAMILY_NAMES + SHARED_NAMES
    assert [r.level for r in report.results[:6]] == [
        Level.L0,
        Level.L1,
        Level.L1,
        Level.L2,
        Level.L2,
        Level.L3,
    ]
    assert report.passed is True
    assert _result(report, "count").details == {"observed": 3, "expected": 3}
    assert _result(report, "pair_uniqueness").details == {
        "n_pairs": 3,
        "n_duplicated": 0,
        "n_strains": 3,
        "n_environments": 1,
    }
    assert _result(report, "value_fidelity").details == {
        "n_values": 3,
        "n_bad": 0,
        "bad": [],
    }
    assert _result(report, "se_nonnegative").message == "3 values checked"
    assert _result(report, "reference_one").details == {
        "n_values": 3,
        "worst_abs_dev": 0.0,
    }
    assert _result(report, "canonical_gene_names").message == (
        "3 systematic names, one canonical spelling each "
        "(no resolver supplied: spelling checked, annotation not)"
    )
    assert _result(report, "uncertainty_sanity").message == (
        "0 labeled uncertainties, none a zero dispersion; 0 records report "
        "n_samples >= 2 with no uncertainty"
    )
    assert _result(report, "media_membership").details["matched_media"] == {
        SC.name: "library:SC"
    }
    assert _result(report, "compound_identity").details["n_identified"] == 0


def test_sgd_genes_add_l4_results_and_min_containment_is_forwarded() -> None:
    sgd = {"YAL001C", "YBR085W"}
    strict = _verify(_good_records(), sgd_genes=sgd)
    assert [r.name for r in strict.results] == FAMILY_NAMES + SHARED_NAMES + [
        "gene_containment_sgd",
        "current_genome_genes",
    ]
    containment = _result(strict, "gene_containment_sgd")
    assert containment.passed is False
    assert containment.details["overlap"] == pytest.approx(2 / 3)
    assert containment.message == (
        "0.667 of 3 measured genes are S288C reference genes (>= 0.9)"
    )
    off_genome = _result(strict, "current_genome_genes")
    assert off_genome.passed is False
    assert off_genome.details == {
        "n_missing_genes": 1,
        "n_records": 1,
        "missing_records": {"YJR155W": 1},
    }
    assert strict.passed is False

    relaxed = _verify(_good_records(), sgd_genes=sgd, min_containment=0.5)
    assert _result(relaxed, "gene_containment_sgd").passed is True
    assert _result(relaxed, "gene_containment_sgd").message.endswith("(>= 0.5)")


@pytest.mark.parametrize(
    "records",
    [
        [
            _record([], f, temperature=t)
            for f, t in ((0.9, 26.0), (1.0, 30.0), (0.7, 37.0))
        ],
        [_record([], 1.0)],
    ],
    ids=["three-temperatures", "one-wild-type"],
)
def test_a_dataset_with_no_gene_perturbations_fails_for_no_measured_genes(
    records: list[dict[str, Any]],
) -> None:
    """With ``sgd_genes`` given and no record carrying a gene perturbation, the report
    fails on ``measured_genes_present`` alone (issue #541 review).

    Both containment results pass vacuously and say the set is empty; the failing row
    names the true reason instead of a 0.000 overlap. Every other rule passes, so
    without that row the report would pass with nothing measured.
    """
    report = _verify(records, sgd_genes={"YAL001C"})
    assert [r.name for r in report.results][-3:] == [
        "measured_genes_present",
        "gene_containment_sgd",
        "current_genome_genes",
    ]
    assert [r.name for r in report.results if not r.passed] == [
        "measured_genes_present"
    ]
    assert report.passed is False
    assert _result(report, "measured_genes_present").message == (
        f"no measured genes: none of the {len(records)} records carries a gene "
        "perturbation outside the background genes, so the SGD gene rules have "
        "nothing to check"
    )
    assert _result(report, "gene_containment_sgd").message == (
        "no measured genes (the measured gene set is empty); containment holds "
        "vacuously"
    )
    assert _result(report, "current_genome_genes").message == (
        "no measured genes (the measured gene set is empty); genome membership holds "
        "vacuously"
    )


def test_resolver_is_forwarded_to_the_canonical_name_rule() -> None:
    report = _verify(_good_records(), resolve_gene_name=_retiring_resolver)
    names = _result(report, "canonical_gene_names")
    assert names.passed is False
    assert names.details["not_current"] == ["YJR155W (retired -> YJR155W)"]
    assert names.details["resolver"] is True
    # the common name IS the systematic name here, so the retired gene is also reported
    # as a common name the resolver cannot place (reported, not failed)
    assert names.details["unresolved_common_names"] == [
        "YJR155W (retired; stored YJR155W)"
    ]
    assert names.details["common_name_mismatch"] == []
    assert names.message == (
        "0 genes carry conflicting common-name spellings (0 records; 0 case-only); "
        "1 systematic names are not the genome's current name; "
        "0 common names resolve to another gene"
    )


def test_pair_uniqueness_keys_on_strain_id_and_environment() -> None:
    duplicated = _verify([_record(["YAL001C"], 0.8), _record(["YAL001C"], 0.9)])
    dup = _result(duplicated, "pair_uniqueness")
    assert dup.passed is False
    assert dup.details == {
        "n_pairs": 1,
        "n_duplicated": 1,
        "n_strains": 1,
        "n_environments": 1,
    }
    assert dup.message == "1 (strain, environment) pairs appear in multiple records"

    allelic = _verify(
        [
            _record(["YAL001C"], 0.8, strain_id="tsq-1"),
            _record(["YAL001C"], 0.9, strain_id="tsq-2"),
        ]
    )
    assert _result(allelic, "pair_uniqueness").details == {
        "n_pairs": 2,
        "n_duplicated": 0,
        "n_strains": 2,
        "n_environments": 1,
    }

    two_temps = _verify(
        [
            _record(["YAL001C"], 0.8, temperature=30.0),
            _record(["YAL001C"], 0.6, temperature=37.0),
        ]
    )
    assert _result(two_temps, "pair_uniqueness").message == (
        "2 unique (strain, environment) records, one each"
    )


def test_pair_uniqueness_counts_an_environment_perturbation_as_a_condition() -> None:
    """One strain on two carbon sources is two records, not one repeated twice.

    The Tong 2020 case: every strain is grown on one medium at one temperature for one
    duration, and the thirty conditions differ only in an
    ``EnvironmentPhysicalPerturbation(factor=carbon_source)``. Keyed on the scalars
    alone, each strain read as thirty duplicates and L1 failed on a dataset holding
    exactly one record per (strain, carbon source).
    """
    varied = _verify(
        [
            _record(["YAL001C"], 0.8, carbon_source="D-glucose"),
            _record(["YAL001C"], 0.4, carbon_source="D-xylose"),
        ]
    )
    result = _result(varied, "pair_uniqueness")
    assert result.passed is True
    assert result.details == {
        "n_pairs": 2,
        "n_duplicated": 0,
        "n_strains": 1,
        "n_environments": 2,
    }

    # the DOSE is part of the condition, as it is in the environment-response verifier
    two_doses = _verify(
        [
            _record(["YAL001C"], 0.8, carbon_source="D-glucose", carbon_g_per_l=2.0),
            _record(["YAL001C"], 0.5, carbon_source="D-glucose", carbon_g_per_l=0.2),
        ]
    )
    assert _result(two_doses, "pair_uniqueness").passed is True

    # and a genuine double-count is still caught: same strain, same source, same dose
    repeated = _verify(
        [
            _record(["YAL001C"], 0.8, carbon_source="D-glucose"),
            _record(["YAL001C"], 0.5, carbon_source="D-glucose"),
        ]
    )
    assert _result(repeated, "pair_uniqueness").passed is False

    # an unperturbed environment is unchanged: the key gains an empty tuple
    plain = _verify([_record(["YAL001C"], 0.8), _record(["YBR085W"], 0.9)])
    assert _result(plain, "pair_uniqueness").details == {
        "n_pairs": 2,
        "n_duplicated": 0,
        "n_strains": 2,
        "n_environments": 1,
    }


def test_value_fidelity_skips_none_then_indexes_negative_and_nan() -> None:
    records = _good_records()
    records[0]["experiment"]["phenotype"]["fitness"] = None
    records[1]["experiment"]["phenotype"]["fitness"] = -0.5
    records[2]["experiment"]["phenotype"]["fitness"] = math.nan
    fidelity = _result(_verify(records), "value_fidelity")
    assert fidelity.passed is False
    assert fidelity.message == "2/2 values invalid"
    assert fidelity.details == {
        "n_values": 2,
        "n_bad": 2,
        "bad": [
            {"index": 0, "value": -0.5, "reason": "< 0.0"},
            {"index": 1, "value": "nan", "reason": "nan"},
        ],
    }


def test_se_nonnegative_skips_nan_and_none_and_flags_negative() -> None:
    records = [
        _record(["YAL001C"], 0.8, se=math.nan),
        _record(["YBR085W"], 1.05, se=None),
        _record(["YJR155W"], 0.3, se=0.05),
    ]
    records[2]["experiment"]["phenotype"]["fitness_se"] = -0.1
    se = _result(_verify(records), "se_nonnegative")
    assert se.level is Level.L2
    assert se.passed is False
    assert se.details == {
        "n_values": 1,
        "n_bad": 1,
        "bad": [{"index": 0, "value": -0.1, "reason": "< 0.0"}],
    }


def test_reference_one_reports_the_worst_deviation_and_skips_none() -> None:
    records = [
        _record(["YAL001C"], 0.8, ref_fitness=1.02),
        _record(["YBR085W"], 1.05),
        _record(["YJR155W"], 0.3),
    ]
    records[1]["reference"]["phenotype_reference"]["fitness"] = None
    reference = _result(_verify(records), "reference_one")
    assert reference.passed is False
    assert reference.details["n_values"] == 2
    assert reference.details["worst_abs_dev"] == pytest.approx(0.02)
    assert reference.message == "reference fitness not identically 1.0: max|v-1|=0.02"


def test_uncertainty_sanity_counts_replicated_records_without_an_uncertainty() -> None:
    records = [_record([g], f, se=None) for g, f in zip(GENES, FITNESS)]
    sanity = _result(_verify(records), "uncertainty_sanity")
    assert sanity.passed is True
    assert sanity.details["n_no_uncertainty_with_replicates"] == 3
    assert sanity.details["n_checked"] == 0
    # verbatim from common.py, American spelling since issue #541
    assert sanity.message == (
        "0 labeled uncertainties, none a zero dispersion; 3 records report "
        "n_samples >= 2 with no uncertainty"
    )


def test_wrong_count_and_an_invalid_record_fail_their_own_levels() -> None:
    records = _good_records()
    records[2]["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"] = (
        "not-a-gene"
    )
    report = verify_fitness_dataset(
        records, dataset_name="fit", provenance=PROV, expected_count=99
    )
    structural = _result(report, "structural")
    assert structural.passed is False
    assert structural.message == "1/3 records failed schema validation"
    assert structural.details["n_failures"] == 1
    assert structural.details["failures"][0]["index"] == 2
    assert _result(report, "count").details == {"observed": 3, "expected": 99}
    assert _result(report, "count").passed is False


def test_fitness_gene_set_unions_every_perturbation() -> None:
    records = [_record(["YAL001C", "YBR085W"], 0.5), _record(["YJR155W"], 0.3)]
    assert fitness_gene_set(records) == {"YAL001C", "YBR085W", "YJR155W"}
    assert fitness_gene_set([]) == set()
