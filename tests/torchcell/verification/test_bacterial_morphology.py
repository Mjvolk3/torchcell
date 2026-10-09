# tests/torchcell/verification/test_bacterial_morphology.py
# [[tests.torchcell.verification.test_bacterial_morphology]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_bacterial_morphology.py
"""Unit tests for the bacterial-morphology L0-L4 verifier (synthetic, CI-safe).

Records are built from the real pydantic models, so the schema's own rules (symbol
membership, correct-statistic filing, no NaN) hold by construction and what is tested is
what the verifier adds: the per-feature L2 bound of each declared statistic, the
required-feature coverage that tolerates a non-determined field, the one-assay rule, and
the parity and reference checks.
"""

from __future__ import annotations

import math
from typing import Any, Final

import pytest

from torchcell.datamodels.bacterial_morphology_features import (
    CAMPOS2018_MORPHOLOGY_ASSAY as ASSAY,
)
from torchcell.datamodels.bacterial_morphology_features import (
    MorphologyAssay,
    MorphologyFeature,
    MorphologyFeatureGroup,
    MorphologyStatistic,
)
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialDeletionPerturbation,
    BacterialMorphologyExperiment,
    BacterialMorphologyExperimentReference,
    BacterialMorphologyPhenotype,
    Environment,
    Genotype,
    SampleUnit,
    Temperature,
)
from torchcell.verification.bacterial_morphology import (
    STATISTIC_BOUNDS,
    morphology_gene_set,
    verify_bacterial_morphology_dataset,
)
from torchcell.verification.report import Level, Provenance

PROV = Provenance(source_uri="test://synthetic", citation_key="test2018")
GENES = ["BW25113_0002", "BW25113_0004", "BW25113_0008"]
NAMESPACE: Final = "ecoli_k12_bw25113_locus_tag"
DAPI = frozenset(
    {"<NA>", "CV_NA", "rho_CD", "CDN_C0", "Rel.timing div", "Rel.timing nuc", "%2N"}
)
REQUIRED = frozenset(ASSAY.by_symbol) - DAPI
BY_STATISTIC = {
    MorphologyStatistic.mean: 2.5,
    MorphologyStatistic.coefficient_of_variation: 0.25,
    MorphologyStatistic.pearson_correlation: 0.6,
    MorphologyStatistic.regression_intercept: 0.4,
    MorphologyStatistic.fraction_of_cells: 0.2,
    MorphologyStatistic.inferred_relative_timing: 0.8,
}


def _profile(
    symbols: frozenset[str] | None = None, **overrides: float
) -> tuple[dict[str, float], dict[str, float]]:
    """The two dicts of a profile over ``symbols``, each value inside its own bound."""
    wanted = frozenset(ASSAY.by_symbol) if symbols is None else symbols
    values: dict[str, float] = {}
    coefficients: dict[str, float] = {}
    for feature in ASSAY.features:
        if feature.symbol not in wanted:
            continue
        number = overrides.get(feature.symbol, BY_STATISTIC[feature.statistic])
        if feature.statistic is MorphologyStatistic.coefficient_of_variation:
            coefficients[feature.symbol] = number
        else:
            values[feature.symbol] = number
    return values, coefficients


def _phenotype(
    symbols: frozenset[str] | None = None, **overrides: float
) -> BacterialMorphologyPhenotype:
    values, coefficients = _profile(symbols, **overrides)
    return BacterialMorphologyPhenotype(
        assay="campos2018",
        morphology=values,
        morphology_coefficient_of_variation=coefficients,
        n_samples=245,
        sample_unit=SampleUnit.cell,
    )


def _record(
    gene: str,
    *,
    phenotype: BacterialMorphologyPhenotype | None = None,
    reference_phenotype: BacterialMorphologyPhenotype | None = None,
) -> dict[str, Any]:
    # a MEDIA_LIBRARY medium, so the shared L3 media rule reports a library match
    # rather than free text and the whole report can pass
    environment = Environment(
        media=MEDIA_LIBRARY["M9"].model_copy(deep=True),
        temperature=Temperature(value=30),
    )
    experiment = BacterialMorphologyExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                BacterialDeletionPerturbation(
                    systematic_gene_name=gene,
                    perturbed_gene_name=gene,
                    gene_namespace=NAMESPACE,
                )
            ]
        ),
        environment=environment,
        phenotype=phenotype if phenotype is not None else _phenotype(),
    )
    reference = BacterialMorphologyExperimentReference(
        dataset_name="test",
        genome_reference=AssemblyReferenceGenome(
            species="Escherichia coli",
            strain="BW25113",
            assembly_set="ecoli_K12_BW25113_ASM75055v1",
            assembly_accession="GCA_000750555.1",
        ),
        environment_reference=environment.model_copy(),
        phenotype_reference=(
            reference_phenotype if reference_phenotype is not None else _phenotype()
        ),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _records() -> list[dict[str, Any]]:
    return [_record(gene) for gene in GENES]


def _verify(records: list[dict[str, Any]], **kwargs: Any) -> Any:
    options: dict[str, Any] = {
        "dataset_name": "synthetic",
        "provenance": PROV,
        "expected_count": len(records),
        "required_features": REQUIRED,
    }
    options.update(kwargs)
    return verify_bacterial_morphology_dataset(records, **options)


def _result(report: Any, name: str) -> Any:
    (found,) = [r for r in report.results if r.name == name]
    return found


def test_a_correct_dataset_passes_every_level() -> None:
    report = _verify(_records())
    assert report.passed, report.summary()
    assert {Level.L0, Level.L1, Level.L2, Level.L3} <= report.levels_covered
    assert _result(report, "value_fidelity").details["n_values"] == 3 * 26
    assert _result(report, "value_fidelity").details["n_features"] == 26
    assert _result(report, "cv_nonnegative").details["n_values"] == 3 * 11
    assert _result(report, "assay_coverage").details["coverage_strata"] == {"26": 3}


def test_a_record_missing_only_a_non_determined_feature_still_passes() -> None:
    """278 Campos strains have no DAPI channel; that is data, not a defect."""
    records = _records()
    records.append(_record("BW25113_0010", phenotype=_phenotype(REQUIRED)))
    report = _verify(records)
    assert report.passed, report.summary()
    coverage = _result(report, "assay_coverage")
    assert coverage.details["coverage_strata"] == {"19": 1, "26": 3}
    assert coverage.details["n_records_missing_required"] == 0


def test_a_record_missing_a_required_feature_fails_coverage() -> None:
    records = _records()
    records.append(_record("BW25113_0010", phenotype=_phenotype(REQUIRED - {"<L>"})))
    report = _verify(records)
    coverage = _result(report, "assay_coverage")
    assert not coverage.passed
    assert coverage.details["n_records_missing_required"] == 1
    assert coverage.details["missing_required"][0]["missing_required"] == ["<L>"]
    assert not report.passed


@pytest.mark.parametrize(
    ("symbol", "value"),
    [
        ("rho_CD", 1.5),  # a Pearson correlation above 1
        ("rho_CD", -1.5),  # and below -1
        ("%2N", 1.5),  # a fraction of cells above 1
        ("Rel.timing nuc", 1.5),  # a relative cell age beyond one cycle
        ("<L>", -1.0),  # a negative mean length
        ("CV_L", -0.1),  # a negative coefficient of variation
        ("<L>", math.inf),  # the schema blocks NaN, not inf
    ],
)
def test_a_value_outside_its_statistics_bound_fails_value_fidelity(
    symbol: str, value: float
) -> None:
    records = _records()
    phenotype = _phenotype(None, **{symbol: value})
    records.append(_record("BW25113_0010", phenotype=phenotype))
    report = _verify(records)
    fidelity = _result(report, "value_fidelity")
    assert not fidelity.passed
    assert fidelity.details["n_bad"] == 1
    (bad,) = fidelity.details["bad"]
    assert bad["feature"] == symbol
    assert bad["value"] == value
    assert not report.passed


def test_a_fitted_intercept_is_bounded_only_by_finiteness() -> None:
    """The one statistic whose definition implies no range, so none is asserted."""
    assert STATISTIC_BOUNDS[MorphologyStatistic.regression_intercept] is None
    records = _records()
    records.append(_record("BW25113_0010", phenotype=_phenotype(CDN_C0=-5.0)))
    assert _verify(records).passed
    records.append(_record("BW25113_0012", phenotype=_phenotype(CDN_C0=-math.inf)))
    assert not _result(_verify(records), "value_fidelity").passed


def test_a_negative_coefficient_of_variation_also_fails_its_own_check() -> None:
    records = _records()
    records.append(_record("BW25113_0010", phenotype=_phenotype(CV_W=-0.2)))
    report = _verify(records)
    negative = _result(report, "cv_nonnegative")
    assert not negative.passed
    assert negative.details["n_negative"] == 1
    assert negative.details["negative"][0]["feature"] == "CV_W"


def test_a_wrong_record_count_fails_l1() -> None:
    report = _verify(_records(), expected_count=99)
    assert not _result(report, "count").passed
    assert not report.passed


def test_two_records_on_one_strain_and_environment_fail_uniqueness() -> None:
    records = [*_records(), _record(GENES[0])]
    report = _verify(records)
    unique = _result(report, "pair_uniqueness")
    assert not unique.passed
    assert unique.details["n_duplicated"] == 1
    assert unique.details["n_strains"] == 3


def test_an_under_populated_reference_fails_l3() -> None:
    records = _records()
    records.append(
        _record("BW25113_0010", reference_phenotype=_phenotype(REQUIRED - {"CV_L"}))
    )
    report = _verify(records)
    populated = _result(report, "reference_populated")
    assert not populated.passed
    assert populated.details["n_short"] == 1
    assert populated.details["short"][0]["missing_required"] == ["CV_L"]


def test_the_vocabulary_parity_check_reads_the_named_assay() -> None:
    parity = _result(_verify(_records()), "vocabulary_parity")
    assert parity.passed
    assert "assay campos2018" in parity.message
    assert "15 value symbols + 11 CV symbols == 26 features" in parity.message


def test_a_dataset_mixing_two_assays_is_refused_before_any_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One dataset, one vocabulary: two assays' symbols are not comparable."""
    import torchcell.datamodels.bacterial_morphology_features as features

    other = MorphologyAssay(
        name="other2030",
        description="a second registered assay, for the mixing check",
        features=(
            MorphologyFeature(
                symbol="<L>",
                name="Mean cell length",
                group=MorphologyFeatureGroup.morphological,
                statistic=MorphologyStatistic.mean,
                unit="µm",
            ),
        ),
    )
    monkeypatch.setitem(features.MORPHOLOGY_ASSAYS, other.name, other)
    records = [
        *_records(),
        _record(
            "BW25113_0010",
            phenotype=BacterialMorphologyPhenotype(
                assay=other.name, morphology={"<L>": 2.5}
            ),
        ),
    ]
    with pytest.raises(ValueError, match="carries one assay vocabulary"):
        _verify(records)


def test_a_required_feature_outside_the_assay_is_refused() -> None:
    with pytest.raises(ValueError, match="are not features of assay campos2018"):
        _verify(_records(), required_features=frozenset({"not_a_symbol"}))


def test_the_shared_rules_run_and_the_l4_universe_is_the_caller_s() -> None:
    report = _verify(_records(), sgd_genes=set(GENES), gene_universe_label="BW25113")
    assert report.passed, report.summary()
    names = {result.name for result in report.results}
    assert {"provenance_gaps", "canonical_gene_names", "media_membership"} <= names
    containment = _result(report, "gene_containment_sgd")
    assert containment.passed
    assert "BW25113" in containment.message


def test_the_gene_set_is_the_perturbed_systematic_names() -> None:
    assert morphology_gene_set(_records()) == set(GENES)


def test_every_statistic_the_vocabulary_declares_has_a_bound_entry() -> None:
    """A new statistic without an entry would raise a KeyError mid-verification."""
    assert set(STATISTIC_BOUNDS) == set(MorphologyStatistic)
