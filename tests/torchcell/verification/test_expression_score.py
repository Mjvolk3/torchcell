# tests/torchcell/verification/test_expression_score.py
# [[tests.torchcell.verification.test_expression_score]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_expression_score.py
"""The error-model score family verifier (#1022) on hand-built records.

Records are dumped from real schema objects, as a loader writes them. The L4 cases are
built so every number is derivable by hand: six strains whose second-source profile is
their own profile have matched r = 1 and, against each other, the mismatched r of six
fixed vectors; a seventh check swaps the second source's labels so the matched pairs are
the mismatched ones.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    MicroarrayExpressionScoreExperiment,
    MicroarrayExpressionScoreExperimentReference,
    MicroarrayExpressionScorePhenotype,
    ReferenceGenome,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.expression_score import (
    deleted_gene,
    l4_matched_profile_agreement,
    matched_profile_correlations,
    verify_expression_score_dataset,
)
from torchcell.verification.report import Level, Provenance

PROVENANCE = Provenance(source_uri="synthetic")


def _phenotype(
    scores: dict[str, float], p_values: dict[str, float] | None
) -> MicroarrayExpressionScorePhenotype:
    return MicroarrayExpressionScorePhenotype(
        expression_score=scores,
        expression_score_p_value=p_values,
        measurement_type="error_model_x_score",
        reference_basis="wild type",
        array_platform="GPL0",
        n_replicates=2,
    )


def _record(
    gene: str, scores: dict[str, float], *, reference_value: float = 0.0
) -> dict[str, Any]:
    env = Environment(media=YPD_LIQUID)
    p_values = {g: 2.0 * float(norm.sf(abs(x))) for g, x in scores.items()}
    experiment = MicroarrayExpressionScoreExperiment(
        dataset_name="t",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=gene
                )
            ]
        ),
        environment=env,
        phenotype=_phenotype(scores, p_values),
    )
    reference = MicroarrayExpressionScoreExperimentReference(
        dataset_name="t",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4741",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        ),
        environment_reference=env,
        phenotype_reference=_phenotype(dict.fromkeys(scores, reference_value), None),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _names(report: Any) -> dict[str, Any]:
    return {result.name: result for result in report.results}


def test_a_clean_dataset_passes_every_family_check() -> None:
    records = [
        _record("YAL001C", {"YAL001C": -3.0, "YBR001C": 1.0}),
        _record("YBR001C", {"YAL001C": 0.5, "YBR001C": -1.0}),
    ]
    report = verify_expression_score_dataset(
        records,
        dataset_name="t",
        provenance=PROVENANCE,
        expected_count=2,
        two_sided_normal_tol=1e-12,
    )
    assert report.passed
    results = _names(report)
    assert list(results) == [
        "structural",
        "count",
        "score_finite",
        "p_value_in_unit_interval",
        "p_value_keys_match_scores",
        "reference_score_zero",
        "deletion_lowers_score",
        "p_value_is_two_sided_normal",
    ]
    assert results["deletion_lowers_score"].message == (
        "median deleted-gene score=-2.000 (<0 => correct orientation); frac_neg=1.000 "
        "over 2 deleted genes (0 deleted genes carry no score in their own record)"
    )


def test_a_sign_inversion_and_a_nonzero_reference_fail_l3() -> None:
    records = [
        _record("YAL001C", {"YAL001C": 3.0, "YBR001C": 1.0}, reference_value=0.5),
        _record("YBR001C", {"YAL001C": 0.5}),
    ]
    report = verify_expression_score_dataset(
        records, dataset_name="t", provenance=PROVENANCE, expected_count=3
    )
    results = _names(report)
    assert not report.passed
    assert not results["count"].passed
    assert results["reference_score_zero"].message == (
        "max|reference score|=0.5; 0 references carry p-values"
    )
    assert results["deletion_lowers_score"].message == (
        "median deleted-gene score=3.000 (<0 => correct orientation); frac_neg=0.000 "
        "over 1 deleted genes (1 deleted genes carry no score in their own record)"
    )
    # without a tolerance the two-sided rule is not asserted
    assert "p_value_is_two_sided_normal" not in results


def test_the_two_sided_rule_reports_its_worst_deviation() -> None:
    record = _record("YAL001C", {"YAL001C": -2.0})
    record["experiment"]["phenotype"]["expression_score_p_value"]["YAL001C"] = 0.05
    report = verify_expression_score_dataset(
        [record],
        dataset_name="t",
        provenance=PROVENANCE,
        expected_count=1,
        two_sided_normal_tol=5e-7,
    )
    result = _names(report)["p_value_is_two_sided_normal"]
    expected = abs(2.0 * float(norm.sf(2.0)) - 0.05)
    assert not result.passed
    assert result.message == (
        f"max|p - 2*(1 - Phi(|score|))| = {expected:.3g} over 1 pairs "
        "(tolerance 5e-07, the release's rounding)"
    )


def test_a_p_value_map_keyed_unlike_the_scores_fails_l2() -> None:
    record = _record("YAL001C", {"YAL001C": -2.0, "YBR001C": 0.1})
    del record["experiment"]["phenotype"]["expression_score_p_value"]["YBR001C"]
    report = verify_expression_score_dataset(
        [record], dataset_name="t", provenance=PROVENANCE, expected_count=1
    )
    result = _names(report)["p_value_keys_match_scores"]
    assert (result.passed, result.details["bad_indices"]) == (False, [0])


def test_deleted_gene_is_none_for_a_multi_perturbation_genotype() -> None:
    experiment = {
        "genotype": {
            "perturbations": [
                {"systematic_gene_name": "YAL001C"},
                {"systematic_gene_name": "YBR001C"},
            ]
        }
    }
    assert deleted_gene(experiment) is None
    assert deleted_gene({"genotype": {"perturbations": [{}]}}) is None


# --------------------------------------------------------------------------- #
# L4: matched-strain profile agreement
# --------------------------------------------------------------------------- #
_STRAINS = [f"YAL00{i}C" for i in range(1, 7)]
_GENES = [f"YHR0{i:02d}W" for i in range(1, 21)]
_RNG = np.random.default_rng(7)
_VECTORS = {s: _RNG.normal(size=len(_GENES)) for s in _STRAINS}


def _profiles(labels: dict[str, str]) -> dict[str, dict[str, float]]:
    """Strain -> profile, the profile of ``labels[strain]`` plus the deleted gene."""
    return {
        s: {**dict(zip(_GENES, _VECTORS[labels[s]].tolist())), s: -5.0}
        for s in _STRAINS
    }


def _score_records(profiles: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    return [
        {
            "experiment": {
                "genotype": {"perturbations": [{"systematic_gene_name": s}]},
                "phenotype": {"expression_score": p},
            }
        }
        for s, p in profiles.items()
    ]


def _ratio_records(profiles: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    return [
        {
            "experiment": {
                "genotype": {"perturbations": [{"systematic_gene_name": s}]},
                "phenotype": {"expression_log2_ratio": p},
            }
        }
        for s, p in profiles.items()
    ]


def test_matched_correlations_exclude_every_deleted_gene() -> None:
    identity = {s: s for s in _STRAINS}
    matched, mismatched, n_genes = matched_profile_correlations(
        _profiles(identity), _profiles(identity)
    )
    assert n_genes == len(_GENES)
    assert all(math.isclose(r, 1.0) for r in matched.values())
    expected = [
        float(np.corrcoef(_VECTORS[a], _VECTORS[b])[0, 1])
        for a in _STRAINS
        for b in _STRAINS
        if a != b
    ]
    assert np.allclose(mismatched, expected)


def test_l4_passes_when_matched_pairs_beat_mismatched_ones() -> None:
    identity = {s: s for s in _STRAINS}
    result = l4_matched_profile_agreement(
        _score_records(_profiles(identity)),
        _ratio_records(_profiles(identity)),
        other_name="other",
        other_field="expression_log2_ratio",
        alpha=1e-3,
    )
    assert result.level is Level.L4
    assert result.name == "matched_profile_agreement_other"
    assert result.passed
    assert result.details["n_shared"] == 6
    assert result.details["n_mismatched"] == 30
    assert result.details["mann_whitney_u"] == 180.0
    assert math.isclose(result.details["median_matched_r"], 1.0)


def test_l4_fails_when_the_second_source_labels_are_shuffled() -> None:
    shifted = {s: _STRAINS[(i + 1) % 6] for i, s in enumerate(_STRAINS)}
    result = l4_matched_profile_agreement(
        _score_records(_profiles({s: s for s in _STRAINS})),
        _ratio_records(_profiles(shifted)),
        other_name="other",
        other_field="expression_log2_ratio",
        alpha=1e-3,
    )
    assert not result.passed
    assert result.details["mann_whitney_p"] > 1e-3


def test_l4_refuses_too_few_shared_genes_rather_than_dropping_them() -> None:
    hu = _score_records({"YAL001C": {"YAL001C": -2.0}, "YAL002C": {"YHR001W": 1.0}})
    other = _ratio_records({"YAL001C": {"YHR001W": 0.5}, "YAL002C": {"YHR001W": 0.2}})
    result = l4_matched_profile_agreement(
        hu, other, other_name="other", other_field="expression_log2_ratio", alpha=0.5
    )
    assert not result.passed
    assert result.message == "2 shared strains; too few or undefined correlations"


@pytest.mark.parametrize("n_shared", [0, 1])
def test_l4_refuses_fewer_than_two_shared_strains(n_shared: int) -> None:
    hu = _score_records(_profiles({s: s for s in _STRAINS}))
    other = _ratio_records(
        {s: p for s, p in _profiles({s: s for s in _STRAINS}).items()}
    )[:n_shared]
    result = l4_matched_profile_agreement(
        hu, other, other_name="o", other_field="expression_log2_ratio", alpha=0.5
    )
    assert (result.passed, result.details["n_shared"]) == (False, n_shared)
