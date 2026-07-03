"""Tests for the fitness uncertainty ontology (UncertaintyType, derive_se, fields)."""

import math

import pytest

from torchcell.datamodels.schema import (
    FitnessPhenotype,
    SampleUnit,
    UncertaintyType,
    derive_se,
)


# --------------------------------------------------------------------------- #
# derive_se: the derivation table
# --------------------------------------------------------------------------- #
def test_derive_se_as_is_kinds():
    assert derive_se(0.02, UncertaintyType.standard_error, None) == 0.02
    assert derive_se(0.02, UncertaintyType.bootstrap_se, None) == 0.02


def test_derive_se_sample_sd_divides_by_sqrt_n():
    assert derive_se(0.06, UncertaintyType.sample_sd, 4) == pytest.approx(0.03)


def test_derive_se_variance():
    assert derive_se(0.04, UncertaintyType.variance, 4) == pytest.approx(0.1)


def test_derive_se_ci95():
    assert derive_se(1.96, UncertaintyType.ci95, None) == pytest.approx(1.0, rel=1e-3)


def test_derive_se_none_when_unreported():
    assert derive_se(None, None, 4) is None


def test_derive_se_requires_n_for_divided_kinds():
    with pytest.raises(ValueError):
        derive_se(0.06, UncertaintyType.sample_sd, None)


# --------------------------------------------------------------------------- #
# FitnessPhenotype: strict reported<->type + auto-derived fitness_se
# --------------------------------------------------------------------------- #
def test_dmf_sample_sd_derives_se():
    # Costanzo DMF: reported sample SD over 4 colonies -> SE = sd/sqrt(4).
    ph = FitnessPhenotype(
        fitness=0.87,
        fitness_uncertainty=0.06,
        fitness_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=4,
        sample_unit=SampleUnit.colony,
    )
    assert ph.fitness_se == pytest.approx(0.03)


def test_smf_bootstrap_se_used_as_is():
    ph = FitnessPhenotype(
        fitness=0.95,
        fitness_uncertainty=0.02,
        fitness_uncertainty_type=UncertaintyType.bootstrap_se,
    )
    assert ph.fitness_se == 0.02


def test_explicit_se_not_overridden():
    ph = FitnessPhenotype(
        fitness=0.9,
        fitness_se=0.01,
        fitness_uncertainty=0.06,
        fitness_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=4,
        sample_unit=SampleUnit.colony,
    )
    assert ph.fitness_se == 0.01  # supplied value wins over derivation


def test_reported_without_type_rejected():
    with pytest.raises(ValueError):
        FitnessPhenotype(fitness=0.9, fitness_uncertainty=0.06)


def test_type_without_reported_rejected():
    with pytest.raises(ValueError):
        FitnessPhenotype(
            fitness=0.9, fitness_uncertainty_type=UncertaintyType.sample_sd
        )


def test_sample_sd_requires_n_and_unit():
    with pytest.raises(ValueError):
        FitnessPhenotype(
            fitness=0.9,
            fitness_uncertainty=0.06,
            fitness_uncertainty_type=UncertaintyType.sample_sd,
        )  # missing n_samples + sample_unit


def test_round_trip():
    ph = FitnessPhenotype(
        fitness=0.87,
        fitness_uncertainty=0.06,
        fitness_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=4,
        sample_unit=SampleUnit.colony,
    )
    assert ph == FitnessPhenotype(**ph.model_dump())
    assert ph.fitness_se is not None and not math.isnan(
        ph.fitness_se
    )  # derived, finite
