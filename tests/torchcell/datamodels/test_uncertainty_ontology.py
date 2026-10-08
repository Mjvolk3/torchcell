"""Tests for the fitness uncertainty ontology (UncertaintyType, derive_se, fields)."""

import math

import pytest

from torchcell.datamodels.schema import (
    FitnessPhenotype,
    PromoterActivityPhenotype,
    ReporterReadout,
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


# --------------------------------------------------------------------------- #
# PromoterActivityPhenotype: the same ontology on a reporter signal
# --------------------------------------------------------------------------- #
def _activity(**kw: object) -> PromoterActivityPhenotype:
    """A reporter reading with the fields every record of the family carries."""
    fields: dict[str, object] = dict(
        promoter_activity=12.98,
        promoter_name="thrA",
        readout=ReporterReadout.plate_reader_fluorescence,
        reporter_gene="gfp",
        activity_units="GFP fluorescence in the reader's own units",
        well_id="Untreated|AZ01|A1",
    )
    fields.update(kw)
    return PromoterActivityPhenotype(**fields)  # type: ignore[arg-type]  # a kwargs table, validated by pydantic


def test_promoter_activity_derives_the_se_from_a_reported_sd() -> None:
    phenotype = _activity(
        promoter_activity_uncertainty=0.4,
        promoter_activity_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=4,
        sample_unit=SampleUnit.biological_replicate,
    )
    assert phenotype.promoter_activity_se == pytest.approx(0.2)


def test_promoter_activity_keeps_a_supplied_se_over_the_derivation() -> None:
    phenotype = _activity(
        promoter_activity_se=0.01,
        promoter_activity_uncertainty=0.4,
        promoter_activity_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=4,
        sample_unit=SampleUnit.biological_replicate,
    )
    assert phenotype.promoter_activity_se == 0.01


def test_promoter_activity_leaves_the_se_unfilled_when_an_sd_has_no_n() -> None:
    """sample_sd divides by sqrt(n), so with no n there is nothing to derive."""
    phenotype = PromoterActivityPhenotype.model_construct(
        promoter_activity=12.98,
        promoter_activity_uncertainty=0.4,
        promoter_activity_uncertainty_type=UncertaintyType.sample_sd,
        promoter_name="thrA",
        readout=ReporterReadout.plate_reader_fluorescence,
        reporter_gene="gfp",
        activity_units="a.u.",
        well_id="Untreated|AZ01|A1",
    )
    assert phenotype.promoter_activity_se is None


def test_promoter_activity_refuses_an_unlabelled_uncertainty() -> None:
    with pytest.raises(ValueError, match="must both be set or both"):
        _activity(promoter_activity_uncertainty=0.4)
    with pytest.raises(ValueError, match="must both be set or both"):
        _activity(promoter_activity_uncertainty_type=UncertaintyType.sample_sd)


def test_promoter_activity_sd_requires_n_and_unit() -> None:
    with pytest.raises(ValueError, match="n_samples and sample_unit are required"):
        _activity(
            promoter_activity_uncertainty=0.4,
            promoter_activity_uncertainty_type=UncertaintyType.sample_sd,
            n_samples=4,
        )


def test_promoter_activity_refuses_a_non_finite_signal() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="must be finite"):
            _activity(promoter_activity=value)


def test_promoter_activity_refuses_a_non_positive_n_samples() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _activity(n_samples=0)


@pytest.mark.parametrize(
    "field", ["promoter_name", "well_id", "reporter_gene", "activity_units"]
)
def test_promoter_activity_refuses_a_blank_identifier(field: str) -> None:
    """A blank identifier is the free-text absence the typed fields replace."""
    with pytest.raises(ValueError, match=f"{field} cannot be empty"):
        _activity(**{field: "   "})


def test_promoter_activity_round_trips_through_its_dump() -> None:
    phenotype = _activity(
        promoter_gene="b0002",
        promoter_activity_uncertainty=0.4,
        promoter_activity_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=4,
        sample_unit=SampleUnit.biological_replicate,
    )
    assert phenotype == PromoterActivityPhenotype(**phenotype.model_dump())
    assert phenotype.label_name == "promoter_activity"
    assert phenotype.label_statistic_name == "promoter_activity_se"
