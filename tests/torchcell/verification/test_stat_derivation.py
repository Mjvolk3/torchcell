"""Tests for StatDerivation (roadmap WS3 -- records how a derived stat was fixed)."""

import pytest

from torchcell.verification import DerivationMethod, Provenance, StatDerivation


def test_backsolve_kuzmin_n_round_trips() -> None:
    """The real Kuzmin combined-mutant n=4 back-solve, as a record."""
    d = StatDerivation(
        field="combined_mutant_n_samples",
        method=DerivationMethod.back_solve,
        value=4,
        range_low=4,
        range_high=8,
        statistic="reported interaction p-value median",
        diagnostics={
            "n4_median": 0.377,
            "n8_median": 0.211,
            "reported_median": 0.358,
            "spearman": 0.985,
        },
        rationale="single-term normal model matches reported p median best at n=4",
    )
    assert d == StatDerivation(**d.model_dump())
    assert d.value == 4


def test_sourced_needs_no_range() -> None:
    """A value stated in the SI (e.g. Costanzo DMF n=4) needs no range/statistic."""
    d = StatDerivation(
        field="double_mutant_n_samples",
        method=DerivationMethod.sourced,
        value=4,
        provenance=Provenance(source_uri="doi:10.1126/science.aaf1420", page="SI p.5"),
        rationale="'4 replicate colonies per double mutant' (Costanzo 2016 SOM)",
    )
    assert d.method is DerivationMethod.sourced


def test_conservative_low_requires_range() -> None:
    with pytest.raises(ValueError):
        StatDerivation(
            field="n", method=DerivationMethod.conservative_low, value=4
        )  # no range bounds


def test_range_method_value_must_lie_in_range() -> None:
    with pytest.raises(ValueError):
        StatDerivation(
            field="n",
            method=DerivationMethod.median,
            value=12,  # outside [4, 8]
            range_low=4,
            range_high=8,
        )


def test_back_solve_requires_statistic() -> None:
    with pytest.raises(ValueError):
        StatDerivation(
            field="n", method=DerivationMethod.back_solve, value=4
        )  # no statistic named


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValueError):
        StatDerivation(
            field="n",
            method=DerivationMethod.sourced,
            value=4,
            bogus="x",  # type: ignore[call-arg]  # intentional: extra=forbid
        )
