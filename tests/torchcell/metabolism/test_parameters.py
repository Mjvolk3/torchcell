# tests/torchcell/metabolism/test_parameters.py

"""Tests for the parameter resolution cascade: measured, then predicted, then default.

Two kinds of test here and the split is deliberate. The **rule** tests run on synthetic
files in a tmp directory, so the selection rule and the aggregation are checked without any
data root. The **table** tests run against the real yeast-GEM and the real predictor mirror
and are skipped when either is absent; they are what pins the coverage counts a run reports,
and a count that moves silently is exactly the failure the provenance machinery exists to
catch.
"""

import json
import os
import os.path as osp
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from dotenv import load_dotenv

from torchcell.metabolism.parameters import (
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    THERMO_EQUILIBRATOR_MIRROR,
    ParameterProvenance,
    PredictorSelection,
    ThermoSource,
    load_predicted_unit_table,
    resolve_reaction_delta_g,
)

# WITHOUT THIS the real-data tests below skip silently on a machine that has the data.
# `DATA_ROOT` and `EXPERIMENT_ROOT` live in the repo's `.env`, not in the shell pytest is
# launched from, so a bare `os.environ.get` resolves to the sentinel and every count these
# tests pin goes unchecked while the run still reports green.
load_dotenv()

DATA_ROOT = os.environ.get("DATA_ROOT", "/nonexistent")
# The plausibility study is read from THIS checkout's experiments tree when `.env` points
# elsewhere, so a worktree tests its own result file rather than the primary checkout's.
EXPERIMENT_ROOT = os.environ.get(
    "EXPERIMENT_ROOT",
    osp.join(
        osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))), "experiments"
    ),
)
GEM_CHECKOUT = osp.join(DATA_ROOT, "data/torchcell/yeast-GEM/yeast-GEM-9.0.2")
KINETICS_DIR = osp.join(DATA_ROOT, KINETICS_MIRROR)
PLAUSIBILITY = osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
THERMO_DIR = osp.join(DATA_ROOT, THERMO_EQUILIBRATOR_MIRROR)

_HAVE_REAL = all(
    osp.exists(p) for p in (GEM_CHECKOUT, KINETICS_DIR, PLAUSIBILITY, THERMO_DIR)
)
real_data = pytest.mark.skipif(
    not _HAVE_REAL,
    reason="yeast-GEM / kinetics mirror / plausibility study not present",
)


def _study(
    tmp_path: Path, rows: dict[str, tuple[float, float]], baseline: float
) -> str:
    """A minimal within-class study file: per-predictor fraction and Wilson interval."""
    payload = {
        "parameters": {
            "k_cat": {
                "own_class_band": {
                    "predictors": {
                        name: {"n": 47, "k": 30, "frac": frac, "ci95": [lower, 1.0]}
                        for name, (frac, lower) in rows.items()
                    }
                },
                "ec_median_lookup": {"own_class": {"frac": baseline}},
            }
        }
    }
    path = str(tmp_path / "within_class_plausibility.json")
    with open(path, "w") as handle:
        json.dump(payload, handle)
    return path


def test_selection_keeps_only_predictors_whose_interval_clears_the_lookup(tmp_path):
    """The point estimate is not the test; the Wilson lower bound is.

    ``good`` beats the baseline on both. ``marginal`` beats it on the point estimate and not
    on the interval, which is the whole reason the rule is written on the bound: with 47
    scorable pairs, three of the six real kcat predictors sit in exactly that position.
    """
    path = _study(
        tmp_path,
        {"good": (0.83, 0.70), "marginal": (0.66, 0.52), "bad": (0.40, 0.28)},
        baseline=0.53,
    )
    selection = PredictorSelection.from_study("k_cat", path)
    assert selection.selected == ["good"]
    assert [name for name, _, _ in selection.ranked] == ["good", "marginal", "bad"]
    assert selection.sha256 and selection.source_path == path


def test_selection_with_no_admitted_predictor_raises_rather_than_defaulting(tmp_path):
    """A study in which nothing beats the lookup is a real answer, not a reason to guess."""
    path = _study(tmp_path, {"bad": (0.40, 0.28)}, baseline=0.53)
    with pytest.raises(ValueError, match="admits no predictor"):
        load_predicted_unit_table("k_cat", 4, str(tmp_path), path)


def _predictor_parquet(
    tmp_path: Path, name: str, rows: list[tuple[int, str, float]]
) -> None:
    directory = tmp_path / name / "processed"
    directory.mkdir(parents=True)
    pd.DataFrame(rows, columns=["unit_id", "gene_id", "k_cat"]).to_parquet(
        directory / "k_cat.parquet"
    )


def test_predicted_value_is_a_log_median_over_predictors_of_a_min_over_subunits(
    tmp_path,
):
    r"""Both aggregations, checked on numbers whose right answer is arithmetic.

    Unit 0 is a two-subunit complex. Each predictor reduces it to the SLOWER subunit, then
    the three predictors are combined by the median in :math:`\log_{10}`. With per-predictor
    minima of 1, 10 and 100, the log median is 1 and the value is 10 -- where an arithmetic
    mean would give 37, dominated by the largest.
    """
    path = _study(
        tmp_path,
        {"a": (0.90, 0.80), "b": (0.88, 0.78), "c": (0.86, 0.76)},
        baseline=0.53,
    )
    _predictor_parquet(tmp_path, "a", [(0, "g0", 5.0), (0, "g1", 1.0), (1, "g2", 2.0)])
    _predictor_parquet(tmp_path, "b", [(0, "g0", 10.0), (0, "g1", 50.0)])
    _predictor_parquet(tmp_path, "c", [(0, "g0", 100.0), (0, "g1", 900.0)])
    table = load_predicted_unit_table("k_cat", 3, str(tmp_path), path)

    assert bool(table.mask[0]) and bool(table.mask[1])
    assert not bool(table.mask[2]), "a unit no predictor priced must stay unmasked"
    assert float(table.values[0]) == pytest.approx(10.0)
    assert int(table.n_predictors[0]) == 3
    # Only predictor `a` reaches unit 1, and one contributor is reported as one.
    assert float(table.values[1]) == pytest.approx(2.0)
    assert int(table.n_predictors[1]) == 1
    assert set(table.selection.selected) == {"a", "b", "c"}


def test_non_positive_predictions_are_dropped_not_logged(tmp_path):
    """A zero or negative turnover is a failed prediction; log10 of it is not a value."""
    path = _study(tmp_path, {"a": (0.90, 0.80)}, baseline=0.53)
    _predictor_parquet(tmp_path, "a", [(0, "g0", 0.0), (1, "g1", 4.0)])
    table = load_predicted_unit_table("k_cat", 2, str(tmp_path), path)
    assert not bool(table.mask[0])
    assert float(table.values[1]) == pytest.approx(4.0)


def test_unit_id_beyond_the_gem_is_an_error_not_a_silent_truncation(tmp_path):
    """A predictor table built against another model must not be quietly half-applied."""
    path = _study(tmp_path, {"a": (0.90, 0.80)}, baseline=0.53)
    _predictor_parquet(tmp_path, "a", [(7, "g0", 4.0)])
    with pytest.raises(ValueError, match="built against a different model"):
        load_predicted_unit_table("k_cat", 2, str(tmp_path), path)


# --------------------------------------------------------------------------------------
# Against the real GEM and the real mirrors.
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_gem():
    from torchcell.metabolism.constraints import build_gem_tensors
    from torchcell.metabolism.yeast_GEM import YeastGEM

    source = YeastGEM()
    gem = build_gem_tensors(
        source.model, model_dir=source.model_dir, with_independent_rows=False
    )
    return source, gem


@real_data
def test_predicted_unit_ids_align_with_the_gpr_parse(real_gem):
    """``unit_id`` in the predictor tables IS ``parse_catalytic_units``' index.

    The predicted tier joins on that integer and nothing else, so if the two ever drifted
    every predicted value would land on the wrong reaction while every count still looked
    healthy. Checked by reaction id and gene membership on all 3,552 units the tables carry.
    """
    source, gem = real_gem
    units = gem.catalytic_units
    genes_of_unit: dict[int, set[str]] = {}
    for u, g in zip(
        units.unit_gene_index[0].tolist(), units.unit_gene_index[1].tolist()
    ):
        genes_of_unit.setdefault(u, set()).add(units.gene_ids[g])

    frame = pd.read_parquet(
        osp.join(KINETICS_DIR, "unikp", "processed", "k_cat.parquet"),
        columns=["unit_id", "reaction_id", "gene_id"],
    )
    grouped = frame.groupby("unit_id").agg(
        reaction=("reaction_id", "first"), genes=("gene_id", set)
    )
    assert len(grouped) > 3000
    for raw_unit_id, row in grouped.iterrows():
        # `iterrows` types the index as `Hashable`; the column holds the integer unit id.
        unit_id = int(cast(int, raw_unit_id))
        j = int(units.unit_reaction[unit_id])
        assert gem.rxn_ids[j] == row["reaction"], unit_id
        assert set(row["genes"]) <= genes_of_unit[unit_id], unit_id


@real_data
def test_predicted_tier_fills_gaps_and_never_overwrites_a_measurement(real_gem):
    """The three-way count, and the invariant that makes it safe.

    Measured units are IDENTICAL with and without the predicted tier: a predictor trained on
    BRENDA reproduces values it memorized, so using one where a measurement exists both
    discards information and inflates any apparent agreement between the two.
    """
    from torchcell.metabolism.parameters import resolve_kcat_table

    source, gem = real_gem
    units = gem.catalytic_units
    oed = osp.join(DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/scerevisiae")
    predicted = load_predicted_unit_table(
        "k_cat", units.n_units, KINETICS_DIR, PLAUSIBILITY
    )
    measured_only = resolve_kcat_table(units, source.model_dir, oed)
    with_predicted = resolve_kcat_table(
        units, source.model_dir, oed, predicted=predicted
    )

    before, after = measured_only.tier_counts(), with_predicted.tier_counts()
    assert before["predicted"] == 0
    assert after["measured"] == before["measured"]
    assert after["predicted"] > 0
    assert after["default"] == before["default"] - after["predicted"]
    assert sum(after.values()) == units.n_units

    is_measured = measured_only.experimental_mask
    assert torch.equal(
        measured_only.values[is_measured], with_predicted.values[is_measured]
    )
    # The preflight floor in 027 reads this, and the predicted tier must not move it.
    assert (
        with_predicted.experimental_coverage.fraction
        == measured_only.experimental_coverage.fraction
    )
    assert ParameterProvenance.PREDICTED_MEDIAN in with_predicted.provenance, (
        "the predicted tier must be tagged as its own provenance, not as a measurement"
    )


@real_data
def test_union_thermo_prices_more_reactions_than_either_route_alone(real_gem):
    """The union is a gain, and each route's provenance is recorded per reaction.

    A reaction tagged ``mixed`` is priceable under neither route alone: it draws one
    participant from each table. That cell is the reason the merge is done on the METABOLITE
    tables rather than by taking the union of the two reaction sets.
    """
    _, gem = real_gem
    shipped = resolve_reaction_delta_g(gem, THERMO_DIR, ThermoSource.SHIPPED)
    equilibrator = resolve_reaction_delta_g(gem, THERMO_DIR, ThermoSource.EQUILIBRATOR)
    union = resolve_reaction_delta_g(gem, THERMO_DIR, ThermoSource.UNION)

    assert union.counts["known"] > shipped.counts["known"]
    assert union.counts["known"] > equilibrator.counts["known"]
    assert union.counts["mixed"] > 0
    assert set(shipped.provenance) <= {"shipped_metabolite_sum", "none"}
    assert set(equilibrator.provenance) <= {"equilibrator", "none"}
    for table in (shipped, equilibrator, union):
        assert not torch.isnan(table.delta_r_g0).any()
        # Masked-out reactions carry a hard zero, never a partial sum that looks like data.
        assert float(table.delta_r_g0[~table.mask].abs().max()) == 0.0
        assert len(table.provenance) == gem.n_reactions

    # The uncertainty is eQuilibrator's and is only claimed where it reaches EVERY
    # participant, so it covers strictly fewer reactions than the energies do.
    assert 0 < union.counts["sigma_known"] < union.counts["known"]
    assert bool((union.sigma[union.sigma_mask] >= 0).all())
    assert float(union.sigma[~union.sigma_mask].abs().max()) == 0.0
    assert bool((union.sigma[union.sigma_mask] > 0).any())


@real_data
def test_a_transport_reaction_carries_zero_component_contribution_uncertainty(real_gem):
    r"""Exactly zero is the right answer for a step that moves one species, not a bug.

    Measured: 745 of the union's 1,755 covered reactions propagate to :math:`\sigma=0`. The
    covariance factor is per COMPOUND, and a transport or isomerization step whose two sides
    are the same compound in different compartments has :math:`S_j^\top \Sigma^{1/2}=0` term
    by term, so the group-contribution uncertainty cancels. What remains for such a step is
    the pH and membrane-potential term, which is a different quantity computed elsewhere in
    the layer, so nothing is being hidden by the zero.
    """
    _, gem = real_gem
    union = resolve_reaction_delta_g(gem, THERMO_DIR, ThermoSource.UNION)
    zero_covered = union.sigma_mask & (union.sigma == 0)
    assert int(zero_covered.sum()) > 0
    # r_1096, "(R)-mevalonate transport", is one species crossing one membrane.
    assert bool(zero_covered[gem.rxn_ids.index("r_1096")])
    # And a multi-compound reaction does get a positive one, so the zero is not universal.
    assert float(union.sigma[union.sigma_mask].max()) > 1.0


@real_data
def test_union_agrees_with_the_shipped_route_where_only_the_shipped_route_prices(
    real_gem,
):
    """A reaction the merge assembles from shipped values alone keeps the shipped number."""
    _, gem = real_gem
    shipped = resolve_reaction_delta_g(gem, THERMO_DIR, ThermoSource.SHIPPED)
    union = resolve_reaction_delta_g(gem, THERMO_DIR, ThermoSource.UNION)
    same = np.array([p == "shipped_metabolite_sum" for p in union.provenance])
    selector = torch.from_numpy(same) & shipped.mask
    assert int(selector.sum()) > 0
    assert torch.allclose(
        union.delta_r_g0[selector], shipped.delta_r_g0[selector], atol=1e-3
    )


@real_data
def test_shipped_delta_g_is_converted_from_kcal_and_lands_near_alberty(real_gem):
    r"""The shipped CSVs are kcal/mol and say kJ/mol; reading them raw is a 4.184x error.

    MEASURED 2026-09-10 by the thermodynamics provenance study
    (``results/thermo_yeast9_provenance.json``, note
    ``[[experiments.026-metabolism-flux.yeast9-delta-g-provenance]]``): yeast-GEM 9 copied
    yETFL's ``metDeltaGFtr`` without converting, yETFL declares ``thermo_units =
    "kcal/mol"``, and regressing an independent kJ/mol recomputation on the shipped column
    over 1,927 shared metabolites gives slope 4.187 at Pearson 0.987 against the exact 4.184
    kJ per kcal.

    Pinned on two metabolites whose transformed formation energies are tabulated
    independently by Alberty: cytosolic water (``s_0803``, shipped -37.21) and cytosolic ATP
    (``s_0434``, shipped -564.41). Read as kJ/mol they are off Alberty's -157.6 and -2292 by
    a factor of four; converted they land within 2 and 70 kJ/mol, which is group-contribution
    accuracy rather than a unit error.
    """
    from torchcell.metabolism.constraints import KCAL_PER_MOL_TO_KJ_PER_MOL

    _, gem = real_gem
    assert gem.thermo is not None
    assert gem.thermo.converted_to == "kJ/mol"
    assert gem.thermo.shipped_units_in_file == "kcal/mol"
    assert gem.thermo.unit_conversion_factor == pytest.approx(4.184)

    index = {m: i for i, m in enumerate(gem.met_ids)}
    water = float(gem.thermo.met_delta_g[index["s_0803"]])
    atp = float(gem.thermo.met_delta_g[index["s_0434"]])

    # The conversion actually happened: the stored value is the file's value times 4.184.
    assert water == pytest.approx(-37.21 * KCAL_PER_MOL_TO_KJ_PER_MOL, abs=1e-2)
    assert atp == pytest.approx(-564.41 * KCAL_PER_MOL_TO_KJ_PER_MOL, abs=1e-2)

    # And the converted values are on Alberty's scale, where the raw ones are not.
    alberty_water, alberty_atp = -157.6, -2292.0
    assert abs(water - alberty_water) < 10.0, water
    assert abs(atp - alberty_atp) < 100.0, atp
    assert abs(-37.21 - alberty_water) > 100.0
    assert abs(-564.41 - alberty_atp) > 1000.0


@real_data
def test_the_union_merges_two_tables_that_are_now_in_the_same_unit(real_gem):
    """A merged energy is only meaningful once both sides are kJ/mol.

    Before the conversion the union added kcal-scale shipped values to kJ-scale eQuilibrator
    values in the same sum, which is what the ``mixed`` provenance flag marks. The check is
    that the two routes now agree in SCALE on the reactions both can price; they are
    different estimators, so they are not expected to agree in value.
    """
    _, gem = real_gem
    shipped = resolve_reaction_delta_g(gem, THERMO_DIR, ThermoSource.SHIPPED)
    equilibrator = resolve_reaction_delta_g(gem, THERMO_DIR, ThermoSource.EQUILIBRATOR)
    both = shipped.mask & equilibrator.mask
    assert int(both.sum()) > 1000
    a = shipped.delta_r_g0[both].abs()
    b = equilibrator.delta_r_g0[both].abs()
    ratio = float(a[a > 1.0].median() / b[a > 1.0].median())
    # Read as kcal this ratio sat near 1/4; in a common unit it is order 1.
    assert 0.25 < ratio < 4.0, ratio
