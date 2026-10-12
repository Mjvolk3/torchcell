# tests/torchcell/metabolism/test_kcat_sigma.py

r"""The :math:`\sigma_u` ladder is ADDITIVE: it must move nothing that already shipped.

The whole reason ``log10_sigma`` is a separate array rather than a replacement for
``values`` is that every 027 arm's recorded provenance, every tier count and the preflight's
``min_kcat_experimental_fraction`` are all functions of the four fields that shipped. So the
property under test is an equality, not a behavior: resolving a table with the width and
resolving it without must agree element-wise on ``values``, ``provenance``, ``known_mask``
and ``experimental_mask``.

The rule tests run on synthetic parquets in a tmp directory. The real-data test runs against
the yeast-GEM and the predictor mirror and skips when either is absent, in the same shape as
``test_parameters.py``.
"""

import os
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from dotenv import load_dotenv

from torchcell.metabolism.parameters import (
    GECKO3_YEAST_PROTEIN_POOL,
    KCAT_SIGMA_LADDER,
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    VERIFIED_PAIRS_FEATURE,
    ParameterProvenance,
    ParameterTable,
    SubstrateMatch,
    load_predicted_unit_table,
    resolve_kcat_table,
)

load_dotenv()

DATA_ROOT = os.environ.get("DATA_ROOT", "/nonexistent")
EXPERIMENT_ROOT = os.environ.get(
    "EXPERIMENT_ROOT",
    osp.join(
        osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))), "experiments"
    ),
)
GEM_CHECKOUT = osp.join(DATA_ROOT, "data/torchcell/yeast-GEM/yeast-GEM-9.0.2")
KINETICS_DIR = osp.join(DATA_ROOT, KINETICS_MIRROR)
PLAUSIBILITY = osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)

_HAVE_REAL = all(osp.exists(p) for p in (GEM_CHECKOUT, KINETICS_DIR, PLAUSIBILITY))
real_data = pytest.mark.skipif(
    not _HAVE_REAL,
    reason="yeast-GEM / kinetics mirror / plausibility study not present",
)

PROVENANCE_ORDER = [
    ParameterProvenance.OPEN_ENZYME_DATABASE,
    ParameterProvenance.OPEN_ENZYME_DATABASE_EC_HOMOLOG,
    ParameterProvenance.PREDICTED_MEDIAN,
    ParameterProvenance.PREDICTED_MEDIAN,
    ParameterProvenance.PREDICTED_MEDIAN,
    ParameterProvenance.ORGANISM_DEFAULT,
]


def test_adding_a_width_leaves_every_shipped_field_bit_identical():
    """``ParameterTable.build`` with and without ``log10_sigma`` agree on all four fields."""
    values = torch.tensor([1.0, 2.5, 0.3, 11.0, 7.0, 9.6])
    without = ParameterTable.build(values.clone(), list(PROVENANCE_ORDER), unit="1/s")
    with_sigma = ParameterTable.build(
        values.clone(),
        list(PROVENANCE_ORDER),
        unit="1/s",
        log10_sigma=KCAT_SIGMA_LADDER.sigma(PROVENANCE_ORDER, None),
    )
    assert torch.equal(without.values, with_sigma.values)
    assert without.provenance == with_sigma.provenance
    assert torch.equal(without.known_mask, with_sigma.known_mask)
    assert torch.equal(without.experimental_mask, with_sigma.experimental_mask)
    assert without.log10_sigma is None
    assert with_sigma.log10_sigma is not None


def test_the_ladder_prices_each_tier_at_its_own_measured_width():
    """A measured value is not treated as exact and a homolog is the widest of the three.

    The ordering is the finding: the tiers are a provenance ordering, so a ladder that
    decreased monotonically down the cascade would be encoding the provenance rather than
    the uncertainty.
    """
    sigma = KCAT_SIGMA_LADDER.sigma(PROVENANCE_ORDER, None).numpy()
    assert sigma[0] == pytest.approx(0.50, abs=1e-6)
    assert sigma[1] == pytest.approx(0.73, abs=1e-6)
    assert sigma[2] == pytest.approx(0.47, abs=1e-6)
    assert sigma[5] == pytest.approx(1.23, abs=1e-6)
    assert sigma[1] > sigma[0] > sigma[2]
    assert sigma[5] == max(sigma)


def test_the_predicted_rung_is_a_floor_and_never_a_ceiling():
    """A unit whose predictors disagree past the floor keeps its own wider width."""
    per_unit = np.array([np.nan, np.nan, 0.9, 0.1, np.nan, np.nan])
    sigma = KCAT_SIGMA_LADDER.sigma(PROVENANCE_ORDER, per_unit).numpy()
    assert sigma[2] == pytest.approx(0.9, abs=1e-6)
    # 0.1 is narrower than the ensemble's own residual against a measurement, and agreement
    # between predictors is not accuracy, so the floor wins.
    assert sigma[3] == pytest.approx(0.47, abs=1e-6)
    # A nan is "no spread observed", which is not "no spread".
    assert sigma[4] == pytest.approx(0.47, abs=1e-6)


def test_a_measured_tier_width_is_not_read_from_the_per_unit_spread():
    """Only the predicted rung is per-unit; a wide predictor spread cannot widen a measurement."""
    per_unit = np.full(len(PROVENANCE_ORDER), 3.0)
    sigma = KCAT_SIGMA_LADDER.sigma(PROVENANCE_ORDER, per_unit).numpy()
    assert sigma[0] == pytest.approx(0.50, abs=1e-6)
    assert sigma[1] == pytest.approx(0.73, abs=1e-6)
    assert sigma[2] == pytest.approx(3.0, abs=1e-6)


def test_the_protein_pool_record_factorizes_to_the_value_it_carries():
    """GECKO's pool is sigma x f x P_tot, and the record's factors must multiply to it."""
    pool = GECKO3_YEAST_PROTEIN_POOL
    assert pool.saturation_sigma is not None
    assert pool.metabolic_fraction_f is not None
    assert pool.total_protein_g_per_gdw is not None
    product = (
        pool.saturation_sigma * pool.metabolic_fraction_f * pool.total_protein_g_per_gdw
    )
    assert product == pytest.approx(pool.p_avail_g_per_gdw, abs=1e-9)
    assert pool.artifact_sha256
    assert pool.quote


@pytest.fixture(scope="module")
def real_kcat_inputs():
    """Everything ``resolve_kcat_table`` needs on the real mirror, built once."""
    from torchcell.metabolism.constraints import build_gem_tensors
    from torchcell.metabolism.yeast_GEM import YeastGEM

    source = YeastGEM()
    gem = build_gem_tensors(
        source.model, model_dir=source.model_dir, with_independent_rows=False
    )
    units = gem.catalytic_units
    substrates = SubstrateMatch.from_verified_pairs(
        osp.join(KINETICS_DIR, VERIFIED_PAIRS_FEATURE)
    )
    predicted = load_predicted_unit_table(
        "k_cat", units.n_units, KINETICS_DIR, PLAUSIBILITY
    )
    oed = osp.join(DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/scerevisiae")
    homolog = osp.join(DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/all")
    return units, source.model_dir, predicted, substrates, oed, homolog


def _write_predictor(path: str, rows: list[tuple[int, str, float]]) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    pd.DataFrame(rows, columns=["unit_id", "gene_id", "k_cat"]).to_parquet(path)


def test_the_across_predictor_sd_is_the_width_of_the_stack_the_median_came_from(
    tmp_path,
):
    """``log10_sd`` is the sd of exactly the values ``values`` is the median of."""
    import json

    study = {
        "parameters": {
            "k_cat": {
                "own_class_band": {
                    "predictors": {
                        name: {"n": 47, "k": 40, "frac": 0.85, "ci95": [0.7, 1.0]}
                        for name in ("alpha", "beta", "gamma")
                    }
                },
                "ec_median_lookup": {"own_class": {"frac": 0.5}},
            }
        }
    }
    study_path = str(tmp_path / "study.json")
    with open(study_path, "w") as handle:
        json.dump(study, handle)

    kinetics = str(tmp_path / "kinetics")
    for name, value in (("alpha", 1.0), ("beta", 10.0), ("gamma", 100.0)):
        _write_predictor(
            osp.join(kinetics, name, "verified", "k_cat.parquet"),
            [(0, "g1", value), (1, "g1", 5.0)],
        )
    os.makedirs(osp.join(kinetics, "_features"), exist_ok=True)
    pd.DataFrame(
        {
            "unit_id": [0, 1],
            "reaction_id": ["r1", "r2"],
            "substrate_met_id": ["s_1", "s_2"],
            "substrate_name": ["alanine", "serine"],
            "smiles": ["C[C@@H](N)C(=O)O", "N[C@@H](CO)C(=O)O"],
            "reaction_ec": ["1.1.1.1", "1.1.1.2"],
        }
    ).to_parquet(osp.join(kinetics, "_features", "verified_pairs.parquet"))

    table = load_predicted_unit_table("k_cat", 2, kinetics, study_path)
    assert table.log10_sd is not None
    sd = table.log10_sd.numpy()
    # log10 of 1, 10, 100 is 0, 1, 2: median 1, sample sd exactly 1.
    assert float(table.values[0]) == pytest.approx(10.0, rel=1e-6)
    assert sd[0] == pytest.approx(1.0, abs=1e-6)
    # Three predictors agreeing exactly is a zero width, which is a real reading and is
    # why the predicted rung carries a floor.
    assert sd[1] == pytest.approx(0.0, abs=1e-6)


@real_data
def test_the_real_predicted_table_carries_a_width_without_moving_a_value():
    """On the real mirror: ``values``, ``mask`` and ``n_predictors`` are what they were."""
    from torchcell.metabolism.constraints import build_gem_tensors
    from torchcell.metabolism.yeast_GEM import YeastGEM

    source = YeastGEM()
    gem = build_gem_tensors(
        source.model, model_dir=source.model_dir, with_independent_rows=False
    )
    n_units = gem.catalytic_units.n_units
    table = load_predicted_unit_table("k_cat", n_units, KINETICS_DIR, PLAUSIBILITY)
    assert table.log10_sd is not None
    sd = table.log10_sd.numpy()
    covered = table.n_predictors.numpy()
    # A width exists exactly where two or more admitted predictors covered the unit, and
    # nowhere else. Anything else means the sd was computed off a different stack.
    assert np.array_equal(np.isfinite(sd), covered >= 2)
    assert float(table.values[table.mask].min()) > 0.0
    assert np.nanmedian(sd) == pytest.approx(0.387, abs=0.02)


# --------------------------------------------------------------------------- #
# complex_aggregation and kcat_ceiling_sigma_k, both additive and default-off
# --------------------------------------------------------------------------- #


def _three_predictor_mirror(tmp_path: Path) -> tuple[str, str]:
    """A mirror whose unit 0 is a two-subunit complex with subunits at 1 and 100 1/s."""
    import json

    study = {
        "parameters": {
            "k_cat": {
                "own_class_band": {
                    "predictors": {
                        name: {"n": 47, "k": 40, "frac": 0.85, "ci95": [0.7, 1.0]}
                        for name in ("alpha", "beta", "gamma")
                    }
                },
                "ec_median_lookup": {"own_class": {"frac": 0.5}},
            }
        }
    }
    study_path = str(tmp_path / "study.json")
    with open(study_path, "w") as handle:
        json.dump(study, handle)
    kinetics = str(tmp_path / "kinetics")
    for name in ("alpha", "beta", "gamma"):
        _write_predictor(
            osp.join(kinetics, name, "verified", "k_cat.parquet"),
            [(0, "g1", 1.0), (0, "g2", 10.0), (0, "g3", 100.0), (1, "g4", 5.0)],
        )
    os.makedirs(osp.join(kinetics, "_features"), exist_ok=True)
    pd.DataFrame(
        {
            "unit_id": [0, 1],
            "reaction_id": ["r1", "r2"],
            "substrate_met_id": ["s_1", "s_2"],
            "substrate_name": ["alanine", "serine"],
            "smiles": ["C[C@@H](N)C(=O)O", "N[C@@H](CO)C(=O)O"],
            "reaction_ec": ["1.1.1.1", "1.1.1.2"],
        }
    ).to_parquet(osp.join(kinetics, "_features", "verified_pairs.parquet"))
    return kinetics, study_path


@pytest.mark.parametrize(
    "aggregation,expected", [("min", 1.0), ("max", 100.0), ("median", 10.0)]
)
def test_complex_aggregation_picks_the_named_subunit(tmp_path, aggregation, expected):
    """min, max and median over a complex's subunits, on the same predictions."""
    kinetics, study = _three_predictor_mirror(tmp_path)
    table = load_predicted_unit_table(
        "k_cat", 2, kinetics, study, complex_aggregation=aggregation
    )
    assert float(table.values[0]) == pytest.approx(expected, rel=1e-6)
    # A single-gene unit is untouched by the convention, which is what makes this a change
    # to complexes rather than a rescaling of the table.
    assert float(table.values[1]) == pytest.approx(5.0, rel=1e-6)
    assert table.complex_aggregation == aggregation


def test_the_default_aggregation_is_the_shipped_one(tmp_path):
    """Omitting the keyword must give exactly what the pre-option code gave."""
    kinetics, study = _three_predictor_mirror(tmp_path)
    default = load_predicted_unit_table("k_cat", 2, kinetics, study)
    explicit = load_predicted_unit_table(
        "k_cat", 2, kinetics, study, complex_aggregation="min"
    )
    assert torch.equal(default.values, explicit.values)
    assert torch.equal(default.mask, explicit.mask)
    assert torch.equal(default.n_predictors, explicit.n_predictors)
    assert default.complex_aggregation == "min"


@real_data
def test_the_ceiling_widening_is_bit_identical_at_k_zero(real_kcat_inputs):
    """K = 0.0 must leave values, provenance and both masks exactly as they were."""
    units, model_dir, predicted, substrates, oed, homolog = real_kcat_inputs
    base = resolve_kcat_table(
        units,
        model_dir,
        oed,
        predicted=predicted,
        substrates=substrates,
        homolog_mirror_dir=homolog,
    )
    explicit = resolve_kcat_table(
        units,
        model_dir,
        oed,
        predicted=predicted,
        substrates=substrates,
        homolog_mirror_dir=homolog,
        kcat_ceiling_sigma_k=0.0,
        complex_aggregation="min",
    )
    assert torch.equal(base.values, explicit.values)
    assert base.provenance == explicit.provenance
    assert torch.equal(base.known_mask, explicit.known_mask)
    assert torch.equal(base.experimental_mask, explicit.experimental_mask)
    assert base.ceiling is None and explicit.ceiling is None
    # The width travels with the table whether or not the ceiling moved.
    assert base.log10_sigma is not None


@real_data
@pytest.mark.parametrize("k", [1.0, 2.0])
def test_the_ceiling_widening_moves_every_unit_by_its_own_sigma(real_kcat_inputs, k):
    """The shift is 10^(k sigma_u) per unit, and the record says so."""
    units, model_dir, predicted, substrates, oed, homolog = real_kcat_inputs
    base = resolve_kcat_table(
        units,
        model_dir,
        oed,
        predicted=predicted,
        substrates=substrates,
        homolog_mirror_dir=homolog,
    )
    widened = resolve_kcat_table(
        units,
        model_dir,
        oed,
        predicted=predicted,
        substrates=substrates,
        homolog_mirror_dir=homolog,
        kcat_ceiling_sigma_k=k,
    )
    assert widened.ceiling is not None
    assert widened.ceiling.sigma_k == k
    assert base.log10_sigma is not None
    ratio = torch.log10(widened.values / base.values)
    assert torch.allclose(ratio, k * base.log10_sigma, atol=2e-4)
    # Strictly looser, never tighter: a widened ceiling that lowered a value would be a
    # different table, not a wider bound.
    assert bool((widened.values >= base.values).all())
    assert widened.provenance == base.provenance
    assert torch.equal(widened.known_mask, base.known_mask)


@real_data
def test_the_aggregation_moves_complexes_and_leaves_the_center_alone(real_kcat_inputs):
    """Max over subunits changes complexes without rescaling the table's median."""
    units, model_dir, predicted, substrates, oed, homolog = real_kcat_inputs
    shipped = resolve_kcat_table(
        units,
        model_dir,
        oed,
        predicted=predicted,
        substrates=substrates,
        homolog_mirror_dir=homolog,
    )
    predicted_max = load_predicted_unit_table(
        "k_cat", units.n_units, KINETICS_DIR, PLAUSIBILITY, complex_aggregation="max"
    )
    maximum = resolve_kcat_table(
        units,
        model_dir,
        oed,
        predicted=predicted_max,
        substrates=substrates,
        homolog_mirror_dir=homolog,
        complex_aggregation="max",
    )
    assert maximum.provenance == shipped.provenance
    single_gene = torch.zeros(units.n_units, dtype=torch.bool)
    counts = np.bincount(units.unit_gene_index[0].numpy(), minlength=units.n_units)
    single_gene[torch.from_numpy(counts == 1)] = True
    assert torch.equal(shipped.values[single_gene], maximum.values[single_gene])
    assert bool((maximum.values >= shipped.values - 1e-6).all())
    ratio = float(maximum.values.median() / shipped.values.median())
    assert 0.9 <= ratio <= 1.2, ratio
