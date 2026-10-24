# tests/torchcell/data/test_target_transform_043.py
# [[tests.torchcell.data.test_target_transform_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_target_transform_043.py
"""Second-stage target maps of ``torchcell.data.target_transform_043``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtri

from torchcell.data.experiment_ontology import (
    ExperimentOntologyVocabulary,
    OntologyNode,
)
from torchcell.data.target_transform_043 import (
    QUANTILE_GRID,
    TargetTransform043,
    fit_group_map,
    profile_keys,
)


def _vocab() -> ExperimentOntologyVocabulary:
    return ExperimentOntologyVocabulary(
        nodes=[
            OntologyNode(name="Experiment", kind="class"),
            OntologyNode(name="Phenotype", kind="class", parents=["Experiment"]),
            OntologyNode(name="ExprPhenotype", kind="class", parents=["Phenotype"]),
            OntologyNode(
                name="ArrayPhenotype", kind="class", parents=["ExprPhenotype"]
            ),
            OntologyNode(
                name="ArrayPhenotype.readout:x",
                kind="value",
                parents=["ArrayPhenotype"],
            ),
            OntologyNode(name="Environment", kind="class", parents=["Experiment"]),
            OntologyNode(name="Media", kind="class", parents=["Environment"]),
        ]
    )


def test_profile_key_is_the_phenotype_lane_only() -> None:
    v = _vocab()
    rows = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],  # array + readout, in some medium
            [1, 1, 1, 1, 1, 1, 0],  # same phenotype, other environment: same profile
            [1, 1, 1, 0, 0, 1, 1],  # the parent expression class only
        ],
        dtype=np.float32,
    )
    keys = profile_keys(v, rows)
    assert keys[0] == keys[1] == "ArrayPhenotype|ArrayPhenotype.readout:x|ExprPhenotype"
    assert keys[2] == "ExprPhenotype"
    with pytest.raises(ValueError, match="no phenotype node"):
        profile_keys(v, np.array([[1, 0, 0, 0, 0, 1, 1]], dtype=np.float32))


def test_quantile_normal_is_monotone_rank_gaussian_and_inverts() -> None:
    rng = np.random.default_rng(0)
    u = rng.exponential(2.0, 5000)  # skewed released scale
    m = fit_group_map("g", "quantile_normal", u)
    t = m.forward(u)
    assert np.all(np.diff(t[np.argsort(u)]) >= -1e-12)  # monotone
    assert abs(float(np.mean(t))) < 0.02 and abs(float(np.std(t)) - 1.0) < 0.02
    assert float(np.max(np.abs(m.forward(np.array([-1e9, 1e9]))))) == pytest.approx(
        float(-ndtri(0.5 / QUANTILE_GRID)), rel=1e-6
    )
    inside = (u >= m.grid_u[0]) & (u <= m.grid_u[-1])  # tails clip by design
    assert inside.mean() > 0.999
    np.testing.assert_allclose(m.inverse(t)[inside], u[inside], rtol=1e-6, atol=1e-9)
    assert np.isnan(m.forward(np.array([np.nan])))[0]


def test_quantile_normal_merges_tied_values() -> None:
    u = np.concatenate([np.zeros(600), np.linspace(1.0, 2.0, 400)])
    m = fit_group_map("g", "quantile_normal", u)
    assert len(m.grid_u) == len(set(m.grid_u))  # strictly increasing grid
    assert m.grid_u[0] == 0.0
    # the levels below 0.6 all hit the tied zero and merge to their mean, about 0.3
    assert m.grid_p[0] == pytest.approx(0.3, abs=0.01)


def test_transform_rows_route_through_group_and_dataset_affine() -> None:
    zs = fit_group_map("prof", "zscore", np.array([0.0, 2.0, 4.0, 6.0]))
    assert zs.mu == 3.0 and zs.sd == pytest.approx(np.std([0, 2, 4, 6]))
    tr = TargetTransform043(
        kind="zscore",
        by="profile",
        groups=[zs, fit_group_map("o", "none", np.zeros(3))],
    )
    z = np.array([[1.0, -1.0], [0.5, np.nan]])
    group = np.array([0, 1])
    mu_d, sd_d = np.array([2.0, 10.0]), np.array([2.0, 4.0])
    t = tr.z_to_t(z, group, mu_d, sd_d)
    # row 0: u = 2 + 2 z -> (u - 3) / sd_g ; row 1 (none): u = 10 + 4 z
    np.testing.assert_allclose(t[0], (np.array([4.0, 0.0]) - 3.0) / zs.sd, rtol=1e-6)
    np.testing.assert_allclose(t[1, 0], 12.0)
    assert np.isnan(t[1, 1])
    np.testing.assert_allclose(tr.t_to_z(t, group, mu_d, sd_d), z, rtol=1e-6)
    assert len(tr.sha256()) == 64
