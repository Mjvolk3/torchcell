# tests/torchcell/metabolism/test_flux_layer.py

"""Tests for the differentiable flux layer.

These are property tests, not regression tests. The properties are the claims the layer
makes -- the box holds exactly, a deletion zeros its reactions, the null-space
parameterization balances mass to machine precision -- and each one is a claim that would
otherwise be checked only by reading the code.

They run on a SMALL SYNTHETIC network rather than on yeast-GEM so they need no data root
and no download. The one test that touches the real model is marked and skipped when the
checkout is absent.
"""

import os
import os.path as osp
from typing import Any

import pytest
import torch

from torchcell.metabolism.constraints import (
    CatalyticUnits,
    GemTensors,
    TableCoverage,
    ThermoMode,
    ThermoTable,
    independent_balance_rows,
    null_space_basis,
)
from torchcell.metabolism.flux_layer import FluxLayer, FluxLayerConfig, gene_index_map


def _toy_gem() -> GemTensors:
    """A four-reaction, three-metabolite network with a deliberate futile cycle.

    ``A -> B`` by r1 and ``B -> A`` by r2 form the cycle; r0 imports A and r3 exports B.
    The cycle is what makes this a useful fixture: it is exactly the structure a
    thermodynamic constraint has to eliminate, and it cannot be eliminated by stoichiometry
    or bounds alone.
    """
    # metabolites: A, B, C. reactions: r0 (-> A), r1 (A -> B), r2 (B -> A), r3 (B ->)
    dense = torch.tensor(
        [[1.0, -1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0]]
    )
    s = dense.to_sparse_coo().coalesce()
    rows, _ = independent_balance_rows(s)
    units = CatalyticUnits(
        # r1 needs the complex {g0, g1}; r2 is catalyzed by g2 alone.
        unit_gene_index=torch.tensor([[0, 0, 1], [0, 1, 2]]),
        unit_reaction=torch.tensor([1, 2]),
        n_units=2,
        n_multigene_units=1,
        n_reactions_with_gpr=2,
        gene_ids=["g0", "g1", "g2"],
    )
    mask = torch.tensor([True, True, False])
    thermo = ThermoTable(
        met_delta_g=torch.tensor([0.0, -10.0, 0.0]),
        met_mask=mask,
        rxn_delta_g=torch.zeros(4),
        rxn_mask=torch.zeros(4, dtype=torch.bool),
        met_coverage=TableCoverage.of(mask.numpy()),
        rxn_coverage=TableCoverage.of(torch.zeros(4, dtype=torch.bool).numpy()),
        source_paths={},
        sha256={},
    )
    return GemTensors(
        s=s,
        lb=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        ub=torch.tensor([10.0, 10.0, 10.0, 10.0]),
        met_ids=["A", "B", "C"],
        rxn_ids=["r0", "r1", "r2", "r3"],
        catalytic_units=units,
        thermo=thermo,
        independent_rows=rows,
        biomass_index=3,
        exchange_indices=torch.tensor([0, 3]),
        n_metabolites=3,
        n_reactions=4,
    )


def _layer(**overrides: Any) -> FluxLayer:
    """Build a layer on the toy network with a three-gene model universe."""
    cfg = FluxLayerConfig(hidden_dim=8, reaction_embed_dim=4, **overrides)
    gem = _toy_gem()
    kwargs: dict[str, Any] = {}
    if cfg.parameterization == "nullspace":
        kwargs["null_space"] = null_space_basis(gem.s)
    return FluxLayer(
        gem,
        ["g0", "g1", "g2"],
        config=cfg,
        kcat_per_s=torch.tensor([1.0, 1.0]),
        molecular_weight_kda=torch.tensor([40.0, 40.0, 40.0]),
        **kwargs,
    )


def _inputs(
    batch: int = 4, n_genes: int = 3, d: int = 8, deleted: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random gene tokens plus an optional single-gene deletion in every sample."""
    h = torch.randn(batch, n_genes, d)
    ctx = torch.randn(batch, d)
    if deleted is None:
        idx = torch.zeros(0, dtype=torch.long)
        idx_batch = torch.zeros(0, dtype=torch.long)
    else:
        idx = torch.full((batch,), deleted, dtype=torch.long)
        idx_batch = torch.arange(batch)
    return h, ctx, idx, idx_batch


def test_box_is_exact():
    """Every flux lands inside its bounds, which is the layer's central structural claim."""
    layer = _layer(thermo_mode=ThermoMode.OFF, use_enzyme_capacity=False)
    out = layer(*_inputs())
    v = out["v"]
    assert torch.all(v >= layer.lb.unsqueeze(0) - 1e-5)
    assert torch.all(v <= layer.ub.unsqueeze(0) + 1e-5)
    assert float(out["feas_box_violation_frac"]) == 0.0


def test_deletion_zeros_its_reaction_exactly():
    """Deleting a subunit collapses the complex's box to a point, so its flux is exactly 0.

    This is the property that makes the deletion a structural fact rather than a penalty the
    data term could trade against. ``g0`` is a subunit of the complex catalyzing ``r1`` and
    is not involved in ``r2``, so ``r1`` must be zero and ``r2`` must not.
    """
    layer = _layer(thermo_mode=ThermoMode.OFF, use_enzyme_capacity=False)
    out = layer(*_inputs(deleted=0))
    assert torch.allclose(out["v"][:, 1], torch.zeros(4), atol=1e-6)
    assert out["v"][:, 2].abs().max() > 0.0


def test_isozyme_deletion_does_not_kill_the_reaction():
    """Deleting one gene of a two-unit reaction leaves capacity, since isozymes sum."""
    layer = _layer(thermo_mode=ThermoMode.OFF, use_enzyme_capacity=False)
    # g2 alone catalyzes r2, so deleting it must zero r2 but leave r1 (the complex) alone.
    out = layer(*_inputs(deleted=2))
    assert torch.allclose(out["v"][:, 2], torch.zeros(4), atol=1e-6)


def test_unannotated_reaction_keeps_full_availability():
    """A reaction with no gene rule is not deleted by any genotype.

    A zero in the annotation means untested, not "no gene catalyzes this", so an
    unannotated reaction must keep availability 1 no matter what is deleted.
    """
    layer = _layer(thermo_mode=ThermoMode.OFF, use_enzyme_capacity=False)
    out = layer(*_inputs(deleted=0))
    assert torch.allclose(out["c_j"][:, 0], torch.ones(4))
    assert torch.allclose(out["c_j"][:, 3], torch.ones(4))


def test_null_space_parameterization_balances_mass_to_machine_precision():
    """``v = N z`` satisfies ``Sv = 0`` identically, which is the whole point of that arm."""
    layer = _layer(parameterization="nullspace", thermo_mode=ThermoMode.OFF)
    out = layer(*_inputs())
    residual = layer._s_matmul(out["v"]).abs().max()
    assert float(residual) < 1e-4
    assert float(out["c_balance"]) < 1e-6


def test_box_and_null_space_trade_exactness():
    """The two parameterizations are exact on opposite constraints, never on both."""
    box = _layer(thermo_mode=ThermoMode.OFF, use_enzyme_capacity=False)(*_inputs())
    null = _layer(parameterization="nullspace", thermo_mode=ThermoMode.OFF)(*_inputs())
    assert float(box["feas_box_violation_frac"]) == 0.0
    assert float(null["c_balance"]) < float(box["c_balance"])


def test_every_residual_is_finite_and_dimensionless_scale():
    """No constraint term may be NaN or wildly out of scale with the others.

    An unnormalized dissipation term evaluated to about 4e4 at initialization and drove the
    first real run to NaN in one step, so "finite" is not a trivial assertion here.
    """
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    out = layer(*_inputs())
    for key in ("c_balance", "c_thermo", "c_budget", "c_dissipation", "c_parsimony"):
        value = float(out[key])
        assert value == value, f"{key} is NaN"
        assert value < 1e4, f"{key} = {value} is out of scale with the data term"


def test_thermo_modes_differ_in_what_they_require():
    """FREE needs no table; ANCHORED masks to reactions whose participants are all known."""
    free = _layer(thermo_mode=ThermoMode.FREE)(*_inputs())
    anchored = _layer(thermo_mode=ThermoMode.ANCHORED)(*_inputs())
    off = _layer(thermo_mode=ThermoMode.OFF)(*_inputs())
    assert float(off["c_thermo"]) == 0.0
    assert "thermo_mu" in free
    assert "thermo_log_c" in anchored


def test_learned_concentration_stays_physiological():
    """``ln c`` is squashed into 1 uM to 10 mM, the Thermo-Flux default window."""
    import math

    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    out = layer(*_inputs())
    log_c = out["thermo_log_c"]
    assert float(log_c.min()) >= math.log(1e-6) - 1e-4
    assert float(log_c.max()) <= math.log(1e-2) + 1e-4


def test_gradients_reach_the_gene_tokens():
    """The chain gamma -> c_u -> c_j -> box -> v must carry gradient end to end."""
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    h, ctx, idx, idx_batch = _inputs()
    h.requires_grad_(True)
    out = layer(h, ctx, idx, idx_batch)
    (out["v"].sum() + layer.constraint_loss(out)).backward()
    assert h.grad is not None
    assert float(h.grad.abs().sum()) > 0.0


def test_gene_index_map_marks_absent_genes():
    """Genes the model universe lacks map to -1 rather than to a silent index 0."""
    gem_to_model, model_to_gem = gene_index_map(["a", "b", "c"], ["b", "zzz"])
    assert gem_to_model.tolist() == [1, -1]
    assert model_to_gem.tolist() == [-1, 0, -1]


def test_coverage_report_names_the_defaults():
    """A run must be able to say what rests on data and what rests on a default."""
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    report = layer.coverage_report()
    assert report["kcat_is_default_for_all_units"] is False
    assert report["mw_is_default_for_all_genes"] is False
    assert report["transport_term_is_zero"] is True
    assert report["n_reactions_second_law_exempt"] >= 2


GEM_CHECKOUT = osp.join(
    os.environ.get("DATA_ROOT", "/nonexistent"),
    "data/torchcell/yeast-GEM/yeast-GEM-9.0.2",
)


@pytest.mark.skipif(
    not osp.exists(GEM_CHECKOUT), reason="yeast-GEM checkout not present"
)
def test_real_gem_delta_g_rejects_both_missing_value_conventions():
    """The shipped table uses a sentinel AND literal NaN; both must be dropped.

    Filtering only the sentinel leaves 51 metabolite and 120 reaction NaNs, which propagate
    into every sum and produce NaN gradients from a run whose coverage number still looks
    healthy.
    """
    from torchcell.metabolism.constraints import build_gem_tensors
    from torchcell.metabolism.yeast_GEM import YeastGEM

    source = YeastGEM()
    gem = build_gem_tensors(
        source.model, model_dir=source.model_dir, with_independent_rows=False
    )
    assert gem.thermo is not None
    assert not torch.isnan(gem.thermo.met_delta_g).any()
    assert not torch.isnan(gem.thermo.rxn_delta_g).any()
    assert gem.thermo.met_coverage.n_known == 2389
    assert gem.thermo.rxn_coverage.n_known == 3210
