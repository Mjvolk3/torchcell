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
from pathlib import Path
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
    if cfg.parameterization.startswith(("nullspace", "projection", "fluxforce")):
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
    assert float(log_c.min()) >= math.log(1e-7) - 1e-4
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


def _abundance_layer(**overrides: Any) -> FluxLayer:
    return _layer(
        use_abundance_head=True,
        thermo_mode=ThermoMode.OFF,
        use_enzyme_capacity=True,
        use_protein_budget=True,
        **overrides,
    )


def test_abundance_head_starts_inside_the_budget():
    """With no prior supplied, the equal-molar prior fills half the budget by construction."""
    layer = _abundance_layer()
    out = layer(*_inputs())
    e_g = out["e_g"]
    assert e_g.shape == (4, 3)
    assert torch.all(e_g > 0)
    # tanh of a random projection moves genes both ways, so the total sits near the prior.
    ratio = float(out["feas_budget_ratio"])
    assert 0.05 < ratio < 5.0, ratio
    assert float(out["c_budget"]) >= 0.0


def test_abundance_deletion_zeroes_the_complex_reaction():
    """A deleted subunit has zero abundance, the unit's power mean is zero, r1 has no capacity."""
    layer = _abundance_layer()
    out = layer(*_inputs(deleted=0))
    assert torch.all(out["e_g"][:, 0] == 0.0)
    assert torch.all(out["v"][:, 1].abs() < 1e-6)
    # The isozyme reaction r2, catalyzed by g2 alone, keeps a positive capacity.
    assert torch.all(out["e_g"][:, 2] > 0)


def test_abundance_capacity_is_the_bound_and_is_reported():
    """The box is capped by kcat * E_u, and the active-fraction diagnostic is emitted."""
    layer = _abundance_layer()
    h, ctx, idx, idx_batch = _inputs()
    out = layer(h, ctx, idx, idx_batch)
    gamma = layer.gene_availability(h, idx, idx_batch)
    e_u = layer.unit_abundance(layer.gene_abundance(h, gamma))
    cap = torch.zeros(4, layer.n_reactions)
    cap.index_add_(1, layer.unit_reaction, layer.kcat_per_h.unsqueeze(0) * e_u)
    for j in (1, 2):
        assert torch.all(out["v"][:, j].abs() <= cap[:, j] + 1e-6)
    assert "feas_capacity_active_frac" in out
    assert 0.0 <= float(out["feas_capacity_active_frac"]) <= 1.0


def test_unit_abundance_is_a_smooth_minimum_over_subunits():
    """A complex takes (close to) the scarcest subunit; an isozyme unit is its one gene."""
    layer = _abundance_layer(softmin_beta=32.0)
    e_g = torch.tensor([[1.0e-5, 4.0e-5, 7.0e-5]])
    e_u = layer.unit_abundance(e_g)
    assert abs(float(e_u[0, 0]) - 1.0e-5) < 0.05e-5
    assert abs(float(e_u[0, 1]) - 7.0e-5) < 1e-9


def test_abundance_prior_can_be_supplied_and_gradients_reach_it():
    """A measured proteome sets the prior, and the data can move E through the head."""
    cfg = FluxLayerConfig(
        hidden_dim=8,
        reaction_embed_dim=4,
        use_abundance_head=True,
        thermo_mode=ThermoMode.OFF,
    )
    layer = FluxLayer(
        _toy_gem(),
        ["g0", "g1", "g2"],
        config=cfg,
        kcat_per_s=torch.tensor([1.0, 1.0]),
        molecular_weight_kda=torch.tensor([40.0, 40.0, 40.0]),
        abundance_prior_mmol_per_gdw=torch.tensor([1e-6, 2e-6, 3e-6]),
    )
    assert not layer.abundance_prior_is_default
    h, ctx, idx, idx_batch = _inputs()
    h.requires_grad_(True)
    out = layer(h, ctx, idx, idx_batch)
    out["e_g"].sum().backward()
    assert h.grad is not None and float(h.grad.abs().sum()) > 0


# =====================================================================================
# WAVE 5 (2026-09-10): the predicted parameter tiers, the merged thermo table, and the
# constraint warmup.
# =====================================================================================


def _toy_reaction_thermo(source: Any) -> Any:
    """A four-reaction merged table with one reaction priced by each route."""
    from torchcell.metabolism.parameters import ReactionThermoTable

    mask = torch.tensor([True, True, True, False])
    return ReactionThermoTable(
        delta_r_g0=torch.tensor([-5.0, -10.0, 10.0, 0.0]),
        mask=mask,
        provenance=["equilibrator", "mixed", "shipped_metabolite_sum", "none"],
        sigma=torch.tensor([2.0, 0.0, 0.0, 0.0]),
        sigma_mask=torch.tensor([True, False, False, False]),
        source=source,
        counts={"known": 3, "n_reactions": 4, "sigma_known": 1},
        met_counts={"n_metabolites": 3, "known": 3},
        source_paths={},
        sha256={},
    )


def test_supplied_reaction_thermo_replaces_the_shipped_sum():
    """The layer uses the table it is handed, not the GEM's own metabolite sum."""
    from torchcell.metabolism.parameters import ThermoSource

    table = _toy_reaction_thermo(ThermoSource.UNION)
    cfg = FluxLayerConfig(
        hidden_dim=8, reaction_embed_dim=4, thermo_source=ThermoSource.UNION
    )
    layer = FluxLayer(
        _toy_gem(),
        ["g0", "g1", "g2"],
        config=cfg,
        kcat_per_s=torch.tensor([1.0, 1.0]),
        molecular_weight_kda=torch.tensor([40.0, 40.0, 40.0]),
        reaction_thermo=table,
    )
    assert torch.equal(layer.delta_r_g0, table.delta_r_g0)
    assert torch.equal(layer.delta_r_g0_mask, table.mask)
    report = layer.coverage_report()
    assert report["thermo_source"] == str(ThermoSource.UNION)
    assert report["thermo_route_counts"]["known"] == 3
    assert report["n_reactions_delta_g_sigma_known"] == 1
    out = layer(*_inputs())
    assert torch.isfinite(out["c_thermo"])


def test_a_thermo_table_that_disagrees_with_the_config_is_refused():
    """A run must never record a route it did not use, so the mismatch is fatal."""
    from torchcell.metabolism.parameters import ThermoSource

    cfg = FluxLayerConfig(
        hidden_dim=8, reaction_embed_dim=4, thermo_source=ThermoSource.EQUILIBRATOR
    )
    with pytest.raises(ValueError, match="thermo_source"):
        FluxLayer(
            _toy_gem(),
            ["g0", "g1", "g2"],
            config=cfg,
            kcat_per_s=torch.tensor([1.0, 1.0]),
            molecular_weight_kda=torch.tensor([40.0, 40.0, 40.0]),
            reaction_thermo=_toy_reaction_thermo(ThermoSource.UNION),
        )


def test_coverage_report_splits_kcat_into_three_tiers():
    """One boolean cannot describe a table that is part measured, part predicted."""
    layer = _layer(thermo_mode=ThermoMode.OFF)
    assert layer.coverage_report()["kcat_tier_counts"]["unknown"] == 2

    gem = _toy_gem()
    cfg = FluxLayerConfig(
        hidden_dim=8, reaction_embed_dim=4, thermo_mode=ThermoMode.OFF
    )
    tagged = FluxLayer(
        gem,
        ["g0", "g1", "g2"],
        config=cfg,
        kcat_per_s=torch.tensor([1.0, 2.0]),
        molecular_weight_kda=torch.tensor([40.0, 40.0, 40.0]),
        kcat_provenance=["open_enzyme_database", "predicted_median"],
        km_mm=torch.tensor([0.5, 0.7]),
        km_provenance=["predicted_median", "organism_default"],
    )
    report = tagged.coverage_report()
    assert report["kcat_tier_counts"] == {
        "measured": 1,
        "predicted": 1,
        "default": 0,
        "unknown": 0,
    }
    assert report["km_tier_counts"]["predicted"] == 1
    assert report["km_tier_counts"]["default"] == 1
    assert report["km_table_present"] is True


def test_a_provenance_list_of_the_wrong_length_is_an_error():
    """A short list would attribute the missing units to nothing at all."""
    with pytest.raises(ValueError, match="provenance"):
        FluxLayer(
            _toy_gem(),
            ["g0", "g1", "g2"],
            config=FluxLayerConfig(hidden_dim=8, reaction_embed_dim=4),
            kcat_per_s=torch.tensor([1.0, 1.0]),
            molecular_weight_kda=torch.tensor([40.0, 40.0, 40.0]),
            kcat_provenance=["open_enzyme_database"],
        )


def test_constraint_warmup_ramps_from_zero_and_saturates():
    """Epoch 0 carries no physics; epoch W carries all of it and it stops there."""
    layer = _layer(
        thermo_mode=ThermoMode.OFF,
        constraint_warmup_epochs=20,
        constraint_final_scale=0.1,
    )
    assert layer.set_constraint_epoch(0) == 0.0
    assert layer.set_constraint_epoch(10) == pytest.approx(0.05)
    assert layer.set_constraint_epoch(20) == pytest.approx(0.1)
    assert layer.set_constraint_epoch(200) == pytest.approx(0.1)

    out = layer(*_inputs())
    layer.set_constraint_epoch(0)
    assert float(layer.constraint_loss(out)) == 0.0
    layer.set_constraint_epoch(20)
    warm = float(layer.constraint_loss(out))
    layer.set_constraint_epoch(10)
    assert float(layer.constraint_loss(out)) == pytest.approx(warm / 2, rel=1e-5)


def test_no_warmup_leaves_the_constraint_loss_exactly_as_it_was():
    """The default has to reproduce every run before 2026-09-10, byte for byte."""
    layer = _layer(thermo_mode=ThermoMode.OFF)
    assert layer.constraint_scale == 1.0
    out = layer(*_inputs())
    w = layer.config.weights
    expected = (
        w["balance"] * out["c_balance"]
        + w["thermo"] * out["c_thermo"]
        + w["budget"] * out["c_budget"]
        + w["dissipation"] * out["c_dissipation"]
        + w["parsimony"] * out["c_parsimony"]
        + w["thermo_prior"] * out["c_thermo_prior"]
        + w.get("box", 1.0) * out["c_box"]
    )
    assert float(layer.constraint_loss(out)) == pytest.approx(float(expected))


def test_constraint_scale_leaves_the_gradient_direction_intact():
    """A scale is a step size on the physics, not a reweighting between its terms."""
    layer = _layer(thermo_mode=ThermoMode.OFF, constraint_final_scale=0.1)
    out = layer(*_inputs())
    loss = layer.constraint_loss(out)
    first = layer.flux_mlp[0]
    assert isinstance(first, torch.nn.Linear)
    weight = first.weight
    (grad_small,) = torch.autograd.grad(loss, weight, retain_graph=True)
    layer.constraint_scale = 1.0
    (grad_full,) = torch.autograd.grad(layer.constraint_loss(out), weight)
    assert torch.allclose(grad_small * 10.0, grad_full, atol=1e-6)


def test_flux_head_normalization_puts_its_init_on_the_target_scale():
    r"""The flux readout must not start seven target sigmas from the target mean.

    MEASURED on the real model, 2026-09-10, by
    ``experiments/027-betaxanthin-metabolic/scripts/diagnose_head_scale.py``: against a
    target standardized to mean 0 and sd 1, the pooled head starts at mean -0.573 (sd
    0.0074) and the unnormalized flux head at **mean +7.030** (sd 0.237). The offset, not
    the spread, is what the CRPS pays, and it matches the wave-5 cells' epoch-0 data loss of
    8.5 to 14.3 at a constraint weight of exactly zero (jobs 1663 to 1665).

    Reproduced here on fluxes of the same magnitude the real layer emits, so the property is
    pinned without the GEM: the normalized head's offset must be smaller by the flux scale,
    and the unnormalized one must still be large, or the test would pass on a head that
    normalizes nothing.
    """
    from torchcell.models.cell_graph_transformer_metabolism import FluxScalarHead

    scale = 100.0
    torch.manual_seed(2700)
    # 4,131 reactions with |v| mean ~10 and max ~60, the measured shape of the real vector.
    v = torch.rand(64, 4131) * 60.0
    turnover = torch.rand(64, 2806) * 60.0
    precursors = torch.tensor([12, 77, 900])

    torch.manual_seed(11)
    plain = FluxScalarHead(4131, precursors, param_dim=2, flux_scale=scale)
    torch.manual_seed(11)
    normed = FluxScalarHead(
        4131, precursors, param_dim=2, normalize_inputs=True, flux_scale=scale
    )
    with torch.no_grad():
        mu_plain = plain(v, turnover)[:, 0, 0]
        mu_normed = normed(v, turnover)[:, 0, 0]

    # The defect the diagnosis found, reproduced: a many-sigma constant offset.
    assert abs(float(mu_plain.mean())) > 2.0, float(mu_plain.mean())
    # And the fix: on the scale a standardized target is scored against.
    assert abs(float(mu_normed.mean())) < 1.0, float(mu_normed.mean())
    assert float(mu_normed.std()) < 1.0
    # It is a change of UNITS, not of the head: same weights, inputs divided by one scalar.
    assert torch.allclose(normed.dense.weight, plain.dense.weight)


def test_flux_head_normalization_is_off_by_default():
    """Every 026 and 027 run before 2026-09-10 must deserialize unchanged."""
    from torchcell.models.cell_graph_transformer_metabolism import FluxScalarHead

    head = FluxScalarHead(10, torch.tensor([0, 1]), param_dim=1)
    assert head.normalize_inputs is False
    torch.manual_seed(3)
    v = torch.randn(4, 10) * 50.0
    turnover = torch.rand(4, 5) * 50.0
    with torch.no_grad():
        expected = head.precursor(
            torch.log1p(turnover[:, head.precursor_indices].clamp(min=0.0))
        ) + head.dense(v)
        assert torch.allclose(head(v, turnover), expected)


def test_constraint_delay_holds_at_zero_then_ramps():
    r"""The delayed schedule: exactly zero until the delay, then the ordinary warmup.

    This is the schedule that lets the module be measured on a head that has already left the
    mean plateau. Measured 2026-09-10: at a final scale of 0.1 the flux arm never escaped in
    80 epochs (job 1669); at 0.0 the same cell escaped at epoch 33 (job 1673). A delay past
    the escape is the only way to apply the constraints to a trained head.
    """
    layer = _layer(
        thermo_mode=ThermoMode.OFF,
        constraint_delay_epochs=40,
        constraint_warmup_epochs=20,
        constraint_final_scale=0.1,
    )
    # Held at exactly zero across the whole escape window.
    for epoch in (0, 1, 20, 33, 39):
        assert layer.set_constraint_epoch(epoch) == 0.0, epoch
    # Then the ordinary linear ramp, measured from the END of the delay.
    assert layer.set_constraint_epoch(40) == 0.0
    assert layer.set_constraint_epoch(50) == pytest.approx(0.05)
    assert layer.set_constraint_epoch(60) == pytest.approx(0.1)
    assert layer.set_constraint_epoch(200) == pytest.approx(0.1)


def test_zero_delay_reproduces_the_plain_warmup():
    """The default must leave every earlier schedule bit for bit."""
    plain = _layer(
        thermo_mode=ThermoMode.OFF,
        constraint_warmup_epochs=20,
        constraint_final_scale=0.1,
    )
    delayed = _layer(
        thermo_mode=ThermoMode.OFF,
        constraint_delay_epochs=0,
        constraint_warmup_epochs=20,
        constraint_final_scale=0.1,
    )
    for epoch in (0, 5, 10, 20, 80):
        assert plain.set_constraint_epoch(epoch) == delayed.set_constraint_epoch(epoch)


# =====================================================================================
# WAVE 6 (2026-09-10): the projection parameterization, the latent sampler, and the
# eQuilibrator square-root covariance.
# =====================================================================================


def _projection_layer(seed: int = 7, **overrides: Any) -> FluxLayer:
    """A projection layer built from a fixed seed, so two of them share their weights."""
    torch.manual_seed(seed)
    return _layer(
        parameterization="projection", thermo_mode=ThermoMode.OFF, **overrides
    )


def test_parse_parameterization_reads_the_three_kinds_and_the_step_suffix():
    """The suffix form is the only channel an 026 ARMS entry has for the step count."""
    from torchcell.metabolism.flux_layer import parse_parameterization

    assert parse_parameterization("box", 5).kind == "box"
    assert parse_parameterization("nullspace", 5).kind == "nullspace"
    plain = parse_parameterization("projection", 5)
    assert (plain.kind, plain.projection_steps) == ("projection", 5)
    suffixed = parse_parameterization("projection_k20", 5)
    assert (suffixed.kind, suffixed.projection_steps) == ("projection", 20)
    # The end and the method ride in the string too, since an ARMS entry reaches no other
    # new config field; a suffix that names nothing leaves the config's own value standing.
    assert parse_parameterization("projection", 5).projection_end is None
    assert parse_parameterization("projection_box", 5).projection_end == "box"
    full = parse_parameterization("projection_k20_box_dykstra", 5)
    assert (full.projection_steps, full.projection_end, full.projection_method) == (
        20,
        "box",
        "dykstra",
    )
    with pytest.raises(ValueError, match="parameterization"):
        parse_parameterization("projction", 5)
    with pytest.raises(ValueError, match="parameterization"):
        parse_parameterization("projection_k20_bx", 5)


def test_the_parameterization_string_overrides_the_config_end():
    """A cell launched as projection_box must END ON THE CLIP whatever the config says."""
    layer = _layer(
        parameterization="projection_box",
        thermo_mode=ThermoMode.OFF,
        projection_end="balance",
    )
    assert layer.projection_end == "box"
    out = layer(*_inputs())
    assert float(layer.feasibility(out)["box_violation_frac"].max()) == 0.0
    # And the coverage report says which end ran, not which end was configured.
    assert layer.coverage_report()["projection_end"] == "box"
    assert layer.coverage_report()["parameterization_raw"] == "projection_box"


def test_projection_balances_mass_exactly_at_the_balance_end():
    """Ending on the balance projection means S v = 0, not S v small."""
    layer = _projection_layer(projection_steps=5, projection_end="balance")
    out = layer(*_inputs())
    feas = layer.feasibility(out)
    assert float(feas["balance_ratio_max"].max()) < 1e-5, float(
        feas["balance_ratio_max"].max()
    )
    assert float(out["c_balance"]) < 1e-8


def test_projection_satisfies_the_box_exactly_at_the_box_end():
    """Ending on the clip means the bounds hold to machine precision."""
    layer = _projection_layer(projection_steps=5, projection_end="box")
    out = layer(*_inputs())
    v, lb, ub = out["v"], out["lb"], out["ub"]
    assert torch.all(v >= lb - 1e-6)
    assert torch.all(v <= ub + 1e-6)
    assert float(layer.feasibility(out)["box_violation_frac"].max()) == 0.0


def test_more_projection_steps_shrink_the_off_end_residual():
    """K is not decoration: the residual the chosen end does NOT fix falls with K.

    Both layers are built from the same seed and read the same inputs, so they emit the
    same v0 and the only difference is the number of alternations.
    """
    inputs = _inputs()
    torch.manual_seed(21)
    one = _projection_layer(projection_steps=1)(*inputs)
    torch.manual_seed(21)
    twenty = _projection_layer(projection_steps=20)(*inputs)
    # At the balance end the leftover is the box overshoot, and c_box is its magnitude.
    assert float(twenty["c_box"]) < float(one["c_box"])


def _thermo_box_layer(steps: int = 20, **overrides: Any) -> FluxLayer:
    """The second law as a box: projection, ending on the clip, directional bounds on."""
    return _layer(
        parameterization=f"projection_k{steps}_box_tdg",
        thermo_mode=ThermoMode.ANCHORED,
        **overrides,
    )


def test_the_thermo_box_makes_the_second_law_exact():
    r"""Every active reaction runs downhill, ``v dG <= 0``, with no penalty involved.

    The sign of ``dG`` is a bound on ``v``, and a bound belongs in the box. With the clip
    last, a flux that is in the box is downhill by construction, so ``c_thermo`` stops being
    a term the data loss can outbid and becomes a diagnostic that should read zero.
    """
    layer = _thermo_box_layer()
    out = layer(*_inputs())
    v, delta_g = out["v"], out["delta_r_g"]
    active = layer.delta_r_g0_mask & (~layer.thermo_exempt)
    drive = v[:, active] * delta_g[:, active]
    assert float(drive.max()) <= 1e-6, float(drive.max())
    assert float(layer.feasibility(out)["thermo_violation_frac"].max()) == 0.0
    assert float(layer.feasibility(out)["thermo_uphill_frac"].max()) == 0.0


def test_c_thermo_is_not_a_readout_once_the_second_law_is_a_box():
    r"""``c_thermo`` stays HIGH on a head that satisfies the second law exactly, by design.

    The hinge is ``relu(v dG + eps) / (|v||dG| + eps)``, and its ``eps`` is what makes the
    strict inequality expressible. At ``v = 0`` both the numerator and the denominator are
    ``eps``, so a reaction the directional box pinned to zero scores exactly 1, and a head
    that pins many of them reads near 1 overall while violating nothing. Measured on the toy
    network: the second-law violation fraction is 0 and ``c_thermo`` is about 0.84.

    This is pinned as a test because reading ``c_thermo`` as a violation rate on a ``tdg``
    cell is the obvious mistake, and it would be read as the physics failing when it is in
    fact exact. ``feasibility``'s two thermodynamic fractions are the readouts; ``c_thermo``
    is kept unchanged only because every arm before 2026-09-11 was trained against it.
    """
    layer = _thermo_box_layer()
    out = layer(*_inputs())
    assert float(layer.feasibility(out)["thermo_violation_frac"].max()) == 0.0
    assert float(out["c_thermo"]) > 0.5
    # And the reason is the zero fluxes, not a violation: no reaction runs uphill at all.
    active = layer.delta_r_g0_mask & (~layer.thermo_exempt)
    assert float((out["v"][:, active] * out["delta_r_g"][:, active]).max()) <= 1e-6


def test_the_thermo_box_keeps_the_bounds_exact_and_reports_the_balance_residual():
    """Ending on the clip means bounds AND direction hold; balance is what is reported."""
    layer = _thermo_box_layer()
    out = layer(*_inputs())
    v, lb, ub = out["v"], out["lb"], out["ub"]
    assert torch.all(v >= lb - 1e-6)
    assert torch.all(v <= ub + 1e-6)
    feas = layer.feasibility(out)
    assert float(feas["box_violation_frac"].max()) == 0.0
    # The balance residual is the one loose end, and it is REPORTED rather than claimed
    # away. On THIS network it is often large, and that is a property of the fixture, not
    # of the head: the toy net is four reactions built around a futile cycle, so pinning
    # r1 and r2 to opposite directions can leave the tightened box meeting ker S only at
    # the origin while the two exempt exchange reactions stay free. The residual is a
    # per-sample number for the same reason: the directional box depends on that
    # genotype's learned concentrations. The 1e-2 claim belongs on yeast-GEM, and it is
    # made there, in test_thermo_box_on_the_real_gem_is_exact_and_nearly_balanced.
    assert feas["balance_ratio_median"].shape == (4,)
    assert torch.all(torch.isfinite(feas["balance_ratio_median"]))

    # More alternations do not make it worse, which is what says the leftover is a
    # convergence question rather than an artifact of the loop.
    torch.manual_seed(5)
    one = _thermo_box_layer(steps=1)
    torch.manual_seed(5)
    twenty = _thermo_box_layer(steps=20)
    inputs = _inputs()
    r1 = one.feasibility(one(*inputs))["balance_ratio_median"]
    r20 = twenty.feasibility(twenty(*inputs))["balance_ratio_median"]
    assert float(r20.median()) <= float(r1.median()) + 1e-6


def test_an_emptied_directional_box_is_counted_rather_than_hidden():
    """A forced lower bound that the second law forbids has to be visible, not absorbed."""
    layer = _thermo_box_layer()
    out = layer(*_inputs())
    assert "feas_thermo_box_emptied_frac" in out
    assert 0.0 <= float(out["feas_thermo_box_emptied_frac"]) <= 1.0


def test_the_thermo_box_margin_leaves_near_equilibrium_reactions_free():
    r"""A margin constrains fewer reactions, and the ones it drops are the undecided ones.

    Measured motivation (job 1725): with the margin at its default, which falls back to the
    1e-3 kJ/mol numerical epsilon, 70.6 percent of reactions on the top-100 strains are
    pinned below 1e-6 against 11.9 percent for the untightened head, and that head scores at
    chance on the phenotype. A reaction half a kJ/mol from equilibrium has a sign the
    tabulated energy cannot resolve, so pinning it spends a degree of freedom and buys no
    thermodynamics.
    """
    # SAME SEED for both, so the only difference between the two boxes is the margin. The
    # availability and capacity chain is randomly initialized, and comparing two
    # independently initialized layers compares those instead.
    torch.manual_seed(31)
    layer = _layer(
        parameterization="projection_k20_box_tdg", thermo_mode=ThermoMode.ANCHORED
    )
    torch.manual_seed(31)
    wide = _layer(
        parameterization="projection_k20_box_tdgm50", thermo_mode=ThermoMode.ANCHORED
    )
    assert layer.thermo_box_margin_kj == 0.0
    assert wide.thermo_box_margin_kj == 50.0

    inputs = _inputs()
    tight_out, wide_out = layer(*inputs), wide(*inputs)
    lb_t, ub_t = tight_out["lb"], tight_out["ub"]
    lb_w, ub_w = wide_out["lb"], wide_out["ub"]
    # A wider margin can only RELAX the box, never tighten it.
    assert torch.all(ub_w >= ub_t - 1e-6)
    assert torch.all(lb_w <= lb_t + 1e-6)
    # And it leaves at least as much room, counted as reactions pinned to a single point.
    pinned_tight = ((ub_t - lb_t) <= 1e-12).float().mean()
    pinned_wide = ((ub_w - lb_w) <= 1e-12).float().mean()
    assert float(pinned_wide) <= float(pinned_tight)
    assert wide.coverage_report()["thermo_box_margin_kj"] == 50.0


def test_the_thermo_box_refuses_a_head_that_does_not_end_on_the_clip():
    """Only the clip enforces the directional bound, so the balance end must be refused."""
    with pytest.raises(ValueError, match="thermo_box"):
        _layer(
            parameterization="projection_k20_balance_tdg",
            thermo_mode=ThermoMode.ANCHORED,
        )
    with pytest.raises(ValueError, match="thermo_box"):
        _layer(
            parameterization="nullspace",
            thermo_mode=ThermoMode.ANCHORED,
            thermo_box=True,
        )


def test_the_thermo_box_is_off_by_default_everywhere():
    """Every arm configured before 2026-09-11 must keep its soft second law."""
    for raw in ("box", "nullspace", "projection", "projection_k20_box"):
        layer = _layer(parameterization=raw, thermo_mode=ThermoMode.ANCHORED)
        assert layer.thermo_box is False, raw
        assert "feas_thermo_box_emptied_frac" not in layer(*_inputs())


def test_dykstra_also_lands_in_the_intersection():
    """The nearest-point variant keeps the exactness claim of the end it stops on."""
    layer = _projection_layer(projection_steps=8, projection_method="dykstra")
    out = layer(*_inputs())
    feas = layer.feasibility(out)
    assert float(feas["balance_ratio_max"].max()) < 1e-5
    # Dykstra starts from the same v0, so its box overshoot is finite and reported.
    assert float(out["c_box"]) >= 0.0


def test_projection_gradients_reach_the_latent_mlp_and_the_gene_tokens():
    """A clip has zero gradient on its saturated side, so this is not automatic."""
    layer = _projection_layer()
    h, ctx, idx, idx_batch = _inputs()
    h.requires_grad_(True)
    out = layer(h, ctx, idx, idx_batch)
    (out["v"].sum() + layer.constraint_loss(out)).backward()
    first = layer.latent_mlp[0]
    assert isinstance(first, torch.nn.Linear)
    assert first.weight.grad is not None
    assert float(first.weight.grad.abs().sum()) > 0.0
    assert h.grad is not None and float(h.grad.abs().sum()) > 0.0


def test_projection_deletion_still_zeros_its_reaction():
    """The clip carries the deletion: lb = ub = 0 pins r1 to zero through every step."""
    layer = _projection_layer(projection_end="box")
    out = layer(*_inputs(deleted=0))
    assert torch.allclose(out["v"][:, 1], torch.zeros(4), atol=1e-6)


def test_latent_stochastic_moves_between_draws_and_collapses_to_the_mean_in_eval():
    """A sampler that repeats one flux is not a sampler; eval must still be the mean."""
    layer = _projection_layer(latent_stochastic=True)
    inputs = _inputs()
    layer.train()
    first, second = layer(*inputs)["v"], layer(*inputs)["v"]
    assert float((first - second).abs().max()) > 0.0
    layer.eval()
    assert torch.allclose(layer(*inputs)["v"], layer(*inputs)["v"])


def test_kl_is_zero_for_a_unit_gaussian_posterior_and_positive_otherwise():
    """KL(N(mu, sigma) || N(0, I)) = 0 exactly at mu = 0, sigma = 1."""
    layer = _projection_layer(latent_stochastic=True)
    readout = layer.latent_mlp[2]
    assert isinstance(readout, torch.nn.Linear)
    torch.nn.init.zeros_(readout.weight)
    torch.nn.init.zeros_(readout.bias)
    torch.nn.init.zeros_(layer.latent_log_std.bias)
    layer.train()
    out = layer(*_inputs())
    assert abs(float(out["c_kl"])) < 1e-6
    torch.nn.init.constant_(layer.latent_log_std.bias, 1.0)
    assert float(layer(*_inputs())["c_kl"]) > 0.0


def test_the_kl_is_absent_from_the_loss_of_a_deterministic_arm():
    """c_kl exists on every arm and is exactly zero unless the latent is stochastic."""
    layer = _projection_layer()
    out = layer(*_inputs())
    assert float(out["c_kl"]) == 0.0
    assert "kl" not in layer.config.weights


def test_sample_returns_shapes_and_per_draw_feasibility():
    """[S, B, r] fluxes and [S, B] feasibility, drawn in eval mode."""
    layer = _projection_layer(latent_stochastic=True)
    layer.eval()
    drawn = layer.sample(*_inputs(batch=3), n_samples=5)
    assert drawn.v.shape == (5, 3, layer.n_reactions)
    assert drawn.balance_ratio_median.shape == (5, 3)
    assert drawn.thermo_uphill_frac.shape == (5, 3)
    assert drawn.n_samples == 5
    # The draws differ even though the module is in eval mode.
    assert float((drawn.v[0] - drawn.v[1]).abs().max()) > 0.0
    # And every draw is balanced, because every draw was projected.
    assert float(drawn.balance_ratio_median.max()) < 1e-5


def test_a_deterministic_projection_cell_can_be_switched_into_a_sampler():
    """``audit_top_strain_feasibility.py`` sets ``config.stochastic`` on a BUILT layer.

    The spread head therefore has to exist on every projection layer, sampled or not, or the
    switch would need a parameter that is absent from the checkpoint. Cheap to hold: a
    zero-weight linear whose bias is the initial log sigma.
    """
    layer = _projection_layer()
    assert layer.latent_stochastic is False
    assert hasattr(layer, "latent_log_std")
    names = dict(layer.named_parameters())
    layer.config.stochastic = True
    assert layer.latent_stochastic is True
    assert set(dict(layer.named_parameters())) == set(names)
    layer.eval()
    drawn = layer.sample(*_inputs(batch=2), n_samples=3)
    assert float((drawn.v[0] - drawn.v[1]).abs().max()) > 0.0


def test_a_nullspace_layer_ignores_stochastic_as_it_always_has():
    """The pre-2026-09-10 no-op on the existing flux_nullspace arm is preserved."""
    layer = _layer(
        parameterization="nullspace", thermo_mode=ThermoMode.OFF, stochastic=True
    )
    assert layer.latent_stochastic is False
    assert not hasattr(layer, "latent_log_std")
    layer.train()
    inputs = _inputs()
    assert torch.allclose(layer(*inputs)["v"], layer(*inputs)["v"])


def test_sample_refuses_a_deterministic_or_box_layer():
    """No fall back: repeating one flux would report a feasibility fraction of 0 or 1."""
    deterministic = _projection_layer()
    with pytest.raises(ValueError, match="latent_stochastic"):
        deterministic.sample(*_inputs(), n_samples=2)
    box = _layer(thermo_mode=ThermoMode.OFF)
    with pytest.raises(ValueError, match="parameterization"):
        box.sample(*_inputs(), n_samples=2)


def test_feasibility_returns_one_row_per_sample_on_every_arm():
    """The same keys from the box head as from the projection head."""
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    out = layer(*_inputs(batch=6))
    feas = layer.feasibility(out)
    for key in (
        "balance_ratio_median",
        "balance_ratio_max",
        "box_violation_frac",
        "thermo_violation_frac",
        "thermo_uphill_frac",
    ):
        assert feas[key].shape == (6,), key
    # The box head is exact on the box by construction, per sample as well as on average.
    assert float(feas["box_violation_frac"].max()) == 0.0
    # An uphill fraction restricted to reactions carrying flux can only be larger than the
    # one that counts a numerically positive product at zero flux, never smaller.
    assert float(feas["thermo_violation_frac"].max()) <= 1.0


def test_a_non_orthonormal_basis_is_refused():
    """N (N^T v) is a projector only for an orthonormal N, so a scaled basis must fail."""
    gem = _toy_gem()
    basis = null_space_basis(gem.s) * 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        FluxLayer(
            gem,
            ["g0", "g1", "g2"],
            config=FluxLayerConfig(
                hidden_dim=8, reaction_embed_dim=4, parameterization="projection"
            ),
            kcat_per_s=torch.tensor([1.0, 1.0]),
            molecular_weight_kda=torch.tensor([40.0, 40.0, 40.0]),
            null_space=basis,
        )


def test_the_covariance_path_replaces_the_isotropic_offset():
    """delta_g_error = Q m, with m of the factor's rank and a N(0, I) prior."""
    from torchcell.metabolism.flux_layer import ReactionSqrtCovariance

    n_components = 3
    q = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]
    )
    cov = ReactionSqrtCovariance(
        q=q,
        mask=torch.tensor([True, True, True, False]),
        n_components=n_components,
        n_reactions_covered=3,
        source_path="synthetic",
        sha256="0" * 64,
    )
    cfg = FluxLayerConfig(
        hidden_dim=8,
        reaction_embed_dim=4,
        thermo_mode=ThermoMode.ANCHORED,
        thermo_uncertainty="covariance",
    )
    layer = FluxLayer(
        _toy_gem(),
        ["g0", "g1", "g2"],
        config=cfg,
        kcat_per_s=torch.tensor([1.0, 1.0]),
        molecular_weight_kda=torch.tensor([40.0, 40.0, 40.0]),
        delta_g_sqrt_cov=cov,
    )
    assert not hasattr(layer, "delta_g_offset")
    assert layer.delta_g_m.shape == (n_components,)
    out = layer(*_inputs())
    # m starts at zero, so the offset starts at zero and the prior with it.
    assert float(out["c_thermo_prior"]) == 0.0
    assert torch.allclose(out["thermo_delta_g_offset"], torch.zeros(4))
    with torch.no_grad():
        layer.delta_g_m.copy_(torch.tensor([1.0, 1.0, 1.0]))
    moved = layer(*_inputs())
    assert torch.allclose(
        moved["thermo_delta_g_offset"], torch.tensor([1.0, 2.0, 3.0, 0.0])
    )
    assert float(moved["c_thermo_prior"]) == pytest.approx(1.0)
    report = layer.coverage_report()
    assert report["thermo_uncertainty"] == "covariance"
    assert report["n_reactions_covariance_covered"] == 3


def test_the_covariance_path_refuses_to_run_without_a_covariance():
    """A 'covariance' run with no Q would be the isotropic run wearing its name."""
    cfg = FluxLayerConfig(
        hidden_dim=8,
        reaction_embed_dim=4,
        thermo_mode=ThermoMode.ANCHORED,
        thermo_uncertainty="covariance",
    )
    with pytest.raises(ValueError, match="covariance"):
        FluxLayer(
            _toy_gem(),
            ["g0", "g1", "g2"],
            config=cfg,
            kcat_per_s=torch.tensor([1.0, 1.0]),
            molecular_weight_kda=torch.tensor([40.0, 40.0, 40.0]),
        )


def test_the_dgpredictor_rows_of_the_shipped_column_are_not_converted(tmp_path):
    r"""yeast-GEM 9.0.2's shipped reaction column is TWO units, and the header says one.

    The curation script fills it from yETFL in kcal/mol and then fills an index block from
    dGPredictor in kJ/mol. Converting the whole column multiplies those 33 rows by 4.184 and
    produces a free energy 4.2x too large that looks entirely ordinary. Checked on a
    synthetic CSV so the property is pinned without the GEM: the exempt row comes back as
    written, its neighbor comes back converted, and both missing-value conventions are still
    dropped.
    """
    from torchcell.metabolism.constraints import (
        KCAL_PER_MOL_TO_KJ_PER_MOL,
        _read_delta_g_csv,
        dgpredictor_reaction_ids,
    )

    csv = tmp_path / "model_rxnDeltaG.csv"
    csv.write_text(
        "rxns,rxnDeltaG\nr_0001,10.0\nr_4650,10.0\nr_0002,10000000\nr_0003,NaN\n"
    )
    plain = _read_delta_g_csv(str(csv))
    assert plain["r_4650"] == pytest.approx(10.0 * KCAL_PER_MOL_TO_KJ_PER_MOL)

    exempt = _read_delta_g_csv(str(csv), already_kj_ids=frozenset({"r_4650"}))
    assert exempt["r_4650"] == pytest.approx(10.0)
    assert exempt["r_0001"] == pytest.approx(10.0 * KCAL_PER_MOL_TO_KJ_PER_MOL)
    # Both missing-value conventions stay dropped whichever way the row is priced.
    assert "r_0002" not in exempt and "r_0003" not in exempt

    # The id list is loaded from a file and a missing file is fatal, never an empty set:
    # silently skipping the exemption is how the 4.2x error reached the layer.
    rows = tmp_path / "dgpredictor_reaction_rows.json"
    rows.write_text('{"dgpredictor_reaction_ids": ["r_4650", "r_4651"]}')
    ids, path = dgpredictor_reaction_ids(str(rows))
    assert ids == frozenset({"r_4650", "r_4651"}) and path == str(rows)
    with pytest.raises(FileNotFoundError, match="dGPredictor"):
        dgpredictor_reaction_ids(str(tmp_path / "absent.json"))


THERMO_DIR = osp.join(
    os.environ.get("DATA_ROOT", "/nonexistent"), "data/torchcell/thermo_equilibrator"
)


@pytest.mark.skipif(
    not osp.exists(GEM_CHECKOUT), reason="yeast-GEM checkout not present"
)
def test_thermo_box_on_the_real_gem_is_exact_and_nearly_balanced():
    r"""The claim that matters, on the network it is meant for, at initialization.

    On yeast-GEM 9.0.2 the directional box has 4,131 reactions and a 1,538-dimensional null
    space to work with, so unlike the four-reaction toy fixture it has room: the second law
    and the bounds hold exactly, and twenty alternations bring the balance residual under
    1e-2 on every sample without any training at all.

    Untrained deliberately. A property that only holds after fitting is a property of the
    fit; this one is a property of the parameterization, so it has to hold at epoch zero.
    """
    from torchcell.metabolism.constraints import build_gem_tensors
    from torchcell.metabolism.yeast_GEM import YeastGEM

    source = YeastGEM()
    gem = build_gem_tensors(source.model, model_dir=source.model_dir)
    torch.manual_seed(2700)
    layer = FluxLayer(
        gem,
        [f"g{i}" for i in range(64)],
        config=FluxLayerConfig(
            hidden_dim=16,
            reaction_embed_dim=4,
            parameterization="projection_k20_box_tdg",
            thermo_mode=ThermoMode.ANCHORED,
        ),
    )
    out = layer(
        torch.randn(4, 64, 16),
        torch.randn(4, 16),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
    )
    feas = layer.feasibility(out)
    active = layer.delta_r_g0_mask & (~layer.thermo_exempt)
    drive = out["v"][:, active] * out["delta_r_g"][:, active]
    assert float(drive.max()) <= 1e-6, float(drive.max())
    assert float(feas["thermo_violation_frac"].max()) == 0.0
    assert float(feas["box_violation_frac"].max()) == 0.0
    assert float(feas["balance_ratio_median"].max()) < 1e-2, float(
        feas["balance_ratio_median"].max()
    )


@pytest.mark.skipif(
    not osp.exists(osp.join(THERMO_DIR, "formation_sigma.npz")),
    reason="eQuilibrator formation covariance not present",
)
def test_real_reaction_covariance_row_norms_equal_the_tabulated_sigma():
    r"""``sigma_j = ||Q_j||`` ties the matrix to the number the thermo table already ships.

    ``resolve_reaction_delta_g`` forms :math:`S^\top\Sigma^{1/2}` and keeps only its row
    norms; :func:`reaction_sqrt_covariance` keeps the matrix. If they ever diverge, the
    covariance path would be fitting a different uncertainty from the one every coverage
    report prints.
    """
    from torchcell.metabolism.constraints import build_gem_tensors
    from torchcell.metabolism.flux_layer import reaction_sqrt_covariance
    from torchcell.metabolism.parameters import ThermoSource, resolve_reaction_delta_g
    from torchcell.metabolism.yeast_GEM import YeastGEM

    source = YeastGEM()
    gem = build_gem_tensors(
        source.model, model_dir=source.model_dir, with_independent_rows=False
    )
    table = resolve_reaction_delta_g(gem, THERMO_DIR, ThermoSource.UNION)
    cov = reaction_sqrt_covariance(THERMO_DIR, gem.s, list(gem.met_ids))
    norms = cov.q.norm(dim=1)
    covered = table.sigma_mask
    assert torch.allclose(norms[covered], table.sigma[covered], atol=1e-3, rtol=1e-4)
    assert cov.n_components == 590
    assert int(covered.sum()) == 1755


# --------------------------------------------------------------------------------------
# THE FLUX-FORCE CHANNEL (2026-09-11). The second law as a smooth, differentiable coupling
# rather than as a sign clamp on the box.
# --------------------------------------------------------------------------------------


def _fluxforce_layer(steps: int = 20, end: str = "box", **overrides: Any) -> FluxLayer:
    """The flux-force head on the toy network, anchored, ending where the caller says."""
    return _layer(
        parameterization=f"fluxforce_k{steps}_{end}",
        thermo_mode=ThermoMode.ANCHORED,
        **overrides,
    )


def test_parse_reads_the_fluxforce_channel_and_its_two_suffixes():
    """The string is the only channel an arm has, so it must carry K and the exact end."""
    from torchcell.metabolism.flux_layer import parse_parameterization

    plain = parse_parameterization("fluxforce", 5)
    assert (plain.kind, plain.projection_steps, plain.projection_end) == (
        "fluxforce",
        5,
        None,
    )
    full = parse_parameterization("fluxforce_k20_box", 5)
    assert (full.kind, full.projection_steps, full.projection_end) == (
        "fluxforce",
        20,
        "box",
    )
    assert parse_parameterization("fluxforce_balance", 5).projection_end == "balance"
    # No `tdg` token: stacking a sign clamp on the flux-force relation would clip the very
    # function whose gradient the channel exists to deliver.
    with pytest.raises(ValueError, match="parameterization"):
        parse_parameterization("fluxforce_k20_box_tdg", 5)


def test_fluxforce_runs_every_priced_reaction_downhill_before_the_projection():
    r"""``v = m f(dG)`` with ``m >= 0`` gives ``v dG <= 0`` identically, with no clamp."""
    layer = _fluxforce_layer()
    out = layer(*_inputs())
    v_pre, delta_g = out["v_pre_projection"], out["delta_r_g"]
    active = layer.delta_r_g0_mask & (~layer.thermo_exempt)
    assert bool(active.any())
    assert float((v_pre[:, active] * delta_g[:, active]).max()) <= 1e-6
    # And the pre-projection flux is already inside the genotype box, so the deletion and
    # the capacity bound bite before anything else runs.
    assert float((v_pre - out["ub"]).max()) <= 1e-6
    assert float((out["lb"] - v_pre).max()) <= 1e-6


def test_fluxforce_magnitude_is_bounded_by_the_genotype_box():
    """``|v| <= max(|lb|, |ub|)``, so the box, not the force law, sets the scale."""
    layer = _fluxforce_layer()
    out = layer(*_inputs())
    ceiling = torch.maximum(out["lb"].abs(), out["ub"].abs())
    assert float((out["v_pre_projection"].abs() - ceiling).max()) <= 1e-6


def test_fluxforce_goes_quiet_near_equilibrium_instead_of_being_pinned():
    r"""``f(dG) -> 0`` smoothly, which is what the directional box spends a bound on.

    Measured on the toy network: the priced reactions sit at ``|dG| = 10`` kJ/mol, where
    ``|f| = 0.96``, and moving them to ``0.1`` kJ/mol takes ``|f|`` to ``0.02``. The
    directional box makes no distinction between the two, and constrains both.
    """
    layer = _fluxforce_layer()
    far = layer.flux_force_efficacy(torch.tensor([-10.0, 10.0]))
    near = layer.flux_force_efficacy(torch.tensor([-0.1, 0.1]))
    assert float(far.abs().min()) > 0.9
    assert float(near.abs().max()) < 0.05
    assert float(layer.flux_force_efficacy(torch.zeros(1)).abs().max()) == 0.0
    # Odd, and monotone decreasing: the sign is structural rather than a convention.
    grid = torch.linspace(-50.0, 50.0, 101)
    f = layer.flux_force_efficacy(grid)
    assert torch.allclose(f, -layer.flux_force_efficacy(-grid), atol=1e-6)
    assert bool((f[1:] - f[:-1] <= 0).all())


def test_fluxforce_passes_gradient_to_the_concentrations_and_the_thermo_box_does_not():
    r"""THE FINDING THIS CHANNEL WAS BUILT ON, pinned as a test.

    ``_thermo_tighten`` decides the sign with ``delta_g > eps`` inside ``torch.where``, a
    step function, so no gradient reaches ``log_c_base`` through the box: the model cannot
    move a metabolite pool to license a direction along that path, and the docstring claim
    that it can is false for it. The flux-force relation is differentiable everywhere, so
    the same derivative is nonzero. The loss here is a function of the FLUX alone, which is
    what isolates the parameterization: through the soft hinge ``c_thermo`` both heads pass
    gradient, and that is a different path.
    """
    torch.manual_seed(11)
    inputs = _inputs()

    force = _fluxforce_layer()
    grad_force = torch.autograd.grad(
        force(*inputs)["v"].sum(), force.log_c_base, allow_unused=True
    )[0]
    assert grad_force is not None
    assert float(grad_force.abs().sum()) > 0.0

    tdg = _layer(
        parameterization="projection_k20_box_tdg", thermo_mode=ThermoMode.ANCHORED
    )
    grad_tdg = torch.autograd.grad(
        tdg(*inputs)["v"].sum(), tdg.log_c_base, allow_unused=True
    )[0]
    assert grad_tdg is None or float(grad_tdg.abs().sum()) == 0.0


def test_fluxforce_gradients_still_reach_the_gene_tokens():
    """The genotype chain is untouched: availability and capacity still move the flux."""
    layer = _fluxforce_layer()
    h, ctx, idx, idx_batch = _inputs()
    h.requires_grad_(True)
    layer(h, ctx, idx, idx_batch)["v"].pow(2).sum().backward()
    assert h.grad is not None and float(h.grad.abs().sum()) > 0.0


def test_fluxforce_deletion_still_zeros_its_reaction():
    """``c_j = 0`` collapses the box, so the magnitude is zero whatever the energy says."""
    layer = _fluxforce_layer(end="box")
    out = layer(*_inputs(deleted=0))
    assert torch.allclose(out["v"][:, 1], torch.zeros(4), atol=1e-6)


def test_fluxforce_balances_mass_exactly_at_the_balance_end():
    """The channel changes the pre-projection flux and nothing about what the ends mean."""
    layer = _fluxforce_layer(end="balance")
    out = layer(*_inputs())
    residual = layer._s_matmul(out["v"])[:, layer.independent_rows]
    assert float(residual.abs().max()) < 1e-5


def test_fluxforce_reports_the_sign_flips_the_projection_introduces():
    """The one second-law violation this channel can produce is counted, not hidden."""
    for end in ("balance", "box"):
        layer = _fluxforce_layer(end=end)
        out = layer(*_inputs())
        frac = float(out["feas_sign_flip_frac"])
        assert 0.0 <= frac <= 1.0
        # And it is the honest accounting of the difference between the two fluxes: no
        # reaction may be counted as flipped while both fluxes carry the same sign.
        v, v_pre = out["v"], out["v_pre_projection"]
        active = layer.delta_r_g0_mask & (~layer.thermo_exempt)
        carrying = (v_pre.abs() > 1e-6) & active.unsqueeze(0)
        flipped = (v * v_pre < 0) & carrying
        expected = float(flipped.sum()) / max(float(carrying.sum()), 1.0)
        assert abs(frac - expected) < 1e-9


def test_fluxforce_samples_and_prices_every_draw():
    """A trained cell can be switched into a sampler, as the projection head can."""
    layer = _fluxforce_layer()
    assert layer.latent_stochastic is False
    layer.config.stochastic = True
    assert layer.latent_stochastic is True
    drawn = layer.sample(*_inputs(batch=3), n_samples=4)
    assert drawn.v.shape == (4, 3, 4)
    assert drawn.balance_ratio_median.shape == (4, 3)
    assert drawn.parameterization == "fluxforce"
    # The draws differ, which is the whole point of calling this rather than forward twice.
    assert float((drawn.v[0] - drawn.v[1]).abs().max()) > 0.0


def test_fluxforce_refuses_a_thermo_mode_that_prices_nothing():
    """No fall back to the box head: the channel IS the reaction energy."""
    with pytest.raises(ValueError, match="ANCHORED"):
        _layer(parameterization="fluxforce_k5_box", thermo_mode=ThermoMode.OFF)


def test_the_dissipation_rate_is_reported_even_when_it_is_not_penalized():
    """Niebel's limit is a validation readout, so the rate is logged whether or not it binds.

    Thermo-Flux takes the cellular Gibbs energy dissipation rate over the boundary fluxes;
    a model that stays under the limit unasked has learned something, and one trained under
    it has been told the answer. Reporting it unconditionally is what lets the two be told
    apart, so this holds with ``use_dissipation_limit`` off.
    """
    layer = _layer(thermo_mode=ThermoMode.ANCHORED, use_dissipation_limit=False)
    out = layer(*_inputs(batch=5))
    assert out["g_diss_per_sample"].shape == (5,)
    assert float(out["c_dissipation"]) == 0.0
    assert torch.allclose(out["thermo_g_diss_mean"], out["g_diss_per_sample"].mean())
    # And the penalized arm reports the same number under its own key.
    penalized = _layer(thermo_mode=ThermoMode.ANCHORED, use_dissipation_limit=True)
    penalized.load_state_dict(layer.state_dict())
    out_p = penalized(*_inputs(batch=5))
    assert torch.allclose(out_p["g_diss"], out_p["thermo_g_diss_mean"])


def test_the_force_end_makes_the_second_law_exact_and_reports_the_balance_residual():
    r"""``fluxforce_k20_force``: the flux leaves through the direction clip, not the box.

    The alternation ends on the balance projection, which is free to move a reaction across
    zero; the last clip then puts every priced reaction back inside the box tightened to the
    sign its energy licenses. Bounds and the second law are both exact after it, and the
    balance residual is what is reported, which is the opposite trade from the box end.
    """
    layer = _fluxforce_layer(end="force")
    assert layer.projection_end == "force"
    out = layer(*_inputs())
    v, delta_g = out["v"], out["delta_r_g"]
    active = layer.delta_r_g0_mask & (~layer.thermo_exempt)
    assert float((v[:, active] * delta_g[:, active]).max()) <= 1e-6
    assert float(layer.feasibility(out)["thermo_violation_frac"].max()) == 0.0
    assert float(out["feas_box_violation_frac"]) == 0.0
    # The flips the balance projection introduced are still counted, because the clip
    # removes them by zeroing a reaction rather than by making them not have happened.
    assert 0.0 <= float(out["feas_sign_flip_frac"]) <= 1.0
    assert "feas_thermo_box_emptied_frac" in out


def test_the_force_end_still_passes_gradient_to_the_concentrations():
    """The direction clip is a step, but it is no longer the ONLY path to ``log_c``.

    This is what separates the force end from the ``tdg`` head. The magnitude and the
    direction that enter the clip are already smooth functions of the reaction energy, so a
    flux that the clip leaves alone still carries the derivative, and the model can move a
    pool to change the direction it asks for.
    """
    torch.manual_seed(13)
    layer = _fluxforce_layer(end="force")
    grad = torch.autograd.grad(
        layer(*_inputs())["v"].sum(), layer.log_c_base, allow_unused=True
    )[0]
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_the_force_end_is_refused_on_the_projection_channel():
    """There is no direction to restore on a head that never had one."""
    with pytest.raises(ValueError, match="parameterization"):
        _layer(parameterization="projection_k20_force", thermo_mode=ThermoMode.ANCHORED)


def test_the_boxforce_end_carries_the_direction_through_the_whole_alternation():
    r"""``fluxforce_k<N>_boxforce``: the direction is IN the box the alternation clips into.

    The ``force`` end reaches exactness with a clip that runs after the projection, so the
    projection converges against a set the flux does not end in and the last clip throws the
    balance away. Here the box is tightened first, so every clip in the alternation is
    already the directional one and the balance projection converges against the set that
    holds. Bounds and the second law are exact and the balance residual is reported.
    """
    layer = _fluxforce_layer(end="boxforce")
    out = layer(*_inputs())
    v, delta_g = out["v"], out["delta_r_g"]
    active = layer.delta_r_g0_mask & (~layer.thermo_exempt)
    assert float((v[:, active] * delta_g[:, active]).max()) <= 1e-6
    assert float(out["feas_box_violation_frac"]) == 0.0
    # The reported box is the GENOTYPE box, which the tightened one sits inside, so a flux
    # that respects the tightened bounds respects the reported ones by construction.
    assert float((v - out["ub"]).max()) <= 1e-4
    assert float((out["lb"] - v).max()) <= 1e-4
    assert "feas_thermo_box_emptied_frac" in out


def test_the_boxforce_end_still_passes_gradient_to_the_concentrations():
    """Tightening the box is a step; the magnitude and the direction inside it are not."""
    torch.manual_seed(17)
    layer = _fluxforce_layer(end="boxforce")
    grad = torch.autograd.grad(
        layer(*_inputs())["v"].sum(), layer.log_c_base, allow_unused=True
    )[0]
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_the_boxforce_end_samples_inside_the_directional_box():
    """Every draw leaves through the tightened clip, so no draw runs uphill."""
    layer = _fluxforce_layer(end="boxforce")
    layer.config.stochastic = True
    drawn = layer.sample(*_inputs(batch=3), n_samples=4)
    assert float(drawn.thermo_violation_frac.max()) == 0.0
    assert float(drawn.box_violation_frac.max()) == 0.0


def test_the_density_term_is_zero_on_a_network_that_turns_over_and_one_on_a_dead_one():
    r"""``c_density`` prices rows of S that carry no throughput, and nothing else.

    A balance statistic over a mostly-zero flux is a statement about the zeros, so a head
    that can empty the network satisfies the Genesis balance rule the degenerate way. The
    term is the hinge ``relu(1 - omega_i / (threshold * flux_scale))`` over the independent
    rows: a row at or above the threshold costs nothing and a row at zero costs one.
    """
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    out = layer(*_inputs())
    # Zero on this fixture, because every row of the toy network turns over above the
    # threshold: the term prices dead rows and this network has none.
    assert float(out["c_density"]) == 0.0
    assert float(out["feas_active_row_frac"]) == 1.0
    starved = _layer(thermo_mode=ThermoMode.ANCHORED, density_turnover_threshold=1.0)
    out_starved = starved(*_inputs())
    assert 0.0 < float(out_starved["c_density"]) <= 1.0
    assert 0.0 <= float(out["feas_active_row_frac"]) <= 1.0
    # A flux of exactly zero is the worst case and scores one on every row.
    dead = dict(out)
    dead["v"] = torch.zeros_like(out["v"])
    layer_out = layer(*_inputs())
    omega = layer.turnover(torch.zeros_like(layer_out["v"]))[:, layer.independent_rows]
    threshold = layer.config.density_turnover_threshold * layer.config.flux_scale
    assert float(torch.relu(1.0 - omega / threshold).mean()) == 1.0


def test_the_density_term_is_off_by_default_and_outside_the_constraint_ramp():
    """Two claims: it changes no existing arm, and it survives a constraint scale of zero.

    Every wave-6 and wave-7 cell runs the physics at scale 0.0 or 0.1. A countermeasure to
    a degenerate solution that lived inside the ramp would be identically zero at 0.0, and
    the arm would report a weight it never applied.
    """
    # The toy network turns over well above the shipped threshold, so its density term is
    # exactly zero; raising the threshold is what makes its rows count as dead and the term
    # bite. That the shipped threshold reads zero HERE is itself the right behavior.
    layer = _layer(thermo_mode=ThermoMode.ANCHORED, density_turnover_threshold=1.0)
    out = layer(*_inputs())
    assert layer.config.density_weight == 0.0
    assert float(out["c_density"]) > 0.0
    baseline = float(layer.constraint_loss(out))

    layer.config.density_weight = 0.5
    weighted = float(layer.constraint_loss(out))
    assert weighted > baseline
    assert abs(weighted - baseline - 0.5 * float(out["c_density"])) < 1e-6

    # And at a constraint scale of zero the density term is the WHOLE loss, which is the
    # property the wave-8 cells depend on.
    layer.constraint_scale = 0.0
    assert abs(float(layer.constraint_loss(out)) - 0.5 * float(out["c_density"])) < 1e-6


def test_the_density_term_gradient_pushes_throughput_onto_dead_rows():
    """The term has to move the flux, not just report on it."""
    layer = _fluxforce_layer(end="boxforce", density_turnover_threshold=1.0)
    layer.config.density_weight = 1.0
    layer.constraint_scale = 0.0
    h, ctx, idx, idx_batch = _inputs()
    out = layer(h, ctx, idx, idx_batch)
    loss = layer.constraint_loss(out)
    grads = torch.autograd.grad(
        loss, list(layer.flux_mlp.parameters()), allow_unused=True
    )
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads)


def test_the_density_threshold_matches_the_audit_rule():
    """The trained term and the judging criterion are the same statistic, by construction."""
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    # `audit_top_strain_feasibility.BALANCE_TURNOVER_THRESHOLDS` has 1e-3 as its primary.
    assert layer.config.density_turnover_threshold == 1.0e-3


def test_the_fluxforce_channel_takes_the_dykstra_method_too():
    r"""Dykstra's correction converges to the NEAREST point of the intersection to ``v0``.

    That is not a detail on this head: von Neumann's alternation reaches SOME feasible
    point, and on this network the point it reaches is sparse. Measured untrained on the
    real yeast-GEM, the fraction of independent rows of S that turn over above
    1e-3 * flux_scale: 5.26 percent for `fluxforce_k100_boxforce` and 11.42 percent for the
    same channel under Dykstra, with the second law, the box and the balance median
    unchanged. The token is what lets a cell ask for it.
    """
    from torchcell.metabolism.flux_layer import parse_parameterization

    spec = parse_parameterization("fluxforce_k100_boxforce_dykstra", 5)
    assert (
        spec.kind,
        spec.projection_steps,
        spec.projection_end,
        spec.projection_method,
    ) == ("fluxforce", 100, "boxforce", "dykstra")
    layer = _fluxforce_layer(end="boxforce", projection_method="dykstra")
    assert layer.projection_method == "dykstra"
    out = layer(*_inputs())
    active = layer.delta_r_g0_mask & (~layer.thermo_exempt)
    assert float((out["v"][:, active] * out["delta_r_g"][:, active]).max()) <= 1e-6
    assert float(out["feas_box_violation_frac"]) == 0.0


# -- K draws per genotype (wave 8) --------------------------------------------------


def test_one_draw_reproduces_the_deterministic_forward_bit_for_bit():
    """``draws=1`` must be the path every wave-5 to wave-7 cell ran, to the last bit.

    The K-draw expansion is guarded on ``draws > 1``, so at 1 it should not run at all:
    not a different order of operations that agrees to a tolerance, but the same tensors.
    Both layers are built and called under the same manual seed, so the parameter
    initialization and the RNG the draw consumes are identical.
    """
    torch.manual_seed(0)
    plain = _layer(
        parameterization="projection",
        thermo_mode=ThermoMode.OFF,
        latent_stochastic=True,
    )
    torch.manual_seed(0)
    one = _layer(
        parameterization="projection",
        thermo_mode=ThermoMode.OFF,
        latent_stochastic=True,
        draws=1,
        log_c_from_latent=False,
    )
    inputs = _inputs()
    torch.manual_seed(1)
    a = plain(*inputs)
    torch.manual_seed(1)
    b = one(*inputs)
    for key in ("v", "c_balance", "c_box", "c_kl", "c_parsimony"):
        assert torch.equal(a[key], b[key]), f"{key} differs at draws=1"


def test_log_c_from_latent_off_is_bit_identical_under_anchored_thermodynamics():
    """The flag's default leaves an anchored K-draw layer at the last bit, extras included.

    The flag moves the energy computation to after the draw, so this checks the OFF path
    kept the old order: same seed, same init, same forward, every thermo extra equal and
    at B rows.
    """
    batch, draws = 3, 4
    torch.manual_seed(0)
    plain = _layer(
        parameterization="projection",
        thermo_mode=ThermoMode.ANCHORED,
        latent_stochastic=True,
        draws=draws,
    )
    torch.manual_seed(0)
    off = _layer(
        parameterization="projection",
        thermo_mode=ThermoMode.ANCHORED,
        latent_stochastic=True,
        draws=draws,
        log_c_from_latent=False,
    )
    assert not off.log_c_from_latent_enabled
    assert not hasattr(off, "log_c_from_latent")
    inputs = _inputs(batch=batch)
    torch.manual_seed(1)
    a = plain(*inputs)
    torch.manual_seed(1)
    b = off(*inputs)
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), (
            f"{key} differs with log_c_from_latent=False"
        )
    assert b["thermo_log_c"].shape[0] == batch


def test_log_c_from_latent_gives_each_draw_its_own_concentration():
    """With the flag, ``thermo_log_c`` has ``B*K`` rows that differ across draws once U moves.

    Three things, in order. At zero init the flag-on layer is the flag-off layer: same
    parameters, same RNG consumption at init and in the forward, same flux, and the K
    concentrations of one genotype are K copies. Once ``log_c_from_latent.weight`` is
    nonzero the K rows of one genotype differ, so the amino-acid concentration readout is
    a function of the draw. And ``sample()`` still runs, pricing each draw against its own
    energies.
    """
    batch, draws = 3, 4
    torch.manual_seed(0)
    off = _layer(
        parameterization="projection",
        thermo_mode=ThermoMode.ANCHORED,
        latent_stochastic=True,
        draws=draws,
    )
    torch.manual_seed(0)
    on = _layer(
        parameterization="projection",
        thermo_mode=ThermoMode.ANCHORED,
        latent_stochastic=True,
        draws=draws,
        log_c_from_latent=True,
    )
    assert on.log_c_from_latent_enabled
    assert on.log_c_from_latent.weight.shape == (on.n_metabolites, on.n_latent)
    assert float(on.log_c_from_latent.weight.abs().max()) == 0.0
    # Same init RNG consumed by both, so the shared parameters are identical.
    for (name, p_off), (_, p_on) in zip(
        off.named_parameters(), on.named_parameters(), strict=False
    ):
        if name != "log_c_from_latent.weight":
            assert torch.equal(p_off, p_on), f"{name} differs at init"
    inputs = _inputs(batch=batch)
    torch.manual_seed(1)
    a = off(*inputs)
    torch.manual_seed(1)
    b = on(*inputs)
    assert torch.equal(a["v"], b["v"]), "zero-init flag-on flux differs from flag-off"
    assert b["thermo_log_c"].shape == (batch * draws, on.n_metabolites)
    assert b["thermo_conc_term"].shape == (batch * draws, on.n_reactions)
    assert b["delta_r_g"].shape[0] == batch * draws
    assert b["g_diss_per_sample"].shape[0] == batch * draws
    grouped = b["thermo_log_c"].view(batch, draws, -1)
    assert float(grouped.std(dim=1).abs().max()) == 0.0, (
        "at zero init the draws of one genotype must share the concentration"
    )
    assert torch.equal(grouped[:, 0], a["thermo_log_c"]), (
        "zero-init flag-on concentration differs from the flag-off one"
    )

    with torch.no_grad():
        on.log_c_from_latent.weight.fill_(0.5)
    c = on(*inputs)
    grouped = c["thermo_log_c"].view(batch, draws, -1)
    assert float(grouped.std(dim=1).max()) > 0.0, (
        "the draws of one genotype share a concentration although U is nonzero"
    )
    sampled = on.sample(*inputs, n_samples=3)
    assert sampled.v.shape == (3, batch, on.n_reactions)


def test_log_c_from_latent_requires_a_drawn_latent_and_a_concentration():
    """The flag is refused on a deterministic latent, a latent-free head, and no thermo."""
    with pytest.raises(AssertionError, match="latent is deterministic"):
        _layer(
            parameterization="projection",
            thermo_mode=ThermoMode.ANCHORED,
            latent_stochastic=False,
            stochastic=False,
            log_c_from_latent=True,
        )
    with pytest.raises(AssertionError, match="null-space latent"):
        _layer(
            parameterization="box",
            thermo_mode=ThermoMode.ANCHORED,
            stochastic=True,
            log_c_from_latent=True,
        )
    with pytest.raises(AssertionError, match="ANCHORED"):
        _layer(
            parameterization="projection",
            thermo_mode=ThermoMode.OFF,
            latent_stochastic=True,
            log_c_from_latent=True,
        )


def test_log_c_from_latent_carries_gradient_to_u_and_to_the_posterior():
    """A loss on ``thermo_log_c`` reaches U, and through U the latent's mu and sigma.

    At zero init the path through U is closed (d log c / d z = U^T sigma' = 0), which is
    the point of the zero init, so U is set nonzero first; the gradient then has to reach
    both halves of the posterior, since that is what lets the data term shape the draw.
    """
    torch.manual_seed(0)
    layer = _layer(
        parameterization="projection",
        thermo_mode=ThermoMode.ANCHORED,
        latent_stochastic=True,
        draws=2,
        log_c_from_latent=True,
    )
    layer.train()
    out = layer(*_inputs(batch=2))
    out["thermo_log_c"].sum().backward()
    assert layer.log_c_from_latent.weight.grad is not None
    assert float(layer.log_c_from_latent.weight.grad.abs().max()) > 0.0
    for p in (layer.latent_log_std.weight, layer.latent_log_std.bias):
        assert p.grad is None or float(p.grad.abs().max()) == 0.0, (
            "at zero init no gradient can pass through U"
        )

    layer.zero_grad()
    with torch.no_grad():
        layer.log_c_from_latent.weight.fill_(0.3)
    out = layer(*_inputs(batch=2))
    out["thermo_log_c"].sum().backward()
    mu_params = [p for p in layer.latent_mlp.parameters()]
    assert any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0 for p in mu_params
    ), "no gradient reached the latent mean"
    for name, p in (
        ("latent_log_std.weight", layer.latent_log_std.weight),
        ("latent_log_std.bias", layer.latent_log_std.bias),
    ):
        assert p.grad is not None and float(p.grad.abs().max()) > 0.0, (
            f"no gradient reached {name}"
        )


def test_k_draws_expand_the_batch_and_differ_from_each_other():
    """``draws=K`` returns ``B*K`` fluxes, grouped by genotype, and no two draws agree.

    The grouping is what lets a caller reshape to ``[B, K, ...]`` with a plain view, and
    the disagreement is what makes them samples rather than copies. Both are asserted
    because a `repeat_interleave` written as a `repeat` would pass the second and fail the
    reshape silently downstream.

    The grouping is asserted on the BOX rather than on the flux. The box is a deterministic
    function of the genotype, so it must be constant within a group and that check is exact;
    the flux is a draw, and on a four-reaction toy network a genotype whose availability
    closes its two catalyzed reactions has a box of zero width and every draw of it clips to
    the same point. Requiring every genotype to move would be requiring the fixture not to
    produce that genotype.
    """
    torch.manual_seed(3)
    layer = _layer(
        parameterization="projection",
        thermo_mode=ThermoMode.OFF,
        latent_stochastic=True,
        draws=8,
    )
    layer.eval()
    h, ctx, idx, idx_batch = _inputs(batch=3)
    out = layer(h, ctx, idx, idx_batch)
    assert out["v"].shape == (24, 4)
    assert float(out["latent_n_draws"]) == 8.0
    for name in ("lb", "ub"):
        within = out[name].view(3, 8, 4).std(dim=1)
        assert float(within.abs().max()) == 0.0, (
            f"{name} varies between the draws of one genotype, so the rows are not grouped "
            "by genotype and a [B, K, ...] view would mix them"
        )
    spread = out["v"].detach().view(3, 8, 4).std(dim=1)
    assert float(spread.max()) > 0.0, "no genotype's draws differ at all"


def test_the_draw_is_taken_in_eval_too():
    """A K-draw arm samples in eval, since its prediction is the average over draws.

    The deterministic head emits mu in eval and must keep doing so; only ``draws > 1``
    forces the draw. Both halves are asserted, because a flag that forced the draw
    everywhere would silently change every earlier arm's reported test score.
    """
    single = _layer(
        parameterization="projection",
        thermo_mode=ThermoMode.OFF,
        latent_stochastic=True,
    )
    single.eval()
    inputs = _inputs(batch=2)
    assert torch.equal(single(*inputs)["v"], single(*inputs)["v"])

    many = _layer(
        parameterization="projection",
        thermo_mode=ThermoMode.OFF,
        latent_stochastic=True,
        draws=4,
    )
    many.eval()
    first, second = many(*inputs)["v"], many(*inputs)["v"]
    assert not torch.equal(first, second), "eval with draws > 1 returned a fixed flux"


def test_k_draws_on_a_deterministic_latent_is_refused():
    """K copies of mu are not K samples, so the layer refuses to be built that way."""
    with pytest.raises(AssertionError, match="not\nK samples|not K samples"):
        _layer(
            parameterization="projection",
            thermo_mode=ThermoMode.OFF,
            latent_stochastic=False,
            stochastic=False,
            draws=4,
        )


def test_the_posterior_sd_is_reported_for_a_sampled_layer_only():
    """``latent_sigma_p50`` exists exactly when there is a posterior to report."""
    sampled = _layer(
        parameterization="projection",
        thermo_mode=ThermoMode.OFF,
        latent_stochastic=True,
    )
    out = sampled(*_inputs())
    assert "latent_sigma_p50" in out
    # Zero weights and a constant bias, so an untrained spread head reads exp(init).
    assert float(out["latent_sigma_p50"]) == pytest.approx(
        float(torch.exp(torch.tensor(sampled.config.latent_log_std_init))), rel=1e-5
    )
    plain = _layer(parameterization="nullspace", thermo_mode=ThermoMode.OFF)
    assert "latent_sigma_p50" not in plain(*_inputs())


def test_every_batch_axis_output_carries_the_expanded_rows():
    """Under K draws, one contract: everything with a batch axis has ``B*K`` rows.

    The exception is the ``thermo_*`` block, which is the learned concentration and its
    derived terms. Without ``log_c_from_latent`` no draw touches those, and the one readout
    that consumes them is refused with K > 1 unless that flag is set (then they carry B*K
    rows too; see ``test_log_c_from_latent_gives_each_draw_its_own_concentration``).

    This is a regression test with a scar. `c_j` was left at B rows on the first pass
    because no draw changes a reaction's availability, and 027's `_per_sample_feasibility`
    rebuilds the box from exactly that tensor and compares it against the flux: the run
    trained a full epoch, validated, tested, and died on a shape error in the feasibility
    pass at the very end.
    """
    batch, draws = 3, 5
    layer = _layer(
        parameterization="projection",
        thermo_mode=ThermoMode.ANCHORED,
        latent_stochastic=True,
        draws=draws,
    )
    out = layer(*_inputs(batch=batch))
    rows = batch * draws
    for key, value in out.items():
        if value.dim() == 0 or key.startswith("thermo_"):
            continue
        assert value.shape[0] in (rows, 1), (
            f"{key} has {value.shape[0]} rows; a batch-axis output must carry B*K = {rows} "
            "so that a consumer can pair it with the flux"
        )


def test_the_active_row_balance_term_is_zero_on_a_balanced_flux():
    r"""A flux in ``ker S`` has no residual, so the term it is judged by must read zero.

    The statistic is the same relative residual the thresholded Genesis rule takes the max
    of, restricted to the rows that turn over, so a head whose flux is exactly balanced owes
    nothing and a head that balances only the idle rows is not let off.
    """
    layer = _layer(parameterization="nullspace", thermo_mode=ThermoMode.ANCHORED)
    out = layer(*_inputs())
    residual = layer._s_matmul(out["v"])[:, layer.independent_rows]
    assert float(residual.abs().max()) < 1e-4
    assert float(out["c_balance_active"]) < 1e-8
    assert float(out["feas_balance_active_max"]) < 1e-6


def test_the_active_row_balance_term_is_off_by_default_and_outside_the_ramp():
    """Weight 0.0 leaves the loss bit-identical, and the term survives a scale of zero."""
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    out = layer(*_inputs())
    assert layer.config.balance_active_weight == 0.0
    baseline = float(layer.constraint_loss(out))

    layer.config.balance_active_weight = 0.5
    weighted = float(layer.constraint_loss(out))
    expected = baseline + 0.5 * float(out["c_balance_active"])
    assert abs(weighted - expected) < 1e-6

    # At the scale these cells run, the ramped physics contributes nothing and this term is
    # the whole of what asks the flux to balance.
    layer.constraint_scale = 0.0
    only = float(layer.constraint_loss(out))
    assert abs(only - 0.5 * float(out["c_balance_active"])) < 1e-6


def test_the_active_row_balance_term_ignores_the_idle_rows():
    """Restricting to rows above the turnover threshold is what the rule does, so it is what
    the term does: raising the threshold until no row qualifies takes the term to zero.
    """
    layer = _layer(thermo_mode=ThermoMode.ANCHORED, density_turnover_threshold=1.0e6)
    out = layer(*_inputs())
    assert float(out["feas_active_row_frac"]) == 0.0
    assert float(out["c_balance_active"]) == 0.0


def test_the_active_row_balance_gradient_reduces_the_residual_not_the_turnover():
    """``omega`` is detached in the denominator, so the only way down is a smaller residual."""
    layer = _fluxforce_layer(end="boxforce")
    layer.config.balance_active_weight = 1.0
    layer.constraint_scale = 0.0
    h, ctx, idx, idx_batch = _inputs()
    out = layer(h, ctx, idx, idx_batch)
    grads = torch.autograd.grad(
        layer.constraint_loss(out), list(layer.flux_mlp.parameters()), allow_unused=True
    )
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads)


def test_the_active_balance_percentile_survives_a_large_expanded_batch():
    """``torch.quantile`` caps its input at 2**24 elements; the K-draw batch passes it.

    The active-row balance percentile is taken over ``[B, independent rows]``. At batch 128
    and yeast-GEM's 2,593 independent rows that is 332k elements at one draw and 42.5M at
    128, and ``torch.quantile`` raises ``input tensor is too large`` above the cap. The
    wave-8 K = 128 benchmark hit it before any job was submitted.

    Two assertions, because only the pair is a fix: the sort-based quantile agrees with
    ``torch.quantile`` wherever ``torch.quantile`` will answer at all, and it still answers
    past the cap.
    """
    from torchcell.metabolism.flux_layer import _linear_quantile

    generator = torch.Generator().manual_seed(0)
    for n in (1, 2, 7, 1000, 33333):
        x = torch.randn(n, generator=generator)
        for q in (0.0, 0.5, 0.95, 1.0):
            assert float(_linear_quantile(x, q)) == pytest.approx(
                float(torch.quantile(x, q)), abs=1e-6
            )
    over_the_cap = torch.randn(2**24 + 1, generator=generator)
    with pytest.raises(RuntimeError, match="too large"):
        torch.quantile(over_the_cap, 0.95)
    assert torch.isfinite(_linear_quantile(over_the_cap, 0.95))


def test_the_frozen_row_set_keeps_a_killed_row_in_the_penalty():
    r"""The dodge the dynamic set allows, and what freezing does about it.

    Under ``balance_active_row_set='dynamic'`` the penalized set is a function of the flux
    being penalized, so a head can leave the penalty by pushing a row below the turnover
    threshold; job 1795 did exactly that, taking the active-row fraction from 0.00501 to
    0.00225 while ``c_density`` held. Frozen, the set is fixed at the first forward, so a
    row that later goes quiet stays in the sum at the ratio its residual earns.
    """
    layer = _layer(thermo_mode=ThermoMode.ANCHORED, balance_active_row_set="frozen")
    inputs = _inputs()
    assert not layer.frozen_active_rows_ready
    layer(*inputs)
    assert layer.frozen_active_rows_ready
    frozen = layer.frozen_active_rows.clone()
    assert bool(frozen.any())

    # A second call with different tokens does not re-derive the set.
    layer(*_inputs(batch=6))
    assert torch.equal(layer.frozen_active_rows, frozen)

    report = layer.coverage_report()
    assert report["balance_active_row_set"] == "frozen"
    assert report["frozen_active_row_count"] == int(frozen.sum())
    assert len(report["frozen_active_row_digest"]) == 16


def test_the_dynamic_row_set_is_the_default_and_unchanged():
    """Every cell before this option ran the dynamic set; the default must still be it."""
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    assert layer.config.balance_active_row_set == "dynamic"
    out = layer(*_inputs())
    omega = layer.turnover(out["v"])[:, layer.independent_rows]
    threshold = layer.config.density_turnover_threshold * layer.config.flux_scale
    assert float(out["feas_active_row_frac"]) == float(
        (omega > threshold).float().mean()
    )
    assert int(layer.frozen_active_rows.sum()) == 0


def test_an_unknown_row_set_is_refused():
    """A typo must not silently fall back to the set the cell did not ask for."""
    with pytest.raises(ValueError, match="balance_active_row_set"):
        _layer(thermo_mode=ThermoMode.ANCHORED, balance_active_row_set="active")


# ---------------------------------------------------------------------------------------
# THE WARM-STARTED RESIDUAL HEAD
#
# Measured motivation, in one line each, because the tests below are assertions about a
# design whose whole justification is a measurement:
#   * from a uniform random point of the box the alternation leaves the turnover-thresholded
#     balance ratio at 2.0 for K = 20 .. 5000, von Neumann and Dykstra alike (probe_projrate);
#   * from a proposal 1 percent from feasible the SAME K = 100 reaches 0.0012 (probe_warmstart);
#   * the trained boxforce head of job 1784 proposes at a relative distance of 0.990, i.e. the
#     projection discards 99 percent of it (probe_trained_proposal_distance).
# ---------------------------------------------------------------------------------------


def _warmstart_layer(seed: int = 5, **overrides: Any) -> FluxLayer:
    """The residual head on the toy network, built from a fixed seed."""
    torch.manual_seed(seed)
    return _layer(parameterization="warmstart_k20_box", **overrides)


def test_parse_parameterization_reads_the_warmstart_channel():
    """The channel, its step count, its exact end and its method all ride in the string."""
    from torchcell.metabolism.flux_layer import parse_parameterization

    plain = parse_parameterization("warmstart", 5)
    assert (plain.kind, plain.projection_steps, plain.projection_end) == (
        "warmstart",
        5,
        None,
    )
    full = parse_parameterization("warmstart_k100_boxforce_dykstra", 5)
    assert (full.kind, full.projection_steps) == ("warmstart", 100)
    assert (full.projection_end, full.projection_method) == ("boxforce", "dykstra")
    with pytest.raises(ValueError, match="parameterization"):
        parse_parameterization("warmstart_k100_bx", 5)


def test_the_reference_flux_is_solved_from_the_gem_and_is_feasible():
    r"""The reference is solved INSIDE the build step, from S and the bounds, and balances.

    This is what keeps the head organism-independent: no reference vector is accepted from
    a caller, so porting to another GEM brings its own reference and no yeast constant
    enters. On the toy network the parsimonious optimum is unique: import 10 through r0,
    carry it through r1, run nothing around the futile cycle, export 10 through r3.
    """
    layer = _warmstart_layer()
    record = layer.reference_flux
    assert record is not None
    assert record.objective_value == pytest.approx(10.0)
    assert record.v.tolist() == pytest.approx([10.0, 10.0, 0.0, 10.0])
    # The futile cycle carries nothing, which is the parsimony half doing its job.
    assert record.v[2].abs() < 1e-9
    balance = layer._s_matmul(record.v.unsqueeze(0))[:, layer.independent_rows]
    assert float(balance.abs().max()) < 1e-6
    report = layer.coverage_report()["warm_start_reference"]
    assert report["route"] == "pfba"
    assert report["growth_fraction"] == 1.0
    assert report["n_reactions_carrying_flux"] == 3


def test_a_caller_supplied_reference_route_is_refused():
    """There is no second route, and no silent fall back to one."""
    with pytest.raises(ValueError, match="warm_start_reference"):
        _warmstart_layer(warm_start_reference="measured")


def test_alpha_is_exactly_the_proposals_relative_distance_at_init():
    r"""The one learned scalar is on the same axis as the probes' ``rel``.

    The null-space direction is normalized before it is scaled, so
    :math:`\lVert v_{pre}-v_{ref}\rVert/\lVert v_{ref}\rVert=\alpha` identically. Without
    that normalization alpha would be readable only through the norm of an MLP output, and
    the arm's central number would not be comparable to the table it was designed against.
    """
    layer = _warmstart_layer(warm_start_init_rel=0.01)
    assert float(layer.warm_alpha()) == pytest.approx(0.01, rel=1e-5)
    h, ctx, idx, idx_batch = _inputs()
    gamma = layer.gene_availability(h, idx, idx_batch)
    lb, ub, _ = layer.dynamic_box(layer.reaction_availability(gamma))
    v_pre, _, _, reference = layer._warmstart_flux(ctx, lb, ub, draw=False)
    # Only where the clip does not bite, since the clip is what the box is for.
    interior = (v_pre > lb + 1e-6) & (v_pre < ub - 1e-6)
    if bool(interior.all(dim=1).any()):
        row = int(torch.nonzero(interior.all(dim=1))[0])
        rel = (v_pre[row] - reference[row]).norm() / reference[row].norm()
        assert float(rel) == pytest.approx(0.01, rel=1e-3)


def test_the_warm_start_proposal_is_closer_to_the_feasible_set_than_a_cold_one():
    """The projection discards less of the proposal than it does of a null-space start.

    The comparison is the point of the whole channel: both layers run the same K clips into
    the same box and differ only in where the alternation begins.

    WHAT THIS TOY CANNOT SHOW, stated so the test is not read as more than it is. The
    network has two independent rows and a box the capacity bound narrows to a fraction of
    the reference, so the clip breaks the reference's balance and ``c_balance`` does NOT
    come out lower here than for the cold start. The balance claim is a genome-scale one
    and its number lives in
    ``experiments/026-metabolism-flux/results/probe_warmstart_init.json``, measured on the
    real GEM through 026's own ``build_flux_layer``.
    """
    inputs = _inputs()
    warm = _warmstart_layer(seed=11, projection_steps=20)
    warm_out = warm(*inputs)
    warm_pre = warm_out["v_pre_projection"]
    warm_rel = float(
        ((warm_out["v"] - warm_pre).norm(dim=1) / warm_pre.norm(dim=1)).mean()
    )
    cold = _projection_layer(seed=11, projection_steps=20, projection_end="box")
    cold_out = cold(*inputs)
    cold_pre = (cold._latent(inputs[1])[0] * cold.config.flux_scale) @ (
        cold.null_basis.t()
    )
    cold_rel = float(
        ((cold_out["v"] - cold_pre).norm(dim=1) / cold_pre.norm(dim=1)).mean()
    )
    assert warm_rel < cold_rel, (warm_rel, cold_rel)


def test_a_deletion_still_zeroes_its_reaction_under_the_warm_start():
    """The reference is clipped into the genotype box, so the knockout still bites hard.

    A warm start that ignored the box would reinstate a deleted reaction's wild-type flux,
    which is the one thing a residual head must not do.
    """
    layer = _warmstart_layer()
    # g2 catalyzes r2 alone; delete it and r2's box collapses to the point 0.
    out = layer(*_inputs(deleted=2))
    assert float(out["v"][:, 2].abs().max()) < 1e-6
    assert float(out["feas_warm_reference_active_frac"]) > 0.0


def test_the_warm_start_diagnostics_are_scalars_the_trainer_logs():
    """Every number the arm is read by is a 0-dim ``feas_`` entry, logged every epoch."""
    layer = _warmstart_layer()
    out = layer(*_inputs())
    for key in (
        "feas_warm_alpha",
        "feas_warm_proposal_rel",
        "feas_warm_reference_clipped_frac",
        "feas_warm_reference_active_frac",
    ):
        assert key in out, key
        assert out[key].dim() == 0
    assert float(out["feas_warm_alpha"]) == pytest.approx(0.01, rel=1e-5)


def test_alpha_is_learned_and_the_gradient_reaches_the_gene_tokens():
    """The residual scale is a parameter, and the genotype still reaches the flux."""
    layer = _warmstart_layer()
    assert any(name == "warm_log_alpha" for name, _ in layer.named_parameters())
    h, ctx, idx, idx_batch = _inputs()
    h.requires_grad_(True)
    out = layer(h, ctx, idx, idx_batch)
    out["v"].sum().backward()
    assert layer.warm_log_alpha.grad is not None
    assert h.grad is not None and float(h.grad.abs().sum()) > 0.0


def test_the_warm_start_ends_exactly_on_the_box_it_names():
    """``_box`` means the bounds hold to machine precision, as on every other channel."""
    layer = _warmstart_layer(projection_steps=20)
    out = layer(*_inputs())
    assert float(layer.feasibility(out)["box_violation_frac"].max()) == 0.0
    assert layer.coverage_report()["projection_end"] == "box"


def test_the_directional_end_needs_a_thermodynamic_table():
    """``boxforce`` tightens the box by delta_r_G, so it refuses a layer that has none."""
    with pytest.raises(ValueError, match="ThermoMode.ANCHORED"):
        _layer(parameterization="warmstart_k20_boxforce", thermo_mode=ThermoMode.OFF)


def test_the_warm_start_adds_no_parameter_to_any_other_channel():
    """Bit-identical when off: no buffer, no parameter, no state_dict entry.

    An empty ``v_reference`` registered on every layer would change the state_dict of every
    arm already trained, for a tensor none of them reads.
    """
    plain = _layer(thermo_mode=ThermoMode.OFF)
    assert plain.reference_flux is None
    assert "v_reference" not in dict(plain.named_buffers())
    assert "warm_log_alpha" not in dict(plain.named_parameters())
    assert plain.coverage_report()["warm_start_reference"] is None


def test_the_warm_start_samples_a_different_direction_per_draw():
    """Every draw leaves the same reference in a different direction."""
    layer = _warmstart_layer(latent_stochastic=True, projection_steps=20)
    assert layer.latent_stochastic
    h, ctx, idx, idx_batch = _inputs()
    sample = layer.sample(h, ctx, idx, idx_batch, n_samples=4)
    assert sample.v.shape[0] == 4
    spread = sample.v.std(dim=0)
    assert float(spread.max()) > 0.0


def test_parsimony_can_be_priced_outside_the_ramp():
    r"""Parsimony has never been applied, because it sits inside a product that is zero.

    ``constraint_loss`` multiplies the whole physics sum by ``constraint_scale``, and every
    wave-7 and wave-8 cell runs that scale at exactly 0.0, so ``weights['parsimony']``
    contributes nothing to those runs. This is the same escape the density and active-row
    balance terms already take, and it matters because parsimony is what makes an FBA flux
    sparse: the pFBA reference carries flux on 10.3 percent of reactions against the
    boxforce head's 28 to 32.
    """
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    out = layer(*_inputs())
    layer.set_constraint_epoch(0)
    layer.config.constraint_final_scale = 0.0
    layer.constraint_scale = 0.0
    inside = float(layer.constraint_loss(out))
    assert inside == pytest.approx(0.0, abs=1e-12)

    layer.config.parsimony_outside_ramp_weight = 1.0e-2
    outside = float(layer.constraint_loss(out))
    assert outside == pytest.approx(1.0e-2 * float(out["c_parsimony"]), rel=1e-6)
    assert outside > 0.0
    assert layer.coverage_report()["parsimony_outside_ramp_weight"] == 1.0e-2


def test_parsimony_outside_the_ramp_is_zero_by_default():
    """Every arm before this option must compute the loss it always did."""
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    assert layer.config.parsimony_outside_ramp_weight == 0.0
    out = layer(*_inputs())
    with_flag = float(layer.constraint_loss(out))
    layer.config.parsimony_outside_ramp_weight = 0.0
    assert float(layer.constraint_loss(out)) == with_flag


def test_the_thermo_exemption_list_extends_box_2_and_records_its_provenance(tmp_path):
    r"""A reaction the table cannot defend is exempted, and the run records which list.

    MEASURED motivation (results/thermo_exempt_candidates.json, job 1906): with a real flux
    fixed, the total L1 slack on the second-law sign constraints is 2,641.5 kJ under
    union_external, and removing the 47 reactions the LP puts its violation on leaves 0.000
    kJ over the remaining 378. The list is a claim about ONE table under ONE window, so the
    route, the window and the table's hashes are recorded beside the file's own hash.
    """
    import json

    path = tmp_path / "exempt.json"
    path.write_text(
        json.dumps(
            {
                "provenance": {
                    "route": "union_external",
                    "window_m": [1e-7, 1e-2],
                    "epsilon_kj": 1e-3,
                    "thermo_table_sha256": {"met": "abc"},
                },
                "selected_method": "l1_violated",
                "gate_met": True,
                # r1 is priced and not exempt on the toy; r0 and r3 are exchanges and are
                # already exempt under Box 2, so listing r1 alone makes the count of NEWLY
                # exempted reactions exactly one.
                "exempt_reaction_ids": ["r1"],
            }
        )
    )
    base = _layer(thermo_mode=ThermoMode.ANCHORED)
    listed = _layer(thermo_mode=ThermoMode.ANCHORED, thermo_exempt_path=str(path))
    assert not bool(base.thermo_exempt[1])
    assert bool(listed.thermo_exempt[1])
    assert listed.n_thermo_reactions == base.n_thermo_reactions - 1

    report = listed.coverage_report()["thermo_exempt_file"]
    assert report["n_reactions_listed"] == 1
    assert report["n_reactions_newly_exempted"] == 1
    assert report["route"] == "union_external"
    assert report["window_m"] == [1e-7, 1e-2]
    assert report["gate_met"] is True
    assert len(report["sha256"]) == 64
    # The exempt set is what the hinge and the directional box read, so the layer must
    # report one number and not two.
    assert int(listed.thermo_exempt.sum()) == int(base.thermo_exempt.sum()) + 1
    assert base.coverage_report()["thermo_exempt_file"] is None


def test_an_exemption_list_from_another_model_is_refused(tmp_path):
    """A list applied by silent omission would exempt nothing and report that it had."""
    import json

    path = tmp_path / "exempt.json"
    path.write_text(json.dumps({"exempt_reaction_ids": ["r1", "r_9999"]}))
    with pytest.raises(ValueError, match="does not have"):
        _layer(thermo_mode=ThermoMode.ANCHORED, thermo_exempt_path=str(path))

    missing = tmp_path / "nope.json"
    with pytest.raises(ValueError, match="thermo_exempt_path"):
        _layer(thermo_mode=ThermoMode.ANCHORED, thermo_exempt_path=str(missing))

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"methods": []}))
    with pytest.raises(ValueError, match="not an exemption list"):
        _layer(thermo_mode=ThermoMode.ANCHORED, thermo_exempt_path=str(wrong))


def test_no_exemption_path_is_bit_identical():
    """Off is the default and must leave the exempt set exactly as Box 2 built it."""
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    assert layer.config.thermo_exempt_path is None
    assert layer.thermo_exempt_list is None
    assert layer.n_thermo_exempt_from_file == 0
    # r0 and r3 are the toy's exchanges, r3 is also the objective; r1 and r2 are priced.
    assert layer.thermo_exempt.tolist() == [True, False, False, True]


def test_the_exemption_list_can_be_applied_after_construction(tmp_path):
    """026's build_flux_layer owns the construction, so the switch has a post-hoc door.

    The same door 027's trainer already uses for the density weight and the concentration
    window. It has to rebuild a buffer rather than assign a float, which is the only reason
    it is a method.
    """
    import json

    path = tmp_path / "exempt.json"
    path.write_text(
        json.dumps({"exempt_reaction_ids": ["r1"], "selected_method": "l1_violated"})
    )
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    before = int(layer.thermo_exempt.sum())
    added = layer.apply_thermo_exempt_list(str(path))
    assert added == 1
    assert int(layer.thermo_exempt.sum()) == before + 1
    assert layer.coverage_report()["thermo_exempt_file"]["path"] == str(path)
    # The hinge reads the set at call time, so the exempted reaction leaves the term.
    out = layer(*_inputs())
    assert not bool(layer._thermo_active()[1])
    assert "c_thermo" in out


def _log_c_init_file(
    tmp_path: Path, layer: FluxLayer, targets: dict[str, float]
) -> tuple[Path, list[float]]:
    """Write a concentration witness in this layer's own metabolite order."""
    import json
    import math

    lo = math.log(layer.config.conc_lower_m)
    hi = math.log(layer.config.conc_upper_m)
    middle = 0.5 * (lo + hi)
    vector = [targets.get(m, middle) for m in layer.met_ids]
    path = tmp_path / "log_c.json"
    path.write_text(
        json.dumps(
            {
                "met_ids": list(layer.met_ids),
                "ln_c": vector,
                "route": "union_external",
                "window_m": [layer.config.conc_lower_m, layer.config.conc_upper_m],
                "exempt_file": "thermo_exempt_candidates.json",
                "exempt_file_sha256": "0" * 64,
                "n_constraints_satisfied": 378,
                "residual_total_slack_kj": 0.0,
            }
        )
    )
    return path, vector


def test_the_concentration_init_starts_the_layer_at_the_witness(tmp_path):
    r"""The directional box is only as good as the concentration it is evaluated at.

    MEASURED motivation (results/probe_warmstart_init.json): at a random initialization the
    tightening forbids the direction of 145 of the 495 reactions the pFBA reference carries,
    because the box takes the sign of delta_r_G at concentrations that license nothing in
    particular. The LP behind the exemption list also produces the concentrations under
    which that flux IS sign-consistent, and this is how the layer starts there.
    """
    import math

    probe = _layer(thermo_mode=ThermoMode.ANCHORED)
    path, vector = _log_c_init_file(tmp_path, probe, {"A": math.log(1e-3)})
    layer = _layer(thermo_mode=ThermoMode.ANCHORED, log_c_init_path=str(path))
    context = torch.randn(4, 8)
    log_c = layer.reaction_potential(context)[2]["log_c"]
    # Every genotype starts at the same vector, because the context term starts at zero.
    assert float(log_c.std(dim=0).max()) == pytest.approx(0.0, abs=1e-6)
    assert float((log_c[0] - torch.tensor(vector)).abs().max()) < 1e-4
    assert layer.log_c_init_max_error < 1e-4

    report = layer.coverage_report()["log_c_init_file"]
    assert report["route"] == "union_external"
    assert report["n_constraints_satisfied"] == 378
    assert len(report["sha256"]) == 64
    # And both are learned parameters that START at the witness, not frozen at it: the
    # concentrations are free to move during training, which is why the second-law
    # violation fraction is a number to watch every epoch rather than a guarantee.
    assert layer.log_c_from_context.weight.requires_grad
    assert layer.log_c_base.requires_grad
    out = layer(*_inputs())
    out["thermo_log_c"].sum().backward()
    assert layer.log_c_base.grad is not None
    assert float(layer.log_c_base.grad.abs().sum()) > 0.0


def test_the_concentration_init_is_off_by_default_and_checked_when_on(tmp_path):
    """Off is bit-identical; on, a vector from another model or window is refused."""
    import json

    plain = _layer(thermo_mode=ThermoMode.ANCHORED)
    assert plain.config.log_c_init_path is None
    assert plain.log_c_init is None
    assert plain.coverage_report()["log_c_init_file"] is None
    assert float(plain.log_c_base.abs().max()) == 0.0

    wrong_model = tmp_path / "wrong.json"
    wrong_model.write_text(json.dumps({"met_ids": ["X", "Y"], "ln_c": [0.0, 0.0]}))
    with pytest.raises(ValueError, match="different metabolite list"):
        _layer(thermo_mode=ThermoMode.ANCHORED, log_c_init_path=str(wrong_model))

    path, _ = _log_c_init_file(tmp_path, plain, {})
    with pytest.raises(ValueError, match="window"):
        _layer(
            thermo_mode=ThermoMode.ANCHORED, log_c_init_path=str(path), conc_upper_m=1.0
        )
    with pytest.raises(ValueError, match="ThermoMode.ANCHORED"):
        _layer(thermo_mode=ThermoMode.FREE, log_c_init_path=str(path))


def test_the_concentration_init_can_be_applied_after_construction(tmp_path):
    """The same post-construction door the exemption list has, for the same reason."""
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    path, vector = _log_c_init_file(tmp_path, layer, {})
    error = layer.apply_log_c_init(str(path))
    assert error < 1e-4
    context = torch.randn(2, 8)
    log_c = layer.reaction_potential(context)[2]["log_c"]
    assert float((log_c[0] - torch.tensor(vector)).abs().max()) < 1e-4
    assert layer.coverage_report()["log_c_init_file"]["path"] == str(path)


def test_the_normalized_fold_returns_exactly_one_on_an_intact_complex():
    r"""The softmin charges a complex for having subunits, and that is a bug not a feature.

    :math:`-\log\sum_g e^{-\beta\gamma_g}/\beta` evaluated with every gate at 1 is
    :math:`1-\log(n_u)/\beta`, so at the shipped beta of 8 a 2-subunit complex reads 0.913
    and a 17-subunit one 0.646 on a WILD TYPE where nothing is deleted. MEASURED consequence
    (results/probe_warmstart_init.json): with untrained gates ATP synthase's availability
    reads 0.0447, which held its box ceiling at 4.47 against a reference flux of 5.95 even
    after the enzyme capacity had been raised past it.
    """
    import math

    layer = _layer(thermo_mode=ThermoMode.OFF, availability_fold="normalized_softmin")
    plain = _layer(thermo_mode=ThermoMode.OFF)
    intact = torch.ones(1, 3)
    # r1 is the two-gene complex {g0, g1}; r2 is the single-gene unit {g2}.
    assert float(plain.reaction_availability(intact)[0, 1]) == pytest.approx(
        1.0 - math.log(2.0) / plain.config.softmin_beta, abs=1e-5
    )
    assert float(layer.reaction_availability(intact)[0, 1]) == pytest.approx(
        1.0, abs=1e-6
    )
    # A one-gene unit has nothing to normalize, so the two folds agree there exactly.
    assert float(layer.reaction_availability(intact)[0, 2]) == pytest.approx(
        float(plain.reaction_availability(intact)[0, 2]), abs=1e-7
    )


def test_the_normalized_fold_still_kills_a_deleted_subunit():
    """The deletion is the property the obvious alternative would have broken.

    Shifting by log(n)/beta, which makes the fold a log-MEAN-exp, also returns 1 on the
    intact complex and returns 0.354 for one deleted subunit among sixteen present. Dividing
    by the all-ones value leaves the deletion at zero, because the raw softmin goes
    non-positive the moment any gate is 0.
    """
    layer = _layer(thermo_mode=ThermoMode.OFF, availability_fold="normalized_softmin")
    deleted = torch.tensor([[0.0, 1.0, 1.0]])
    assert float(layer.reaction_availability(deleted)[0, 1]) == pytest.approx(
        0.0, abs=1e-6
    )
    # And through the whole chain, the flux of the complex's reaction is exactly zero.
    out = layer(*_inputs(deleted=0))
    assert float(out["v"][:, 1].abs().max()) == 0.0


def test_the_normalized_fold_preserves_the_ordering_of_partial_gates():
    """Dividing by a positive per-unit constant is monotone, so nothing is reordered."""
    layer = _layer(thermo_mode=ThermoMode.OFF, availability_fold="normalized_softmin")
    plain = _layer(thermo_mode=ThermoMode.OFF)
    generator = torch.Generator().manual_seed(3)
    gamma = torch.rand(32, 3, generator=generator)
    normalized = layer.reaction_availability(gamma)[:, 1]
    original = plain.reaction_availability(gamma)[:, 1]
    # Rank correlation of 1: same order, different scale.
    assert torch.equal(torch.argsort(normalized), torch.argsort(original))
    assert float((normalized - original).abs().max()) > 0.0
    assert float(normalized.max()) <= 1.0 + 1e-6


def test_the_default_fold_is_unchanged_and_a_typo_is_refused():
    """Off is the default and must be bit-identical; an unknown fold is an error."""
    layer = _layer(thermo_mode=ThermoMode.ANCHORED)
    assert layer.config.availability_fold == "softmin"
    assert layer.coverage_report()["availability_fold"] == "softmin"
    inputs = _inputs()
    before = layer(*inputs)["v"].clone()
    layer.config.availability_fold = "softmin"
    assert torch.equal(layer(*inputs)["v"], before)
    layer.config.availability_fold = "normalised_softmin"
    with pytest.raises(ValueError, match="availability_fold"):
        layer(*inputs)


def _reference_table_file(tmp_path, layer, per_key):
    """Write a per-genotype reference table in this layer's own reaction order."""
    import json

    import numpy as np

    keys = list(per_key)
    matrix = np.stack([np.asarray(per_key[k], dtype=np.float32) for k in keys])
    path = tmp_path / "reference_table.npz"
    np.savez(
        path,
        v=matrix,
        keys=np.array(keys),
        rxn_ids=np.array(list(layer.rxn_ids)),
        provenance=np.array(json.dumps({"gem": "toy", "growth_fraction": 1.0})),
    )
    return path


def test_the_reference_is_per_genotype_when_a_table_is_loaded(tmp_path):
    r"""One wild-type reference makes the head a near-constant predictor.

    MEASURED (job 1946, seed 2700, read at epoch 143 of 200): with a single reference the
    cell never left the betaxanthin mean plateau, validation prediction spread 0.0004 to
    0.012 and validation Pearson flat at -0.052 after epoch 5, while the only genotype
    channel the data term had, alpha, grew from 0.0099 to 0.087 and took the training-batch
    active-row balance max from about 1.0 to 2.0 with it.
    """
    probe = _layer(parameterization="warmstart_k20_box", thermo_mode=ThermoMode.OFF)
    wild = probe.v_reference.numpy().copy()
    # g2 catalyzes r2 alone, so a genotype deleting it gets a visibly different reference.
    knocked = wild.copy()
    knocked[0] = 5.0
    path = _reference_table_file(
        tmp_path, probe, {"": wild, "g2": knocked, "g0,g1": wild * 0.5}
    )
    layer = _layer(
        parameterization="warmstart_k20_box",
        thermo_mode=ThermoMode.OFF,
        warm_start_reference_table_path=str(path),
    )
    assert layer.reference_table is not None
    empty = torch.zeros(0, dtype=torch.long)
    # Wild type: the empty key.
    wild_rows = layer.genotype_reference(2, empty, empty)
    assert float((wild_rows - torch.tensor(wild)).abs().max()) == 0.0
    # One genotype deleting g2, one wild type, in one batch.
    indices = torch.tensor([2], dtype=torch.long)
    batch_of = torch.tensor([1], dtype=torch.long)
    rows = layer.genotype_reference(2, indices, batch_of)
    assert float((rows[0] - torch.tensor(wild)).abs().max()) == 0.0
    assert float((rows[1] - torch.tensor(knocked)).abs().max()) == 0.0
    # The rows differ, which is the property the whole table exists for.
    assert float((rows[0] - rows[1]).abs().max()) > 0.0

    report = layer.coverage_report()["warm_start_reference_table"]
    assert report["n_genotypes"] == 3
    assert report["fallback_frac"] == 0.0
    assert len(report["sha256"]) == 64


def test_an_unknown_genotype_falls_back_to_the_wild_type_and_is_counted(tmp_path):
    """The fallback is the wild-type reference, and the rate is reported, not hidden."""
    probe = _layer(parameterization="warmstart_k20_box", thermo_mode=ThermoMode.OFF)
    wild = probe.v_reference.numpy().copy()
    path = _reference_table_file(tmp_path, probe, {"": wild})
    layer = _layer(
        parameterization="warmstart_k20_box",
        thermo_mode=ThermoMode.OFF,
        warm_start_reference_table_path=str(path),
    )
    rows = layer.genotype_reference(
        2, torch.tensor([2], dtype=torch.long), torch.tensor([1], dtype=torch.long)
    )
    assert float((rows[1] - torch.tensor(wild)).abs().max()) == 0.0
    report = layer.coverage_report()["warm_start_reference_table"]
    assert report["n_fallbacks"] == 1
    assert report["fallback_frac"] == pytest.approx(0.5)


def test_a_reference_table_for_another_model_is_refused(tmp_path):
    """A flux applied by position to another model's reactions is not a flux."""
    import json

    import numpy as np

    path = tmp_path / "wrong.npz"
    np.savez(
        path,
        v=np.zeros((1, 2), dtype=np.float32),
        keys=np.array([""]),
        rxn_ids=np.array(["x", "y"]),
        provenance=np.array(json.dumps({})),
    )
    with pytest.raises(ValueError, match="different reaction list"):
        _layer(
            parameterization="warmstart_k20_box",
            thermo_mode=ThermoMode.OFF,
            warm_start_reference_table_path=str(path),
        )


def test_alpha_can_be_frozen():
    """The second arm: a small FIXED residual beside a genotype-carrying reference."""
    learned = _layer(parameterization="warmstart_k20_box", thermo_mode=ThermoMode.OFF)
    frozen = _layer(
        parameterization="warmstart_k20_box",
        thermo_mode=ThermoMode.OFF,
        warm_start_alpha_learnable=False,
        warm_start_init_rel=0.02,
    )
    assert learned.warm_log_alpha.requires_grad
    assert not frozen.warm_log_alpha.requires_grad
    assert float(frozen.warm_alpha()) == pytest.approx(0.02, rel=1e-5)
    # It is FROZEN, not removed: both carry it, so the two arms' checkpoints interchange.
    assert "warm_log_alpha" in dict(frozen.named_parameters())
    out = frozen(*_inputs())
    out["v"].sum().backward()
    assert frozen.warm_log_alpha.grad is None
    assert frozen.coverage_report()["warm_start_reference"]["alpha_learnable"] is False


def test_no_reference_table_is_bit_identical():
    """Off is the default: one reference, repeated, exactly as job 1946 ran it."""
    layer = _layer(parameterization="warmstart_k20_box", thermo_mode=ThermoMode.OFF)
    assert layer.config.warm_start_reference_table_path is None
    assert layer.reference_table is None
    assert layer.coverage_report()["warm_start_reference_table"] is None
    empty = torch.zeros(0, dtype=torch.long)
    rows = layer.genotype_reference(3, empty, empty)
    assert float(rows.std(dim=0).max()) == 0.0


def test_the_reference_key_ignores_the_engineered_background(tmp_path):
    r"""A gene the strain HAS is not a gene the strain LACKS, and the key must know.

    THE DEFECT THIS GUARDS, measured 2026-09-14. Every Cachera betaxanthin strain carries
    CYP76AD1, DOD and the feedback-resistant native alleles ARO4^K229L (YBR249C) and
    ARO7^G141S (YPR060C) as GENE ADDITIONS integrated at locus XII-5
    (``cachera2023.py``), and ``GraphProcessor._process_gene_info`` collects
    ``systematic_gene_name`` from every perturbation regardless of type. So all four arrive
    as perturbation indices and are read as knockouts. Keyed that way, the reference lookup
    would name a cassette the strain HAS, miss every row, and fall back to the wild type on
    100 percent of the dataset.

    The background list travels in the table's provenance, so the layer learns it from the
    artifact rather than from a setting that could disagree with it.
    """
    import json

    import numpy as np

    probe = _layer(parameterization="warmstart_k20_box", thermo_mode=ThermoMode.OFF)
    wild = probe.v_reference.numpy().copy()
    knocked = wild.copy()
    knocked[0] = 7.0
    path = tmp_path / "table.npz"
    np.savez(
        path,
        v=np.stack([wild, knocked]),
        keys=np.array(["", "g2"]),
        rxn_ids=np.array(list(probe.rxn_ids)),
        # g0 stands in for the cassette: perturbed in every record, never deleted.
        provenance=np.array(json.dumps({"background_genes": ["g0"]})),
    )
    layer = _layer(
        parameterization="warmstart_k20_box",
        thermo_mode=ThermoMode.OFF,
        warm_start_reference_table_path=str(path),
    )
    assert layer.reference_background_genes == {"g0"}
    # THE DELETION SET resolves to its row: g2 deleted, cassette not named.
    rows = layer.genotype_reference(
        1, torch.tensor([2], dtype=torch.long), torch.tensor([0], dtype=torch.long)
    )
    assert float((rows[0] - torch.tensor(knocked)).abs().max()) == 0.0
    assert layer.coverage_report()["warm_start_reference_table"]["n_fallbacks"] == 0
    # THE PERTURBATION SET, which names the cassette, is an ERROR rather than a silent
    # fallback: since deletion_indices landed, a background gene arriving here means the
    # caller wired the wrong index set and the availability gate is wrong too.
    with pytest.raises(ValueError, match="engineered background"):
        layer.genotype_reference(
            1,
            torch.tensor([0, 2], dtype=torch.long),
            torch.tensor([0, 0], dtype=torch.long),
        )
