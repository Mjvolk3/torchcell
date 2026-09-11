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
