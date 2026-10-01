# tests/torchcell/metabolism/test_flux_layer.py
# [[tests.torchcell.metabolism.test_flux_layer]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/metabolism/test_flux_layer.py

"""Tests for the differentiable flux layer.

These are property tests, not regression tests. The properties are the claims the layer
makes -- the box holds exactly, a deletion zeros its reactions, the null-space
parameterization balances mass to machine precision -- and each one is a claim that would
otherwise be checked only by reading the code.

They run on a SMALL SYNTHETIC network rather than on yeast-GEM so they need no data root
and no download. The one test that touches the real model is marked and skipped when the
checkout is absent.

2026.09.30 - closed forms (Phase 15). The exact tests pin the network's arithmetic by
zeroing the last ``flux_mlp`` layer (so every logit ``z`` is 0 and ``sigmoid(z) = 1/2``,
putting each flux at its box midpoint) and, where gene availability matters, the
``availability`` gate (so every mapped gene has ``gamma = sigmoid(0) = 1/2``).

* Two-reaction balance (``r0: -> A`` in ``[0, 10]``, ``r1: A ->`` in ``[0, 4]``, no GPR):
  ``v = (5, 2)``, residual ``Sv = 5 - 2 = 3``, turnover ``omega = (5 + 2) / 2 = 3.5``,
  ``c_balance = (3 / (3.5 + 1e-6))^2 = 0.7346935``, ``c_parsimony = 3.5 / 100``. A
  constant logit parks every flux at its box midpoint, so mass balance holds only when
  the midpoints happen to balance; and on an irreversible box ``[0, ub]`` the flux is
  ``ub * sigmoid(z) > 0`` for every finite ``z``, so an exact zero (a sparse flux) is
  out of reach.
* Softmin and isozyme sum at ``beta = 8``: a two-gene complex at ``gamma = (1, 1)`` gives
  ``c_u = 1 - ln(2) / 8 = 0.9133566``; at ``gamma = (1/2, 0)`` gives
  ``-(1/8) ln(e^-4 + 1) = -0.0022639``, clamped to 0. Two single-gene isozymes at
  ``1/4`` each sum to ``1/2``; at ``3/4`` each they sum to ``3/2``, clamped to 1.
* Enzyme capacity: ``kcat = 1 /s`` gives ``3600 /h``, times the nominal ``1e-3`` is a
  ceiling of ``3.6`` per unit, scaled by ``c_j``.
* Second law on ``EX_A: -> A``, ``R1: A -> B``, ``EX_B: B ->`` with ``Delta_f G = (0,
  -10)`` kJ/mol, both known: ``Delta_r G'0 = (0, -10, 10)``; with the learned
  log-concentrations at the window midpoint ``m = (ln 1e-7 + ln 1e-2) / 2`` the
  concentration term cancels on R1, so ``Delta_r G_R1 = -10``. Run backwards (``v_R1 =
  -5``) the drive is ``+50`` and ``c_thermo = (50 + 1e-3) / (50 + 1e-3) = 1``; run
  forwards it is 0. Dissipation over the two exchanges is ``5 (RT m) + 5 (10 - RT m) =
  50`` J/gDW/h.
"""

import math
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


# -- Phase 15: closed forms, refusals and Decision 12 identities ----------------------


def _units(
    pairs: list[tuple[int, int]], unit_reaction: list[int], gene_ids: list[str]
) -> CatalyticUnits:
    """CatalyticUnits from ``(unit, gene)`` pairs and the reaction of each unit."""
    index = (
        torch.tensor(pairs, dtype=torch.long).t()
        if pairs
        else torch.zeros(2, 0, dtype=torch.long)
    )
    return CatalyticUnits(
        unit_gene_index=index,
        unit_reaction=torch.tensor(unit_reaction, dtype=torch.long),
        n_units=len(unit_reaction),
        n_multigene_units=0,
        n_reactions_with_gpr=len(set(unit_reaction)),
        gene_ids=gene_ids,
    )


def _thermo(met_delta_g: list[float], met_mask: list[bool], n_rxn: int) -> ThermoTable:
    mask = torch.tensor(met_mask)
    rxn_mask = torch.zeros(n_rxn, dtype=torch.bool)
    return ThermoTable(
        met_delta_g=torch.tensor(met_delta_g),
        met_mask=mask,
        rxn_delta_g=torch.zeros(n_rxn),
        rxn_mask=rxn_mask,
        met_coverage=TableCoverage.of(mask.numpy()),
        rxn_coverage=TableCoverage.of(rxn_mask.numpy()),
        source_paths={},
        sha256={},
    )


def _gem(
    dense: list[list[float]],
    lb: list[float],
    ub: list[float],
    rxn_ids: list[str],
    units: CatalyticUnits,
    thermo: ThermoTable | None = None,
    exchange: list[int] | None = None,
    biomass: int | None = None,
) -> GemTensors:
    s = torch.tensor(dense).to_sparse_coo().coalesce()
    return GemTensors(
        s=s,
        lb=torch.tensor(lb),
        ub=torch.tensor(ub),
        met_ids=[f"m{i}" for i in range(len(dense))],
        rxn_ids=rxn_ids,
        catalytic_units=units,
        thermo=thermo,
        biomass_index=biomass,
        exchange_indices=None if exchange is None else torch.tensor(exchange),
        n_metabolites=len(dense),
        n_reactions=len(rxn_ids),
    )


def _head(layer: FluxLayer) -> torch.nn.Linear:
    """The last ``flux_mlp`` layer, which emits the logit ``z``."""
    head = layer.flux_mlp[2]
    assert isinstance(head, torch.nn.Linear)
    return head


def _zero_heads(layer: FluxLayer) -> None:
    """Zero the last flux layer (z = 0) and the availability gate (gamma = 1/2)."""
    with torch.no_grad():
        _head(layer).weight.zero_()
        _head(layer).bias.zero_()
        layer.availability.weight.zero_()
        layer.availability.bias.zero_()
        if layer.thermo_mode is ThermoMode.ANCHORED:
            layer.log_c_from_context.weight.zero_()
            layer.log_c_from_context.bias.zero_()


def _no_pert() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)


def _two_reaction_layer(**overrides: Any) -> FluxLayer:
    """``r0: -> A`` in [0, 10], ``r1: A ->`` in [0, 4]; neither reaction has a GPR."""
    gem = _gem(
        [[1.0, -1.0]], [0.0, 0.0], [10.0, 4.0], ["r0", "r1"], _units([], [], ["g0"])
    )
    cfg = FluxLayerConfig(
        hidden_dim=2, reaction_embed_dim=2, thermo_mode=ThermoMode.OFF, **overrides
    )
    layer = FluxLayer(gem, ["g0"], config=cfg)
    _zero_heads(layer)
    return layer


def test_two_reaction_mass_balance_closed_form() -> None:
    """At z = 0 both fluxes sit at their box midpoints and the residual is 3 of 3.5.

    v = (5, 2); Sv = 3; omega = 3.5; c_balance = (3 / 3.500001)^2; the median residual
    ratio is the one row's 3 / 3.500001; c_parsimony = mean(|v|) / 100 = 0.035; no GPR
    means no enzyme demand, so the budget terms are exactly 0; the loss is
    c_balance + 1e-3 * c_parsimony.
    """
    layer = _two_reaction_layer()
    out = layer(torch.zeros(1, 1, 2), torch.zeros(1, 2), *_no_pert())
    assert out["v"].tolist() == [[5.0, 2.0]]
    ratio = 3.0 / (3.5 + 1e-6)
    assert float(out["c_balance"]) == pytest.approx(ratio**2, rel=1e-6)
    assert float(out["feas_balance_median"]) == pytest.approx(ratio, rel=1e-6)
    assert float(out["c_parsimony"]) == pytest.approx(0.035, rel=1e-6)
    assert float(out["protein_used"]) == 0.0
    assert float(out["c_budget"]) == 0.0
    assert float(out["c_thermo"]) == 0.0
    assert float(out["c_dissipation"]) == 0.0
    assert float(layer.constraint_loss(out)) == pytest.approx(
        ratio**2 + 1e-3 * 0.035, rel=1e-6
    )


def test_sigmoid_box_cannot_reach_a_sparse_flux() -> None:
    """On an irreversible box [0, ub], v = ub * sigmoid(z) > 0 for every finite z.

    With the last layer's bias at -20 (and at -5), v = (10, 4) * sigmoid(-20) and
    (10, 4) * sigmoid(-5): tiny, strictly positive, never the exact zero a sparse FBA
    solution has on an unused reaction. Only availability c_j = 0 (a deletion) gives 0.
    """
    layer = _two_reaction_layer()
    for bias in (-20.0, -5.0):
        with torch.no_grad():
            _head(layer).bias.fill_(bias)
        v = layer(torch.zeros(1, 1, 2), torch.zeros(1, 2), *_no_pert())["v"]
        s = torch.sigmoid(torch.tensor(bias))
        assert torch.equal(v, torch.tensor([[10.0, 4.0]]) * s)
        assert bool((v > 0).all())


def test_bounds_are_clamped_to_the_flux_scale() -> None:
    """The +/-1000 "unbounded" convention is clamped to +/-flux_scale in the buffers."""
    gem = _gem(
        [[1.0, -1.0]],
        [-1000.0, 0.0],
        [1000.0, 50.0],
        ["r0", "r1"],
        _units([], [], ["g0"]),
    )
    layer = FluxLayer(gem, ["g0"], config=FluxLayerConfig(flux_scale=20.0))
    assert layer.lb.tolist() == [-20.0, 0.0]
    assert layer.ub.tolist() == [20.0, 20.0]


def test_reaction_availability_softmin_and_isozyme_sum() -> None:
    """Complexes take the softmin (clamped at 0), isozymes add (clamped at 1).

    Reactions: r0 has no GPR (always 1), r1 is the complex {g0, g1}, r2 the isozymes
    {g2} or {g3}. Row 1 gamma = (1, 1, 1/4, 1/4): r1 = 1 - ln(2)/8, r2 = 1/4 + 1/4.
    Row 2 gamma = (1/2, 0, 3/4, 3/4): r1 = -(1/8) ln(e^-4 + 1) < 0 -> 0; r2 = 3/2 -> 1.
    """
    gem = _gem(
        [[1.0, -1.0, -1.0]],
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        ["r0", "r1", "r2"],
        _units([(0, 0), (0, 1), (1, 2), (2, 3)], [1, 2, 2], ["g0", "g1", "g2", "g3"]),
    )
    layer = FluxLayer(gem, ["g0", "g1", "g2", "g3"])
    gamma = torch.tensor([[1.0, 1.0, 0.25, 0.25], [0.5, 0.0, 0.75, 0.75]])
    c_j = layer.reaction_availability(gamma)
    assert -(1 / 8) * math.log(math.exp(-4) + 1) < 0
    expected = torch.tensor([[1.0, 1.0 - math.log(2) / 8, 0.5], [1.0, 0.0, 1.0]])
    assert torch.allclose(c_j, expected, atol=1e-6)


def test_dynamic_box_enzyme_capacity() -> None:
    """Kcat 1 /s is a ceiling of 3.6 per unit, scaled by c_j; off leaves the scaled box.

    r0 (no GPR) in [-10, 5], r1 (one unit) in [-10, 10], r2 (no GPR) in [-10, 0]. At
    c = 1: r1 becomes [-3.6, 3.6]. At c = 1/2: the box is halved (r1 [-5, 5]) and the
    ceiling too (1.8). With capacity off the box is only scaled by c.

    Contract (issue #534): capacity bounds ``|v_j| <= cap_j`` and applies only to a
    reaction with a GPR. A reaction without one keeps its own scaled ``[lb, ub]``, so the
    asymmetric r0 stays [-10, 5] (not [-5, 5]) and the reverse-only r2 stays [-10, 0]
    (not the point 0).
    """
    gem = _gem(
        [[1.0, -1.0, 1.0]],
        [-10.0, -10.0, -10.0],
        [5.0, 10.0, 0.0],
        ["r0", "r1", "r2"],
        _units([(0, 0)], [1], ["g0"]),
    )
    layer = FluxLayer(gem, ["g0"], kcat_per_s=torch.tensor([1.0]))
    lb, ub = layer.dynamic_box(torch.tensor([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]))
    assert torch.allclose(lb, torch.tensor([[-10.0, -3.6, -10.0], [-5.0, -1.8, -5.0]]))
    assert torch.allclose(ub, torch.tensor([[5.0, 3.6, 0.0], [2.5, 1.8, 0.0]]))
    off = FluxLayer(
        gem,
        ["g0"],
        config=FluxLayerConfig(use_enzyme_capacity=False),
        kcat_per_s=torch.tensor([1.0]),
    )
    lb_off, ub_off = off.dynamic_box(torch.tensor([[0.5, 0.5, 0.5]]))
    assert lb_off.tolist() == [[-5.0, -5.0, -5.0]]
    assert ub_off.tolist() == [[2.5, 5.0, 0.0]]


def test_default_kcat_and_molecular_weight() -> None:
    """No kcat: 13.7 /s for every unit, 49320 /h; no MW: 40 kDa each; both reported."""
    gem = _gem(
        [[1.0, -1.0]],
        [0.0, 0.0],
        [10.0, 10.0],
        ["r0", "r1"],
        _units([(0, 0), (1, 1)], [0, 1], ["g0", "g1"]),
    )
    layer = FluxLayer(gem, ["g0", "g1"])
    assert torch.allclose(layer.kcat_per_h, torch.tensor([49320.0, 49320.0]))
    assert layer.mw_kda.tolist() == [40.0, 40.0]
    report = layer.coverage_report()
    assert report["kcat_is_default_for_all_units"] is True
    assert report["mw_is_default_for_all_genes"] is True


def test_constitutive_gene_is_pinned_to_one() -> None:
    """A cassette gene with no token keeps gamma 1; mapped genes read the gate (1/2).

    Model universe (g0, g1); GEM genes (g0, g1, g2); g2 constitutive. Sample 0 is
    unperturbed; sample 1 deletes token 0 (g0), which is hard-zeroed. Without the
    constitutive list g2 has no token and reads 0.
    """
    gem = _gem(
        [[1.0, -1.0, -1.0]],
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        ["r0", "r1", "r2"],
        _units([(0, 0), (1, 1), (2, 2)], [0, 1, 2], ["g0", "g1", "g2"]),
    )
    layer = FluxLayer(gem, ["g0", "g1"], constitutive_genes=["g2"])
    plain = FluxLayer(gem, ["g0", "g1"])
    for lay in (layer, plain):
        _zero_heads(lay)
    h = torch.zeros(2, 2, 32)
    pert = (torch.tensor([0]), torch.tensor([1]))
    assert layer.gene_availability(h, *pert).tolist() == [
        [0.5, 0.5, 1.0],
        [0.0, 0.5, 1.0],
    ]
    assert plain.gene_availability(h, *pert).tolist() == [
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
    ]
    assert layer.coverage_report()["n_constitutive_genes"] == 1
    assert layer.coverage_report()["n_gem_genes_in_model_universe"] == 2


def test_constitutive_gene_absent_from_the_gem_is_refused() -> None:
    """A constitutive gene the GEM does not contain raises, naming the sorted unknowns."""
    gem = _gem(
        [[1.0, -1.0]], [0.0, 0.0], [1.0, 1.0], ["r0", "r1"], _units([], [], ["g0"])
    )
    with pytest.raises(
        ValueError,
        match=(
            r"^constitutive_genes not present in the GEM: \['gA', 'gZ'\]\. "
            r"Apply the pathway to the model before building its tensors\.$"
        ),
    ):
        FluxLayer(gem, ["g0"], constitutive_genes=["gZ", "g0", "gA"])


def test_nullspace_without_a_basis_is_refused() -> None:
    """``parameterization='nullspace'`` with no basis raises before any parameter."""
    with pytest.raises(
        ValueError,
        match=(
            r"^parameterization='nullspace' needs the basis; build it with "
            r"torchcell\.metabolism\.constraints\.null_space_basis\(gem\.s\)\.$"
        ),
    ):
        FluxLayer(
            _toy_gem(), ["g0"], config=FluxLayerConfig(parameterization="nullspace")
        )


def test_second_law_exemptions_by_name_without_exchange_or_biomass() -> None:
    """No exchange or biomass index: only ids with H2O (any case) or "water" are exempt."""
    gem = _gem(
        [[1.0, -1.0, 1.0, -1.0]],
        [0.0] * 4,
        [1.0] * 4,
        ["r0", "h2ot", "Water_transport", "r3"],
        _units([], [], ["g0"]),
    )
    layer = FluxLayer(gem, ["g0"])
    assert layer.thermo_exempt.tolist() == [False, True, True, False]
    assert layer.is_exchange.tolist() == [False] * 4
    assert layer.n_thermo_reactions == 2
    assert layer.independent_rows.tolist() == [0]


def test_no_thermo_table_uses_the_compartment_temperature() -> None:
    """No table: zero energies, empty masks, RT from the compartments' 303.15 K."""
    gem = _gem(
        [[1.0, -1.0]], [0.0, 0.0], [1.0, 1.0], ["r0", "r1"], _units([], [], ["g0"])
    )
    layer = FluxLayer(gem, ["g0"])
    assert layer.rt == 303.15 * 8.314462618e-3
    assert layer.delta_r_g0.tolist() == [0.0, 0.0]
    assert layer.delta_r_g0_mask.tolist() == [False, False]
    assert layer.met_g_mask.tolist() == [False]
    report = layer.coverage_report()
    assert report["n_reactions_second_law_applied"] == 0
    assert report["frac_metabolites_delta_f_g_known"] == 0.0
    assert report["thermo_mode"] == "anchored"


def test_shipped_transport_term_fills_only_degenerate_reactions() -> None:
    """The shipped value is used where the recomputed energy is exactly 0 and shipped.

    Metabolites A_c, A_e, B with Delta_f G (-5, -5, -8), all known. t1: A_c -> A_e
    recomputes to 0 and ships -7.5: sourced. r2: A_c -> B recomputes to -3 and ships 4:
    not degenerate, not sourced. t3: A_e -> A_c recomputes to 0 but ships nothing: not
    sourced. With the switch off every entry is 0.
    """
    thermo = _thermo([-5.0, -5.0, -8.0], [True, True, True], 3)
    thermo.rxn_delta_g = torch.tensor([-7.5, 4.0, 0.0])
    thermo.rxn_mask = torch.tensor([True, True, False])
    gem = _gem(
        [[-1.0, -1.0, 1.0], [1.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [0.0] * 3,
        [1.0] * 3,
        ["t1", "r2", "t3"],
        _units([], [], ["g0"]),
        thermo=thermo,
    )
    on = FluxLayer(
        gem, ["g0"], config=FluxLayerConfig(use_shipped_transport_delta_g=True)
    )
    assert on.delta_r_g0.tolist() == [0.0, -3.0, 0.0]
    assert on.delta_r_g_transport.tolist() == [-7.5, 0.0, 0.0]
    assert on.coverage_report()["n_transport_terms_sourced"] == 1
    assert on.coverage_report()["transport_term_is_zero"] is False
    off = FluxLayer(gem, ["g0"])
    assert off.delta_r_g_transport.tolist() == [0.0, 0.0, 0.0]


def _second_law_layer(r1_lb: float, r1_ub: float, **overrides: Any) -> FluxLayer:
    """``EX_A: -> A``, ``R1: A -> B``, ``EX_B: B ->``; Delta_f G (0, -10); exchanges 0, 2."""
    gem = _gem(
        [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]],
        [0.0, r1_lb, 0.0],
        [10.0, r1_ub, 10.0],
        ["EX_A", "R1", "EX_B"],
        _units([], [], ["g0"]),
        thermo=_thermo([0.0, -10.0], [True, True], 3),
        exchange=[0, 2],
    )
    cfg = FluxLayerConfig(hidden_dim=2, reaction_embed_dim=2, **overrides)
    layer = FluxLayer(gem, ["g0"], config=cfg)
    _zero_heads(layer)
    return layer


def test_second_law_hinge_and_dissipation_closed_form() -> None:
    """R1 run backwards against Delta_r G = -10 is fully uphill; forwards it is free.

    Enzyme capacity is on (the default): R1 has no GPR, so its reverse-only box
    [-10, 0] is kept and carries the nonzero flux -5. v = (5, -5, 5) on the reversed box. Delta_r G = (RT m, -10, 10 - RT m) with m the
    window midpoint, so the drive on R1 is (-5)(-10) = 50 and c_thermo = 1; dissipation
    over the exchanges is 5 RT m + 5 (10 - RT m) = 50, a ratio 50 / 10 - 1 = 4 over a
    10 J/gDW/h limit. The zero-initialized offset gives c_thermo_prior exactly 0.
    """
    rt = 303.15 * 8.314462618e-3
    m = (math.log(1e-7) + math.log(1e-2)) / 2
    back = _second_law_layer(-10.0, 0.0, g_diss_limit=10.0)
    out = back(torch.zeros(1, 1, 2), torch.zeros(1, 2), *_no_pert())
    assert out["v"].tolist() == [[5.0, -5.0, 5.0]]
    assert torch.allclose(
        out["delta_r_g"], torch.tensor([[rt * m, -10.0, 10.0 - rt * m]]), atol=1e-5
    )
    assert torch.allclose(out["thermo_log_c"], torch.tensor([[m, m]]))
    assert float(out["c_thermo"]) == pytest.approx(1.0, abs=1e-6)
    assert float(out["feas_thermo_violation_frac"]) == 1.0
    assert float(out["g_diss"]) == pytest.approx(50.0, abs=1e-4)
    assert float(out["c_dissipation"]) == pytest.approx(4.0, abs=1e-5)
    assert float(out["c_thermo_prior"]) == 0.0
    fwd = _second_law_layer(0.0, 10.0, use_enzyme_capacity=False)
    out_fwd = fwd(torch.zeros(1, 1, 2), torch.zeros(1, 2), *_no_pert())
    assert float(out_fwd["c_thermo"]) == 0.0
    assert float(out_fwd["feas_thermo_violation_frac"]) == 0.0
    assert float(out_fwd["c_dissipation"]) == 0.0
    with torch.no_grad():
        fwd.delta_g_offset.copy_(torch.tensor([1.0, 2.0, 3.0]))
    out_off = fwd(torch.zeros(1, 1, 2), torch.zeros(1, 2), *_no_pert())
    assert float(out_off["c_thermo_prior"]) == pytest.approx(14.0 / 3.0)


def test_protein_budget_closed_form_and_switch() -> None:
    """One unit on r1, kcat 1 /s, gene at gamma 1/2: v1 = 0.9, protein 1e-5 g/gDW.

    c_j(r1) = 1/2 halves the box to [0, 2] and the ceiling to 3.6 / 2 = 1.8; v1 = 0.9;
    demand 0.9 / 3600 = 2.5e-4 g; protein 2.5e-4 * 40 kDa * 1e-3 = 1e-5; over a budget
    of 5e-6 the hinge is 1e-5 / 5e-6 - 1 = 1. Budget off: no ``protein_used`` key and
    c_budget exactly 0.
    """
    gem = _gem(
        [[1.0, -1.0]],
        [0.0, 0.0],
        [10.0, 4.0],
        ["r0", "r1"],
        _units([(0, 0)], [1], ["g0"]),
    )

    def build(**overrides: Any) -> FluxLayer:
        cfg = FluxLayerConfig(
            hidden_dim=2, reaction_embed_dim=2, thermo_mode=ThermoMode.OFF, **overrides
        )
        layer = FluxLayer(gem, ["g0"], config=cfg, kcat_per_s=torch.tensor([1.0]))
        _zero_heads(layer)
        return layer

    out = build(p_avail=5e-6)(torch.zeros(1, 1, 2), torch.zeros(1, 2), *_no_pert())
    assert torch.allclose(out["v"], torch.tensor([[5.0, 0.9]]))
    assert torch.allclose(out["e_g"], torch.tensor([[2.5e-4]]))
    assert float(out["protein_used"]) == pytest.approx(1e-5, rel=1e-5)
    assert float(out["feas_budget_ratio"]) == pytest.approx(2.0, rel=1e-5)
    assert float(out["c_budget"]) == pytest.approx(1.0, rel=1e-4)
    out_off = build(use_protein_budget=False)(
        torch.zeros(1, 1, 2), torch.zeros(1, 2), *_no_pert()
    )
    assert "protein_used" not in out_off
    assert "e_g" not in out_off
    assert float(out_off["c_budget"]) == 0.0


def test_stochastic_head_samples_in_training_and_uses_the_mean_in_eval() -> None:
    """Z = mean + eps * exp(clamp(log_std, -6, 2)) in training; z = mean in eval.

    The last layer is zeroed with a log-std bias of 10, clamped to 2, so in training
    v = lb + (ub - lb) * sigmoid(e^2 * eps) with eps the seeded draw; in eval v is the
    box midpoint (5, 2).
    """
    layer = _two_reaction_layer(stochastic=True)
    with torch.no_grad():
        _head(layer).bias.copy_(torch.tensor([0.0, 10.0]))
    inputs = (torch.zeros(1, 1, 2), torch.zeros(1, 2), *_no_pert())
    layer.eval()
    assert layer(*inputs)["v"].tolist() == [[5.0, 2.0]]
    layer.train()
    torch.manual_seed(7)
    v = layer(*inputs)["v"]
    torch.manual_seed(7)
    eps = torch.randn(1, 2)
    expected = torch.tensor([[10.0, 4.0]]) * torch.sigmoid(eps * math.exp(2.0))
    assert torch.allclose(v, expected, atol=1e-6)


def test_seeded_construction_is_deterministic() -> None:
    """Two layers built under the same seed give bitwise-equal outputs on one input."""
    torch.manual_seed(3)
    first = _layer(thermo_mode=ThermoMode.ANCHORED)
    torch.manual_seed(3)
    second = _layer(thermo_mode=ThermoMode.ANCHORED)
    torch.manual_seed(11)
    inputs = _inputs()
    out_a, out_b = first(*inputs), second(*inputs)
    for key in ("v", "c_balance", "c_thermo", "c_budget", "thermo_log_c"):
        assert torch.equal(out_a[key], out_b[key]), key


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            ThermoMode.ANCHORED,
            [
                "availability.bias",
                "availability.weight",
                "delta_g_offset",
                "flux_mlp.0.bias",
                "flux_mlp.0.weight",
                "flux_mlp.2.bias",
                "flux_mlp.2.weight",
                "log_c_base",
                "log_c_from_context.bias",
                "log_c_from_context.weight",
                "reaction_embedding.weight",
            ],
        ),
        (
            ThermoMode.FREE,
            [
                "availability.bias",
                "availability.weight",
                "flux_mlp.0.bias",
                "flux_mlp.0.weight",
                "flux_mlp.2.bias",
                "flux_mlp.2.weight",
                "mu_base",
                "mu_from_context.bias",
                "mu_from_context.weight",
                "reaction_embedding.weight",
            ],
        ),
    ],
)
def test_gradient_reaches_every_parameter(
    mode: ThermoMode, expected: list[str]
) -> None:
    """Box arm: every parameter of the mode gets a nonzero gradient from v and dG."""
    torch.manual_seed(0)
    layer = _layer(thermo_mode=mode)
    out = layer(*_inputs())
    (out["v"].sum() + out["delta_r_g"].sum()).backward()
    names = sorted(n for n, _ in layer.named_parameters())
    assert names == expected
    for name, param in layer.named_parameters():
        assert param.grad is not None, name
        assert float(param.grad.abs().sum()) > 0.0, name


def test_each_arm_builds_only_the_modules_it_reads() -> None:
    """Each parameterization owns exactly the parameters its forward reads (issue #534).

    Contract: the nullspace arm builds ``latent_mlp`` and the availability gate and no box
    head; the box arm builds ``reaction_embedding`` and ``flux_mlp`` and no latent head.
    On the toy network (hidden 8, reaction embedding 4, null-space dimension 2, thermo
    OFF) the counts are: nullspace ``(8*8 + 8) + (8*2 + 2) + (8 + 1) = 99``; box
    ``4*4 + (12*8 + 8) + (8*1 + 1) + (8 + 1) = 138``. In the nullspace arm the gate
    reaches the loss only through the soft box penalty, so the loss is ``v.sum() +
    c_box``; in the box arm ``v.sum()`` reaches every parameter.
    """
    torch.manual_seed(0)
    null = _layer(parameterization="nullspace", thermo_mode=ThermoMode.OFF)
    assert null.n_latent == 2
    assert sorted(n for n, _ in null.named_parameters()) == [
        "availability.bias",
        "availability.weight",
        "latent_mlp.0.bias",
        "latent_mlp.0.weight",
        "latent_mlp.2.bias",
        "latent_mlp.2.weight",
    ]
    assert sum(p.numel() for p in null.parameters()) == 99
    out = null(*_inputs())
    assert float(out["c_box"]) > 0.0
    (out["v"].sum() + out["c_box"]).backward()
    for name, param in null.named_parameters():
        assert param.grad is not None, name
        assert float(param.grad.abs().sum()) > 0.0, name

    box = _layer(thermo_mode=ThermoMode.OFF)
    assert sorted(n for n, _ in box.named_parameters()) == [
        "availability.bias",
        "availability.weight",
        "flux_mlp.0.bias",
        "flux_mlp.0.weight",
        "flux_mlp.2.bias",
        "flux_mlp.2.weight",
        "reaction_embedding.weight",
    ]
    assert sum(p.numel() for p in box.parameters()) == 138
    box(*_inputs())["v"].sum().backward()
    for name, param in box.named_parameters():
        assert param.grad is not None, name
        assert float(param.grad.abs().sum()) > 0.0, name
