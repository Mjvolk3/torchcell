# experiments/026-metabolism-flux/scripts/probe_warmstart_init.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_warmstart_init
r"""What does the warm-started residual head read at INITIALIZATION on the real GEM?

THE CHECK THIS IS, AND WHAT IT GATES. ``probe_warmstart.py`` measures that a proposal whose
projection displacement is 0.008 clears the turnover-thresholded balance rule at 0.00118
with 82 active rows and a flux-carrying fraction of 0.21, on the real yeast-GEM but with a
SYNTHETIC proposal and no genotype box. The ``warmstart`` parameterization is supposed to
put a trainable head in that regime. Whether it does is a property of the real arm, because
the genotype box is what the reference has to survive.

THE 2x2, AND WHY THESE TWO AXES. The first run of this check failed, and it named two
measured causes with different remedies, so the second run varies exactly those two:

``exemption``
    whether the 47 reactions no concentration in the window can make downhill are exempted
    from the second law (``FluxLayerConfig.thermo_exempt_path``, list from
    ``thermo_exempt_gate.py``). This is what decides how many of the reference's reactions
    the DIRECTIONAL tightening forbids outright.
``abundance``
    ``prior`` is the budget-consistent per-gene abundance the abundance head starts from,
    which gives a capacity ceiling with a median near 0.045 mmol/gDW/h; ``nominal`` is the
    ``E_bar = 1e-3`` path the layer takes when ``use_abundance_head`` is off, which is two
    to three orders of magnitude looser. This is what decides how many of the reference's
    reactions exceed their own ceiling.

Both are selected through switches that already exist: ``use_abundance_head`` in the arm
dict, and the exemption list on the built layer. The ``fluxforce_k100_boxforce`` control is
built identically at the same seed in the first row.

THE REACHABILITY LP. Beside the five statistics, this asks the question the statistics
cannot: is there ANY point of the genotype box with :math:`Sv=0` that turns over more than a
handful of rows? Total turnover is :math:`\sum_j c_j|v_j|` with
:math:`c_j=\tfrac12\sum_i|S_{ij}|`, so maximizing it over a polytope is maximizing a CONVEX
function and is not an LP. What is solved instead is the split relaxation
:math:`v=p-n,\ p,n\ge0` with the original box preserved and :math:`p_j+n_j` capped at the
box width, which gives BOTH bounds at once: its objective is an upper bound on the
reachable turnover, and the point :math:`v=p-n` it returns is genuinely feasible, so the
active-row count of that point is a lower bound witness. If the upper bound is small, the
box cannot carry a dense balanced flux and the box, not the proposal, was the enemy.

Run from the repo root, on CPU::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_warmstart_init.py --batch 4

Writes ``experiments/026-metabolism-flux/results/probe_warmstart_init.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
SCRIPTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
sys.path.insert(0, SCRIPTS_DIR)

from train_flux import ARMS, HOMOLOG_MIRROR, OED_MIRROR, build_flux_layer  # noqa: E402

from torchcell.metabolism.parameters import (  # noqa: E402
    KCAT_SIGMA_LADDER,
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    VERIFIED_PAIRS_FEATURE,
    PredictedUnitTable,
    SubstrateMatch,
    _per_unit_from_predictor,
    load_predicted_unit_table,
    resolve_kcat_table,
)

#: The arm every wave-7 and wave-8 boxforce cell is built on. The channel string and the
#: abundance switch are the only things the comparison changes.
BASE_ARM = "flux_projection_anchored_k20_boxend"
#: The registry entry a 027 arm file can name to get the NOMINAL abundance path, since an
#: arm file may set `base_arm` and the channel and nothing else. Its switches are checked
#: against BASE_ARM below rather than assumed.
NOMINAL_BASE_ARM = "flux_nullspace"
EXEMPT_JSON = "thermo_exempt_candidates.json"
LOG_C_JSON = "thermo_consistent_log_c.json"
LOG_C_CENTERED_JSON = "thermo_consistent_log_c_centered.json"
#: The exemption list augmented with the reactions the tightening still forbids at the
#: consistent-concentration initialization. Written by this script, not hand-authored.
EXEMPT_AUGMENTED_JSON = "thermo_exempt_candidates_augmented.json"
TURNOVER_THRESHOLD = 1.0e-3

#: The capacity rules the grid varies. `shipped` is the table every cell to date ran.
CAPACITY_RULES: tuple[str, ...] = (
    "shipped",
    "complex_max",
    "complex_median",
    "sigma_plus_1",
    "sigma_plus_2",
)

#: The control, the 2x2, and the capacity grid. Every row is built from the same seed.
SETTINGS: tuple[dict[str, object], ...] = (
    {
        "name": "control_fluxforce_prior_exempt_off",
        "channel": "fluxforce_k100_boxforce",
        "abundance": "prior",
        "exemption": False,
    },
    {
        "name": "warmstart_prior_exempt_off",
        "channel": "warmstart_k100_boxforce",
        "abundance": "prior",
        "exemption": False,
    },
    {
        "name": "warmstart_prior_exempt_on",
        "channel": "warmstart_k100_boxforce",
        "abundance": "prior",
        "exemption": True,
    },
    {
        "name": "warmstart_nominal_exempt_off",
        "channel": "warmstart_k100_boxforce",
        "abundance": "nominal",
        "exemption": False,
    },
    {
        "name": "warmstart_nominal_exempt_on",
        "channel": "warmstart_k100_boxforce",
        "abundance": "nominal",
        "exemption": True,
    },
    {
        # THE DIRECTION FROM CONSISTENT CONCENTRATIONS. The tightening reads the sign of
        # delta_r_G at the layer's own learned concentrations, and at a random
        # initialization those license nothing in particular. The LP behind the exemption
        # list also produced the concentrations under which the reference flux IS
        # sign-consistent; this row starts the layer there.
        "name": "warmstart_nominal_exempt_on_logc",
        "channel": "warmstart_k100_boxforce",
        "abundance": "nominal",
        "exemption": True,
        "log_c_init": True,
    },
    {
        "name": "warmstart_prior_exempt_on_logc",
        "channel": "warmstart_k100_boxforce",
        "abundance": "prior",
        "exemption": True,
        "log_c_init": True,
    },
    # THE CAPACITY GRID. Everything above holds the k_cat table fixed at the committed one;
    # these vary it on principled rules at the best setting found so far (nominal abundance,
    # exemption on, consistent concentrations), crossed with the two concentration witnesses.
    *(
        {
            "name": f"warmstart_nominal_exempt_on_logc_{rule}_{variant}",
            "channel": "warmstart_k100_boxforce",
            "abundance": "nominal",
            "exemption": True,
            "log_c_init": True,
            "log_c_variant": variant,
            "capacity_rule": rule,
        }
        for rule in CAPACITY_RULES
        for variant in ("vertex", "centered")
    ),
    # THE AVAILABILITY GATE, the second "minimum over 17 subunits" rule stacked on ATP
    # synthase. `normalized_softmin` divides the softmin by its own value on the all-ones
    # vector, so an intact complex reads exactly 1 instead of 1 - log(n_u)/beta.
    *(
        {
            "name": f"warmstart_nominal_exempt_on_logc_{rule}_vertex_gatefix",
            "channel": "warmstart_k100_boxforce",
            "abundance": "nominal",
            "exemption": True,
            "log_c_init": True,
            "log_c_variant": "vertex",
            "capacity_rule": rule,
            "availability_fold": "normalized_softmin",
        }
        for rule in ("complex_max", "sigma_plus_2")
    ),
    # THE ATTRIBUTION OF THE RESIDUAL. The best cell above misses the maximum by 3.1x with
    # every watched reaction held and a clip cost of 0.0073, so the residual is not the
    # reference failing to fit the box. Two candidates remain and these four cells separate
    # them: the reference reactions the tightening still forbids, and the spread in the
    # untrained availability gate across genotypes.
    *(
        {
            "name": f"attrib_sigma_plus_2_gatefix_{tag}",
            "channel": "warmstart_k100_boxforce",
            "abundance": "nominal",
            "exemption": True,
            "log_c_init": True,
            "log_c_variant": "vertex",
            "capacity_rule": "sigma_plus_2",
            "availability_fold": "normalized_softmin",
            "exempt_augmented": augmented,
            "force_availability_one": gate_open,
        }
        for tag, augmented, gate_open in (
            ("baseline", False, False),
            ("exempt14", True, False),
            ("gateopen", False, True),
            ("both", True, True),
        )
    ),
)

#: The three reactions the previous run left over the capacity ceiling.
WATCHED_REACTIONS: tuple[str, ...] = ("r_0226", "r_0439", "r_0961")

#: The row of `probe_warmstart.json` this check is trying to reach.
TARGET = {
    "balance_active_max": 0.00118,
    "balance_active_p95": 0.00038,
    "balance_active_rows": 82.0,
    "flux_active_frac": 0.21,
    "proposal_distance_rel": 0.008,
}
#: The decision rule the follow-up set: a setting wins when it clears both balance
#: statistics at the Genesis tolerance with a row set worth evaluating them over.
WIN_BALANCE_TOLERANCE = 1.0e-2
WIN_MINIMUM_ACTIVE_ROWS = 40



def _per_unit_with(
    parquet_path: str, column: str, n_units: int, aggregation: str
) -> np.ndarray:
    r"""One predictor's :math:`\log_{10}` value per unit, under a chosen subunit rule.

    THIS IS ``parameters._per_unit_from_predictor`` WITH ONE LINE PARAMETERIZED, and the
    equivalence is checked rather than asserted: at ``aggregation='min'`` the caller
    compares this against the shipped helper itself and refuses to continue if they differ.
    The substrate step is unchanged, because the contested convention is the subunit one.

    WHY IT IS CONTESTED. We take the minimum over a complex's subunits, "a complex cannot
    turn over faster than its slowest subunit"; GECKO 3 takes the highest predicted k_cat
    across subunits, reading the spread as noise around one complex rate. Both are readings
    of what a per-subunit predictor returns when asked a question it was not trained on, and
    the reaction that sets wild-type growth under our capacity bound, ATP synthase, is a
    17-subunit complex whose predictors disagree by 0.993 decades.
    """
    frame = pd.read_parquet(parquet_path, columns=["unit_id", "gene_id", column])
    frame = frame[np.isfinite(frame[column]) & (frame[column] > 0)]
    per_gene = frame.groupby(["unit_id", "gene_id"], as_index=False)[column].median()
    grouped = per_gene.groupby("unit_id")[column]
    per_unit = {"min": grouped.min, "max": grouped.max, "median": grouped.median}[
        aggregation
    ]()
    values = np.full(n_units, np.nan)
    index = per_unit.index.to_numpy().astype(int)
    values[index] = np.log10(per_unit.to_numpy())
    return values


def _predicted_table_with(
    shipped: PredictedUnitTable, n_units: int, aggregation: str
) -> PredictedUnitTable:
    """The admitted ensemble's median per unit, under a chosen subunit rule.

    The predictor SELECTION, the file paths and their hashes are the shipped table's, so
    nothing about which predictors are admitted changes; only the rule that turns a
    complex's per-subunit predictions into one number for the complex.
    """
    stacked = [
        _per_unit_with(path, "k_cat", n_units, aggregation)
        for path in shipped.source_paths.values()
    ]
    matrix = np.vstack(stacked)
    n_predictors = np.isfinite(matrix).sum(axis=0)
    mask = n_predictors > 0
    log_median = np.full(n_units, np.nan)
    log_median[mask] = np.nanmedian(matrix[:, mask], axis=0)
    values = np.zeros(n_units)
    values[mask] = np.power(10.0, log_median[mask])
    spread = n_predictors >= 2
    log_sd = np.full(n_units, np.nan)
    log_sd[spread] = np.nanstd(matrix[:, spread], axis=0, ddof=1)
    return shipped.model_copy(
        update={
            "values": torch.from_numpy(values).float(),
            "mask": torch.from_numpy(mask),
            "log10_sd": torch.from_numpy(log_sd).float(),
            "n_predictors": torch.from_numpy(n_predictors.astype(np.int64)),
        }
    )


def capacity_rules(units, model_dir: str) -> dict[str, dict[str, object]]:
    """One k_cat table per capacity rule, all from the shipped predictor parquets.

    Two families, and they answer different questions.

    ``complex_max`` / ``complex_median``
        the subunit aggregation swapped. A convention change, not an uncertainty
        statement: it asks what the same predictions say under GECKO 3's reading.
    ``sigma_plus_1`` / ``sigma_plus_2``
        the shipped table widened by its own tier ladder,
        ``k_cat <- k_cat * 10^(k * sigma_u)`` with sigma from
        ``parameters.KCAT_SIGMA_LADDER`` (measured 0.50, homolog 0.73, predicted
        max(0.47, per-unit sd), default 1.23 decades). An uncertainty statement, not a
        convention change: +1 sigma is about 3x and +2 sigma about 10x.
    """
    kinetics_dir = osp.join(os.environ["DATA_ROOT"], KINETICS_MIRROR)
    substrates = SubstrateMatch.from_verified_pairs(
        osp.join(kinetics_dir, VERIFIED_PAIRS_FEATURE)
    )
    plausibility = osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
    shipped_predicted = load_predicted_unit_table(
        "k_cat", units.n_units, kinetics_dir, plausibility
    )
    # THE EQUIVALENCE CHECK. If this reimplementation of the aggregation differs from the
    # shipped helper at the shipped setting, every number below is about a different table.
    for path in shipped_predicted.source_paths.values():
        mine = _per_unit_with(path, "k_cat", units.n_units, "min")
        theirs, _ = _per_unit_from_predictor(path, "k_cat", units.n_units)
        assert np.allclose(mine, theirs, equal_nan=True), (
            f"the probe's subunit aggregation does not reproduce "
            f"parameters._per_unit_from_predictor on {path}."
        )

    def resolve(predicted: PredictedUnitTable):
        return resolve_kcat_table(
            units,
            model_dir,
            OED_MIRROR,
            predicted=predicted,
            substrates=substrates,
            homolog_mirror_dir=HOMOLOG_MIRROR,
        )

    shipped_table = resolve(shipped_predicted)
    sigma = KCAT_SIGMA_LADDER.sigma(
        shipped_table.provenance, per_unit_sd=shipped_predicted.log10_sd
    )
    out: dict[str, dict[str, object]] = {
        "shipped": {
            "kcat_per_s": shipped_table.values.clone(),
            "description": "the committed table: median over admitted predictors, MINIMUM over a complex's subunits",
        }
    }
    for rule in ("max", "median"):
        table = resolve(_predicted_table_with(shipped_predicted, units.n_units, rule))
        out[f"complex_{rule}"] = {
            "kcat_per_s": table.values.clone(),
            "description": f"the same predictors with the {rule.upper()} over a complex's subunits",
        }
    for k in (1, 2):
        widened = shipped_table.values * torch.pow(
            10.0, float(k) * sigma.float()
        )
        out[f"sigma_plus_{k}"] = {
            "kcat_per_s": widened,
            "description": (
                f"the shipped table times 10^({k} sigma_u) with sigma from the tier "
                "ladder (measured 0.50, homolog 0.73, predicted max(0.47, per-unit sd), "
                "default 1.23 decades)"
            ),
        }
    for name, payload in out.items():
        values = payload["kcat_per_s"]
        payload["median_kcat_per_s"] = float(values.median())
        payload["ratio_to_shipped_median"] = float(
            (values / out["shipped"]["kcat_per_s"].clamp(min=1e-12)).median()
        )
    return out




def sign_forbidden_ids(layer, h: torch.Tensor, context: torch.Tensor) -> list[str]:
    """Every reference reaction whose direction the tightened box forbids, over the batch.

    The union across genotypes, not a top-N listing: the augmented exemption list below is
    built from it, so a reaction missed here would be silently left priced.
    """
    with torch.no_grad():
        _, _, tight_lb, tight_ub, _, _ = genotype_boxes(layer, h, context)
        reference = layer.v_reference.unsqueeze(0)
        forbidden = ((reference > 1e-9) & (tight_ub < 1e-9)) | (
            (reference < -1e-9) & (tight_lb > -1e-9)
        )
        columns = torch.nonzero(forbidden.any(dim=0), as_tuple=False).flatten()
    return [layer.rxn_ids[int(j)] for j in columns.tolist()]


def write_augmented_exempt(
    base_path: str, extra: list[str], layer, out_path: str
) -> str:
    """The base exemption list plus the reactions the tightening still forbids at init.

    A MEASUREMENT, NOT A POLICY, and the file says so. The base 47 come from the LP: no
    concentration in the window can make them downhill. These extra ones are a different
    claim, weaker and initialization-dependent: at the concentration the LP chose, with this
    capacity rule and this availability fold, the directional box excludes the direction the
    pFBA reference runs them in. They are written to their own file with that sentence
    attached so the two can never be quoted as one set.
    """
    import json

    with open(base_path) as handle:
        base = json.load(handle)
    base_ids = list(base["exempt_reaction_ids"])
    added = [r for r in extra if r not in set(base_ids)]
    payload = dict(base)
    payload["exempt_reaction_ids"] = base_ids + added
    payload["selected_method"] = "l1_violated_plus_init_sign_forbidden"
    payload["augmentation"] = {
        "base_file": osp.basename(base_path),
        "base_sha256": hashlib.sha256(
            json.dumps(base, indent=2).encode()
        ).hexdigest(),
        "n_base": len(base_ids),
        "n_added": len(added),
        "added_reaction_ids": added,
        "how_added_were_found": (
            "reactions carrying flux in the layer's own pFBA reference whose direction the "
            "boxforce tightening forbids, measured on a batch of 4 untrained genotypes at "
            "the consistent-concentration initialization, capacity rule sigma_plus_2, "
            "availability fold normalized_softmin"
        ),
        "what_this_is_not": (
            "the base list is an LP result: no concentration in the window makes those "
            "reactions downhill. The added ones are initialization-dependent and weaker: "
            "they are forbidden at ONE concentration vector under ONE capacity rule. The "
            "two sets must not be quoted as one."
        ),
        "log_c_init": LOG_C_JSON,
        "capacity_rule": "sigma_plus_2",
        "availability_fold": "normalized_softmin",
    }
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return out_path


def watched_capacity(layer, h: torch.Tensor, context: torch.Tensor) -> list[dict]:
    """The capacity of the three named reactions against the reference flux they must hold.

    Reported per capacity rule, because "the over-ceiling count fell" is only actionable
    with the reaction that moved and by how much.
    """
    with torch.no_grad():
        _, _, tight_lb, tight_ub, capacity, _ = genotype_boxes(layer, h, context)
        reference = layer.v_reference
        rows = []
        for name in WATCHED_REACTIONS:
            j = layer.rxn_ids.index(name)
            ceiling = float(
                tight_ub[:, j].median()
                if float(reference[j]) > 0
                else -tight_lb[:, j].median()
            )
            rows.append(
                {
                    "reaction": name,
                    "reference_flux": float(reference[j]),
                    "capacity": (
                        None if capacity is None else float(capacity[:, j].median())
                    ),
                    "box_ceiling": ceiling,
                    "excess": (
                        float(abs(float(reference[j])) / ceiling)
                        if ceiling > 1e-12
                        else None
                    ),
                    "held": bool(abs(float(reference[j])) <= ceiling),
                }
            )
    return rows


def statistics(layer, out: dict[str, torch.Tensor]) -> dict[str, float]:
    """The audit's own statistics, computed off one forward pass."""
    v = out["v"]
    scale = float(layer.config.flux_scale)
    residual = layer._s_matmul(v)[:, layer.independent_rows].abs()
    omega = layer.turnover(v)[:, layer.independent_rows]
    ratio = residual / (omega + layer.config.balance_epsilon)
    active = omega > TURNOVER_THRESHOLD * scale
    selected = ratio[active]
    stats = {
        "balance_active_max": float(selected.max()) if selected.numel() else float("nan"),
        "balance_active_p95": (
            float(torch.quantile(selected, 0.95)) if selected.numel() else float("nan")
        ),
        "balance_active_rows": float(active.float().sum(dim=1).mean()),
        "flux_active_frac": float(out["feas_flux_active_frac"]),
        "balance_median_all": float(ratio.median()),
        "box_violation_frac": float(out["feas_box_violation_frac"]),
        "thermo_violation_frac": float(out["feas_thermo_violation_frac"]),
        "c_balance": float(out["c_balance"]),
        "c_density": float(out["c_density"]),
    }
    if "v_pre_projection" in out:
        pre = out["v_pre_projection"]
        stats["proposal_distance_rel"] = float(
            ((v - pre).norm(dim=1) / pre.norm(dim=1).clamp(min=1e-12)).mean()
        )
        stats["v_pre_active_frac"] = float(
            (pre.abs() > 1.0e-6 * scale).float().mean()
        )
    for key in (
        "feas_warm_alpha",
        "feas_warm_proposal_rel",
        "feas_warm_reference_clipped_frac",
        "feas_warm_reference_active_frac",
    ):
        if key in out:
            stats[key] = float(out[key].detach())
    return stats


def genotype_boxes(layer, h: torch.Tensor, context: torch.Tensor):
    """The box before and after the directional tightening, as ``forward`` builds them."""
    empty = torch.zeros(0, dtype=torch.long)
    gamma = layer.gene_availability(h, empty, empty)
    c_j = layer.reaction_availability(gamma)
    e_u = (
        layer.unit_abundance(layer.gene_abundance(h, gamma))
        if layer.config.use_abundance_head
        else None
    )
    lb, ub, capacity = layer.dynamic_box(c_j, e_u)
    delta_g, _, _ = layer.reaction_potential(context)
    tight_lb, tight_ub, emptied = layer._thermo_tighten(lb, ub, delta_g)
    return lb, ub, tight_lb, tight_ub, capacity, emptied


def reference_fit(layer, h: torch.Tensor, context: torch.Tensor) -> dict[str, object]:
    """Does the genotype box CONTAIN the reference flux, and what does the clip cost?

    This is the diagnosis half of the check, and it is the half that says what to do when
    the statistics miss their target. The residual head's whole mechanism is that the clip
    is small at small alpha; if the box is narrower than the reference, the clip is not
    small, the clipped reference is not balanced, and the projection is back to repairing an
    O(1)-wrong proposal, which ``probe_projrate.py`` shows it cannot do at any K.

    Three things are separated, because they have different remedies:

    ``sign_forbidden``
        reference reactions whose direction the box excludes outright. Only the DIRECTIONAL
        tightening can do this, so this count is zero under ``_box`` and positive under
        ``_boxforce``, and its remedy is the thermodynamic table or an exemption list.
    ``over_ceiling``
        reference reactions inside the licensed direction whose MAGNITUDE exceeds the box,
        which is the enzyme-capacity bound. Its remedy is the abundance path.
    ``clip_cost``
        the two together, as a fraction of the reference's norm.
    """
    with torch.no_grad():
        lb, ub, tight_lb, tight_ub, capacity, emptied = genotype_boxes(
            layer, h, context
        )
        reference = layer.v_reference
        carrying = reference.abs() > 1e-9
        out: dict[str, object] = {
            "reference_n_carrying_flux": int(carrying.sum()),
            "reference_max_abs_flux": float(reference.abs().max()),
            "reference_norm": float(reference.norm()),
            "thermo_emptied_box_frac": float(emptied.float().mean()),
        }
        if capacity is not None:
            gpr = layer.has_gpr
            out["capacity_median_over_gpr_reactions"] = float(capacity[:, gpr].median())
            out["capacity_min_over_gpr_reactions"] = float(capacity[:, gpr].min())
        for tag, (low, high) in {
            "box": (lb, ub),
            "boxforce": (tight_lb, tight_ub),
        }.items():
            clipped = torch.minimum(
                torch.maximum(reference.unsqueeze(0).expand_as(low), low), high
            )
            forbidden = ((reference.unsqueeze(0) > 1e-9) & (high < 1e-9)) | (
                (reference.unsqueeze(0) < -1e-9) & (low > -1e-9)
            )
            ceiling = torch.where(reference.unsqueeze(0) > 0, high, -low)
            over = (
                carrying.unsqueeze(0)
                & (~forbidden)
                & (reference.abs().unsqueeze(0) > ceiling)
            )
            excess = (reference.abs().unsqueeze(0) / ceiling.clamp(min=1e-12))[over]
            block = {
                "clip_cost_rel": float(
                    ((clipped - reference).norm(dim=1) / reference.norm()).mean()
                ),
                "n_sign_forbidden": float(forbidden.float().sum(dim=1).mean()),
                "n_over_ceiling": float(over.float().sum(dim=1).mean()),
                "median_excess_over_ceiling": (
                    float(excess.median()) if excess.numel() else None
                ),
                # NAMED, because "3 reactions over their ceiling" is only actionable with
                # the ids: the clip cost is dominated by whichever of them carries the
                # reference's largest flux, and the remedy is that reaction's kcat or its
                # enzyme abundance rather than anything in the head.
                "over_ceiling_reactions": _named_offenders(
                    layer, reference, ceiling, over, forbidden
                ),
                "clipped": _flux_statistics(layer, clipped),
            }
            projected = clipped.clone()
            for step in range(layer.projection_steps):
                projected = torch.minimum(torch.maximum(projected, low), high)
                if step < layer.projection_steps - 1:
                    projected = layer._project_balanced(projected)
            block["clipped_then_projected"] = _flux_statistics(layer, projected)
            out[tag] = block
    return out


def _named_offenders(
    layer,
    reference: torch.Tensor,
    ceiling: torch.Tensor,
    over: torch.Tensor,
    forbidden: torch.Tensor,
    limit: int = 12,
) -> dict[str, list[dict[str, object]]]:
    """The reactions the box cannot hold, by name, split by which stage excludes them."""

    def rows(mask: torch.Tensor, by_excess: bool) -> list[dict[str, object]]:
        columns = torch.nonzero(mask.any(dim=0), as_tuple=False).flatten()
        entries = []
        for j in columns.tolist():
            entries.append(
                {
                    "reaction": layer.rxn_ids[j],
                    "reference_flux": float(reference[j]),
                    "ceiling": float(ceiling[:, j].median()),
                    "excess": (
                        float(reference[j].abs() / ceiling[:, j].median().clamp(min=1e-12))
                        if by_excess
                        else None
                    ),
                    "norm_share": float(reference[j] ** 2 / reference.pow(2).sum()),
                }
            )
        entries.sort(key=lambda row: -float(row["norm_share"]))
        return entries[:limit]

    return {"over_ceiling": rows(over, True), "sign_forbidden": rows(forbidden, False)}


def _flux_statistics(layer, v: torch.Tensor) -> dict[str, float]:
    """The balance rule and the density, on a flux the layer did not produce."""
    scale = float(layer.config.flux_scale)
    residual = layer._s_matmul(v)[:, layer.independent_rows].abs()
    omega = layer.turnover(v)[:, layer.independent_rows]
    ratio = residual / (omega + layer.config.balance_epsilon)
    active = omega > TURNOVER_THRESHOLD * scale
    selected = ratio[active]
    return {
        "balance_active_max": (
            float(selected.max()) if selected.numel() else float("nan")
        ),
        "balance_active_p95": (
            float(torch.quantile(selected, 0.95)) if selected.numel() else float("nan")
        ),
        "balance_active_rows": float(active.float().sum(dim=1).mean()),
        "flux_active_frac": float((v.abs() > 1.0e-6 * scale).float().mean()),
    }


def turnover_reachability(
    layer, low: torch.Tensor, high: torch.Tensor, sample: int = 0
) -> dict[str, object]:
    r"""Can ANY balanced point of this genotype's box turn over more than a few rows?

    THE RELAXATION, stated plainly because the answer is only as good as it is. Total
    turnover is :math:`\sum_j c_j|v_j|`, a convex function, so its maximum over a polytope
    is not an LP. Splitting :math:`v=p-n` with :math:`p,n\ge0`, keeping the ORIGINAL box on
    each side rather than the lazy relaxation, and capping :math:`p_j+n_j` at the box width
    gives a linear program whose objective is an UPPER BOUND on reachable turnover (it can
    count a two-sided reaction twice) and whose solution :math:`v=p-n` is a genuinely
    feasible point.

    So the two numbers answer the question from both ends: ``active_rows_at_lp_point`` is a
    witness that the box CAN carry that many turning-over rows, and
    ``turnover_upper_bound`` caps what any point could do. Reported beside the pFBA
    reference's own 82 rows.

    The objective is scored ONLY on the reactions whose box lies on one side of zero, where
    the split is exact; the two-sided reactions contribute their box width to the upper
    bound as ``two_sided_turnover_headroom`` instead. Rewarding them in the objective lets
    the LP buy its bound with a p and an n that cancel, which returns a witness that turns
    nothing over.
    """
    dense = torch.sparse_coo_tensor(
        layer.s_indices, layer.s_values, (layer.n_metabolites, layer.n_reactions)
    ).to_dense().double().numpy()
    scale = float(layer.config.flux_scale)
    lo = low[sample].double().numpy()
    hi = high[sample].double().numpy()
    n = dense.shape[1]
    weight = 0.5 * np.abs(dense).sum(axis=0)
    sparse_s = csr_matrix(dense)
    # The split box, exactly equal to the original one, as in parsimonious_reference_flux.
    p_bounds: list[tuple[float, float]] = []
    n_bounds: list[tuple[float, float]] = []
    two_sided: list[int] = []
    for j in range(n):
        a, b = float(lo[j]), float(hi[j])
        if a > b:  # an inverted box, which the clip resolves to the upper bound
            a = b
        if a >= 0.0:
            p_bounds.append((a, b))
            n_bounds.append((0.0, 0.0))
        elif b <= 0.0:
            p_bounds.append((0.0, 0.0))
            n_bounds.append((-b, -a))
        else:
            p_bounds.append((0.0, b))
            n_bounds.append((0.0, -a))
            two_sided.append(j)
    # REACTIONS WHOSE BOX EXCLUDES ZERO. The GEM forces a nonzero flux on its maintenance
    # reaction, and the directional tightening can leave that force pointing into a
    # direction the rest of the network cannot balance, in which case the box holds no
    # balanced point at all and the LP below is infeasible before it optimizes anything.
    # That is an answer, not a failure, so it is measured rather than worked around: the
    # program is solved twice, once as built and once with those reactions relaxed to admit
    # zero, and both statuses are reported.
    forced = [
        j
        for j in range(n)
        if p_bounds[j][0] > 0.0 or n_bounds[j][0] > 0.0
    ]
    a_eq = hstack([sparse_s, -sparse_s], format="csr")
    a_ub = None
    b_ub = None
    if two_sided:
        # p_j + n_j <= max(ub_j, -lb_j) on the reactions where the split can double count.
        columns = np.asarray(two_sided)
        row_index = np.repeat(np.arange(len(two_sided)), 2)
        column_index = np.empty(2 * len(two_sided), dtype=int)
        column_index[0::2] = columns
        column_index[1::2] = columns + n
        a_ub = csr_matrix(
            (np.ones(column_index.size), (row_index, column_index)),
            shape=(len(two_sided), 2 * n),
        )
        b_ub = np.maximum(hi[columns], -lo[columns])
    started = time.time()
    # THE OBJECTIVE IS EXACT ON THE ONE-SIDED REACTIONS AND ZERO ON THE OTHERS. For a
    # reaction whose box lies on one side of zero, exactly one of p and n can be nonzero, so
    # c_j (p_j + n_j) IS c_j |v_j| and the split introduces no gap at all. Rewarding the
    # two-sided reactions instead lets the LP buy objective with a p and an n that cancel,
    # which inflates the bound and leaves a witness that turns nothing over. After the
    # directional tightening only 86 to 117 of 4,131 reactions are two-sided, so scoring the
    # rest exactly costs almost nothing and makes the returned point a real maximizer.
    exact_weight = weight.copy()
    if two_sided:
        exact_weight[np.asarray(two_sided, dtype=int)] = 0.0

    def solve(pb, nb, w):
        return linprog(
            -np.concatenate([w, w]),
            A_eq=a_eq,
            b_eq=np.zeros(dense.shape[0]),
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=pb + nb,
            method="highs",
        )

    program = solve(p_bounds, n_bounds, exact_weight)
    out: dict[str, object] = {
        "lp_status_as_built": int(program.status),
        "lp_message": str(program.message)[:120],
        "n_two_sided_reactions": len(two_sided),
        "n_reactions_forced_off_zero": len(forced),
        "forced_reactions": [layer.rxn_ids[j] for j in forced[:10]],
        "forced_relaxed": False,
    }
    if not program.success and forced:
        relaxed_p = list(p_bounds)
        relaxed_n = list(n_bounds)
        for j in forced:
            relaxed_p[j] = (0.0, relaxed_p[j][1])
            relaxed_n[j] = (0.0, relaxed_n[j][1])
        program = solve(relaxed_p, relaxed_n, exact_weight)
        out["forced_relaxed"] = True
        out["lp_status_forced_relaxed"] = int(program.status)
    out["lp_seconds"] = round(time.time() - started, 1)
    if not program.success:
        return out
    # The remaining gap, so the upper bound below is honest: the most the two-sided
    # reactions could add to the turnover is their box width times their row weight.
    slack = (
        float(
            sum(
                weight[j] * max(float(hi[j]), -float(lo[j]))
                for j in two_sided
            )
        )
        if two_sided
        else 0.0
    )
    out["two_sided_turnover_headroom"] = slack
    v = torch.from_numpy(program.x[:n] - program.x[n:]).float().unsqueeze(0)
    omega = layer.turnover(v)[:, layer.independent_rows]
    residual = layer._s_matmul(v)[:, layer.independent_rows].abs()
    active = omega > TURNOVER_THRESHOLD * scale
    out.update(
        {
            "one_sided_turnover_at_optimum": float(-program.fun),
            "turnover_upper_bound": float(-program.fun) + slack,
            "turnover_at_lp_point": float(omega.sum()),
            "active_rows_at_lp_point": int(active.sum()),
            "flux_active_frac_at_lp_point": float(
                (v.abs() > 1.0e-6 * scale).float().mean()
            ),
            "balance_residual_max_at_lp_point": float(residual.max()),
            "reference_active_rows": int(
                (
                    layer.turnover(layer.v_reference.unsqueeze(0))[
                        :, layer.independent_rows
                    ]
                    > TURNOVER_THRESHOLD * scale
                ).sum()
            )
            if hasattr(layer, "v_reference")
            else None,
        }
    )
    return out


def _gene_ids() -> list[str]:
    """The GEM's own gene ids, which is the gene universe an init check needs.

    A real cell passes the transformer's 6,607-token universe; the flux layer only ever
    reads the GEM's subset of it, so an untrained check on the GEM's own gene list exercises
    exactly the same chain with no dataset build.
    """
    from torchcell.metabolism.constraints import build_gem_tensors
    from torchcell.metabolism.yeast_GEM import YeastGEM

    source = YeastGEM()
    gem = build_gem_tensors(source.model, model_dir=source.model_dir)
    return list(gem.catalytic_units.gene_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-lp", action="store_true")
    parser.add_argument(
        "--only",
        default="",
        help=(
            "Comma-separated setting names to run, or a prefix. Empty runs every setting. "
            "The attribution cells need the augmented exemption list, which is written "
            "from the baseline cell's own measurement before they build."
        ),
    )
    parser.add_argument("--out", default="probe_warmstart_init.json")
    args = parser.parse_args()

    started = time.time()
    torch.set_num_threads(8)
    exempt_path = osp.join(RESULTS_DIR, EXEMPT_JSON)
    log_c_path = osp.join(RESULTS_DIR, LOG_C_JSON)
    centered_path = osp.join(RESULTS_DIR, LOG_C_CENTERED_JSON)
    gene_ids = _gene_ids()
    from torchcell.metabolism.constraints import build_gem_tensors
    from torchcell.metabolism.yeast_GEM import YeastGEM

    source = YeastGEM()
    gem = build_gem_tensors(source.model, model_dir=source.model_dir)
    rules = capacity_rules(gem.catalytic_units, source.model_dir)
    print("capacity rules:")
    for name, payload in rules.items():
        print(
            f"  {name:16s} median k_cat {payload['median_kcat_per_s']:.4g} 1/s, "
            f"ratio to shipped {payload['ratio_to_shipped_median']:.4g}"
        )

    # THE NOMINAL PATH HAS TO BE REACHABLE FROM AN ARM FILE, not only from this script. A
    # 027 arm file may name `base_arm` and the channel and nothing else, so the claim that
    # `flux_nullspace` is the nominal-abundance twin of BASE_ARM is checked rather than
    # asserted: the two must differ in the abundance switch and the parameterization alone.
    base = dict(ARMS[BASE_ARM])
    nominal = dict(ARMS[NOMINAL_BASE_ARM])
    differing = sorted(
        key
        for key in set(base) | set(nominal)
        if base.get(key) != nominal.get(key)
    )
    arm_twin_check = {
        "base_arm": BASE_ARM,
        "nominal_base_arm": NOMINAL_BASE_ARM,
        "differing_keys": differing,
        "is_twin": differing == ["parameterization", "use_abundance_head"],
    }
    print(f"arm twin check: {arm_twin_check}")

    wanted = [w.strip() for w in args.only.split(",") if w.strip()]
    selected_settings = [
        setting
        for setting in SETTINGS
        if not wanted or any(str(setting["name"]).startswith(w) for w in wanted)
    ]
    print(f"running {len(selected_settings)} of {len(SETTINGS)} settings")

    # THE AUGMENTED EXEMPTION LIST IS MEASURED, NOT CONFIGURED. It is built from the
    # baseline attribution cell's own forbidden set before any cell that consumes it is
    # constructed, so the file and the measurement can never drift apart.
    augmented_path = osp.join(RESULTS_DIR, EXEMPT_AUGMENTED_JSON)
    if any(setting.get("exempt_augmented") for setting in selected_settings):
        arm = dict(ARMS[BASE_ARM])
        arm["parameterization"] = "warmstart_k100_boxforce"
        arm["kcat_tier"] = "predicted"
        arm["thermo_source"] = "union_external"
        arm["use_abundance_head"] = False
        torch.manual_seed(args.seed)
        probe_layer, _ = build_flux_layer(arm, gene_ids, args.hidden)
        probe_layer.apply_thermo_exempt_list(exempt_path)
        probe_layer.apply_log_c_init(log_c_path)
        probe_layer.config.availability_fold = "normalized_softmin"
        probe_layer.kcat_per_h = (
            rules["sigma_plus_2"]["kcat_per_s"].to(probe_layer.kcat_per_h) * 3600.0
        )
        torch.manual_seed(args.seed)
        probe_h = torch.randn(args.batch, len(gene_ids), args.hidden)
        probe_context = torch.randn(args.batch, args.hidden)
        forbidden_ids = sign_forbidden_ids(probe_layer, probe_h, probe_context)
        write_augmented_exempt(exempt_path, forbidden_ids, probe_layer, augmented_path)
        print(
            f"augmented exemption list: {len(forbidden_ids)} reactions forbidden at init, "
            f"written to {augmented_path}"
        )

    rows: list[dict[str, object]] = []
    for setting in selected_settings:
        arm = dict(ARMS[BASE_ARM])
        arm["parameterization"] = setting["channel"]
        arm["kcat_tier"] = "predicted"
        arm["thermo_source"] = "union_external"
        arm["use_abundance_head"] = setting["abundance"] == "prior"
        torch.manual_seed(args.seed)
        build_started = time.time()
        layer, meta = build_flux_layer(arm, gene_ids, args.hidden)
        assert layer is not None
        n_exempted = 0
        if setting["exemption"]:
            n_exempted = layer.apply_thermo_exempt_list(
                augmented_path if setting.get("exempt_augmented") else exempt_path
            )
        if setting.get("force_availability_one"):
            # THE GATE OPENED, as a probe of the box and not as a config option. Zeroing the
            # availability head's weight and pushing its bias positive sends every gate to
            # 1, so with the normalized fold an intact complex reads exactly 1 and the box
            # is the GEM's own bound times the capacity. It answers what the untrained
            # per-genotype spread costs, which no setting of the layer can answer.
            with torch.no_grad():
                nn.init.zeros_(layer.availability.weight)
                nn.init.constant_(layer.availability.bias, 20.0)
        rule = str(setting.get("capacity_rule", "shipped"))
        if rule != "shipped":
            # The capacity bound is recomputed from `kcat_per_h` inside `dynamic_box` at
            # call time, so swapping the buffer swaps the rule for everything downstream.
            layer.kcat_per_h = rules[rule]["kcat_per_s"].to(layer.kcat_per_h) * 3600.0
        variant = str(setting.get("log_c_variant", "vertex"))
        # Read at call time by `reaction_availability`, so the assignment takes effect on a
        # layer 026's build_flux_layer has already constructed.
        layer.config.availability_fold = str(
            setting.get("availability_fold", "softmin")
        )
        log_c_error = None
        if setting.get("log_c_init"):
            log_c_error = layer.apply_log_c_init(
                log_c_path if variant == "vertex" else centered_path
            )
        build_seconds = time.time() - build_started
        torch.manual_seed(args.seed)
        h = torch.randn(args.batch, len(gene_ids), args.hidden)
        context = torch.randn(args.batch, args.hidden)
        empty = torch.zeros(0, dtype=torch.long)
        with torch.no_grad():
            out = layer(h, context, empty, empty)
        row: dict[str, object] = {
            "name": setting["name"],
            "channel": setting["channel"],
            "abundance": setting["abundance"],
            "exemption": setting["exemption"],
            "n_reactions_newly_exempted": n_exempted,
            "log_c_init": bool(setting.get("log_c_init")),
            "exempt_augmented": bool(setting.get("exempt_augmented")),
            "force_availability_one": bool(setting.get("force_availability_one")),
            "log_c_variant": variant if setting.get("log_c_init") else None,
            "capacity_rule": rule,
            "availability_fold": layer.config.availability_fold,
            "capacity_rule_description": rules[rule]["description"],
            "capacity_rule_median_kcat_per_s": rules[rule]["median_kcat_per_s"],
            "capacity_rule_ratio_to_shipped": rules[rule]["ratio_to_shipped_median"],
            "log_c_init_max_error_ln_c": log_c_error,
            "coverage_log_c_init_file": layer.coverage_report().get("log_c_init_file"),
            "n_reactions_second_law_applied": int(
                (layer.delta_r_g0_mask & (~layer.thermo_exempt)).sum()
            ),
            "base_arm": BASE_ARM,
            "build_seconds": round(build_seconds, 1),
            "coverage_warm_start_reference": layer.coverage_report().get(
                "warm_start_reference"
            ),
            "coverage_thermo_exempt_file": layer.coverage_report().get(
                "thermo_exempt_file"
            ),
            "statistics": statistics(layer, out),
        }
        if layer.parameterization == "warmstart":
            row["reference_fit"] = reference_fit(layer, h, context)
            row["watched_reactions"] = watched_capacity(layer, h, context)
        if not args.skip_lp:
            with torch.no_grad():
                lb, ub, tight_lb, tight_ub, _, _ = genotype_boxes(layer, h, context)
            # BOTH ENDS. The tightened box is what the arm runs in; the untightened one is
            # the ceiling the same box reaches once the sign constraint is taken OUT of it,
            # which is what says whether the tightening or the capacity is binding.
            row["reachability"] = turnover_reachability(layer, tight_lb, tight_ub)
            row["reachability_untightened"] = turnover_reachability(layer, lb, ub)
        rows.append(row)
        print(f"-- {setting['name']} (built in {build_seconds:.1f}s)")
        print(json.dumps(row["statistics"], indent=2))
        if "reference_fit" in row:
            print("reference fit:", json.dumps(row["reference_fit"], indent=2))
        if "reachability" in row:
            print("reachability (boxforce):", json.dumps(row["reachability"], indent=2))
            print(
                "reachability (box):",
                json.dumps(row["reachability_untightened"], indent=2),
            )

    def wins(row: dict[str, object]) -> bool:
        stats = row["statistics"]
        return (
            row["channel"].startswith("warmstart")
            and stats["balance_active_max"] < WIN_BALANCE_TOLERANCE
            and stats["balance_active_p95"] < WIN_BALANCE_TOLERANCE
            and stats["balance_active_rows"] >= WIN_MINIMUM_ACTIVE_ROWS
        )

    winners = [row["name"] for row in rows if wins(row)]
    payload = {
        "definitions": {
            "target": (
                "the rel-0.01 row of probe_warmstart.json: balance_active_max 0.00118, "
                "p95 0.00038, flux_active_frac 0.21, 82 active rows, projection "
                "displacement 0.008. That row has no genotype box on it, so it is the "
                "target rather than the prediction."
            ),
            "win_rule": (
                f"balance_active_max and p95 both below {WIN_BALANCE_TOLERANCE} with at "
                f"least {WIN_MINIMUM_ACTIVE_ROWS} active rows."
            ),
            "reachability": (
                "an LP over the split box: the objective is an UPPER bound on reachable "
                "turnover and the returned point is a feasible witness, so a small upper "
                "bound means the box cannot carry a dense balanced flux at all."
            ),
        },
        "target": TARGET,
        "batch": args.batch,
        "hidden": args.hidden,
        "seed": args.seed,
        "untrained": True,
        "exempt_list_path": exempt_path,
        "log_c_init_path": log_c_path,
        "log_c_centered_path": centered_path,
        "capacity_rules": {
            name: {
                k: v for k, v in payload.items() if k != "kcat_per_s"
            }
            for name, payload in rules.items()
        },
        "arm_twin_check": arm_twin_check,
        "winners": winners,
        "channels": rows,
    }
    out_path = osp.join(RESULTS_DIR, args.out)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"winners: {winners or 'none'}")
    print(f"wrote {out_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
