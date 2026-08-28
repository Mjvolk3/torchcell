# experiments/024-perturb-seq-costing/scripts/scaling_analysis.py
# [[experiments.024-perturb-seq-costing.scripts.scaling_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/024-perturb-seq-costing/scripts/scaling_analysis
"""How a multiplexed library scales: delivery, combination recovery, environments.

Section 7 of the review asks three questions the earlier sections do not, and all
three are arithmetic rather than opinion:

1. **Delivery.** If several guides per cell come from several *plasmids* rather
   than from an array on one plasmid, how many distinct guides does a cell
   actually carry? Selection conditions on carrying at least one, so the count is
   a zero-truncated Poisson, and two plasmids can carry the same guide.
2. **Recovery.** Do you have to observe the same combination more than once? That
   depends entirely on the estimand, and the two answers differ by orders of
   magnitude. Main effects never require a repeat; a named combination's joint
   transcriptome requires `c` cells carrying exactly it.
3. **Environments.** The other scaling axis. It is linear in cell demand where
   named combinations are quadratic, which is the whole reason it is attractive.
   The cost side of that axis -- what a multi-environment screen costs on
   split-pool against droplet -- reuses the Sec. 5 cost model
   (``cost_model.budget_for``) rather than introducing a new one.

Nothing here is a new statistical result: (2) reuses Eq. 3 of the review, which
is Yao et al.'s expression. What is new is applying it to the delivery routes
that are actually available in yeast, and being explicit that the copy-number
route buys *combinations* without buying *transformants*.

Run:  python experiments/024-perturb-seq-costing/scripts/scaling_analysis.py
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp

from dotenv import load_dotenv
from pydantic import BaseModel

import cost_model as CM
import uiuc_core_data as UC

load_dotenv()
RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "024-perturb-seq-costing", "results"
)

# Whole numbers on purpose. The dial a marker attenuation actually sets is "how
# many plasmid molecules does a surviving cell carry on average", and a reader
# asked, reasonably, what a target of 1.5 was supposed to mean. Integers make the
# column read as what it is: copy number per cell.
TARGET_PLASMIDS_PER_CELL = [2, 3, 4, 5, 8]

# Cells needed to power one second-order interaction to the precision a
# first-order effect gets at 100 cells. Yao et al.'s constant; Sec. 4.5 quotes
# the sentence. Repeated here rather than imported so this file reads standalone.
CELLS_PER_PAIR = 400
FIRST_ORDER_FLOOR = 100


# --- 1. Delivery: how many DISTINCT guides does a cell carry? ----------------
class DeliveryPoint(BaseModel):
    """One operating point of the multi-plasmid ("copy number") delivery route."""

    lam: float  # Poisson mean uptake events per cell, BEFORE selection
    mean_plasmids: float  # E[m | m >= 1], what selection leaves
    mean_distinct: float  # E[distinct guides], after same-guide collisions
    p_at_least_2: float  # fraction of surviving cells carrying >= 2 guides
    p_exactly_1: float


def zero_truncated_poisson_mean(lam: float) -> float:
    """E[m | m >= 1]. Selection for the marker conditions on at least one plasmid."""
    return lam / (1.0 - math.exp(-lam))


def lam_for_target_mean(target: float, lo: float = 1e-6, hi: float = 50.0) -> float:
    """Invert the above: what lambda gives a post-selection mean of `target`?"""
    if target <= 1.0:
        raise ValueError("post-selection mean is > 1 by construction")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if zero_truncated_poisson_mean(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def expected_distinct(lam: float, n_library: int) -> float:
    """E[# distinct guides | m >= 1] when m plasmids are drawn from n guides.

    Two plasmids can carry the same guide. Given m draws with replacement from a
    library of n, the expected number of distinct guides is n(1-(1-1/n)^m). We
    average that over the zero-truncated Poisson on m.
    """
    denom = 1.0 - math.exp(-lam)
    total = 0.0
    for m in range(1, 200):
        p_m = math.exp(-lam) * lam**m / math.factorial(m) / denom
        if p_m < 1e-12 and m > lam + 10:
            break
        total += p_m * n_library * (1.0 - (1.0 - 1.0 / n_library) ** m)
    return total


def delivery_table(n_library: int, targets: list[float]) -> list[DeliveryPoint]:
    out = []
    for t in targets:
        lam = lam_for_target_mean(t)
        denom = 1.0 - math.exp(-lam)
        p1 = (math.exp(-lam) * lam) / denom
        out.append(
            DeliveryPoint(
                lam=lam,
                mean_plasmids=zero_truncated_poisson_mean(lam),
                mean_distinct=expected_distinct(lam, n_library),
                p_at_least_2=1.0 - p1,
                p_exactly_1=p1,
            )
        )
    return out


# --- 2. Recovery: main effects vs a named combination ------------------------
class RecoveryPoint(BaseModel):
    n_targets: int
    k: int
    cells_for_main_effects: float  # every gene at FIRST_ORDER_FLOOR
    n_pairs: int
    p_specific_pair: float  # chance a random k-subset contains a given pair
    cells_for_all_pairs: float  # Eq. 3
    expected_repeats_per_pair: float  # at the main-effect budget, how often is a
    #                                   given pair seen? This is the user's
    #                                   "will I ever see it twice" question.


def recovery(n_targets: int, k: int, floor: int = FIRST_ORDER_FLOOR) -> RecoveryPoint:
    n_pairs = n_targets * (n_targets - 1) // 2
    # Main effects: each cell reports on k genes, so N*k/T cells inform each gene.
    cells_main = floor * n_targets / k
    p_pair = (
        (k * (k - 1)) / (n_targets * (n_targets - 1)) if k >= 2 and n_targets >= 2 else 0.0
    )
    cells_pairs = (
        CELLS_PER_PAIR * n_pairs / (k * (k - 1) / 2) if k >= 2 else float("inf")
    )
    return RecoveryPoint(
        n_targets=n_targets,
        k=k,
        cells_for_main_effects=cells_main,
        n_pairs=n_pairs,
        p_specific_pair=p_pair,
        cells_for_all_pairs=cells_pairs,
        expected_repeats_per_pair=cells_main * p_pair,
    )


def max_panel_for_one_observation(
    k: int, floor: int = FIRST_ORDER_FLOOR, t_hi: int = 1_000_000
) -> int:
    """Largest panel T at which a named pair is still seen once, on average.

    The main-effect budget buys a fixed number of cells; how often a NAMED pair
    turns up in them falls as the panel grows. This back-solves the panel size
    at which that expectation is still 1, which is the threshold separating "the
    joint transcriptome of this pair is in the dataset" from "it is not".

    Bisection on ``recovery(...).expected_repeats_per_pair`` rather than an
    inverted closed form, so the arithmetic has exactly one definition and this
    cannot drift away from the table.
    """
    if k < 2:
        raise ValueError("a pair needs k >= 2")
    lo, hi = 2, t_hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if recovery(mid, k, floor).expected_repeats_per_pair >= 1.0:
            lo = mid
        else:
            hi = mid - 1
    return lo


# --- 3. Environments: the linear axis ----------------------------------------
def environment_scaling(cells_per_condition: float, n_env: list[int]) -> dict:
    """Environments multiply cell demand linearly. That is the entire point."""
    return {str(e): cells_per_condition * e for e in n_env}


class EnvironmentCostPoint(BaseModel):
    """Recurring cost of one multi-environment screen, on each platform."""

    n_env: int
    splitpool_usd: float
    splitpool_runs: int
    splitpool_sublibraries: int
    splitpool_sequencing_usd: float
    droplet_usd: float
    droplet_channels: int
    droplet_sequencing_usd: float
    droplet_preindexed_usd: float
    droplet_preindexed_channels: int


def environment_cost(
    n_env_values: list[int], cells_per_gene: int = 100
) -> list[EnvironmentCostPoint]:
    """Recurring cost against environments, split-pool vs droplet.

    Every dollar figure comes from the Sec. 5 cost model: ``cost_model``'s
    platforms (Brettner's per-run and per-sublibrary rates for split-pool, the
    UIUC per-channel rate for 10x) and ``uiuc_core_data``'s lane pricing. This
    function only composes them along the environment axis, and the composition
    is where the two platforms differ:

    * **Split-pool** carries sample identity in the round-1 plate, so cells from
      every condition are pooled into the same protocol runs and sublibraries.
      The multi-environment design is therefore costed directly with
      ``budget_for`` -- batch and sublibrary counts round up over the POOLED
      cell total, exactly as a real run would.
    * **Preindexed droplet** has a round-1 plate too, of 384 wells, so it pools
      the same way: conditions preindexed into different wells are loaded into
      shared channels and told apart afterwards by the well index. It is costed
      with ``budget_for`` for that reason. The saving is only the rounding --
      one condition already fills 8 channels, so 96 conditions need 690 pooled
      channels against 768 rounded up per condition -- and the slope of the
      curve is set by cells, not by the per-condition floor.
    * **Unmodified droplet** cannot pool: nothing in a channel says which
      condition a cell came from, so channel count is rounded up PER CONDITION
      and summed. The indexed libraries still share sequencing lanes, so the
      sequencing term is priced on the pooled read total.

    ``cells_per_gene=100`` puts the single-condition screen at 600,000 usable
    cells (100 x 6,000 genes), the per-condition figure ``environment_scaling``
    and Sec. 7.3 use. The split-pool platform is SPLiT-seq + rRNA depletion and
    the droplet platform is the 10x Chromium X at UIUC rates, the same pair
    Sec. 5's budget comparison resolves to.
    """
    single = CM.ScreenDesign(cells_per_gene=cells_per_gene, n_environments=1)
    droplet_one = CM.budget_for(single, CM.TENX)
    out = []
    for e in n_env_values:
        design = CM.ScreenDesign(cells_per_gene=cells_per_gene, n_environments=e)
        sp = CM.budget_for(design, CM.SPLITSEQ_DEPLETED)
        # Preindexed droplet is carried as its own series because Sec. 5 promotes
        # it to a co-equal candidate, and showing only the un-preindexed platform
        # would overstate the environment penalty by the factor preindexing
        # removes.
        pi = CM.budget_for(design, CM.TENX_SCIFI_PROJECTED)
        droplet_protocol = e * droplet_one.protocol_usd
        droplet_seq = UC.cost_for_read_pairs(e * droplet_one.read_pairs)
        out.append(
            EnvironmentCostPoint(
                n_env=e,
                splitpool_usd=sp.recurring_usd,
                splitpool_runs=sp.n_batches,
                splitpool_sublibraries=sp.n_sublibraries,
                splitpool_sequencing_usd=sp.sequencing_usd,
                droplet_usd=droplet_protocol + droplet_seq,
                droplet_channels=e * droplet_one.n_batches,
                droplet_sequencing_usd=droplet_seq,
                droplet_preindexed_usd=pi.recurring_usd,
                droplet_preindexed_channels=pi.n_batches,
            )
        )
    return out


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)

    # Delivery, evaluated on the two library sizes the review uses.
    delivery = {}
    for n_lib in (200, 6000):
        pts = delivery_table(n_lib, TARGET_PLASMIDS_PER_CELL)
        delivery[str(n_lib)] = [p.model_dump() for p in pts]
        print(f"\n--- delivery, library of {n_lib} guides ---")
        print(f"{'target m':>9} {'lambda':>8} {'E[m|m>=1]':>10} "
              f"{'E[distinct]':>12} {'P(>=2)':>8}")
        for t, p in zip(TARGET_PLASMIDS_PER_CELL, pts):
            print(f"{t:>9.1f} {p.lam:>8.3f} {p.mean_plasmids:>10.3f} "
                  f"{p.mean_distinct:>12.3f} {p.p_at_least_2:>8.3f}")

    # Recovery, main effects vs named pairs.
    rec = []
    print("\n--- recovery: main effects are cheap, named pairs are not ---")
    print(f"{'T':>6} {'k':>3} {'cells(main)':>13} {'pairs':>12} "
          f"{'P(pair)':>11} {'cells(all pairs)':>18} {'repeats/pair':>13}")
    for T in (200, 6000):
        for k in (1, 2, 3, 5, 8):
            r = recovery(T, k)
            rec.append(r.model_dump())
            print(f"{T:>6} {k:>3} {r.cells_for_main_effects:>13,.0f} "
                  f"{r.n_pairs:>12,} {r.p_specific_pair:>11.2e} "
                  f"{r.cells_for_all_pairs:>18,.0f} "
                  f"{r.expected_repeats_per_pair:>13.2e}")

    ceilings = {str(k): max_panel_for_one_observation(k) for k in (2, 3, 4, 5, 8)}
    print("\n--- largest panel at which a named pair is still seen once ---")
    for k, t in ceilings.items():
        print(f"  k = {k:>2}: T = {t:>6,}")

    envs = environment_scaling(600_000, [1, 4, 12, 96])
    print("\n--- environments (linear), at 600k cells per condition ---")
    for e, c in envs.items():
        print(f"  {e:>3} environments: {c:>12,.0f} cells")

    # Environment cost, every integer 1..96 so the plotted staircase is real.
    env_cost = environment_cost(list(range(1, 97)))
    by_env = {p.n_env: p for p in env_cost}
    print("\n--- environment cost: split-pool vs droplet, Sec. 5 model at "
          "600k usable cells per condition ---")
    print(f"{'envs':>5} {'split-pool $':>13} {'runs':>5} {'sublibs':>8} "
          f"{'droplet $':>12} {'channels':>9} {'preindexed $':>13} {'chan':>6}")
    for e in (1, 4, 12, 24, 48, 96):
        p = by_env[e]
        print(f"{e:>5} {p.splitpool_usd:>13,.0f} {p.splitpool_runs:>5} "
              f"{p.splitpool_sublibraries:>8} {p.droplet_usd:>12,.0f} "
              f"{p.droplet_channels:>9} {p.droplet_preindexed_usd:>13,.0f} "
              f"{p.droplet_preindexed_channels:>6}")
    marginal_sp = (by_env[96].splitpool_usd - by_env[1].splitpool_usd) / 95
    marginal_dr = (by_env[96].droplet_usd - by_env[1].droplet_usd) / 95
    marginal_pi = (
        by_env[96].droplet_preindexed_usd - by_env[1].droplet_preindexed_usd
    ) / 95
    print(f"  marginal cost per added environment: split-pool "
          f"${marginal_sp:,.0f}, droplet ${marginal_dr:,.0f}, "
          f"preindexed droplet ${marginal_pi:,.0f}")
    env_cost_summary = {
        "cells_per_gene": 100,
        "usable_cells_per_condition": CM.ScreenDesign(
            cells_per_gene=100, n_environments=1
        ).usable_cells_needed,
        "splitpool_platform": CM.SPLITSEQ_DEPLETED.name,
        "droplet_platform": CM.TENX.name,
        "droplet_channel_usd": CM.TENX.cost_per_batch_usd,
        "droplet_channels_per_condition": by_env[1].droplet_channels,
        "droplet_preindexed_platform": CM.TENX_SCIFI_PROJECTED.name,
        "droplet_preindexed_channels_per_condition": (
            by_env[1].droplet_preindexed_channels
        ),
        "marginal_usd_per_env_splitpool": marginal_sp,
        "marginal_usd_per_env_droplet": marginal_dr,
        "marginal_usd_per_env_droplet_preindexed": marginal_pi,
    }

    out = {
        "delivery": delivery,
        "recovery": rec,
        "max_panel_for_one_observation": ceilings,
        "environments": envs,
        "environment_costs": [p.model_dump() for p in env_cost],
        "environment_cost_summary": env_cost_summary,
        "constants": {
            "cells_per_pair": CELLS_PER_PAIR,
            "first_order_floor": FIRST_ORDER_FLOOR,
        },
    }
    path = osp.join(RESULTS, "scaling_analysis.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
