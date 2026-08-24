# experiments/019-perturb-seq-costing/scripts/scaling_analysis.py
# [[experiments.019-perturb-seq-costing.scripts.scaling_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/scaling_analysis
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

Nothing here is a new statistical result: (2) reuses Eq. 3 of the review, which
is Yao et al.'s expression. What is new is applying it to the delivery routes
that are actually available in yeast, and being explicit that the copy-number
route buys *combinations* without buying *transformants*.

Run:  python experiments/019-perturb-seq-costing/scripts/scaling_analysis.py
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "019-perturb-seq-costing", "results"
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


# --- 3. Environments: the linear axis ----------------------------------------
def environment_scaling(cells_per_condition: float, n_env: list[int]) -> dict:
    """Environments multiply cell demand linearly. That is the entire point."""
    return {str(e): cells_per_condition * e for e in n_env}


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

    envs = environment_scaling(600_000, [1, 4, 12, 96])
    print("\n--- environments (linear), at 600k cells per condition ---")
    for e, c in envs.items():
        print(f"  {e:>3} environments: {c:>12,.0f} cells")

    out = {
        "delivery": delivery,
        "recovery": rec,
        "environments": envs,
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
