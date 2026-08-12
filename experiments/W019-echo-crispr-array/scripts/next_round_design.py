# experiments/W019-echo-crispr-array/scripts/next_round_design.py
# [[experiments.W019-echo-crispr-array.scripts.next_round_design]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/next_round_design
"""Sampling design for the next assay round (12 singles + ~14 doubles), from run-3 noise.

Answers three questions with measured numbers rather than assumptions:
  1. WHICH noise term limits us -- between-plate or within-plate (colony)?
  2. Can it be reduced by sampling, and by sampling WHAT (more colonies, or more plates)?
  3. What does that imply for the ordered-trend claim (WT < single < double < triple)?

Variance model for one strain's normalised fitness, P plates x c colonies per plate:
    Var(mean) = sigma_plate^2 / P  +  sigma_colony^2 / (P*c)
so colonies only ever divide the SMALLER term -- plates divide both.

For the Jonckheere-Terpstra trend test the groups are strain CLASSES and the observations
within a class are distinct STRAINS, so each observation carries biological strain-to-strain
spread PLUS that strain's measurement error. Colonies within a plate are pseudo-replicates
for this purpose and must not be counted as n.

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/next_round_design.py
"""

from __future__ import annotations

import math
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS_DIR = osp.join(EXP_DIR, "results")
SEED = 7
N_SINGLES, N_DOUBLES = 12, 14


def jt_pvalue(groups: list[np.ndarray]) -> float:
    """One-sided Jonckheere-Terpstra p-value (normal approximation)."""
    k = len(groups)
    n = [len(g) for g in groups]
    total = sum(n)
    j_stat = sum(
        int((np.subtract.outer(groups[b], groups[a]) > 0).sum())
        for a in range(k)
        for b in range(a + 1, k)
    )
    mu = (total**2 - sum(x**2 for x in n)) / 4
    var = (total**2 * (2 * total + 3) - sum(x**2 * (2 * x + 3) for x in n)) / 72
    return 1 - 0.5 * (1 + math.erf((j_stat - mu) / math.sqrt(var) / math.sqrt(2)))


def main() -> None:
    boot = pd.read_csv(osp.join(RESULTS_DIR, "run3_strain_bootstrap.csv"))
    fit = pd.read_csv(osp.join(RESULTS_DIR, "run3_fitness_by_condition.csv"))
    sd_plate = float(boot["across_plate_sd"].mean())
    sd_colony = float(fit["fitness_sd"].mean())
    sd_strain = float(boot["boot_fitness"].std(ddof=1))
    c_now = float(fit["n_used"].mean())
    var_p, var_c = sd_plate**2, sd_colony**2

    print("[1] measured noise (run 3)")
    print(f"    sigma_plate  (between-plate, per strain) = {sd_plate:.3f}")
    print(f"    sigma_colony (within-plate colonies)     = {sd_colony:.3f}")
    print(f"    sigma_strain (biological, across strains)= {sd_strain:.3f}")
    share = 100 * var_p / (var_p + var_c / c_now)
    print(f"    at c={c_now:.0f} colonies/strain/plate the PLATE term is {share:.1f}% of variance")

    print("\n[2] SE of a strain mean by design (rows = plates, cols = colonies/plate)")
    cols = (8, 12, 20, 29)
    print("      " + "".join(f"{('c=' + str(c)):>9}" for c in cols))
    for n_plates in (2, 3, 4, 6, 8, 12):
        row = f"P={n_plates:<4}"
        for c in cols:
            row += f"{math.sqrt(var_p / n_plates + var_c / (n_plates * c)):>9.3f}"
        print(row)
    print("    -> colonies barely move it; SE ~ sigma_plate/sqrt(P). PLATES are the lever.")

    print("\n[3] capacity: 384 wells, 26 strains + WT + blanks + cushion")
    for wt_n, blank_n, cushion in ((24, 6, 40),):
        avail = 384 - wt_n - blank_n - cushion
        c = avail // (N_SINGLES + N_DOUBLES)
        print(f"    384 - {wt_n} WT - {blank_n} blank - {cushion} cushion = {avail}"
              f" -> {c} replicates x {N_SINGLES + N_DOUBLES} strains")

    print("\n[4] JT power for WT < single < double (observation = one STRAIN)")
    rng = np.random.default_rng(SEED)
    n_per_class = (N_SINGLES, N_SINGLES, N_DOUBLES)
    plates = (3, 4, 6, 8)
    print("    gap  " + "".join(f"{('P=' + str(p)):>8}" for p in plates))
    for gap in (0.05, 0.10, 0.15, 0.20):
        row = f"   {gap:>4.2f} "
        for n_plates in plates:
            se = math.sqrt(var_p / n_plates + var_c / (n_plates * 12))
            spread = math.sqrt(sd_strain**2 + se**2)
            hits = sum(
                jt_pvalue([rng.normal(1 + i * gap, spread, n)
                           for i, n in enumerate(n_per_class)]) < 0.05
                for _ in range(2000)
            )
            row += f"{hits / 2000:>7.0%} "
        print(row)

    print("\n[5] per-STRAIN resolution (the harder claim): min detectable difference ~ 2.8*SE")
    for n_plates in plates:
        se = math.sqrt(var_p / n_plates + var_c / (n_plates * 12))
        print(f"    P={n_plates}: SE={se:.3f} -> need a gap of ~{2.8 * se:.2f} between two strains")


if __name__ == "__main__":
    main()
