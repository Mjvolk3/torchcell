# experiments/W019-echo-crispr-array/scripts/next_doubles_selection.py
# [[experiments.W019-echo-crispr-array.scripts.next_doubles_selection]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/next_doubles_selection
"""SUPERSEDED 2026.08.23 -- kept for history, do not use to plan a round.

This ran on the 10-gene / 31-target basis. YLR104W (LCL2) is BOTH built and a panel-12
prediction node, so the correct basis is 11 genes / 39 targets. The round actually being
built comes from `triple_design_rank_sampling.py`, checked by
`verify_triple_build_list.py`; the bench list is
[[experiments.W019-echo-crispr-array.build-list]]. The outputs still on disk
(`results/next_doubles_*.csv`, `notes/assets/images/.../next_doubles_selection.*`) are on
the superseded basis too.

Which doubles to build next, under four competing objectives.

A triple is SCORABLE only when all three of its doubles exist (tau subtracts a digenic
term per pair). None of the 31 target triples is scorable today, so the question is which
new doubles to build -- and that answer depends entirely on what you are optimising.

Strategies compared at equal budget:

  rank      walk the ranked targets top-down, buying whatever doubles each needs.
            Maximises predicted interaction strength. Concentrates hard on YLR312C-B,
            which sits in 8 of the top 10 targets.
  count     greedy on the number of targets made scorable. Ignores rank.
  balanced  greedy on a CONCAVE gene-coverage objective, sum_g sqrt(scorable triples
            containing g). Concavity means the second triple for a gene is worth less
            than the first, so the selection spreads across genes instead of stacking
            one hub. This is the "do not bet everything on one node" objective.
  no_ylr    count-greedy restricted to targets that do NOT contain YLR312C-B -- the SGD
            "ORF, Merged" node the inference-3 notes recommend swapping out.

Target set is pinned to the same file the original set-cover used; see
triple_coverage_from_built_doubles.py for why the two neighbouring files are wrong.

Outputs
  results/next_doubles_strategies.csv       per strategy x budget summary
  results/next_doubles_picks.csv            the ordered picks per strategy
  results/next_doubles_gene_coverage.csv    per-gene scorable counts at the chosen budget
  notes/assets/images/W019-echo-crispr-array/next_doubles_selection.{png,svg}

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/next_doubles_selection.py
"""

from __future__ import annotations

import itertools
import math
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
REPO = osp.dirname(osp.dirname(EXP_DIR))
RESULTS = osp.join(EXP_DIR, "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "W019-echo-crispr-array")

TARGETS_CSV = osp.join(
    REPO, "experiments/010-kuzmin-tmi/results/inference_3",
    "top_k_constructible_panel12_k200.csv",
)
STRAINS = osp.join(
    EXP_DIR, "data/run4_doubles_2026-08-06/Single-and-Double-KO-Strains-List-Order.csv"
)
TEN = {
    "YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W",
    "YKL033W-A", "YLL012W", "YLR312C-B", "YPL046C", "YPL081W",
}
FLAGGED = "YLR312C-B"          # SGD "ORF, Merged"; swap recommended in inference-3 notes
BUDGETS = (3, 6, 9, 12)
SHOW_AT = 9


def pairs(t):
    return [frozenset(p) for p in itertools.combinations(sorted(t), 2)]


def read_built() -> set[frozenset[str]]:
    sl = pd.read_csv(STRAINS)
    is_d = sl["#"].astype(str).str.match(r"d\d")
    orf = lambda c: str(c).split(" ")[0].strip()  # noqa: E731
    return {
        frozenset((orf(r.KO1), orf(r.KO2)))
        for r, d in zip(sl.itertuples(), is_d, strict=True) if d
    }


def read_targets() -> list[dict]:
    df = pd.read_csv(TARGETS_CSV).sort_values("prediction", ascending=False)
    out = []
    for r in df.itertuples():
        t = frozenset((r.gene1, r.gene2, r.gene3))
        if t <= TEN:
            out.append({"triple": t, "prediction": r.prediction})
    for i, d in enumerate(out, start=1):
        d["rank"] = i
    return out


def scorable(targets, have):
    return [d for d in targets if all(p in have for p in pairs(d["triple"]))]


def gene_cov(sc):
    c = {g: 0 for g in TEN}
    for d in sc:
        for g in d["triple"]:
            c[g] += 1
    return c


def select(targets, built, budget, mode):
    """Return the ordered list of doubles chosen under `mode`."""
    pool = [d for d in targets if FLAGGED not in d["triple"]] if mode == "no_ylr" else targets
    have, picks = set(built), []

    if mode == "rank":
        for d in sorted(pool, key=lambda x: -x["prediction"]):
            for p in pairs(d["triple"]):
                if p not in have and len(picks) < budget:
                    have.add(p)
                    picks.append(p)
            if len(picks) >= budget:
                break
        return picks

    # NOTE: iterate candidates in a STABLE sorted order. Iterating the raw set makes
    # tie-breaks depend on PYTHONHASHSEED, so the picks (and every downstream count)
    # differ between runs. Sorting makes the selection reproducible.
    cand = sorted({p for d in pool for p in pairs(d["triple"])} - built,
                  key=lambda p: sorted(p))
    while len(picks) < budget and cand:
        best, best_key = None, None
        for c in cand:
            after = have | {c}
            sc = scorable(pool, after)
            if mode == "balanced":
                cov = gene_cov(sc)
                key = (sum(math.sqrt(v) for v in cov.values()), len(sc))
            else:  # count / no_ylr
                key = (len(sc), max((x["prediction"] for x in scorable(pool, after)), default=0))
            if best_key is None or key > best_key:
                best, best_key = c, key
        have.add(best)
        picks.append(best)
        cand.remove(best)
    return picks


def main() -> None:
    os.makedirs(IMG_DIR, exist_ok=True)
    built = read_built()
    targets = read_targets()
    print(f"targets: {len(targets)}   built doubles: {len(built)}")
    print(f"targets containing {FLAGGED}: "
          f"{sum(1 for d in targets if FLAGGED in d['triple'])}\n")

    rows, pickrows = [], []
    for mode in ("rank", "count", "balanced", "no_ylr"):
        for b in BUDGETS:
            picks = select(targets, built, b, mode)
            have = built | set(picks)
            sc = scorable(targets, have)
            cov = gene_cov(sc)
            nz = [v for v in cov.values() if v > 0]
            rows.append(
                {
                    "strategy": mode, "budget": b, "scorable": len(sc),
                    "genes_covered": len(nz),
                    "max_gene_share": max(cov.values()) if sc else 0,
                    "ylr312cb_share": cov[FLAGGED],
                    "best_pred": max((d["prediction"] for d in sc), default=float("nan")),
                }
            )
            if b == SHOW_AT:
                for i, p in enumerate(picks, 1):
                    pickrows.append({"strategy": mode, "order": i,
                                     "double": " + ".join(sorted(p))})
    summ = pd.DataFrame(rows)
    summ.to_csv(osp.join(RESULTS, "next_doubles_strategies.csv"), index=False)
    pd.DataFrame(pickrows).to_csv(
        osp.join(RESULTS, "next_doubles_picks.csv"), index=False)

    print(f"{'strategy':<10}{'budget':>7}{'scorable':>10}{'genes':>7}"
          f"{'max/gene':>10}{'YLR312C-B':>11}{'best pred':>11}")
    for r in summ.itertuples():
        print(f"{r.strategy:<10}{r.budget:>7}{r.scorable:>10}{r.genes_covered:>7}"
              f"{r.max_gene_share:>10}{r.ylr312cb_share:>11}{r.best_pred:>11.4f}")

    # ---- milestones: doubles needed to reach a target number of scorable triples -----
    # greedy orders are prefix-consistent, so evaluate prefixes of one full-length run.
    print("\nTO REACH A GIVEN NUMBER OF SCORABLE TRIPLES")
    print("  CONSTRUCT = new strains the bench must make (doubles + triples; the 12")
    print("  singles and 13 existing doubles already exist). MEASURE = strains that go")
    print("  on the plate, which is only what tau needs: the singles, doubles and")
    print("  triples appearing in the scorable set, plus WT.\n")
    print(f"{'want':>5}{'strategy':>10}│{'new dbl':>8}{'new tri':>8}{'CONSTRUCT':>11}│"
          f"{'sgl':>5}{'dbl':>5}{'tri':>5}{'MEASURE':>9}│{'wells':>7}"
          f"{'tube@1':>8}{'tube@2':>8}│{'genes':>6}{'YLR':>5}")
    miles = []
    for mode in ("rank", "count", "balanced", "no_ylr"):
        order = select(targets, built, 26, mode)
        for want in (10, 15, 20):
            hit = None
            for n in range(len(order) + 1):
                sc = scorable(targets, built | set(order[:n]))
                if len(sc) >= want:
                    hit = (n, sc)
                    break
            if hit is None:
                print(f"{want:>5}{mode:>10}   unreachable")
                continue
            n, sc = hit
            cov = gene_cov(sc)
            # MEASURE = the minimal set tau needs (drops built doubles no triple uses)
            m_sgl = {g for d in sc for g in d["triple"]}
            m_dbl = {p for d in sc for p in pairs(d["triple"])}
            measure = len(m_sgl) + len(m_dbl) + len(sc)
            construct = n + len(sc)          # new doubles + all triples are new
            wells = (378 - 28) // measure
            row = {
                "target": want, "strategy": mode,
                "new_doubles": n, "new_triples": len(sc), "construct_total": construct,
                "measure_singles": len(m_sgl), "measure_doubles": len(m_dbl),
                "measure_triples": len(sc), "measure_plus_wt": measure + 1,
                "wells_per_strain": wells,
                "tubes_1pick": measure + 1, "tubes_2pick": measure * 2 + 7,
                "genes": sum(1 for v in cov.values() if v > 0), "ylr_share": cov[FLAGGED],
                "picks": ";".join(" + ".join(sorted(p)) for p in order[:n]),
            }
            miles.append(row)
            print(f"{want:>5}{mode:>10}│{n:>8}{len(sc):>8}{construct:>11}│"
                  f"{len(m_sgl):>5}{len(m_dbl):>5}{len(sc):>5}{measure + 1:>9}│"
                  f"{wells:>7}{measure + 1:>8}{measure * 2 + 7:>8}│"
                  f"{row['genes']:>6}{cov[FLAGGED]:>5}")
    pd.DataFrame(miles).to_csv(
        osp.join(RESULTS, "next_doubles_milestones.csv"), index=False)

    print(f"\npicks at budget {SHOW_AT}:")
    pk = pd.DataFrame(pickrows)
    for mode in ("rank", "count", "balanced", "no_ylr"):
        s = pk[pk.strategy == mode]["double"].tolist()
        print(f"  {mode:<9} " + "; ".join(s))

    # per-gene coverage at SHOW_AT
    covrows = []
    for mode in ("rank", "count", "balanced", "no_ylr"):
        picks = select(targets, built, SHOW_AT, mode)
        cov = gene_cov(scorable(targets, built | set(picks)))
        for g, v in cov.items():
            covrows.append({"strategy": mode, "gene": g, "scorable_triples": v})
    covdf = pd.DataFrame(covrows)
    covdf.to_csv(osp.join(RESULTS, "next_doubles_gene_coverage.csv"), index=False)

    # ---- figure ---------------------------------------------------------------------
    plt.rcParams.update({"font.family": "Arial", "font.size": 6,
                         "svg.fonttype": "none", "axes.linewidth": 0.5})
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(74)))
    for i, mode in enumerate(("rank", "count", "balanced", "no_ylr")):
        s = summ[summ.strategy == mode]
        ax.plot(s["budget"], s["scorable"], "-o", ms=3.5, lw=1.0,
                color=PLOT_PALETTE[i], mec="black", mew=0.3, label=mode)
    ax.set_xlabel("new doubles built")
    ax.set_ylabel("target triples scorable")
    ax.set_title("Yield vs budget", fontsize=6)
    ax.xaxis.set_major_locator(MultipleLocator(3))
    ax.grid(axis="y", lw=0.3, color="0.85")
    ax.legend(frameon=False, fontsize=5, loc="upper left")

    genes = sorted(TEN)
    w = 0.2
    x = np.arange(len(genes))
    for i, mode in enumerate(("rank", "count", "balanced", "no_ylr")):
        v = [covdf[(covdf.strategy == mode) & (covdf.gene == g)]["scorable_triples"].iloc[0]
             for g in genes]
        bx.bar(x + (i - 1.5) * w, v, w, color=PLOT_PALETTE[i], edgecolor="black",
               linewidth=0.3, label=mode)
    bx.set_xticks(x)
    bx.set_xticklabels([g.replace("YLR312C-B", "YLR312C-B*") for g in genes],
                       rotation=45, ha="right", fontsize=4.5)
    bx.set_ylabel("scorable triples containing the gene")
    bx.set_title(f"Gene coverage at {SHOW_AT} new doubles  (* = merged ORF)", fontsize=6)
    bx.grid(axis="y", lw=0.3, color="0.85")
    bx.legend(frameon=False, fontsize=5)

    fig.tight_layout()
    fig.savefig(osp.join(IMG_DIR, "next_doubles_selection.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "next_doubles_selection.svg"))
    plt.close(fig)
    print(f"\nwrote -> {RESULTS}/next_doubles_strategies.csv")
    print(f"wrote -> {IMG_DIR}/next_doubles_selection.svg")


if __name__ == "__main__":
    main()
