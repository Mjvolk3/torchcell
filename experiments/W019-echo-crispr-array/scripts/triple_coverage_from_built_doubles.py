# experiments/W019-echo-crispr-array/scripts/triple_coverage_from_built_doubles.py
# [[experiments.W019-echo-crispr-array.scripts.triple_coverage_from_built_doubles]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/triple_coverage_from_built_doubles
"""SUPERSEDED 2026.08.23 -- kept for history, do not use to plan a round.

This ran on the 10-gene / 31-target basis. YLR104W (LCL2) is BOTH built and a panel-12
prediction node, so the correct basis is 11 genes / 39 targets. The round actually being
built comes from `triple_design_rank_sampling.py`, checked by
`verify_triple_build_list.py`; the bench list is
[[experiments.W019-echo-crispr-array.build-list]]. The outputs still on disk
(`results/triple_coverage_*.csv`, `results/triple_scorability_*.csv`,
`notes/assets/images/.../triple_coverage_from_built_doubles.*`,
`.../triple_scorability_by_double.*`) are on the superseded basis too. The BUILDABLE vs
SCORABLE distinction below is still the right one and still holds.

For each candidate NEW double, how many high-ranking triples it makes SCORABLE.

Two different requirements are easy to conflate, and the panel satisfies one but not the
other:

  BUILDABLE  -- needs >= 1 of the triple's three doubles (cross double AB with single C).
                This is what optimized_doubles_setcover_constructed_10.py covered, and it
                succeeded: 31/31 of the target triples are buildable from the 13 built
                doubles (and from the designed 14 -- the one construction failure,
                YKL033W-A x YJR060W, enabled no triple).
  SCORABLE   -- needs ALL THREE doubles, because

                    tau_ijk = f_ijk - f_i f_j f_k - eps_ij f_k - eps_ik f_j - eps_jk f_i

                subtracts a digenic term for every pair. This was never the cover's
                objective, and 0/31 targets currently satisfy it.

TARGET SET (pinned, do not substitute):
  experiments/010-kuzmin-tmi/results/inference_3/top_k_constructible_panel12_k200.csv
  restricted to the 10 properly-constructed genes -- 31 triples. This is the SAME file and
  the SAME restriction that optimized_doubles_setcover_constructed_10.py used. Two nearby
  files are NOT the target set and must not be used here:
    - results/inference_3/triples_table_panel12_k200.csv  (122 rows: all constructible,
      not the top-k the cover ran on)
    - results/constructible_triples_panel12_*.parquet     (an EARLIER inference over a
      different gene panel; shares 1 gene with the built panel)

Outputs
  results/triple_scorability_by_double.csv   per candidate double (the main table)
  results/triple_scorability_greedy.csv      greedy add-order, cumulative scorable
  notes/assets/images/W019-echo-crispr-array/triple_scorability_by_double.{png,svg}

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/triple_coverage_from_built_doubles.py
"""

from __future__ import annotations

import itertools
import os
import os.path as osp

import matplotlib.pyplot as plt
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
    REPO,
    "experiments/010-kuzmin-tmi/results/inference_3",
    "top_k_constructible_panel12_k200.csv",
)
STRAINS = osp.join(
    EXP_DIR, "data/run4_doubles_2026-08-06/Single-and-Double-KO-Strains-List-Order.csv"
)

# The 10 properly-constructed genes, verbatim from optimized_doubles_setcover_constructed_10.py
TEN = {
    "YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W",
    "YKL033W-A", "YLL012W", "YLR312C-B", "YPL046C", "YPL081W",
}
COMMON = {
    "YPL081W": "RPS9A", "YDR057W": "YOS9", "YPL046C": "ELC1", "YBR203W": "COS111",
    "YGL087C": "MMS2", "YLL012W": "YEH1", "YJR060W": "CBF1", "YLR312C-B": "",
    "YER079W": "", "YKL033W-A": "",
}


def pair_name(p) -> str:
    return " + ".join(sorted(p))


def read_built() -> set[frozenset[str]]:
    sl = pd.read_csv(STRAINS)
    k = sl["#"].astype(str)
    orf = lambda c: str(c).split(" ")[0].strip()  # noqa: E731
    return {
        frozenset((orf(r.KO1), orf(r.KO2)))
        for r, is_d in zip(sl.itertuples(), k.str.match(r"d\d"), strict=True)
        if is_d
    }


def read_targets() -> list[dict]:
    df = pd.read_csv(TARGETS_CSV)
    df = df.sort_values("prediction", ascending=False).reset_index(drop=True)
    out = []
    for i, r in enumerate(df.itertuples(), start=1):
        t = frozenset((r.gene1, r.gene2, r.gene3))
        if not t <= TEN:
            continue
        out.append({"topk_rank": i, "triple": t, "prediction": r.prediction})
    for j, d in enumerate(out, start=1):
        d["target_rank"] = j
    return out


def pairs(t) -> list[frozenset[str]]:
    return [frozenset(p) for p in itertools.combinations(sorted(t), 2)]


def per_double_table(targets, built) -> pd.DataFrame:
    cand = {p for d in targets for p in pairs(d["triple"])} - built
    rows = []
    for c in sorted(cand, key=lambda p: sorted(p)):
        touched = [d for d in targets if c in pairs(d["triple"])]
        closes = [d for d in touched if all(p in built or p == c for p in pairs(d["triple"]))]
        # after adding c, how many targets sit exactly one double short
        after = built | {c}
        one_away = [
            d for d in targets
            if sum(p not in after for p in pairs(d["triple"])) == 1
        ]
        rows.append(
            {
                "double": pair_name(c),
                "targets_touched": len(touched),
                "closes_alone": len(closes),
                "targets_1_away_after": len(one_away),
                "best_target_rank": min((d["target_rank"] for d in touched), default=None),
                "best_prediction": max((d["prediction"] for d in touched), default=None),
                "closed_triples": ";".join(
                    "+".join(sorted(d["triple"])) for d in closes
                ),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["closes_alone", "targets_touched", "best_prediction"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def greedy(targets, built, n_steps=26) -> pd.DataFrame:
    have = set(built)
    rows = [{"step": 0, "double_added": "(13 built)", "newly_scorable": 0,
             "cumulative_scorable": 0, "of_targets": len(targets)}]
    prev = 0
    for step in range(1, n_steps + 1):
        # first choice: a double that CLOSES a target outright (takes it 1-missing -> 0)
        need: dict[frozenset[str], list[dict]] = {}
        for d in targets:
            miss = [p for p in pairs(d["triple"]) if p not in have]
            if len(miss) == 1:
                need.setdefault(miss[0], []).append(d)
        if not need:
            # nothing is one away -- fall back to the double appearing in the most
            # still-unscorable targets, so progress continues on 2-away triples.
            for d in targets:
                miss = [p for p in pairs(d["triple"]) if p not in have]
                for p in miss:
                    need.setdefault(p, []).append(d)
        if not need:
            break
        best = max(need, key=lambda p: (len(need[p]), max(x["prediction"] for x in need[p])))
        have.add(best)
        cum = sum(1 for d in targets if all(p in have for p in pairs(d["triple"])))
        rows.append(
            {"step": step, "double_added": pair_name(best),
             "newly_scorable": cum - prev, "cumulative_scorable": cum,
             "of_targets": len(targets)}
        )
        prev = cum
    return pd.DataFrame(rows)


def plot(tab: pd.DataFrame, path: pd.DataFrame) -> None:
    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 6, "svg.fonttype": "none",
         "axes.linewidth": 0.5}
    )
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(78))
    )
    t = tab.head(20).iloc[::-1]
    y = range(len(t))
    ax.barh(y, t["targets_touched"], color=PLOT_PALETTE[3], edgecolor="black",
            linewidth=0.4, label="targets the double appears in")
    ax.barh(y, t["closes_alone"], color=PLOT_PALETTE[0], edgecolor="black",
            linewidth=0.4, label="made scorable by adding it alone")
    ax.set_yticks(list(y))
    ax.set_yticklabels(t["double"], fontsize=4.5)
    ax.set_xlabel("target triples (of 31)")
    ax.set_title("Per candidate double", fontsize=6)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend(frameon=False, fontsize=5, loc="lower right")

    bx.step(path["step"], path["cumulative_scorable"], where="post",
            color=PLOT_PALETTE[0], lw=1.0)
    bx.scatter(path["step"], path["cumulative_scorable"], s=10,
               c=PLOT_PALETTE[0], edgecolor="black", linewidth=0.3, zorder=4)
    bx.axhline(len(path) and path["of_targets"].iloc[0], ls="--", lw=0.5,
               color=PLOT_PALETTE[5])
    bx.annotate("all 31 targets", (0.5, path["of_targets"].iloc[0]), fontsize=5,
                xytext=(2, 2), textcoords="offset points")
    bx.set_xlabel("new doubles built (greedy order)")
    bx.set_ylabel("target triples with all 3 doubles")
    bx.set_title("Cumulative, greedy order", fontsize=6)
    bx.grid(axis="y", lw=0.3, color="0.85")

    fig.tight_layout()
    fig.savefig(osp.join(IMG_DIR, "triple_scorability_by_double.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "triple_scorability_by_double.svg"))
    plt.close(fig)


def main() -> None:
    os.makedirs(IMG_DIR, exist_ok=True)
    built = read_built()
    targets = read_targets()
    print(f"built doubles: {len(built)}")
    print(f"target triples (top-k, within the 10 constructed genes): {len(targets)}")

    buildable = sum(1 for d in targets if any(p in built for p in pairs(d["triple"])))
    scorable = sum(1 for d in targets if all(p in built for p in pairs(d["triple"])))
    print(f"  BUILDABLE (>=1 double): {buildable}/{len(targets)}")
    print(f"  SCORABLE  (all 3)     : {scorable}/{len(targets)}")
    dist = pd.Series(
        [sum(p in built for p in pairs(d["triple"])) for d in targets]
    ).value_counts().sort_index()
    for k, v in dist.items():
        print(f"    {k} of 3 doubles built: {v} triples")

    tab = per_double_table(targets, built)
    tab.to_csv(osp.join(RESULTS, "triple_scorability_by_double.csv"), index=False)
    print(f"\ncandidate doubles: {len(tab)}")
    print(f"\n{'double':<26}{'in targets':>11}{'closes':>8}{'1-away after':>14}"
          f"{'best rank':>11}{'best pred':>11}")
    for r in tab.itertuples():
        print(f"{r.double:<26}{r.targets_touched:>11}{r.closes_alone:>8}"
              f"{r.targets_1_away_after:>14}{r.best_target_rank:>11}"
              f"{r.best_prediction:>11.4f}")

    path = greedy(targets, built)
    path.to_csv(osp.join(RESULTS, "triple_scorability_greedy.csv"), index=False)
    print(f"\nGREEDY: {'step':>4}  {'double':<26}{'new':>5}{'cumulative':>12}")
    for r in path.itertuples():
        print(f"        {r.step:>4}  {r.double_added:<26}{r.newly_scorable:>5}"
              f"{r.cumulative_scorable:>12}")

    plot(tab, path)
    print(f"\nwrote -> {RESULTS}/triple_scorability_by_double.csv")
    print(f"wrote -> {IMG_DIR}/triple_scorability_by_double.svg")


if __name__ == "__main__":
    main()
