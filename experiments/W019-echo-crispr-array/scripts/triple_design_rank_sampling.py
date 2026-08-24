# experiments/W019-echo-crispr-array/scripts/triple_design_rank_sampling.py
# [[experiments.W019-echo-crispr-array.scripts.triple_design_rank_sampling]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/triple_design_rank_sampling
"""Where on the ranked constructible triples each design samples, and how evenly it
spreads genes.

GENE BASIS -- 11, not 10. `optimized_doubles_setcover_constructed_10.py` froze a TEN set
= panel-12 minus YIL174W and YLR104W, because the plan at that time was to swap LCL2/
YLR104W out for LCL1/YPL056C. Run 4 did NOT do that: its s7 is YLR104W (LCL2), so
YLR104W is BOTH built and a panel-12 prediction node. The constructible basis is
therefore 11 genes, and the top-k target set is 39 triples, not 31 -- YLR104W adds 8,
one of them at prediction 0.7012. YLR313C (SPH1) is also built but was never a panel
node, so it contributes no predicted triple.

BLOCKED DOUBLE -- YKL033W-A x YJR060W failed to construct and is excluded from every
candidate set here. It blocks 0 of the 39 targets (no top-k triple contains that pair),
so the failure costs nothing for tau, but it must never be proposed.

Kuzmin thresholds for a "high-ranking" trigenic interaction, sourced:
  kuzminExploringWholegenomeDuplicate2020/si/si1.md -- "an established interaction
  magnitude cut-off for digenic interactions (|eps| > 0.08, p < 0.05) and trigenic
  interactions (|tau| > 0.08, p < 0.05)"; Fig. S2 caption gives intermediate
  (|tau| > 0.08) and stringent (|tau| > 0.12).
  kuzminSystematicAnalysisComplex2018/si/si1.md -- intermediate tau < -0.08, very
  stringent tau < -0.2.

Outputs
  results/triple_design_rank_sampling_summary.csv
  results/triple_design_rank_sampling_selection.csv
  results/triple_design_rank_sampling_gene_frequency.csv
  notes/assets/images/W019-echo-crispr-array/triple_design_rank_sampling.{png,svg}
  notes/assets/images/W019-echo-crispr-array/triple_design_gene_frequency.{png,svg}

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/triple_design_rank_sampling.py
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
GENES = [
    "YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W", "YKL033W-A",
    "YLL012W", "YLR104W", "YLR312C-B", "YPL046C", "YPL081W",
]
COMMON = {"YBR203W": "COS111", "YDR057W": "YOS9", "YGL087C": "MMS2", "YJR060W": "CBF1",
          "YLL012W": "YEH1", "YLR104W": "LCL2", "YPL046C": "ELC1", "YPL081W": "RPS9A"}
FLAGGED = "YLR312C-B"
BLOCKED = frozenset(("YKL033W-A", "YJR060W"))   # failed construction, never propose
WANT = 20
# Genes with ZERO Kuzmin trigenic records. The model was trained on TmiKuzmin2018 +
# TmiKuzmin2020 ONLY (see queries/001_small_build.cql -- every other dataset block is
# commented out), so for these two it has no labelled example of any kind and its
# predictions are extrapolation. They carry ranks 1-11 of the 39 targets: the entire top
# of the ranking is flagged, and the first clean target is rank 12.
NO_TRIGENIC_DATA = frozenset(("YER079W", "YLR312C-B"))
CAP_FLAGGED = 6      # max triples in `capped` that may touch a NO_TRIGENIC_DATA gene
CAP_N = 20           # `capped` size. 16 of 39 targets are clean but only 14 of those have
                     # an already-built parent double, so a one-wave selection can draw at
                     # most 14 clean triples. Cap 6 is therefore the SMALLEST cap that
                     # reaches 20 in a single wave (6 flagged + 14 clean); cap 5 tops out
                     # at 19. The cap is set by that construction constraint, not by a
                     # risk budget chosen in advance.
MODES = ("rank", "count", "balanced", "uniform", "capped", "no_ylr")

# Plate arithmetic, every term sourced rather than folded into one magic number.
PLATE_WELLS = 384
ORIENTATION_BLANKS = 6    # generate_picklist.py
WT_WELLS_MIN = 20         # next_round_layout.py:186, "keep >= 20 WT wells"
STRAIN_WELLS = PLATE_WELLS - ORIENTATION_BLANKS - WT_WELLS_MIN


def pairs(t):
    return [frozenset(p) for p in itertools.combinations(sorted(t), 2)]


def read_built():
    sl = pd.read_csv(STRAINS)
    is_d = sl["#"].astype(str).str.match(r"d\d")
    orf = lambda c: str(c).split(" ")[0].strip()  # noqa: E731
    return {frozenset((orf(r.KO1), orf(r.KO2)))
            for r, d in zip(sl.itertuples(), is_d, strict=True) if d}


def read_targets():
    # mergesort (stable) is load-bearing, not stylistic: the 39 in-basis targets contain
    # exactly one tied prediction, 0.42041015625, shared by ranks 34 and 35. Rank 34 is
    # clean and selected; rank 35 touches YLR312C-B and would consume cap budget. Under
    # pandas' default quicksort that tie can resolve either way between runs, so the
    # selection itself is not reproducible. A stable sort pins it to input order.
    df = pd.read_csv(TARGETS_CSV).sort_values(
        "prediction", ascending=False, kind="mergesort")
    out = []
    for r in df.itertuples():
        t = frozenset((r.gene1, r.gene2, r.gene3))
        if t <= set(GENES) and not (BLOCKED <= t):
            out.append({"triple": t, "prediction": r.prediction})
    for i, d in enumerate(out, start=1):
        d["rank"] = i
    return out


def gene_counts(sel):
    c = dict.fromkeys(GENES, 0)
    for d in sel:
        for g in d["triple"]:
            c[g] += 1
    return c


def new_doubles(sel, built):
    return {p for d in sel for p in pairs(d["triple"])} - built


def imbalance(sel):
    c = gene_counts(sel)
    return max(c.values()) - min(c.values())


def uniform_select(targets, built, want):
    idx = {id(d): i for i, d in enumerate(targets)}
    sel, remaining = [], list(targets)
    while len(sel) < want and remaining:
        best, best_key = None, None
        for d in remaining:
            c = gene_counts(sel)
            gain = sum(math.sqrt(c[g] + 1) - math.sqrt(c[g]) for g in d["triple"])
            marg = len(new_doubles(sel + [d], built)) - len(new_doubles(sel, built))
            key = (gain - 0.05 * marg, d["prediction"], -idx[id(d)])
            if best_key is None or key > best_key:
                best, best_key = d, key
        sel.append(best)
        remaining.remove(best)

    def obj(s):
        return (imbalance(s), len(new_doubles(s, built)),
                -sum(x["prediction"] for x in s))

    improved = True
    while improved:
        improved = False
        cur = obj(sel)
        for i in range(len(sel)):
            for u in sorted(remaining, key=lambda d: idx[id(d)]):
                trial = sel[:i] + sel[i + 1:] + [u]
                if obj(trial) < cur:
                    out = sel[i]
                    sel, improved = trial, True
                    remaining.remove(u)
                    remaining.append(out)
                    break
            if improved:
                break
    return sel


def doubles_greedy(targets, built, want, mode):
    pool = ([d for d in targets if FLAGGED not in d["triple"]]
            if mode == "no_ylr" else targets)
    have = set(built)
    cand = sorted(({p for d in pool for p in pairs(d["triple"])} - built) - {BLOCKED},
                  key=lambda p: sorted(p))
    while cand:
        sc = [d for d in pool if all(p in have for p in pairs(d["triple"]))]
        if len(sc) >= want:
            break
        best, best_key = None, None
        for c in cand:
            after = have | {c}
            s2 = [d for d in pool if all(p in after for p in pairs(d["triple"]))]
            key = ((sum(math.sqrt(v) for v in gene_counts(s2).values()), len(s2))
                   if mode == "balanced"
                   else (len(s2), max((x["prediction"] for x in s2), default=0.0)))
            if best_key is None or key > best_key:
                best, best_key = c, key
        have.add(best)
        cand.remove(best)
    return [d for d in targets if all(p in have for p in pairs(d["triple"]))]


def capped_select(targets, built, want=CAP_N, cap=CAP_FLAGGED):
    """Take the `cap` best-predicted triples that touch a NO_TRIGENIC_DATA gene, then fill
    with 'clean' triples chosen to add the FEWEST new doubles (ties by prediction).

    At the committed configuration the fill rule has no effect: exactly 14 clean targets
    have a built parent and the fill needs exactly 14, so cheapest-first and rank-order
    return the same 20 triples at the same 25 new doubles. The rule is kept because it
    binds at any other cap or size, but it is not what produces this round's selection.
    """
    touches = lambda t: bool(t & NO_TRIGENIC_DATA)  # noqa: E731
    # A triple with NO already-built double cannot be made in the same wave as the new
    # doubles -- you must build one of its doubles first, then the triple off it. Exclude
    # those so the whole round stays parallel. Three targets are affected (ranks 26, 32,
    # 37) and all three contain YLR104W, which has zero built doubles because it sat
    # outside the TEN gene set the original set-cover ran on.
    has_parent = lambda t: any(p in built for p in pairs(t))  # noqa: E731
    ranked = [d for d in sorted(targets, key=lambda d: -d["prediction"])
              if has_parent(d["triple"])]
    sel = [d for d in ranked if touches(d["triple"])][:cap]
    clean = [d for d in ranked if not touches(d["triple"])]
    have = {p for d in sel for p in pairs(d["triple"])} | set(built)
    while len(sel) < want and clean:
        best = min(
            clean,
            key=lambda d: (len([p for p in pairs(d["triple"]) if p not in have]),
                           -d["prediction"]),
        )
        have |= set(pairs(best["triple"]))
        sel.append(best)
        clean.remove(best)
    return sel


def select(targets, built, mode):
    if mode == "rank":
        return sorted(targets, key=lambda d: -d["prediction"])[:WANT]
    if mode == "uniform":
        return uniform_select(targets, built, WANT)
    if mode == "capped":
        return capped_select(targets, built)
    return doubles_greedy(targets, built, WANT, mode)


def main() -> None:
    os.makedirs(IMG_DIR, exist_ok=True)
    built = read_built()
    targets = read_targets()
    print(f"gene basis: {len(GENES)}   top-k targets: {len(targets)}   "
          f"built doubles: {len(built)}")
    sels = {m: select(targets, built, m) for m in MODES}

    rows, selrows, freqrows = [], [], []
    for m in MODES:
        s, nd, c = sels[m], new_doubles(sels[m], built), gene_counts(sels[m])
        m_sgl = {g for d in s for g in d["triple"]}
        m_dbl = {p for d in s for p in pairs(d["triple"])}
        measure = len(m_sgl) + len(m_dbl) + len(s)
        rows.append({
            "strategy": m, "triples": len(s), "new_doubles": len(nd),
            "construct_total": len(nd) + len(s), "measure_plus_wt": measure + 1,
            "wells_per_strain": STRAIN_WELLS // measure,
            "tubes_1pick": measure + 1, "tubes_2pick": (measure + 1) * 2,
            # A triple with no already-built parent double cannot be made in the same
            # wave as the new doubles. `no_ylr` is the one strategy that selects such a
            # triple (rank 26), so its build is not a single wave.
            "one_wave": all(any(p in built for p in pairs(d["triple"])) for d in s),
            "zero_data_triples": sum(1 for d in s
                                     if d["triple"] & NO_TRIGENIC_DATA),
            "gene_min": min(c.values()), "gene_max": max(c.values()),
            "imbalance": max(c.values()) - min(c.values()), "ylr_share": c[FLAGGED],
            "median_rank": float(np.median([d["rank"] for d in s])),
            "mean_prediction": float(np.mean([d["prediction"] for d in s])),
            "top10_captured": sum(1 for d in s if d["rank"] <= 10),
        })
        for d in sorted(s, key=lambda x: x["rank"]):
            selrows.append({"strategy": m, "rank": d["rank"],
                            "prediction": d["prediction"],
                            "triple": " + ".join(sorted(d["triple"]))})
        for g in GENES:
            freqrows.append({"strategy": m, "gene": g, "count": c[g]})

    summ = pd.DataFrame(rows)
    summ.to_csv(osp.join(RESULTS, "triple_design_rank_sampling_summary.csv"), index=False)
    pd.DataFrame(selrows).to_csv(
        osp.join(RESULTS, "triple_design_rank_sampling_selection.csv"), index=False)
    freq = pd.DataFrame(freqrows)
    freq.to_csv(osp.join(RESULTS, "triple_design_rank_sampling_gene_frequency.csv"),
                index=False)

    print(f"\n{'strategy':<10}{'tri':>5}{'newdbl':>8}{'CONSTR':>8}{'MEAS':>6}{'wells':>7}"
          f"{'tube@2':>8}{'gene':>9}{'imbal':>7}{'YLR':>5}{'top10':>7}{'medrank':>9}"
          f"{'meanpred':>10}")
    for r in summ.itertuples():
        print(f"{r.strategy:<10}{r.triples:>5}{r.new_doubles:>8}{r.construct_total:>8}"
              f"{r.measure_plus_wt:>6}{r.wells_per_strain:>7}{r.tubes_2pick:>8}"
              f"{f'{r.gene_min}-{r.gene_max}':>9}{r.imbalance:>7}{r.ylr_share:>5}"
              f"{r.top10_captured:>7}{r.median_rank:>9.0f}{r.mean_prediction:>10.4f}")

    # ---- construction feasibility check ----------------------------------------------
    # A triple ABC is BUILT by crossing an existing double with the third single, so it
    # needs >= 1 of its three doubles to already exist. If a selected triple had zero
    # built parents the round would be SERIAL: build a new double, then the triple off
    # it. This checks for that, and for each new double that both its singles exist.
    built_singles = set()
    sl = pd.read_csv(STRAINS)
    _k = sl["#"].astype(str)
    for r, m in zip(sl.itertuples(), _k.str.match(r"s\d"), strict=True):
        if m:
            built_singles.add(str(r.KO1).split(" ")[0].strip())
    chk = []
    for m in MODES:
        need = new_doubles(sels[m], built)
        bad_d = [d for d in need if not set(d) <= built_singles or d == BLOCKED]
        for d in sels[m]:
            t = d["triple"]
            parents = [p for p in pairs(t) if p in built]
            chk.append({
                "strategy": m, "rank": d["rank"], "triple": " + ".join(sorted(t)),
                "built_parents": len(parents),
                "parent_routes": ";".join(
                    f"({' + '.join(sorted(p))}) x {sorted(set(t) - set(p))[0]}"
                    for p in parents),
                "new_doubles_needed": 3 - len(parents),
            })
        serial = [d["rank"] for d in sels[m] if not any(p in built for p in pairs(d["triple"]))]
        print(f"  [{m}] singles missing: {sorted({g for x in sels[m] for g in x['triple']} - built_singles) or 'none'}"
              f" | new doubles not constructible: {bad_d or 'none'}"
              f" | triples with no built parent: {serial or 'none'}")
    pd.DataFrame(chk).to_csv(
        osp.join(RESULTS, "triple_build_construction_check.csv"), index=False)

    # ---- rank-sampling dot plot ------------------------------------------------------
    plt.rcParams.update({"font.family": "Arial", "font.size": 6,
                         "svg.fonttype": "none", "axes.linewidth": 0.5})
    ordered = sorted(targets, key=lambda d: -d["prediction"])
    ranks = [d["rank"] for d in ordered]
    preds = [d["prediction"] for d in ordered]
    fig, (px, ax) = plt.subplots(
        1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 2.6]},
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(150)))

    px.barh(ranks, preds, height=0.75, color=PLOT_PALETTE[5], edgecolor="black",
            linewidth=0.25)
    px.set_xlabel("predicted interaction")
    px.set_ylabel("triple rank (1 = strongest)")
    px.set_xlim(0, max(preds) * 1.05)
    px.grid(axis="x", lw=0.3, color="0.9")

    for i, m in enumerate(MODES):
        chosen = {d["rank"] for d in sels[m]}
        x = np.full(len(ordered), i, dtype=float)
        unsel = [j for j, d in enumerate(ordered) if d["rank"] not in chosen]
        sel = [j for j, d in enumerate(ordered) if d["rank"] in chosen]
        ax.scatter(x[unsel], [ranks[j] for j in unsel], s=22, c="0.86",
                   edgecolor="0.55", linewidth=0.3, zorder=2)
        ax.scatter(x[sel], [ranks[j] for j in sel], s=26, c=PLOT_PALETTE[1],
                   edgecolor="black", linewidth=0.4, zorder=3)
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(
        [f"{m}\n{len(sels[m])} tri\n{len(new_doubles(sels[m], built))} dbl" for m in MODES],
        fontsize=5)
    ax.set_xlim(-0.6, len(MODES) - 0.4)
    ax.grid(axis="y", lw=0.3, color="0.93", zorder=0)
    ax.set_title(f"Which of the {len(targets)} constructible triples each design samples "
                 f"(red = selected)", fontsize=6.5)
    px.invert_yaxis()
    px.yaxis.set_major_locator(MultipleLocator(5))
    fig.tight_layout()
    fig.savefig(osp.join(IMG_DIR, "triple_design_rank_sampling.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "triple_design_rank_sampling.svg"))
    plt.close(fig)

    # ---- gene frequency ---------------------------------------------------------------
    fig, bx = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62)))
    w, xs = 0.16, np.arange(len(GENES))
    for i, m in enumerate(MODES):
        v = [freq[(freq.strategy == m) & (freq.gene == g)]["count"].iloc[0] for g in GENES]
        bx.bar(xs + (i - 2) * w, v, w, color=PLOT_PALETTE[i], edgecolor="black",
               linewidth=0.3, label=m)
    ideal = 3 * WANT / len(GENES)
    bx.axhline(ideal, ls="--", lw=0.6, color="black")
    bx.annotate(f"uniform = {ideal:.1f}", (len(GENES) - 0.4, ideal), fontsize=5,
                ha="right", va="bottom")
    bx.set_xticks(xs)
    bx.set_xticklabels(
        [(f"{g}\n({COMMON[g]})" if g in COMMON else g) + ("*" if g == FLAGGED else "")
         for g in GENES], fontsize=4.5)
    bx.set_ylabel(f"triples containing the gene (of {WANT})")
    bx.set_title("Gene participation per design  (* = merged ORF, swap recommended)",
                 fontsize=6.5)
    bx.grid(axis="y", lw=0.3, color="0.9")
    bx.legend(frameon=False, fontsize=5, ncol=5, loc="upper center")
    fig.tight_layout()
    fig.savefig(osp.join(IMG_DIR, "triple_design_gene_frequency.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "triple_design_gene_frequency.svg"))
    plt.close(fig)
    print(f"\nwrote -> {IMG_DIR}/triple_design_rank_sampling.svg")


if __name__ == "__main__":
    main()
