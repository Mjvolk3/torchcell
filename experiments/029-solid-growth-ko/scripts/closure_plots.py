# experiments/029-solid-growth-ko/scripts/closure_plots.py
# [[experiments.029-solid-growth-ko.scripts.closure_recompute]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/029-solid-growth-ko/scripts/closure_plots

"""Figure and LaTeX tables of the 029 closure recompute (called by closure_recompute.py).

One full-width figure, closure_query_comparison: (a) the identity's Pearson r by label
policy on the 029 build with the 025 build beside it, digenic and trigenic; (b) the same
r per stored screen, policy by screen; (c) stored trigenic score against the score
recomputed under the 025 join replayed (mean of every entry); (d) the same under the
Kuzmin-first policy. Repo figure standards: Arial 6 pt,
boxed axes, palette from torchcell.utils, true-size SVG plus a PNG, no bbox_inches.
"""

import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import MultipleLocator

from torchcell.timestamp import timestamp
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, PLOT_PALETTE_FILL, mm_to_in, savefig_true_size_svg

ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
CMAP = LinearSegmentedColormap.from_list(
    "tc_red", ["#FFFFFF", PLOT_PALETTE_FILL[1], PLOT_PALETTE[1], PLOT_PALETTE[7]]
)
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "svg.fonttype": "none",
        "axes.linewidth": 0.5,
        "hatch.linewidth": 0.5,
    }
)
TS = timestamp()
POLICY_LABEL = {
    "mean_all": "mean of\nevery entry",
    "measured_else_0": "measured,\n0 if none",
    "kuzmin_first": "Kuzmin\nfirst",
    "costanzo_first": "Costanzo\nfirst",
}
SCREEN_COLS = [(2, "C26", "Costanzo 26 C, digenic"), (2, "C30", "Costanzo 30 C, digenic"), (2, "K18", "Kuzmin 2018, digenic"),
               (2, "K20", "Kuzmin 2020, digenic"), (3, "K18", "Kuzmin 2018, trigenic"), (3, "K20", "Kuzmin 2020, trigenic")]


def _box(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)


def _save(fig, img_dir: str, name: str) -> None:
    os.makedirs(img_dir, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(img_dir, f"{name}.svg"))
    fig.savefig(osp.join(img_dir, f"{name}.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(img_dir, f"{name}_{TS}.svg"))
    plt.close(fig)


def _hex(ax, x, y, lim, xl, yl, title, s: dict) -> None:
    m = np.isfinite(x) & np.isfinite(y)
    ax.hexbin(x[m], y[m], gridsize=70, cmap=CMAP, norm=LogNorm(), linewidths=0.1, extent=(*lim, *lim))
    ax.plot(lim, lim, color="black", lw=0.5, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
    ax.text(0.03, 0.97, f"n = {s['n']:,}\nr = {s['pearson']:.3f}\nrho = {s['spearman']:.3f}\nslope = {s['slope']:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6)
    _box(ax)


def _get(pol_stats: pd.DataFrame, policy: str, order: int, src: str = "all") -> dict:
    r = pol_stats[(pol_stats["policy"] == policy) & (pol_stats["order"] == order) & (pol_stats["stored_source"] == src)]
    assert len(r) == 1, (policy, order, src)
    return r.iloc[0].to_dict()


def make_figure(gd: pd.DataFrame, gt: pd.DataFrame, pol_stats: pd.DataFrame, summary: dict, img_dir: str) -> None:
    w = mm_to_in(PANEL_WIDTHS_MM["full"])
    fig, axes = plt.subplots(1, 4, figsize=(w, mm_to_in(66)))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.32, top=0.91, wspace=0.55)

    # (a) r by policy, 025 beside 029
    ax = axes[0]
    b025 = summary["build_025"]
    groups = [("025 build\nas built", b025["digenic_all"]["pearson"], b025["trigenic_all"]["pearson"], True)]
    for pol in ["mean_all", "measured_else_0", "kuzmin_first", "costanzo_first"]:
        groups.append((POLICY_LABEL[pol], _get(pol_stats, pol, 2)["pearson"], _get(pol_stats, pol, 3)["pearson"], False))
    xs = np.arange(len(groups))
    for i, (lab, rd, rt, is025) in enumerate(groups):
        h = "//" if is025 else None
        ax.bar(i - 0.2, rd, 0.38, color=RED, edgecolor="black", lw=0.5, hatch=h, label="digenic" if i == 1 else None)
        ax.bar(i + 0.2, rt, 0.38, color=ORANGE, edgecolor="black", lw=0.5, hatch=h, label="trigenic" if i == 1 else None)
        ax.text(i - 0.2, rd + 0.02, f"{rd:.2f}", ha="center", va="bottom", fontsize=4.5)
        ax.text(i + 0.2, rt + 0.02, f"{rt:.2f}", ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=5, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.set_ylabel("Pearson r, stored score vs identity from fitness")
    ax.set_title("every stored entry, by policy")
    ax.legend(frameon=False, loc="upper left")
    _box(ax)

    # (b) r per stored screen x policy
    ax = axes[1]
    pols = ["mean_all", "measured_else_0", "kuzmin_first", "costanzo_first"]
    mat = np.array([[_get(pol_stats, pol, o, src)["pearson"] for (o, src, _) in SCREEN_COLS] for pol in pols])
    ax.imshow(mat, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if mat[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(SCREEN_COLS)))
    ax.set_xticklabels([c[2] for c in SCREEN_COLS], fontsize=5, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(pols)))
    ax.set_yticklabels([POLICY_LABEL[p] for p in pols], fontsize=5)
    ax.set_title("r per stored screen, by policy")
    ax.tick_params(length=0)
    _box(ax)

    # (c), (d) trigenic hexbins
    _hex(axes[2], gt["value"].to_numpy(), gt["tau_mean_all"].to_numpy(), (-0.8, 0.5),
         "stored tmi (source row)", "tau from build fitness, mean of every entry",
         "trigenic, 029, the 025 join replayed", _get(pol_stats, "mean_all", 3))
    _hex(axes[3], gt["value"].to_numpy(), gt["tau_kuzmin_first"].to_numpy(), (-0.8, 0.5),
         "stored tmi (source row)", "tau from build fitness, Kuzmin first",
         "trigenic, 029, Kuzmin-first policy", _get(pol_stats, "kuzmin_first", 3))
    for ax, letter in zip(axes, "abcd"):
        ax.text(-0.3, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
    _save(fig, img_dir, "closure_query_comparison")


# --------------------------------------------------------------------------- tables
HEAD = (
    "%% SOURCE: experiments/029-solid-growth-ko/scripts/closure_recompute.py "
    "(results/closure_recompute_summary.json) -- GENERATED, do not edit\n"
)
SRC_LABEL = {"all": "every stored entry", "K18": "Kuzmin 2018", "K20": "Kuzmin 2020", "C30": "Costanzo 30 C", "C26": "Costanzo 26 C"}
POLICY_TEXT = {
    "mean_all": "mean of every entry (the 025 join)",
    "measured_else_0": "measured entries; 0 only without a measurement",
    "kuzmin_first": "Kuzmin first",
    "costanzo_first": "Costanzo first",
}


def write_tables(summary: dict, pol_stats: pd.DataFrame, table_dir: str) -> None:
    os.makedirs(table_dir, exist_ok=True)
    b = summary["build_025"]
    lines = [HEAD, r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"build, policy & stored score & $n$ & $r$ & $\rho$ & slope & rmse \\", r"\midrule"]

    def _fmt(name, src, v, best):
        r_txt = f"{v['pearson']:.3f}"
        if best:
            r_txt = r"\textbf{" + r_txt + "}"
        return f"{name} & {src} & {v['n']:,} & {r_txt} & {v['spearman']:.3f} & {v['slope']:.2f} & {v['rmse']:.3f} \\\\"

    for order, word in ((2, "digenic"), (3, "trigenic")):
        lines.append(r"\multicolumn{7}{l}{\emph{" + word + r"}} \\")
        rows = [(f"025, as built", "merged record", b[f"{'digenic' if order == 2 else 'trigenic'}_all"])]
        for pol in POLICY_TEXT:
            rows.append((f"029, {POLICY_TEXT[pol]}", SRC_LABEL["all"], _get(pol_stats, pol, order)))
            if order == 3:
                for src in ("K18", "K20"):
                    rows.append((r"\quad by screen", SRC_LABEL[src], _get(pol_stats, pol, order, src)))
        best_r = max(v["pearson"] for _, _, v in rows)
        for name, src, v in rows:
            lines.append(_fmt(name, src, v, v["pearson"] == best_r))
        if order == 2:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(osp.join(table_dir, "t6-query-comparison.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    s, sv, sg = summary["singles"], summary["pinned_survival"], summary["stored_p_is_source_p"]
    n_by = summary["n_records_by_order"]
    lines = [HEAD, r"\begin{tabular}{lrr}", r"\toprule", r"quantity & 025 build & 029 build \\", r"\midrule",
             f"singles & {b['n_by_order']['1']:,} & {n_by[1]:,} \\\\",
             f"closure doubles & {b['n_by_order']['2']:,} & {n_by[2]:,} \\\\",
             f"triples (gene sets) & {b['n_by_order']['3']:,} & {n_by[3]:,} \\\\",
             f"triples in both builds & \\multicolumn{{2}}{{r}}{{{sv['triples_in_both']:,}}} \\\\",
             f"pinned 010 train triples present & {sv['train']['n_025']:,} & {sv['train']['n_in_029']:,} \\\\",
             f"pinned 010 validation triples present & {sv['val']['n_025']:,} & {sv['val']['n_in_029']:,} \\\\",
             f"pinned 010 test triples present & {sv['test']['n_025']:,} & {sv['test']['n_in_029']:,} \\\\",
             f"singles carrying a converted 0 & {b['singles_with_converted_zero']:,} & {s['n_with_converted_zero']:,} \\\\",
             f"\\quad of which with no measurement & {b['singles_zero_only_no_measurement']:,} & {s['n_zero_only_no_measurement']:,} \\\\",
             f"singles with a Kuzmin query fitness & -- & {s['n_kuzmin_query_fitness']:,} \\\\",
             f"doubles merged across screens ($p$ replaced) & {b['n_merged_doubles']:,} & 0 \\\\",
             f"triples merged across screens ($p$ replaced) & {b['n_merged_triples']:,} & 0 \\\\",
             f"interaction entries carrying the source $p$ & single-screen only & {sg['n_with_p']:,} of {sg['n_interaction_entries']:,} \\\\",
             r"\bottomrule", r"\end{tabular}"]
    with open(osp.join(table_dir, "t7-survival.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
