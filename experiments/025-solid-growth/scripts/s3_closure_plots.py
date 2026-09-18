# experiments/025-solid-growth/scripts/s3_closure_plots.py
# [[experiments.025-solid-growth.scripts.s3_closure_recompute]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/s3_closure_plots

"""Figures and LaTeX tables of the S3 closure recompute (called by s3_closure_recompute.py).

Three figures, each a full-width row of panels at Nature print size:

- s3_closure_strength      within-screen control | digenic recompute | trigenic recompute
- s3_closure_confidence    stored vs propagated p (doubles) | (triples) | merged p vs source p | calls
- s3_closure_hazards       essentiality-tainted singles | duplicate screens per double | p of merged vs single

Repo figure standards: Arial 6 pt, boxed axes, palette from torchcell.utils, true-size SVG
plus a PNG fallback, no bbox_inches="tight".
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
# red is the primary series color of this document: the hexbin ramp runs white ->
# red fill -> red -> dark red (palette slots 2, 2 fill, 8).
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
    }
)
TS = timestamp()


def _box(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)


def _save(fig, img_dir: str, name: str) -> None:
    os.makedirs(img_dir, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(img_dir, f"{name}.svg"))
    fig.savefig(osp.join(img_dir, f"{name}.png"), dpi=300)
    # timestamped copies for the iteration history
    savefig_true_size_svg(fig, osp.join(img_dir, f"{name}_{TS}.svg"))
    plt.close(fig)


def _hex(ax, x, y, lim, xl, yl, title, stats_txt, gridsize=70):
    m = np.isfinite(x) & np.isfinite(y)
    ax.hexbin(x[m], y[m], gridsize=gridsize, cmap=CMAP, norm=LogNorm(), linewidths=0.1, extent=(*lim, *lim))
    ax.plot(lim, lim, color="black", lw=0.5, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
    ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, va="top", ha="left", fontsize=6)
    _box(ax)


def _fmt(s: dict) -> str:
    return f"n = {s['n']:,}\nr = {s['pearson']:.3f}\nrho = {s['spearman']:.3f}\nslope = {s['slope']:.2f}"


def make_figures(singles: pd.DataFrame, doubles: pd.DataFrame, triples: pd.DataFrame, dm: pd.DataFrame,
                 tmerged: pd.DataFrame, rawd: pd.DataFrame, summary: dict, img_dir: str) -> None:
    w = mm_to_in(PANEL_WIDTHS_MM["full"])

    # ------------------------------------------------------------- strength
    fig, axes = plt.subplots(1, 3, figsize=(w, mm_to_in(62)))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.9, wspace=0.42)
    ctrl = summary["within_screen_control"]
    src = "costanzo2016" if "costanzo2016" in ctrl else next(iter(ctrl))
    r = rawd[rawd["source"] == src]
    _hex(axes[0], r["eps"].to_numpy(), r["eps_rec_raw"].to_numpy(), (-0.8, 0.5),
         "reported eps (source row)", "f_ab - f_a f_b (same row)",
         f"within one screen ({src})", _fmt(ctrl[src]))
    d = doubles[~doubles["merged"]] if (~doubles["merged"]).sum() > 1000 else doubles
    s = summary["digenic_strength"]["all"]
    _hex(axes[1], doubles["gi"].to_numpy(), doubles["eps_rec"].to_numpy(), (-0.8, 0.5),
         "stored dmi (build)", "f_ab - f_a f_b (build)", "digenic, S3 closure doubles", _fmt(s))
    s = summary["trigenic_strength"]["all"]
    _hex(axes[2], triples["gi"].to_numpy(), triples["tau_rec"].to_numpy(), (-0.8, 0.5),
         "stored tmi (build)", "tau from build fitness", "trigenic, S3 triples", _fmt(s))
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.22, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
    _save(fig, img_dir, "s3_closure_strength")

    # ------------------------------------------------------------- confidence
    fig, axes = plt.subplots(1, 4, figsize=(w, mm_to_in(52)))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.88, wspace=0.5)
    lim = (1e-6, 1.0)
    single = doubles[~doubles["merged"]]
    c = summary["digenic_confidence"]

    def _logp(ax, x, y, xl, yl, title, rho, n):
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x, y = np.clip(x[m], *lim), np.clip(y[m], *lim)
        ax.hexbin(x, y, gridsize=55, cmap=CMAP, norm=LogNorm(), xscale="log", yscale="log",
                  linewidths=0.1, extent=(np.log10(lim[0]), 0, np.log10(lim[0]), 0))
        ax.plot(lim, lim, color="black", lw=0.5, ls="--")
        ax.axvline(0.05, color=BLUE, lw=0.5)
        ax.axhline(0.05, color=BLUE, lw=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.text(0.03, 0.97, f"n = {n:,}\nrho = {rho:.3f}", transform=ax.transAxes, va="top", fontsize=6)
        _box(ax)

    _logp(axes[0], single["gi_p"].to_numpy(), single["p_rec_obs"].to_numpy(),
          "stored p (= source p)", r"z-test p, double's own sd / $\sqrt{4}$", "doubles, one screen",
          c["spearman_single_screen_stored_vs_obs"], c["n_single_screen"])
    tsingle = triples[~triples["merged"]]
    ct = summary["trigenic_confidence"]
    _logp(axes[1], tsingle["gi_p"].to_numpy(), tsingle["p_rec_obs"].to_numpy(),
          "stored p (= source p)", r"z-test p, triple's own sd / $\sqrt{4}$", "triples, one screen",
          ct["spearman_p_stored_vs_obs_only"], ct["n"])
    merged = dm[dm["merged"] & np.isfinite(dm["p_source_median"])]
    cm = summary["merged_p_against_sources"]
    _logp(axes[2], merged["p_source_median"].to_numpy(), merged["gi_p"].to_numpy(),
          "median source p", "stored p (t-test over screens)", "doubles, merged screens",
          cm["spearman_stored_vs_source_median_p"], cm["n_merged_with_source_rows"])
    # calls
    ax = axes[3]
    rows = [
        ("doubles\none screen", c["calls_single_screen_stored_vs_obs_on_recomputed_eps"]),
        ("doubles\nmerged", cm["calls_stored_vs_source_median"]),
        ("triples\none screen", ct["calls_stored_vs_propagated"]),
    ]
    xs = np.arange(len(rows))
    stored = [r[1]["stored_called"] / r[1]["n"] for r in rows]
    recomp = [r[1]["recomputed_called"] / r[1]["n"] for r in rows]
    both = [r[1]["both"] / r[1]["n"] for r in rows]
    ax.bar(xs - 0.27, stored, 0.25, color=RED, edgecolor="black", lw=0.5, label="stored")
    ax.bar(xs, recomp, 0.25, color=ORANGE, edgecolor="black", lw=0.5, label="recomputed / source")
    ax.bar(xs + 0.27, both, 0.25, color=PURPLE, edgecolor="black", lw=0.5, label="both")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("fraction called (|score| > 0.08, p < 0.05)")
    ax.set_title("interaction calls")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 3.0)
    ax.legend(frameon=False, loc="upper center", fontsize=5, ncol=1, bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", which="major", lw=0.3, color="#DDDDDD")
    ax.set_axisbelow(True)
    _box(ax)
    for ax, letter in zip(axes, "abcd"):
        ax.text(-0.3, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
    _save(fig, img_dir, "s3_closure_confidence")

    # ------------------------------------------------------------- hazards
    fig, axes = plt.subplots(1, 3, figsize=(w, mm_to_in(55)))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.88, wspace=0.42)
    ax = axes[0]
    ok = singles[~singles["has_ess"] & ~singles["has_sl"]]
    bad = singles[singles["has_ess"] & ~singles["ess_only"]]
    only = singles[singles["ess_only"]]
    ax.scatter(ok["fit_measured"], ok["fit_mean"], s=2, color=GRAY, lw=0, label=f"measured only (n = {len(ok):,})")
    ax.scatter(bad["fit_measured"], bad["fit_mean"], s=3, color=RED, lw=0,
               label=f"measured + essentiality 0 (n = {len(bad):,})")
    ax.scatter(np.full(len(only), 1.22), only["fit_mean"], s=3, color=PURPLE, lw=0,
               label=f"essentiality only, no measurement (n = {len(only):,})")
    ax.plot([0, 1.3], [0, 1.3], color="black", lw=0.5, ls="--")
    ax.set_xlim(0, 1.3)
    ax.set_ylim(-0.02, 1.6)
    ax.set_xlabel("measured single-mutant fitness (0 taken back out)")
    ax.set_ylabel("stored fitness of the single")
    ax.set_title("singles: an essentiality 0 in the mean")
    ax.legend(frameon=False, loc="upper left", fontsize=5)
    _box(ax)
    ax = axes[1]
    dup = doubles["gi_dup"].clip(upper=8)
    counts = dup.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=RED, edgecolor="black", lw=0.5)
    ax.set_xticks(range(1, 9))
    ax.set_xticklabels([str(i) for i in range(1, 8)] + ["8+"])
    ax.set_xlabel("screens merged into one closure double")
    ax.set_ylabel("doubles")
    ax.set_title("duplicate screens per double")
    _box(ax)
    ax = axes[2]
    bins = np.linspace(-6, 0, 49)

    def _lp(x):
        return np.log10(np.clip(x.dropna().to_numpy(), 1e-6, 1))

    ax.hist(_lp(doubles.loc[~doubles["merged"], "gi_p"]), bins=bins, density=True,
            histtype="step", color=ORANGE, lw=0.8, label="one screen: source p")
    ax.hist(_lp(doubles.loc[doubles["merged"], "gi_p"]), bins=bins, density=True,
            histtype="step", color=RED, lw=0.8, label="merged: t-test p")
    if len(merged):
        ax.hist(_lp(merged["p_source_median"]), bins=bins, density=True,
                histtype="step", color=PURPLE, lw=0.8, label="merged: median source p")
    ax.axvline(np.log10(0.05), color=BLUE, lw=0.5)
    ax.set_xlabel("log10 p-value (clipped at 1e-6)")
    ax.set_ylabel("density")
    ax.set_title("p-values of closure doubles, by origin")
    ax.legend(frameon=False, loc="upper left")
    _box(ax)
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.22, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
    _save(fig, img_dir, "s3_closure_hazards")


# --------------------------------------------------------------------------- tables
def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def write_tables(summary: dict, comp: pd.DataFrame, table_dir: str) -> None:
    os.makedirs(table_dir, exist_ok=True)
    head = (
        "%% SOURCE: experiments/025-solid-growth/scripts/s3_closure_recompute.py "
        "(results/s3_closure_recompute_summary.json) -- GENERATED, do not edit\n"
    )

    # t1: composition of the pool by order and source combination
    lines = [head, r"\begin{tabular}{llr}", r"\toprule", r"order & fitness sources joined & records \\", r"\midrule"]
    for order, g in comp.groupby("order"):
        first = True
        for _, row in g.iterrows():
            src = _tex_escape(row["fit_src"].replace("Dataset", "").replace("+", " + "))
            lines.append(f"{order if first else ''} & {src} & {int(row['n']):,} \\\\")
            first = False
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(osp.join(table_dir, "t1-composition.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # t2: strength by stratum
    rows = [
        ("within one screen, Costanzo 2016 (raw row)", summary["within_screen_control"].get("costanzo2016")),
        ("within one screen, Kuzmin 2018 (raw row)", summary["within_screen_control"].get("kuzmin2018")),
        ("within one screen, Kuzmin 2020 (raw row)", summary["within_screen_control"].get("kuzmin2020")),
        ("digenic, all closure doubles", summary["digenic_strength"]["all"]),
        ("digenic, one screen only", summary["digenic_strength"]["single_screen"]),
        ("digenic, merged screens", summary["digenic_strength"]["merged_screens"]),
        ("digenic, no essential single", summary["digenic_strength"]["no_essential_single"]),
        ("digenic, an essential single (stored mean)", summary["digenic_strength"]["essential_single"]),
        ("digenic, an essential single (measured mean restored)", summary["digenic_strength"]["essential_single_measured_mean_restored"]),
        ("digenic, an essential-only single (fitness 0)", summary["digenic_strength"]["essential_only_single"]),
        ("trigenic, all triples", summary["trigenic_strength"]["all"]),
        ("trigenic, one screen only", summary["trigenic_strength"]["single_screen"]),
        ("trigenic, merged screens", summary["trigenic_strength"]["merged_screens"]),
        ("trigenic, no essential single", summary["trigenic_strength"]["no_essential_single"]),
        ("trigenic, an essential single (stored mean)", summary["trigenic_strength"]["essential_single"]),
        ("trigenic, an essential single (measured mean restored)", summary["trigenic_strength"]["essential_single_measured_mean_restored"]),
        ("trigenic, an essential-only single (fitness 0)", summary["trigenic_strength"]["essential_only_single"]),
        ("trigenic, from stored dmi", summary["trigenic_strength"]["from_stored_dmi"]),
    ]
    lines = [head, r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"stratum & $n$ & $r$ & $\rho$ & slope & rmse \\", r"\midrule"]
    best_r = max(v["pearson"] for _, v in rows if v and "pearson" in v)
    for name, v in rows:
        if not v or "pearson" not in v:
            continue
        r_txt = f"{v['pearson']:.3f}"
        if v["pearson"] == best_r:
            r_txt = r"\textbf{" + r_txt + "}"
        lines.append(f"{name} & {v['n']:,} & {r_txt} & {v['spearman']:.3f} & {v['slope']:.2f} & {v['rmse']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(osp.join(table_dir, "t2-strength.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # t3: confidence
    c, ct, cm, cmt = (summary["digenic_confidence"], summary["trigenic_confidence"],
                      summary["merged_p_against_sources"], summary["merged_triples_against_sources"])
    lines = [head, r"\begin{tabular}{lrrrr}", r"\toprule",
             r"comparison & $n$ & $\rho$ of $p$ & called stored & called other \\", r"\midrule"]

    def _row(name, n, rho, calls):
        lines.append(
            f"{name} & {n:,} & {rho:.3f} & {calls['stored_called'] / calls['n']:.3f} & "
            f"{calls['recomputed_called'] / calls['n']:.3f} \\\\"
        )

    _row("doubles, one screen: source $p$ vs $z$ from the double's sd",
         c["calls_single_screen_stored_vs_obs"]["n"], c["spearman_single_screen_stored_vs_obs"],
         c["calls_single_screen_stored_vs_obs"])
    _row("doubles, one screen: source $p$ vs $z$ with the singles' sd propagated",
         c["calls_single_screen_stored_vs_propagated"]["n"], c["spearman_single_screen_stored_vs_propagated"],
         c["calls_single_screen_stored_vs_propagated"])
    _row("doubles, one screen: source call vs call on recomputed $\\varepsilon$",
         c["calls_single_screen_stored_vs_obs_on_recomputed_eps"]["n"], c["spearman_single_screen_stored_vs_obs"],
         c["calls_single_screen_stored_vs_obs_on_recomputed_eps"])
    _row("doubles, merged: stored ($t$-test) vs $z$ from the double's sd",
         c["calls_merged_stored_vs_obs"]["n"], c["spearman_merged_stored_vs_obs"],
         c["calls_merged_stored_vs_obs"])
    _row("doubles, merged: stored ($t$-test) vs median source $p$",
         cm["calls_stored_vs_source_median"]["n"], cm["spearman_stored_vs_source_median_p"],
         cm["calls_stored_vs_source_median"])
    _row("triples, one screen: source $p$ vs $z$ from the triple's sd",
         ct["calls_stored_vs_obs_only_on_stored_tau"]["n"], ct["spearman_p_stored_vs_obs_only"],
         ct["calls_stored_vs_obs_only_on_stored_tau"])
    _row("triples, one screen: source $p$ vs $z$ with every term propagated",
         ct["calls_stored_vs_propagated"]["n"], ct["spearman_p_stored_vs_propagated"],
         ct["calls_stored_vs_propagated"])
    lines.append(
        f"triples, merged: stored ($t$-test) vs source $p$ & {cmt['n']:,} & -- & "
        f"{cmt['frac_stored_p_below_0.05']:.3f} & {cmt['frac_source_median_p_below_0.05']:.3f} \\\\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(osp.join(table_dir, "t3-confidence.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # t4: hazards, counts
    fc, e = summary["fitness_conflicts"], summary["essentiality_in_singles"]
    lines = [head, r"\begin{tabular}{lr}", r"\toprule", r"hazard & records \\", r"\midrule",
             f"singles carrying an SGD essentiality entry (fitness 0 in the mean) & {fc['n_singles_with_essentiality_entry']:,} \\\\",
             f"\\quad of which essentiality only, no measurement (stored fitness 0) & {e['n_essential_only_no_measurement']:,} \\\\",
             f"singles carrying a SynthLethDB entry (fitness 0) & {fc['n_singles_with_synthleth_entry']:,} \\\\",
             f"records with two fitness values (masked by the trainer) & {fc['n_records_two_fitness_entries']:,} \\\\",
             f"closure doubles touching an essentiality-tainted single & {e['n_doubles_touching']:,} \\\\",
             f"\\quad of which touching an essentiality-only single & {e['n_doubles_touching_essential_only']:,} \\\\",
             f"triples touching an essentiality-tainted single & {e['n_triples_touching']:,} \\\\",
             f"\\quad of which touching an essentiality-only single & {e['n_triples_touching_essential_only']:,} \\\\",
             f"closure doubles merged from several screens ($p$ replaced by a $t$-test) & {fc['n_doubles_merged_interaction']:,} \\\\",
             f"triples merged from several screens ($p$ replaced by a $t$-test) & {fc['n_triples_merged_interaction']:,} \\\\",
             f"merged doubles significant in every source, not significant in the build & {cm['n_all_sources_significant_but_stored_not']:,} \\\\",
             f"merged triples significant in every source, not significant in the build & {cmt['n_all_sources_significant_but_stored_not']:,} \\\\",
             r"\bottomrule", r"\end{tabular}"]
    with open(osp.join(table_dir, "t4-hazards.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
