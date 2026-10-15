# experiments/030-solid-growth-multi/scripts/figure2_gate_numbers.py
# [[experiments.030-solid-growth-multi.scripts.figure2_gate_numbers]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/figure2_gate_numbers
"""Every number of notes-tex/interaction-fitness-figure-gate, from the result files.

The gate document for manuscript Figure 2 (fitness and gene interactions) prints nothing
typed by hand: each value in its prose and captions is a macro written here, each table
row is written here, and the one panel drawn here (the fitness side) reads the same
files. Sources, all committed result files:

    experiments/030-solid-growth-multi/results/lower_order_baselines_030.csv
        the lower-order ladder on both arms (lower_order_baselines_030.py, job 3902)
    experiments/030-solid-growth-multi/results/lower_order_baselines_030_paired.csv
    experiments/030-solid-growth-multi/results/lower_order_baselines_030_summary.json
        with the transformer's W&B validation rows
    experiments/025-solid-growth/results/additive_baselines_025_summary.json
        the gene-identity ladder B0 to B5 on both arms, the 010 checkpoints, job 1598
    experiments/010-kuzmin-tmi/results/additive_baseline_gene_interaction.csv
        the three 010 checkpoints' test scores
    experiments/006-kuzmin-tmi/results/dcell_training/checkpoints.csv
        DCell's three checkpoints on the 006 build (another build; comparator only)
    experiments/005-kuzmin2018-tmi/results/dango_string_version_summary.csv
        DANGO on the 005 Kuzmin 2018 build (another build; comparator only)
    experiments/007-kuzmin-tm/results/fba_baseline_si/stats.json
        Yeast9 FBA against the released trigenic score (another build; comparator only)

W&B (``--wandb``): the 030 arms' validation fitness Pearson on the held-out triples
(``val/fitness/order3/Pearson``), mean of the last ten synced epochs, the same window
the interaction row uses, one reference per seed (``wandb_seed_histories.py``; seeds are never
stitched together). Seed 0 writes the
unsuffixed macros; every seed writes ``...Seed<Word>...`` macros; a seed with fewer
epochs than the window is on file as partial and drawn nowhere.

Writes:
    notes-tex/interaction-fitness-figure-gate/tables/gate_numbers.tex      \\newcommand macros
    notes-tex/interaction-fitness-figure-gate/tables/gate_ladder_rows.tex  the performance table
    notes-tex/interaction-fitness-figure-gate/tables/gate_comparator_rows.tex
    $ASSET_IMAGES_DIR/030-solid-growth-multi/figure2_gate_fitness_<ts>.{svg,png}

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/figure2_gate_numbers.py --wandb
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from pydantic import BaseModel

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    apply_paper_style,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
REPO = osp.dirname(EXPERIMENT_ROOT)
R030 = osp.join(EXPERIMENT_ROOT, "030-solid-growth-multi", "results")
R025 = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
R010 = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
R006 = osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi", "results", "dcell_training")
R005 = osp.join(EXPERIMENT_ROOT, "005-kuzmin2018-tmi", "results")
R007 = osp.join(EXPERIMENT_ROOT, "007-kuzmin-tm", "results", "fba_baseline_si")
TABLES = osp.join(REPO, "notes-tex", "interaction-fitness-figure-gate", "tables")
IMAGES = osp.join(ASSET_IMAGES_DIR, "030-solid-growth-multi")

from wandb_seed_histories import (  # noqa: E402  (sibling module, scripts dir)
    WANDB_ARM_TAG,
    WANDB_HEADLINE_SEED,
    WANDB_WINDOW,
    wandb_seed_histories,
)

SEED_WORD = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four"}

DIGIT_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

#: rung -> (label, regime). The regime names what a rung reads at scoring time: identity
#: (the triple's genes and nothing measured), or the measured lower-order terms of the
#: same strains (a lookup, transductive over the closure).
RUNGS = {
    "N0_screen_mean": ("N0 screen mean", "screen only"),
    "N1_mult_query_double": ("N1 digenic controls, zero parameters", "lower-order terms"),
    "N2_mult_singles": ("N2 singles product, zero parameters", "lower-order terms"),
    "N1s_mult_strain_matched": ("N1s strain-matched controls", "lower-order terms"),
    "B1_additive_gene": ("B1 additive gene ridge", "identity"),
    "L1_ridge_lower_order": ("L1 ridge, lower-order terms", "lower-order terms"),
    "L2_hgb_lower_order": ("L2 boosted trees, lower-order terms", "lower-order terms"),
    "L3_ridge_gene_plus_lower_order": ("L3 ridge, genes + lower-order terms", "both"),
    "F1_ridge_lower_order_fitness": ("F1 ridge, lower-order terms, target f_ijk", "lower-order terms"),
}
B_RUNGS = {
    "B0_train_mean": "B0 train mean",
    "B3_hierarchical_mean": "B3 screen mean",
    "B4_query_pair_only": "B4 query pair only",
    "B1_additive_gene": "B1 additive gene ridge",
    "B2_additive_plus_pair": "B2 additive plus pair",
    "B5_gene_embedding_mlp": "B5 gene-embedding MLP, three seeds",
}


class FitnessRef(BaseModel):
    """One seed of one arm's fitness head, stitched over its rank-0 runs by epoch."""

    arm: str
    seed: int
    run_ids: list[str]
    n_epochs_logged: int
    last_epoch: int
    val_fitness_order3_last_window_mean: float
    window: int
    partial: bool


def macro_name(key: str) -> str:
    """A LaTeX macro name: letters only, digits spelled out."""
    return "".join(DIGIT_WORDS[int(ch)] if ch.isdigit() else ch for ch in key if ch.isalnum())


def fmt(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if abs(v) >= 1 and float(v).is_integer():
            return f"{int(v):,}"
        if abs(v) >= 100:
            return f"{v:,.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


def signed(v: float) -> str:
    return f"{v:+.3f}"


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


# ---------------------------------------------------------------------------- readers
def read_ladder() -> pd.DataFrame:
    return pd.read_csv(osp.join(R030, "lower_order_baselines_030.csv"))


def read_paired() -> pd.DataFrame:
    return pd.read_csv(osp.join(R030, "lower_order_baselines_030_paired.csv"))


def read_summary_030() -> dict[str, Any]:
    with open(osp.join(R030, "lower_order_baselines_030_summary.json")) as f:
        return json.load(f)


def read_summary_025() -> dict[str, Any]:
    with open(osp.join(R025, "additive_baselines_025_summary.json")) as f:
        return json.load(f)


def read_010() -> pd.DataFrame:
    df = pd.read_csv(osp.join(R010, "additive_baseline_gene_interaction.csv"))
    return df[df["model"].str.startswith("CGT")]


def read_dcell() -> pd.DataFrame:
    return pd.read_csv(osp.join(R006, "checkpoints.csv"))


def read_dango() -> pd.DataFrame:
    return pd.read_csv(osp.join(R005, "dango_string_version_summary.csv"))


def read_fba() -> dict[str, Any]:
    with open(osp.join(R007, "stats.json")) as f:
        return json.load(f)["correlations"]["tau"]


def wandb_fitness(arm: str) -> list[FitnessRef]:
    """The arm's fitness head on the held-out triples, one reference per seed."""
    key = "val/fitness/order3/Pearson"
    hists = wandb_seed_histories(WANDB_ARM_TAG[arm], key)
    assert WANDB_HEADLINE_SEED in hists, f"{arm}: headline seed {WANDB_HEADLINE_SEED} absent"
    return [
        FitnessRef(
            arm=arm,
            seed=seed,
            run_ids=hist.attrs["run_ids"],
            n_epochs_logged=int(len(hist)),
            last_epoch=int(hist.index[-1]),
            val_fitness_order3_last_window_mean=float(hist.iloc[-WANDB_WINDOW:].mean()),
            window=WANDB_WINDOW,
            partial=bool(len(hist) < WANDB_WINDOW),
        )
        for seed, hist in hists.items()
    ]


# ------------------------------------------------------------------------------ macros
def macros(
    ladder: pd.DataFrame,
    paired: pd.DataFrame,
    s030: dict[str, Any],
    s025: dict[str, Any],
    cgt010: pd.DataFrame,
    dcell: pd.DataFrame,
    dango: pd.DataFrame,
    fba: dict[str, Any],
    fitness: list[FitnessRef],
) -> dict[str, str]:
    m: dict[str, str] = {}
    for arm in ("R", "Q"):
        sub = ladder[ladder["arm"] == arm].set_index(["rung", "split"])
        for rung in RUNGS:
            for split in ("val", "test"):
                row = sub.loc[(rung, split)]
                m[f"lo{arm}{rung}{split}"] = fmt(float(row["pearson"]))
                m[f"lo{arm}{rung}{split}Within"] = fmt(float(row["pearson_within_screen"]))
                m[f"lo{arm}{rung}{split}Spearman"] = fmt(float(row["spearman"]))
                m[f"lo{arm}{rung}{split}Kz2018"] = fmt(float(row["pearson_kuzmin2018"]))
                m[f"lo{arm}{rung}{split}Kz2020"] = fmt(float(row["pearson_kuzmin2020"]))
                m[f"lo{arm}{rung}{split}Coverage"] = fmt(float(row["coverage"]))
            m[f"lo{arm}{rung}N"] = fmt(int(sub.loc[(rung, "test")]["n"]))
        m[f"n{arm}Train"] = fmt(int(sub.loc[("N0_screen_mean", "train")]["n"]))
        m[f"n{arm}Val"] = fmt(int(sub.loc[("N0_screen_mean", "val")]["n"]))
        m[f"n{arm}Test"] = fmt(int(sub.loc[("N0_screen_mean", "test")]["n"]))
        m["nTriples"] = fmt(sum(int(sub.loc[("N0_screen_mean", s)]["n"]) for s in ("train", "val", "test")))
        for _, p in paired[paired["arm"] == arm].iterrows():
            key = f"pair{arm}{p['a']}{p['b']}"
            m[key] = signed(float(p["gain"]))
            m[key + "Lo"] = signed(float(p["lo"]))
            m[key + "Hi"] = signed(float(p["hi"]))
        cov = s030["coverage"][arm]["masked_by_term"]
        m[f"mask{arm}Fijq"] = fmt(int(cov["f_ij_q"]))
        m[f"mask{arm}Epsij"] = fmt(int(cov["eps_ij"]))
        # the gene-identity ladder of 025 on the same arm
        tp = s025["arms"][arm]["test_pearson"]
        for rung in B_RUNGS:
            m[f"b{arm}{rung}"] = fmt(float(tp[rung]["mean"]))
            if tp[rung]["sd"] is not None:
                m[f"b{arm}{rung}Sd"] = fmt(float(tp[rung]["sd"]))
    # the transformer: the headline seed writes the unsuffixed macros, every seed writes
    # its own (``cgtQSeedOneValWindow``), and ``cgt<arm>SeedsLogged`` lists them all
    for ref in s030["wandb"]:
        a, seed = ref["arm"], int(ref["seed"])
        prefixes = [f"cgt{a}Seed{SEED_WORD[seed]}"]
        if seed == WANDB_HEADLINE_SEED:
            prefixes.append(f"cgt{a}")
        for pre in prefixes:
            m[f"{pre}ValWindow"] = fmt(float(ref["val_pearson_last_window_mean"]))
            m[f"{pre}ValMax"] = fmt(float(ref["val_pearson_max"]))
            m[f"{pre}ValMaxEpoch"] = fmt(int(ref["val_pearson_max_epoch"]))
            m[f"{pre}LastEpoch"] = fmt(int(ref["last_epoch"]))
            m[f"{pre}FirstEpoch"] = fmt(int(ref["last_epoch"]) - int(ref["window"]) + 1)
            m[f"{pre}EpochsSynced"] = fmt(int(ref["n_epochs_logged"]))
            m[f"{pre}Runs"] = ", ".join(ref["run_ids"])
        m[f"cgt{a}Seed"] = fmt(WANDB_HEADLINE_SEED)
    for a in ("R", "Q"):
        refs = sorted((r for r in s030["wandb"] if r["arm"] == a), key=lambda r: r["seed"])
        m[f"cgt{a}NSeeds"] = fmt(len(refs))
        m[f"cgt{a}SeedsLogged"] = "; ".join(
            f"seed {r['seed']}, {r['n_epochs_logged']} epochs to {r['last_epoch']}"
            + (" (partial, under the window)" if r["partial"] else "")
            for r in refs
        )
    for f in fitness:
        prefixes = [f"fit{f.arm}CgtSeed{SEED_WORD[f.seed]}"]
        if f.seed == WANDB_HEADLINE_SEED:
            prefixes.append(f"fit{f.arm}Cgt")
        for pre in prefixes:
            m[f"{pre}ValWindow"] = fmt(f.val_fitness_order3_last_window_mean)
            m[f"{pre}LastEpoch"] = fmt(f.last_epoch)
            m[f"{pre}FirstEpoch"] = fmt(f.last_epoch - f.window + 1)
            m[f"{pre}EpochsSynced"] = fmt(f.n_epochs_logged)
    # 010 checkpoints and the 1598 replication on arm R
    t = s025["transformer"]["R"]
    for name, v in t["010_checkpoints_test_pearson"].items():
        m["cgtOld" + macro_name(name.split("_")[1])] = fmt(float(v))
    tests = cgt010[cgt010["split"] == "test"]["pearson"].astype(float)
    m["cgtOldTestMean"] = fmt(float(tests.mean()))
    m["cgtOldTestMin"] = fmt(float(tests.min()))
    m["cgtOldTestMax"] = fmt(float(tests.max()))
    m["cgtOldN"] = fmt(int(len(tests)))
    m["cgtRepValBest"] = fmt(float(t["job_1598_replication"]["val_pearson_best_epoch"]))
    m["cgtRepBestEpoch"] = fmt(int(t["job_1598_replication"]["best_epoch"]))
    m["cgtRepRun"] = str(t["job_1598_replication"]["run"])
    # comparators on other builds
    m["dcellBestVal"] = fmt(float(dcell["val_pearson"].max()))
    m["dcellMeanVal"] = fmt(float(dcell["val_pearson"].mean()))
    m["dcellN"] = fmt(int(len(dcell)))
    m["dcellDays"] = fmt(float(dcell["wall_clock_days"].max()))
    best = dango.sort_values("mean", ascending=False).iloc[0]
    m["dangoBestMean"] = fmt(float(best["mean"]))
    m["dangoBestRelease"] = str(best["string_version"]).replace("_", ".")
    m["dangoBestSchedule"] = str(best["loss_schedule"])
    m["dangoBestN"] = fmt(int(best["n"]))
    m["fbaPearson"] = f"{float(fba['pearson_r']):.4f}"
    m["fbaSpearman"] = f"{float(fba['spearman_rho']):.4f}"
    m["fbaN"] = fmt(int(fba["n"]))
    m["fbaFracZero"] = fmt(100 * float(fba["frac_abs_below_1e-3"]))
    return m


def write_macros(m: dict[str, str], path: str) -> None:
    lines = [
        "%% SOURCE: written by experiments/030-solid-growth-multi/scripts/figure2_gate_numbers.py",
        "%% from the result files named in its docstring; regenerate, do not edit.",
    ]
    for k, v in sorted(m.items()):
        lines.append(f"\\newcommand{{\\g{macro_name(k)}}}{{{v}}}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(m)} macros)")


def write_figure_names(path: str) -> None:
    """The newest render of each figure the document places, by stem, as macros."""
    import glob

    stems = {
        "figLadder": "lower_order_baselines_030_ladder_",
        "figFitness": "figure2_gate_fitness_",
        "figSheet": "figure2_gate_sheet_",
        "figSheetTwo": "figure2_gate_sheet2_",
    }
    lines = [
        "%% SOURCE: written by figure2_gate_numbers.py; the newest SVG of each stem under",
        "%% notes/assets/images/030-solid-growth-multi/, converted by `make plots`.",
    ]
    for macro, stem in stems.items():
        files = sorted(glob.glob(osp.join(IMAGES, stem + "*.svg")))
        assert files, f"no render of {stem}* under {IMAGES}"
        name = osp.splitext(osp.basename(files[-1]))[0]
        lines.append(f"\\newcommand{{\\{macro}}}{{figures/{name}.pdf}}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def write_ladder_rows(ladder: pd.DataFrame, path: str) -> None:
    lines = [
        "%% SOURCE: experiments/030-solid-growth-multi/results/lower_order_baselines_030.csv,",
        "%% rows written by figure2_gate_numbers.py; regenerate, do not edit.",
    ]
    for arm in ("R", "Q"):
        sub = ladder[ladder["arm"] == arm].set_index(["rung", "split"])
        for rung, (label, regime) in RUNGS.items():
            v = sub.loc[(rung, "val")]
            t = sub.loc[(rung, "test")]
            lines.append(
                f"{arm} & {tex_escape(label)} & {regime} & {float(v['pearson']):.3f} & "
                f"{float(t['pearson']):.3f} & {float(t['pearson_within_screen']):.3f} & "
                f"{float(t['spearman']):.3f} & {float(t['coverage']):.2f} \\\\"
            )
        if arm == "R":
            lines.append(r"\midrule")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def write_comparator_rows(
    s025: dict[str, Any], cgt010: pd.DataFrame, dcell: pd.DataFrame, dango: pd.DataFrame, fba: dict[str, Any], path: str
) -> None:
    lines = [
        "%% SOURCE: rows written by figure2_gate_numbers.py from the 010, 006, 005 and 007",
        "%% result files named in its docstring; regenerate, do not edit.",
    ]
    for _, r in cgt010[cgt010["split"] == "test"].iterrows():
        lines.append(
            f"CGT 010 checkpoint {tex_escape(r['model'].split('_')[1])} & 010 build, arm R & test & "
            f"{float(r['pearson']):.3f} & one checkpoint \\\\"
        )
    rep = s025["transformer"]["R"]["job_1598_replication"]
    lines.append(
        f"CGT 025 replication, job 1598 & 025 build, arm R & val, best epoch {int(rep['best_epoch'])} & "
        f"{float(rep['val_pearson_best_epoch']):.3f} & max over epochs, no test \\\\"
    )
    lines.append(
        f"DCell, three checkpoints & 006 build & val & {float(dcell['val_pearson'].max()):.3f} & "
        f"best of {len(dcell)}, one {float(dcell['wall_clock_days'].max()):.0f}-day run \\\\"
    )
    best = dango.sort_values("mean", ascending=False).iloc[0]
    lines.append(
        f"DANGO, STRING {str(best['string_version']).replace('_', '.')}, {tex_escape(str(best['loss_schedule']))} & "
        f"005 build, Kuzmin 2018 & test & {float(best['mean']):.3f} & mean of {int(best['n'])} \\\\"
    )
    lines.append(
        f"Yeast9 FBA, nothing fitted & 007 build & all {int(fba['n']):,} triples & "
        f"{float(fba['pearson_r']):.4f} & {100 * float(fba['frac_abs_below_1e-3']):.1f}\\% of predictions at zero \\\\"
    )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


# ------------------------------------------------------------------------------ figure
def plot_fitness(ladder: pd.DataFrame, fitness: list[FitnessRef], stem: str) -> None:
    """The fitness side: f_ijk of held-out triples from the lower-order ridge (test) and
    from the transformer's fitness head (validation), both arms."""
    apply_paper_style()
    COLOR = {"R": PLOT_PALETTE[0], "Q": PLOT_PALETTE[1]}
    FILL = {"R": PLOT_PALETTE_FILL[0], "Q": PLOT_PALETTE_FILL[1]}
    BAR = {"edgecolor": "black", "linewidth": 0.5}
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["third"]), mm_to_in(52.0)))
    cats, x = [], 0
    for arm in ("R", "Q"):
        sub = ladder[ladder["arm"] == arm].set_index(["rung", "split"])["pearson"]
        ax.bar(x, sub.loc[("F1_ridge_lower_order_fitness", "test")], 0.7, color=FILL[arm], **BAR)
        cats.append(f"F1 ridge\narm {arm}")
        x += 1
        # one bar per seed with a full window; a partial seed is on file, not drawn
        refs = [f for f in fitness if f.arm == arm and not f.partial]
        for f in refs:
            ax.bar(x, f.val_fitness_order3_last_window_mean, 0.7, color=COLOR[arm], **BAR)
            seed = f", seed {f.seed}" if len(refs) > 1 else ""
            cats.append(f"CGT 030\narm {arm}{seed}")
            x += 1
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=5)
    ax.set_ylabel("Pearson r, held-out triple fitness")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    handles = [
        Patch(facecolor=FILL["R"], label="ridge, test", **BAR),
        Patch(facecolor=COLOR["R"], label="transformer, validation", **BAR),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1, handlelength=1.4, borderpad=0.4)
    panel_label(ax, "a")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true", help="fetch the fitness-head rows")
    ap.add_argument(
        "--names-only",
        action="store_true",
        help="rewrite only the figure-name macros, after the sheet script has rendered",
    )
    args = ap.parse_args()
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(IMAGES, exist_ok=True)
    if args.names_only:
        write_figure_names(osp.join(TABLES, "gate_figure_names.tex"))
        return
    ladder, paired = read_ladder(), read_paired()
    s030, s025 = read_summary_030(), read_summary_025()
    cgt010, dcell, dango, fba = read_010(), read_dcell(), read_dango(), read_fba()
    cache = osp.join(R030, "figure2_gate_fitness_wandb.json")
    if args.wandb:
        fitness = [f for arm in ("R", "Q") for f in wandb_fitness(arm)]
        with open(cache, "w") as f:
            json.dump([x.model_dump() for x in fitness], f, indent=2)
    else:
        with open(cache) as f:
            fitness = [FitnessRef(**x) for x in json.load(f)]
    m = macros(ladder, paired, s030, s025, cgt010, dcell, dango, fba, fitness)
    write_macros(m, osp.join(TABLES, "gate_numbers.tex"))
    write_ladder_rows(ladder, osp.join(TABLES, "gate_ladder_rows.tex"))
    write_comparator_rows(s025, cgt010, dcell, dango, fba, osp.join(TABLES, "gate_comparator_rows.tex"))
    plot_fitness(ladder, fitness, osp.join(IMAGES, f"figure2_gate_fitness_{timestamp()}"))
    write_figure_names(osp.join(TABLES, "gate_figure_names.tex"))
    print("finished")


if __name__ == "__main__":
    main()
