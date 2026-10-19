# experiments/040-inhibitor-synergy-wetlab/scripts/r11_gate_section.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.r11_gate_section]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/r11_gate_section
r"""Round 11's section of the environment case-study gate: macros, two tables, one figure.

Reads the round-11 score tables through ``r11_readout`` (fold ensembles, averaged over
the folds an arm has) and writes, into ``notes-tex/wet-lab/env-case-study-gate``:

``tables/r11_numbers.tex``       every number section 6 prints, as ``\rEleven...`` macros
``tables/r11_host.tex``          the wet-lab table: the decoder's functional forms
``tables/r11_public.tex``        the public-side table: operator, public bar, sources,
                                 held-out dose, gene-cold
``tables/r11_figure_names.tex``  ``\figRElevenHost``, the newest host figure

and the figure itself, ``r11_host_<stamp>.svg`` (true size) and ``.png`` into
``$ASSET_IMAGES_DIR/040-inhibitor-synergy-wetlab``, with ``r11_host.pdf`` beside them
for review. The model-free bars (Loewe, Bliss) are the reference values the trainer
wrote into every score table, read back from them rather than typed.

    python experiments/040-inhibitor-synergy-wetlab/scripts/r11_gate_section.py
"""

from __future__ import annotations

import glob
import os
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from r11_readout import CONTROL, arm_row, read_scores, seed_floor  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
    text_clashes,
)

load_dotenv()
EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
REPO = osp.dirname(osp.dirname(EXPERIMENT))
RESULTS = osp.join(EXPERIMENT, "results")
DOC = osp.join(REPO, "notes-tex", "wet-lab", "env-case-study-gate")
TABLES = osp.join(DOC, "tables")
SLUG = "040-inhibitor-synergy-wetlab"
SOURCE = (
    "%% SOURCE: written by experiments/040-inhibitor-synergy-wetlab/scripts/"
    "r11_gate_section.py from results/mixture-delta/r11_*/*_scores.csv\n"
)
BLISS, LOEWE, MODEL, CONTROL_GRAY = (
    PLOT_PALETTE[1],
    PLOT_PALETTE[2],
    PLOT_PALETTE[4],
    PLOT_PALETTE[5],
)

#: The wet-lab arms the figure and the host table show, in reading order, with the
#: label each carries. ``zs`` marks an arm that never saw a titration or combination
#: well, which the figure hatches.
HOST_ARMS: list[tuple[str, str, bool]] = [
    (CONTROL, "control: round-10 Bliss arm", False),
    ("van_e150_cell_bliss_hill", "Hill + Bliss", False),
    ("van_e150_cell_bliss_hill_loewe", "Hill + Loewe", False),
    ("van_e150_cell_bliss_hill_hsa", "Hill + highest single", False),
    ("van_e150_cell_bliss_head_hsa", "head + highest single", False),
    ("van_e150_cell_bliss_hill_ft", "Hill, titration fine-tune", False),
    ("van_e150_cell_bliss_hill_loewe_doff", "Hill + Loewe + dose gain", False),
    ("van_e150_cell_bliss_attend_doff", "head + dose gain", False),
    ("van_e150_cell_bliss_displace_doff", "displacement + dose gain", False),
    ("van_e150_cell_bliss_hill_anchors", "Hill + Bliss, zero-shot", True),
    ("van_e150_cell_bliss_hill_loewe_anchors", "Hill + Loewe, zero-shot", True),
    (
        "vanhophet_e150_cell_anchors_bliss_s345",
        "head + Bliss, zero-shot, 3 screens",
        True,
    ),
]

#: The public-side arms, grouped by the question each answers.
PUBLIC_ARMS: list[tuple[str, str, str]] = [
    ("operator", CONTROL, "control: tokens attend to the genes"),
    ("operator", "van_e150_cell_bliss_attend_co", "attend, head reads the cell only"),
    ("operator", "van_e150_cell_bliss_attend_L2", "attend, two medium layers"),
    ("operator", "van_e150_cell_bliss_displace", "displace the gene embeddings"),
    (
        "operator",
        "van_e150_cell_bliss_displace_co",
        "displace, head reads the cell only",
    ),
    ("public bar", "van_e150_cell_bliss_cgt2", "two CGT layers"),
    ("public bar", "van_e150_cell_bliss_do0p4", "dropout 0.4"),
    ("public bar", "van_e150_cell_bliss_wd1e3", "weight decay 1e-3"),
    ("public bar", "van_e150_cell_bliss_lr5e4", "learning rate 5e-4"),
    ("public bar", "van_e150_cell_bliss_cgt0p1", "graph prior 0.1"),
    ("public bar", "van_e150_cell_bliss_cgt1", "graph prior 1"),
    ("sources", "van_e150_cell_bliss_hill_loewe", "Vanacloig, Hill + Loewe"),
    ("sources", "vanhop_e150_cell_bliss_hill_loewe", "+ Hoepfner"),
    ("sources", "vanhet_e150_cell_bliss_hill_loewe", "+ Hillenmeyer"),
    ("sources", "vanwil_e150_cell_bliss_hill_loewe", "+ Wildenhain"),
    ("sources", "vanhophet_e150_cell_bliss_hill_loewe", "+ Hoepfner + Hillenmeyer"),
    ("sources", "vanhophetwil_e150_cell_bliss_hill_loewe", "+ all three"),
    ("held-out dose", "vanhop_e150_cell_bliss_doff_heldout", "no dose channel"),
    ("held-out dose", "vanhop_e150_cell_bliss_dfilm_heldout", "FiLM"),
    (
        "held-out dose",
        "vanhop_e150_cell_bliss_doffset_heldout",
        "dose gain on the offset",
    ),
    (
        "held-out dose",
        "vanhop_e150_cell_bliss_hill_doff_heldout",
        "Hill, no dose channel",
    ),
    ("held-out dose", "vanhop_e150_cell_bliss_hill_dfilm_heldout", "Hill + FiLM"),
    (
        "held-out dose",
        "vanhop_e150_cell_bliss_hill_doffset_heldout",
        "Hill + dose gain",
    ),
    ("gene-cold", "van_e150_cell_bliss_attend_gene", "attend"),
    ("gene-cold", "van_e150_cell_bliss_attend_dev_gene", "attend + pair term"),
    ("gene-cold", "van_e150_cell_bliss_displace_gene", "displace"),
    ("gene-cold", "van_e150_cell_bliss_displace_dev_gene", "displace + pair term"),
]

DIGITS = dict(enumerate("Zero One Two Three Four Five Six Seven Eight Nine".split()))


def macro_name(key: str) -> str:
    r"""``\rEleven`` + the key with every digit spelled out (macros cannot hold digits)."""
    return "rEleven" + "".join(DIGITS[int(c)] if c.isdigit() else c for c in key)


def fmt(value: float, places: int = 3) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{places}f}"


def ensemble_by_fold(
    scores: pd.DataFrame, arm: str, task: str, metric: str, subset: str
) -> pd.Series:
    rows = scores[
        (scores["arm"] == arm)
        & (scores["member"] == "ensemble")
        & (scores["task"] == task)
        & (scores["metric"] == metric)
        & (scores["subset"] == subset)
        & (scores["call"] == "served")
    ]
    return rows.set_index("fold")["value"].sort_index()


def single_seeds(
    scores: pd.DataFrame, arm: str, task: str, metric: str, subset: str
) -> pd.Series:
    rows = scores[
        (scores["arm"] == arm)
        & (scores["member"] != "ensemble")
        & (scores["task"] == task)
        & (scores["metric"] == metric)
        & (scores["subset"] == subset)
        & (scores["call"] == "served")
    ]
    return rows["value"]


def reference(scores: pd.DataFrame, task: str, metric: str, subset: str) -> float:
    rows = scores[
        (scores["task"] == task)
        & (scores["metric"] == metric)
        & (scores["subset"] == subset)
        & (scores["call"] == "served")
    ]["reference_value"].dropna()
    assert rows.nunique() == 1, f"{task} {metric}: reference differs across tables"
    return float(rows.iloc[0])


def seconds_per_seed(arm: str) -> float:
    """Mean wall seconds of one seed's training, from the arm's history tables."""
    values = []
    for path in glob.glob(
        osp.join(RESULTS, "mixture-delta", "r11_*", f"mix_{arm}_f*_history.csv")
    ):
        history = pd.read_csv(path, usecols=["seed", "sys/seconds"])
        values.extend(history.groupby("seed")["sys/seconds"].max().tolist())
    return float(np.mean(values)) if values else float("nan")


def host_figure(scores: pd.DataFrame, table: pd.DataFrame) -> dict[str, str]:
    """Three panels, one row per arm: growth AUROC, worst-scenario and grown Spearman."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    panels = [
        ("growth_auroc", "combinations", "growth AUROC, 63 combinations", (0.5, 1.0)),
        (
            "worst_scenario_spearman",
            "combinations",
            "Spearman, all 63, no growth at 0",
            (0.0, 0.9),
        ),
        (
            "fitness_spearman",
            "combinations_that_grew",
            "Spearman, combinations that grew",
            (0.0, 0.9),
        ),
    ]
    rules = {
        "growth_auroc": [("Loewe rule", LOEWE)],
        "worst_scenario_spearman": [("Loewe rule", LOEWE)],
        "fitness_spearman": [("Bliss rule", BLISS)],
    }
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(66)),
        sharey=True,
        gridspec_kw={"wspace": 0.06},
        constrained_layout=True,
    )
    arms = [a for a, _, _ in HOST_ARMS if a in set(table["arm"])]
    labels = {a: label for a, label, _ in HOST_ARMS}
    zero_shot = {a for a, _, zs in HOST_ARMS if zs}
    y = np.arange(len(arms))[::-1]
    for ax, (metric, subset, xlabel, xlim) in zip(axes, panels):
        for yi, arm in zip(y, arms):
            folds = ensemble_by_fold(scores, arm, "ex23", metric, subset)
            mean = float(folds.mean())
            ax.barh(
                yi,
                mean - xlim[0],
                left=xlim[0],
                height=0.62,
                color=PLOT_PALETTE_FILL[4] if arm == CONTROL else MODEL,
                edgecolor="black",
                linewidth=0.4,
                hatch="//////" if arm in zero_shot else None,
            )
            ax.plot(
                folds.values,
                [yi] * len(folds),
                "o",
                ms=1.6,
                mfc="white",
                mec="black",
                mew=0.4,
            )
        control = float(
            ensemble_by_fold(scores, CONTROL, "ex23", metric, subset).mean()
        )
        ax.axvline(control, color=CONTROL_GRAY, lw=0.6)
        task_ref = (
            "combinations" if metric != "fitness_spearman" else "combinations_that_grew"
        )
        for name, color in rules[metric]:
            ax.axvline(reference(scores, "ex23", metric, task_ref), color=color, lw=0.9)
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(axis="x", which="both", lw=0.3, color="0.85")
        ax.tick_params(axis="x", which="minor", length=0)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    axes[0].set_yticks(y, [labels[a] for a in arms])
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    for letter, ax in zip("abc", axes):
        ax.set_title(letter, loc="left", fontsize=8, fontweight="bold")
    handles = [
        Patch(
            facecolor=MODEL,
            edgecolor="black",
            lw=0.4,
            label="trained with the titration wells",
        ),
        Patch(
            facecolor=MODEL,
            edgecolor="black",
            lw=0.4,
            hatch="//////",
            label="zero-shot: public anchors only",
        ),
        plt.Line2D([], [], color=CONTROL_GRAY, lw=0.6, label="control"),
        plt.Line2D([], [], color=LOEWE, lw=0.9, label="Loewe rule, ex21 Hill fits"),
        plt.Line2D([], [], color=BLISS, lw=0.9, label="Bliss rule, measured singles"),
        plt.Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=1.6,
            mfc="white",
            mec="black",
            mew=0.4,
            label="one fold, three-seed ensemble",
        ),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=5.5
    )
    for a, b in text_clashes(fig):
        print(f"  CLASH r11_host: {a!r} x {b!r}", flush=True)
    out_dir = osp.join(os.environ["ASSET_IMAGES_DIR"], SLUG)
    os.makedirs(out_dir, exist_ok=True)
    stamp = timestamp()
    svg = osp.join(out_dir, f"r11_host_{stamp}.svg")
    fig.savefig(osp.join(out_dir, f"r11_host_{stamp}.png"), dpi=300)
    savefig_true_size_svg(fig, svg)
    fig.savefig(
        osp.join(out_dir, "r11_host.pdf"),
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return {"name": f"r11_host_{stamp}", "svg": svg}


def host_table(table: pd.DataFrame, scores: pd.DataFrame) -> str:
    by = table.set_index("arm")
    rows = [
        "Loewe rule, ex21 Hill fits & -- & "
        f"{fmt(reference(scores, 'ex23', 'growth_auroc', 'combinations'))} & "
        f"{fmt(reference(scores, 'ex23', 'worst_scenario_spearman', 'combinations'))}"
        " & -- & -- \\\\"
    ]
    rows.append(
        "Bliss rule, measured singles & -- & -- & -- & "
        f"{fmt(reference(scores, 'ex23', 'fitness_spearman', 'combinations_that_grew'))} & -- \\\\"
    )
    rows.append("\\midrule")
    for arm, label, _ in HOST_ARMS:
        if arm not in by.index:
            continue
        r = by.loc[arm]
        rows.append(
            f"{label} & {int(r['folds'])} & {fmt(r['auroc'])} & {fmt(r['worst'])} & "
            f"{fmt(r['grown'])} & {fmt(r['tit'])} \\\\"
        )
    return (
        SOURCE + "\\begin{table}[htbp]\\centering\\footnotesize\n"
        "\\caption{\\textbf{Round eleven on the wet-lab wells.} Fold ensembles of three seeds,"
        " averaged over the folds named, fold seed zero, served growth call. Growth AUROC and"
        " the worst-scenario Spearman are over the \\hydnCombinations{} combinations (no growth"
        " read as zero for the second); grown is the fitness Spearman over the combinations"
        " that grew; titration is the Spearman over all \\extitration{} wells, a train fit"
        " except for the zero-shot arms, which never saw one.}\\label{tab:r11host}\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "arm & folds & growth AUROC & worst-scenario Spearman & grown Spearman & titration Spearman \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def public_table(table: pd.DataFrame) -> str:
    by = table.set_index("arm")
    rows, last = [], ""
    for group, arm, label in PUBLIC_ARMS:
        if arm not in by.index:
            continue
        r = by.loc[arm]
        if group != last and last:
            rows.append("\\midrule")
        held = [r[k] for k in ("hop", "het", "wil") if np.isfinite(r[k])]
        held_txt = " / ".join(fmt(v) for v in held) if held else "--"
        rows.append(
            f"{group if group != last else ''} & {label} & {int(r['folds'])} & {fmt(r['van'])} & "
            f"{held_txt} & {fmt(r['auroc'])} \\\\"
        )
        last = group
    return (
        SOURCE + "\\begin{table}[htbp]\\centering\\footnotesize\n"
        "\\caption{\\textbf{Round eleven on the public side.} The Vanacloig column is the"
        " compound-cold centered Spearman (gene-cold rows: over held-out genes); nested ridge"
        " on fingerprint counts reads \\hydladderRidge{}. The held-out column is each"
        " auxiliary screen's centered Spearman on its held-out compounds, in the order"
        " Hoepfner / Hillenmeyer / Wildenhain, or on a held-out dose of a seen compound in"
        " the dose rows. Growth AUROC is beside each for the cost on the wet-lab side.}"
        "\\label{tab:r11public}\n"
        "\\begin{tabular}{llrrlr}\n\\toprule\n"
        "question & arm & folds & Vanacloig & held-out public & growth AUROC \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def numbers(scores: pd.DataFrame, table: pd.DataFrame) -> dict[str, str]:
    by = table.set_index("arm")
    floor = seed_floor(scores[scores["arm"] == CONTROL])
    zs = "van_e150_cell_bliss_hill_anchors"
    zs_folds = ensemble_by_fold(scores, zs, "ex23", "growth_auroc", "combinations")
    zs_worst = ensemble_by_fold(
        scores, zs, "ex23", "worst_scenario_spearman", "combinations"
    )
    zs_seed = single_seeds(scores, zs, "ex23", "growth_auroc", "combinations")
    control_folds = ensemble_by_fold(
        scores, CONTROL, "ex23", "growth_auroc", "combinations"
    )
    operator = [
        "van_e150_cell_bliss_attend_co",
        "van_e150_cell_bliss_attend_L2",
        "van_e150_cell_bliss_displace",
        "van_e150_cell_bliss_displace_co",
    ]
    dose_hill = [
        "vanhop_e150_cell_bliss_hill_doff_heldout",
        "vanhop_e150_cell_bliss_hill_dfilm_heldout",
        "vanhop_e150_cell_bliss_hill_doffset_heldout",
    ]
    gene = [a for g, a, _ in PUBLIC_ARMS if g == "gene-cold"]
    het = [by.loc[a, "het"] for a in by.index if np.isfinite(by.loc[a, "het"])]
    get = lambda arm, col: float(by.loc[arm, col])  # noqa: E731
    out: dict[str, float | int] = {
        "nJobs": 105,
        "nConfigs": 186,
        "nArms": 62,
        "nScored": int(scores["name"].nunique()),
        "controlVan": get(CONTROL, "van"),
        "controlAuroc": get(CONTROL, "auroc"),
        "controlWorst": get(CONTROL, "worst"),
        "controlGrown": get(CONTROL, "grown"),
        "controlFoldSd": float(control_folds.std()),
        "seedSdVan": floor["van"],
        "seedSdAuroc": floor["auroc"],
        "zsAuroc": float(zs_folds.mean()),
        "zsAurocMin": float(zs_folds.min()),
        "zsAurocMax": float(zs_folds.max()),
        "zsAurocSd": float(zs_folds.std()),
        "zsWorst": float(zs_worst.mean()),
        "zsWorstMin": float(zs_worst.min()),
        "zsSeedMin": float(zs_seed.min()),
        "zsSeedMax": float(zs_seed.max()),
        "zsGrown": get(zs, "grown"),
        "zsTit": get(zs, "tit"),
        "zsIso": get(zs, "iso"),
        "zsLoeweAuroc": get("van_e150_cell_bliss_hill_loewe_anchors", "auroc"),
        "hillAuroc": get("van_e150_cell_bliss_hill", "auroc"),
        "hillGrown": get("van_e150_cell_bliss_hill", "grown"),
        "hillIso": get("van_e150_cell_bliss_hill", "iso"),
        "loeweAuroc": get("van_e150_cell_bliss_hill_loewe", "auroc"),
        "loeweGrown": get("van_e150_cell_bliss_hill_loewe", "grown"),
        "loeweIso": get("van_e150_cell_bliss_hill_loewe", "iso"),
        "loeweTit": get("van_e150_cell_bliss_hill_loewe", "tit"),
        "loeweFsOneAuroc": get("van_e150_cell_bliss_hill_loewe_fs1", "auroc"),
        "loeweFsOneGrown": get("van_e150_cell_bliss_hill_loewe_fs1", "grown"),
        "hsaAuroc": get("van_e150_cell_bliss_hill_hsa", "auroc"),
        "hsaGrown": get("van_e150_cell_bliss_hill_hsa", "grown"),
        "hsaIso": get("van_e150_cell_bliss_hill_hsa", "iso"),
        "ftAuroc": get("van_e150_cell_bliss_hill_ft", "auroc"),
        "ftWorst": get("van_e150_cell_bliss_hill_ft", "worst"),
        "ftGrown": get("van_e150_cell_bliss_hill_ft", "grown"),
        "ftTit": get("van_e150_cell_bliss_hill_ft", "tit"),
        "monotoneCeiling": get("van_e150_cell_bliss_hill", "dose"),
        "operatorVanMin": min(get(a, "van") for a in operator),
        "operatorVanMax": max(get(a, "van") for a in operator),
        "operatorAurocMin": min(get(a, "auroc") for a in operator),
        "operatorAurocMax": max(get(a, "auroc") for a in operator),
        "displaceDoffTit": get("van_e150_cell_bliss_displace_doff", "tit"),
        "displaceDoffAuroc": get("van_e150_cell_bliss_displace_doff", "auroc"),
        "displaceDoffIso": get("van_e150_cell_bliss_displace_doff", "iso"),
        "attendDoffTit": get("van_e150_cell_bliss_attend_doff", "tit"),
        "attendDoffAuroc": get("van_e150_cell_bliss_attend_doff", "auroc"),
        "devZsAuroc": get("van_e150_cell_bliss_dev_anchors", "auroc"),
        "zsHeadAuroc": get("van_e150_cell_anchors_bliss_s345", "auroc"),
        "zsHeadThreeAuroc": get("vanhophet_e150_cell_anchors_bliss_s345", "auroc"),
        "zsHeadTwoAuroc": get("vanhop_e150_cell_anchors_bliss_s345", "auroc"),
        "devThreeIso": get("vanhophet_e150_cell_bliss_dev", "iso"),
        "controlIso": get(CONTROL, "iso"),
        "hetMin": float(min(het)),
        "hetMax": float(max(het)),
        "doseOff": get("vanhop_e150_cell_bliss_doff_heldout", "hop"),
        "doseFilm": get("vanhop_e150_cell_bliss_dfilm_heldout", "hop"),
        "doseOffset": get("vanhop_e150_cell_bliss_doffset_heldout", "hop"),
        "doseHillMin": min(get(a, "hop") for a in dose_hill),
        "doseHillMax": max(get(a, "hop") for a in dose_hill),
        "loeweVan": get("van_e150_cell_bliss_hill_loewe", "van"),
        "cgtTwoVan": get("van_e150_cell_bliss_cgt2", "van"),
        "priorLowVan": get("van_e150_cell_bliss_cgt0p1", "van"),
        "priorHighVan": get("van_e150_cell_bliss_cgt1", "van"),
        "priorFolds": int(get("van_e150_cell_bliss_cgt0p1", "folds")),
        "priorMinutes": seconds_per_seed("van_e150_cell_bliss_cgt0p1") / 60,
        "plainMinutes": seconds_per_seed(CONTROL) / 60,
        "wilHeld": get("vanwil_e150_cell_bliss_hill_loewe", "wil"),
        "wilGrown": get("vanwil_e150_cell_bliss_hill_loewe", "grown"),
        "sourceVanMin": min(
            get(a, "van")
            for g, a, _ in PUBLIC_ARMS
            if g == "sources" and a != "van_e150_cell_bliss_hill_loewe"
        ),
        "sourceVanMax": max(
            get(a, "van")
            for g, a, _ in PUBLIC_ARMS
            if g == "sources" and a != "van_e150_cell_bliss_hill_loewe"
        ),
        "geneMin": min(get(a, "van") for a in gene),
        "geneMax": max(get(a, "van") for a in gene),
    }
    text = {}
    for key, v in out.items():
        if isinstance(v, int):
            text[key] = f"{v:,}"
        elif key.endswith("Minutes"):
            text[key] = f"{v:.0f}"
        else:
            text[key] = fmt(v)
    return text


#: The mermaid render of the leading arm's wiring (Supplement nine), and where the
#: document reads it. mermaid_pdf.sh fits the page to the diagram, which comes out wider
#: than the text block, so it is scaled here to the full Nature width.
WIRING_PDF = osp.join(
    REPO,
    "notes",
    "assets",
    "pdf-output",
    "experiments.040-inhibitor-synergy-wetlab.mermaid.hill-bliss-model.pdf",
)
WIRING_OUT = osp.join(DOC, "figures", "r11_model_wiring.pdf")


def wiring_figure() -> float:
    """Scale the rendered wiring diagram to the full panel width; returns its width in mm."""
    from pypdf import PdfReader, PdfWriter, Transformation

    page = PdfReader(WIRING_PDF).pages[0]
    width_pt = float(page.mediabox.width)
    target_pt = PANEL_WIDTHS_MM["full"] / 25.4 * 72
    factor = target_pt / width_pt
    page.add_transformation(Transformation().scale(factor, factor))
    page.mediabox.lower_left = (0, 0)
    page.mediabox.upper_right = (target_pt, float(page.mediabox.height) * factor)
    page.cropbox = page.mediabox
    writer = PdfWriter()
    writer.add_page(page)
    with open(WIRING_OUT, "wb") as f:
        writer.write(f)
    return target_pt / 72 * 25.4


def main() -> None:
    scores = read_scores()
    table = pd.DataFrame([arm_row(arm, frame) for arm, frame in scores.groupby("arm")])
    os.makedirs(TABLES, exist_ok=True)
    text = numbers(scores, table)
    with open(osp.join(TABLES, "r11_numbers.tex"), "w") as f:
        f.write(SOURCE)
        for key, value in sorted(text.items()):
            f.write(f"\\newcommand{{\\{macro_name(key)}}}{{{value}}}\n")
    with open(osp.join(TABLES, "r11_host.tex"), "w") as f:
        f.write(host_table(table, scores))
    with open(osp.join(TABLES, "r11_public.tex"), "w") as f:
        f.write(public_table(table))
    figure = host_figure(scores, table)
    with open(osp.join(TABLES, "r11_figure_names.tex"), "w") as f:
        f.write(SOURCE)
        f.write(f"\\newcommand{{\\figRElevenHost}}{{figures/{figure['name']}.pdf}}\n")
    wiring_mm = wiring_figure()
    print(
        f"{len(text)} macros, {text['nScored']} configs, figure {figure['svg']}, "
        f"wiring diagram {wiring_mm:.0f} mm"
    )


if __name__ == "__main__":
    main()
