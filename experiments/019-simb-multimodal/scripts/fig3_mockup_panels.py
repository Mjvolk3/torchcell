# experiments/019-simb-multimodal/scripts/fig3_mockup_panels.py
# [[experiments.019-simb-multimodal.scripts.fig3_mockup_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/fig3_mockup_panels
"""Preliminary panels for the Figure 3 gate (notes-tex/figure-3-gate/).

Small panels, each one idea, each drawn ONLY from committed result files under
experiments/019-simb-multimodal/results/ (and, for the co-variation panel,
experiments/028-knockout-expression/results/). Nothing is fetched from W&B and nothing is
recomputed from raw data here; a quantity that is in no results file is drawn as an
explicit "not yet run" or "not in file" marker.

  ceiling_vs_model      replicate ceiling, best genotype-only model, best linear baseline
       and best kNN baseline, three modalities. Ceilings and model as before:
       expression_ceiling_replicate.json, round_leaderboards.csv (strand "expression",
       maximum primary_roll_max), proteome_ceiling_replicate.json,
       morphology_noise_ceiling.json. Linear: baselines_embedding_study.json, the largest
       {expression,proteome}_B2_val mean over embeddings. kNN: the larger of the largest
       {expression,proteome}_B3_val mean in baselines_embedding_study.json and the largest
       graph or ProtT5 key in graph_retrieval_baseline.json (rounds v13, v14,
       val/all/pearson_per_feature); morphology kNN from knn_embedding_probe.json (largest
       best_pearson_per_feature over arms). No linear baseline for morphology is on file.
  triangle              out-of-fold ridge between measured modalities, six directed pairs.
       proteome_morphology_covariation.json, expression_morphology_covariation.json,
       joint_checkpoint_readout.json["triangle"] for one shared-strain count.
  v19_paired            joint minus single per split seed, both heads.
       joint_checkpoint_readout.json["v19"]["tests"].
  v20_partial           conditioned minus control at the same epochs. PARTIAL.
       joint_checkpoint_readout.json["v20_partial"]["runs"].
  v19_timing            epoch of the loss minimum and of the proteome score peak.
       joint_checkpoint_readout.json["v19"]["runs"].
  v19_per_seed          the single-head and joint window scores behind v19_paired.
       joint_checkpoint_readout.json["v19"]["runs"].
  baselines_by_seed_*   linear, kNN and the v19 single-head model per split seed.
       expression_baselines_split/seed*.json, baselines_split_fig3_proteome/seed*.json
       (B2_bilinear and B3_neighbor_average, best embedding selected on validation) and
       joint_checkpoint_readout.json["v19"]["runs"] (arms K_expr, K_prot, *_window).
       The seed NUMBER is shared; the stores are not: the baselines ran on fig3_core and on
       the full fig3_proteome store, the v19 arms on the both-label subset.
  graph_effect_*        pair-form AUC at one step against the rewired control, per graph.
       graph_prior_probe.json["pair_form"], strands expression and morphology. The file
       has no proteome strand, which the panel says in a labeled empty axis.
  graph_retrieval       the graph as a retrieval key minus its rewired control, expression
       and proteome. graph_retrieval_baseline.json rounds v13 and v14, summary.paired.
  covariation           agreement between panels on which gene pairs co-vary.
       experiments/028-knockout-expression/results/proteome_expression_covariation.json
       ["gene_covariation"].

  embedding_reduced, embedding_full   ridge (B2) and nearest-neighbor (B3) validation mean
       and sd over four split seeds per gene representation, expression and proteome.
       baselines_embedding_study.json. The model's input composite is the row model_stack:
       expression_baselines_split.py defines it as (fudt_upstream, calm, prot_T5_all,
       fudt_downstream), the node_embeddings list of conf/cgt_expr_v13_split.yaml that the
       v19 configuration inherits. The reduced set is chosen by rank on the mean of the four
       validation means: best single language model, the model composite, best other
       composite, codon frequency, best random control.
  size_vs_score         parameter count against score, one point per expression-strand run.
       round_leaderboards.csv (total_param_count, primary_roll_max, is_collapsed,
       n_train_supervised); the v19 size from joint_checkpoint_readout.json v19.runs
       total_param_count and its training strains from split_gene_overlap_audit.json.
  curves_proteome, curves_expression   validation loss and validation Pearson against
       epoch for one split seed, single-head and joint arms. joint_checkpoint_curves.csv
       (written by joint_checkpoint_readout.py) and the marker epochs of
       joint_checkpoint_readout.json. The split seed is chosen by rule: the complete split
       seed whose joint-minus-single proteome window difference is closest to the median
       over complete split seeds (the lower seed on a tie).

Panels carry no panel letter: the same panel takes a different letter in each mockup, so
the letters are set in the draw.io compositions (fig3_mockup_drawio.py).

Writes $ASSET_IMAGES_DIR/019-simb-multimodal/fig3_mockup_<name>.{svg,png}, stable names.

    python experiments/019-simb-multimodal/scripts/fig3_mockup_panels.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import os.path as osp
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from torchcell.utils import (  # noqa: E402
    MAX_HEIGHT_MM,
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS = experiment_results_dir("019-simb-multimodal", __file__)
RESULTS_028 = experiment_results_dir("028-knockout-expression", __file__)
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-simb-multimodal")
PANEL_HEIGHT_MM = 48.0
FAMILY_COLORS = dict(
    zip(
        ["protein LM", "coding sequence", "regulatory DNA", "composite", "graph", "control"],
        PLOT_PALETTE[:6],
    )
)
MODEL_COMPOSITE = "model_stack"
EMBEDDING_LABELS = {
    "prot_T5_all": "ProtT5",
    "prot_T5_no_dubious": "ProtT5, no dubious",
    "esm2_650M_all": "ESM2 650M",
    "esm2_650M_no_dubious": "ESM2 650M, no dubious",
    "calm": "CaLM",
    "codon_frequency": "codon frequency",
    "species_lm_five_prime": "species LM, 5' flank",
    "species_lm_three_prime": "species LM, 3' flank",
    "species_lm_5p_3p": "species LM, both flanks",
    "nt_window_5979": "NT, 5,979 window (mean)",
    "nt_window_5979_max": "NT, 5,979 window (max)",
    "nt_window_five_prime_1003": "NT, 5' 1,003",
    "nt_window_three_prime_300": "NT, 3' 300",
    "nt_window_five_prime_5979": "NT, 5' 5,979",
    "nt_window_three_prime_5979": "NT, 3' 5,979",
    "nt_5prime_3prime": "NT, 5' 1,003 + 3' 300",
    "normalized_chrom_pathways": "chromatin pathways",
    "random_1024": "random (1,024)",
    "random_100": "random (100)",
    "random_10": "random (10)",
    "prot_T5+calm": "ProtT5 + CaLM",
    "prot_T5+esm2": "ProtT5 + ESM2",
    "prot_T5+species_lm_5p_3p": "ProtT5 + species LM",
    "esm2+calm": "ESM2 + CaLM",
    "esm2+species_lm_5p_3p": "ESM2 + species LM",
    "model_stack": "CGT input stack",
    "model_stack+esm2": "model stack + ESM2",
    "all_sequence": "all sequence (+ NT window)",
}
SPLIT_LABEL = "split seed (data partition)"
ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
FILL_ORANGE, FILL_RED, FILL_PURPLE = PLOT_PALETTE_FILL[:3]
FILL_GRAY = PLOT_PALETTE_FILL[5]
LEGEND = {"frameon": True, "edgecolor": "black", "fancybox": False, "framealpha": 1.0}
GRAPH_LABELS = {
    "physical_interaction": "physical",
    "physical": "physical",
    "regulatory_interaction": "regulatory",
    "regulatory": "regulatory",
    "tflink": "TFLink",
    "string12_0_neighborhood": "neighborhood",
    "string12_0_fusion": "fusion",
    "string12_0_cooccurence": "co-occurrence",
    "string12_0_coexpression": "coexpression",
    "string12_0_experimental": "experimental",
    "string12_0_database": "database",
    "all9_union": "union of nine",
}
PANEL_NAMES = {
    "messner": "Messner",
    "kemmeren": "Kemmeren",
    "caudal": "Caudal",
    "nadalA": "Nadal-Ribelles",
}

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
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "patch.linewidth": 0.5,
        "hatch.linewidth": 0.5,
    }
)


def _load(name: str, root: str = RESULTS) -> dict[str, Any]:
    with open(osp.join(root, name)) as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _figure(
    width_key: str,
    ncols: int = 1,
    nrows: int = 1,
    height_mm: float = PANEL_HEIGHT_MM,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    assert height_mm <= MAX_HEIGHT_MM
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(mm_to_in(PANEL_WIDTHS_MM[width_key]), mm_to_in(height_mm)),
        **kwargs,
    )
    return fig, axes


def _box(ax: Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("black")
    ax.set_axisbelow(True)


def _grid(ax: Axes, major: float, minor: float, axis: str = "y") -> None:
    """Gridlines at every ``minor`` step, labeled at every ``major`` step."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(MultipleLocator(major))
    target.set_minor_locator(MultipleLocator(minor))
    ax.tick_params(axis=axis, which="minor", length=0)
    ax.grid(axis=axis, which="both", color="#CCCCCC", lw=0.4)


def _top_legend(
    fig: Figure, ax: Axes, handles: list[Any], ncol: int, x: float | None = None
) -> None:
    """A framed legend above the axes, left edge on the first axes' left edge.

    ``x`` (figure fraction) moves the left edge when the legend is wider than the axes;
    keep it right of the panel-letter corner.
    """
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(ax.get_position().x0 if x is None else x, 0.985),
        borderaxespad=0.0,
        ncol=ncol,
        columnspacing=1.0,
        handletextpad=0.5,
        **LEGEND,
    )


def _save(fig: Figure, name: str) -> None:
    os.makedirs(IMG_DIR, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, f"fig3_mockup_{name}.svg"))
    fig.savefig(osp.join(IMG_DIR, f"fig3_mockup_{name}.png"), dpi=300)
    plt.close(fig)
    print(f"wrote {osp.join(IMG_DIR, f'fig3_mockup_{name}.svg')}")


def _best_baselines() -> dict[str, dict[str, tuple[str, float] | None]]:
    """Best linear and best kNN baseline per modality, each with the key that won.

    Every entry is a maximum over keys selected on the validation score it reports, so
    each is an upward-biased order statistic.
    """
    study = _load("baselines_embedding_study.json")["embeddings"]
    retrieval = _load("graph_retrieval_baseline.json")["rounds"]
    probe = _load("knn_embedding_probe.json")["arms"]
    out: dict[str, dict[str, tuple[str, float] | None]] = {}
    for modality, round_key in (("expression", "v13"), ("proteome", "v14")):
        linear = max(
            ((f"ridge on {e}", v[f"{modality}_B2_val"]["mean"]) for e, v in study.items()),
            key=lambda kv: kv[1],
        )
        knn_embedding = [
            (f"embedding {e}", v[f"{modality}_B3_val"]["mean"]) for e, v in study.items()
        ]
        keys = retrieval[round_key]["summary"]["keys"]
        knn_graph = [
            (f"graph key {k}", v["val/all/pearson_per_feature"]["mean"])
            for k, v in keys.items()
            if not k.endswith("_rewired")
            and k not in ("oracle", "train_mean", "random_embedding")
        ]
        knn = max(knn_embedding + knn_graph, key=lambda kv: kv[1])
        out[modality] = {"linear": linear, "knn": knn}
    morph = [
        (f"embedding {arm}", v["modalities"]["morphology"]["best_pearson_per_feature"])
        for arm, v in probe.items()
    ]
    out["morphology"] = {
        "linear": None,
        "knn": max((kv for kv in morph if not math.isnan(kv[1])), key=lambda kv: kv[1]),
    }
    for modality, entry in out.items():
        print(f"best baselines, {modality}: {entry}")
    return out


def panel_ceiling_vs_model() -> None:
    expr = _load("expression_ceiling_replicate.json")
    prot = _load("proteome_ceiling_replicate.json")
    morph = _load("morphology_noise_ceiling.json")
    board = pd.read_csv(osp.join(RESULTS, "round_leaderboards.csv"), low_memory=False)
    baselines = _best_baselines()
    expr_model = float(
        board.loc[board["strand"] == "expression", "primary_roll_max"].max()
    )
    expr_ceiling = float(expr["primary_ceiling_mean_sqrt_r"]["ceiling"])
    prot_low = float(prot["route_d_duplicate_strains"]["mean_ceiling"])
    prot_high = float(prot["route_w_his3_replicate"]["mean_ceiling"])
    prot_model = float(prot["observed"]["v14_partition_mean"])
    morph_ceiling = float(morph["ceiling_mean_model_features"])
    morph_model = float(morph["observed_best"]["roll_max"])
    fractions = [
        f"{100 * expr_model / expr_ceiling:.0f}%",
        f"{100 * prot['route_w_his3_replicate']['v14_frac_of_ceiling']:.0f}–"
        f"{100 * prot['route_d_duplicate_strains']['v14_frac_of_ceiling']:.0f}%",
        f"{100 * morph['fraction_of_ceiling_realized']:.0f}%",
    ]
    modalities = ["expression", "proteome", "morphology"]
    ceilings = [expr_ceiling, prot_low, morph_ceiling]
    models = [expr_model, prot_model, morph_model]
    lines = [ORANGE, RED, PURPLE]
    fills = [FILL_ORANGE, FILL_RED, FILL_PURPLE]

    fig, ax = _figure("half")
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.12, top=0.72)
    w = 0.18
    x_ceiling, x_model = -0.33, -0.09
    x_base = {"linear": 0.09, "knn": 0.27}
    bar = {"edgecolor": "black", "lw": 0.5, "zorder": 2}
    for i, modality in enumerate(modalities):
        ax.bar(i + x_ceiling, ceilings[i], w, color=fills[i], **bar)
        ax.bar(i + x_model, models[i], w, color=lines[i], **bar)
        for kind, hatch in (("linear", "////"), ("knn", "xxxx")):
            xb = i + x_base[kind]
            entry = baselines[modality][kind]
            if entry is None:
                ax.bar(
                    xb, 0.44, 0.7 * w, facecolor="none", edgecolor="black", lw=0.5,
                    ls="--", zorder=2,
                )
                ax.text(xb, 0.03, "not yet run", rotation=90, ha="center", va="bottom")
                continue
            ax.bar(xb, entry[1], w, color=lines[i], hatch=hatch, **bar)
    ax.bar(
        [1 + x_ceiling], [prot_high - prot_low], w, bottom=[prot_low], color=FILL_RED,
        hatch="....", **bar,
    )
    top_labels = [
        f"{expr_ceiling:.3f}",
        f"{prot_low:.3f}–{prot_high:.3f}",
        f"{morph_ceiling:.3f}",
    ]
    tops = [expr_ceiling, prot_high, morph_ceiling]
    for i in range(3):
        ax.text(i + x_ceiling, tops[i] + 0.02, top_labels[i], ha="center", va="bottom")
        label_y = models[i] + 0.03
        for kind in ("linear", "knn"):
            entry = baselines[modalities[i]][kind]
            if entry is not None:
                label_y = max(label_y, entry[1] + 0.03)
        if "–" in fractions[i]:
            ax.text(i + x_model - w / 2, label_y, fractions[i], ha="left", va="bottom")
        else:
            ax.text(
                i + x_model + w / 2 - 0.02, models[i] + 0.03, fractions[i], ha="right",
                va="bottom",
            )
    ax.set_xticks(range(3))
    ax.set_xticklabels(modalities)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Pearson per feature")
    _grid(ax, 0.2, 0.1)
    _box(ax)
    _top_legend(
        fig,
        ax,
        [
            Patch(facecolor=FILL_GRAY, edgecolor="black", label="replicate ceiling"),
            Patch(
                facecolor=FILL_GRAY, edgecolor="black", hatch="....",
                label="ceiling, HIS3 replicates",
            ),
            Patch(
                facecolor=GRAY, edgecolor="black",
                label="best genotype-only model (% of ceiling)",
            ),
            Patch(
                facecolor=GRAY, edgecolor="black", hatch="////",
                label="best linear baseline",
            ),
            Patch(
                facecolor=GRAY, edgecolor="black", hatch="xxxx",
                label="best kNN baseline",
            ),
        ],
        ncol=2,
    )
    _save(fig, "ceiling_vs_model")


def panel_triangle() -> None:
    pm = _load("proteome_morphology_covariation.json")
    em = _load("expression_morphology_covariation.json")
    joint = _load("joint_checkpoint_readout.json")
    n_pe = {
        (t["given"], t["predicted"]): int(t["n_strains"])
        for t in joint["triangle"]
        if {t["given"], t["predicted"]} == {"proteome", "expression"}
    }
    ctx = pm["context"]
    # (given, predicted, all features, moving features, permuted null, shared strains)
    pairs: list[tuple[str, str, float, float | None, float | None, int]] = [
        (
            "proteome", "expression", ctx["proteome_to_expression_ridge_per_feature"],
            None, None, n_pe[("proteome", "expression")],
        ),
        (
            "expression", "proteome", ctx["expression_to_proteome_ridge_per_feature"],
            None, None, n_pe[("expression", "proteome")],
        ),
        (
            "proteome", "morphology",
            pm["proteome_to_morphology"]["all_features"]["median"],
            pm["proteome_to_morphology"]["moving_features"]["median"],
            pm["proteome_to_morphology"]["null_all"]["median"],
            pm["n_shared_strains"],
        ),
        (
            "morphology", "proteome",
            pm["morphology_to_proteome"]["all_proteins"]["median"], None,
            pm["morphology_to_proteome"]["null"]["median"], pm["n_shared_strains"],
        ),
        (
            "expression", "morphology",
            em["expression_to_morphology"]["all_features"]["median"],
            em["expression_to_morphology"]["moving_features"]["median"],
            em["expression_to_morphology"]["null_all"]["median"],
            em["n_shared_strains"],
        ),
        (
            "morphology", "expression",
            em["morphology_to_expression"]["all_genes"]["median"], None,
            em["morphology_to_expression"]["null"]["median"], em["n_shared_strains"],
        ),
    ]
    fig, ax = _figure("half")
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.25, top=0.76)
    w = 0.36
    for i, (_given, _pred, all_r, moving_r, null_r, n) in enumerate(pairs):
        ax.bar(i - w / 2, all_r, w, color=ORANGE, edgecolor="black", lw=0.5, zorder=2)
        top = all_r
        if moving_r is not None:
            ax.bar(
                i + w / 2, moving_r, w, color=RED, edgecolor="black", lw=0.5, zorder=2
            )
            top = max(top, moving_r)
        if null_r is not None:
            ax.plot([i - w, i + w], [null_r, null_r], color="black", lw=1.0, zorder=3)
        ax.text(i, top + 0.012, f"n = {n:,}", ha="center", va="bottom")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"{g}\n→\n{p}" for g, p, *_ in pairs])
    ax.set_xlim(-0.6, len(pairs) - 0.4)
    ax.set_ylim(-0.02, 0.42)
    ax.set_ylabel("held-out Pearson per feature")
    _grid(ax, 0.2, 0.1)
    _box(ax)
    _top_legend(
        fig,
        ax,
        [
            Patch(facecolor=ORANGE, edgecolor="black", label="all features"),
            Patch(
                facecolor=RED, edgecolor="black",
                label="moving features (morphology targets only)",
            ),
            Line2D([], [], color="black", lw=1.0, label="strain-permuted null"),
        ],
        ncol=2,
    )
    _save(fig, "triangle")


def panel_v19_paired() -> None:
    tests = _load("joint_checkpoint_readout.json")["v19"]["tests"]
    heads = [
        ("expression head", tests["H1a_expression_superiority"], ORANGE),
        ("proteome head", tests["H1b_proteome_noninferiority"], RED),
    ]
    fig, axes = _figure("half", ncols=2, sharey=True)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.2, top=0.9, wspace=0.08)
    for ax, (title, test, color) in zip(axes, heads):
        parts = sorted(test["per_partition"], key=int)
        diffs = [test["per_partition"][p] for p in parts]
        ax.bar(
            range(len(parts)), diffs, 0.7, color=color, edgecolor="black", lw=0.5,
            zorder=2,
        )
        ax.axhline(0.0, color="black", lw=0.8, zorder=3)
        margin = float(test["margin"])
        stat = (
            f"mean {test['mean_diff']:+.3f}, n = {test['n_partitions']}\n"
            f"{test['n_positive']} of {test['n_partitions']} above 0\n"
            f"one-sided t, p = {test['p_one_sided_t']:.2f}"
        )
        if margin > 0:
            ax.axhline(-margin, color="black", lw=0.8, ls="--", zorder=3)
            stat += f"\ndashed: margin {-margin:.3f}"
        ax.text(
            0.04, 0.96, stat, transform=ax.transAxes, ha="left", va="top", zorder=5,
            bbox={"facecolor": "white", "edgecolor": "black", "lw": 0.5, "pad": 2},
        )
        ax.set_title(title, pad=3)
        ax.set_xticks(range(len(parts)))
        ax.set_xticklabels(parts)
        ax.set_xlabel(SPLIT_LABEL)
        _grid(ax, 0.04, 0.02)
        _box(ax)
    axes[0].set_ylim(-0.07, 0.13)
    axes[0].set_ylabel("joint − single, Pearson")
    _save(fig, "v19_paired")


def panel_v20_partial() -> None:
    runs = _load("joint_checkpoint_readout.json")["v20_partial"]["runs"]
    arms = ["C_expr", "C_exprperm", "C_prot", "C_protperm"]
    labels = ["C_expr", "C_expr\nperm", "C_prot", "C_prot\nperm"]
    splits = sorted({int(r["split"]) for r in runs})
    colors = PLOT_PALETTE[: len(splits)]
    epochs = [int(r["last_epoch"]) for r in runs]
    fig, ax = _figure("third")
    # Two legend rows once more than two split seeds are on file; a third-width
    # panel holds two entries per row.
    legend_rows = (len(splits) + 1) // 2
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.2, top=0.86 - 0.07 * (legend_rows - 1))
    w = 0.8 / len(splits)
    for j, split in enumerate(splits):
        vals = [
            next(
                r["diff_same_epochs"]
                for r in runs
                if r["arm"] == arm and int(r["split"]) == split
            )
            for arm in arms
        ]
        ax.bar(
            np.arange(len(arms)) - 0.4 + w / 2 + j * w, vals, w, color=colors[j],
            edgecolor="black", hatch="////", lw=0.5, zorder=2,
            label=f"split seed {split}",
        )
    ax.axhline(0.0, color="black", lw=0.8, zorder=3)
    ax.text(
        0.04, 0.96, f"PARTIAL\nepochs {min(epochs)} to {max(epochs)}\nnot a result",
        transform=ax.transAxes, ha="left", va="top", fontsize=7, fontweight="bold",
        color="black", zorder=5,
        bbox={"facecolor": "white", "edgecolor": "black", "lw": 0.5, "pad": 2},
    )
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.02, 0.16)
    ax.set_ylabel("conditioned − control,\nsame epochs (Pearson)")
    _grid(ax, 0.04, 0.02)
    _box(ax)
    handles, _ = ax.get_legend_handles_labels()
    _top_legend(fig, ax, handles, ncol=min(len(splits), 2))
    _save(fig, "v20_partial")


def panel_v19_timing() -> None:
    v19 = _load("joint_checkpoint_readout.json")["v19"]
    complete = [int(p) for p in v19["complete_partitions"]]
    by = {(r["arm"], int(r["split"])): r for r in v19["runs"]}
    series = [
        ("K_prot", "loss_min_epoch", ORANGE, "v", "single head, loss minimum", -0.24),
        ("K_joint", "loss_min_epoch", RED, "v", "joint, loss minimum", -0.08),
        ("K_prot", "proteome_roll_max_epoch", ORANGE, "o", "single head, score peak", 0.08),
        ("K_joint", "proteome_roll_max_epoch", RED, "o", "joint, score peak", 0.24),
    ]
    fig, ax = _figure("half")
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.2, top=0.8)
    handles = []
    for arm, key, color, marker, label, dx in series:
        handles.append(
            ax.scatter(
                [p + dx for p in complete], [by[(arm, p)][key] for p in complete],
                s=14, marker=marker, color=color, edgecolor="black", lw=0.4, zorder=3,
                label=label,
            )
        )
    budget = max(int(by[("K_prot", p)]["last_epoch"]) for p in complete)
    ax.set_xticks(complete)
    ax.set_xlim(-0.6, max(complete) + 0.6)
    ax.set_ylim(0, budget * 1.04)
    ax.set_xlabel(SPLIT_LABEL)
    ax.set_ylabel("epoch (proteome head)")
    _grid(ax, 400, 200)
    _box(ax)
    _top_legend(fig, ax, handles, ncol=2)
    _save(fig, "v19_timing")


def panel_v19_per_seed() -> None:
    v19 = _load("joint_checkpoint_readout.json")["v19"]
    complete = [int(p) for p in v19["complete_partitions"]]
    by = {(r["arm"], int(r["split"])): r for r in v19["runs"]}
    heads = [
        ("expression head, epochs {} to {}", "K_expr", "expression"),
        ("proteome head, epochs {} to {}", "K_prot", "proteome"),
    ]
    fig, axes = _figure("half", ncols=2, sharey=True)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.2, top=0.8, wspace=0.08)
    handles = []
    for ax, (title, single_arm, head) in zip(axes, heads):
        single = [by[(single_arm, p)][f"{head}_window"] for p in complete]
        joint = [by[("K_joint", p)][f"{head}_window"] for p in complete]
        ax.vlines(complete, single, joint, color=GRAY, lw=0.6, zorder=2)
        handles = [
            ax.scatter(
                complete, single, s=14, color=ORANGE, edgecolor="black", lw=0.4,
                zorder=3, label="single head",
            ),
            ax.scatter(
                complete, joint, s=14, color=RED, edgecolor="black", lw=0.4, zorder=3,
                label="joint",
            ),
        ]
        ax.axhline(0.0, color="black", lw=0.8, zorder=1)
        ax.set_title(title.format(*v19["windows"][head]), pad=3)
        ax.set_xticks(complete)
        ax.set_xlim(-0.6, max(complete) + 0.6)
        ax.set_xlabel(SPLIT_LABEL)
        _grid(ax, 0.1, 0.05)
        _box(ax)
    axes[0].set_ylim(-0.02, 0.2)
    axes[0].set_ylabel("validation Pearson,\nwindow mean")
    _top_legend(fig, axes[0], handles, ncol=2)
    _save(fig, "v19_per_seed")


def _split_baselines(folder: str) -> dict[int, tuple[float, float]]:
    """split seed -> (best linear, best kNN) validation Pearson on that split."""
    out: dict[int, tuple[float, float]] = {}
    for path in sorted(glob.glob(osp.join(RESULTS, folder, "seed*.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        if rec["split"]["fold_test_into_train"]:
            continue
        scores = []
        for key in ("B2_bilinear", "B3_neighbor_average"):
            best = rec[key]["by_embedding"][rec[key]["best_embedding"]]
            scores.append(float(best["selected_on_val"]["val_pearson_per_feature"]))
        out[int(rec["split"]["split_seed"])] = (scores[0], scores[1])
    return out


def panel_baselines_by_seed(width_key: str) -> None:
    v19 = _load("joint_checkpoint_readout.json")["v19"]
    complete = {int(p) for p in v19["complete_partitions"]}
    by = {(r["arm"], int(r["split"])): r for r in v19["runs"]}
    heads = [
        ("expression", "K_expr", _split_baselines("expression_baselines_split")),
        ("proteome", "K_prot", _split_baselines("baselines_split_fig3_proteome")),
    ]
    seeds = [sorted(set(b) & complete) for _, _, b in heads]
    fig, axes = _figure(
        width_key, ncols=2, sharey=True,
        gridspec_kw={"width_ratios": [len(s) + 1 for s in seeds]},
    )
    left = 0.2 if width_key == "third" else 0.13
    fig.subplots_adjust(left=left, right=0.975, bottom=0.2, top=0.8, wspace=0.08)
    handles = []
    for ax, (head, arm, base), head_seeds in zip(axes, heads, seeds):
        model = [by[(arm, s)][f"{head}_window"] for s in head_seeds]
        series = [
            (model, ORANGE, "o", "CGT" if width_key == "third" else "CGT single head"),
            ([base[s][0] for s in head_seeds], RED, "s", "linear"),
            ([base[s][1] for s in head_seeds], PURPLE, "^", "kNN"),
        ]
        handles = [
            ax.plot(
                head_seeds, values, color=color, marker=marker, ms=3, lw=0.6,
                mec="black", mew=0.4, zorder=3, label=label,
            )[0]
            for values, color, marker, label in series
        ]
        ax.axhline(0.0, color="black", lw=0.8, zorder=1)
        ax.set_title(head, pad=3)
        step = 2 if width_key == "third" and len(head_seeds) > 6 else 1
        ax.set_xticks(head_seeds[::step])
        ax.set_xlim(min(head_seeds) - 0.6, max(head_seeds) + 0.6)
        _grid(ax, 0.1, 0.05)
        _box(ax)
    axes[0].set_ylim(-0.02, 0.2)
    axes[0].set_ylabel("validation Pearson\nper feature")
    fig.supxlabel(SPLIT_LABEL, fontsize=6, y=0.02)
    _top_legend(fig, axes[0], handles, ncol=3)
    _save(fig, f"baselines_by_seed_{width_key}")


def panel_graph_effect(width_key: str) -> None:
    pair = _load("graph_prior_probe.json")["pair_form"]
    strands = [("expression", ORANGE), ("proteome", RED), ("morphology", PURPLE)]
    graphs = list(pair["expression"]["graphs"])
    fig, axes = _figure(width_key, ncols=3, sharey=True)
    left = 0.3 if width_key == "third" else 0.15
    fig.subplots_adjust(left=left, right=0.975, bottom=0.2, top=0.8, wspace=0.1)
    y = np.arange(len(graphs))
    for ax, (strand, color) in zip(axes, strands):
        ax.set_title(strand, pad=3)
        ax.set_xlim(0.44, 0.7)
        ax.set_xticks([0.5, 0.6])
        _box(ax)
        if strand not in pair:
            ax.text(
                0.5, 0.5, "pair form\nnot in file", transform=ax.transAxes, ha="center",
                va="center",
            )
            continue
        ax.axvline(0.5, color="black", lw=0.5, zorder=1)
        ax.grid(axis="y", color="#CCCCCC", lw=0.4)
        cells = [pair[strand]["graphs"][g]["t1"] for g in graphs]
        ax.scatter(
            [c["auc_rewired"] for c in cells], y, s=12, facecolor="white",
            edgecolor="black", lw=0.6, zorder=3,
        )
        ax.scatter(
            [c["auc"] for c in cells], y, s=12, color=color, edgecolor="black", lw=0.4,
            zorder=4,
        )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([GRAPH_LABELS[g] for g in graphs])
    axes[0].set_ylim(len(graphs) - 0.4, -0.6)
    fig.supxlabel("AUC, adjacent against background pairs", fontsize=6, y=0.02)
    _top_legend(
        fig,
        axes[0],
        [
            Line2D([], [], marker="o", ls="", ms=3.5, mfc=GRAY, mec="black", mew=0.4,
                   label="graph"),
            Line2D([], [], marker="o", ls="", ms=3.5, mfc="white", mec="black",
                   mew=0.6, label="rewired control"),
        ],
        ncol=2,
    )
    _save(fig, f"graph_effect_{width_key}")


def panel_graph_retrieval() -> None:
    rounds = _load("graph_retrieval_baseline.json")["rounds"]
    strands = [("expression", "v13", ORANGE), ("proteome", "v14", RED)]
    graphs = [
        k for k in rounds["v13"]["summary"]["paired"]
        if "val/all/minus_rewired" in rounds["v13"]["summary"]["paired"][k]
    ]
    fig, axes = _figure("third", ncols=2, sharey=True)
    fig.subplots_adjust(left=0.3, right=0.965, bottom=0.2, top=0.86, wspace=0.12)
    y = np.arange(len(graphs))
    for ax, (strand, round_key, color) in zip(axes, strands):
        paired = rounds[round_key]["summary"]["paired"]
        cells = [paired[g]["val/all/minus_rewired"] for g in graphs]
        n = {int(c["n"]) for c in cells}
        ax.axvline(0.0, color="black", lw=0.5, zorder=1)
        ax.grid(axis="y", color="#CCCCCC", lw=0.4)
        ax.errorbar(
            [c["mean"] for c in cells], y, xerr=[c["sd"] for c in cells], fmt="o",
            ms=3, color=color, mec="black", mew=0.4, ecolor="black", elinewidth=0.5,
            zorder=3,
        )
        ax.set_title(f"{strand}, n = {n.pop()}", pad=3)
        ax.set_xlim(-0.04, 0.2)
        ax.set_xticks([0.0, 0.1])
        _box(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([GRAPH_LABELS[g] for g in graphs])
    axes[0].set_ylim(len(graphs) - 0.4, -0.6)
    fig.supxlabel("graph key − rewired key, Pearson", fontsize=6, y=0.02)
    _save(fig, "graph_retrieval")


def panel_covariation() -> None:
    cov = _load("proteome_expression_covariation.json", RESULTS_028)["gene_covariation"]
    rows = sorted(cov.items(), key=lambda kv: -kv[1]["spearman"])
    fig, ax = _figure("third")
    fig.subplots_adjust(left=0.43, right=0.965, bottom=0.26, top=0.93)
    y = np.arange(len(rows))
    ax.barh(
        y, [r["spearman"] for _, r in rows], 0.7, color=ORANGE, edgecolor="black",
        lw=0.5, zorder=2,
    )
    for i, (_, r) in enumerate(rows):
        ax.text(r["spearman"] + 0.015, i, f"n = {r['n_genes']:,}", ha="left", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [" /\n".join(PANEL_NAMES[p] for p in key.split("_vs_")) for key, _ in rows]
    )
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Spearman of gene-gene r\nbetween two panels")
    _grid(ax, 0.2, 0.1, axis="x")
    _box(ax)
    _save(fig, "covariation")


def _embedding_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = _load("baselines_embedding_study.json")["embeddings"]
    return rows


def _embedding_rank(row: dict[str, Any]) -> float:
    """Mean of the four validation means (expression and proteome, B2 and B3)."""
    return float(
        np.mean(
            [row[f"{m}_{b}_val"]["mean"] for m in ("expression", "proteome") for b in ("B2", "B3")]
        )
    )


def _reduced_embedding_set(rows: dict[str, dict[str, Any]]) -> list[str]:
    """Five rows by a fixed rule, each the top of its class by _embedding_rank."""

    def best(keys: list[str]) -> str:
        return max(keys, key=lambda k: _embedding_rank(rows[k]))

    single = [
        k
        for k, v in rows.items()
        if v["family"] in ("protein LM", "coding sequence", "regulatory DNA")
        and k != "codon_frequency"
    ]
    composite = [
        k for k, v in rows.items() if v["family"] == "composite" and k != MODEL_COMPOSITE
    ]
    control = [k for k, v in rows.items() if v["family"] == "control"]
    chosen = [best(single), MODEL_COMPOSITE, best(composite), "codon_frequency", best(control)]
    print(f"reduced embedding set: {chosen}")
    return chosen


def _embedding_panel(
    name: str, keys: list[str], width_key: str, height_mm: float, adjust: dict[str, float]
) -> None:
    rows = _embedding_rows()
    fig, axes = _figure(width_key, ncols=2, height_mm=height_mm, sharey=True)
    fig.subplots_adjust(**adjust)
    y = np.arange(len(keys))
    for ax, modality in zip(axes, ("expression", "proteome")):
        ax.axvline(0.0, color="black", lw=0.5, zorder=1)
        ax.grid(axis="y", color="#CCCCCC", lw=0.4)
        for baseline, marker, dy in (("B2", "s", -0.17), ("B3", "^", 0.17)):
            for i, key in enumerate(keys):
                cell = rows[key][f"{modality}_{baseline}_val"]
                ax.errorbar(
                    cell["mean"], i + dy, xerr=cell["sd"], fmt=marker, ms=3,
                    color=FAMILY_COLORS[rows[key]["family"]], mec="black", mew=0.4,
                    ecolor="black", elinewidth=0.5, zorder=3,
                )
        n = {rows[k][f"{modality}_B2_val"]["n"] for k in keys}
        ax.set_title(f"{modality}, n = {n.pop()}", pad=3)
        ax.set_xlim(-0.03, 0.17)
        ax.set_xticks([0.0, 0.1])
        _box(ax)
    axes[0].set_yticks(y)
    labels = axes[0].set_yticklabels([EMBEDDING_LABELS[k] for k in keys])
    for key, label in zip(keys, labels):
        if key == MODEL_COMPOSITE:
            label.set_fontweight("bold")
    axes[0].set_ylim(len(keys) - 0.4, -0.6)
    fig.supxlabel("validation Pearson per feature", fontsize=6, y=0.01)
    families = list(dict.fromkeys(rows[k]["family"] for k in keys))
    handles = [
        Line2D([], [], marker="s", ls="", ms=3.5, mfc="white", mec="black", mew=0.5,
               label="ridge"),
        Line2D([], [], marker="^", ls="", ms=3.5, mfc="white", mec="black", mew=0.5,
               label="kNN"),
    ] + [Patch(facecolor=FAMILY_COLORS[f], edgecolor="black", label=f) for f in families]
    _top_legend(fig, axes[0], handles, ncol=2 if width_key == "third" else 3, x=0.13)
    _save(fig, name)


def panel_embedding_reduced() -> None:
    rows = _embedding_rows()
    fig_keys = _reduced_embedding_set(rows)
    _embedding_panel(
        "embedding_reduced", fig_keys, "third", PANEL_HEIGHT_MM,
        {"left": 0.38, "right": 0.965, "bottom": 0.17, "top": 0.6, "wspace": 0.12},
    )


def panel_embedding_full() -> None:
    rows = _embedding_rows()
    order = list(FAMILY_COLORS)
    keys = sorted(rows, key=lambda k: (order.index(rows[k]["family"]), -_embedding_rank(rows[k])))
    _embedding_panel(
        "embedding_full", keys, "half", 96.0,
        {"left": 0.33, "right": 0.975, "bottom": 0.08, "top": 0.845, "wspace": 0.08},
    )


def panel_size_vs_score() -> None:
    board = pd.read_csv(osp.join(RESULTS, "round_leaderboards.csv"), low_memory=False)
    runs = board[board["strand"] == "expression"].dropna(
        subset=["total_param_count", "primary_roll_max"]
    )
    collapsed = runs["is_collapsed"].astype(str) == "True"
    v19 = _load("joint_checkpoint_readout.json")["v19"]["runs"]
    v19_params = {int(r["total_param_count"]) for r in v19 if r["arm"] == "K_expr"}
    assert len(v19_params) == 1
    v19_size = v19_params.pop()
    audit = _load("split_gene_overlap_audit.json")["fig3_proteome_full"]
    v19_train = [int(audit[s]["train"]["both"]) for s in sorted(audit)]
    r = float(
        np.corrcoef(np.log10(runs["total_param_count"]), runs["primary_roll_max"])[0, 1]
    )
    n_train = runs["n_train_supervised"].dropna()
    fig, ax = _figure("third")
    fig.subplots_adjust(left=0.2, right=0.965, bottom=0.2, top=0.74)
    ax.scatter(
        runs.loc[collapsed, "total_param_count"], runs.loc[collapsed, "primary_roll_max"],
        s=5, facecolor="white", edgecolor=GRAY, lw=0.4, zorder=2,
    )
    ax.scatter(
        runs.loc[~collapsed, "total_param_count"], runs.loc[~collapsed, "primary_roll_max"],
        s=5, color=ORANGE, edgecolor="black", lw=0.2, zorder=3,
    )
    ax.axvline(v19_size, color="black", lw=0.8, ls="--", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("best validation Pearson\n(rolling maximum)")
    ax.set_ylim(-0.05, 0.55)
    _grid(ax, 0.1, 0.05)
    _box(ax)
    ax.text(
        0.03, 0.97,
        f"n = {len(runs):,} runs\n"
        f"r(log10 parameters, score) = {r:+.3f}\n"
        f"training strains: median {int(n_train.median()):,}\n"
        f"({int(n_train.min()):,} to {int(n_train.max()):,});"
        f" v19: {min(v19_train):,} to {max(v19_train):,}\n"
        f"v19 CGT: {v19_size / 1e6:.2f} M parameters",
        transform=ax.transAxes, ha="left", va="top", zorder=5,
        bbox={"facecolor": "white", "edgecolor": "black", "lw": 0.5, "pad": 2},
    )
    _top_legend(
        fig,
        ax,
        [
            Line2D([], [], marker="o", ls="", ms=3, mfc=ORANGE, mec="black", mew=0.3,
                   label="run"),
            Line2D([], [], marker="o", ls="", ms=3, mfc="white", mec=GRAY, mew=0.5,
                   label="collapsed run"),
            Line2D([], [], color="black", lw=0.8, ls="--",
                   label="v19 CGT"),
        ],
        ncol=3,
    )
    _save(fig, "size_vs_score")


def _prototypical_seed() -> int:
    """The complete split seed whose joint-minus-single proteome window difference is
    closest to the median over complete split seeds; the lower seed on a tie."""
    test = _load("joint_checkpoint_readout.json")["v19"]["tests"]["H1b_proteome_noninferiority"]
    diffs = {int(k): float(v) for k, v in test["per_partition"].items()}
    median = float(np.median(list(diffs.values())))
    seed = min(diffs, key=lambda k: (abs(diffs[k] - median), k))
    print(f"prototypical split seed: {seed} (difference {diffs[seed]:+.4f}, median {median:+.4f})")
    return seed


def _curves_panel(name: str, head: str, single_arm: str) -> None:
    readout = _load("joint_checkpoint_readout.json")["v19"]
    seed = _prototypical_seed()
    curves = pd.read_csv(osp.join(RESULTS, "joint_checkpoint_curves.csv"))
    runs = {(r["arm"], int(r["split"])): r for r in readout["runs"]}
    key = f"val/{head}/pearson_per_feature"
    lo, hi = readout["windows"][head]
    fig, (ax_loss, ax_score) = _figure("half", nrows=2, height_mm=60.0, sharex=True)
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.14, top=0.86, hspace=0.12)
    handles = []
    for arm, color, fill, label in (
        (single_arm, ORANGE, FILL_ORANGE, "single head"),
        ("K_joint", RED, FILL_RED, "joint"),
    ):
        c = curves[(curves["arm"] == arm) & (curves["split"] == seed)].set_index("epoch")
        run = runs[(arm, seed)]
        ax_loss.plot(c.index, c["val/loss_roll5"], color=color, lw=0.8, zorder=3)
        score = c[key].dropna()
        smooth = score.rolling(5, center=True).mean()
        ax_score.plot(score.index, score, color=fill, lw=0.5, zorder=2)
        handles.append(ax_score.plot(smooth.index, smooth, color=color, lw=0.8, zorder=3,
                                     label=label)[0])
        e_loss = int(run["loss_min_epoch"])
        e_peak = int(run[f"{head}_roll_max_epoch"])
        ax_loss.scatter([e_loss], [c.loc[e_loss, "val/loss_roll5"]], s=16, marker="v",
                        color=color, edgecolor="black", lw=0.4, zorder=4)
        ax_score.scatter([e_peak], [smooth.loc[e_peak]], s=16, marker="o", color=color,
                         edgecolor="black", lw=0.4, zorder=4)
    for ax in (ax_loss, ax_score):
        ax.axvspan(lo, hi, color=FILL_GRAY, zorder=0)
        ax.axvline(lo, color=GRAY, lw=0.4, zorder=1)
        ax.axvline(hi, color=GRAY, lw=0.4, zorder=1)
        ax.grid(axis="y", color="#CCCCCC", lw=0.4)
        _box(ax)
    ax_loss.set_ylabel("validation loss\n(5-epoch mean)")
    ax_score.set_ylabel(f"{head} Pearson\nper feature")
    ax_score.set_xlabel(f"epoch, split seed {seed}")
    ax_score.set_xlim(0, 1200)
    ax_score.xaxis.set_major_locator(MultipleLocator(200))
    handles += [
        Line2D([], [], marker="v", ls="", ms=3.5, mfc=GRAY, mec="black", mew=0.4,
               label="loss minimum"),
        Line2D([], [], marker="o", ls="", ms=3.5, mfc=GRAY, mec="black", mew=0.4,
               label="score peak"),
        Patch(facecolor=FILL_GRAY, edgecolor=GRAY, label=f"window, {lo} to {hi}"),
    ]
    _top_legend(fig, ax_loss, handles, ncol=3)
    _save(fig, name)


def panel_curves_proteome() -> None:
    _curves_panel("curves_proteome", "proteome", "K_prot")


def panel_curves_expression() -> None:
    _curves_panel("curves_expression", "expression", "K_expr")


def main() -> None:
    panel_ceiling_vs_model()
    panel_triangle()
    panel_v19_paired()
    panel_v20_partial()
    panel_v19_timing()
    panel_v19_per_seed()
    for width_key in ("third", "half"):
        panel_baselines_by_seed(width_key)
    for width_key in ("third", "wide"):
        panel_graph_effect(width_key)
    panel_graph_retrieval()
    panel_covariation()
    panel_size_vs_score()
    panel_curves_proteome()
    panel_curves_expression()
    panel_embedding_reduced()
    panel_embedding_full()


if __name__ == "__main__":
    main()
