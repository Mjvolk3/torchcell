# experiments/010-kuzmin-tmi/scripts/build_list_20_triples_corrected.py
# [[experiments.010-kuzmin-tmi.scripts.build_list_20_triples_corrected]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/build_list_20_triples_corrected
"""Corrected predictions for the 20 triples the bench was told to construct.

The build list is the ``capped`` strategy of
``experiments/W019-echo-crispr-array/results/triple_design_rank_sampling_selection.csv``,
25 doubles plus these 20 triples. Its ranking came from ``inference_3``, whose
gene indices were shifted by 28 and which scored any triple containing a gene
outside the model's gene space as a double. So the numbers the list was ordered
by are not predictions of those triples.

This rescores all 20 under the 6,607-gene index space the checkpoints were
trained on, with all three checkpoints, and plots the result against what the
list was chosen on.

``YLR312C-B`` is resolved to ``YLR313C``, its current systematic name; R64 records
one verified gene there with ``Alias: ['SPH1', 'YLR312C-B']``. That gene had no
index in the original run, which is why the six triples containing it took the
top six ranks: each was scored as the double formed by its other two genes.
"""

import glob
import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import score_010_checkpoints_directly as S  # noqa: E402

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
SELECTION_CSV = osp.join(
    EXPERIMENT_ROOT,
    "W019-echo-crispr-array",
    "results",
    "triple_design_rank_sampling_selection.csv",
)
STRATEGY = "capped"
BUILD_DIR = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
)

CHECKPOINT_GLOBS = {
    "M01_lzs9pcj3": "models/checkpoints/compute-3-3-2027905_*/lzs9pcj3-best-pearson-epoch=24-val/gene_interaction/*.ckpt",
    "M02_yv4r30bi": "models/checkpoints/compute-3-3-2027907_*/yv4r30bi-best-pearson-epoch=25-val/gene_interaction/*.ckpt",
    "M03_c7671wgj": "models/checkpoints/compute-3-3-2036902_*/c7671wgj-best-pearson-epoch=24-val/gene_interaction/*.ckpt",
}


def build_list() -> pd.DataFrame:
    sel = pd.read_csv(SELECTION_CSV)
    df = sel[sel["strategy"] == STRATEGY].copy()
    df["genes"] = df["triple"].str.split(r"\s*\+\s*", regex=True)
    assert df["genes"].map(len).eq(3).all(), "every build-list entry must be a triple"
    print(f"{STRATEGY} build list: {len(df)} triples")
    return df.sort_values("rank").reset_index(drop=True)


def load_checkpoint(cell_graph, embeddings, path: str, device):
    from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer

    model = CellGraphTransformer(
        cell_graph=cell_graph,
        graph_regularization_config=S.GRAPH_REG_CONFIG,
        perturbation_head_config=S.PERT_HEAD_CONFIG,
        graph_reg_lambda=1.0,
        node_embeddings=embeddings,
        learnable_embedding_config=S.LEARNABLE_EMBEDDING_CONFIG,
        **S.MODEL_KWARGS,
    ).to(device)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = {
        k[len("model.") :]: v
        for k, v in ck["state_dict"].items()
        if k.startswith("model.")
    }
    inc = model.load_state_dict(state, strict=False)
    assert not inc.unexpected_keys, inc.unexpected_keys
    return model


def label_spread() -> float:
    """Training-label standard deviation, the scale any prediction is read against."""
    label = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    return float(label["gene_interaction"].std(ddof=0))


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = build_list()

    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    gene_set = set(genome.gene_set)

    resolved: dict[str, str] = {}
    for g in sorted({g for genes in df["genes"] for g in genes}):
        r = genome.resolve_gene_name(g)
        assert r.systematic_name in gene_set, (
            f"build-list gene {g} resolves to {r.systematic_name}, outside the "
            f"model's gene set ({r.status.value}: {r.note})"
        )
        resolved[g] = r.systematic_name
        if r.systematic_name != g:
            print(f"  resolved {g} -> {r.systematic_name} ({r.status.value})")

    cell_graph, embeddings = S.build_cell_graph()
    node_to_idx = {g: i for i, g in enumerate(cell_graph["gene"].node_ids)}
    assert int(cell_graph["gene"].num_nodes) == len(gene_set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = np.array(
        [[node_to_idx[resolved[g]] for g in genes] for genes in df["genes"]],
        dtype=np.int64,
    )
    for tag, pattern in CHECKPOINT_GLOBS.items():
        matches = glob.glob(osp.join(DATA_ROOT, pattern))
        assert matches, f"no checkpoint for {tag}"
        model = load_checkpoint(cell_graph, embeddings, matches[0], device)
        df[tag] = S.score(model, cell_graph, idx, device)

    cols = list(CHECKPOINT_GLOBS)
    df["corrected_mean"] = df[cols].mean(axis=1)
    df["corrected_sd"] = df[cols].std(axis=1)
    df["corrected_min"] = df[cols].min(axis=1)
    df["corrected_max"] = df[cols].max(axis=1)
    df["sign_agree"] = (df[cols] > 0).sum(axis=1).isin([0, 3])
    # Training support, counted under the RESOLVED name. Counting under the name
    # the build list uses returns 0 for YLR312C-B, because the perturbed-gene
    # index is keyed on current systematic names and that string is an alias.
    # The same alias miss is what dropped the gene's index in inference_3.
    with open(
        osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")
    ) as f:
        gene_index = json.load(f)
    support = {g: len(gene_index.get(resolved[g], [])) for g in resolved}
    for g in sorted(support, key=lambda x: support[x]):
        print(
            f"  {g:12} -> {resolved[g]:10} trigenic training records {support[g]:>5d}"
        )
    df["min_train_records"] = df["genes"].map(lambda gs: min(support[g] for g in gs))
    df["has_untrained_gene"] = df["min_train_records"] == 0
    df = df.rename(columns={"prediction": "as_listed"})

    sd = label_spread()
    print(f"\ntraining-label sd {sd:.4f}")
    print(f"as listed  : {df['as_listed'].min():.4f} to {df['as_listed'].max():.4f}")
    print(
        f"corrected  : {df['corrected_mean'].min():+.4f} to "
        f"{df['corrected_mean'].max():+.4f}"
    )
    print(
        f"positive after correction: {int((df['corrected_mean'] > 0).sum())} of "
        f"{len(df)}; all three checkpoints agree on sign for "
        f"{int(df['sign_agree'].sum())}"
    )
    out = df.drop(columns=["genes"])
    path = osp.join(RESULTS_DIR, "build_list_20_triples_corrected.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    print(
        df.sort_values("corrected_mean", ascending=False)[
            ["triple", "as_listed", "corrected_mean", "corrected_sd", "sign_agree"]
        ].to_string(index=False)
    )
    plot_hist(df, sd)
    plot_ranked(df)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )


def save(fig, stem: str) -> None:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    p = osp.join(IMAGE_DIR, stem)
    fig.savefig(p + ".png", dpi=300)
    savefig_true_size_svg(fig, p + ".svg")
    print(f"wrote {p}.svg")
    plt.close(fig)


def plot_hist(df: pd.DataFrame, sd: float) -> None:
    """The histogram, both scorings on one axis so the shift is the message."""
    style()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(78.0)),
        sharex=True,
    )
    lo = min(df["corrected_mean"].min(), 0.0) - 0.05
    hi = df["as_listed"].max() + 0.05
    bins = np.linspace(lo, hi, 60)

    axes[0].hist(
        df["as_listed"],
        bins=bins,
        color=PLOT_PALETTE[5],
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].set_ylabel("Triples")
    axes[0].set_title(
        "Predicted trigenic interaction for the 20 triples on the build list",
        fontsize=6,
    )
    axes[0].text(
        0.98,
        0.9,
        "as listed (invalid, shifted index)",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=5.5,
    )

    axes[1].hist(
        df["corrected_mean"],
        bins=bins,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].set_ylabel("Triples")
    axes[1].set_xlabel(r"Predicted $\tau$")
    axes[1].text(
        0.98,
        0.9,
        "corrected, mean of 3 checkpoints",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=5.5,
    )

    for ax in axes:
        ax.axvline(0.0, color="black", linewidth=0.6)
        # The scale a prediction has to be read against.
        ax.axvspan(-sd, sd, color=PLOT_PALETTE[2], alpha=0.10, linewidth=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.grid(which="both", linewidth=0.3, color="0.85")
        ax.set_axisbelow(True)
    axes[1].text(
        sd,
        axes[1].get_ylim()[1] * 0.55,
        f"  $\\pm$1 label SD = {sd:.3f}",
        fontsize=5,
        va="center",
    )
    fig.tight_layout()
    save(fig, "build_list_20_triples_hist")


def plot_ranked(df: pd.DataFrame) -> None:
    """Every one of the 20 named, so the list can be re-ordered directly."""
    style()
    d = df.sort_values("corrected_mean", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(80.0))
    )
    y = np.arange(len(d))[::-1]
    colors = [
        PLOT_PALETTE[1] if flag else PLOT_PALETTE[0] for flag in d["has_untrained_gene"]
    ]
    ax.barh(
        y,
        d["corrected_mean"],
        xerr=[
            d["corrected_mean"] - d["corrected_min"],
            d["corrected_max"] - d["corrected_mean"],
        ],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 0.4, "capthick": 0.4, "capsize": 1.2},
    )
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([t.replace(" + ", "+") for t in d["triple"]], fontsize=5)
    ax.set_xlabel(r"Corrected predicted $\tau$, bars span the 3 checkpoints")
    ax.set_title(
        "The 20 build-list triples, re-ranked\n"
        f"red contains a gene with zero trigenic training records "
        f"({int(d['has_untrained_gene'].sum())} of {len(d)}); "
        "every gene here has a trained embedding row",
        fontsize=6,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(axis="x", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "build_list_20_triples_ranked")


if __name__ == "__main__":
    main()
