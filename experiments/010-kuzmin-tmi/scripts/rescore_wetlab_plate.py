# experiments/010-kuzmin-tmi/scripts/rescore_wetlab_plate.py
# [[experiments.010-kuzmin-tmi.scripts.rescore_wetlab_plate]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/rescore_wetlab_plate
"""Score every triple over the strains actually on the wet-lab plate.

The panel handed to the bench is not the model-selected panel-12. It is read
here from the Echo transfer report of the plating run itself
(``experiments/W019-echo-crispr-array/data/run3_2026-07-23/P1_transfer_report.csv``),
so the gene list comes from what was physically plated rather than from a note.

Two things have to be fixed before the model can be asked anything about it.

**The plate carries one locus twice.** ``YLR312C-B`` and ``SPH1`` are separate
wells, and R64 records ``YLR313C`` with ``Alias: ['SPH1', 'YLR312C-B']``, one
verified gene, SGD:S000004305. Twelve strain labels are eleven distinct loci.
Every name is therefore resolved through
``SCerevisiaeGenome.resolve_gene_name`` before indexing, and the duplicate is
collapsed with both labels reported.

**The predictions the panel was chosen from are invalid.** Those came from
``inference_3``, whose gene indices were shifted by 28 and which silently scored
triples containing an unindexable gene as doubles. This script scores under the
6,607-gene index space the checkpoints were trained on, with all three
checkpoints rather than one, since two training runs share only 0.39 to 0.47 of
their top 100.

Outputs a ranked triple table with the spread across checkpoints, and the same
for the doubles, since doubles are what the assay builds first.
"""

import glob
import os
import os.path as osp
import re
import sys
from itertools import combinations

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

TRANSFER_REPORT = osp.join(
    EXPERIMENT_ROOT,
    "W019-echo-crispr-array",
    "data",
    "run3_2026-07-23",
    "P1_transfer_report.csv",
)
# Not strains: the wild-type control and the blank wells.
NON_STRAIN = {"BY4741", "Blank_media", "Sample Name", ""}

CHECKPOINT_GLOBS = {
    "M01_lzs9pcj3": "models/checkpoints/compute-3-3-2027905_*/lzs9pcj3-best-pearson-epoch=24-val/gene_interaction/*.ckpt",
    "M02_yv4r30bi": "models/checkpoints/compute-3-3-2027907_*/yv4r30bi-best-pearson-epoch=25-val/gene_interaction/*.ckpt",
    "M03_c7671wgj": "models/checkpoints/compute-3-3-2036902_*/c7671wgj-best-pearson-epoch=24-val/gene_interaction/*.ckpt",
}


def plate_strains() -> list[str]:
    """The strain labels actually transferred, read from the Echo report."""
    labels: list[str] = []
    with open(TRANSFER_REPORT) as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) > 13:
                labels.append(parts[12].strip())
    seen = sorted({label for label in labels if label not in NON_STRAIN})
    print(f"strain labels on the plate: {len(seen)}")
    for label in seen:
        print(f"  {label}")
    return seen


def resolve(labels: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Map each plate label to a systematic name, and group labels per locus."""
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    gene_set = set(genome.gene_set)

    label_to_gene: dict[str, str] = {}
    for label in labels:
        r = genome.resolve_gene_name(label)
        assert r.systematic_name in gene_set, (
            f"plate strain {label} resolves to {r.systematic_name}, which is not "
            f"in the model's gene set ({r.status.value}: {r.note}). It cannot be "
            "scored and must be reported to the bench rather than silently dropped."
        )
        label_to_gene[label] = r.systematic_name
        if r.systematic_name != label:
            print(f"  {label} -> {r.systematic_name} ({r.status.value})")

    per_locus: dict[str, list[str]] = {}
    for label, gene in label_to_gene.items():
        per_locus.setdefault(gene, []).append(label)
    dups = {g: ls for g, ls in per_locus.items() if len(ls) > 1}
    if dups:
        print("\nSAME LOCUS PLATED UNDER MORE THAN ONE LABEL:")
        for gene, ls in dups.items():
            print(f"  {gene}: {', '.join(sorted(ls))}")
    print(f"\n{len(labels)} strain labels resolve to {len(per_locus)} distinct loci")
    return label_to_gene, per_locus


@torch.no_grad()
def score_any_order(model, cell_graph, idx: np.ndarray, device) -> np.ndarray:
    """Predictions for [n, k] index rows, for any k.

    ``score_010_checkpoints_directly.score`` hardcodes three perturbations per
    record. The doubles here need k=2, so the batch assignment is built from the
    row width instead.
    """
    model.eval()
    n_rows, k = idx.shape
    out = np.empty(n_rows, dtype=np.float64)
    for start in range(0, n_rows, S.BATCH):
        chunk = idx[start : start + S.BATCH]
        n = chunk.shape[0]
        pert = torch.from_numpy(chunk.reshape(-1)).to(device)
        batch_assign = torch.arange(n, device=device).repeat_interleave(k)
        batch = {
            "gene": type(
                "G",
                (),
                {
                    "perturbation_indices": pert,
                    "perturbation_indices_batch": batch_assign,
                },
            )()
        }
        preds, _ = model(cell_graph, batch, return_attention=False)
        out[start : start + n] = preds.squeeze(-1).float().cpu().numpy()
    return out * S.NORM_STD + S.NORM_MEAN


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


def label_for(gene: str, per_locus: dict[str, list[str]]) -> str:
    """Prefer a readable standard name when the plate used one."""
    labels = sorted(per_locus[gene])
    named = [x for x in labels if not re.fullmatch(r"Y[A-P][LR]\d{3}[CW](-[A-Z])?", x)]
    return named[0] if named else labels[0]


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    labels = plate_strains()
    label_to_gene, per_locus = resolve(labels)
    genes = sorted(per_locus)

    cell_graph, embeddings = S.build_cell_graph()
    n_nodes = int(cell_graph["gene"].num_nodes)
    print(f"cell graph gene nodes: {n_nodes}")
    node_ids = list(cell_graph["gene"].node_ids)
    node_to_idx = {g: i for i, g in enumerate(node_ids)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for tag, pattern in CHECKPOINT_GLOBS.items():
        matches = glob.glob(osp.join(DATA_ROOT, pattern))
        assert matches, f"no checkpoint for {tag}"
        models[tag] = load_checkpoint(cell_graph, embeddings, matches[0], device)

    frames = {}
    for order, name in ((2, "doubles"), (3, "triples")):
        combos = [tuple(c) for c in combinations(genes, order)]
        idx = np.array([[node_to_idx[g] for g in c] for c in combos], dtype=np.int64)
        df = pd.DataFrame(
            {
                "order": order,
                "systematic": ["+".join(c) for c in combos],
                "plate_label": [
                    "+".join(label_for(g, per_locus) for g in c) for c in combos
                ],
            }
        )
        for tag, model in models.items():
            df[tag] = score_any_order(model, cell_graph, idx, device)
        cols = list(CHECKPOINT_GLOBS)
        df["mean"] = df[cols].mean(axis=1)
        df["sd"] = df[cols].std(axis=1)
        df["min"] = df[cols].min(axis=1)
        df["max"] = df[cols].max(axis=1)
        df = df.sort_values("mean", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        # A sign flip between checkpoints means the three runs do not even agree
        # on the direction of the interaction.
        df["sign_agree"] = (df[cols] > 0).sum(axis=1).isin([0, 3])
        # The 010 model saw only 3-perturbation records. A 2-perturbation input
        # is outside its training distribution and its output there is not a
        # calibrated tau. This is also why the corrupted inference_3 run's
        # collapsed doubles scored around 0.71, an order of magnitude past the
        # label's own spread.
        df["in_training_distribution"] = order == 3
        frames[name] = df
        print(f"\n=== {name}: {len(df)} combinations over {len(genes)} loci ===")
        if order != 3:
            print(
                "  OUT OF DISTRIBUTION: the 010 model was trained on triples "
                "only, so these are diagnostic, not tau predictions"
            )
        print(
            df.head(12)[
                ["rank", "plate_label", "mean", "sd", "min", "max", "sign_agree"]
            ].to_string(index=False)
        )
        print(
            f"checkpoints agree on sign for {int(df['sign_agree'].sum())} of "
            f"{len(df)} ({df['sign_agree'].mean():.0%})"
        )

    out = pd.concat(frames.values(), ignore_index=True)
    path = osp.join(RESULTS_DIR, "wetlab_plate_rescored.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    plot(frames["triples"], frames["doubles"], len(genes))


def plot(triples: pd.DataFrame, doubles: pd.DataFrame, n_genes: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )

    # Panel 1: every triple ranked, with the spread across the three checkpoints.
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(70.0))
    )
    x = np.arange(len(triples))
    ax.fill_between(
        x,
        triples["min"],
        triples["max"],
        color=PLOT_PALETTE[5],
        alpha=0.35,
        linewidth=0,
        label="checkpoint range",
    )
    ax.plot(x, triples["mean"], color=PLOT_PALETTE[0], linewidth=0.8, label="mean")
    ax.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel(f"Triple, ranked ({len(triples)} over {n_genes} loci)")
    ax.set_ylabel(r"Predicted trigenic interaction $\tau$")
    ax.legend(frameon=False, loc="upper right")
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    fig.tight_layout()
    stem = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi", "wetlab_plate_triples_ranked")
    os.makedirs(osp.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")

    # Panel 2: the top and bottom triples named, so the bench can read them off.
    top = pd.concat([triples.head(15), triples.tail(15)])
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(95.0))
    )
    y = np.arange(len(top))[::-1]
    colors = [PLOT_PALETTE[0] if v > 0 else PLOT_PALETTE[1] for v in top["mean"]]
    ax.barh(
        y,
        top["mean"],
        xerr=[top["mean"] - top["min"], top["max"] - top["mean"]],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 0.4, "capthick": 0.4, "capsize": 1.2},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(top["plate_label"], fontsize=5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel(r"Predicted $\tau$, mean of 3 checkpoints, bars span the 3")
    ax.set_title("Top 15 and bottom 15 triples on the plate")
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(axis="x", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    fig.tight_layout()
    stem = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi", "wetlab_plate_triples_named")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")

    # Panel 3: the doubles, which is what the assay builds first.
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(80.0))
    )
    y = np.arange(len(doubles))[::-1]
    ax.barh(
        y,
        doubles["mean"],
        xerr=[doubles["mean"] - doubles["min"], doubles["max"] - doubles["mean"]],
        color=[PLOT_PALETTE[0] if v > 0 else PLOT_PALETTE[1] for v in doubles["mean"]],
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 0.4, "capthick": 0.4, "capsize": 1.2},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(doubles["plate_label"], fontsize=5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel(r"Model output on a 2-gene input, not a calibrated $\tau$")
    ax.set_title(
        "All plate doubles, OUT OF DISTRIBUTION\n(this model was trained on triples only)"
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(axis="x", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    fig.tight_layout()
    stem = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi", "wetlab_plate_doubles")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")


if __name__ == "__main__":
    main()
