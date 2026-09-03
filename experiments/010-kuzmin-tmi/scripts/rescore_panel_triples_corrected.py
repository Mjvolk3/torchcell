# experiments/010-kuzmin-tmi/scripts/rescore_panel_triples_corrected.py
# [[experiments.010-kuzmin-tmi.scripts.rescore_panel_triples_corrected]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/rescore_panel_triples_corrected
"""Rescore every triple over the 010 construction panels under the correct index.

The panel-12 and panel-24 selections were made from ``inference_3``, whose gene
indices were shifted by 28: the genome gene set had dropped from 6,607 to 6,579
entries, the 28 missing mitochondrial open reading frames sort first, and the
checkpoint's embedding table still meant the 6,607-gene ordering. Every stored
prediction is therefore the model's answer for a different triple.

Strain construction has already started on those genes, so the practical
question is not which genes to pick but which triples among the picked genes the
model actually favors. This script answers that: it enumerates every triple over
the panel genes and scores each one with all three checkpoints under the correct
6,607-gene index space.

It also scores the same triples under the 6,579-gene space, which reproduces
what ``inference_3`` recorded, so the size of the error is measured rather than
asserted. Where a stored panel prediction exists the two are compared directly.

Three checkpoints rather than one, because the previous round established that
two training runs share only 0.39 to 0.47 of their top 100, so a single
checkpoint's ranking of a small design space is not the model's ranking.

Panels scored:

    constructed_10   the 10 genes strain construction actually started on
    panel_12         the inference_3 panel-12 k=200 selection those came from
    panel_24         the inference_3 panel-24 k=200 selection

Every triple over the union is scored once, then each panel reads its own rows.
"""

import glob
import json
import os
import os.path as osp
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import score_010_checkpoints_directly as S  # noqa: E402

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
EXP_DATA = osp.join(DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi")
BUILD_GENE_SET = osp.join(EXP_DATA, "001-small-build", "processed", "gene_set.json")

# The genes strain construction started on, read from the artifact that records
# them rather than retyped.
CONSTRUCTED_10_CSV = osp.join(RESULTS_DIR, "constructed_10_dmf_costanzo_kuzmin.csv")
SELECTION_CSV = osp.join(RESULTS_DIR, "inference_3", "gene_selection_results.csv")
SELECTION_K = 200

CHECKPOINT_GLOBS = {
    "M01_lzs9pcj3": "models/checkpoints/compute-3-3-2027905_*/lzs9pcj3-best-pearson-epoch=24-val/gene_interaction/*.ckpt",
    "M02_yv4r30bi": "models/checkpoints/compute-3-3-2027907_*/yv4r30bi-best-pearson-epoch=25-val/gene_interaction/*.ckpt",
    "M03_c7671wgj": "models/checkpoints/compute-3-3-2036902_*/c7671wgj-best-pearson-epoch=24-val/gene_interaction/*.ckpt",
}


def panels() -> dict[str, list[str]]:
    built = pd.read_csv(CONSTRUCTED_10_CSV)
    constructed = sorted(set(built["gene1"]) | set(built["gene2"]))

    sel = pd.read_csv(SELECTION_CSV)
    out = {"constructed_10": constructed}
    for size in (12, 24):
        row = sel[(sel["panel_size"] == size) & (sel["k"] == SELECTION_K)]
        if row.empty:
            continue
        out[f"panel_{size}"] = sorted(
            g.strip() for g in row.iloc[0]["selected_genes"].split(",")
        )
    for name, genes in out.items():
        print(f"{name}: {len(genes)} genes")
    return out


def index_maps(
    genes: list[str],
) -> tuple[dict[str, int], dict[str, int], dict[str, str], dict[str, str]]:
    """Index spaces plus the name resolution the panel needs.

    Returns the correct 6,607-gene map, the 6,579-gene map ``inference_3`` used,
    a map from panel name to the systematic name that can be scored, and the
    panel names that cannot be scored at all with the reason.

    Five panel genes are in neither index space. Three are outdated systematic
    names that resolve to current genes. Two are typed ``pseudogene`` in R64, and
    ``gene_set`` is built from ``features_of_type("gene")``, so they fall outside
    the model's index space and no embedding row was ever learned for them. They
    are real loci with SGD records, Costanzo measured deletion strains for both,
    and one of them is SDC25. Their absence here is a property of how we build
    the gene set, not of the genes.
    """
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    correct = {g: i for i, g in enumerate(sorted(genome.gene_set))}
    with open(BUILD_GENE_SET) as f:
        shifted = {g: i for i, g in enumerate(sorted(json.load(f)))}
    print(f"correct index space {len(correct)} genes, shifted {len(shifted)}")

    resolved: dict[str, str] = {}
    unscoreable: dict[str, str] = {}
    for g in genes:
        if g in correct:
            resolved[g] = g
            continue
        r = genome.resolve_gene_name(g)
        if r.systematic_name in correct:
            resolved[g] = r.systematic_name
            print(f"  resolved {g} -> {r.systematic_name} ({r.status.value})")
        else:
            unscoreable[g] = f"{r.status.value}: {r.note}"
            print(f"  OUTSIDE GENE SET {g}: {unscoreable[g]}")

    absent_shifted = [g for g in genes if g not in shifted]
    if absent_shifted:
        print(
            f"{len(absent_shifted)} panel genes had no index in the shifted "
            f"space, so inference_3 dropped them and scored those triples as "
            f"doubles or singles: {absent_shifted}"
        )
    return correct, shifted, resolved, unscoreable


def load_checkpoint(cell_graph, embeddings, path: str, device) -> object:
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
    incompatible = model.load_state_dict(state, strict=False)
    assert not incompatible.unexpected_keys, incompatible.unexpected_keys
    unresolved = [
        k
        for k in incompatible.missing_keys
        if not k.startswith("perturbation_transform.")
    ]
    assert not unresolved, unresolved
    return model


def stored_panel_predictions() -> pd.DataFrame:
    """What inference_3 recorded for the panel triples it did score."""
    frames = []
    for path in sorted(
        glob.glob(
            osp.join(RESULTS_DIR, "inference_3", "constructible_triples_*.parquet")
        )
    ):
        d = pd.read_parquet(path)
        d["source"] = osp.basename(path)
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["triple", "stored_prediction"])
    d = pd.concat(frames, ignore_index=True)
    d["triple"] = [
        "+".join(sorted([a, b, c]))
        for a, b, c in zip(d["gene1"], d["gene2"], d["gene3"])
    ]
    return (
        d.groupby("triple", as_index=False)["prediction"]
        .first()
        .rename(columns={"prediction": "stored_prediction"})
    )


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    panel_genes = panels()
    union = sorted({g for genes in panel_genes.values() for g in genes})
    print(f"union of panel genes: {len(union)}")

    correct, shifted, resolved, unscoreable = index_maps(union)
    scoreable = [g for g in union if g in resolved]
    print(f"scoreable panel genes: {len(scoreable)} of {len(union)}")

    triples = [tuple(t) for t in combinations(scoreable, 3)]
    print(f"triples over the scoreable union: {len(triples)}")
    df = pd.DataFrame(triples, columns=["gene1", "gene2", "gene3"])
    df["triple"] = ["+".join(t) for t in triples]
    df["systematic"] = ["+".join(sorted(resolved[g] for g in t)) for t in triples]

    idx_correct = np.array(
        [
            [correct[resolved[a]], correct[resolved[b]], correct[resolved[c]]]
            for a, b, c in triples
        ],
        dtype=np.int64,
    )
    # A gene absent from the shifted space simply produced fewer perturbation
    # indices in the original run. Scoring those rows here would not reproduce
    # what happened, so they are left out of the shifted column.
    shifted_ok = np.array([all(g in shifted for g in t) for t in triples], dtype=bool)
    idx_shifted = np.array(
        [
            [shifted[a], shifted[b], shifted[c]]
            for a, b, c in triples
            if all(g in shifted for g in (a, b, c))
        ],
        dtype=np.int64,
    )

    cell_graph, embeddings = S.build_cell_graph()
    n_nodes = int(cell_graph["gene"].num_nodes)
    print(f"cell graph gene nodes: {n_nodes}")
    assert n_nodes == len(correct), (
        "the cell graph must be built on the same gene set as the correct index "
        f"map; got {n_nodes} nodes against {len(correct)} genes"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for tag, pattern in CHECKPOINT_GLOBS.items():
        matches = glob.glob(osp.join(DATA_ROOT, pattern))
        assert matches, f"no checkpoint for {tag} at {pattern}"
        model = load_checkpoint(cell_graph, embeddings, matches[0], device)
        df[f"corrected_{tag}"] = S.score(model, cell_graph, idx_correct, device)
        col = np.full(len(triples), np.nan)
        col[shifted_ok] = S.score(model, cell_graph, idx_shifted, device)
        df[f"asrun_{tag}"] = col
        print(
            f"{tag}: corrected mean {df[f'corrected_{tag}'].mean():+.5f} "
            f"sd {df[f'corrected_{tag}'].std():.5f}"
        )

    corrected_cols = [f"corrected_{t}" for t in CHECKPOINT_GLOBS]
    asrun_cols = [f"asrun_{t}" for t in CHECKPOINT_GLOBS]
    df["corrected_mean"] = df[corrected_cols].mean(axis=1)
    df["corrected_sd"] = df[corrected_cols].std(axis=1)
    df["asrun_mean"] = df[asrun_cols].mean(axis=1)

    # Confirm the as-run column really is what inference_3 recorded.
    stored = stored_panel_predictions()
    df = df.merge(stored, on="triple", how="left")
    have = df["stored_prediction"].notna() & df["asrun_M03_c7671wgj"].notna()
    if have.any():
        d = (
            df.loc[have, "asrun_M03_c7671wgj"] - df.loc[have, "stored_prediction"]
        ).abs()
        print(
            f"\nas-run column vs the stored inference_3 predictions, "
            f"n={int(have.sum())}: max |diff| {d.max():.2e}, mean {d.mean():.2e}"
        )

    rows = []
    unsc_rows = []
    for name, genes in panel_genes.items():
        for g in genes:
            if g in unscoreable:
                unsc_rows.append({"panel": name, "gene": g, "reason": unscoreable[g]})
        gs = {g for g in genes if g in resolved}
        sub = df[df.apply(lambda r: {r.gene1, r.gene2, r.gene3} <= gs, axis=1)].copy()
        sub = sub.sort_values("corrected_mean", ascending=False).reset_index(drop=True)
        sub["corrected_rank"] = np.arange(1, len(sub) + 1)
        sub["asrun_rank"] = sub["asrun_mean"].rank(ascending=False, method="min")
        sub["panel"] = name
        rows.append(sub)
        top = sub.head(10)
        overlap10 = len(
            set(sub.nlargest(10, "corrected_mean")["triple"])
            & set(sub.nlargest(10, "asrun_mean")["triple"])
        )
        r = sub[["corrected_mean", "asrun_mean"]].corr().iloc[0, 1]
        print(f"\n=== {name}: {len(sub)} triples ===")
        print(f"corrected vs as-run Pearson {r:+.4f}; top-10 overlap {overlap10}/10")
        print(
            top[
                ["triple", "corrected_mean", "corrected_sd", "asrun_mean", "asrun_rank"]
            ].to_string(index=False)
        )

    out = pd.concat(rows, ignore_index=True)
    path = osp.join(RESULTS_DIR, "panel_triples_rescored.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")

    unsc = pd.DataFrame(unsc_rows)
    upath = osp.join(RESULTS_DIR, "panel_genes_unscoreable.csv")
    unsc.to_csv(upath, index=False)
    print(f"wrote {upath}")
    if not unsc.empty:
        print("\npanel genes the model cannot score:")
        print(unsc.to_string(index=False))

    rename = pd.DataFrame(
        [{"panel_name": k, "systematic_name": v} for k, v in resolved.items() if k != v]
    )
    rpath = osp.join(RESULTS_DIR, "panel_genes_renamed.csv")
    rename.to_csv(rpath, index=False)
    print(f"wrote {rpath}")
    plot(out)


def plot(out: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    from torchcell.utils import (
        PANEL_WIDTHS_MM,
        PLOT_PALETTE,
        apply_paper_style,
        mm_to_in,
        savefig_true_size_svg,
    )

    apply_paper_style()
    names = [n for n in out["panel"].unique()]
    fig, axes = plt.subplots(
        1, len(names), figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(58.0))
    )
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        sub = out[out["panel"] == name]
        ax.errorbar(
            sub["asrun_mean"],
            sub["corrected_mean"],
            yerr=sub["corrected_sd"],
            fmt="o",
            ms=2,
            mfc=PLOT_PALETTE[0],
            mec="black",
            mew=0.3,
            ecolor=PLOT_PALETTE[5],
            elinewidth=0.4,
            capsize=0,
            linestyle="none",
        )
        lo = float(np.nanmin([sub["asrun_mean"].min(), sub["corrected_mean"].min()]))
        hi = float(np.nanmax([sub["asrun_mean"].max(), sub["corrected_mean"].max()]))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.5, linestyle=":")
        ax.set_title(f"{name} ({len(sub)} triples)")
        ax.set_xlabel("As run, shifted index")
        ax.set_ylabel("Corrected, 3-checkpoint mean")
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.grid(which="both", linewidth=0.3, color="0.85")
        ax.set_axisbelow(True)
    fig.tight_layout()

    stem = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi", "panel_triples_rescored")
    os.makedirs(osp.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")


if __name__ == "__main__":
    main()
