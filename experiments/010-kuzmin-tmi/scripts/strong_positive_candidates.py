# experiments/010-kuzmin-tmi/scripts/strong_positive_candidates.py
# [[experiments.010-kuzmin-tmi.scripts.strong_positive_candidates]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/strong_positive_candidates

"""How many inference_1 triples the model calls a STRONG positive interaction.

The panel design assumed a triple is worth building if its predicted $\\tau$ is
positive and the three checkpoints agree. Restricting to interactions the
literature would call strong is a much harder filter, and it changes the size of
the pool by two orders of magnitude.

**Where the threshold comes from.** Kuzmin 2018 defines the ordinary call as a
conjunction, magnitude and significance together, and its trigenic form is
one-sided negative. Baryshnikova 2010 adds a stringent tier that is
sign-asymmetric, $\\varepsilon < -0.12$ on the negative side and
$\\varepsilon > 0.16$ on the positive side. The positive arm of that tier,
$\\tau > +0.16$, is the strong cut used here. It is a digenic threshold applied to
a trigenic score, which is a real approximation and is flagged as one: no
published stringent tier for positive trigenic interactions exists.

**Two ways to count, and they differ by 2.6 times.** Ranking by the mean of the
three checkpoints admits triples one optimistic run carries. Requiring every
checkpoint to clear the cut is the conservative reading and is what a build list
should use, since two training runs share only 0.04 to 0.14 of their top 100 on
this space.

The gene gates are the same ones the panel selection applies: resolution through
the shared reconciler, membership in the model's gene set, no two genes on one
locus, and a floor on trigenic training records.

Outputs, under ``experiments/010-kuzmin-tmi/results/``:

    strong_positive_counts.csv      pool size at each cut, both rules
    strong_positive_triples.csv     the strong triples themselves, gated
    strong_positive_summary.json    concentration and tau-closure cost
"""

import glob
import itertools
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
BUILD_DIR = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
)
INFERENCE_DIR = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_1/inferred"
)

CHECKPOINTS = {"M01_lzs9pcj3": "0.4520", "M02_yv4r30bi": "0.4472", "M03_c7671wgj": "0.4619"}
TAGS = list(CHECKPOINTS)

# The published ordinary call, the stringent tier, its positive arm, and two
# stricter cuts so the shape of the decay is visible rather than asserted.
CUTS = [0.08, 0.12, 0.16, 0.20, 0.30, 0.50]
STRONG_CUT = 0.16
MIN_TRAIN_RECORDS = 200


def prediction_path(pearson: str) -> str:
    matches = [
        f
        for f in glob.glob(osp.join(INFERENCE_DIR, f"*Pearson={pearson}*.parquet"))
        if not f.endswith((".rank0", ".rank1", ".rank2", ".rank3"))
    ]
    assert len(matches) == 1, f"expected one file for Pearson={pearson}, got {matches}"
    return matches[0]


def load_space() -> tuple[np.ndarray, np.ndarray, list[str]]:
    preds: dict[str, np.ndarray] = {}
    triples: np.ndarray | None = None
    vocab: list[str] = []
    index_ref: np.ndarray | None = None
    for tag, pearson in CHECKPOINTS.items():
        table = pq.read_table(
            prediction_path(pearson),
            columns=["index", "gene1", "gene2", "gene3", "prediction"],
        )
        idx = table["index"].to_numpy()
        if index_ref is None:
            index_ref = idx
        else:
            assert np.array_equal(idx, index_ref), f"{tag} row order differs"
        if triples is None:
            col_of: dict[str, int] = {}
            cols = []
            for c in ("gene1", "gene2", "gene3"):
                s = table[c].to_pandas()
                for g in s.unique():
                    if g not in col_of:
                        col_of[g] = len(vocab)
                        vocab.append(g)
                cols.append(s.map(col_of).to_numpy(dtype=np.int32))
            triples = np.sort(np.stack(cols, axis=1), axis=1)
        preds[tag] = table["prediction"].to_numpy().astype(np.float64)
        del table
    assert triples is not None
    return triples, np.stack([preds[t] for t in TAGS], axis=1), vocab


def gene_support() -> dict[str, int]:
    with open(osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")) as f:
        return {g: len(v) for g, v in json.load(f).items()}


def resolve(vocab: list[str]) -> tuple[dict[str, str], set[str]]:
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    gene_set = set(genome.gene_set)
    return {g: genome.resolve_gene_name(g).systematic_name for g in vocab}, gene_set


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("loading inference_1 ...")
    triples, preds, vocab = load_space()
    worst = preds.min(axis=1)
    mean = preds.mean(axis=1)
    names = np.array(vocab, dtype=object)
    print(f"  {len(triples):,} triples over {len(vocab)} genes")

    rows = []
    for cut in CUTS:
        by_mean = mean > cut
        by_worst = worst > cut
        rows.append(
            {
                "cut": cut,
                "n_by_ensemble_mean": int(by_mean.sum()),
                "n_by_worst_checkpoint": int(by_worst.sum()),
                "genes_by_worst_checkpoint": int(np.unique(triples[by_worst]).size)
                if by_worst.any()
                else 0,
            }
        )
    counts = pd.DataFrame(rows)
    counts.to_csv(osp.join(RESULTS_DIR, "strong_positive_counts.csv"), index=False)
    print("\npool size against the predicted cut:")
    print(counts.to_string(index=False))

    # The strong pool, conservative rule.
    sel = np.where(worst > STRONG_CUT)[0]
    order = sel[np.argsort(worst[sel])[::-1]]
    support = gene_support()
    resolved, gene_set = resolve(
        sorted({str(g) for g in names[triples[order]].reshape(-1)})
    )

    recs = []
    for i in order:
        gs = sorted(str(g) for g in names[triples[i]])
        sysnames = [resolved[g] for g in gs]
        sup = [support.get(s, 0) for s in sysnames]
        fails = []
        if len(set(sysnames)) < 3:
            fails.append("two genes resolve to one locus")
        if any(s not in gene_set for s in sysnames):
            fails.append("gene outside the model gene set")
        if min(sup) < MIN_TRAIN_RECORDS:
            fails.append(f"min training records {min(sup)}")
        recs.append(
            {
                "triple": " + ".join(gs),
                "worst_checkpoint": float(worst[i]),
                "ensemble_mean": float(mean[i]),
                "spread": float(preds[i].max() - preds[i].min()),
                "min_train_records": int(min(sup)),
                "passes_gates": not fails,
                "gate_failures": "; ".join(fails),
            }
        )
    strong = pd.DataFrame(recs)
    strong.to_csv(osp.join(RESULTS_DIR, "strong_positive_triples.csv"), index=False)
    print(f"\nstrong pool at tau > +{STRONG_CUT}, every checkpoint above the cut:")
    print(strong.to_string(index=False))

    kept = strong[strong["passes_gates"]]
    kept_genes = sorted({g for t in kept["triple"] for g in t.split(" + ")})
    part = pd.Series(
        [g for t in kept["triple"] for g in t.split(" + ")]
    ).value_counts()
    doubles = sorted(
        {frozenset(p) for t in kept["triple"] for p in itertools.combinations(sorted(t.split(" + ")), 2)}
    )

    summary = {
        "strong_cut": STRONG_CUT,
        "n_strong_by_worst_checkpoint": int(len(strong)),
        "n_strong_by_ensemble_mean": int((mean > STRONG_CUT).sum()),
        "n_after_gene_gates": int(len(kept)),
        "genes_after_gates": kept_genes,
        "n_genes_after_gates": len(kept_genes),
        "top_gene": str(part.index[0]),
        "top_gene_count": int(part.iloc[0]),
        "triples_without_top_gene": int(
            (~kept["triple"].str.contains(str(part.index[0]), regex=False)).sum()
        ),
        "n_doubles_for_tau_closure": len(doubles),
        "n_strains_total": 1 + len(kept_genes) + len(doubles) + len(kept),
        "gate_min_train_records": MIN_TRAIN_RECORDS,
    }
    with open(osp.join(RESULTS_DIR, "strong_positive_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nafter gene gates:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\ngene participation in the gated strong set:")
    print(part.to_string())

    plot(counts, strong, part, summary)


def plot(
    counts: pd.DataFrame, strong: pd.DataFrame, part: pd.Series, summary: dict
) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(58.0))
    )

    # a. how fast the pool collapses with the threshold
    ax = axes[0]
    ax.plot(
        counts["cut"], counts["n_by_ensemble_mean"], marker="o", ms=2.5, linewidth=0.9,
        color=PLOT_PALETTE[0], markeredgecolor="black", markeredgewidth=0.3,
        label="ensemble mean above cut",
    )
    ax.plot(
        counts["cut"], counts["n_by_worst_checkpoint"], marker="o", ms=2.5, linewidth=0.9,
        color=PLOT_PALETTE[1], markeredgecolor="black", markeredgewidth=0.3,
        label="every checkpoint above cut",
    )
    ax.axvline(STRONG_CUT, color="black", linewidth=0.7, linestyle="--")
    ax.text(STRONG_CUT, ax.get_ylim()[1], " strong", fontsize=5, va="top")
    ax.set_yscale("log")
    ax.set_xlabel(r"Predicted $\tau$ cut")
    ax.set_ylabel("Triples of 4,370,595")
    ax.set_title("The pool collapses with the cut", fontsize=6)
    ax.legend(frameon=False, fontsize=4.5, loc="upper right")

    # b. the strong triples themselves
    ax = axes[1]
    show = strong.iloc[::-1]
    y = np.arange(len(show))
    colors = [
        PLOT_PALETTE[0] if ok else PLOT_PALETTE[5] for ok in show["passes_gates"]
    ]
    ax.barh(y, show["worst_checkpoint"], color=colors, edgecolor="black", linewidth=0.4, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(show["triple"], fontsize=3.8)
    ax.axvline(STRONG_CUT, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel(r"Predicted $\tau$, worst of three checkpoints")
    ax.set_title(
        f"All {len(strong)} strong triples (gray fails a gate)", fontsize=6
    )

    # c. the concentration that survives the cut
    ax = axes[2]
    x = np.arange(len(part))
    ax.bar(x, part.to_numpy(), color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.4, width=0.66)
    ax.set_xticks(x)
    ax.set_xticklabels(part.index, rotation=90, fontsize=4.5)
    ax.set_ylabel(f"Of {summary['n_after_gene_gates']} gated strong triples")
    ax.set_title("One gene carries the strong tier", fontsize=6)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.grid(which="both", linewidth=0.3, color="0.85")
        ax.set_axisbelow(True)

    fig.suptitle(
        "Restricting inference_1 to strong positive interactions, "
        r"$\tau > +0.16$",
        fontsize=6.5,
    )
    fig.tight_layout()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    stem = osp.join(IMAGE_DIR, "strong_positive_candidates")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"\nwrote {stem}.svg")


if __name__ == "__main__":
    main()
