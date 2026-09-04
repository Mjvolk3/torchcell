# experiments/010-kuzmin-tmi/scripts/screen_diversity_audit.py
# [[experiments.010-kuzmin-tmi.scripts.screen_diversity_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/screen_diversity_audit

"""A record count is the wrong support statistic; distinct screens is the right one.

The panel selection gated genes on trigenic training records, with a floor of
200, because the previous panel's cliff ran 519 down to 10 down to 0. That gate
passes ``YER059W`` easily at 1,097 records. Every one of those records comes from
a single Kuzmin screen.

Kuzmin crosses a query DOUBLE mutant against an array of single mutants, so a
gene's records are grouped by the query double they were measured under. A gene
that is an array gene appears against hundreds of different query doubles; a
query gene appears only under its own. ``YER059W`` sits at the extreme of that:
one query double, ``YER059W + YIL050W``, and 1,097 array genes crossed against
it. The model has never seen ``YER059W`` in any combination not containing
``YIL050W``.

That matters because the panel's tier A pairs ``YER059W`` with genes that were
array genes in exactly that screen, and their measured $\\tau$ there runs +0.43 to
+0.80. Predicting a triple that swaps ``YIL050W`` out for one of those partners
is an extrapolation off one screen rather than a three-way inference, and the
predictions track the memorized values closely enough to say so.

This measures three things and adds the gate the record floor should have been:

    screen diversity   distinct query doubles each gene was measured under
    memorization       correlation between the model's prediction for a
                       YER059W triple and the measured screen values of its two
                       partners
    gate impact        which panel genes a distinct-screen floor removes

Outputs, under ``experiments/010-kuzmin-tmi/results/``:

    screen_diversity_per_gene.csv     every gene, records against screens
    screen_diversity_panel.csv        the panel genes, with the verdict
    screen_diversity_summary.json     the memorization correlations
"""

import glob
import json
import os
import os.path as osp
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from scipy.stats import pearsonr

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
QUERY_PAIR_MIN_COUNT = 5
HUB = "YER059W"
PANEL = [
    "YDR423C", "YER059W", "YGL060W", "YGR121C", "YHR003C",
    "YHR204W", "YJR082C", "YKR028W", "YLR258W", "YPL181W",
]
# A gene must have been measured under this many distinct query doubles for its
# behavior to be separable from any one screen's offset.
MIN_DISTINCT_SCREENS = 50


def training_structure():
    """Triples, labels, gene names, and each record's query double."""
    label_df = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    label_df = label_df.sort_values("index")
    tau = label_df["gene_interaction"].to_numpy()
    row_of = {int(r): i for i, r in enumerate(label_df["index"].to_numpy())}
    with open(osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")) as f:
        gene_index = json.load(f)
    names = sorted(gene_index)
    col = {g: j for j, g in enumerate(names)}
    rows = np.full((len(tau), 3), -1, dtype=np.int32)
    fill = np.zeros(len(tau), dtype=np.int8)
    for g, ids in gene_index.items():
        c = col[g]
        for r in ids:
            i = row_of[int(r)]
            rows[i, fill[i]] = c
            fill[i] += 1
    assert (fill == 3).all()
    rows = np.sort(rows, axis=1)

    base = len(names) + 1
    a, b, c = (rows[:, j].astype(np.int64) for j in range(3))
    pairs = np.stack([a * base + b, a * base + c, b * base + c], axis=1)
    keys, counts = np.unique(pairs.reshape(-1), return_counts=True)
    recurring = set(keys[counts >= QUERY_PAIR_MIN_COUNT].tolist())
    query = np.array(
        [next(int(p) for p in r if int(p) in recurring) for r in pairs], dtype=np.int64
    )
    return rows, tau, names, col, base, recurring, query


def per_gene(rows, names, query) -> pd.DataFrame:
    """Records and distinct screens per gene."""
    recs: list[dict[str, object]] = []
    for c, g in enumerate(names):
        m = (rows == c).any(axis=1)
        n = int(m.sum())
        if n == 0:
            continue
        screens = len(set(query[m].tolist()))
        recs.append(
            {
                "gene": g,
                "n_records": n,
                "n_distinct_screens": screens,
                "records_per_screen": n / screens,
            }
        )
    return pd.DataFrame(recs).sort_values("n_distinct_screens")


def memorization(rows, tau, names, col, base, query) -> dict[str, object]:
    """Do predictions for hub triples track the hub screen's measured values?"""
    hub = col[HUB]
    m = (rows == hub).any(axis=1)
    hub_screens = set(query[m].tolist())
    assert len(hub_screens) == 1, f"{HUB} spans {len(hub_screens)} screens"
    qd = next(iter(hub_screens))
    partner = qd // base if qd % base == hub else qd % base

    # Measured tau of each array gene against that one query double.
    in_screen: dict[int, float] = {}
    for r, v in zip(rows[m], tau[m]):
        rest = [int(x) for x in r if int(x) not in (hub, partner)]
        if len(rest) == 1:
            in_screen[rest[0]] = float(v)

    preds: dict[str, np.ndarray] = {}
    triples = None
    vocab: list[str] = []
    for tag, pearson in CHECKPOINTS.items():
        matches = [
            f
            for f in glob.glob(osp.join(INFERENCE_DIR, f"*Pearson={pearson}*.parquet"))
            if not f.endswith((".rank0", ".rank1", ".rank2", ".rank3"))
        ]
        table = pq.read_table(
            matches[0], columns=["gene1", "gene2", "gene3", "prediction"]
        )
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
            triples = np.stack(cols, axis=1)
        preds[tag] = table["prediction"].to_numpy().astype(np.float64)
        del table
    pred = np.mean([preds[t] for t in CHECKPOINTS], axis=0)
    worst = np.min([preds[t] for t in CHECKPOINTS], axis=0)

    vcode = {g: i for i, g in enumerate(vocab)}
    hub_v = vcode[HUB]
    got_pred, got_mean, got_min = [], [], []
    for i in np.where((triples == hub_v).any(axis=1))[0]:
        others = [str(vocab[x]) for x in triples[i] if x != hub_v]
        cs = [col.get(g) for g in others]
        if any(c is None or c not in in_screen for c in cs):
            continue
        got_pred.append(pred[i])
        got_mean.append(float(np.mean([in_screen[c] for c in cs])))
        got_min.append(float(min(in_screen[c] for c in cs)))

    # How the strong tail relates to one-screen genes, over the whole space.
    screens_of = {g: 0 for g in vocab}
    return_extra: dict[str, object] = {}
    p, q, r = np.array(got_pred), np.array(got_mean), np.array(got_min)
    return_extra["_vocab"] = vocab
    return_extra["_triples"] = triples
    return_extra["_pred_all"] = worst
    del screens_of
    return {
        **return_extra,
        "hub": HUB,
        "hub_query_double": f"{HUB} + {names[partner]}",
        "hub_records": int(m.sum()),
        "hub_distinct_screens": 1,
        "n_array_genes_in_hub_screen": len(in_screen),
        "n_inference_triples_with_both_partners_measured": int(len(p)),
        "pearson_pred_vs_partner_mean": float(pearsonr(p, q)[0]),
        "pearson_pred_vs_partner_min": float(pearsonr(p, r)[0]),
        "_pred": p,
        "_mean": q,
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows, tau, names, col, base, recurring, query = training_structure()
    print(f"{len(rows):,} records, {len(recurring)} query doubles")

    genes = per_gene(rows, names, query)
    genes.to_csv(osp.join(RESULTS_DIR, "screen_diversity_per_gene.csv"), index=False)
    print(f"\ndistinct screens per gene over {len(genes):,} genes:")
    print(genes["n_distinct_screens"].describe(percentiles=[0.01, 0.5, 0.99]).to_string())

    panel = genes[genes["gene"].isin(PANEL)].copy()
    panel["passes_screen_gate"] = panel["n_distinct_screens"] >= MIN_DISTINCT_SCREENS
    panel = panel.sort_values("n_distinct_screens")
    panel.to_csv(osp.join(RESULTS_DIR, "screen_diversity_panel.csv"), index=False)
    print(f"\npanel genes against a floor of {MIN_DISTINCT_SCREENS} distinct screens:")
    print(panel.to_string(index=False))

    mem = memorization(rows, tau, names, col, base, query)
    pred, pmean = mem.pop("_pred"), mem.pop("_mean")
    vocab, inf_triples, inf_pred = (
        mem.pop("_vocab"), mem.pop("_triples"), mem.pop("_pred_all")
    )

    # Enrichment of one-screen genes in the strong tail, against the space's
    # own base rate. Without the base rate the tail figure means nothing: 4.5
    # percent of all triples already contain such a gene.
    screens = dict(zip(genes["gene"], genes["n_distinct_screens"]))
    sc = np.array([screens.get(str(g), 0) for g in vocab])
    min_screens = sc[inf_triples].min(axis=1)
    one = min_screens == 1
    tail = {}
    for cut in (0.08, 0.16):
        k = inf_pred > cut
        tail[f"frac_one_screen_above_{cut}"] = float(one[k].mean())
        tail[f"enrichment_above_{cut}"] = float(one[k].mean() / one.mean())
        tail[f"n_above_{cut}"] = int(k.sum())
    tail["base_rate_one_screen"] = float(one.mean())
    tail["median_min_screens_whole_space"] = int(np.median(min_screens))
    mem.update(tail)
    with open(osp.join(RESULTS_DIR, "screen_diversity_summary.json"), "w") as f:
        json.dump(
            {
                **mem,
                "min_distinct_screens_gate": MIN_DISTINCT_SCREENS,
                "panel_genes_failing_gate": panel.loc[
                    ~panel["passes_screen_gate"], "gene"
                ].tolist(),
                "genes_below_gate_overall": int(
                    (genes["n_distinct_screens"] < MIN_DISTINCT_SCREENS).sum()
                ),
            },
            f,
            indent=2,
        )
    print("\nmemorization check:")
    for k, v in mem.items():
        print(f"  {k}: {v}")

    plot(genes, panel, pred, pmean, mem)


def plot(genes, panel, pred, pmean, mem) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(58.0))
    )

    # a. records is not support: the two statistics come apart
    ax = axes[0]
    ax.scatter(
        genes["n_records"], genes["n_distinct_screens"], s=1.5,
        color=PLOT_PALETTE[5], edgecolors="none", alpha=0.35, rasterized=True,
    )
    ax.scatter(
        panel["n_records"], panel["n_distinct_screens"], s=9,
        color=PLOT_PALETTE[0], edgecolors="black", linewidths=0.3, zorder=3,
        label="panel genes",
    )
    hub = panel[panel["gene"] == HUB]
    ax.scatter(
        hub["n_records"], hub["n_distinct_screens"], s=16,
        color=PLOT_PALETTE[1], edgecolors="black", linewidths=0.4, zorder=4,
        label=HUB,
    )
    ax.axhline(
        MIN_DISTINCT_SCREENS, color="black", linewidth=0.7, linestyle="--",
        label=f"floor {MIN_DISTINCT_SCREENS} screens",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Trigenic training records")
    ax.set_ylabel("Distinct query doubles measured under")
    ax.set_title("A record count is not support", fontsize=6)
    ax.legend(frameon=False, fontsize=4.5, loc="lower right")

    # b. the panel genes, ordered
    ax = axes[1]
    p = panel.sort_values("n_distinct_screens")
    x = np.arange(len(p))
    colors = [
        PLOT_PALETTE[0] if ok else PLOT_PALETTE[1] for ok in p["passes_screen_gate"]
    ]
    ax.bar(x, p["n_distinct_screens"], color=colors, edgecolor="black", linewidth=0.4, width=0.66)
    ax.axhline(MIN_DISTINCT_SCREENS, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(p["gene"], rotation=90, fontsize=4.5)
    ax.set_yscale("log")
    ax.set_ylabel("Distinct screens")
    n_fail = int((~p["passes_screen_gate"]).sum())
    ax.set_title(f"{n_fail} panel genes sit under the floor", fontsize=6)

    # c. predictions against the one screen's memorized values
    ax = axes[2]
    ax.scatter(
        pmean, pred, s=1.5, color=PLOT_PALETTE[0], edgecolors="none", alpha=0.3,
        rasterized=True,
    )
    lim = [min(pmean.min(), pred.min()), max(pmean.max(), pred.max())]
    ax.plot(lim, lim, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel(r"Partners' measured $\tau$ in the hub screen, mean")
    ax.set_ylabel(r"Predicted $\tau$ for the novel triple")
    ax.set_title(
        f"r = {mem['pearson_pred_vs_partner_mean']:.2f} over "
        f"{mem['n_inference_triples_with_both_partners_measured']:,} triples",
        fontsize=6,
    )

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.grid(which="both", linewidth=0.3, color="0.85")
        ax.set_axisbelow(True)

    fig.suptitle(
        "Screen diversity, and why YER059W's 1,097 records are one measurement",
        fontsize=6.5,
    )
    fig.tight_layout()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    stem = osp.join(IMAGE_DIR, "screen_diversity_audit")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"\nwrote {stem}.svg")


if __name__ == "__main__":
    main()
