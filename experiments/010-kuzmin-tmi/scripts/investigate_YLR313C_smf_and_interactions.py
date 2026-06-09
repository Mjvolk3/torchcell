# experiments/010-kuzmin-tmi/scripts/investigate_YLR313C_smf_and_interactions.py
# [[experiments.010-kuzmin-tmi.scripts.investigate_YLR313C_smf_and_interactions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/investigate_YLR313C_smf_and_interactions

"""
Investigate YLR312C-B (SGD "Merged" ORF — does not encode a discrete protein)
versus YLR313C / SPH1, the authentic gene the feature was merged into.

YLR312C-B is now an *alias* of SPH1/YLR313C (chr XII:760750-762342, - strand): the
old small-ORF span lies inside the SPH1 CDS, so a full-ORF KanMX replacement of
YLR312C-B deletes SPH1. This script collects the single- and double-mutant
phenotype for both nodes to make the swap decision.

Efficiency note: the experimental doubles are read straight from each dataset's
flat `preprocess/data.csv` (with a substring pre-filter on the target genes),
NOT by deserializing the 20.7M-record LMDB. Costanzo's Dmi preprocess file
carries both double-mutant fitness AND epsilon/P-value, so dmf_costanzo2016 is
never touched. Singles still use the small Smf LMDB datasets (~20k rows, ~1 s).

Run from repo root:
  ~/miniconda3/envs/torchcell/bin/python \
    experiments/010-kuzmin-tmi/scripts/investigate_YLR313C_smf_and_interactions.py

Outputs:
  1. results/inference_3/YLR313C_investigation_singles_queried.csv
  2. results/inference_3/YLR313C_investigation_doubles_queried.csv
  3. assets/images/010-kuzmin-tmi/YLR312C-B_triple_interaction_predictions_<date>.png
  4. Triple-participation counts + a markdown snippet (stdout) for the note.
"""

import csv
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from torchcell.datasets.scerevisiae import SmfCostanzo2016Dataset, SmfKuzmin2018Dataset
from torchcell.datasets.scerevisiae.kuzmin2020 import SmfKuzmin2020Dataset
from torchcell.timestamp import timestamp

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

# ── Investigation targets ────────────────────────────────────────────
# YLR312C-B: merged small ORF, the node the inference-3 model used.
# YLR313C:   SPH1, the authentic gene the feature was merged into.
TARGETS = ["YLR312C-B", "YLR313C"]
TARGET_ROLE = {
    "YLR312C-B": "merged small ORF (model node)",
    "YLR313C": "SPH1 (authentic gene)",
}

# The 12-gene inference-3 panel (partners for the doubles query).
PANEL_GENES = [
    "YBR203W", "YDR057W", "YER079W", "YGL087C", "YIL174W", "YJR060W",
    "YKL033W-A", "YLL012W", "YLR104W", "YLR312C-B", "YPL046C", "YPL081W",
]

HIGHLIGHT_GENE = "YLR312C-B"

# Significance thresholds for "real" digenic interaction.
P_THRESH = 0.05
EPS_THRESH = 0.08

COLOR_HIGHLIGHT = "#C0392B"  # crimson — triples containing YLR312C-B
COLOR_OTHER = "#9AA0A6"      # gray — all other triples

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi/results/inference_3")
IMAGES_SUBDIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")

# ── Flat preprocess CSVs for doubles (per study) ─────────────────────
# Each Dmi preprocess file carries epsilon, P-value, and double-mutant fitness.
DOUBLES_STUDIES = [
    {
        "study": "Costanzo2016",
        "csv": "data/torchcell/dmi_costanzo2016/preprocess/data.csv",
        "gene_cols": ["Query Systematic Name", "Array Systematic Name"],
        "eps": "Genetic interaction score (ε)",
        "pval": "P-value",
        "dmf": "Double mutant fitness",
        "dmf_std": "Double mutant fitness standard deviation",
    },
    {
        "study": "Kuzmin2018",
        "csv": "data/torchcell/dmi_kuzmin2018/preprocess/data.csv",
        "gene_cols": ["Query systematic name no ho", "Array systematic name"],
        "eps": "Adjusted genetic interaction score (epsilon or tau)",
        "pval": "P-value",
        "dmf": "Combined mutant fitness",
        "dmf_std": "Combined mutant fitness standard deviation",
    },
    {
        "study": "Kuzmin2020",
        "csv": "data/torchcell/dmi_kuzmin2020/preprocess/data.csv",
        "gene_cols": ["Query systematic name no ho", "Array systematic name"],
        "eps": "Adjusted genetic interaction score (epsilon or tau)",
        "pval": "P-value",
        "dmf": "Double/triple mutant fitness",
        "dmf_std": "Double/triple mutant fitness standard deviation",
    },
]


# ══════════════════════════════════════════════════════════════════════
# Singles — small Smf LMDB datasets (fast)
# ══════════════════════════════════════════════════════════════════════
def build_single_index(dataset, name: str) -> dict:
    """frozenset({gene}) -> {fitness, fitness_std, strain_id}."""
    index = {}
    for i in tqdm(range(len(dataset)), desc=f"Indexing {name}"):
        item = dataset[i]
        genes = frozenset(
            p["systematic_gene_name"]
            for p in item["experiment"]["genotype"]["perturbations"]
        )
        ph = item["experiment"]["phenotype"]
        index[genes] = {
            "fitness": ph["fitness"],
            "fitness_std": ph["fitness_std"],
            "strain_id": item["experiment"]["genotype"]["perturbations"][0]["strain_id"],
        }
    return index


def query_singles(genes: list) -> pd.DataFrame:
    rows = {g: {"gene": g, "role": TARGET_ROLE[g]} for g in genes}
    configs = [
        ("SmfCostanzo2016", SmfCostanzo2016Dataset, "smf_costanzo2016"),
        ("SmfKuzmin2018", SmfKuzmin2018Dataset, "smf_kuzmin2018"),
        ("SmfKuzmin2020", SmfKuzmin2020Dataset, "smf_kuzmin2020"),
    ]
    for name, cls, subdir in configs:
        print(f"\nProcessing {name}...")
        ds = cls(root=osp.join(DATA_ROOT, f"data/torchcell/{subdir}"), io_workers=4)
        idx = build_single_index(ds, name)
        for g in genes:
            data = idx.get(frozenset([g]))
            rows[g][f"{name}_fitness"] = data["fitness"] if data else None
            rows[g][f"{name}_std"] = data["fitness_std"] if data else None
            rows[g][f"{name}_strain_id"] = data["strain_id"] if data else None
        hits = sum(rows[g][f"{name}_fitness"] is not None for g in genes)
        print(f"  Matches found: {hits}/{len(genes)}")
    return pd.DataFrame([rows[g] for g in genes])


# ══════════════════════════════════════════════════════════════════════
# Doubles — streamed from flat preprocess CSVs with substring pre-filter
# ══════════════════════════════════════════════════════════════════════
def collect_doubles(targets: list, panel: list) -> pd.DataFrame:
    """For each target × panel partner, pull epsilon / P-value / DMF from each
    study's flat preprocess CSV. Keeps every directional measurement (Costanzo
    SGA stores query↔array separately, which exposes directional asymmetry)."""
    want = set(targets)
    panel_set = set(panel)
    records = []

    for cfg in DOUBLES_STUDIES:
        path = osp.join(DATA_ROOT, cfg["csv"])
        if not osp.exists(path):
            print(f"  [skip] {cfg['study']}: {path} not found")
            continue
        g1c, g2c = cfg["gene_cols"]
        n_scanned = n_parsed = 0
        with open(path, newline="") as fh:
            header = next(csv.reader([fh.readline()]))
            col = {c: header.index(c) for c in cfg["gene_cols"] + [
                cfg["eps"], cfg["pval"], cfg["dmf"], cfg["dmf_std"]
            ]}
            for line in fh:
                n_scanned += 1
                # cheap substring gate — targets are rare, skips ~99.9% of rows
                if "YLR312C-B" not in line and "YLR313C" not in line:
                    continue
                rec = next(csv.reader([line]))
                n_parsed += 1
                a, b = rec[col[g1c]], rec[col[g2c]]
                pair = frozenset([a, b])
                if len(pair) != 2:
                    continue
                target = a if a in want else (b if b in want else None)
                if target is None:
                    continue
                partner = b if a == target else a
                if partner not in panel_set:
                    continue
                records.append({
                    "study": cfg["study"],
                    "target": target,
                    "partner": partner,
                    "gene1": a,
                    "gene2": b,
                    "epsilon": rec[col[cfg["eps"]]],
                    "p_value": rec[col[cfg["pval"]]],
                    "dmf": rec[col[cfg["dmf"]]],
                    "dmf_std": rec[col[cfg["dmf_std"]]],
                })
        print(f"  {cfg['study']}: scanned {n_scanned:,}, parsed {n_parsed:,}, "
              f"matched {sum(r['study'] == cfg['study'] for r in records)}")
    return pd.DataFrame(records)


def summarize_doubles(doubles: pd.DataFrame, target: str) -> dict:
    """Per-partner significance summary for one target (Costanzo only carries data)."""
    sub = doubles[doubles["target"] == target].copy()
    if sub.empty:
        return {"measured": 0, "significant": 0, "sig_partners": []}
    sub["eps_f"] = pd.to_numeric(sub["epsilon"], errors="coerce")
    sub["p_f"] = pd.to_numeric(sub["p_value"], errors="coerce")
    sig_mask = (sub["p_f"] < P_THRESH) & (sub["eps_f"].abs() > EPS_THRESH)
    sig_partners = sorted(sub.loc[sig_mask, "partner"].unique())
    return {
        "measured": sub["partner"].nunique(),
        "significant": len(sig_partners),
        "sig_partners": sig_partners,
    }


# ══════════════════════════════════════════════════════════════════════
# Genome-wide Costanzo interactions (for the interaction histogram)
# ══════════════════════════════════════════════════════════════════════
def scan_costanzo_genomewide(targets: list) -> pd.DataFrame:
    """One pass over the Costanzo Dmi flat CSV; collect EVERY measured pair
    involving a target (all partners genome-wide, not just the panel)."""
    cfg = DOUBLES_STUDIES[0]  # Costanzo2016 carries epsilon + P-value
    path = osp.join(DATA_ROOT, cfg["csv"])
    g1c, g2c = cfg["gene_cols"]
    want = set(targets)
    rows = []
    with open(path, newline="") as fh:
        header = next(csv.reader([fh.readline()]))
        col = {c: header.index(c) for c in [g1c, g2c, cfg["eps"], cfg["pval"]]}
        for line in fh:
            if "YLR312C-B" not in line and "YLR313C" not in line:
                continue
            rec = next(csv.reader([line]))
            a, b = rec[col[g1c]], rec[col[g2c]]
            if a == b:
                continue
            target = a if a in want else (b if b in want else None)
            if target is None:
                continue
            rows.append({
                "target": target,
                "partner": b if a == target else a,
                "epsilon": rec[col[cfg["eps"]]],
                "p_value": rec[col[cfg["pval"]]],
            })
    df = pd.DataFrame(rows)
    df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    return df.dropna(subset=["epsilon"])


def plot_costanzo_interaction_histogram(gw_df: pd.DataFrame, out_path: str) -> dict:
    """Overlaid histogram of genome-wide Costanzo epsilon for each target node."""
    plt.style.use("torchcell/torchcell.mplstyle")
    colors = {"YLR312C-B": "#C0392B", "YLR313C": "#2C6FA6"}
    lo, hi = np.percentile(gw_df["epsilon"].values, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=(10, 6))
    stats = {}
    for gene in TARGETS:
        eps = gw_df.loc[gw_df["target"] == gene, "epsilon"].values
        if len(eps) == 0:
            continue
        ax.hist(eps, bins=bins, alpha=0.55, color=colors.get(gene, "#888888"),
                label=f"{gene} (n={len(eps)}, mean ε={eps.mean():+.4f})")
        ax.axvline(eps.mean(), color=colors.get(gene, "#888888"), ls="--", lw=1.6)
        stats[gene] = {
            "n": int(len(eps)), "mean": float(eps.mean()),
            "median": float(np.median(eps)), "std": float(eps.std()),
        }
    ax.axvline(0.0, color="black", lw=0.9, ls=":")
    ax.set_xlabel("Costanzo2016 digenic interaction score (ε)")
    ax.set_ylabel("Partner count")
    ax.set_title("Genome-wide Costanzo digenic interactions\n"
                 "YLR312C-B (merged/UTR region) vs YLR313C / SPH1 (gene)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved histogram: {out_path}")
    return stats


# ══════════════════════════════════════════════════════════════════════
# Triple participation + plot
# ══════════════════════════════════════════════════════════════════════
def triple_participation(triples_df: pd.DataFrame, gene: str) -> dict:
    mask = (
        (triples_df["gene1"] == gene)
        | (triples_df["gene2"] == gene)
        | (triples_df["gene3"] == gene)
    )
    contains, other = triples_df[mask], triples_df[~mask]
    return {
        "n_total": len(triples_df),
        "n_contains": int(mask.sum()),
        "mean_pred_contains": float(contains["prediction"].mean()),
        "mean_pred_other": float(other["prediction"].mean()),
        "max_pred_contains": float(contains["prediction"].max()),
        "best_rank": int(contains.index.min()) + 1,  # file is pre-sorted desc
    }


def plot_predictions(triples_df: pd.DataFrame, gene: str, out_path: str) -> None:
    plt.style.use("torchcell/torchcell.mplstyle")
    df = triples_df.sort_values("prediction", ascending=False).reset_index(drop=True)
    contains = (
        (df["gene1"] == gene) | (df["gene2"] == gene) | (df["gene3"] == gene)
    ).values
    preds = df["prediction"].values
    ranks = np.arange(1, len(df) + 1)
    n_hit, n_other = int(contains.sum()), len(df) - int(contains.sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.scatter(ranks[~contains], preds[~contains], s=22, c=COLOR_OTHER,
               label=f"Other triples (n={n_other})", zorder=2)
    ax.scatter(ranks[contains], preds[contains], s=40, c=COLOR_HIGHLIGHT,
               edgecolors="black", linewidths=0.4,
               label=f"Contains {gene} (n={n_hit})", zorder=3)
    ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Triple rank (by predicted interaction, desc)")
    ax.set_ylabel("Predicted gene interaction")
    ax.set_title(f"Panel-12 inference-3 triple predictions\n{gene} highlighted")
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1]
    bins = np.linspace(preds.min(), preds.max(), 24)
    ax.hist(preds[~contains], bins=bins, color=COLOR_OTHER, alpha=0.75,
            label=f"Other (μ={preds[~contains].mean():.3f})")
    ax.hist(preds[contains], bins=bins, color=COLOR_HIGHLIGHT, alpha=0.75,
            label=f"{gene} (μ={preds[contains].mean():.3f})")
    ax.set_xlabel("Predicted gene interaction")
    ax.set_ylabel("Triple count")
    ax.set_title("Distribution of predictions")
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(IMAGES_SUBDIR, exist_ok=True)
    print("=" * 64)
    print("Investigate YLR312C-B (merged ORF) vs YLR313C / SPH1")
    print("=" * 64)

    # ── Single-mutant fitness ────────────────────────────────────────
    singles_result = query_singles(TARGETS)
    singles_out = osp.join(RESULTS_DIR, "YLR313C_investigation_singles_queried.csv")
    singles_result.to_csv(singles_out, index=False)
    print(f"\nSaved: {singles_out}")
    print(singles_result.to_string())

    # ── Plot (no DB dependency — render before the doubles scan) ──────
    triples_df = pd.read_csv(osp.join(RESULTS_DIR, "triples_table_panel12_k200.csv"))
    topk_df = pd.read_csv(osp.join(RESULTS_DIR, "top_k_constructible_panel12_k200.csv"))
    full = triple_participation(triples_df, HIGHLIGHT_GENE)
    topk_n = int(
        ((topk_df["gene1"] == HIGHLIGHT_GENE)
         | (topk_df["gene2"] == HIGHLIGHT_GENE)
         | (topk_df["gene3"] == HIGHLIGHT_GENE)).sum()
    )
    print(f"\n── Triple participation for {HIGHLIGHT_GENE} ──")
    print(f"  full k=200 set : {full['n_contains']}/{full['n_total']} "
          f"(best rank {full['best_rank']}, max {full['max_pred_contains']:.3f}, "
          f"mean {full['mean_pred_contains']:.3f} vs {full['mean_pred_other']:.3f})")
    print(f"  top-52 subset  : {topk_n}/{len(topk_df)}")

    date_str = timestamp()[:10]  # YYYY-MM-DD
    img_name = f"YLR312C-B_triple_interaction_predictions_{date_str}.png"
    plot_predictions(triples_df, HIGHLIGHT_GENE, osp.join(IMAGES_SUBDIR, img_name))

    # ── Double-mutant fitness + interaction (fast flat-CSV scan) ──────
    print("\nCollecting doubles from flat preprocess CSVs...")
    doubles = collect_doubles(TARGETS, PANEL_GENES)
    doubles_out = osp.join(RESULTS_DIR, "YLR313C_investigation_doubles_queried.csv")
    doubles.to_csv(doubles_out, index=False)
    print(f"Saved: {doubles_out}")

    summaries = {t: summarize_doubles(doubles, t) for t in TARGETS}
    for t in TARGETS:
        s = summaries[t]
        sig = ", ".join(s["sig_partners"]) if s["sig_partners"] else "none"
        print(f"  {t}: {s['measured']} partners measured, "
              f"{s['significant']} significant (p<{P_THRESH}, |ε|>{EPS_THRESH}): {sig}")

    # ── Genome-wide Costanzo interaction histogram ───────────────────
    print("\nScanning Costanzo for genome-wide interactions of both nodes...")
    gw = scan_costanzo_genomewide(TARGETS)
    gw_out = osp.join(RESULTS_DIR, "YLR313C_investigation_costanzo_genomewide.csv")
    gw.to_csv(gw_out, index=False)
    print(f"Saved: {gw_out}")
    hist_name = f"YLR312C-B_vs_YLR313C_costanzo_interaction_histogram_{date_str}.png"
    gw_stats = plot_costanzo_interaction_histogram(gw, osp.join(IMAGES_SUBDIR, hist_name))
    for g in TARGETS:
        if g in gw_stats:
            s = gw_stats[g]
            print(f"  {g}: n={s['n']} genome-wide partners, mean ε={s['mean']:+.4f}, "
                  f"median={s['median']:+.4f}, std={s['std']:.4f}")

    # ── Markdown snippet for the note ────────────────────────────────
    print("\n" + "=" * 64)
    print("MARKDOWN SNIPPET (bottom-of-table rows for the note):")
    print("=" * 64)

    def fmt(r, name):
        f, s = r[f"{name}_fitness"], r[f"{name}_std"]
        if pd.isna(f):
            return "—"
        return f"{float(f):.4f} ± {float(s):.4f}" if not pd.isna(s) else f"{float(f):.4f}"

    for _, r in singles_result.iterrows():
        s = summaries[r["gene"]]
        sig = ", ".join(s["sig_partners"]) if s["sig_partners"] else "**0**"
        print(f"| {r['gene']} | {r['role']} | {fmt(r, 'SmfCostanzo2016')} | "
              f"{fmt(r, 'SmfKuzmin2018')} | {fmt(r, 'SmfKuzmin2020')} | "
              f"{s['measured']} | {sig} |")
    print(f"\n![{img_name}](./assets/images/010-kuzmin-tmi/{img_name})")
    print(f"\n![{hist_name}](./assets/images/010-kuzmin-tmi/{hist_name})")
    for g in TARGETS:
        if g in gw_stats:
            s = gw_stats[g]
            print(f"  {g}: genome-wide mean ε={s['mean']:+.4f} (n={s['n']})")


if __name__ == "__main__":
    main()
