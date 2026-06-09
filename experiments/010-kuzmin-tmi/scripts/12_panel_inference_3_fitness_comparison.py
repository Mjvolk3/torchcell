# experiments/010-kuzmin-tmi/scripts/12_panel_inference_3_fitness_comparison.py
# [[experiments.010-kuzmin-tmi.scripts.12_panel_inference_3_fitness_comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/12_panel_inference_3_fitness_comparison

"""
Panel 12 Inference 3: Fitness Measurement Comparison Across Sources.

Reads the queried singles/doubles CSV files and produces:
  1. Formatted tables (stdout) of multi-source measurements
  2. Forest plots (SMF + DMF) — point ± error bar per source
  3. Gaussian overlay plots (SMF + DMF) — N(μ,σ) density curves
  4. Summary statistics CSV (between-source spread vs within-source noise)
  5. Gene–triples summary table
  6. All doubles (lowest-std source) table
  7. Triple fitness trajectory summary
  8. Best-path trajectory plot (WT → max SMF → max DMF → f_ijk)
  9. Hero triple plots showing all 6 mutation orderings
"""

import os
import os.path as osp
from itertools import permutations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import norm


load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

# ── Color palette (first 3 from torchcell.mplstyle prop_cycle) ────────
COLORS = {
    "Costanzo2016": "#000000",
    "Kuzmin2018": "#D86E2F",
    "Kuzmin2020": "#7191A9",
}

IMAGES_SUBDIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════


def load_data():
    """Load the queried singles, doubles, and triples CSVs."""
    results_dir = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi/results/inference_3")
    singles_df = pd.read_csv(
        osp.join(results_dir, "singles_table_panel12_k200_queried.csv")
    )
    doubles_df = pd.read_csv(
        osp.join(results_dir, "doubles_table_panel12_k200_queried.csv")
    )
    triples_df = pd.read_csv(
        osp.join(results_dir, "triples_table_panel12_k200.csv")
    )
    return singles_df, doubles_df, triples_df


def _extract_singles(singles_df, min_sources=1):
    """Extract singles with ≥min_sources SMF fitness measurements.

    Returns list of dicts: {gene, measurements: [{source, mean, std}, ...]}
    """
    sources = [
        ("Costanzo2016", "SmfCostanzo2016_fitness", "SmfCostanzo2016_std"),
        ("Kuzmin2018", "SmfKuzmin2018_fitness", "SmfKuzmin2018_std"),
        ("Kuzmin2020", "SmfKuzmin2020_fitness", "SmfKuzmin2020_std"),
    ]
    results = []
    for _, row in singles_df.iterrows():
        measurements = []
        for source_name, fit_col, std_col in sources:
            if pd.notna(row[fit_col]):
                mean_val = float(row[fit_col])
                std_val = float(row[std_col]) if pd.notna(row[std_col]) else None
                measurements.append(
                    {"source": source_name, "mean": mean_val, "std": std_val}
                )
        if len(measurements) >= min_sources:
            results.append({"gene": row["gene"], "measurements": measurements})
    return results


def extract_all_singles(singles_df):
    """Extract all singles with ≥1 SMF source."""
    return _extract_singles(singles_df, min_sources=1)


def extract_singles_multi_source(singles_df):
    """Extract singles with ≥2 SMF sources (for summary stats)."""
    return _extract_singles(singles_df, min_sources=2)


def sort_by_source_combo(entries, gene_key="gene"):
    """Sort entries by source combination: multi-source first, then by combo name."""
    def _sort_key(entry):
        sources = tuple(sorted(m["source"] for m in entry["measurements"]))
        return (-len(sources), sources, entry[gene_key])
    return sorted(entries, key=_sort_key)


def extract_doubles_multi_source(doubles_df):
    """Extract doubles that have ≥2 DMF fitness measurements across sources.

    Returns list of dicts: {gene1, gene2, label, measurements: [{source, mean, std}, ...]}
    """
    sources = [
        ("Costanzo2016", "DmfCostanzo2016_fitness", "DmfCostanzo2016_std"),
        ("Kuzmin2018", "DmfKuzmin2018_fitness", "DmfKuzmin2018_std"),
        ("Kuzmin2020", "DmfKuzmin2020_fitness", "DmfKuzmin2020_std"),
    ]
    results = []
    for _, row in doubles_df.iterrows():
        measurements = []
        for source_name, fit_col, std_col in sources:
            if pd.notna(row[fit_col]):
                mean_val = float(row[fit_col])
                std_val = float(row[std_col]) if pd.notna(row[std_col]) else None
                measurements.append(
                    {"source": source_name, "mean": mean_val, "std": std_val}
                )
        if len(measurements) >= 2:
            results.append(
                {
                    "gene1": row["gene1"],
                    "gene2": row["gene2"],
                    "label": f"{row['gene1']} – {row['gene2']}",
                    "measurements": measurements,
                }
            )
    return results


# ══════════════════════════════════════════════════════════════════════
# Output 1: Formatted tables
# ══════════════════════════════════════════════════════════════════════


def print_formatted_tables(singles_multi, doubles_multi):
    """Print aligned tables of multi-source measurements to stdout."""
    print("=" * 80)
    print("SINGLES WITH ≥2 SMF SOURCES")
    print("=" * 80)

    header = f"{'Gene':<14} {'Costanzo2016':>20} {'Kuzmin2018':>20} {'Kuzmin2020':>20}"
    print(header)
    print("-" * 80)

    for entry in singles_multi:
        vals = {}
        for m in entry["measurements"]:
            if m["std"] is not None:
                vals[m["source"]] = f"{m['mean']:.4f} ± {m['std']:.4f}"
            else:
                vals[m["source"]] = f"{m['mean']:.4f} (no std)"
        row = (
            f"{entry['gene']:<14}"
            f" {vals.get('Costanzo2016', '—'):>20}"
            f" {vals.get('Kuzmin2018', '—'):>20}"
            f" {vals.get('Kuzmin2020', '—'):>20}"
        )
        print(row)

    print(f"\n({len(singles_multi)} of 12 genes have ≥2 sources)\n")

    print("=" * 80)
    print("DOUBLES WITH ≥2 DMF SOURCES")
    print("=" * 80)

    header = f"{'Pair':<28} {'Costanzo2016':>20} {'Kuzmin2018':>20} {'Kuzmin2020':>20}"
    print(header)
    print("-" * 94)

    for entry in doubles_multi:
        vals = {}
        for m in entry["measurements"]:
            if m["std"] is not None:
                vals[m["source"]] = f"{m['mean']:.4f} ± {m['std']:.4f}"
            else:
                vals[m["source"]] = f"{m['mean']:.4f} (no std)"
        pair_label = f"{entry['gene1']}–{entry['gene2']}"
        row = (
            f"{pair_label:<28}"
            f" {vals.get('Costanzo2016', '—'):>20}"
            f" {vals.get('Kuzmin2018', '—'):>20}"
            f" {vals.get('Kuzmin2020', '—'):>20}"
        )
        print(row)

    print(f"\n({len(doubles_multi)} of 66 pairs have ≥2 sources)\n")


# ══════════════════════════════════════════════════════════════════════
# Output 2: Forest plots
# ══════════════════════════════════════════════════════════════════════


def make_forest_plot(entries, labels, title, xlabel, filename, annotate_side="right"):
    """Create a forest plot (point ± error bar) for multi-source measurements.

    Parameters
    ----------
    entries : list of dicts with 'measurements' key
    labels : list of str, y-axis labels (gene names or pair names)
    title : str, figure title
    xlabel : str, x-axis label
    filename : str, output filename (without path)
    annotate_side : str, "left" or "right" — where to place annotations and legend
    """
    plt.style.use("torchcell/torchcell.mplstyle")

    n = len(entries)
    fig_height = max(4, 0.7 * n + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_positions = list(range(n - 1, -1, -1))  # top to bottom
    labeled_sources = set()

    for i, (entry, y_base) in enumerate(zip(entries, y_positions)):
        measurements = entry["measurements"]
        n_sources = len(measurements)
        # Offset multiple sources within the same gene group
        offsets = np.linspace(-0.15 * (n_sources - 1), 0.15 * (n_sources - 1), n_sources)

        for j, m in enumerate(measurements):
            color = COLORS[m["source"]]
            xerr = m["std"] if m["std"] is not None else 0
            marker = "o"
            label = m["source"] if m["source"] not in labeled_sources else None
            if label:
                labeled_sources.add(m["source"])
            ax.errorbar(
                m["mean"],
                y_base + offsets[j],
                xerr=xerr,
                fmt=marker,
                color=color,
                markersize=8,
                linewidth=2,
                label=label,
            )

    # Reference line at fitness = 1.0
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=0)

    ax.set_xlim(0, 1.4)
    ax.set_xticks(np.arange(0, 1.6, 0.2))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    # De-duplicate legend, placed BELOW the axes so it can never cover data
    # or the per-row annotations (in-axes corners overlapped them previously).
    handles, leg_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(leg_labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(by_label),
        frameon=False,
    )

    # Per-row spread annotations: anchored in axes-fraction x (fixed margin off
    # the spine, independent of the data scale) and data y (row-aligned), on a
    # white background so they read cleanly instead of sitting on the border.
    if annotate_side == "left":
        ann_x, ann_ha = 0.02, "left"
    else:
        ann_x, ann_ha = 0.98, "right"

    for i, (entry, y_base) in enumerate(zip(entries, y_positions)):
        measurements = entry["measurements"]
        if len(measurements) < 2:
            continue
        means = [m["mean"] for m in measurements]
        stds_available = [m["std"] for m in measurements if m["std"] is not None]
        between_spread = max(means) - min(means)
        avg_std = np.mean(stds_available) if stds_available else float("nan")
        annotation = f"|Δμ| = {between_spread:.3f}"
        if not np.isnan(avg_std):
            annotation += f"\nmean(σ) = {avg_std:.3f}"
        ax.annotate(
            annotation,
            xy=(ann_x, y_base),
            xycoords=("axes fraction", "data"),
            fontsize=13,
            va="center",
            ha=ann_ha,
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )

    plt.tight_layout()

    filepath = osp.join(IMAGES_SUBDIR, f"{filename}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


# ══════════════════════════════════════════════════════════════════════
# Output 3: Gaussian overlay plots
# ══════════════════════════════════════════════════════════════════════


def make_gaussian_overlay(entries, labels, suptitle, filename):
    """Create multi-panel Gaussian overlay plot.

    For each entry, plot N(μ,σ) density curves per source.
    Sources without std are shown as vertical dashed lines.
    """
    plt.style.use("torchcell/torchcell.mplstyle")

    n = len(entries)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False
    )

    for idx, (entry, label) in enumerate(zip(entries, labels)):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        # Fixed x-range: 0 to 1.4 for all subplots
        x = np.linspace(0, 1.4, 400)

        for m in entry["measurements"]:
            color = COLORS[m["source"]]
            if m["std"] is not None and m["std"] > 0:
                y = norm.pdf(x, m["mean"], m["std"])
                ax.plot(x, y, color=color, label=m["source"])
                ax.fill_between(x, y, alpha=0.15, color=color)
            else:
                # Point estimate — vertical dashed line
                ax.axvline(
                    x=m["mean"],
                    color=color,
                    linestyle="--",
                    linewidth=2,
                    label=m["source"],
                )

        ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlim(0, 1.4)
        ax.set_xticks(np.arange(0, 1.6, 0.2))
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Fitness")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9, loc="upper left")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(suptitle, fontsize=18, y=1.02)
    plt.tight_layout()

    filepath = osp.join(IMAGES_SUBDIR, f"{filename}.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved: {filepath}")


# ══════════════════════════════════════════════════════════════════════
# Output 4: Summary statistics
# ══════════════════════════════════════════════════════════════════════


def compute_summary(singles_multi, doubles_multi):
    """Compute between-source spread vs within-source noise for each entry.

    Returns a DataFrame with columns:
      type, label, between_source_spread, avg_within_source_std, spread_to_noise_ratio
    """
    rows = []
    for entry in singles_multi:
        means = [m["mean"] for m in entry["measurements"]]
        stds = [m["std"] for m in entry["measurements"] if m["std"] is not None]
        spread = max(means) - min(means)
        avg_std = np.mean(stds) if stds else float("nan")
        ratio = spread / avg_std if stds and avg_std > 0 else float("nan")
        rows.append(
            {
                "type": "single",
                "label": entry["gene"],
                "between_source_spread": round(spread, 4),
                "avg_within_source_std": round(avg_std, 4) if stds else None,
                "spread_to_noise_ratio": round(ratio, 4) if stds else None,
            }
        )
    for entry in doubles_multi:
        means = [m["mean"] for m in entry["measurements"]]
        stds = [m["std"] for m in entry["measurements"] if m["std"] is not None]
        spread = max(means) - min(means)
        avg_std = np.mean(stds) if stds else float("nan")
        ratio = spread / avg_std if stds and avg_std > 0 else float("nan")
        rows.append(
            {
                "type": "double",
                "label": entry["label"],
                "between_source_spread": round(spread, 4),
                "avg_within_source_std": round(avg_std, 4),
                "spread_to_noise_ratio": round(ratio, 4),
            }
        )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# Output 5: Gene–triples summary table
# ══════════════════════════════════════════════════════════════════════


def print_gene_triples_summary(singles_df, triples_df):
    """Print table of each gene's mean SMF fitness and triple-appearance count."""
    # Count appearances across gene1, gene2, gene3 columns
    gene_counts = pd.concat(
        [triples_df["gene1"], triples_df["gene2"], triples_df["gene3"]]
    ).value_counts()

    # Build summary rows — pick the source with lowest std for each gene
    sources = [
        ("Costanzo2016", "SmfCostanzo2016_fitness", "SmfCostanzo2016_std"),
        ("Kuzmin2018", "SmfKuzmin2018_fitness", "SmfKuzmin2018_std"),
        ("Kuzmin2020", "SmfKuzmin2020_fitness", "SmfKuzmin2020_std"),
    ]
    rows = []
    for _, row in singles_df.iterrows():
        gene = row["gene"]
        best_fitness, best_std, best_source = None, float("inf"), None
        fallback_fitness, fallback_source = None, None
        for source_name, fit_col, std_col in sources:
            if pd.notna(row[fit_col]):
                fitness = float(row[fit_col])
                if pd.notna(row[std_col]):
                    std = float(row[std_col])
                    if std < best_std:
                        best_fitness, best_std, best_source = fitness, std, source_name
                elif fallback_fitness is None:
                    fallback_fitness, fallback_source = fitness, source_name
        # Use lowest-std source; if none have std, use first available
        if best_fitness is not None:
            smf, std, source = best_fitness, best_std, best_source
        else:
            smf, std, source = fallback_fitness, None, fallback_source
        rows.append(
            {
                "gene": gene,
                "smf_fitness": smf,
                "smf_std": std,
                "source": source,
                "triple_count": int(gene_counts.get(gene, 0)),
            }
        )

    df = pd.DataFrame(rows).sort_values("smf_fitness", ascending=False)

    print("=" * 80)
    print("GENE SUMMARY: SMF Fitness (lowest σ source) & Triple Appearances")
    print("=" * 80)
    header = f"{'Gene':<14} {'Fitness':>9} {'± Std':>9} {'Source':>16} {'Triples':>9}"
    print(header)
    print("-" * 59)
    for _, r in df.iterrows():
        fit_str = f"{r['smf_fitness']:.4f}" if pd.notna(r["smf_fitness"]) else "—"
        std_str = f"{r['smf_std']:.4f}" if pd.notna(r["smf_std"]) else "—"
        print(
            f"{r['gene']:<14} {fit_str:>9} {std_str:>9} {r['source']:>16} {r['triple_count']:>9}"
        )
    print(f"\nTotal triples: {len(triples_df)}")

    return df


def print_all_doubles_lowest_std(doubles_df):
    """Print all 66 doubles with DMF fitness from the source with lowest std."""
    sources = [
        ("Costanzo2016", "DmfCostanzo2016_fitness", "DmfCostanzo2016_std"),
        ("Kuzmin2018", "DmfKuzmin2018_fitness", "DmfKuzmin2018_std"),
        ("Kuzmin2020", "DmfKuzmin2020_fitness", "DmfKuzmin2020_std"),
    ]
    rows = []
    for _, row in doubles_df.iterrows():
        gene1, gene2 = row["gene1"], row["gene2"]
        best_fitness, best_std, best_source = None, float("inf"), None
        fallback_fitness, fallback_source = None, None
        for source_name, fit_col, std_col in sources:
            if pd.notna(row[fit_col]):
                fitness = float(row[fit_col])
                if pd.notna(row[std_col]):
                    std = float(row[std_col])
                    if std < best_std:
                        best_fitness, best_std, best_source = fitness, std, source_name
                elif fallback_fitness is None:
                    fallback_fitness, fallback_source = fitness, source_name
        if best_fitness is not None:
            dmf, std, source = best_fitness, best_std, best_source
        else:
            dmf, std, source = fallback_fitness, None, fallback_source
        rows.append(
            {
                "pair": f"{gene1}–{gene2}",
                "dmf_fitness": dmf,
                "dmf_std": std,
                "source": source if source else "—",
            }
        )

    df = pd.DataFrame(rows).sort_values("dmf_fitness", ascending=True)

    print("=" * 80)
    print("ALL DOUBLES: DMF Fitness (lowest σ source)")
    print("=" * 80)
    header = f"{'Pair':<28} {'Fitness':>9} {'± Std':>9} {'Source':>16}"
    print(header)
    print("-" * 64)
    for _, r in df.iterrows():
        fit_str = f"{r['dmf_fitness']:.4f}" if pd.notna(r["dmf_fitness"]) else "—"
        std_str = f"{r['dmf_std']:.4f}" if pd.notna(r["dmf_std"]) else "—"
        src_str = r["source"] if r["source"] else "—"
        print(f"{r['pair']:<28} {fit_str:>9} {std_str:>9} {src_str:>16}")
    print(f"\n({len(df)} pairs total, {df['dmf_fitness'].notna().sum()} with data)")

    return df


# ══════════════════════════════════════════════════════════════════════
# Output 7-9: Fitness trajectory analysis
# ══════════════════════════════════════════════════════════════════════


def build_smf_lookup(singles_df):
    """Build gene → SMF fitness lookup using the source with lowest std."""
    sources = [
        ("Costanzo2016", "SmfCostanzo2016_fitness", "SmfCostanzo2016_std"),
        ("Kuzmin2018", "SmfKuzmin2018_fitness", "SmfKuzmin2018_std"),
        ("Kuzmin2020", "SmfKuzmin2020_fitness", "SmfKuzmin2020_std"),
    ]
    lookup = {}
    for _, row in singles_df.iterrows():
        best_fitness, best_std = None, float("inf")
        fallback_fitness = None
        for _, fit_col, std_col in sources:
            if pd.notna(row[fit_col]):
                fitness = float(row[fit_col])
                if pd.notna(row[std_col]):
                    std = float(row[std_col])
                    if std < best_std:
                        best_fitness, best_std = fitness, std
                elif fallback_fitness is None:
                    fallback_fitness = fitness
        lookup[row["gene"]] = (
            best_fitness if best_fitness is not None else fallback_fitness
        )
    return lookup


def build_dmf_lookup(doubles_df):
    """Build frozenset({gene1, gene2}) → DMF fitness lookup using lowest std source."""
    sources = [
        ("Costanzo2016", "DmfCostanzo2016_fitness", "DmfCostanzo2016_std"),
        ("Kuzmin2018", "DmfKuzmin2018_fitness", "DmfKuzmin2018_std"),
        ("Kuzmin2020", "DmfKuzmin2020_fitness", "DmfKuzmin2020_std"),
    ]
    lookup = {}
    for _, row in doubles_df.iterrows():
        best_fitness, best_std = None, float("inf")
        fallback_fitness = None
        for _, fit_col, std_col in sources:
            if pd.notna(row[fit_col]):
                fitness = float(row[fit_col])
                if pd.notna(row[std_col]):
                    std = float(row[std_col])
                    if std < best_std:
                        best_fitness, best_std = fitness, std
                elif fallback_fitness is None:
                    fallback_fitness = fitness
        key = frozenset({row["gene1"], row["gene2"]})
        dmf = best_fitness if best_fitness is not None else fallback_fitness
        if dmf is not None:
            lookup[key] = dmf
    return lookup


def compute_triple_fitness(triples_df, smf_lookup, dmf_lookup):
    """Compute f_ijk from τ_ijk and SMF/DMF values for each triple.

    Uses: f_ijk = τ_ijk + f_ij·f_k + f_ik·f_j + f_jk·f_i − 2·f_i·f_j·f_k
    """
    rows = []
    for _, row in triples_df.iterrows():
        g1, g2, g3 = row["gene1"], row["gene2"], row["gene3"]
        tau = float(row["prediction"])

        f_i = smf_lookup[g1]
        f_j = smf_lookup[g2]
        f_k = smf_lookup[g3]

        f_ij = dmf_lookup[frozenset({g1, g2})]
        f_ik = dmf_lookup[frozenset({g1, g3})]
        f_jk = dmf_lookup[frozenset({g2, g3})]

        f_ijk = tau + f_ij * f_k + f_ik * f_j + f_jk * f_i - 2 * f_i * f_j * f_k

        max_smf = max(f_i, f_j, f_k)
        max_dmf = max(f_ij, f_ik, f_jk)

        # Check if any permutation gives a monotonically increasing path
        monotonic = False
        for perm in permutations([g1, g2, g3]):
            a, b, _ = perm
            f_a = smf_lookup[a]
            f_ab = dmf_lookup[frozenset({a, b})]
            if 1.0 < f_a < f_ab < f_ijk:
                monotonic = True
                break

        rows.append(
            {
                "gene1": g1,
                "gene2": g2,
                "gene3": g3,
                "tau_ijk": tau,
                "f_i": f_i,
                "f_j": f_j,
                "f_k": f_k,
                "f_ij": f_ij,
                "f_ik": f_ik,
                "f_jk": f_jk,
                "f_ijk": f_ijk,
                "max_smf": max_smf,
                "max_dmf": max_dmf,
                "monotonic_path_exists": monotonic,
            }
        )

    return pd.DataFrame(rows)


def print_triple_trajectory_summary(trajectory_df):
    """Print trajectory summary table sorted by f_ijk descending."""
    df = trajectory_df.sort_values("f_ijk", ascending=False)

    print("=" * 100)
    print("TRIPLE FITNESS TRAJECTORIES (sorted by f_ijk descending)")
    print("=" * 100)
    header = (
        f"{'Genes':<38} {'max_smf':>9} {'max_dmf':>9} "
        f"{'f_ijk':>9} {'tau_ijk':>9} {'Mono':>6}"
    )
    print(header)
    print("-" * 100)

    for _, r in df.iterrows():
        genes = f"{r['gene1']}--{r['gene2']}--{r['gene3']}"
        mono_flag = "Y" if r["monotonic_path_exists"] else ""
        print(
            f"{genes:<38} {r['max_smf']:>9.4f} {r['max_dmf']:>9.4f} "
            f"{r['f_ijk']:>9.4f} {r['tau_ijk']:>9.4f} {mono_flag:>6}"
        )

    n_mono = trajectory_df["monotonic_path_exists"].sum()
    print(
        f"\n{n_mono} of {len(trajectory_df)} triples have a monotonically "
        "increasing path (1.0 < f_a < f_ab < f_ijk)"
    )


def plot_best_path_trajectories(trajectory_df, filename):
    """Plot WT → max(SMF) → max(DMF) → f_ijk trajectories for all triples."""
    plt.style.use("torchcell/torchcell.mplstyle")

    fig, ax = plt.subplots(figsize=(10, 7))

    x = [0, 1, 2, 3]
    x_labels = ["WT", "Single", "Double", "Triple"]

    mono_df = trajectory_df[trajectory_df["monotonic_path_exists"]]
    nonmono_df = trajectory_df[~trajectory_df["monotonic_path_exists"]]

    # Plot non-monotonic lines first (background)
    for _, r in nonmono_df.iterrows():
        y = [1.0, r["max_smf"], r["max_dmf"], r["f_ijk"]]
        ax.plot(x, y, color="black", alpha=0.1, linewidth=1)

    # Plot monotonic lines on top (highlighted)
    for _, r in mono_df.iterrows():
        y = [1.0, r["max_smf"], r["max_dmf"], r["f_ijk"]]
        ax.plot(x, y, color="#D86E2F", alpha=0.8, linewidth=1.5)

    # Legend proxies
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            alpha=0.4,
            linewidth=1,
            label=f"Non-monotonic (N={len(nonmono_df)})",
        ),
        Line2D(
            [0],
            [0],
            color="#D86E2F",
            alpha=0.8,
            linewidth=1.5,
            label=f"Monotonic (N={len(mono_df)})",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # Reference line at y=1.0
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Mutation Count")
    ax.set_ylabel("Fitness")
    ax.set_title(
        "Fitness Trajectories: WT \u2192 Single \u2192 Double \u2192 Triple"
        " (Panel 12, Inference 3)"
    )

    plt.tight_layout()
    filepath = osp.join(IMAGES_SUBDIR, f"{filename}.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_all_paths_hero_triples(
    trajectory_df,
    smf_lookup,
    dmf_lookup,
    n_heroes=6,
    filename="fitness_trajectory_paths_panel12_inference_3",
):
    """Plot all 6 mutation orderings for the top n_heroes triples by f_ijk."""
    plt.style.use("torchcell/torchcell.mplstyle")

    # Get color cycle from mplstyle
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    cycle_colors = [c["color"] for c in prop_cycle]

    top = trajectory_df.nlargest(n_heroes, "f_ijk")

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))

    x = [0, 1, 2, 3]
    x_labels = ["WT", "Single", "Double", "Triple"]

    for idx, (_, r) in enumerate(top.iterrows()):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]

        genes = (r["gene1"], r["gene2"], r["gene3"])
        f_ijk = r["f_ijk"]

        for perm_idx, perm in enumerate(permutations(genes)):
            a, b, _ = perm
            f_a = smf_lookup[a]
            f_ab = dmf_lookup[frozenset({a, b})]
            y = [1.0, f_a, f_ab, f_ijk]

            color = cycle_colors[perm_idx % len(cycle_colors)]
            label = f"{a} \u2192 {b} \u2192 {perm[2]}"
            ax.plot(
                x, y, color=color, linewidth=2, marker="o", markersize=5, label=label
            )

        ax.axhline(
            y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=0
        )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(
            f"{r['gene1']}\u2013{r['gene2']}\u2013{r['gene3']}"
            f"\nf_ijk = {f_ijk:.4f}",
            fontsize=11,
        )
        ax.set_ylabel("Fitness")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "All Mutation Paths: Top 6 Triples by f_ijk (Panel 12, Inference 3)",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    filepath = osp.join(IMAGES_SUBDIR, f"{filename}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    os.makedirs(IMAGES_SUBDIR, exist_ok=True)

    singles_df, doubles_df, triples_df = load_data()

    # Extract entries
    all_singles = sort_by_source_combo(extract_all_singles(singles_df))
    singles_multi = extract_singles_multi_source(singles_df)
    doubles_multi = extract_doubles_multi_source(doubles_df)

    # ── Output 1: Formatted tables ────────────────────────────────
    print_formatted_tables(singles_multi, doubles_multi)

    # ── Output 2: Forest plots ────────────────────────────────────
    # SMF forest plot: all 12 genes (annotations only on multi-source)
    all_singles_labels = [e["gene"] for e in all_singles]
    make_forest_plot(
        all_singles,
        all_singles_labels,
        title="SMF Fitness: Between-Source Comparison (Panel 12, Inference 3)",
        xlabel="SMF Fitness",
        filename="smf_forest_plot_panel12_inference_3",
    )

    # DMF forest plot: only multi-source doubles (66 pairs is too many)
    doubles_labels = [e["label"] for e in doubles_multi]
    make_forest_plot(
        doubles_multi,
        doubles_labels,
        title="DMF Fitness: Between-Source Comparison (Panel 12, Inference 3)",
        xlabel="DMF Fitness",
        filename="dmf_forest_plot_panel12_inference_3",
        annotate_side="left",
    )

    # ── Output 3: Gaussian overlays (all 12, sorted by source combo) ─
    all_singles_labels = [e["gene"] for e in all_singles]
    make_gaussian_overlay(
        all_singles,
        all_singles_labels,
        suptitle="SMF Fitness Distributions (Panel 12, Inference 3)",
        filename="smf_gaussian_overlay_panel12_inference_3",
    )

    make_gaussian_overlay(
        doubles_multi,
        doubles_labels,
        suptitle="DMF Fitness Distributions (Panel 12, Inference 3)",
        filename="dmf_gaussian_overlay_panel12_inference_3",
    )

    # ── Output 4: Summary statistics ──────────────────────────────
    summary_df = compute_summary(singles_multi, doubles_multi)

    print("=" * 80)
    print("SUMMARY: Between-Source Spread vs Within-Source Noise")
    print("=" * 80)
    print("  between_source_spread : max(μ) − min(μ) across sources")
    print("  avg_within_source_std : mean of available σ values")
    print("  spread_to_noise_ratio : spread / avg_std  (>1 → sources disagree beyond noise)")
    print("-" * 80)
    print(summary_df.to_string(index=False))

    results_dir = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi/results/inference_3")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = osp.join(results_dir, "fitness_comparison_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Output 5: Gene–triples summary ────────────────────────────
    print_gene_triples_summary(singles_df, triples_df)

    # ── Output 6: All doubles (lowest-std source) ─────────────────
    print_all_doubles_lowest_std(doubles_df)

    # ── Output 7-9: Trajectory analysis ───────────────────────────
    smf_lookup = build_smf_lookup(singles_df)
    dmf_lookup = build_dmf_lookup(doubles_df)
    trajectory_df = compute_triple_fitness(triples_df, smf_lookup, dmf_lookup)
    print_triple_trajectory_summary(trajectory_df)
    plot_best_path_trajectories(
        trajectory_df,
        filename="fitness_trajectory_maxpath_panel12_inference_3",
    )
    plot_all_paths_hero_triples(
        trajectory_df,
        smf_lookup,
        dmf_lookup,
        n_heroes=6,
        filename="fitness_trajectory_paths_panel12_inference_3",
    )


if __name__ == "__main__":
    main()
