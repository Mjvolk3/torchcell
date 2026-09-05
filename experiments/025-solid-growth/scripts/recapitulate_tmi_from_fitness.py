# experiments/025-solid-growth/scripts/recapitulate_tmi_from_fitness.py
# [[experiments.025-solid-growth.scripts.recapitulate_tmi_from_fitness]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/recapitulate_tmi_from_fitness
"""Recompute trigenic interaction scores from the fitness values stored in the 025 build.

The trigenic interaction is defined from fitness alone,

    tau_abc = f_abc - f_ab*f_c - f_ac*f_b - f_bc*f_a + 2*f_a*f_b*f_c,

so if the 025 dataset's smf/dmf/tmf values cannot reproduce its own stored tmi labels,
a model asked to route fitness supervision into interaction prediction is being asked
to learn an identity the data does not satisfy. Two variants are computed per triple:

- variant A "aggregate": every fitness value is the mean over ALL fitness entries in the
  genotype's group (Costanzo + Kuzmin + essentiality-derived), i.e. exactly the target a
  per-entry MSE trains toward.
- variant B "kuzmin": fitness values restricted to Kuzmin-sourced entries (the same
  screen family that produced the tau labels), where available for all seven terms.
- variant C "dmi": the digenic terms come from the STORED dmi labels instead of
  recomputed products, tau = f_abc - eps_ab*f_c - eps_ac*f_b - eps_bc*f_a - f_a*f_b*f_c,
  which is exactly the information set the S3 training arm hands the model.

The digenic identity eps_ab = f_ab - f_a*f_b is checked the same way against stored dmi.
Essential genes are a known contamination: a Kuzmin ts-allele gene's single-mutant group
merges its Costanzo ts smf with the SGD-essentiality-derived fitness 0, pulling f_a
toward 0.5 and wrecking every product term, so all trigenic statistics are also reported
on the stratum whose three singles carry no essentiality entry.
"""

import gzip
import json
import os
import os.path as osp
import re
from collections import defaultdict
from itertools import combinations

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from scipy import stats

from torchcell.timestamp import timestamp
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BUILD = osp.join(
    DATA_ROOT, "data/torchcell/experiments/025-solid-growth/001-full-build/processed"
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth/results")
TABLE_DIR = osp.join(
    DATA_ROOT, "data/torchcell/experiments/025-solid-growth/recapitulation"
)
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "025-solid-growth")
GENE_RE = re.compile(rb'"systematic_gene_name": "([^"]+)"')
TS = timestamp()


def group_values(entries: list[dict]) -> dict:
    """Fitness and interaction values of one genotype group, overall and Kuzmin-only."""
    fit_all: list[float] = []
    fit_kuz: list[float] = []
    gi_all: list[float] = []
    for item in entries:
        e = item["experiment"]
        ds = e["dataset_name"]
        ph = e["phenotype"]
        if e["experiment_type"] == "fitness":
            v = ph["fitness"]
            if v is not None:
                fit_all.append(v)
                if "Kuzmin" in ds:
                    fit_kuz.append(v)
        elif e["experiment_type"] == "gene interaction":
            v = ph["gene_interaction"]
            if v is not None:
                gi_all.append(v)
    has_ess = any(
        "GeneEssentialitySgd" in item["experiment"]["dataset_name"] for item in entries
    )
    return {
        "fit": float(np.mean(fit_all)) if fit_all else None,
        "fit_kuz": float(np.mean(fit_kuz)) if fit_kuz else None,
        "gi": float(np.mean(gi_all)) if gi_all else None,
        "n_fit": len(fit_all),
        "has_ess": has_ess,
    }


def scan() -> tuple[dict, dict, dict]:
    with open(osp.join(BUILD, "perturbation_count_index.json")) as f:
        count_index = json.load(f)
    idx_single = count_index["1"]
    idx_double = count_index["2"]
    idx_triple = count_index["3"]
    print(
        f"index sizes: singles {len(idx_single)}, doubles {len(idx_double)}, "
        f"triples {len(idx_triple)}",
        flush=True,
    )

    env = lmdb.open(
        osp.join(BUILD, "lmdb"), readonly=True, lock=False, max_readers=32
    )
    singles: dict[str, dict] = {}
    triples: dict[tuple, dict] = {}
    with env.begin() as txn:
        for i in idx_single:
            entries = json.loads(txn.get(str(i).encode()).decode())
            genes = {
                p["systematic_gene_name"]
                for p in entries[0]["experiment"]["genotype"]["perturbations"]
            }
            assert len(genes) == 1
            singles[next(iter(genes))] = group_values(entries)
        print(f"singles parsed: {len(singles)}", flush=True)

        for n, i in enumerate(idx_triple):
            raw = txn.get(str(i).encode())
            entries = json.loads(raw.decode())
            genes = {
                p["systematic_gene_name"]
                for p in entries[0]["experiment"]["genotype"]["perturbations"]
            }
            assert len(genes) == 3
            triples[tuple(sorted(genes))] = group_values(entries) | {"idx": i}
            if n % 100000 == 0:
                print(f"triples parsed: {n}", flush=True)
        print(f"triples parsed: {len(triples)}", flush=True)

        needed_pairs: set[frozenset] = set()
        for gs in triples:
            for pair in combinations(gs, 2):
                needed_pairs.add(frozenset(pair))
        print(f"needed pairs (closure of triples): {len(needed_pairs)}", flush=True)

        # The reference block carries no perturbations, so gene names in raw bytes come
        # only from the experiment genotypes; the cheap regex is exact for membership.
        doubles: dict[frozenset, dict] = {}
        for n, i in enumerate(idx_double):
            raw = txn.get(str(i).encode())
            pair = frozenset(m.decode() for m in set(GENE_RE.findall(raw)))
            if pair in needed_pairs:
                doubles[pair] = group_values(json.loads(raw.decode()))
            if n % 1000000 == 0:
                print(f"doubles scanned: {n}, kept {len(doubles)}", flush=True)
        print(f"doubles kept: {len(doubles)} of {len(idx_double)} scanned", flush=True)
    env.close()
    return singles, doubles, triples


def recompute(singles: dict, doubles: dict, triples: dict) -> dict:
    rows = []
    for gs, tv in triples.items():
        a, b, c = gs
        s = [singles.get(g) for g in gs]
        d_ab = doubles.get(frozenset((a, b)))
        d_ac = doubles.get(frozenset((a, c)))
        d_bc = doubles.get(frozenset((b, c)))
        row: dict = {
            "gene_a": a,
            "gene_b": b,
            "gene_c": c,
            "idx_025": tv["idx"],
            "tmi_stored": tv["gi"],
            "tmf_stored": tv["fit"],
            "n_singles": sum(x is not None for x in s),
            "n_doubles": sum(x is not None for x in (d_ab, d_ac, d_bc)),
            "tau_aggregate": None,
            "tau_kuzmin": None,
            "tau_dmi": None,
            "any_ess_single": any(x is not None and x["has_ess"] for x in s),
        }
        for variant, key in (("tau_aggregate", "fit"), ("tau_kuzmin", "fit_kuz")):
            f_abc = tv[key]
            vals = [x[key] if x else None for x in s]
            dd = [x[key] if x else None for x in (d_ab, d_ac, d_bc)]
            if f_abc is None or any(v is None for v in vals) or any(v is None for v in dd):
                continue
            fa, fb, fc = vals
            f_ab, f_ac, f_bc = dd
            row[variant] = (
                f_abc - f_ab * fc - f_ac * fb - f_bc * fa + 2.0 * fa * fb * fc
            )
        f_abc = tv["fit"]
        vals = [x["fit"] if x else None for x in s]
        ee = [x["gi"] if x else None for x in (d_ab, d_ac, d_bc)]
        if f_abc is not None and all(v is not None for v in vals) and all(
            e is not None for e in ee
        ):
            fa, fb, fc = vals
            e_ab, e_ac, e_bc = ee
            row["tau_dmi"] = (
                f_abc - e_ab * fc - e_ac * fb - e_bc * fa - fa * fb * fc
            )
        rows.append(row)
    return {"rows": rows}


def digenic_check(singles: dict, doubles: dict) -> list[dict]:
    rows = []
    for pair, dv in doubles.items():
        a, b = sorted(pair)
        sa, sb = singles.get(a), singles.get(b)
        if sa is None or sb is None or dv["fit"] is None or dv["gi"] is None:
            continue
        rows.append(
            {
                "gene_a": a,
                "gene_b": b,
                "dmi_stored": dv["gi"],
                "eps_aggregate": dv["fit"] - sa["fit"] * sb["fit"],
            }
        )
    return rows


def paired_stats(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"n": int(len(x))}
    lr = stats.linregress(x, y)
    return {
        "n": int(len(x)),
        "pearson": float(stats.pearsonr(x, y)[0]),
        "spearman": float(stats.spearmanr(x, y)[0]),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "median_abs_residual": float(np.median(np.abs(x - y))),
    }


def call_confusion(stored: np.ndarray, recomputed: np.ndarray, cut: float) -> dict:
    mask = np.isfinite(stored) & np.isfinite(recomputed)
    s, r = stored[mask] > cut, recomputed[mask] > cut
    return {
        "cut": cut,
        "stored_pos": int(s.sum()),
        "recomputed_pos": int(r.sum()),
        "both": int((s & r).sum()),
        "recall_of_stored": float((s & r).sum() / s.sum()) if s.sum() else None,
    }


ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
CMAP = LinearSegmentedColormap.from_list(
    "tc_orange", ["#FFFFFF", PLOT_PALETTE[3], PLOT_PALETTE[0], PLOT_PALETTE[6]]
)


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("black")


def hexbin_panel(x, y, xlabel, ylabel, title, stats_d, path_base, lim):
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(70))
    )
    mask = np.isfinite(x) & np.isfinite(y)
    hb = ax.hexbin(
        x[mask], y[mask], gridsize=80, bins="log", cmap=CMAP, extent=(*lim, *lim),
        linewidths=0.0,
    )
    ax.plot(lim, lim, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.set_title(title, fontsize=6)
    ax.tick_params(labelsize=6)
    ax.text(
        0.03,
        0.97,
        f"n = {stats_d['n']:,}\nr = {stats_d['pearson']:.3f}\n"
        f"rho = {stats_d['spearman']:.3f}\nslope = {stats_d['slope']:.3f}",
        transform=ax.transAxes,
        fontsize=6,
        va="top",
    )
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("triples (log10)", fontsize=6)
    fig.savefig(path_base + ".png", dpi=300, bbox_inches="tight")
    savefig_true_size_svg(fig, path_base + ".svg")
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none"})

    singles, doubles, triples = scan()
    recap = recompute(singles, doubles, triples)
    rows = recap["rows"]
    digenic = digenic_check(singles, doubles)

    tmi = np.array([r["tmi_stored"] if r["tmi_stored"] is not None else np.nan for r in rows])
    tau_a = np.array([r["tau_aggregate"] if r["tau_aggregate"] is not None else np.nan for r in rows])
    tau_k = np.array([r["tau_kuzmin"] if r["tau_kuzmin"] is not None else np.nan for r in rows])
    tau_d = np.array([r["tau_dmi"] if r["tau_dmi"] is not None else np.nan for r in rows])
    no_ess = np.array([not r["any_ess_single"] for r in rows])
    dmi = np.array([r["dmi_stored"] for r in digenic])
    eps_a = np.array([r["eps_aggregate"] for r in digenic])

    n_triples = len(rows)
    closure_full = sum(r["n_singles"] == 3 and r["n_doubles"] == 3 for r in rows)
    summary = {
        "generated": TS,
        "n_triples": n_triples,
        "n_singles_in_build": len(singles),
        "n_closure_pairs_needed": len({
            frozenset(p)
            for r in rows
            for p in combinations((r["gene_a"], r["gene_b"], r["gene_c"]), 2)
        }),
        "n_closure_pairs_found": len(doubles),
        "n_triples_full_closure": closure_full,
        "frac_triples_full_closure": closure_full / n_triples,
        "n_triples_no_ess_single": int(no_ess.sum()),
        "trigenic_aggregate": paired_stats(tmi, tau_a),
        "trigenic_aggregate_no_ess": paired_stats(tmi[no_ess], tau_a[no_ess]),
        "trigenic_kuzmin": paired_stats(tmi, tau_k),
        "trigenic_kuzmin_no_ess": paired_stats(tmi[no_ess], tau_k[no_ess]),
        "trigenic_dmi": paired_stats(tmi, tau_d),
        "trigenic_dmi_no_ess": paired_stats(tmi[no_ess], tau_d[no_ess]),
        "digenic_aggregate": paired_stats(dmi, eps_a),
        "positive_calls": [
            call_confusion(tmi, tau_a, c) for c in (0.08, 0.16, 0.20)
        ],
        "negative_calls_below_-0.08": call_confusion(-tmi, -tau_a, 0.08),
    }
    print(json.dumps(summary, indent=2), flush=True)

    with gzip.open(
        osp.join(TABLE_DIR, "recapitulation_per_triple.csv.gz"), "wt"
    ) as f:
        cols = list(rows[0].keys())
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")

    with open(osp.join(RESULTS_DIR, "recapitulation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    lim_t = (-0.6, 0.6)
    hexbin_panel(
        tmi, tau_a,
        "stored tmi (mean over group entries)",
        "recomputed tau, aggregate fitness",
        "Trigenic recapitulation, aggregate",
        summary["trigenic_aggregate"],
        osp.join(IMG_DIR, f"recapitulation_trigenic_aggregate_{TS}"), lim_t,
    )
    hexbin_panel(
        tmi, tau_k,
        "stored tmi (mean over group entries)",
        "recomputed tau, Kuzmin-sourced fitness",
        "Trigenic recapitulation, Kuzmin-only",
        summary["trigenic_kuzmin"],
        osp.join(IMG_DIR, f"recapitulation_trigenic_kuzmin_{TS}"), lim_t,
    )
    hexbin_panel(
        tmi[no_ess], tau_d[no_ess],
        "stored tmi (mean over group entries)",
        "recomputed tau from stored dmi + fitness",
        "Trigenic recapitulation, dmi-based, no essential single",
        summary["trigenic_dmi_no_ess"],
        osp.join(IMG_DIR, f"recapitulation_trigenic_dmi_no_ess_{TS}"), lim_t,
    )
    hexbin_panel(
        dmi, eps_a,
        "stored dmi",
        "recomputed eps = f_ab - f_a f_b",
        "Digenic recapitulation, closure pairs",
        summary["digenic_aggregate"],
        osp.join(IMG_DIR, f"recapitulation_digenic_aggregate_{TS}"), (-0.8, 0.8),
    )

    # residual histogram, both variants
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(55)))
    for arr, color, label in (
        (tau_a, ORANGE, "aggregate"),
        (tau_k, RED, "Kuzmin-only"),
        (tau_d, PURPLE, "dmi-based"),
    ):
        mask = np.isfinite(arr) & np.isfinite(tmi)
        ax.hist(
            (arr - tmi)[mask], bins=200, range=(-0.4, 0.4), histtype="step",
            color=color, linewidth=0.8, label=f"{label} (n={mask.sum():,})",
        )
    ax.set_yscale("log")
    ax.set_xlabel("recomputed tau - stored tmi", fontsize=6)
    ax.set_ylabel("triples", fontsize=6)
    ax.tick_params(labelsize=6)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.legend(fontsize=6, frameon=False)
    style_axis(ax)
    base = osp.join(IMG_DIR, f"recapitulation_residuals_{TS}")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    savefig_true_size_svg(fig, base + ".svg")
    plt.close(fig)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
