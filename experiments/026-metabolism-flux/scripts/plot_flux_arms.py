# experiments/026-metabolism-flux/scripts/plot_flux_arms.py
# [[experiments.026-metabolism-flux.scripts.plot_flux_arms]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_flux_arms.py

r"""Plot the flux-layer arm comparison and the feasibility traces.

Run from the worktree root, after ``train_flux.py``::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/plot_flux_arms.py

Three panels, and the third is the one that keeps the first two honest:

a. **Prediction.** Peak validation Pearson per arm, per phenotype, with the seed spread
   drawn as individual points rather than an error bar. With three seeds an error bar
   implies a distribution nobody measured; three dots do not.
b. **Feasibility.** Mass-balance residual and second-law violation fraction over training,
   per arm. A flux vector that fits the phenotype while violating mass balance has not
   learned metabolism, and this panel is where that shows.
c. **The exactness budget.** Mass-balance residual against box-violation fraction, one
   point per arm. The box arm sits on one axis at zero, the null-space arm on the other.
   **Neither can sit at the origin** without a projection step, and seeing that is the
   argument for where the budget should be spent.
"""

import json
import os
import os.path as osp
from collections import defaultdict
from glob import glob
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

RESULTS_DIR = osp.join(os.environ["EXPERIMENT_ROOT"], "026-metabolism-flux", "results")
IMAGES_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "026-metabolism-flux")

ARM_ORDER = ["pooled", "flux_off", "flux_free", "flux_anchored", "flux_nullspace"]
#: Short labels. A five-category axis at 88 mm has about 16 mm per tick, so anything
#: longer than roughly ten characters per line collides with its neighbours.
ARM_LABEL = {
    "pooled": "pooled",
    "flux_off": "flux\nbare",
    "flux_free": "flux\nfree $\\mu$",
    "flux_anchored": "flux\nanchored",
    "flux_nullspace": "flux\nnull sp.",
}
#: Betaxanthin replicate-based noise ceiling, from
#: experiments/019-simb-multimodal/results/pigment_noise_ceiling.json.
BETAXANTHIN_CEILING = 0.914


def _style(ax: plt.Axes) -> None:
    """Boxed axes, 0.5 pt spines, tenth gridlines."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    ax.tick_params(labelsize=6, width=0.5, length=2)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def load_runs() -> list[dict[str, Any]]:
    """Read every checkpointed arm-sweep result file in the results directory.

    The sweep runs as two processes, one per GPU, each writing its own file, and each
    rewrites its file after every completed (arm, seed). So a partial sweep is readable and
    the union across files is the experiment. Globbing rather than naming the files keeps a
    third GPU or a follow-up confirmation run from needing a code change.
    """
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    # Oldest file first, so a later file's run REPLACES an earlier one for the same
    # (arm, seed). This is what keeps a corrected rerun from being averaged together with
    # the run it corrects: the null-space arm was re-run after the collapse-detection fix
    # to `masked_pearson`, and silently pooling both would report a mean over two
    # different metrics.
    for path in sorted(
        glob(osp.join(RESULTS_DIR, "flux_arms*.json")), key=osp.getmtime
    ):
        if path.endswith("_summary.json"):
            continue
        with open(path) as f:
            payload = json.load(f)
        for run in payload.get("runs", []):
            run["_source_file"] = osp.basename(path)
            latest[(run["arm"], int(run["seed"]))] = run
    return list(latest.values())


def main() -> None:
    """Build the three-panel arm comparison from every checkpointed sweep file."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    runs = load_runs()
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_arm[r["arm"]].append(r)
    arms = [a for a in ARM_ORDER if a in by_arm]
    print(f"loaded {len(runs)} runs across {len(arms)} arms")

    plt.rcParams.update({"font.family": "Arial", "font.size": 6, "svg.fonttype": "none"})
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(54.0))
    )

    # -- a. prediction --------------------------------------------------------
    ax = axes[0]
    width = 0.36
    x = np.arange(len(arms))
    # Two series take the first two palette entries, per the repo ordering rule.
    for k, (metric, color) in enumerate(
        [("val_betaxanthin", PLOT_PALETTE[0]), ("val_mulleder19", PLOT_PALETTE[1])]
    ):
        means, points = [], []
        for a in arms:
            vals = [r["best"][metric] for r in by_arm[a]]
            means.append(float(np.mean(vals)))
            points.append(vals)
        ax.bar(
            x + (k - 0.5) * width,
            means,
            width,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            label="betaxanthin" if k == 0 else "amino acids (19)",
        )
        for xi, vals in zip(x + (k - 0.5) * width, points):
            ax.scatter(
                [xi] * len(vals), vals, s=3, color="black", zorder=3, linewidths=0
            )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a] for a in arms], fontsize=6)
    ax.set_ylabel("peak validation Pearson $r$")
    ax.set_ylim(-0.10, 0.20)
    ax.legend(fontsize=6, frameon=False, loc="upper left", handlelength=1.2)
    ax.text(
        0.98,
        0.97,
        f"noise ceiling $r$ = {BETAXANTHIN_CEILING}",
        transform=ax.transAxes,
        fontsize=6,
        va="top",
        ha="right",
    )
    _style(ax)

    # -- b. feasibility over training ----------------------------------------
    ax = axes[1]
    for i, a in enumerate(arms):
        hist = by_arm[a][0]["history"]
        bal = [h.get("feas_balance_median") for h in hist]
        if not any(v is not None for v in bal):
            continue
        ax.plot(
            [h["epoch"] for h in hist],
            bal,
            color=PLOT_PALETTE[i],
            linewidth=0.8,
            label=a.replace("_", " "),
        )
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"median $|[Sv]_i| / \omega_i$")
    ax.legend(fontsize=6, frameon=False)
    _style(ax)

    # -- c. the exactness budget ---------------------------------------------
    # The three box-parameterized arms land on ONE point by construction, so annotating
    # each at its own coordinate stacks three labels on top of each other. Group the
    # arms by their (balance, box) corner and label the corner once.
    ax = axes[2]
    corners: dict[tuple[float, float], list[str]] = defaultdict(list)
    for a in arms:
        hist = by_arm[a][0]["history"]
        if not hist or "feas_balance_median" not in hist[-1]:
            continue
        bal = max(hist[-1]["feas_balance_median"], 1e-12)
        box = max(hist[-1].get("feas_box_violation_frac", 0.0), 1e-12)
        # Round to a decade so numerically identical corners collapse together.
        corners[(round(np.log10(bal), 1), round(np.log10(box), 1))].append(a)
    for i, ((lb, lx), members) in enumerate(sorted(corners.items())):
        ax.scatter(
            10**lb,
            10**lx,
            s=22,
            color=PLOT_PALETTE[i],
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )
        label = "\n".join(m.replace("flux_", "").replace("_", " ") for m in members)
        # Label inward. A corner sitting at the right edge of the axis has no room to its
        # right, and the box arms sit at a residual of ~2, which IS the right edge.
        # Label inward on BOTH axes. A corner at the right edge has no room to its right,
        # and one at the bottom edge has none below, where it would land on the tick labels.
        right = lb > -4
        low = lx < -6
        ax.annotate(
            label,
            (10**lb, 10**lx),
            fontsize=6,
            xytext=(-5 if right else 5, 4 if low else -2),
            textcoords="offset points",
            va="bottom" if low else "top",
            ha="right" if right else "left",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-9, 1e2)
    ax.set_ylim(1e-13, 1e1)
    ax.set_xlabel(r"mass-balance residual (soft $\to$ right)")
    ax.set_ylabel(r"box violation fraction (soft $\to$ up)")
    _style(ax)

    fig.tight_layout()
    stem = osp.join(IMAGES_DIR, "flux_arm_comparison")
    fig.savefig(f"{stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}.svg")
    plt.close(fig)
    print(f"wrote {stem}.svg")

    # -- summary table --------------------------------------------------------
    summary = {}
    for a in arms:
        summary[a] = {
            "n_seeds": len(by_arm[a]),
            "betaxanthin_mean": float(
                np.mean([r["best"]["val_betaxanthin"] for r in by_arm[a]])
            ),
            "betaxanthin_sd": float(
                np.std([r["best"]["val_betaxanthin"] for r in by_arm[a]], ddof=1)
                if len(by_arm[a]) > 1
                else 0.0
            ),
            "betaxanthin_per_seed": [r["best"]["val_betaxanthin"] for r in by_arm[a]],
            "mulleder19_mean": float(
                np.mean([r["best"]["val_mulleder19"] for r in by_arm[a]])
            ),
            "mulleder19_per_seed": [r["best"]["val_mulleder19"] for r in by_arm[a]],
            "final_feasibility": {
                k: v
                for k, v in (by_arm[a][0]["history"][-1] or {}).items()
                if k.startswith(("feas_", "c_", "g_diss", "protein_used"))
            },
            "wall_time_s_per_seed": [r["wall_time_s"] for r in by_arm[a]],
        }
    with open(osp.join(RESULTS_DIR, "flux_arms_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
