# experiments/043-expression-proteome-morphology/scripts/figure_043_eval_panels.py
# [[experiments.043-expression-proteome-morphology.scripts.figure_043_eval_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/figure_043_eval_panels
"""Panels of the 043 preliminary-evaluations figure (sheet 2 of ``figure_043_sheets``).

Every number is read from a file under ``results/``:

* ``baselines_043.json``: the model-free baselines per validation head and the recipe-v2
  trained arm (``gs1024_w4_bs16`` seeds 0-2, ``trained.summary``, epochs 41-50);
* ``<round>_window_means.json`` (``wandb_view_043.window_means``): per run, the mean of
  each metric over the run's registered window. The GilaHyper runs that went past epoch
  50 carry the 41-50 means under ``means_41_50`` (``means`` is then 71-80); the others
  carry 41-50 under ``means`` with ``window == [41, 50]``. Every cross-arm comparison here
  is at epochs 41-50, because ``scan_history`` truncated the resumed operator-injection
  seeds 0 and 1 at epoch 50 (their 71-80 rows are empty). Fine-tunes (``<run>_ft``) are
  read over their own last window (Aurora epochs 17-20, GilaHyper 16-20);
* ``diagnostics_043.json``: ``d1`` (composition on the 72 Sameith doubles, ``best.pt``
  of each seed) and ``d2`` (the isolate signal-loss ladder);
* ``morphology_probe_043.json``: median per-feature Pearson r of ridge probes from each
  representation to the 116 moving CalMorph features of the Ohya 2005 held-out strains.

A seed is one point; a mean over seeds carries a whisker of the sample sd (ddof 1) and
the seed count in its label. A one-seed arm is drawn as a single point and labelled
n = 1.

    python experiments/043-expression-proteome-morphology/scripts/figure_043_eval_panels.py
"""

from __future__ import annotations

import os.path as osp
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, osp.dirname(__file__))
import figure_043_common as C  # noqa: E402

DOUBLES = "val/doubles/pearson_per_strain_centered_median"
NULL = "val/doubles/additive_measured_pearson_per_strain_centered_median"
KEMMEREN = "val/kemmeren/pearson_per_gene_median"
MUENZNER = "val/muenzner/pearson_per_gene_median"
GH, AU = "ghyper_sweep_2026_10_24", "aurora_v2_2026_10_26"
WIN = "epochs 41-50"


def metric_key(ds: str) -> str:
    """The trainer's validation key of one head."""
    if ds == "vaishnav":
        return "val/vaishnav/pearson"
    if ds == "doubles":
        return DOUBLES
    return f"val/{ds}/pearson_per_gene_median"


def baseline_stat(ds: str) -> str:
    """The key of the statistic a ``baselines_043`` baseline stores for one head."""
    if ds == "vaishnav":
        return "pearson"
    if ds == "doubles":
        return "pearson_per_strain_centered_median"
    return "pearson_per_gene_median"


def shade(color: str, f: float) -> tuple[float, float, float]:
    """A solid lighter shade of ``color``: mixed with white by ``f`` (no alpha)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def mean_sd(v: list[float]) -> tuple[float, float]:
    a = np.asarray(v, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else float("nan"))


def xgrid(ax: Axes, step: float, label_every: int = 2) -> None:
    """``C.hgrid`` on the x axis, for the horizontal panels."""
    ax.xaxis.set_major_locator(MultipleLocator(step * label_every))
    ax.xaxis.set_minor_locator(MultipleLocator(step))
    ax.grid(axis="x", which="both", color="#DDDDDD", linewidth=0.4, zorder=0)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- runs


def window(run: dict[str, Any], first: int, last: int) -> dict[str, Any]:
    """The means of one run over epochs ``first``-``last`` (see the module docstring)."""
    if f"means_{first}_{last}" in run:
        return run[f"means_{first}_{last}"]
    assert run["window"] == [first, last], (run["window"], first, last)
    return run["means"]


def run_values(
    R: C.Reads, rnd: str, ids: list[str], key: str, first: int, last: int
) -> list[float]:
    """One value per run id; a run that lacks the key contributes nothing."""
    runs = R.window_means(rnd)["runs"]
    out = []
    for rid in ids:
        m = window(runs[rid], first, last)
        if key in m and np.isfinite(m[key]["mean"]):
            out.append(float(m[key]["mean"]))
    return out


def ft_values(R: C.Reads, rnd: str, ids: list[str], key: str) -> list[float]:
    """The fine-tune of each run id (``<id>_ft``) over its own registered window."""
    runs = R.window_means(rnd)["runs"]
    out = []
    for rid in ids:
        r = runs[f"{rid}_ft"]
        m = window(r, *r["window"])
        out.append(float(m[key]["mean"]))
    return out


def aurora(cond: str, loss: str) -> list[str]:
    return [f"a1026_all_{cond}_{loss}_equal_v2_s{s}" for s in range(4)]


#: the arms of the lever ladder and the fine-tune panel, top to bottom; label, round,
#: run ids (stage 1, epochs 41-50), color, whether a Kemmeren fine-tune exists
ARMS: list[tuple[str, str, list[str], Any, bool]] = [
    (
        "HL-Gauss onto, GH",
        GH,
        ["gs1024_w4_bs16", "gs1024_w4_bs16_s1", "gs1024_w4_bs16_s2"],
        C.ARM_COLOR["bs16"],
        True,
    ),
    ("HL-Gauss onto, Au", AU, aurora("onto", "hlgauss"), C.LOSS_COLOR["hlgauss"], True),
    ("HL-Gauss name, Au", AU, aurora("name", "hlgauss"), C.LOSS_COLOR["hlgauss"], True),
    ("MSE onto, Au", AU, aurora("onto", "mse"), C.LOSS_COLOR["mse"], True),
    ("MSE name, Au", AU, aurora("name", "mse"), C.LOSS_COLOR["mse"], True),
    ("pinball onto, Au", AU, aurora("onto", "pinball"), C.LOSS_COLOR["pinball"], True),
    ("pinball name, Au", AU, aurora("name", "pinball"), C.LOSS_COLOR["pinball"], True),
    (
        "HL-Gauss + inj_op, GH",
        GH,
        [
            "gs1024_w5_onto_inject_op",
            "gs1024_w5_onto_inject_op_s1",
            "gs1024_w5_onto_inject_op_s2",
        ],
        C.ARM_COLOR["inject_op"],
        True,
    ),
    ("pinball onto, GH", GH, ["gs1024_w8_pinball"], C.ARM_COLOR["pinball"], True),
    (
        "pinball + inj_op, GH",
        GH,
        ["gs1024_w8_pinball_injop"],
        C.ARM_COLOR["pinball_injop"],
        True,
    ),
    ("route B bs8, GH", GH, ["gs1024_w8_routeB_bs8"], C.ARM_COLOR["routeB"], False),
]


def dot_rows(ax: Axes, rows: list[tuple[str, list[float], Any]], labels: bool) -> None:
    """Horizontal dot-and-whisker rows: each seed an open circle, the mean a filled
    diamond with an sd whisker (none for one seed).
    """
    n = len(rows)
    for i, (_, vals, color) in enumerate(rows):
        y = n - 1 - i
        if not vals:
            continue
        mu, sd = mean_sd(vals)
        ax.scatter(
            vals,
            [y] * len(vals),
            s=7,
            facecolor="white",
            edgecolor=color,
            linewidth=0.6,
            zorder=3,
        )
        if len(vals) > 1:
            ax.errorbar(
                mu,
                y,
                xerr=sd,
                fmt="none",
                ecolor="black",
                elinewidth=0.5,
                capsize=1.5,
                capthick=0.5,
                zorder=4,
            )
            ax.scatter(
                [mu],
                [y],
                s=14,
                marker="D",
                color=color,
                edgecolor="black",
                linewidth=0.4,
                zorder=5,
            )
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{lab} ({len(v)})" for lab, v, _ in reversed(rows)] if labels else []
    )
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="y", length=0 if not labels else 2)


def ladder_rows(R: C.Reads, key: str) -> list[tuple[str, list[float], Any]]:
    return [
        (lab, run_values(R, rnd, ids, key, 41, 50), col)
        for lab, rnd, ids, col, _ in ARMS
    ]


def _ladder(
    ax: Axes, R: C.Reads, key: str, labels: bool
) -> tuple[list[tuple[str, list[float], Any]], float]:
    rows = ladder_rows(R, key)
    dot_rows(ax, rows, labels)
    C.box(ax)
    return rows, 0.0


def draw_ladder(ax: Axes, R: C.Reads) -> str:
    rows, _ = _ladder(ax, R, DOUBLES, labels=True)
    null = R.baselines["datasets"]["doubles"]["baselines"]["additive_measured"][
        "pearson_per_strain_centered_median"
    ]
    ax.axvline(null, color="black", linestyle=(0, (3, 2)), linewidth=0.6)
    ax.text(
        null - 0.004,
        len(rows) - 0.55,
        f"additive null {null:.3f}",
        ha="right",
        va="top",
        fontsize=5,
    )
    ax.set_xlim(0.10, 0.46)
    xgrid(ax, 0.05)
    ax.set_xlabel(
        f"validation doubles r, {WIN}\nrows: arm (seeds); GH GilaHyper, Au Aurora"
    )
    assert all(max(v) < null for _, v, _ in rows if v)
    C.fit_title(
        ax,
        [
            f"every arm and every seed stays under the additive null {null:.2f}",
            f"every arm stays under the null {null:.2f}",
            "all under the additive null",
        ],
    )
    return "ready"


def _ridge_ref(ax: Axes, R: C.Reads, ds: str, n: int) -> float:
    ridge = R.baselines["datasets"][ds]["baselines"]["ridge"]["pearson_per_gene_median"]
    ax.axvline(ridge, color=C.GRAY, linestyle=(0, (3, 2)), linewidth=0.6)
    ax.text(
        ridge - 0.003,
        n - 0.55,
        f"ridge {ridge:.3f}",
        ha="right",
        va="top",
        fontsize=5,
        color=C.GRAY,
    )
    return float(ridge)


def draw_ladder_muenzner(ax: Axes, R: C.Reads) -> str:
    rows, _ = _ladder(ax, R, MUENZNER, labels=False)
    ridge = _ridge_ref(ax, R, "muenzner", len(rows))
    ax.set_xlim(0.0, 0.45)
    xgrid(ax, 0.05)
    ax.set_xlabel(f"validation Muenzner r, {WIN}")
    mse = mean_sd(rows[3][1])[0]
    hlg = mean_sd(rows[1][1])[0]
    C.fit_title(
        ax,
        [
            f"MSE costs Muenzner ({mse:.2f} vs {hlg:.2f}); all arms stay under ridge {ridge:.2f}",
            f"MSE costs Muenzner; ridge {ridge:.2f} leads",
            "MSE costs Muenzner r",
        ],
    )
    return "ready"


def draw_ladder_kemmeren(ax: Axes, R: C.Reads) -> str:
    rows, _ = _ladder(ax, R, KEMMEREN, labels=False)
    ridge = _ridge_ref(ax, R, "kemmeren", len(rows))
    ax.set_xlim(0.04, 0.15)
    xgrid(ax, 0.01, label_every=4)
    ax.set_xlabel(f"validation Kemmeren r, {WIN}")
    pi = rows[9][1][0]
    C.fit_title(
        ax,
        [
            f"pinball + inject_op costs Kemmeren ({pi:.3f}, n = 1); ridge {ridge:.2f} leads",
            "pinball + inject_op costs Kemmeren",
            "pinball + inj_op costs it",
        ],
    )
    return "ready"


# ------------------------------------------------------------------------- fine-tune


def draw_finetune(ax: Axes, R: C.Reads) -> str:
    arms = [a for a in ARMS if a[4]]
    n = len(arms)
    gains: dict[str, list[float]] = {DOUBLES: [], KEMMEREN: []}
    labels = []
    for i, (lab, rnd, ids, col, _) in enumerate(arms):
        y = n - 1 - i
        for key, marker, dy in ((DOUBLES, "o", 0.14), (KEMMEREN, "s", -0.14)):
            s1 = run_values(R, rnd, ids, key, 41, 50)
            ft = ft_values(R, rnd, ids, key)
            assert len(s1) == len(ft) == len(ids), (lab, key)
            (m1, sd1), (m2, sd2) = mean_sd(s1), mean_sd(ft)
            gains[key].append(m2 - m1)
            ax.plot([m1, m2], [y + dy, y + dy], color=col, linewidth=0.8, zorder=2)
            for m, sd, face in ((m1, sd1, "white"), (m2, sd2, col)):
                if np.isfinite(sd):
                    ax.errorbar(
                        m,
                        y + dy,
                        xerr=sd,
                        fmt="none",
                        ecolor="black",
                        elinewidth=0.4,
                        capsize=1.0,
                        capthick=0.4,
                        zorder=3,
                    )
                ax.scatter(
                    [m],
                    [y + dy],
                    s=10,
                    marker=marker,
                    facecolor=face,
                    edgecolor=col if face == "white" else "black",
                    linewidth=0.5,
                    zorder=4,
                )
        labels.append(f"{lab} ({len(ids)})")
    null = R.baselines["datasets"]["doubles"]["baselines"]["additive_measured"][
        "pearson_per_strain_centered_median"
    ]
    ax.axvline(null, color="black", linestyle=(0, (3, 2)), linewidth=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0.0, 0.46)
    xgrid(ax, 0.05)
    ax.set_xlabel(
        "validation r, epochs 41-50 then the fine-tune's\nlast window; rows: arm (seeds)"
    )
    C.box(ax)
    h = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=C.GRAY,
            markersize=3,
            label="stage 1",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=C.GRAY,
            markeredgecolor="black",
            markersize=3,
            label="fine-tuned",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="black",
            markerfacecolor="none",
            markersize=3,
            label="doubles",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            color="black",
            markerfacecolor="none",
            markersize=3,
            label="Kemmeren",
        ),
        Line2D(
            [],
            [],
            color="black",
            linestyle=(0, (3, 2)),
            linewidth=0.6,
            label=f"null {null:.3f}",
        ),
    ]
    ax.legend(
        handles=h,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=3,
        fontsize=5,
        handletextpad=0.3,
        borderaxespad=0.2,
        columnspacing=0.8,
    )
    gd, gk = gains[DOUBLES], gains[KEMMEREN]
    C.fit_title(
        ax,
        [
            f"fine-tuning lifts doubles r by {min(gd):+.3f} to {max(gd):+.3f} and Kemmeren r by {min(gk):+.3f} to {max(gk):+.3f}",
            f"fine-tune: doubles {min(gd):+.2f} to {max(gd):+.2f}, Kemmeren {min(gk):+.2f} to {max(gk):+.2f}",
            f"fine-tune lifts both, doubles +{max(gd):.2f} at most",
        ],
        legend_rows=2,
    )
    return "ready"


# ------------------------------------------------------------------------- baselines


#: model-free baselines: label, face, hatch
BASE_STYLE = {
    "additive_measured": ("additive measured singles", "white", "\\\\\\\\"),
    "env_mean": ("per-environment mean", "white", "...."),
    "ridge": ("ridge on frozen tokens", C.GRAY, None),
    "knn": ("kNN, best k of 5, 20, 50", shade(C.GRAY, 0.45), "////"),
    "hgb": ("gradient-boosted trees", shade(C.GRAY, 0.75), "xxxx"),
}
TRAINED = [
    ("v2", "recipe v2, HL-Gauss, onto, GilaHyper", C.ARM_COLOR["bs16"]),
    ("au_pin", "pinball, onto, Aurora", C.ARM_COLOR["pinball"]),
    ("gh_pinj", "pinball + inject_op, GilaHyper", C.ARM_COLOR["pinball_injop"]),
]


def baseline_values(R: C.Reads, ds: str) -> dict[str, float]:
    """The finite model-free baselines of one head (``constant`` is undefined for every
    per-gene head and is left out).
    """
    b = R.baselines["datasets"][ds]["baselines"]
    st = baseline_stat(ds)
    out: dict[str, float] = {}
    for k in ("additive_measured", "env_mean", "ridge"):
        if k in b:
            out[k] = float(b[k][st])
    knn = [float(b[f"knn_{k}"][st]) for k in (5, 20, 50)]
    knn = [v for v in knn if np.isfinite(v)]
    out["knn"] = max(knn)
    out["hgb"] = float(b["hgb"][st])
    return out


def trained_values(R: C.Reads, ds: str) -> dict[str, list[float]]:
    key = metric_key(ds)
    out: dict[str, list[float]] = {
        "v2": [float(v) for v in R.baselines["trained"]["summary"][key]["per_seed"]]
    }
    au = run_values(R, AU, aurora("onto", "pinball"), key, 41, 50)
    if au:
        out["au_pin"] = au
    gh = run_values(R, GH, ["gs1024_w8_pinball_injop"], key, 41, 50)
    if gh:
        out["gh_pinj"] = gh
    return out


def draw_baselines(ax: Axes, R: C.Reads) -> str:
    fig_w_mm = (
        ax.figure.get_figure(root=True).get_figwidth() * 25.4 * ax.get_position().width
    )
    narrow = fig_w_mm < 120
    width = 0.105
    lead = {}
    for gi, (ds, label, _) in enumerate(C.DATASETS):
        bv, tv = baseline_values(R, ds), trained_values(R, ds)
        slots: list[tuple[str, Any]] = [
            ("t:" + k, tv[k]) for k, _, _ in TRAINED if k in tv
        ]
        slots += [("b:" + k, bv[k]) for k in BASE_STYLE if k in bv]
        x0 = gi - width * (len(slots) - 1) / 2
        for j, (slot, val) in enumerate(slots):
            x = x0 + j * width
            kind, k = slot.split(":")
            if kind == "b":
                _, face, hatch = BASE_STYLE[k]
                ax.bar(
                    x,
                    val,
                    width,
                    color=face,
                    hatch=hatch,
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=2,
                )
            else:
                col = dict((a, c) for a, _, c in TRAINED)[k]
                mu, sd = mean_sd(val)
                if len(val) == 1:
                    ax.scatter(
                        [x],
                        [mu],
                        s=10,
                        marker="D",
                        color=col,
                        edgecolor="black",
                        linewidth=0.4,
                        zorder=4,
                    )
                else:
                    ax.bar(
                        x,
                        mu,
                        width,
                        color=col,
                        edgecolor="black",
                        linewidth=0.4,
                        zorder=2,
                    )
                    ax.errorbar(
                        x,
                        mu,
                        yerr=sd,
                        fmt="none",
                        ecolor="black",
                        elinewidth=0.5,
                        capsize=1.2,
                        capthick=0.5,
                        zorder=3,
                    )
        best_b = max(bv.values())
        best_t = max(mean_sd(v)[0] for v in tv.values())
        lead[ds] = best_t > best_b
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_xticks(range(len(C.DATASETS)))
    if narrow:
        ax.set_xticklabels(
            [lab.split(" ")[0] for _, lab, _ in C.DATASETS],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
    else:
        ax.set_xticklabels([lab.replace(" ", "\n", 1) for _, lab, _ in C.DATASETS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(C.DATASETS) - 0.4)
    ax.set_ylim(-0.2, 1.0)
    C.hgrid(ax, 0.1)
    C.box(ax)
    ax.set_ylabel(f"validation r, {WIN}")
    n_v2 = len(R.baselines["trained"]["summary"][DOUBLES]["per_seed"])
    n_au = len(aurora("onto", "pinball"))
    h = [
        Patch(
            facecolor=C.ARM_COLOR["bs16"],
            edgecolor="black",
            linewidth=0.4,
            label=f"recipe v2, HL-Gauss, onto, GilaHyper (n = {n_v2})",
        ),
        Patch(
            facecolor=C.ARM_COLOR["pinball"],
            edgecolor="black",
            linewidth=0.4,
            label=f"pinball, onto, Aurora (n = {n_au})",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            color=C.ARM_COLOR["pinball_injop"],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=3,
            label="pinball + inject_op, GilaHyper (n = 1)",
        ),
    ]
    h += [
        Patch(facecolor=face, hatch=hatch, edgecolor="black", linewidth=0.4, label=lab)
        for lab, face, hatch in BASE_STYLE.values()
    ]
    ncol = 3 if narrow else 4
    ax.legend(
        handles=h,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=ncol,
        fontsize=5 if narrow else 6,
        borderaxespad=0.2,
        handlelength=1.4,
        columnspacing=0.8,
    )
    if not narrow:
        missing = [
            C.DATASET_LABEL[ds]
            for ds, _, _ in C.DATASETS
            if "au_pin" not in trained_values(R, ds)
        ]
        ax.text(
            0.005,
            0.97,
            "r: median per-gene Pearson; Vaishnav over promoters; doubles: median centered per-double Pearson\n"
            "bars: mean over seeds, whiskers: sd; a one-seed arm is a point\nno pinball run logs "
            + ", ".join(missing),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5,
            color="#333333",
        )
    n_rows = int(np.ceil(len(h) / ncol))
    per_gene = [ds for ds, _, _ in C.DATASETS if ds not in ("vaishnav", "doubles")]
    n_lost = sum(not lead[ds] for ds in per_gene)
    led = [C.DATASET_LABEL[ds].split(" ")[0] for ds in lead if lead[ds]]
    null = R.baselines["datasets"]["doubles"]["baselines"]["additive_measured"][
        "pearson_per_strain_centered_median"
    ]
    v2d = R.baselines["trained"]["summary"][DOUBLES]["mean"]
    C.fit_title(
        ax,
        [
            f"a model-free baseline beats every trained arm on {n_lost} of {len(per_gene)} per-gene heads; doubles sit under the additive null ({v2d:.2f} vs {null:.2f}); "
            f"the model leads only on {', '.join(led)}",
            f"a model-free baseline beats every trained arm on {n_lost} of {len(per_gene)} per-gene heads; doubles sit under the additive null ({v2d:.2f} vs {null:.2f})",
            f"a baseline leads on {n_lost} of {len(per_gene)} per-gene heads; the model on {', '.join(led)}",
        ],
        legend_rows=n_rows,
    )
    return "ready"


# ---------------------------------------------------------------------- diagnostics


D1_BARS = [
    ("model_double_vs_measured_double", "model double vs measured"),
    ("model_singles_sum_vs_measured_double", "sum of model singles vs measured"),
    ("model_double_vs_model_singles_sum", "model double vs sum of model singles"),
    ("model_single_vs_measured_single", "model single vs measured single"),
]
D1_ARMS = [
    ("bs16", "recipe v2", C.ARM_COLOR["bs16"], 3),
    ("injop", "inject_op", C.ARM_COLOR["inject_op"], 2),
]


def d1_values(R: C.Reads, arm: str, n: int, key: str) -> list[float]:
    ck = R.diagnostics["d1"]["checkpoints"]
    return [float(ck[f"{arm}_s{s}/best.pt"][key]["median"]) for s in range(n)]


def draw_composition(ax: Axes, R: C.Reads) -> str:
    width = 0.36
    for j, (arm, lab, col, n) in enumerate(D1_ARMS):
        for i, (key, _) in enumerate(D1_BARS):
            v = d1_values(R, arm, n, key)
            mu, sd = mean_sd(v)
            x = i + (j - 0.5) * width
            ax.bar(x, mu, width, color=col, edgecolor="black", linewidth=0.4, zorder=2)
            ax.errorbar(
                x,
                mu,
                yerr=sd,
                fmt="none",
                ecolor="black",
                elinewidth=0.5,
                capsize=1.2,
                capthick=0.5,
                zorder=3,
            )
            ax.scatter(
                [x] * len(v),
                v,
                s=4,
                facecolor="white",
                edgecolor="black",
                linewidth=0.4,
                zorder=4,
            )
    null = float(
        R.diagnostics["d1"]["checkpoints"]["bs16_s0/best.pt"][
            "measured_singles_sum_vs_measured_double"
        ]["median"]
    )
    ax.axhline(null, color="black", linestyle=(0, (3, 2)), linewidth=0.6)
    ax.set_xticks(range(len(D1_BARS)))
    ax.set_xticklabels(
        [
            "double\nvs\nmeasured",
            "singles\nsum vs\nmeasured",
            "double vs\nsingles\nsum",
            "single\nvs\nmeasured",
        ],
        fontsize=5,
    )
    ax.set_xlabel("model predictions; sum = the two singles added")
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.0)
    C.hgrid(ax, 0.1)
    C.box(ax)
    ax.set_ylabel("median centered per-double r\n(72 doubles; best.pt)")
    h = [
        Patch(facecolor=col, edgecolor="black", linewidth=0.4, label=f"{lab} (n = {n})")
        for _, lab, col, n in D1_ARMS
    ]  # noqa: E501
    h.append(
        Line2D(
            [],
            [],
            color="black",
            linestyle=(0, (3, 2)),
            linewidth=0.6,
            label=f"additive null {null:.3f}",
        )
    )
    ax.legend(
        handles=h, loc="upper left", fontsize=5, borderaxespad=0.3, handlelength=1.4
    )
    comp = mean_sd(d1_values(R, "bs16", 3, "model_double_vs_model_singles_sum"))[0]
    sing = mean_sd(d1_values(R, "bs16", 3, "model_single_vs_measured_single"))[0]
    C.fit_title(
        ax,
        [
            f"the double tracks the model's own singles (r {comp:.2f}); the singles are the limit (r {sing:.2f})",
            f"composes its singles (r {comp:.2f}); singles limit (r {sing:.2f})",
            "the singles are the limit",
        ],
    )
    return "ready"


D2_ROWS = [
    ("frozen_indicators", "frozen calls (ridge)"),
    ("frozen_ctx_mean", "frozen composite"),
    ("input_mean", "input, mean"),
    ("per_gene_input", "input, own gene"),
    ("trunk_mean", "trunk, mean"),
    ("trunk_cls", "trunk, CLS"),
    ("per_gene_trunk", "trunk, own gene"),
    ("op_mean", "operator, mean"),
    ("op_cls", "operator, CLS"),
    ("per_gene_op", "operator, own gene"),
    ("head_output", "model head output"),
]
D2_DS = [
    ("caudal", "Caudal 2024", C.MODALITY_COLOR["mRNA"]),
    ("muenzner", "Muenzner 2024 protein", C.MODALITY_COLOR["protein"]),
]


def d2_values(R: C.Reads, ds: str, key: str) -> list[float]:
    d2 = R.diagnostics["d2"]
    if key.startswith("frozen_"):
        return [float(d2["frozen"][ds][key]["pearson_per_gene_median"])]
    return [
        float(r["datasets"][ds]["representations"][key]["pearson_per_gene_median"])
        for r in d2["runs"].values()
    ]


def draw_isolates(ax: Axes, R: C.Reads) -> str:
    n = len(D2_ROWS)
    h = 0.38
    for j, (ds, lab, col) in enumerate(D2_DS):
        for i, (key, _) in enumerate(D2_ROWS):
            v = d2_values(R, ds, key)
            mu, sd = mean_sd(v)
            y = n - 1 - i + (0.5 - j) * h
            ax.barh(y, mu, h, color=col, edgecolor="black", linewidth=0.4, zorder=2)
            if len(v) > 1:
                ax.errorbar(
                    mu,
                    y,
                    xerr=sd,
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.5,
                    capsize=1.0,
                    capthick=0.5,
                    zorder=3,
                )
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in reversed(D2_ROWS)])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, 0.45)
    xgrid(ax, 0.05)
    C.box(ax)
    n_runs = len(R.diagnostics["d2"]["runs"])
    ax.set_xlabel(
        f"validation per-gene r, ridge per row\nbs16 mean, sd (n = {n_runs}); frozen n = 1"
    )
    hh = [
        Patch(facecolor=col, edgecolor="black", linewidth=0.4, label=lab)
        for _, lab, col in D2_DS
    ]
    ax.legend(
        handles=hh,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        fontsize=5,
        borderaxespad=0.2,
        handlelength=1.4,
    )
    fz = [d2_values(R, ds, "frozen_indicators")[0] for ds, _, _ in D2_DS]
    hd = [mean_sd(d2_values(R, ds, "head_output"))[0] for ds, _, _ in D2_DS]
    C.fit_title(
        ax,
        [
            f"signal falls from the frozen calls ({fz[0]:.2f}, {fz[1]:.2f}) to the model head ({hd[0]:.2f}, {hd[1]:.2f})",
            f"frozen calls {fz[0]:.2f}, {fz[1]:.2f}; head {hd[0]:.2f}, {hd[1]:.2f}",
            "the head loses the frozen signal",
        ],
        legend_rows=1,
    )
    return "ready"


# ------------------------------------------------------------------------ morphology


#: the model representations of the probe (``update_sum`` is ``mean_genes`` up to an
#: offset and a scale, the same ridge input, so it is not drawn); the frozen composite is
#: the reference row and line
MORPH_ROWS = [
    ("trunk_gene", "deleted gene, trunk"),
    ("pert_gene", "deleted gene, operator"),
    ("mean_genes", "gene mean"),
    ("cls", "CLS"),
    ("z_s", "mark key"),
]
MORPH_MODELS = [
    ("bs16", "recipe v2", C.ARM_COLOR["bs16"]),
    ("onto_inject_op", "inject_op", C.ARM_COLOR["inject_op"]),
]


def _morph(reps: dict[str, Any], key: str) -> tuple[float, float, float]:
    f = reps[key]["moving_116"]["fold_split"]
    lo, hi = f["ci_median_pearson"]
    return float(f["median_pearson"]), float(lo), float(hi)


def _hbar_ci(
    ax: Axes, y: float, v: tuple[float, float, float], h: float, color: Any
) -> None:
    mu, lo, hi = v
    ax.barh(y, mu, h, color=color, edgecolor="black", linewidth=0.4, zorder=2)
    ax.errorbar(
        mu,
        y,
        xerr=[[mu - lo], [hi - mu]],
        fmt="none",
        ecolor="black",
        elinewidth=0.5,
        capsize=1.0,
        capthick=0.5,
        zorder=3,
    )


def draw_morphology(ax: Axes, R: C.Reads) -> str:
    reps = R.morphology_probe["representations"]
    n = len(MORPH_ROWS) + 1
    h = 0.38
    fz = _morph(reps, "frozen_composite")
    _hbar_ci(ax, n - 1, fz, 2 * h, C.GRAY)
    ax.axvline(fz[0], color=C.GRAY, linestyle=(0, (3, 2)), linewidth=0.6, zorder=1)
    for i, (key, _) in enumerate(MORPH_ROWS):
        for j, (fam, _, col) in enumerate(MORPH_MODELS):
            _hbar_ci(
                ax, n - 2 - i + (0.5 - j) * h, _morph(reps, f"{fam}_{key}"), h, col
            )
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in reversed(MORPH_ROWS)] + ["frozen composite"])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0.0, 0.3)
    xgrid(ax, 0.05)
    C.box(ax)
    f0 = reps["frozen_composite"]["moving_116"]["fold_split"]
    ax.set_xlabel(
        f"median r, {f0['n_features_scored']} CalMorph features, {f0['n_test_strains']}\nheld-out strains; whiskers 95% bootstrap"
    )
    hh = [
        Patch(facecolor=col, edgecolor="black", linewidth=0.4, label=f"{lab} (seed 0)")
        for _, lab, col in MORPH_MODELS
    ]
    ax.legend(
        handles=hh,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        fontsize=5,
        borderaxespad=0.2,
        handlelength=1.4,
    )
    best_key = max(
        (f"{fam}_{k}" for fam, _, _ in MORPH_MODELS for k, _ in MORPH_ROWS),
        key=lambda k: _morph(reps, k)[0],
    )
    best = _morph(reps, best_key)[0]
    C.fit_title(
        ax,
        [
            f"trained bs16 tokens map to morphology at {best:.2f}, the frozen composite at {fz[0]:.2f}",
            f"morphology: bs16 {best:.2f} vs frozen {fz[0]:.2f}",
            f"bs16 {best:.2f} vs frozen {fz[0]:.2f}",
        ],
        legend_rows=1,
    )
    return "ready"


def draw_scatter(ax: Axes, R: C.Reads) -> str:
    return C.placeholder(
        ax,
        "predicted vs measured per-gene profile for one held-out\nKemmeren deletion and one Sameith double,\nmodel vs additive null",
        "profiles",
    )


PANELS: dict[str, Callable[[Axes, C.Reads], str]] = {
    "baselines": draw_baselines,
    "ladder": draw_ladder,
    "ladder_muenzner": draw_ladder_muenzner,
    "ladder_kemmeren": draw_ladder_kemmeren,
    "finetune": draw_finetune,
    "composition": draw_composition,
    "isolates": draw_isolates,
    "morphology": draw_morphology,
    "scatter": draw_scatter,
}


def full_width(key: str, height_mm: float, name: str) -> dict[str, str]:
    """One panel alone at the full 179 mm width, for checking the baselines strip."""
    R = C.Reads()
    fig = C.new_sheet(height_mm)
    (ax,) = C.rows(fig, 1, 1)
    st = PANELS[key](ax, R)
    C.letters_status(fig, [ax], [st])
    C.status_key(fig)
    return C.save(fig, name)


def main() -> None:
    out = C.demo(PANELS, "figure_043_eval_panels")
    print(out["png"])
    out = full_width("baselines", 62.0, "figure_043_eval_baselines")
    print(out["png"])


if __name__ == "__main__":
    main()
