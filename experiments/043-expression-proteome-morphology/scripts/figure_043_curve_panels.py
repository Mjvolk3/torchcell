# experiments/043-expression-proteome-morphology/scripts/figure_043_curve_panels.py
# [[experiments.043-expression-proteome-morphology.scripts.figure_043_curve_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/figure_043_curve_panels
"""Pretraining-curve panels of the 043 setup-and-pretraining figure.

Every curve is a W&B history read from one cache, ``results/figure_043_curves.csv``,
written by ``--fetch``; without it the panels read only the cache. The cache is long
format, one row per (run, metric, logged step), with the run's arm, loss family,
conditioning, injection point, batch, seed, machine, W&B state and last epoch.

Histories come from ``Run.history(keys=[metric], samples=2000)`` one metric at a time,
not ``scan_history`` (it truncates resumed runs at epoch 50 on this project). A metric
logged per optimizer step (``train/grad_norm``) carries no ``epoch``; its epoch is the
fractional position of its step between the steps of the epoch-level rows.

Panels follow the 040 thesis sheet's pretraining rows: the mean over seeds with the
seeds' min-max band, one color per arm, a dotted reference line.
"""

from __future__ import annotations

import argparse
import os.path as osp
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

sys.path.insert(0, osp.dirname(__file__))
import figure_043_common as C  # noqa: E402

CACHE = osp.join(C.R043, "figure_043_curves.csv")

METRICS: tuple[str, ...] = (
    "train/loss",
    "val/loss",
    "val/kemmeren/pearson_per_gene_median",
    "val/doubles/pearson_per_strain_centered_median",
    "val/doubles/additive_measured_pearson_per_strain_centered_median",
    "val/muenzner/pearson_per_gene_median",
    "val/caudal/pearson_per_gene_median",
    "val/messner/pearson_per_gene_median",
    "val/vaishnav/pearson",
    "operator/relative_norm",
    "train/grad_norm",
)
DOUBLES = "val/doubles/pearson_per_strain_centered_median"
NULL = "val/doubles/additive_measured_pearson_per_strain_centered_median"
OPNORM = "operator/relative_norm"

#: the machine of each round prefix (the run id is <prefix>_<cell>, wandb_view_043.ROUNDS)
MACHINE = {"a1026": "Aurora", "gs1024": "GilaHyper", "o1023": "Polaris"}
ROUND = {
    "a1026": "aurora_v2_2026_10_26",
    "gs1024": "ghyper_sweep_2026_10_24",
    "o1023": "ontology_2026_10_23",
}


def _runs() -> list[tuple[str, str]]:
    """(run id, arm) of every run the panels draw; run id = W&B id = resume_id."""
    out: list[tuple[str, str]] = []
    for cond in ("onto", "name"):
        for loss in ("hlgauss", "pinball", "mse"):
            for s in range(4):
                out.append(
                    (f"a1026_all_{cond}_{loss}_equal_v2_s{s}", f"aurora_{cond}_{loss}")
                )
    for sfx in ("", "_s1", "_s2"):
        out.append((f"gs1024_w4_bs16{sfx}", "bs16"))
        out.append((f"gs1024_w5_onto_inject_op{sfx}", "inject_op"))
    out += [
        ("gs1024_w8_pinball", "pinball_gh"),
        ("gs1024_w8_pinball_injop", "pinball_injop"),
        ("gs1024_w8_routeB_bs8", "routeB_bs8"),
        ("gs1024_w7_routeB", "routeB"),
        ("gs1024_w7_routeB_s1", "routeB"),
        ("gs1024_w7_routeB_lrhalf", "routeB_lrhalf"),
    ]
    for b in (16, 32):
        for s in range(5):
            out.append((f"o1023_all_onto_hlgauss_equal_bs{b}_s{s}", f"polaris_bs{b}"))
    return out


def fetch() -> pd.DataFrame:
    """Read every run's histories from W&B into the long-format cache."""
    import wandb

    api = wandb.Api(timeout=120)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for rid, arm in _runs():
        try:
            run = api.run(f"{C.ENTITY}/{C.PROJECT}/{rid}")
        except (wandb.errors.CommError, ValueError):
            missing.append(rid)
            print(f"  NOT FOUND {rid}", flush=True)
            continue
        cfg = run.config
        ep = run.history(keys=["epoch"], samples=2000, pandas=True)
        ep = ep.dropna(subset=["epoch"]).sort_values("_step")
        # the step at which each whole epoch was last logged: the epoch-level clock
        ends = ep.groupby("epoch")["_step"].max().reset_index().sort_values("_step")
        rows: list[pd.DataFrame] = []
        for key in METRICS:
            # _timestamp is returned only when asked for; it rides on the val/loss rows
            ks = [key, "_timestamp"] if key == "val/loss" else [key]
            h = run.history(keys=ks, samples=2000, pandas=True)
            if h.empty:
                continue
            h = h.dropna(subset=[key]).sort_values("_step")
            exact = h["_step"].map(dict(zip(ends["_step"], ends["epoch"], strict=True)))
            frac = np.interp(h["_step"], ends["_step"], ends["epoch"])
            epoch = exact.fillna(pd.Series(frac, index=h.index))
            rows.append(
                pd.DataFrame(
                    {"step": h["_step"], "epoch": epoch, "metric": key, "value": h[key]}
                )
            )
            if key == "val/loss":
                rows.append(
                    pd.DataFrame(
                        {
                            "step": h["_step"],
                            "epoch": epoch,
                            "metric": "_timestamp",
                            "value": h["_timestamp"],
                        }
                    )
                )
        df = pd.concat(rows, ignore_index=True)
        prefix = rid.split("_")[0]
        df.insert(0, "run_id", rid)
        df.insert(1, "run_name", run.name)
        df.insert(2, "round", ROUND[prefix])
        df.insert(3, "arm", arm)
        df.insert(4, "loss_family", cfg["loss"]["kind"])
        df.insert(5, "conditioning", cfg["model"]["conditioning"]["source"])
        # runs that predate an option carry no key for it: recorded as "unset", not guessed
        df.insert(6, "inject", cfg["model"]["conditioning"].get("inject", "unset"))
        df.insert(7, "batch", cfg["data"]["batch_size"])
        df.insert(8, "seed", cfg["seed"])
        df.insert(9, "machine", MACHINE[prefix])
        df.insert(10, "route", cfg["data"].get("route", "unset"))
        df.insert(11, "state", run.state)
        df.insert(12, "last_epoch", int(ends["epoch"].max()))
        frames.append(df)
        print(
            f"  {rid}: {run.state}, last epoch {int(ends['epoch'].max())}, {len(df)} rows",
            flush=True,
        )
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(CACHE, index=False)
    print(
        f"cache {CACHE}: {len(out)} rows, {out.run_id.nunique()} runs; missing {missing}"
    )
    return out


_CURVES: pd.DataFrame | None = None


def curves() -> pd.DataFrame:
    global _CURVES
    if _CURVES is None:
        _CURVES = pd.read_csv(CACHE)
    return _CURVES


def series(rid: str, metric: str) -> pd.Series:
    """One run's metric indexed by epoch (epoch-level rows only)."""
    d = curves()
    d = d[(d.run_id == rid) & (d.metric == metric)]
    return d.groupby("epoch")["value"].last()


def seed_stats(rids: list[str], metric: str) -> pd.DataFrame:
    """Mean, min, max and seed count per epoch over the runs that reached it."""
    m = pd.concat({r: series(r, metric) for r in rids}, axis=1)
    return pd.DataFrame(
        {
            "mean": m.mean(axis=1),
            "lo": m.min(axis=1),
            "hi": m.max(axis=1),
            "n": m.count(axis=1),
        }
    )


def last_epoch(rid: str) -> int:
    return int(curves().loc[curves().run_id == rid, "last_epoch"].iloc[0])


def draw_mean(
    ax: Axes,
    st: pd.DataFrame,
    color: str,
    ls: str = "-",
    band: bool = True,
    lw: float = 0.8,
) -> None:
    """Mean line where every seed is present, a thinner continuation where fewer are."""
    full = st[st.n == st.n.max()]
    ax.plot(full.index, full["mean"], color=color, ls=ls, lw=lw)
    if band and st.n.max() > 1:
        ax.fill_between(
            full.index, full["lo"], full["hi"], color=color, alpha=0.18, lw=0
        )
    rest = st[st.index >= full.index.max()]
    if len(rest) > 1:
        ax.plot(rest.index, rest["mean"], color=color, ls=ls, lw=0.5, alpha=0.6)


def seed_note(rids: list[str]) -> str:
    """'n seeds' or 'n seeds to E, m to F' from the runs' last epochs."""
    ends = sorted((last_epoch(r) for r in rids), reverse=True)
    if len(set(ends)) == 1:
        return f"{len(rids)} seed{'s' if len(rids) > 1 else ''}"
    low = min(ends)
    return f"{len(rids)} seeds to {low}, {sum(e > low for e in ends)} to {max(ends)}"


def epoch_axis(ax: Axes, n: int) -> None:
    ax.set_xlim(0, n)
    ax.set_xlabel("epoch")
    C.box(ax)


AURORA = {
    loss: [f"a1026_all_onto_{loss}_equal_v2_s{s}" for s in range(4)]
    for loss in ("hlgauss", "pinball", "mse")
}
BS16 = ["gs1024_w4_bs16", "gs1024_w4_bs16_s1", "gs1024_w4_bs16_s2"]
INJOP = [
    "gs1024_w5_onto_inject_op",
    "gs1024_w5_onto_inject_op_s1",
    "gs1024_w5_onto_inject_op_s2",
]


def legend_text(ax: Axes, text: str, x: float, y: float, ha: str = "left") -> None:
    """A gray note inside the axes (seed counts, off-axis values)."""
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va="top", fontsize=5.5, color=C.GRAY
    )


def _loss_family(ax: Axes, loss: str) -> str:
    """Train (dashed) and validation (solid) loss of one loss family, Aurora recipe v2
    under ontology conditioning, mean over 4 seeds with the min-max band.
    """
    rids = AURORA[loss]
    col = C.LOSS_COLOR[loss]
    tr, va = seed_stats(rids, "train/loss"), seed_stats(rids, "val/loss")
    draw_mean(ax, tr, col, ls="--")
    draw_mean(ax, va, col)
    vmin = int(va["mean"].idxmin())
    last = int(va.index.max())
    ax.axvline(vmin, color=C.GRAY, ls=":", lw=0.6)
    epoch_axis(ax, 50)
    ax.set_ylabel(f"{C.LOSS_LABEL[loss]} loss")
    ax.legend(
        handles=[
            Line2D([], [], color=col, ls="--", label="train"),
            Line2D([], [], color=col, label="validation"),
            Line2D([], [], color=C.GRAY, ls=":", lw=0.6, label="val minimum"),
        ],
        # between the curves: validation sits above 0.6 of the axes, train below 0.2
        loc="lower right",
        bbox_to_anchor=(1.0, 0.2),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        title=f"Aurora, {seed_note(rids)}",
        title_fontsize=5.5,
    )
    gap = va["mean"].iloc[-1] - tr["mean"].iloc[-1]
    where = f"still falling at {last}" if vmin == last else f"bottoms at epoch {vmin}"
    short = f"min at {last}" if vmin == last else f"min at epoch {vmin}"
    C.fit_title(
        ax,
        [
            f"{C.LOSS_LABEL[loss]} val {where}, gap {gap:.2f}",
            f"{C.LOSS_LABEL[loss]} val {short}, gap {gap:.2f}",
            f"val {short}, gap {gap:.2f}",
        ],
    )
    return "ready"


def loss(ax: Axes, R: C.Reads) -> str:
    return _loss_family(ax, "hlgauss")


def loss_hlgauss(ax: Axes, R: C.Reads) -> str:
    return _loss_family(ax, "hlgauss")


def loss_pinball(ax: Axes, R: C.Reads) -> str:
    return _loss_family(ax, "pinball")


def loss_mse(ax: Axes, R: C.Reads) -> str:
    return _loss_family(ax, "mse")


HEADS = [
    ("doubles", DOUBLES, C.AMBER),
    ("muenzner", "val/muenzner/pearson_per_gene_median", C.BRICK),
    ("caudal", "val/caudal/pearson_per_gene_median", C.PURPLE),
    ("kemmeren", "val/kemmeren/pearson_per_gene_median", C.BLUE),
    ("messner", "val/messner/pearson_per_gene_median", C.GRAY),
]
HEAD_LABEL = {
    "doubles": "doubles",
    "muenzner": "Muenzner",
    "caudal": "Caudal",
    "kemmeren": "Kemmeren",
    "messner": "Messner",
}


def heads(ax: Axes, R: C.Reads) -> str:
    """Validation r per head of the recipe-v2 baseline (GilaHyper, ontology, HL-Gauss,
    batch 16): mean over the seeds that reached each epoch, band = min-max where all
    three are present, thin continuation where two are.
    """
    handles = []
    peaks = []
    for key, metric, col in HEADS:
        st = seed_stats(BS16, metric)
        draw_mean(ax, st, col)
        full = st[st.n == st.n.max()]
        peaks.append(int(full["mean"].idxmax()))
        handles.append(Line2D([], [], color=col, label=HEAD_LABEL[key]))
    null = seed_stats(BS16, NULL)["mean"].mean()
    ax.axhline(null, color="black", ls=":", lw=0.6)
    handles.append(Line2D([], [], color="black", ls=":", lw=0.6, label="doubles null"))
    va = seed_stats(BS16, "val/vaishnav/pearson")
    v50 = va.loc[50, "mean"]
    d50 = seed_stats(BS16, DOUBLES).loc[50, "mean"]
    ax.set_ylim(-0.05, 0.75)
    C.hgrid(ax, 0.05, 2)
    epoch_axis(ax, 80)
    ax.set_ylabel("validation Pearson r")
    ax.legend(
        handles=handles, loc="upper left", ncol=2, fontsize=5.5, columnspacing=0.8
    )
    legend_text(ax, f"bs16, {seed_note(BS16)}", 0.98, 0.58, ha="right")
    legend_text(ax, f"Vaishnav {v50:.2f} at 50, off axis", 0.98, 0.50, ha="right")
    C.fit_title(
        ax,
        [
            f"heads peak at {min(peaks)}-{max(peaks)}; doubles {d50:.2f} < null {null:.2f}",
            f"heads peak at {min(peaks)}-{max(peaks)}; doubles < null",
        ],
    )
    return "ready"


def levers(ax: Axes, R: C.Reads) -> str:
    """Doubles r of the arms that moved doubles over the bs16 baseline."""
    arms = [
        ("bs16", BS16, "HL-Gauss", "-"),
        # dashed: ARM_COLOR gives injection SAND, close to the baseline's AMBER
        ("inject_op", INJOP, "+ injection", "--"),
        ("pinball", AURORA["pinball"], "pinball", "-"),
        ("pinball_injop", ["gs1024_w8_pinball_injop"], "pinball + inj.", "-"),
    ]
    handles = []
    at50 = {}
    for arm, rids, lab, ls in arms:
        st = seed_stats(rids, DOUBLES)
        draw_mean(ax, st, C.ARM_COLOR[arm], ls=ls)
        at50[arm] = st.loc[50, "mean"]
        handles.append(
            Line2D([], [], color=C.ARM_COLOR[arm], ls=ls, label=f"{lab} ({len(rids)})")
        )
    null = seed_stats(BS16, NULL)["mean"].mean()
    ax.axhline(null, color="black", ls=":", lw=0.6)
    handles.append(Line2D([], [], color="black", ls=":", lw=0.6, label="additive null"))
    ax.set_ylim(0.0, 0.75)
    C.hgrid(ax, 0.05, 2)
    epoch_axis(ax, 80)
    ax.set_ylabel("doubles r")
    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=2,
        fontsize=5.5,
        columnspacing=0.8,
        title="(seeds); thin line: fewer seeds",
        title_fontsize=5.5,
    )
    C.fit_title(
        ax,
        [
            f"doubles at 50: injection {at50['inject_op']:.2f}, bs16 {at50['bs16']:.2f}",
            f"doubles at 50: injection {at50['inject_op']:.2f} vs {at50['bs16']:.2f}",
        ],
    )
    return "ready"


ROUTE_B = [
    ("gs1024_w7_routeB", "B bs16 s0", C.BLUE, "-"),
    ("gs1024_w7_routeB_s1", "B bs16 s1", C.BLUE, "--"),
    ("gs1024_w7_routeB_lrhalf", "B lr/2", C.PURPLE, "-"),
    ("gs1024_w8_routeB_bs8", "B bs8", C.BRICK, "-"),
    ("gs1024_w4_bs16", "A bs16", C.AMBER, "-"),
]
#: operator relative norm above which a run is called locked (route B and batch 32 runs
#: pass it within a few epochs and keep rising; every unlocked run stays under 2)
LOCK = 10.0


def _run_label(rid: str, lab: str) -> str:
    d = curves()
    st = d.loc[d.run_id == rid, "state"].iloc[0]
    n = last_epoch(rid)
    planned = 80 if rid.startswith("o1023") else 50
    return lab if (st == "finished" and n >= planned) else f"{lab} (to {n})"


def _lock_epoch(rid: str) -> int | None:
    s = series(rid, OPNORM)
    hit = s[s > LOCK]
    return int(hit.index.min()) if len(hit) else None


def collapse(ax: Axes, R: C.Reads) -> str:
    """Operator relative norm, log axis, route B runs with route A bs16 for reference."""
    handles = []
    for rid, lab, col, ls in ROUTE_B:
        s = series(rid, OPNORM)
        ax.plot(s.index, s.values, color=col, ls=ls)
        handles.append(Line2D([], [], color=col, ls=ls, label=_run_label(rid, lab)))
    ax.axhline(LOCK, color="black", ls=":", lw=0.6)
    ax.set_yscale("log")
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_ylim(1e-3, 1e4)
    ax.grid(axis="y", which="major", color="#DDDDDD", lw=0.4)
    ax.set_axisbelow(True)
    epoch_axis(ax, 80)
    ax.set_ylabel("operator relative norm")
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        fontsize=5.5,
        columnspacing=0.8,
        title="route, batch",
        title_fontsize=5.5,
    )
    locks = {lab: _lock_epoch(rid) for rid, lab, _, _ in ROUTE_B}
    bs8_max = series("gs1024_w8_routeB_bs8", OPNORM).max()
    C.fit_title(
        ax,
        [
            f"B bs16 locks at {locks['B bs16 s0']}, {locks['B bs16 s1']}; bs8 peaks at {bs8_max:.1f}",
            f"route B locks at {locks['B bs16 s0']}, {locks['B bs16 s1']}; bs8 holds",
        ],
    )
    return "ready"


def collapse_doubles(ax: Axes, R: C.Reads) -> str:
    """Validation doubles r of the same runs; a locked run repeats one value exactly."""
    handles = []
    for rid, lab, col, ls in ROUTE_B:
        s = series(rid, DOUBLES)
        ax.plot(s.index, s.values, color=col, ls=ls)
        handles.append(Line2D([], [], color=col, ls=ls, label=_run_label(rid, lab)))
    flat = series("gs1024_w7_routeB", DOUBLES)
    lock = flat.iloc[-1]
    n_same = int((flat == lock).sum())
    ax.set_ylim(-0.15, 0.55)
    C.hgrid(ax, 0.05, 2)
    epoch_axis(ax, 80)
    ax.set_ylabel("doubles r")
    ax.legend(
        handles=handles,
        loc="upper right",
        ncol=2,
        fontsize=5.5,
        columnspacing=0.8,
        title="route, batch",
        title_fontsize=5.5,
    )
    C.title(ax, f"locked: r {lock:.4f} for {n_same} epochs; bs8 learns")
    return "ready"


def bs32(ax: Axes, R: C.Reads) -> str:
    """Polaris HL-Gauss batch 32 vs batch 16, 5 seeds each: operator norm, log axis; an x
    marks the last logged epoch of a run that ended before epoch 80.
    """
    deaths = []
    locked = 0
    for b, col in ((16, C.AMBER), (32, C.BRICK)):
        for s in range(5):
            rid = f"o1023_all_onto_hlgauss_equal_bs{b}_s{s}"
            v = series(rid, OPNORM)
            ax.plot(v.index, v.values, color=col, lw=0.6)
            if b == 32 and _lock_epoch(rid) is not None:
                locked += 1
            if last_epoch(rid) < 80:
                ax.plot(
                    v.index[-1],
                    v.values[-1],
                    marker="x",
                    color="black",
                    ms=3.5,
                    mew=0.8,
                )
                deaths.append(last_epoch(rid))
    ax.axhline(LOCK, color="black", ls=":", lw=0.6)
    ax.set_yscale("log")
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_ylim(1e-3, 1e6)
    ax.grid(axis="y", which="major", color="#DDDDDD", lw=0.4)
    ax.set_axisbelow(True)
    epoch_axis(ax, 80)
    ax.set_ylabel("operator relative norm")
    ax.legend(
        handles=[
            Line2D([], [], color=C.AMBER, label="batch 16, 5 seeds"),
            Line2D([], [], color=C.BRICK, label="batch 32, 5 seeds"),
            Line2D([], [], color="black", marker="x", ls="", ms=3.5, label="run ended"),
        ],
        loc="lower right",
        fontsize=5.5,
    )
    C.fit_title(
        ax,
        [
            f"batch 32: {locked} of 5 lock, {len(deaths)} die; batch 16 never",
            f"batch 32 locks {locked} of 5; batch 16 never",
            f"batch 32 locks {locked} of 5 seeds",
        ],
    )
    return "ready"


def throughput(ax: Axes, R: C.Reads) -> str:
    """Seconds per epoch: per run, the median over consecutive-epoch gaps between the
    W&B timestamps of the val/loss logs (one per epoch; a resume gap drops out of the
    median); bar = median over the runs of a machine and batch, whisker = their range.
    """
    d = curves()
    t = d[d.metric == "_timestamp"].sort_values(["run_id", "epoch"])
    per_run = []
    for rid, g in t.groupby("run_id"):
        dt = np.diff(g["value"].to_numpy())[np.diff(g["epoch"].to_numpy()) == 1]
        r0 = g.iloc[0]
        per_run.append(
            {
                "run": rid,
                "machine": r0.machine,
                "batch": r0.batch,
                "route": r0.route,
                "s": np.median(dt),
            }
        )
    pr = pd.DataFrame(per_run)
    grp = (
        pr.groupby(["machine", "batch", "route"])["s"]
        .agg(["median", "min", "max", "count"])
        .reset_index()
        .sort_values("median")
        .reset_index(drop=True)
    )
    col = {"GilaHyper": C.AMBER, "Polaris": C.BRICK, "Aurora": C.PURPLE}
    y = np.arange(len(grp))
    ax.barh(
        y,
        grp["median"],
        color=[col[m] for m in grp.machine],
        edgecolor="black",
        lw=0.4,
        height=0.6,
    )
    ax.errorbar(
        grp["median"],
        y,
        xerr=[grp["median"] - grp["min"], grp["max"] - grp["median"]],
        fmt="none",
        ecolor="black",
        elinewidth=0.5,
        capsize=1.5,
    )
    for yi, r in zip(y, grp.itertuples(), strict=True):
        ax.text(
            r.max + 40, yi, f"{r.median:,.0f} s ({r.count})", va="center", fontsize=5.5
        )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{r.machine} {r.batch}{' B' if r.route == 'B' else ''}"
            for r in grp.itertuples()
        ]
    )
    ax.invert_yaxis()
    ax.set_xlim(0, grp["max"].max() * 1.5)
    ax.set_xlabel("s per epoch, one card (runs); batch, B = route B")
    ax.grid(axis="x", color="#DDDDDD", lw=0.4)
    ax.set_axisbelow(True)
    C.box(ax)
    g16 = grp[
        (grp.machine == "GilaHyper") & (grp.batch == 16) & (grp.route == "A")
    ].iloc[0]
    a16 = grp[(grp.machine == "Aurora") & (grp.batch == 16)].iloc[0]
    C.fit_title(
        ax,
        [
            f"bs16 epoch: GilaHyper {g16['median']:.0f} s, Aurora {a16['median']:,.0f} s",
            f"an epoch: GilaHyper {g16['median']:.0f} s, Aurora {a16['median']:,.0f} s",
            f"epoch: {g16['median']:.0f} s on a card, {a16['median']:,.0f} s on a tile",
        ],
    )
    return "ready"


PANELS: dict[str, Callable[[Axes, C.Reads], str]] = {
    "loss": loss,
    "loss_hlgauss": loss_hlgauss,
    "loss_pinball": loss_pinball,
    "loss_mse": loss_mse,
    "heads": heads,
    "levers": levers,
    "collapse": collapse,
    "collapse_doubles": collapse_doubles,
    "bs32": bs32,
    "throughput": throughput,
}


def main() -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fetch", action="store_true", help="refresh the W&B cache")
    args = p.parse_args()
    if args.fetch:
        fetch()
    out = C.demo(PANELS, "figure_043_curve_panels")
    print(out["png"])
    return out


if __name__ == "__main__":
    main()
