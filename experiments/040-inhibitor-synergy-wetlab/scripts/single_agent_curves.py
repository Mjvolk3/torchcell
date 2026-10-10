# experiments/040-inhibitor-synergy-wetlab/scripts/single_agent_curves.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.single_agent_curves]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/single_agent_curves
"""Single-agent dose response of the six inhibitors on bAID in YPD (2021 Bioscreen).

SOURCES, all from ``results/wetlab_wells.csv`` (served raw-curve call, see
[[experiments.040-inhibitor-synergy-wetlab.scripts.wetlab_table]]):

- ex21: nine doses x three biological replicates per inhibitor, 72 h run.
- the isobole axes: the wells of ex26 (FF x AA), ex27 (FA x AA) and ex28 (HMF x AA) whose
  partner dose is zero, nine doses x two plates, 96 h runs. AA has an axis in all three
  grids, which gives a cross-run consistency check on one compound.
- the ex23 singles: one dose x three replicates, 85 h; a check point, not fitted.

MODEL. ``f(d) = top / (1 + (d / IC50)^h)`` fitted by least squares on the DOSED wells
only, with ``top`` FREE: ex21's uninhibited wells grew slower than its low-dose wells
(039), so ex21 is normalized by its fitted top and never by its control wells. IC50 and
IC30 are relative to the fitted top (30% inhibition = f/top = 0.7, so
``IC30 = IC50 (3/7)^(1/h)``). A well that did not grow is fitness 0 in the primary fit
(``zero``) and dropped in the secondary fit (``excluded``); both are reported.

UNCERTAINTY. 1000 bootstrap resamples of the replicate wells within each dose; 95%
percentile intervals. The primary-variant draws are written for ``mixture_rules.py``.

VANACLOIG. The IC30 doses of furfural and 5-HMF in Vanacloig 2022 (anaerobic SYNH3,
48 h; Table S1, served on the rebuilt dev store's perturbation concentration) beside the
ex21 IC30 (aerobic YPD, 72 h): a cross-medium potency comparison, not a replication.
Levulinic acid has no Vanacloig condition in the corrected store (dropped in build 002
as an unreported token), which this script checks.

Writes ``results/single_agent_fits.csv``, ``results/single_agent_bootstrap.csv``,
``results/single_agent_ex23_check.csv``, ``results/vanacloig_ic30.csv`` and a figure.
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from pydantic import BaseModel
from scipy.optimize import least_squares

from torchcell.datasets.scerevisiae.vanacloig2022 import EnvChemgenVanacloig2022Dataset
from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")
IMAGES = osp.join(os.environ["ASSET_IMAGES_DIR"], "040-inhibitor-synergy-wetlab")
VANACLOIG_ROOT = osp.join(
    os.environ["DATA_ROOT"], "data/torchcell/env_chemgen_vanacloig2022"
)

ABBRS = ["FF", "AA", "HMF", "FA", "LVA", "LA"]
NAMES = {
    "FF": "furfural",
    "AA": "acetic acid",
    "HMF": "5-(hydroxymethyl)furfural",
    "FA": "formic acid",
    "LVA": "levulinic acid",
    "LA": "lactic acid",
}
SHORT = {
    "FF": "furfural",
    "AA": "acetic acid",
    "HMF": "5-HMF",
    "FA": "formic acid",
    "LVA": "levulinic acid",
    "LA": "lactic acid",
}
#: Standard formula weights (g/mol), the same as wetlab_table.MOLAR_MASS_G_PER_MOL.
MOLAR_MASS = {
    "FF": 96.08,
    "AA": 60.05,
    "HMF": 126.11,
    "FA": 46.03,
    "LVA": 116.12,
    "LA": 90.08,
}
#: The isobole run whose axis carries each inhibitor (AA: all three).
ISOBOLE_AXES = {"ex26": ["FF", "AA"], "ex27": ["FA", "AA"], "ex28": ["HMF", "AA"]}
#: ex23's one dose per inhibitor, g/L (bioscreen.EX23_G_PER_L).
EX23_DOSE = {"FF": 1.5, "AA": 2.0, "HMF": 2.522, "FA": 1.0, "LVA": 6.0, "LA": 20.0}
#: The Vanacloig compounds shared with the Bioscreen panel, by served compound name.
VANACLOIG_SHARED = ["FF", "HMF", "LVA"]
VANACLOIG_STRIDE = 1000
N_BOOT = 1000
SEED = 0
VARIANTS = ("zero", "excluded")
SOURCE_COLOR = {
    "ex21": PLOT_PALETTE[0],
    "ex26": PLOT_PALETTE[1],
    "ex27": PLOT_PALETTE[2],
    "ex28": PLOT_PALETTE[4],
    "ex23": "black",
}

matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "svg.fonttype": "none",
    }
)


class HillFit(BaseModel):
    """``f(d) = top / (1 + (d / ic50)^h)``."""

    top: float
    ic50: float
    h: float
    cost: float

    def fraction(self, d: np.ndarray | float) -> np.ndarray | float:
        """The fitness fraction f / top at dose ``d``."""
        return 1.0 / (1.0 + (np.asarray(d) / self.ic50) ** self.h)

    def icx(self, x: float) -> float:
        """Dose of ``x`` fractional inhibition relative to top."""
        return self.ic50 * (x / (1.0 - x)) ** (1.0 / self.h)


def fit_hill(dose: np.ndarray, y: np.ndarray, warm: HillFit | None = None) -> HillFit:
    """Least squares in (top, log10 IC50, h), the best of a grid of starts.

    ``warm`` (a bootstrap's point estimate) adds its parameters as one more start and
    thins the grid to two starts, which keeps 1000 resamples fast.
    """
    lo, hi = np.log10(dose.min()) - 2.0, np.log10(dose.max()) + 2.0

    def resid(p: np.ndarray) -> np.ndarray:
        return p[0] / (1.0 + (dose / 10.0 ** p[1]) ** p[2]) - y

    top0 = max(float(y.max()), 0.1)
    if warm is None:
        starts = [
            [top0, ic, h]
            for ic in np.linspace(np.log10(dose.min()), np.log10(dose.max()), 4)
            for h in (1.0, 3.0)
        ]
    else:
        starts = [
            [
                float(np.clip(warm.top, 0.05, 3.0)),
                float(np.clip(np.log10(warm.ic50), lo, hi)),
                float(np.clip(warm.h, 0.2, 20.0)),
            ],
            [top0, float(np.log10(np.median(dose))), 2.0],
        ]
    best = None
    for x0 in starts:
        r = least_squares(resid, x0=x0, bounds=([0.05, lo, 0.2], [3.0, hi, 20.0]))
        if best is None or r.cost < best.cost:
            best = r
    assert best is not None
    return HillFit(
        top=float(best.x[0]),
        ic50=float(10.0 ** best.x[1]),
        h=float(best.x[2]),
        cost=float(best.cost),
    )


def source_wells(wells: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """(source run, inhibitor) -> its dosed single-inhibitor wells, with ``dose``."""
    singles = wells[wells["n_compounds"] == 1]
    out = {}
    runs = {"ex21": ABBRS} | ISOBOLE_AXES
    for run, abbrs in runs.items():
        for a in abbrs:
            g = singles[(singles["run"] == run) & (singles[f"dose_g_per_l_{a}"] > 0)]
            out[(run, a)] = g.assign(dose=g[f"dose_g_per_l_{a}"]).copy()
    return out


def response(g: pd.DataFrame, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Doses and fitness, no-growth as 0 (``zero``) or dropped (``excluded``)."""
    if variant == "zero":
        return g["dose"].to_numpy(), g["fitness"].fillna(0.0).to_numpy()
    kept = g[g["grew"]]
    return kept["dose"].to_numpy(), kept["fitness"].to_numpy()


def bootstrap(
    g: pd.DataFrame, variant: str, rng: np.random.Generator, point: HillFit
) -> list[HillFit]:
    """Refit on replicate wells resampled with replacement within each dose."""
    groups = [d for _, d in g.groupby("dose")]
    fits = []
    for _ in range(N_BOOT):
        sample = pd.concat(
            [d.iloc[rng.integers(0, len(d), len(d))] for d in groups], ignore_index=True
        )
        dose, y = response(sample, variant)
        fits.append(fit_hill(dose, y, warm=point))
    return fits


def vanacloig_ic30() -> pd.DataFrame:
    """Read the shared compounds' Table S1 doses from the rebuilt Vanacloig store.

    The store writes one contiguous block per condition (thousands of records each), so
    a read every ``VANACLOIG_STRIDE`` records lands in every block; the first record of
    each shared compound found that way is the one read for its dose.
    """
    dataset = EnvChemgenVanacloig2022Dataset(root=VANACLOIG_ROOT)
    found: dict[str, tuple[int, float, str, str]] = {}
    blocks: list[str] = []
    for i in range(0, len(dataset), VANACLOIG_STRIDE):
        env = dataset.transform_item(dataset[i])["experiment"].environment
        molecules = [p for p in env.perturbations if hasattr(p, "compound")]
        if len(molecules) != 1:
            raise RuntimeError(f"record {i}: {len(molecules)} small molecules")
        p = molecules[0]
        if not blocks or blocks[-1] != p.compound.name:
            blocks.append(p.compound.name)
        if p.compound.name not in found:
            found[p.compound.name] = (
                i,
                float(p.concentration.value),
                p.concentration.unit.value,
                p.concentration.basis.value,
            )
    rows = []
    for a in VANACLOIG_SHARED:
        name = NAMES[a]
        if name in found:
            i, value, unit, basis = found[name]
            if unit != "mM":
                raise RuntimeError(f"{name}: unit {unit}, expected mM")
            rows.append(
                {
                    "inhibitor": a,
                    "compound": name,
                    "in_store": True,
                    "record_index": i,
                    "vanacloig_ic30_mM": value,
                    "vanacloig_ic30_g_per_l": value * MOLAR_MASS[a] / 1000.0,
                    "basis": basis,
                }
            )
        else:
            rows.append({"inhibitor": a, "compound": name, "in_store": False})
    print(
        f"Vanacloig store: {len(dataset)} records, {len(blocks)} condition blocks read "
        f"at stride {VANACLOIG_STRIDE}"
    )
    return pd.DataFrame(rows)


def box(ax: plt.Axes) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)


def save(fig: plt.Figure, name: str) -> list[str]:
    os.makedirs(IMAGES, exist_ok=True)
    stem = osp.join(IMAGES, f"{name}_{timestamp()}")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    return [stem + ".svg", stem + ".png"]


def plot_curves(
    data: dict[tuple[str, str], pd.DataFrame],
    fits: dict[tuple[str, str], HillFit],
    wells: pd.DataFrame,
) -> list[str]:
    """Six panels: each source's wells over its fitted top, with the fitted curve."""
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(95)), sharey=True
    )
    ex23 = wells[(wells["run"] == "ex23") & (wells["n_compounds"] == 1)]
    for ax, a in zip(axes.ravel(), ABBRS, strict=True):
        for (run, abbr), g in data.items():
            if abbr != a:
                continue
            fit = fits[(run, abbr)]
            grew = g[g["grew"]]
            ax.scatter(
                grew["dose"],
                grew["fitness"] / fit.top,
                s=6,
                color=SOURCE_COLOR[run],
                edgecolor="black",
                linewidth=0.3,
                label=f"{run} wells",
                zorder=3,
            )
            none = g[~g["grew"]]
            ax.scatter(
                none["dose"],
                np.zeros(len(none)),
                s=10,
                marker="x",
                color=SOURCE_COLOR[run],
                linewidth=0.6,
                zorder=3,
            )
            grid = np.geomspace(g["dose"].min() / 2, g["dose"].max() * 1.5, 200)
            ax.plot(grid, fit.fraction(grid), color=SOURCE_COLOR[run], linewidth=0.8)
        e = ex23[ex23[f"dose_g_per_l_{a}"] > 0]
        ax.errorbar(
            [EX23_DOSE[a]],
            [e["fitness"].mean()],
            yerr=[e["fitness"].std(ddof=1)],
            fmt="*",
            markersize=5,
            color="black",
            elinewidth=0.5,
            capsize=1.5,
            label="ex23 single (mean, SD; its own WT = 1)",
            zorder=4,
        )
        ax.axvline(EX23_DOSE[a], color="#999999", linewidth=0.4, linestyle="--")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.45)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(which="minor", axis="y", length=0)
        ax.grid(axis="y", which="both", linewidth=0.3, color="#DDDDDD")
        ax.set_axisbelow(True)
        ax.set_title(f"{SHORT[a]} ({a})")
        ax.set_xlabel("dose (g/L)")
        box(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("fitness / fitted top of its run")
    handles, labels = [], []
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    handles.append(
        plt.Line2D([], [], marker="x", linestyle="none", color="black", markersize=3)
    )
    labels.append("no growth within the run (drawn at 0)")
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.subplots_adjust(
        left=0.07, right=0.99, top=0.95, bottom=0.17, hspace=0.6, wspace=0.08
    )
    return save(fig, "single_agent_curves")


def main() -> None:
    wells = pd.read_csv(osp.join(RESULTS, "wetlab_wells.csv"))
    data = source_wells(wells)
    rng = np.random.default_rng(SEED)
    rows, boot_rows = [], []
    primary: dict[tuple[str, str], HillFit] = {}
    for (run, a), g in data.items():
        for variant in VARIANTS:
            dose, y = response(g, variant)
            fit = fit_hill(dose, y)
            draws = bootstrap(g, variant, rng, fit)
            if variant == "zero":
                primary[(run, a)] = fit
                boot_rows += [
                    {"source": run, "inhibitor": a, "draw": k} | d.model_dump()
                    for k, d in enumerate(draws)
                ]
            stats = {
                "top": [d.top for d in draws],
                "ic50": [d.ic50 for d in draws],
                "h": [d.h for d in draws],
                "ic30": [d.icx(0.3) for d in draws],
            }
            row = {
                "source": run,
                "inhibitor": a,
                "compound": NAMES[a],
                "variant": variant,
                "n_wells": len(g),
                "n_wells_fitted": len(y),
                "n_no_growth": int((~g["grew"]).sum()),
                "max_dose_g_per_l": float(g["dose"].max()),
                "top": fit.top,
                "h": fit.h,
                "ic50_g_per_l": fit.ic50,
                "ic30_g_per_l": fit.icx(0.3),
            }
            for key, value in stats.items():
                lo, hi = np.percentile(value, [2.5, 97.5])
                col = key if key in ("top", "h") else f"{key}_g_per_l"
                row[f"{col}_lo"], row[f"{col}_hi"] = float(lo), float(hi)
            for col in ("ic50", "ic30"):
                for suffix in ("", "_lo", "_hi"):
                    row[f"{col}_mM{suffix}"] = (
                        row[f"{col}_g_per_l{suffix}"] / MOLAR_MASS[a] * 1000.0
                    )
            rows.append(row)
    fits = pd.DataFrame(rows)
    fits.to_csv(osp.join(RESULTS, "single_agent_fits.csv"), index=False)
    pd.DataFrame(boot_rows).to_csv(
        osp.join(RESULTS, "single_agent_bootstrap.csv"), index=False
    )
    show = [
        "source",
        "inhibitor",
        "variant",
        "n_wells_fitted",
        "n_no_growth",
        "top",
        "h",
        "ic50_g_per_l",
        "ic50_g_per_l_lo",
        "ic50_g_per_l_hi",
        "ic30_g_per_l",
        "ic30_mM",
        "ic30_mM_lo",
        "ic30_mM_hi",
    ]
    print(fits[show].round(3).to_string())
    print(
        "acetic acid across runs (zero variant):\n"
        + fits[(fits["inhibitor"] == "AA") & (fits["variant"] == "zero")][show]
        .round(3)
        .to_string()
    )
    ex21_wt = wells[(wells["run"] == "ex21") & (wells["n_compounds"] == 0)]
    print(
        f"ex21 uninhibited wells: mean fitness {ex21_wt['fitness'].mean():.3f} "
        f"(n = {int(ex21_wt['grew'].sum())} grown of {len(ex21_wt)}); fitted ex21 tops "
        + ", ".join(f"{a} {primary[('ex21', a)].top:.3f}" for a in ABBRS)
    )

    # ex21 fit at the ex23 dose against the ex23 single, under both ex23 growth calls
    boot = pd.DataFrame(boot_rows)
    check_rows = []
    for a in ABBRS:
        e = wells[
            (wells["run"] == "ex23")
            & (wells["n_compounds"] == 1)
            & (wells[f"dose_g_per_l_{a}"] > 0)
        ]
        b21 = boot[(boot["source"] == "ex21") & (boot["inhibitor"] == a)]
        pred_draws = 1.0 / (1.0 + (EX23_DOSE[a] / b21["ic50"]) ** b21["h"])
        pred = float(primary[("ex21", a)].fraction(EX23_DOSE[a]))
        for call, col in (("served", "fitness"), ("software", "fitness_software")):
            obs = e[col].fillna(0.0).to_numpy()
            obs_draws = np.array(
                [rng.choice(obs, len(obs)).mean() for _ in range(len(pred_draws))]
            )
            diff = obs_draws - pred_draws.to_numpy()
            check_rows.append(
                {
                    "inhibitor": a,
                    "compound": NAMES[a],
                    "call": call,
                    "ex23_dose_g_per_l": EX23_DOSE[a],
                    "ex21_predicted_fraction": pred,
                    "ex23_observed": float(obs.mean()),
                    "n_ex23_wells": len(obs),
                    "observed_minus_predicted": float(obs.mean()) - pred,
                    "diff_lo": float(np.percentile(diff, 2.5)),
                    "diff_hi": float(np.percentile(diff, 97.5)),
                }
            )
    check = pd.DataFrame(check_rows)
    check.to_csv(osp.join(RESULTS, "single_agent_ex23_check.csv"), index=False)
    print(
        "ex21 Hill fit at the ex23 dose vs the ex23 single:\n"
        + check.round(3).to_string()
    )

    vana = vanacloig_ic30()
    ex21 = fits[(fits["source"] == "ex21") & (fits["variant"] == "zero")].set_index(
        "inhibitor"
    )
    for col in ("ic30_mM", "ic30_mM_lo", "ic30_mM_hi"):
        vana[f"bioscreen_ex21_{col}"] = vana["inhibitor"].map(ex21[col])
    vana["ratio_bioscreen_over_vanacloig"] = (
        vana["bioscreen_ex21_ic30_mM"] / vana["vanacloig_ic30_mM"]
    )
    vana["comparison"] = (
        "cross-medium potency: Vanacloig anaerobic SYNH3 48 h (haploid deletion pool "
        "IC30) vs Bioscreen aerobic YPD 72 h (bAID, ex21 Hill fit)"
    )
    vana.to_csv(osp.join(RESULTS, "vanacloig_ic30.csv"), index=False)
    print(vana.drop(columns="comparison").round(3).to_string())

    print("\n".join(plot_curves(data, primary, wells)))


if __name__ == "__main__":
    main()
