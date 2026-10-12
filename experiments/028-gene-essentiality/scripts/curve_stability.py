# experiments/028-gene-essentiality/scripts/curve_stability.py
# [[experiments.028-gene-essentiality.scripts.curve_stability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/curve_stability
"""How spiky a validation curve is, as a number, so "too spiky" stops being an impression.

The wave-9 essentiality curves look unstable, and the model-size ladder exists to test
whether capacity is the cause. That test needs an endpoint the test AUROC cannot provide:
with 31 essential genes among 195 the AUROC's bootstrap interval is about 0.17 wide, which
is wider than any difference these cells will show. The endpoint is the CURVE.

Four statistics per cell, all on the per-epoch validation score:

* **Median absolute step** between adjacent epochs. The headline. A curve that moves 0.07
  per epoch on a metric whose whole range is 0.3 is resampling noise, not converging.
* **Step-to-range ratio**, that median divided by the curve's own peak-to-trough range, so
  cells whose curves live on different scales stay comparable.
* **Lag-1 autocorrelation** of the first difference. A smooth curve that happens to move
  fast has positive autocorrelation; noise that reverses every epoch has it near -0.5. This
  separates "fast" from "jittery", which the median step alone cannot.
* **Selected-epoch spread across seeds.** If selection lands at epoch 60, 60 and 28 on
  three seeds of one configuration, the selector is picking a point on a noisy curve and the
  reported score is an order statistic of that noise.

Reported per configuration (width, preprocessor depth, layers) and per seed, alongside the
parameter count, so the ladder reads as a curve against size.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/curve_stability.py

Writes ``results/curve_stability.json``.
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
OUT = osp.join(RESULTS, "curve_stability.json")
#: The essentiality cells: the wave-9 baseline at width 90 and the wave-11 size ladder.
PATTERNS = ("wave9_ess_pool_e80_s*.json", "wave11_ess_pool_*_e80_s*.json")
#: A 2-epoch smoke has no curve to describe.
MIN_EPOCHS = 10


class CellCurve(BaseModel):
    """One cell's validation curve, described rather than scored."""

    model_config = ConfigDict(extra="forbid")
    result_file: str
    hidden: int
    preprocessor_layers: int
    layers: int
    attention_heads: int
    seed: int
    n_parameters: int
    n_epochs: int
    select_epoch: int
    median_abs_step: float = Field(
        description="Median |change| in the validation score between adjacent epochs."
    )
    curve_range: float = Field(description="Peak minus trough of the validation curve.")
    step_to_range: float = Field(
        description="median_abs_step / curve_range; scale-free spikiness."
    )
    diff_autocorrelation: float = Field(
        description=(
            "Lag-1 autocorrelation of the first difference. Near -0.5 means the curve "
            "reverses direction every epoch, which is noise; positive means it is moving."
        )
    )
    final_minus_peak: float = Field(
        description="Last epoch's score minus the peak; how far past the peak it ran."
    )
    grad_norm_mean: float | None = None
    grad_clipped_frac: float | None = None


class Configuration(BaseModel):
    """One (width, preprocessor depth, layers) across its seeds."""

    model_config = ConfigDict(extra="forbid")
    label: str
    hidden: int
    preprocessor_layers: int
    layers: int
    n_parameters: int
    seeds: list[int]
    median_abs_step_by_seed: dict[str, float]
    mean_median_abs_step: float
    select_epoch_by_seed: dict[str, int]
    select_epoch_spread: int = Field(
        description="Max minus min selected epoch across seeds."
    )


class StabilityReport(BaseModel):
    """Every essentiality cell's curve, and the ladder summarized by configuration."""

    model_config = ConfigDict(extra="forbid")
    metric: str = "per-epoch validation AUROC (logged as val_pearson on a binary head)"
    cells: list[CellCurve]
    configurations: list[Configuration]
    note: str


def describe(path: str) -> CellCurve | None:
    with open(path) as fh:
        d = json.load(fh)
    history = d.get("history") or []
    if len(history) < MIN_EPOCHS:
        return None
    cell = d["cell"]
    curve = np.array([float(e["val_pearson"]) for e in history], dtype=float)
    finite = curve[np.isfinite(curve)]
    if finite.size < MIN_EPOCHS:
        return None
    steps = np.abs(np.diff(finite))
    diffs = np.diff(finite)
    rng = float(finite.max() - finite.min())
    # Lag-1 autocorrelation of the first difference; a constant difference has none.
    if diffs.size > 2 and float(diffs.std()) > 0:
        autocorr = float(np.corrcoef(diffs[:-1], diffs[1:])[0, 1])
    else:
        autocorr = float("nan")
    selected = int(d["select_epoch"])
    row = next((e for e in history if e["epoch"] == selected), history[-1])
    return CellCurve(
        result_file=osp.basename(path),
        hidden=int(cell["hidden"]),
        preprocessor_layers=int(cell.get("preprocessor_layers", 2)),
        layers=int(cell["layers"]),
        attention_heads=int(cell["attention_heads"]),
        seed=int(cell["seed"]),
        n_parameters=int(d["n_parameters"]["total"]),
        n_epochs=len(history),
        select_epoch=selected,
        median_abs_step=float(np.median(steps)),
        curve_range=rng,
        step_to_range=float(np.median(steps) / rng) if rng > 0 else float("nan"),
        diff_autocorrelation=autocorr,
        final_minus_peak=float(finite[-1] - finite.max()),
        grad_norm_mean=row.get("grad_norm_mean"),
        grad_clipped_frac=row.get("grad_clipped_frac"),
    )


def main() -> None:
    paths: list[str] = []
    for pattern in PATTERNS:
        paths.extend(sorted(glob.glob(osp.join(RESULTS, pattern))))
    cells = [c for c in (describe(p) for p in paths) if c is not None]
    assert cells, f"no essentiality result files with at least {MIN_EPOCHS} epochs."

    configurations: list[Configuration] = []
    keys = sorted({(c.hidden, c.preprocessor_layers, c.layers) for c in cells})
    for hidden, preproc, layers in keys:
        group = [
            c
            for c in cells
            if (c.hidden, c.preprocessor_layers, c.layers) == (hidden, preproc, layers)
        ]
        steps = {str(c.seed): c.median_abs_step for c in group}
        epochs = {str(c.seed): c.select_epoch for c in group}
        configurations.append(
            Configuration(
                label=f"h{hidden}_p{preproc}_l{layers}",
                hidden=hidden,
                preprocessor_layers=preproc,
                layers=layers,
                n_parameters=group[0].n_parameters,
                seeds=sorted(c.seed for c in group),
                median_abs_step_by_seed=steps,
                mean_median_abs_step=float(np.mean(list(steps.values()))),
                select_epoch_by_seed=epochs,
                select_epoch_spread=max(epochs.values()) - min(epochs.values()),
            )
        )

    report = StabilityReport(
        cells=cells,
        configurations=configurations,
        note=(
            "The endpoint of the size ladder is the curve, not the test AUROC: at 31 "
            "essential genes among 195 the AUROC bootstrap interval is about 0.17 wide and "
            "cannot separate these cells. A median absolute step that falls with model size "
            "says capacity drove the instability; one that does not says the cause is "
            "elsewhere, and the 16 percent positive rate and the batch size of 128 against "
            "about 851 training rows are the next candidates."
        ),
    )
    with open(OUT, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)

    print(
        f"{'configuration':16s} {'params':>9s} {'med|step|':>10s} {'step/range':>11s} "
        f"{'autocorr':>9s} {'sel epochs':>18s}"
    )
    for c in sorted(configurations, key=lambda x: -x.n_parameters):
        group = [
            x
            for x in cells
            if (x.hidden, x.preprocessor_layers, x.layers)
            == (c.hidden, c.preprocessor_layers, c.layers)
        ]
        ratio = float(np.mean([x.step_to_range for x in group]))
        auto = float(np.nanmean([x.diff_autocorrelation for x in group]))
        epochs = ",".join(str(v) for v in c.select_epoch_by_seed.values())
        print(
            f"{c.label:16s} {c.n_parameters:9,d} {c.mean_median_abs_step:10.4f} "
            f"{ratio:11.3f} {auto:9.3f} {epochs:>18s}"
        )
    print(f"\n{OUT}")


if __name__ == "__main__":
    main()
