# experiments/027-betaxanthin-metabolic/scripts/analyze_wave5.py
# [[experiments.027-betaxanthin-metabolic.scripts.analyze_wave5]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/analyze_wave5.py

r"""Read the wave-5 cells out of their slurm logs and result files into one table.

Every number the dev note reports for wave 5 comes from here, so a cell that was cancelled
mid-run is summarized from its LOG (which has one row per completed epoch) rather than from
a result file it never wrote. A partial run is labeled partial and never carries a test
score, because the test pass happens once, after the epoch selection, at the end.

Per cell it reports:

* the **escape epoch**, the first epoch whose validation prediction spread exceeds
  ``ESCAPE_SD_RATIO`` of the target spread. Below that the head is a constant plus jitter
  and any Pearson computed on it is jitter too;
* validation Pearson at the **last completed epoch**, and its **maximum with the epoch
  named**. The maximum is an upward-biased order statistic over the epochs run, not a score,
  and it is labeled as one in the output key;
* the **pinned test Spearman** on the 629 Merzbacher genes, present only for a cell that
  finished;
* the constraint scale and the constraint loss **at the escape epoch**, which is the
  quantity the warmup experiment turns on;
* the capacity and budget diagnostics from the layer's own forward pass.

Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/analyze_wave5.py

Writes ``results/wave5_summary.json``.
"""

import glob
import json
import os
import os.path as osp
import re
from typing import Any, cast

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()

EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
SLURM_OUT = osp.join(EXP_DIR, "slurm", "output")
RESULTS = osp.join(EXP_DIR, "results")

#: An escape, not a collapse threshold. The trainer's collapse detector uses 0.01; a spread
#: at 0.01 of the target's is still a near-constant head, so the epoch a run is said to have
#: LEFT the plateau is set five times higher.
ESCAPE_SD_RATIO = 0.05

#: An escape must be SUSTAINED, and this is not a refinement, it is the rule that stops the
#: mistake made by eye on 2026-09-10. Cell G1-kj read 0.0137 then 0.0170 at epochs 43 and 44
#: and was called the start of an escape; epochs 45 to 48 read 0.0025, 0.0048, 0.0011, 0.0017.
#: The pooled baseline's real escape (job 1634) rose over six consecutive epochs and never
#: returned to the floor. A run of consecutive epochs above the threshold is what separates
#: those two shapes, and a single spike cannot satisfy it.
ESCAPE_CONSECUTIVE_EPOCHS = 3

#: The epoch line `train_bx.py` prints, from wave 5 onward.
EPOCH_RE = re.compile(
    r"\[(?P<arm>\S+) s(?P<seed>\d+)\] ep\s+(?P<epoch>\d+) loss (?P<loss>\S+) "
    r"data (?P<data>\S+) cstr (?P<cstr>\S+) scale (?P<scale>\S+) "
    r"val_loss (?P<val_loss>\S+) val_r (?P<val_r>\S+) sd_ratio (?P<sd>\S+)"
)


class EpochRow(BaseModel):
    """One epoch as the log printed it."""

    model_config = ConfigDict(extra="forbid")

    epoch: int
    train_loss: float
    train_data_loss: float
    train_constraint_loss: float
    constraint_scale: float
    val_loss: float
    val_pearson: float
    val_pred_sd_ratio: float


class CellSummary(BaseModel):
    """One wave-5 cell, from whatever evidence exists for it."""

    model_config = ConfigDict(extra="forbid")

    cell: str
    job_id: str
    log_path: str
    arm: str | None = None
    epochs_completed: int = 0
    finished: bool = False
    escape_epoch: int | None = None
    first_epoch_above_threshold: int | None = None
    max_sd_ratio: float | None = None
    max_sd_ratio_epoch: int | None = None
    constraint_scale_at_escape: float | None = None
    constraint_loss_at_escape: float | None = None
    data_loss_at_escape: float | None = None
    val_pearson_last: float | None = None
    val_pred_sd_ratio_last: float | None = None
    val_pearson_max: float | None = None
    val_pearson_max_epoch: int | None = None
    select_epoch: int | None = None
    test_spearman_pinned: float | None = None
    test_pearson_pinned: float | None = None
    n_test_pinned: int | None = None
    kcat_tier_counts: dict[str, int] | None = None
    km_tier_counts: dict[str, int] | None = None
    thermo_source: str | None = None
    thermo_route_counts: dict[str, int] | None = None
    capacity_active_frac_last: float | None = None
    budget_ratio_last: float | None = None
    result_path: str | None = None


def parse_log(path: str) -> list[EpochRow]:
    rows: list[EpochRow] = []
    with open(path, errors="replace") as handle:
        for line in handle:
            match = EPOCH_RE.search(line)
            if match is None:
                continue
            rows.append(
                EpochRow(
                    epoch=int(match["epoch"]),
                    train_loss=float(match["loss"]),
                    train_data_loss=float(match["data"]),
                    train_constraint_loss=float(match["cstr"]),
                    constraint_scale=float(match["scale"]),
                    val_loss=float(match["val_loss"]),
                    val_pearson=float(match["val_r"]),
                    val_pred_sd_ratio=float(match["sd"]),
                )
            )
    return rows


def summarize(cell: str, job_id: str, log_path: str, result_path: str | None) -> CellSummary:
    rows = parse_log(log_path)
    out = CellSummary(cell=cell, job_id=job_id, log_path=log_path, result_path=result_path)
    out.epochs_completed = len(rows)
    if rows:
        last = rows[-1]
        out.val_pearson_last = last.val_pearson
        out.val_pred_sd_ratio_last = last.val_pred_sd_ratio
        best = max(rows, key=lambda r: r.val_pearson)
        out.val_pearson_max = best.val_pearson
        out.val_pearson_max_epoch = best.epoch
        widest = max(rows, key=lambda r: r.val_pred_sd_ratio)
        out.max_sd_ratio = widest.val_pred_sd_ratio
        out.max_sd_ratio_epoch = widest.epoch
        for row in rows:
            if row.val_pred_sd_ratio > ESCAPE_SD_RATIO:
                out.first_epoch_above_threshold = row.epoch
                break
        # The SUSTAINED rule: the first epoch that begins a run of
        # ESCAPE_CONSECUTIVE_EPOCHS consecutive epochs above the threshold. A run truncated
        # by the wall clock cannot satisfy it, and that is correct: an escape that has not
        # yet been sustained has not yet been observed.
        need = ESCAPE_CONSECUTIVE_EPOCHS
        for i in range(len(rows) - need + 1):
            window = rows[i : i + need]
            if all(r.val_pred_sd_ratio > ESCAPE_SD_RATIO for r in window):
                first = window[0]
                out.escape_epoch = first.epoch
                out.constraint_scale_at_escape = first.constraint_scale
                out.constraint_loss_at_escape = first.train_constraint_loss
                out.data_loss_at_escape = first.train_data_loss
                break
    if result_path is not None and osp.exists(result_path):
        with open(result_path) as handle:
            payload = json.load(handle)
        out.finished = True
        out.arm = payload["cell"]["arm"]
        out.select_epoch = payload["select_epoch"]
        pinned = payload["test"]["pinned"]
        out.test_spearman_pinned = pinned["spearman"]
        out.test_pearson_pinned = pinned["pearson"]
        out.n_test_pinned = pinned["n"]
        meta: dict[str, Any] = payload.get("flux_meta") or {}
        coverage: dict[str, Any] = meta.get("coverage") or {}
        out.kcat_tier_counts = coverage.get("kcat_tier_counts")
        out.km_tier_counts = coverage.get("km_tier_counts")
        out.thermo_source = coverage.get("thermo_source")
        out.thermo_route_counts = coverage.get("thermo_route_counts")
        history = payload.get("history") or []
        if history:
            out.capacity_active_frac_last = history[-1].get("feas_capacity_active_frac")
            out.budget_ratio_last = history[-1].get("feas_budget_ratio")
    return out


#: cell -> (slurm job-name prefix, result file the slurm script writes).
CELLS: dict[str, tuple[str, str]] = {
    "G0 flux_off warm20 scale0.1": (
        "027-w5-flux_off-warm0.1",
        "wave5_g0_flux_off_s2700.json",
    ),
    "G1 flux_capacity warm20 scale0.1": (
        "027-w5-flux_capacity-warm0.1",
        "wave5_g1_flux_capacity_w20_s0p1_s2700.json",
    ),
    "G2 flux_abundance warm20 scale0.1": (
        "027-w5-flux_abundance-warm0.1",
        "wave5_g2_flux_abundance_w20_s0p1_s2700.json",
    ),
    "G3 flux_capacity warm20 scale1.0": (
        "027-w5-flux_capacity-warm1.0",
        "wave5_g3_flux_capacity_w20_s1p0_s2700.json",
    ),
    "G4 flux_capacity warm20 scale0.01": (
        "027-w5-flux_capacity-warm0.01",
        "wave5_g4_flux_capacity_w20_s0p01_s2700.json",
    ),
    "G6 flux_capacity headfix": (
        "027-w5-flux_capacity-headfix",
        "wave5_g6_flux_capacity_headfix_s2700.json",
    ),
    # The kcal-to-kJ restarts. Their pre-fix twins above read a union table that added
    # kcal-scale shipped energies to kJ-scale eQuilibrator ones, so those runs are kept
    # under their own keys rather than overwritten: both sides stay on the record.
    "G1-kj flux_capacity warm20 scale0.1": (
        "027-w5-flux_capacity-warm0.1-kj",
        "wave5_g1_flux_capacity_w20_s0p1_s2700_kj.json",
    ),
    "G2-kj flux_abundance warm20 scale0.1": (
        "027-w5-flux_abundance-warm0.1-kj",
        "wave5_g2_flux_abundance_w20_s0p1_s2700_kj.json",
    ),
    "G7 flux_capacity headfix nocstr": (
        "027-w5-flux_capacity-nocstr",
        "wave5_g7_flux_capacity_nocstr_s2700.json",
    ),
    "G8 pooled control today": ("027-w5-pooled-control", "wave5_g8_pooled_s2700.json"),
    # Seed replicates. The seed-2700 pair differ by +0.0030 in pinned test Spearman, far
    # inside the 0.04 between-seed spread the 019 study measured, so one seed cannot say
    # whether the module helps. These are what put an interval on that gap.
    "G7b flux_capacity headfix nocstr s2701": (
        "027-w5-flux_capacity-nocstr-s2701",
        "wave5_g7_flux_capacity_nocstr_s2701.json",
    ),
    "G8b pooled control s2701": (
        "027-w5-pooled-control-s2701",
        "wave5_g8_pooled_s2701.json",
    ),
    "G10 flux_capacity headfix delay40": (
        "027-w5-flux_capacity-headfix-delay40",
        "wave5_g10_flux_capacity_headfix_delay40_s2700.json",
    ),
    "G9 flux_off headfix": (
        "027-w5-flux_off-headfix",
        "wave5_g9_flux_off_headfix_s2700.json",
    ),
}


def main() -> None:
    summaries: list[CellSummary] = []
    for cell, (prefix, result_name) in CELLS.items():
        logs = sorted(glob.glob(osp.join(SLURM_OUT, f"{prefix}_*.out")))
        if not logs:
            continue
        log_path = logs[-1]
        job_id = osp.basename(log_path).rsplit("_", 1)[-1].removesuffix(".out")
        summaries.append(
            summarize(cell, job_id, log_path, osp.join(RESULTS, result_name))
        )

    payload = {
        "escape_sd_ratio": ESCAPE_SD_RATIO,
        "escape_consecutive_epochs": ESCAPE_CONSECUTIVE_EPOCHS,
        "escape_rule": (
            "escape_epoch is the first epoch beginning a run of "
            f"{ESCAPE_CONSECUTIVE_EPOCHS} consecutive epochs with val_pred_sd_ratio > "
            f"{ESCAPE_SD_RATIO}. first_epoch_above_threshold is reported separately so a "
            "single spike is visible without being counted as an escape."
        ),
        "scoring_note": (
            "val_pearson_max is a MAXIMUM over the epochs run, an upward-biased order "
            "statistic whose bias grows with the budget. The reported endpoint is "
            "test_spearman_pinned at the validation-selected epoch. The 019 replicate study "
            "puts the between-seed spread near 0.04 in Pearson, so a gap under that is "
            "within noise at n=1 seed."
        ),
        "cells": [s.model_dump() for s in summaries],
    }
    os.makedirs(RESULTS, exist_ok=True)
    path = osp.join(RESULTS, "wave5_summary.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)

    header = (
        f"{'cell':34s} {'job':>6s} {'eps':>4s} {'esc':>4s} {'cstr@esc':>9s} "
        f"{'val_r_last':>10s} {'val_r_max':>10s} {'@ep':>4s} {'test_rho':>9s}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.cell:34s} {s.job_id:>6s} {s.epochs_completed:>4d} "
            f"{('-' if s.escape_epoch is None else str(s.escape_epoch)):>4s} "
            f"{('-' if s.constraint_loss_at_escape is None else f'{s.constraint_loss_at_escape:.4f}'):>9s} "
            f"{('-' if s.val_pearson_last is None else f'{s.val_pearson_last:+.4f}'):>10s} "
            f"{('-' if s.val_pearson_max is None else f'{s.val_pearson_max:+.4f}'):>10s} "
            f"{('-' if s.val_pearson_max_epoch is None else str(s.val_pearson_max_epoch)):>4s} "
            f"{('partial' if not s.finished else f'{s.test_spearman_pinned:+.4f}'):>9s}"
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
