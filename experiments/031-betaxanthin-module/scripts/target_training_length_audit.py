# experiments/031-betaxanthin-module/scripts/target_training_length_audit.py
# [[experiments.031-betaxanthin-module.scripts.target_training_length_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_training_length_audit

r"""Are the transfer cells trained long enough? The validation curves, per arm and seed.

Every transfer cell (rounds 117 onward) trains 200 epochs and keeps its full per-epoch
history in the result file: ``val_loss`` (the CRPS on the validation pool) and
``val_pearson``. A cell is trained long enough if its validation loss has reached a
minimum inside the budget and is not still falling at epoch 199, and if the moving
average of validation Pearson the selection rule reads has peaked before the end. This
script reads both curves for every landed transfer arm and reports, per cell:

* the epoch of the minimum validation loss (raw and five-epoch centered moving average),
  the loss there, the loss at the last epoch, and whether the minimum lies in the final
  twenty epochs (still falling);
* the slope of the moving-average loss over the final twenty epochs, per epoch;
* the epoch of the maximum moving-average validation Pearson, the selected epoch, and
  whether the maximum lies in the final twenty epochs (still rising).

Cells still running are read from their per-cell slurm logs and marked partial, with the
epoch reached; they have no result and get no verdict. The figure overlays the seed
curves per arm. Writes ``training_length_audit.json`` and a figure per study (031 betaxanthin, 032 amino acids), the figure to
``ASSET_IMAGES_DIR/<study>/``. Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_training_length_audit.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

from torchcell.timestamp import timestamp
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
#: Each target is its own study (2026-09-23): betaxanthin is 031, the amino acids 032.
STUDY_BY_HEAD = {"betaxanthin": "031-betaxanthin-module", "mulleder19": "032-amino-acid-module"}
RESULTS_BY_HEAD = {
    h: osp.join(os.environ["EXPERIMENT_ROOT"], s, "results") for h, s in STUDY_BY_HEAD.items()
}
#: Running cells log under their study; cells launched before the split (jobs up to
#: 2775) log under 028, so both trees are searched.
LOG_DIRS = [
    osp.join(os.environ["EXPERIMENT_ROOT"], s, "slurm", "output")
    for s in (*STUDY_BY_HEAD.values(), "028-gene-essentiality")
]
IMAGES_BY_HEAD = {
    h: osp.join(os.environ["ASSET_IMAGES_DIR"], s) for h, s in STUDY_BY_HEAD.items()
}
SEEDS = tuple(range(2700, 2712))
TAIL = 20
MA = 5
#: (stem, head, short label); every transfer arm, landed or running.
ARMS = [
    ("w134_bxctx_h12l1p1_e200", "betaxanthin", "control"),
    ("w134_bxstore_h12l1p1_e200", "betaxanthin", "store"),
    ("w136_bxratio_h12l1p1_e200", "betaxanthin", "two ratios"),
    ("w136_bxstorewd_h12l1p1_e200", "betaxanthin", "store, decayed"),
    ("w137_bxdel_h12l1p1_e200", "betaxanthin", "deletion-only"),
    ("w137_bxgpr_h12l1p1_e200", "betaxanthin", "rule counts"),
    ("w140_bxdelstore_h12l1p1_e200", "betaxanthin", "deletion-only + store"),
    ("w142_bxgrowth_h12l1p1_e200", "betaxanthin", "growth block"),
    ("w142_bxprod_h12l1p1_e200", "betaxanthin", "production block"),
    ("w139_bxwide_h90l2p1_e200", "betaxanthin", "width 90"),
    ("w140_bxwidestore_h90l2p1_e200", "betaxanthin", "width 90 + store"),
    ("w145_bxwidedelstore_h90l2p1_e200", "betaxanthin", "width 90 + del + store"),
    ("w135_aactx_h12l1p1_e200", "mulleder19", "control"),
    ("w135_aastore_h12l1p1_e200", "mulleder19", "store"),
    ("w138_aaprodstat_h12l1p1_e200", "mulleder19", "20 summaries"),
    ("w138_aaprodfull_h12l1p1_e200", "mulleder19", "vector + 42 summaries"),
    ("w141_aadel_h12l1p1_e200", "mulleder19", "deletion-only"),
    ("w143_aadelstore_h12l1p1_e200", "mulleder19", "deletion-only + store"),
    ("w139_aawide_h90l2p1_e200", "mulleder19", "width 90"),
    ("w147_bxdelstore40_h12l1p1_e200", "betaxanthin", "del + store, affine frozen at 40"),
    ("w148_bxdelstorer15_h12l1p1_e200", "betaxanthin", "del + store, affine from 15"),
    ("w149_bxwidedel_h90l2p1_e200", "betaxanthin", "width 90 + del"),
    ("w152_bxdelgpr_h12l1p1_e200", "betaxanthin", "del + rule counts"),
    ("w153_bxdelhead256_h12l1p1_e200", "betaxanthin", "del + solve head 256"),
    ("w154_bxdelmoma_h12l1p1_e200", "betaxanthin", "del + MOMA store"),
    ("w156_bxdelpfba_h12l1p1_e200", "betaxanthin", "del + HiGHS store"),
    ("w157_bxdelmdf_h12l1p1_e200", "betaxanthin", "del + driving-force store"),
    ("w159_bxdelsampled_h12l1p1_e200", "betaxanthin", "del + sampled-cone store"),
    ("w146_aadel_h12l1p1_e300", "mulleder19", "deletion-only, 300 epochs"),
    ("w150_aadelgpr_h12l1p1_e200", "mulleder19", "del + rule counts"),
    ("w151_aadelhead256_h12l1p1_e200", "mulleder19", "del + solve head 256"),
    ("w155_aadelmoma_h12l1p1_e200", "mulleder19", "del + MOMA store"),
    ("w158_aadelmdf_h12l1p1_e200", "mulleder19", "del + driving-force store"),
    ("w160_aadelsampled_h12l1p1_e200", "mulleder19", "del + sampled-cone store"),
]
LOG_LINE = re.compile(
    r"\] ep\s+(\d+) loss ([\d.]+) .*?val_loss ([\d.]+) val_r ([-+\d.]+)"
)


class CellAudit(BaseModel):
    """One cell's training-length read."""

    stem: str
    seed: int
    partial: bool
    epochs_seen: int
    min_val_loss_epoch: int
    min_val_loss: float
    min_val_loss_ma_epoch: int
    last_val_loss: float
    last_minus_min_val_loss: float
    loss_min_in_tail: bool
    tail_loss_slope_per_epoch: float
    max_val_pearson_ma_epoch: int
    max_val_pearson_ma: float
    select_epoch: int | None
    pearson_max_in_tail: bool
    pearson_ma_at_end: float


class ArmAudit(BaseModel):
    """One arm's cells and the counts that answer the question."""

    stem: str
    head: str
    label: str
    cells: list[CellAudit]
    n_complete: int
    n_loss_min_in_tail: int
    n_pearson_max_in_tail: int
    median_min_val_loss_ma_epoch: float | None
    median_max_val_pearson_ma_epoch: float | None


class Report(BaseModel):
    """The audit."""

    tail_epochs: int
    moving_average: int
    arms: list[ArmAudit]
    note: str


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average with shrinking windows at the ends, as the trainer's."""
    out = np.empty_like(x, dtype=float)
    h = k // 2
    for i in range(x.size):
        lo, hi = max(0, i - h), min(x.size, i + h + 1)
        out[i] = float(np.mean(x[lo:hi]))
    return out


def curves_from_result(path: str) -> tuple[np.ndarray, np.ndarray, int | None]:
    with open(path) as fh:
        r = json.load(fh)
    h = r["history"]
    loss = np.array([e["val_loss"] for e in h], dtype=float)
    pear = np.array([e["val_pearson"] for e in h], dtype=float)
    return loss, pear, int(r["select_epoch"])


def curves_from_log(stem: str, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    # The newest log by modification time: a cancelled pack leaves an older log for the
    # same cell beside the running one.
    paths = sorted(
        (
            p
            for d in LOG_DIRS
            for p in glob.glob(osp.join(d, f"*-pack_*_{stem}_s{seed}.out"))
        ),
        key=osp.getmtime,
    )
    if not paths:
        return None
    loss, pear = [], []
    with open(paths[-1]) as fh:
        for line in fh:
            m = LOG_LINE.search(line)
            if m:
                loss.append(float(m.group(3)))
                pear.append(float(m.group(4)))
    if not loss:
        return None
    return np.array(loss), np.array(pear)


def audit(
    stem: str, seed: int, loss: np.ndarray, pear: np.ndarray, select: int | None, partial: bool
) -> CellAudit:
    n = loss.size
    loss_ma = moving_average(loss, MA)
    pear_ma = moving_average(np.nan_to_num(pear, nan=0.0), MA)
    tail = slice(max(0, n - TAIL), n)
    slope = float(np.polyfit(np.arange(n)[tail], loss_ma[tail], 1)[0]) if n >= 3 else 0.0
    i_loss = int(np.argmin(loss))
    i_loss_ma = int(np.argmin(loss_ma))
    i_pear_ma = int(np.argmax(pear_ma))
    return CellAudit(
        stem=stem,
        seed=seed,
        partial=partial,
        epochs_seen=n,
        min_val_loss_epoch=i_loss,
        min_val_loss=float(loss[i_loss]),
        min_val_loss_ma_epoch=i_loss_ma,
        last_val_loss=float(loss[-1]),
        last_minus_min_val_loss=float(loss[-1] - loss[i_loss]),
        loss_min_in_tail=bool(i_loss_ma >= n - TAIL),
        tail_loss_slope_per_epoch=slope,
        max_val_pearson_ma_epoch=i_pear_ma,
        max_val_pearson_ma=float(pear_ma[i_pear_ma]),
        select_epoch=select,
        pearson_max_in_tail=bool(i_pear_ma >= n - TAIL),
        pearson_ma_at_end=float(pear_ma[-1]),
    )


def main() -> None:
    for d in IMAGES_BY_HEAD.values():
        os.makedirs(d, exist_ok=True)
    arms: list[ArmAudit] = []
    curves: dict[str, list[tuple[int, np.ndarray, np.ndarray, bool]]] = {}
    for stem, head, label in ARMS:
        cells: list[CellAudit] = []
        for seed in SEEDS:
            path = osp.join(RESULTS_BY_HEAD[head], f"{stem}_s{seed}.json")
            if osp.exists(path):
                loss, pear, select = curves_from_result(path)
                cells.append(audit(stem, seed, loss, pear, select, partial=False))
                curves.setdefault(stem, []).append((seed, loss, pear, False))
                continue
            got = curves_from_log(stem, seed)
            if got is None:
                continue
            loss, pear = got
            cells.append(audit(stem, seed, loss, pear, None, partial=True))
            curves.setdefault(stem, []).append((seed, loss, pear, True))
        if not cells:
            continue
        complete = [c for c in cells if not c.partial]
        arms.append(
            ArmAudit(
                stem=stem,
                head=head,
                label=label,
                cells=cells,
                n_complete=len(complete),
                n_loss_min_in_tail=sum(c.loss_min_in_tail for c in complete),
                n_pearson_max_in_tail=sum(c.pearson_max_in_tail for c in complete),
                median_min_val_loss_ma_epoch=(
                    float(np.median([c.min_val_loss_ma_epoch for c in complete]))
                    if complete
                    else None
                ),
                median_max_val_pearson_ma_epoch=(
                    float(np.median([c.max_val_pearson_ma_epoch for c in complete]))
                    if complete
                    else None
                ),
            )
        )
        for c in cells:
            tag = f"partial, epoch {c.epochs_seen}" if c.partial else "complete"
            print(
                f"{stem} s{c.seed} [{tag}]: val_loss min {c.min_val_loss:.4f} at ep "
                f"{c.min_val_loss_epoch} (MA ep {c.min_val_loss_ma_epoch}), last "
                f"{c.last_val_loss:.4f} ({c.last_minus_min_val_loss:+.4f}), tail slope "
                f"{c.tail_loss_slope_per_epoch:+.5f}/ep; val_r MA max {c.max_val_pearson_ma:.3f} "
                f"at ep {c.max_val_pearson_ma_epoch}, end {c.pearson_ma_at_end:.3f}, "
                f"selected {c.select_epoch}"
            )
        a = arms[-1]
        print(
            f"  -> {a.n_complete} complete: loss min in last {TAIL} on "
            f"{a.n_loss_min_in_tail}, val_r max in last {TAIL} on {a.n_pearson_max_in_tail}; "
            f"median loss-min epoch {a.median_min_val_loss_ma_epoch}, median val_r-max "
            f"epoch {a.median_max_val_pearson_ma_epoch}"
        )

    note = (
        "Per cell, from the result file's per-epoch history (or the slurm log for a "
        "running cell, marked partial): the epoch of minimum validation loss (raw and "
        f"{MA}-epoch centered moving average), the loss at the end, the slope of the "
        f"moving-average loss over the last {TAIL} epochs, and the epoch of maximum "
        "moving-average validation Pearson beside the selected epoch. A minimum or "
        f"maximum inside the last {TAIL} epochs means the curve had not settled."
    )
    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 6, "svg.fonttype": "none",
         "axes.linewidth": 0.5}
    )
    stamp = timestamp()
    # One report and one figure per study: validation loss and moving-average Pearson,
    # one color per arm, seeds as thin lines, partial cells dashed, legend outside.
    for head, results in RESULTS_BY_HEAD.items():
        head_arms = [a for a in arms if a.head == head]
        report = Report(tail_epochs=TAIL, moving_average=MA, arms=head_arms, note=note)
        out = osp.join(results, "training_length_audit.json")
        with open(out, "w") as fh:
            fh.write(report.model_dump_json(indent=2))
        print(out)
        fig, axes = plt.subplots(
            1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62)), squeeze=False
        )
        for j, a in enumerate(head_arms):
            color = PLOT_PALETTE[j % len(PLOT_PALETTE)]
            for k, (seed, loss, pear, partial) in enumerate(curves[a.stem]):
                style = "--" if partial else "-"
                axes[0][0].plot(
                    np.arange(loss.size), moving_average(loss, MA), style, color=color,
                    lw=0.6, label=a.label if k == 0 else None,
                )
                axes[0][1].plot(
                    np.arange(pear.size), moving_average(np.nan_to_num(pear), MA), style,
                    color=color, lw=0.6,
                )
        for ax, ylabel in zip(axes[0], ["validation loss (CRPS), 5-epoch MA",
                                        "validation Pearson, 5-epoch MA"]):
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.tick_params(labelsize=6, width=0.5)
            for s in ax.spines.values():
                s.set_linewidth(0.5)
                s.set_visible(True)
            ax.set_xlim(0, 200)
        axes[0][0].legend(
            loc="lower left", bbox_to_anchor=(0, 1.04), ncol=4, fontsize=6, frameon=False
        )
        fig.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.14, wspace=0.3)
        stem = f"training_length_audit_{stamp}"
        fig.savefig(osp.join(IMAGES_BY_HEAD[head], f"{stem}.png"), dpi=300)
        savefig_true_size_svg(fig, osp.join(IMAGES_BY_HEAD[head], f"{stem}.svg"))
        print(osp.join(IMAGES_BY_HEAD[head], f"{stem}.svg"))
        plt.close(fig)


if __name__ == "__main__":
    main()
