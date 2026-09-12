# experiments/027-betaxanthin-metabolic/scripts/mulleder_concentration_window.py
# [[experiments.027-betaxanthin-metabolic.wave7-amino-acid-concentration]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/mulleder_concentration_window.py

r"""Does the flux layer's concentration window cover the measured amino acids?

WHY THIS RUNS BEFORE THE CELLS
------------------------------
``FluxLayer.reaction_potential`` in ``ANCHORED`` mode learns a per-strain log concentration
squashed into ``[ln CONC_LOWER_M, ln CONC_UPPER_M]`` = ``[ln 1e-7, ln 1e-2]``. Wave 7
supervises exactly that quantity with Mulleder 2016's absolute intracellular concentrations,
which are reported in mM. A measured value above the window's ceiling is unreachable by the
head no matter what it learns, and the failure is silent: the sigmoid saturates and the
residual is charged to the model rather than to the parameterization.

This script reads the SAME processed LMDB ``train_bx.py`` opens
(``$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer``) and
reports, per amino acid, the min, median and max in molar, plus the fraction of measurements
outside the shipped window. It writes ``results/mulleder_concentration_window.json``.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/mulleder_concentration_window.py
"""

import json
import os
import os.path as osp

import lmdb
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.metabolism.constraints import CONC_LOWER_M, CONC_UPPER_M

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
LMDB_PATH = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer",
    "processed/lmdb",
)
DATASET_NAME = "AminoAcidMulleder2016Dataset"
#: Mulleder's ``intracellular_concentration_mM`` sheet, so a stored value of 1.0 is 1 mM.
MM_PER_M = 1.0e3


class AminoAcidWindow(BaseModel):
    """The measured range of one amino acid against the layer's concentration window."""

    model_config = ConfigDict(extra="forbid")

    amino_acid: str
    n: int
    min_m: float
    median_m: float
    max_m: float
    frac_above_upper: float
    frac_below_lower: float


class WindowCheck(BaseModel):
    """The whole check, written verbatim into the results file and the note."""

    model_config = ConfigDict(extra="forbid")

    lmdb_path: str
    dataset_name: str
    stored_unit: str
    conc_lower_m: float
    conc_upper_m: float
    n_records: int
    n_measurements: int
    overall_min_m: float
    overall_max_m: float
    frac_measurements_above_upper: float
    frac_measurements_below_lower: float
    amino_acids_exceeding_upper: list[str]
    recommended_conc_lower_m: float
    recommended_conc_upper_m: float
    per_amino_acid: list[AminoAcidWindow]


def read_measurements() -> dict[str, list[float]]:
    """``{amino acid -> concentrations in molar}`` over every Mulleder record in the build."""
    env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
    values: dict[str, list[float]] = {}
    n_records = 0
    with env.begin() as txn:
        for _, payload in txn.cursor():
            for rec in json.loads(payload.decode()):
                experiment = rec["experiment"]
                if experiment["dataset_name"] != DATASET_NAME:
                    continue
                n_records += 1
                for name, mm in experiment["phenotype"]["metabolite_level"].items():
                    values.setdefault(name, []).append(float(mm) / MM_PER_M)
    env.close()
    values["__n_records__"] = [float(n_records)]
    return values


def build_check() -> WindowCheck:
    raw = read_measurements()
    n_records = int(raw.pop("__n_records__")[0])
    per: list[AminoAcidWindow] = []
    every: list[float] = []
    for name in sorted(raw):
        arr = np.asarray(raw[name], dtype=float)
        every.extend(arr.tolist())
        per.append(
            AminoAcidWindow(
                amino_acid=name,
                n=int(arr.size),
                min_m=float(arr.min()),
                median_m=float(np.median(arr)),
                max_m=float(arr.max()),
                frac_above_upper=float((arr > CONC_UPPER_M).mean()),
                frac_below_lower=float((arr < CONC_LOWER_M).mean()),
            )
        )
    allv = np.asarray(every, dtype=float)
    exceeding = [p.amino_acid for p in per if p.max_m > CONC_UPPER_M]
    # A DECADE OF MARGIN ON EACH SIDE of the measured range, rounded to a power of ten, so
    # the supervised value never sits on the sigmoid's saturating tail: the largest measured
    # concentration must be reachable with room for the head to overshoot and come back.
    upper = 10.0 ** float(np.ceil(np.log10(allv.max())) + 1.0)
    lower = min(CONC_LOWER_M, 10.0 ** float(np.floor(np.log10(allv.min())) - 1.0))
    return WindowCheck(
        lmdb_path=LMDB_PATH,
        dataset_name=DATASET_NAME,
        stored_unit="intracellular_concentration_mM",
        conc_lower_m=CONC_LOWER_M,
        conc_upper_m=CONC_UPPER_M,
        n_records=n_records,
        n_measurements=int(allv.size),
        overall_min_m=float(allv.min()),
        overall_max_m=float(allv.max()),
        frac_measurements_above_upper=float((allv > CONC_UPPER_M).mean()),
        frac_measurements_below_lower=float((allv < CONC_LOWER_M).mean()),
        amino_acids_exceeding_upper=exceeding,
        recommended_conc_lower_m=lower,
        recommended_conc_upper_m=upper,
        per_amino_acid=per,
    )


def main() -> None:
    check = build_check()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = osp.join(RESULTS_DIR, "mulleder_concentration_window.json")
    with open(path, "w") as fh:
        json.dump(check.model_dump(), fh, indent=2)
    print(
        f"{check.n_records} Mulleder records, {check.n_measurements} measurements; "
        f"range {check.overall_min_m:.3e} to {check.overall_max_m:.3e} M against the "
        f"window [{check.conc_lower_m:.0e}, {check.conc_upper_m:.0e}] M"
    )
    print(
        f"above the ceiling: {check.frac_measurements_above_upper:.4f} of measurements, "
        f"{len(check.amino_acids_exceeding_upper)} of {len(check.per_amino_acid)} "
        f"amino acids ({', '.join(check.amino_acids_exceeding_upper)})"
    )
    print("| amino acid | n | min (M) | median (M) | max (M) | frac above 1e-2 |")
    print("|---|---:|---:|---:|---:|---:|")
    for p in check.per_amino_acid:
        print(
            f"| {p.amino_acid} | {p.n} | {p.min_m:.3e} | {p.median_m:.3e} | "
            f"{p.max_m:.3e} | {p.frac_above_upper:.4f} |"
        )
    print(
        f"recommended window: [{check.recommended_conc_lower_m:.0e}, "
        f"{check.recommended_conc_upper_m:.0e}] M"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
