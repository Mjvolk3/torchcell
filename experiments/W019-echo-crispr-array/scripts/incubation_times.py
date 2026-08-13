# experiments/W019-echo-crispr-array/scripts/incubation_times.py
# [[experiments.W019-echo-crispr-array.scripts.incubation_times]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/incubation_times
"""Recover the incubation time of every plate, to the minute, from committed artifacts.

Two timestamps are already in the data and neither was being used:

  t0  the Echo `Run Date/Time` in each plate's transfer report -- when that plate was
      actually dispensed, per-plate, to the second.
  t1  the EXIF capture time of that plate's photograph.

so elapsed = t1 - t0 is derivable per plate with no recall and no bench notebook. This
matters because the rounds are NOT at the same incubation, and a cross-round comparison of
colony size that assumes they are is comparing different points on a growth curve.

CAVEAT ON t0, and it is the whole reason this script prints two columns. The Echo run time is
when the plate was DISPENSED, not when it entered the incubator -- plates are sealed and
carried over after all three are dispensed. So `elapsed_from_echo` is an UPPER bound on
incubation and runs slightly long. Where an officially recorded incubation exists it should
be preferred, and the difference between the two tells you the dispense-to-incubator gap.

Run from repo root (reads committed data only; no GPU):
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/incubation_times.py
"""

from __future__ import annotations

import glob
import os
import os.path as osp
from datetime import datetime

import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS

EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
DATA_DIR = osp.join(EXP_DIR, "data")
RESULTS_DIR = osp.join(EXP_DIR, "results")

# (round, plate, transfer report glob, image glob). Only rounds whose images pair 1:1 with a
# transfer report are listed; run 2's extra survey shots have no matching report.
ROUNDS = [
    ("run2", "P1", "run2_2026-07-17/P1_2p5nL_transfer_report.csv",
     "run2_2026-07-17/P1_2p5nL_plate_*.jpeg"),
    ("run2", "P2", "run2_2026-07-17/P2_5nL_transfer_report.csv",
     "run2_2026-07-17/P2_5nL_plate_*.jpeg"),
    ("run3", "P1", "run3_2026-07-23/P1_transfer_report.csv",
     "run3_2026-07-23/P1_OD1-5nL_TCsingleKO.JPG"),
    ("run3", "P2", "run3_2026-07-23/P2_transfer_report.csv",
     "run3_2026-07-23/P2_OD1-5nL_TCsingleKO.JPG"),
    ("run3", "P3", "run3_2026-07-23/P3_transfer_report.csv",
     "run3_2026-07-23/P3_OD1-5nL_TCsingleKO.JPG"),
    ("run4", "P1", "run4_doubles_2026-08-06/P1_Transfer_Report.csv",
     "run4_doubles_2026-08-06/P1_sKO-dKO_OD1-5nL.JPG"),
    ("run4", "P2", "run4_doubles_2026-08-06/P2_Transfer_Report.csv",
     "run4_doubles_2026-08-06/P2_sKO-dKO_OD1-5nL.JPG"),
    ("run4", "P3", "run4_doubles_2026-08-06/P3_Transfer_Report.csv",
     "run4_doubles_2026-08-06/P3_sKO-dKO_OD1-5nL.JPG"),
]

# Officially recorded incubation, where one exists. Only run 4 has been given to us so far;
# the others are BLANK rather than guessed -- an inferred time must not masquerade as a
# recorded one.
OFFICIAL = {("run4", "P1"): "48:12", ("run4", "P2"): "48:12", ("run4", "P3"): "48:12"}


def echo_run_time(path: str) -> datetime:
    """The Echo `Run Date/Time` from a transfer report header."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            k, _, v = line.partition(",")
            if k.strip() == "Run Date/Time":
                return datetime.strptime(v.strip().strip('"'), "%Y-%m-%d %H:%M:%S")
    raise ValueError(f"no 'Run Date/Time' in {path}")


def exif_capture_time(path: str) -> datetime:
    """EXIF capture time of a plate photograph."""
    e = Image.open(path).getexif()
    tags = {TAGS.get(k, k): v for k, v in e.items()}
    dt = tags.get("DateTime") or tags.get("DateTimeOriginal")
    if dt is None:
        ifd = e.get_ifd(0x8769)
        dt = {TAGS.get(k, k): v for k, v in ifd.items()}.get("DateTimeOriginal")
    if dt is None:
        raise ValueError(f"no EXIF capture time in {path}")
    return datetime.strptime(str(dt), "%Y:%m:%d %H:%M:%S")


def hhmm(td) -> str:
    total = int(td.total_seconds())
    return f"{total // 3600}:{(total % 3600) // 60:02d}"


def main() -> None:
    rows = []
    for rnd, plate, rep_glob, img_glob in ROUNDS:
        rep = glob.glob(osp.join(DATA_DIR, rep_glob))
        img = glob.glob(osp.join(DATA_DIR, img_glob))
        if not rep or not img:
            print(f"  SKIP {rnd} {plate}: missing report or image")
            continue
        t0, t1 = echo_run_time(rep[0]), exif_capture_time(img[0])
        official = OFFICIAL.get((rnd, plate), "")
        rows.append(
            dict(
                round=rnd,
                plate=plate,
                echo_dispensed=t0.strftime("%Y-%m-%d %H:%M:%S"),
                image_captured=t1.strftime("%Y-%m-%d %H:%M:%S"),
                elapsed_from_echo=hhmm(t1 - t0),
                official_incubation=official,
            )
        )
    df = pd.DataFrame(rows)
    out = osp.join(RESULTS_DIR, "incubation_times.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")
    print(
        "\nelapsed_from_echo is dispense -> photograph, so it is an UPPER bound on incubation:\n"
        "plates are sealed and moved to the incubator after all plates in the round are\n"
        "dispensed. Where an official incubation is recorded, the gap between the two columns\n"
        "is that dispense-to-incubator delay."
    )


if __name__ == "__main__":
    main()
