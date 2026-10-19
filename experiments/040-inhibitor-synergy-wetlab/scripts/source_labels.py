# experiments/040-inhibitor-synergy-wetlab/scripts/source_labels.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.source_labels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/source_labels
r"""What a label looks like in each source the mixture trainer reads, and which inhibitors
of this study any public screen holds.

The data panel of the gate figure shows, beside each source's size, the shape of its
label: a public screen's value is one deletion strain's growth under one compound at one
dose, relative to that screen's own control, on that screen's own scale (a log2 ratio, a
MAD-scaled sensitivity score, a z-score); a well of this study is the wild-type host's
relative growth rate, 1 at the uninhibited rate and 0 at no growth. The values are read
exactly as the trainer reads them before it standardizes a source: ``_condition_frame``
of ``mixture_data.py`` (single-compound cells with a structure, the trainer's orientation
to sick-negative applied), so the histogram is of the trainer's input, not of a column of
the released file.

Three tables, under ``results/``:

``source_label_histograms.csv``  per source, forty bins between the 0.5th and 99.5th
                                 percentiles of the oriented value, with the count in each
``source_label_summary.csv``     per source: the label's name and unit as the loader
                                 states it, the orientation sentence, the cells and
                                 conditions read, the percentiles and the share below zero
``source_inhibitor_overlap.csv`` the six inhibitors of this study against the four
                                 screens: the cells and conditions a screen holds for the
                                 inhibitor, matched on InChIKey, so the figure can state
                                 which inhibitors any public profile exists for

The host rows come from ``results/wetlab_wells.csv`` (``wetlab_table.py``) by run, and the
public IC30 anchors are stated as the trainer builds them (``anchor_records``): every
anchored compound at fitness 0.70, the compound-free medium at 1.0.

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \\
        experiments/040-inhibitor-synergy-wetlab/scripts/source_labels.py
"""

from __future__ import annotations

import os.path as osp
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from mixture_data import (  # noqa: E402
    CELL_TABLE,
    IC30_FITNESS,
    SPECS,
    _condition_frame,
    wetlab_inhibitors,
)

load_dotenv()
EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")
BINS = 40

#: the label each loader serves, as its sourced value names it (the unit is the scale
#: the number is on; ``SourceSpec.orientation`` states the formula)
LABEL = {
    "vanacloig": ("log2 ratio to the inhibitor-free control", "log2"),
    "hoepfner_hop": ("adjusted MADL sensitivity score", "MAD units"),
    "hillenmeyer_het": ("fitness-defect log2 ratio, negated", "log2"),
    "wildenhain": ("growth-inhibition z-score", "IQR units"),
}
#: this study's runs, as the figure names them
RUN_SPLIT = {
    "ex21": "this study: titration",
    "ex23": "this study: combinations",
    "ex26": "this study: isobole grids",
    "ex27": "this study: isobole grids",
    "ex28": "this study: isobole grids",
}


def histogram(source: str, values: np.ndarray) -> pd.DataFrame:
    lo, hi = np.nanpercentile(values, [0.5, 99.5])
    counts, edges = np.histogram(values[np.isfinite(values)], bins=BINS, range=(lo, hi))
    return pd.DataFrame(
        {"source": source, "bin_lo": edges[:-1], "bin_hi": edges[1:], "count": counts}
    )


def summary(source: str, values: np.ndarray, **extra: object) -> dict[str, object]:
    finite = values[np.isfinite(values)]
    p = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    return {
        "source": source,
        "n_values": int(len(finite)),
        "p01": p[0],
        "p05": p[1],
        "p25": p[2],
        "p50": p[3],
        "p75": p[4],
        "p95": p[5],
        "p99": p[6],
        "share_below_zero": float((finite < 0).mean()),
        **extra,
    }


def main() -> None:
    hists, rows, overlap = [], [], []
    inhibitors = wetlab_inhibitors()
    key_to_name = {key: name for name, key in inhibitors.items()}
    for name, spec in SPECS.items():
        frame = _condition_frame(spec, CELL_TABLE)
        values = frame["response"].to_numpy(dtype=float)
        hists.append(histogram(name, values))
        label, unit = LABEL[name]
        rows.append(
            summary(
                name,
                values,
                label=label,
                unit=unit,
                orientation=spec.orientation,
                value_is="one deletion strain under one compound at one dose",
                n_conditions=int(frame["condition"].nunique()),
                n_compounds=int(frame["inchikeys"].nunique()),
                n_genes=int(frame["query_gene"].nunique()),
            )
        )
        held = frame[frame["inchikeys"].isin(key_to_name)]
        for key, sub in held.groupby("inchikeys"):
            overlap.append(
                {
                    "inhibitor": key_to_name[str(key)],
                    "inchikey": key,
                    "source": name,
                    "n_cells": int(len(sub)),
                    "n_conditions": int(sub["condition"].nunique()),
                    "doses": "|".join(sorted(sub["conc_values"].astype(str).unique())),
                }
            )
        print(
            f"{name}: {len(frame):,} cells read, {rows[-1]['n_conditions']} conditions"
        )
    wells = pd.read_csv(osp.join(RESULTS, "wetlab_wells.csv"))
    wells = wells[wells["run"].isin(RUN_SPLIT)].copy()
    wells["source"] = wells["run"].map(RUN_SPLIT)
    # a well that did not grow has no fitness in the table; the trainer reads it as 0
    wells["fitness_read"] = wells["fitness"].fillna(0.0)
    for source, sub in wells.groupby("source"):
        values = sub["fitness_read"].to_numpy(dtype=float)
        counts, edges = np.histogram(values, bins=BINS, range=(0.0, 1.6))
        hists.append(
            pd.DataFrame(
                {
                    "source": source,
                    "bin_lo": edges[:-1],
                    "bin_hi": edges[1:],
                    "count": counts,
                }
            )
        )
        rows.append(
            summary(
                str(source),
                values,
                label="relative growth rate of the host",
                unit="1 = the run's uninhibited rate, 0 = no growth",
                orientation="served growth call; a well with no growth within the run is 0",
                value_is="one well of the wild-type host",
                n_conditions=int(
                    sub.groupby(
                        [c for c in sub.columns if c.startswith("dose_g_per_l_")]
                    ).ngroups
                ),
                n_compounds=int(sub["n_compounds"].max()),
                n_genes=0,
                n_wells=int(len(sub)),
                share_no_growth=float((~sub["grew"].astype(bool)).mean()),
            )
        )
    rows.append(
        {
            "source": "public IC30 anchors",
            "n_values": 0,
            "label": "host fitness asserted at the screen's IC30 dose",
            "unit": f"{IC30_FITNESS:.2f} at every anchored dose, 1.0 compound-free",
            "orientation": "not measured: IC30 names a 30 percent growth inhibition",
            "value_is": "one public IC30 condition, Vanacloig and Hoepfner only",
        }
    )
    hist = pd.concat(hists, ignore_index=True)
    hist.to_csv(osp.join(RESULTS, "source_label_histograms.csv"), index=False)
    pd.DataFrame(rows).to_csv(
        osp.join(RESULTS, "source_label_summary.csv"), index=False
    )
    over = pd.DataFrame(
        overlap,
        columns=["inhibitor", "inchikey", "source", "n_cells", "n_conditions", "doses"],
    )
    over.to_csv(osp.join(RESULTS, "source_inhibitor_overlap.csv"), index=False)
    print(over.to_string(index=False))
    print(
        pd.DataFrame(rows)[
            ["source", "n_values", "p05", "p50", "p95", "share_below_zero"]
        ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )


if __name__ == "__main__":
    main()
