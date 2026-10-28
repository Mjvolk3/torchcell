# experiments/043-expression-proteome-morphology/scripts/figure_043_sheets.py
# [[experiments.043-expression-proteome-morphology.scripts.figure_043_sheets]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/figure_043_sheets
"""The two 043 figure sheets (owner, 2026-10-27: "I think 2 total. 1 is setup and
pretraining. 2. is our preliminary evaluations", drawn with more synthesis per sheet,
placeholders allowed where a panel is an idea only).

Sheet 1, setup and pretraining: the records read and the split (data panels), the
model in math with its symbol table and the conditioning routes (model panels), the
pretraining curves (loss, heads, levers, the two collapses). Sheet 2, preliminary
evaluations: the model-free baselines beside the trained arms, the lever ladders on
doubles, Muenzner and Kemmeren, the fine-tune gain, the composition and isolate
diagnostics, and two placeholders.

Every panel is drawn by one of the panel modules (``figure_043_data_panels``,
``figure_043_model_panel``, ``figure_043_curve_panels``, ``figure_043_eval_panels``)
from a result file under ``results/`` or the W&B cache
``results/figure_043_curves.csv``; the layout and the status marks live here.

    python experiments/043-expression-proteome-morphology/scripts/figure_043_sheets.py [--only 1|2]
    -> $ASSET_IMAGES_DIR/043-expression-proteome-morphology/figure_043_{setup,evals}{_<stamp>.svg,.png,.pdf}
"""

from __future__ import annotations

import argparse
import os.path as osp
import sys
from typing import Any

sys.path.insert(0, osp.dirname(__file__))

import figure_043_common as C  # noqa: E402

PanelFn = Any


def _panels() -> dict[str, PanelFn]:
    """Every panel of the four modules under one namespace, ``<module>.<key>``."""
    import figure_043_curve_panels as curves
    import figure_043_data_panels as data
    import figure_043_eval_panels as evals
    import figure_043_model_panel as model

    out: dict[str, PanelFn] = {}
    for name, mod in (
        ("data", data),
        ("model", model),
        ("curves", curves),
        ("evals", evals),
    ):
        for key, fn in mod.PANELS.items():
            out[f"{name}.{key}"] = fn
    return out


def _draw(
    fig: Any, axes: list[Any], keys: list[str], panels: dict[str, PanelFn], R: C.Reads
) -> None:
    statuses = []
    for ax, key in zip(axes, keys, strict=True):
        if key not in panels:
            statuses.append(
                C.placeholder(ax, f"panel {key!r} is not drawn by any module yet", key)
            )
            continue
        statuses.append(panels[key](ax, R))
    C.letters_status(fig, axes, statuses)
    C.status_key(fig)


#: sheet 1: four rows, heights in mm (the data row holds 15 to 19 bar rows, the model
#: row the five equation blocks and the symbol table)
SETUP_ROWS: list[list[str]] = [
    ["data.records", "data.split", "data.ontology"],
    ["model.model", "model.symbols"],
    ["model.conditioning", "curves.loss"],
    ["curves.heads", "curves.levers", "curves.collapse", "curves.bs32"],
]
SETUP_HEIGHTS = [46.0, 64.0, 52.0, 36.0]
SETUP_WIDTHS: list[list[float] | None] = [None, [1.05, 0.95], [1.45, 1.0], None]

#: sheet 2: the baselines strip across the top, then three rows of three
EVALS_ROWS: list[list[str]] = [
    ["evals.baselines"],
    ["evals.ladder", "evals.ladder_muenzner", "evals.ladder_kemmeren"],
    ["evals.finetune", "evals.composition", "evals.isolates"],
    ["evals.morphology", "evals.scatter", "curves.throughput"],
]
EVALS_HEIGHTS = [46.0, 42.0, 42.0, 42.0]
EVALS_WIDTHS: list[list[float] | None] = None

#: the letter bands: the top band plus one gap per row boundary, in mm
BAND_MM = 7.0


def sheet(
    name: str,
    layout: list[list[str]],
    heights: list[float],
    widths: list[list[float] | None] | None,
    panels: dict[str, PanelFn],
    R: C.Reads,
) -> dict[str, str]:
    """One sheet; ``heights`` are row heights in mm, the sheet adds the letter bands."""
    fig = C.new_sheet(sum(heights) + BAND_MM * len(heights))
    axes = C.rows(
        fig, len(layout), [len(r) for r in layout], widths=widths, heights=heights
    )
    _draw(fig, axes, [k for row in layout for k in row], panels, R)
    return C.save(fig, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["1", "2"], default=None)
    args = ap.parse_args()
    panels = _panels()
    R = C.Reads()
    if args.only in (None, "1"):
        out = sheet(
            "figure_043_setup", SETUP_ROWS, SETUP_HEIGHTS, SETUP_WIDTHS, panels, R
        )
        print(out["png"])
    if args.only in (None, "2"):
        out = sheet(
            "figure_043_evals", EVALS_ROWS, EVALS_HEIGHTS, EVALS_WIDTHS, panels, R
        )
        print(out["png"])


if __name__ == "__main__":
    main()
