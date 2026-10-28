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

    python experiments/043-expression-proteome-morphology/scripts/figure_043_sheets.py [--only 1|2] [--tex]
    -> $ASSET_IMAGES_DIR/043-expression-proteome-morphology/figure_043_{setup,evals}{_<stamp>.svg,.png,.pdf}

``--tex`` (owner, 2026-10-28, "would prefer to have math in latex") composes sheet 1 a
second time as ``figure_043_setup_tex`` (PDF and PNG, no SVG) with the model panels of
``figure_043_model_panel_tex`` (real LaTeX) in place of the mathtext ones; every other
panel is the same.
"""

from __future__ import annotations

import argparse
import contextlib
import os.path as osp
import sys
from typing import Any

import matplotlib

sys.path.insert(0, osp.dirname(__file__))

import figure_043_common as C  # noqa: E402

PanelFn = Any


def _panels(tex: bool = False) -> dict[str, PanelFn]:
    """Every panel of the four modules under one namespace, ``<module>.<key>``;
    ``tex`` takes the model panels from ``figure_043_model_panel_tex``.
    """
    import figure_043_curve_panels as curves
    import figure_043_data_panels as data
    import figure_043_eval_panels as evals
    import figure_043_model_panel as model

    if tex:
        import figure_043_model_panel_tex as model  # type: ignore[no-redef]

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
#: row the five equation blocks and the symbol table). The epoch share sits between the
#: conditioning and the loss curve: row 1 has no width to spare and a fifth row would
#: take the sheet past 235 mm.
SETUP_ROWS: list[list[str]] = [
    ["data.records", "data.split", "data.ontology"],
    ["model.model", "model.symbols"],
    ["model.conditioning", "data.epoch_share", "curves.loss"],
    ["curves.heads", "curves.levers", "curves.collapse", "curves.bs32"],
]
SETUP_HEIGHTS = [46.0, 64.0, 52.0, 36.0]
SETUP_WIDTHS: list[list[float] | None] = [None, [1.05, 0.95], [1.72, 0.72, 0.66], None]

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
    tex: bool = False,
) -> dict[str, str]:
    """One sheet; ``heights`` are row heights in mm, the sheet adds the letter bands.

    ``tex``: the sheet holds usetex text, so it is drawn and saved under the TeX rc of
    ``figure_043_model_panel_tex`` (entered after ``new_sheet``, which applies the
    sheet rc) and saved as PDF and PNG only.
    """
    fig = C.new_sheet(sum(heights) + BAND_MM * len(heights))
    axes = C.rows(
        fig, len(layout), [len(r) for r in layout], widths=widths, heights=heights
    )
    if tex:
        import figure_043_model_panel_tex as T

        ctx: Any = matplotlib.rc_context(T.SAVE_RC)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        _draw(fig, axes, [k for row in layout for k in row], panels, R)
        return C.save(fig, name, tex=tex)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["1", "2"], default=None)
    ap.add_argument(
        "--tex", action="store_true", help="sheet 1 with LaTeX model panels only"
    )
    args = ap.parse_args()
    R = C.Reads()
    if args.tex:
        out = sheet(
            "figure_043_setup_tex",
            SETUP_ROWS,
            SETUP_HEIGHTS,
            SETUP_WIDTHS,
            _panels(tex=True),
            R,
            tex=True,
        )
        print(out["png"])
        return
    panels = _panels()
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
