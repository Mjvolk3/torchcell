# notes/assets/drawio/Fig2-gene-interactions.compose.py
# [[notes.assets.drawio.Fig2-gene-interactions.compose]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes/assets/drawio/Fig2-gene-interactions.compose
"""Place the data panels of Figure 2 into Fig2-gene-interactions.drawio.svg, headless.

The figure is a three-by-three grid of third-width slots (227.56 draw.io units, 57.8 mm).
Row 1 is the hand-drawn explainer (a) and the two panels placed by hand (b, binned
prediction; c, state of the art), which this script leaves alone. Row 2 held three
sketch placeholders; this script replaces them with the three generated panels at true
size, and row 3 keeps the nested-cell schematic (g) and drops the two remaining
placeholders. Run from the repo root after the panel scripts:

  experiments/010-kuzmin-tmi/scripts/fig2_additive_null_panel.py       -> d
  experiments/025-solid-growth/scripts/fig2_graphs_help_panel.py        -> e
  experiments/025-solid-growth/scripts/fig2_training_pool_panel.py      -> f

then export with the headless desktop build (xvfb). The figure must be closed in
draw.io while this runs; draw.io saves whole-file from memory and would revert it.
Idempotent: a second run finds the placeholders gone and the panels present, and
refreshes the panel payloads in place.
"""

from __future__ import annotations

import os
import os.path as osp
import re
import sys

from dotenv import load_dotenv

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "publish", "scripts"))
from drawio_true_size_import import image_cell, load_drawio, save_drawio  # noqa: E402

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
HERE = osp.dirname(osp.abspath(__file__))
FIG = osp.join(HERE, "Fig2-gene-interactions.drawio.svg")

# Row-2 slots: (cell id of the panel image, SVG, x of the slot frame). The frames
# (cells 182, 181, 189) stay as the rounded boxes every slot has; their sketch labels
# are cleared and their sketch images removed.
PANELS = [
    ("fig2-d", osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi", "fig2_additive_null_panel.svg"), "182"),
    ("fig2-e", osp.join(ASSET_IMAGES_DIR, "025-solid-growth", "fig2_graphs_help_panel.svg"), "181"),
    ("fig2-f", osp.join(ASSET_IMAGES_DIR, "025-solid-growth", "fig2_training_pool_panel.svg"), "189"),
]
REMOVE = ["184", "185", "190", "192", "201", "193", "187", "188"]  # sketches, row-3 boxes, i, j
RELABEL = {"51": "e", "168": "f", "186": "g"}  # letter cells, by id


def cell_block(mxfile: str, cell_id: str) -> re.Match[str] | None:
    return re.search(
        rf'<mxCell id="{cell_id}"[^>]*?(?:/>|>.*?</mxCell>)', mxfile, flags=re.S
    )


def geometry(mxfile: str, cell_id: str) -> tuple[float, float]:
    m = cell_block(mxfile, cell_id)
    assert m is not None, f"cell {cell_id} not found"
    g = re.search(r'<mxGeometry([^>]*)>', m.group(0))
    assert g is not None, f"cell {cell_id} has no geometry"
    x = re.search(r'x="([\d.-]+)"', g.group(1))
    y = re.search(r'y="([\d.-]+)"', g.group(1))
    return (float(x.group(1)) if x else 0.0, float(y.group(1)) if y else 0.0)


def main() -> None:
    text, mxfile, wrapped = load_drawio(FIG)
    for cid in REMOVE:
        m = cell_block(mxfile, cid)
        if m is not None:
            mxfile = mxfile[: m.start()] + mxfile[m.end() :]
            print("removed", cid)
    for cid, letter in RELABEL.items():
        m = cell_block(mxfile, cid)
        assert m is not None, f"letter cell {cid} missing"
        # The letter sits as `<b style="">x</b>` inside an HTML value, which in the
        # attribute reads `&gt;x&lt;/b&gt;`; replace exactly that letter.
        block = re.sub(
            r'value="([^"]*)"',
            lambda v: 'value="'
            + re.sub(r"(&gt;)([a-j])(&lt;/b&gt;)", rf"\g<1>{letter}\g<3>", v.group(1), count=1)
            + '"',
            m.group(0),
            count=1,
        )
        mxfile = mxfile[: m.start()] + block + mxfile[m.end() :]
    new_cells = []
    for cid, svg, frame in PANELS:
        fx, fy = geometry(mxfile, frame)
        # clear the frame's sketch label
        m = cell_block(mxfile, frame)
        assert m is not None
        block = re.sub(r'value="[^"]*"', 'value=""', m.group(0), count=1)
        mxfile = mxfile[: m.start()] + block + mxfile[m.end() :]
        old = cell_block(mxfile, cid)
        if old is not None:
            mxfile = mxfile[: old.start()] + mxfile[old.end() :]
        cell, _ = image_cell(cid, svg, fx)
        cell = cell.replace('y="0"', f'y="{fy + 6:.4f}"', 1)
        new_cells.append(cell)
    mxfile = mxfile.replace("</root>", "\n".join(new_cells) + "\n      </root>", 1)
    save_drawio(FIG, text, mxfile, wrapped)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
