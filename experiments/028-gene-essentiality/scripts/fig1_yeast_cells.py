# experiments/028-gene-essentiality/scripts/fig1_yeast_cells.py
# [[experiments.028-gene-essentiality.scripts.fig1_yeast_cells]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/fig1_yeast_cells
"""The three yeast cell drawings of manuscript Figure 1, written out as PNG assets.

``notes/assets/drawio/Fig1-torchcell-overview.drawio.svg`` embeds its raster images as
base64 PNG. Three of them are the same yeast cell drawing in three tints (the orange,
rose and sand strains of Figure 1's bottom-right block). The gate sheet of experiment 028
draws its label panel with them, so they are written once here, by position in the SVG,
to ``$ASSET_IMAGES_DIR/028-gene-essentiality/yeast_cell_<tint>.png``.

    PYTHONPATH=$PWD python experiments/028-gene-essentiality/scripts/fig1_yeast_cells.py
"""

from __future__ import annotations

import base64
import os
import os.path as osp
import re

from dotenv import load_dotenv

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
SVG = osp.join(ASSET_IMAGES_DIR, "..", "drawio", "Fig1-torchcell-overview.drawio.svg")
OUT = osp.join(ASSET_IMAGES_DIR, "028-gene-essentiality")
#: The three cells, by the x and y the SVG places them at (the only stable handle).
CELLS = {"orange": ("489.3", "124.45"), "rose": ("528.5", "58.94"), "sand": ("489.3", "95.86")}


def main() -> None:
    with open(SVG) as fh:
        svg = fh.read()
    found = {
        (x, y): data
        for x, y, data in re.findall(
            r'<image x="([-\d.]+)" y="([-\d.]+)" width="[\d.]+" height="[\d.]+" '
            r'xlink:href="data:image/png;base64,([^"]+)"',
            svg,
        )
    }
    os.makedirs(OUT, exist_ok=True)
    for tint, key in CELLS.items():
        assert key in found, f"no image at x={key[0]} y={key[1]} in Fig 1"
        path = osp.join(OUT, f"yeast_cell_{tint}.png")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(found[key]))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
