# notes/assets/drawio/Fig2-gene-interactions.explainer.py
# [[notes.assets.drawio.Fig2-gene-interactions.explainer]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes/assets/drawio/Fig2-gene-interactions.explainer
"""Export panel a of Figure 2, the hand-drawn explainer, from Fig2-gene-interactions.drawio.svg.

Keeps every cell whose top-level ancestor lies in the first slot (x < 232, y < 240 draw.io
units) except the slot's frame and letter, writes that subset as a plain .drawio, and
exports it with the draw.io desktop binary under xvfb (DRAWIO names the binary) to
$ASSET_IMAGES_DIR/025-solid-growth/fig2_explainer.png at six times true size, which the
sheet (experiments/025-solid-growth/scripts/fig2_gene_interactions.py) places in panel a.
"""

from __future__ import annotations

import os
import os.path as osp
import subprocess
import sys
import xml.etree.ElementTree as ET

from dotenv import load_dotenv

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "publish", "scripts"))
from drawio_true_size_import import load_drawio  # noqa: E402

load_dotenv()
HERE = osp.dirname(osp.abspath(__file__))
FIG = osp.join(HERE, "Fig2-gene-interactions.drawio.svg")
OUT = osp.join(os.environ["ASSET_IMAGES_DIR"], "025-solid-growth", "fig2_explainer.png")
DRAWIO = os.environ["DRAWIO"]
SLOT = (232.0, 240.0)
EXCLUDE = {"194", "48"}  # the slot's frame and its letter


def main() -> None:
    _, mx, _ = load_drawio(FIG)
    root = ET.fromstring(mx)
    model = root.find(".//mxGraphModel")
    assert model is not None
    cells = model.find("root")
    assert cells is not None
    by_id = {c.get("id"): c for c in cells}

    def top(c: ET.Element) -> ET.Element:
        while c.get("parent") not in ("0", "1", None):
            c = by_id[c.get("parent")]
        return c

    def inside(c: ET.Element) -> bool:
        g = top(c).find("mxGeometry")
        if g is None:
            return False
        return float(g.get("x") or 0) < SLOT[0] and float(g.get("y") or 0) < SLOT[1]

    keep = []
    for c in list(cells):
        if c.get("id") in ("0", "1"):
            continue
        if c.get("edge") == "1":
            ends = [by_id[e] for e in (c.get("source"), c.get("target")) if e in by_id]
            if ends:
                ok = all(inside(e) for e in ends)
            else:
                # A free edge is placed by its own points (the loop arrows of the nested
                # cell slot have no source or target cell).
                pts = [p for p in c.iter("mxPoint") if p.get("x") is not None]
                ok = bool(pts) and all(float(p.get("x") or 0) < SLOT[0] and float(p.get("y") or 0) < SLOT[1] for p in pts)
        else:
            ok = inside(c)
        if ok and top(c).get("id") not in EXCLUDE:
            keep.append(c)
    new_root = ET.Element("root")
    for cid in ("0", "1"):
        new_root.append(by_id[cid])
    for c in keep:
        new_root.append(c)
    model.remove(cells)
    model.append(new_root)
    tmp = osp.join(osp.dirname(OUT), "fig2_explainer.drawio")
    open(tmp, "w").write(ET.tostring(root, encoding="unicode"))
    cmd = [
        "xvfb-run", "-a", DRAWIO, "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
        "--export", "--format", "png", "--scale", "6", "--crop", "--output", OUT, tmp,
    ]
    subprocess.run(cmd, check=True, env={**os.environ, "ELECTRON_DISABLE_SECURITY_WARNINGS": "true"}, capture_output=True)
    os.remove(tmp)
    print(f"{len(keep)} cells ->", OUT)


if __name__ == "__main__":
    main()
