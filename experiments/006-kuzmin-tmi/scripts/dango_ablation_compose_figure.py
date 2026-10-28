# experiments/006-kuzmin-tmi/scripts/dango_ablation_compose_figure.py
# [[experiments.006-kuzmin-tmi.scripts.dango_ablation_compose_figure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/dango_ablation_compose_figure
"""Compose the DANGO ablation SI figure as a draw.io file from the true-size panel SVGs.

One row of three third-width panels (57.8 mm each, 52 mm tall) from ``dango_ablation_si.py``
(this folder): (a) validation Pearson r against epoch per arm over the five earlier STRING
v9.1 runs, (b) best validation r and test r at the minimum-validation-MSE checkpoint per arm,
(c) the validation reconstruction loss per arm. The script refuses any panel that is not third
width. Each panel SVG declares its size in draw.io units (100 per inch, via
``savefig_true_size_svg``), so the image cells are placed at exact physical size.

Panel letters are 8 pt bold lowercase (draw.io fontSize 11.1). Rerun after regenerating any
panel; the draw.io file is overwritten, never edited by hand.

Layout: the "white cross" convention of every composed SI figure (copied from
``experiments/010-kuzmin-tmi/scripts/dango_full_dataset_compose_figure.py``): COL_GAP = 12
units (3 mm) between columns, a TOP_STRIP of 16 units above the row, and each panel letter in
that strip at the panel's top-left (x = panel_x, y = row_top).

Export (``--export``) runs draw.io headless under a private D-Bus session, which the GilaHyper
export needs (Electron otherwise blocks on the session bus), checks that the PDF was written
during the call (an mtime stamp taken before it: neither xvfb-run's nor draw.io's exit status
says whether a file was written), then runs ``drawio_font_band.py --check``. Set ``DRAWIO`` to
the extracted drawio-desktop binary.

Run from the repo root:
    python experiments/006-kuzmin-tmi/scripts/dango_ablation_compose_figure.py [--export]
"""

import argparse
import base64
import os
import os.path as osp
import re
import subprocess
import sys
import time
from xml.sax.saxutils import quoteattr

from dotenv import load_dotenv

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
DRAWIO_DIR = "notes/assets/drawio"
FIG_NAME = "FigS-dango-ablation"
PDF_OUT = f"paper/nature-biotech/figures/{FIG_NAME}.pdf"
FONT_BAND = "paper/nature-biotech/scripts/drawio_font_band.py"

FULL_WIDTH = 709  # 180 mm in draw.io units (cap)
MAX_HEIGHT = 669  # 170 mm
COL_GAP = 12  # 3 mm between columns
TOP_STRIP = 16  # the letter strip above the row
LETTER_W, LETTER_H = 18, 14
LETTER_STYLE = (
    "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;"
    "whiteSpace=wrap;rounded=0;fontFamily=Arial;fontSize=11.1;fontStyle=1;"
)
IMAGE_STYLE = (
    "shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;"
    "verticalAlign=top;aspect=fixed;imageAspect=0;image=data:image/svg+xml,{b64};"
)
P006 = osp.join(ASSET_IMAGES_DIR, "006-kuzmin-tmi")
THIRD_WIDTH_MM = 57.8  # torchcell.utils.PANEL_WIDTHS_MM["third"]
UNITS_PER_MM = 100 / 25.4  # draw.io canvas units (100 per inch)
ROW = [
    ("a", "dango_ablation_curves.svg"),
    ("b", "dango_ablation_scores.svg"),
    ("c", "dango_ablation_reconstruction.svg"),
]


def svg_size(path: str) -> tuple[float, float]:
    head = open(path, encoding="utf-8").read(2000)
    w = float(re.search(r'width="([0-9.]+)"', head).group(1))
    h = float(re.search(r'height="([0-9.]+)"', head).group(1))
    return w, h


def image_cell(cid: int, path: str, x: float, y: float) -> str:
    w, h = svg_size(path)
    b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
    style = IMAGE_STYLE.format(b64=b64)
    return (
        f'<mxCell id="{cid}" value="" style={quoteattr(style)} vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.4f}" height="{h:.4f}" as="geometry"/></mxCell>'
    )


def letter_cell(cid: int, letter: str, x: float, row_top: float) -> str:
    """Panel letter in the TOP_STRIP above the panel, flush with the panel's left edge."""
    return (
        f'<mxCell id="{cid}" value="{letter}" style={quoteattr(LETTER_STYLE)} vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{row_top:.1f}" width="{LETTER_W}" height="{LETTER_H}" as="geometry"/></mxCell>'
    )


def compose() -> str:
    third = THIRD_WIDTH_MM * UNITS_PER_MM
    for _, name in ROW:
        w = svg_size(osp.join(P006, name))[0]
        if abs(w - third) > 0.5:
            raise SystemExit(f"{name} is {w:.2f} units wide; every panel must be third width ({third:.2f})")

    cells, cid = [], 2
    row_top = 0.0
    y = row_top + TOP_STRIP
    extent_w = 0.0
    for col, (letter, name) in enumerate(ROW):
        x = col * (third + COL_GAP)
        cells.append(image_cell(cid, osp.join(P006, name), x, y))
        cells.append(letter_cell(cid + 1, letter, x, row_top))
        cid += 2
        extent_w = max(extent_w, x + svg_size(osp.join(P006, name))[0])
    extent_h = y + max(svg_size(osp.join(P006, name))[1] for _, name in ROW)

    xml = (
        '<mxfile host="dango_ablation_compose_figure.py">'
        f'<diagram id="{FIG_NAME}" name="Page-1">'
        '<mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
        'arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">'
        '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>"
    )
    out = osp.join(DRAWIO_DIR, f"{FIG_NAME}.drawio")
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"wrote {out}: {extent_w:.0f} x {extent_h:.0f} units = {extent_w / UNITS_PER_MM:.1f} x {extent_h / UNITS_PER_MM:.1f} mm")
    if extent_w > FULL_WIDTH + 8 or extent_h > MAX_HEIGHT + 8:
        raise SystemExit(f"{FIG_NAME} exceeds the Nature print box (709 x 669 units)")
    return out


def export(drawio_path: str):
    """Headless draw.io PDF export under a private D-Bus session, then the font-band audit."""
    drawio = os.environ["DRAWIO"]
    stamp = time.time()
    cmd = [
        "env", "-u", "DBUS_SESSION_BUS_ADDRESS", "timeout", "180", "dbus-run-session", "--",
        "xvfb-run", "-a", drawio, drawio_path, "--password-store=basic", "--no-sandbox",
        "--disable-gpu", "--disable-dev-shm-usage", "-x", "-f", "pdf", "--crop", "-o", PDF_OUT,
    ]
    subprocess.run(cmd, check=False)
    if not osp.exists(PDF_OUT) or osp.getmtime(PDF_OUT) < stamp:
        raise SystemExit(f"draw.io did not write {PDF_OUT} during this call")
    print(f"exported {PDF_OUT}")
    subprocess.run([sys.executable, FONT_BAND, "--check", drawio_path], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="export the PDF headless and audit fonts")
    args = ap.parse_args()
    path = compose()
    if args.export:
        export(path)


if __name__ == "__main__":
    main()
