# experiments/008-xue-ffa/scripts/perspective_motivation_drawio.py
# [[experiments.008-xue-ffa.scripts.perspective_motivation_drawio]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/perspective_motivation_drawio
#
# The top row of Fig. 1: two schematic panels that say what the document is for, as a
# native draw.io page 179 mm wide and one row tall. Panel a is the motivation, a yeast in
# nature (robust, cross-coupled, most of its chemistry off the curated map) against a
# yeast in a bioreactor (one product, one number per strain). Panel b is the classic
# picture: an interaction is the departure of a measured combination from an expectation,
# with the genome-wide counts on yeast growth beside it and what predicting such
# departures would buy a design. The drawing functions live in
# epistasis_model_intuition_drawio.py, where the two panels were first built; this script
# places them on their own page so Fig. 1 can carry them above its three measured panels
# (author review, 2026.09.18: "we wanted these as motivation in fig 1").
#
# The letters a and b are drawn here, since the row is one PDF and the document's \panel
# macro places one letter per file. The measured panels under it are lettered c, d, e by
# the document.

import argparse
import os
import os.path as osp
import sys

from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from drawio_doc import U, Doc, export, letter_style  # noqa: E402
from epistasis_model_intuition_drawio import (  # noqa: E402
    H_ROW0,
    LETTER_DY,
    W_MM,
    classic,
    mm,
    motivation,
)

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
DRAWIO_DIR = osp.join(osp.dirname(ASSET_IMAGES_DIR), "drawio")

Y_TOP = 5.0  # the letters sit LETTER_DY above this


def build(out_path):
    doc = Doc(osp.splitext(osp.basename(out_path))[0])
    # Letters at 2 and 90.5 mm, content indented 4 mm past each, and both panels end
    # inside the 177 mm the rows of Fig. 3 use, so the two figures share one left edge.
    half = (W_MM - 6.0) / 2.0
    doc.vertex("letter-a", "a", letter_style(), mm(2), mm(Y_TOP - LETTER_DY), 24, 18)
    motivation(doc, 6.0, Y_TOP, half - 4.0, H_ROW0)
    xb = 2.0 + half + 2.0
    doc.vertex("letter-b", "b", letter_style(), mm(xb), mm(Y_TOP - LETTER_DY), 24, 18)
    classic(doc, xb + 4.0, Y_TOP, 177.0 - (xb + 4.0), H_ROW0)
    # An invisible frame nearly the full width of the page, so the export is as wide as
    # the three panels under it whatever the two schematics' extent. The crop adds half a
    # millimetre to a frame, so 178.8 exports inside the 179.4 cap.
    frame_w = 178.8
    doc.vertex("frame", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;",
               0, 0, mm(frame_w), mm(Y_TOP + H_ROW0 + 1.0))
    doc.write(out_path)
    print(f"wrote {out_path}  ({frame_w:.1f} x {Y_TOP + H_ROW0 + 1.0:.1f} mm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=osp.join(DRAWIO_DIR, "ffa-epistasis-fig1-motivation.drawio"))
    ap.add_argument("--drawio", default=None)
    args = ap.parse_args()
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    build(args.out)
    if args.drawio:
        stem = osp.splitext(osp.basename(args.out))[0]
        export(args.drawio, args.out, osp.join(IMAGES_DIR, stem + ".svg"),
               osp.join(IMAGES_DIR, stem + ".png"))


if __name__ == "__main__":
    main()
