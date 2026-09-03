#!/usr/bin/env python
# notes-tex/common/svg_true_size_pdf.py
# [[notes-tex.common.svg-true-size-pdf]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/common/svg_true_size_pdf.py
#
# Convert a plot SVG to PDF at its TRUE physical size, so a panel drawn 57.8 mm wide with
# 6 pt type arrives 57.8 mm wide with 6 pt type.
#
# WHY THIS EXISTS. `rsvg-convert -f pdf -o out.pdf in.svg` does not preserve physical size
# for either of the two SVG forms this repo produces:
#
#   torchcell.utils.savefig_true_size_svg writes UNITLESS width/height in draw.io's
#   100-units-per-inch canvas (a 57.8 mm panel is `width="227.5591"`). rsvg reads a
#   unitless length as pixels and writes one PDF point per pixel, so the panel comes out
#   227.56 pt = 80.3 mm, a factor of 100/72 = 1.389 too large.
#
#   create_ffa_multigraph_overlays._rescale_svg_to_mm writes explicit millimetres
#   (`width="131.63mm"`). rsvg converts millimetres to pixels at its default 90 dpi and
#   then writes one point per pixel, so that figure comes out 90/72 = 1.25 too large.
#
# Neither error is visible in the PDF on its own, which is what makes it worth a script:
# the figure simply prints bigger than designed, its 6 pt type lands near 8 pt, and the
# panel widths that were chosen to tile a 180 mm page no longer tile it.
#
# The fix is a per-file zoom computed from the header, since the correct factor depends on
# which of the two forms the file uses. Both factors are exact, so no rounding enters.
#
# NOTE FOR OTHER DOCUMENTS: the `plots` rule in notes-tex/common/Makefile.common still
# calls rsvg-convert directly and therefore still oversizes by 1.389. Any document using
# that rule for a savefig_true_size_svg panel should switch to this script.
#
#   python notes-tex/common/svg_true_size_pdf.py IN.svg OUT.pdf
#   python notes-tex/common/svg_true_size_pdf.py --check IN.svg     # print the size only

import argparse
import re
import subprocess
import sys

# rsvg-convert emits one PDF point per pixel, and a PDF point is 1/72 inch.
PT_PER_INCH = 72.0
MM_PER_INCH = 25.4
# The two unit conventions this repo's SVG writers use.
DRAWIO_UNITS_PER_INCH = 100.0
RSVG_DEFAULT_DPI = 90.0

HEADER = re.compile(
    r'<svg[^>]*\swidth="([\d.]+)(mm|pt|px|in)?"[^>]*\sheight="([\d.]+)(mm|pt|px|in)?"'
)


def svg_size_mm(path):
    """The file's intended physical size in millimetres, and the zoom rsvg needs.

    Returns (width_mm, height_mm, zoom). The zoom corrects rsvg's reading of the header,
    which differs by unit, so it is derived rather than assumed.
    """
    with open(path, "r", encoding="utf-8") as fh:
        head = fh.read(4000)
    m = HEADER.search(head)
    if not m:
        raise SystemExit(f"{path}: no width/height on the <svg> element")
    w, w_unit, h, h_unit = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
    if w_unit != h_unit:
        raise SystemExit(f"{path}: width is {w_unit!r} but height is {h_unit!r}")

    if w_unit == "mm":
        # rsvg turns mm into px at 90 dpi, then px into pt one for one.
        return w, h, PT_PER_INCH / RSVG_DEFAULT_DPI
    if w_unit in (None, "px"):
        # Unitless: written by savefig_true_size_svg in 100-units-per-inch.
        return (w / DRAWIO_UNITS_PER_INCH * MM_PER_INCH,
                h / DRAWIO_UNITS_PER_INCH * MM_PER_INCH,
                PT_PER_INCH / DRAWIO_UNITS_PER_INCH)
    if w_unit == "pt":
        return w / PT_PER_INCH * MM_PER_INCH, h / PT_PER_INCH * MM_PER_INCH, 1.0
    if w_unit == "in":
        return w * MM_PER_INCH, h * MM_PER_INCH, 1.0
    raise SystemExit(f"{path}: unhandled unit {w_unit!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("svg")
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("--check", action="store_true",
                    help="report the size without converting")
    args = ap.parse_args()

    w_mm, h_mm, zoom = svg_size_mm(args.svg)
    if args.check or not args.pdf:
        print(f"{args.svg}: {w_mm:.2f} x {h_mm:.2f} mm  (rsvg zoom {zoom:.4f})")
        return 0

    subprocess.run(
        ["rsvg-convert", "-f", "pdf", "-z", f"{zoom:.10f}", "-o", args.pdf, args.svg],
        check=True,
    )
    print(f"  {args.svg} -> {args.pdf}  ({w_mm:.1f} x {h_mm:.1f} mm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
