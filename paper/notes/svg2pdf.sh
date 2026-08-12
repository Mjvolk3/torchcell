#!/bin/bash
# paper/notes/svg2pdf.sh -- convert a repo figure SVG to a TRUE-PHYSICAL-SIZE vector PDF.
#
#   svg2pdf.sh <in.svg> <out.pdf>
#
# Why this is not just `rsvg-convert -f pdf`: the repo has TWO families of figure SVG whose
# root coordinates mean different things, and rsvg's default (90 units/inch) is wrong for
# both. Getting this wrong is silent -- the figure simply renders at the wrong size, which
# breaks the WYSIWYG contract the editing view exists to provide (a 6 pt font must print at
# 6 pt, a 180 mm panel must occupy 180 mm).
#
#   1. PLAIN matplotlib savefig    root is width="416.38pt"  -> PostScript points, 72/inch.
#      rsvg's 90/inch default inflates these by exactly 1.25x.
#
#   2. savefig_true_size_svg       root is width="348.4252"  -> UNITLESS, and per
#      torchcell/utils/utils.py that number is in draw.io's native unit of 100/inch (the
#      function rewrites matplotlib's pt-tagged root precisely so draw.io imports at true
#      size). Read at 90 or 96/inch it comes out oversized. 348.4252/100 = 3.484 in =
#      88.5 mm, exactly PANEL_WIDTHS_MM["half_plus"] -- the intended column width.
#
# So the unit is decided by whether the root width carries an explicit `pt`, which is a real
# property of the file, not a guess. Anything else is an error rather than a default, because
# a silently mis-sized figure is worse than a failed build.
#
# HOW each family is converted, and why they differ -- this part is rsvg's doing, not ours:
#   pt-tagged   --dpi-x/y 72       rsvg honours --dpi for a unit-tagged root, so this lands
#                                  the declared points 1:1.
#   unitless    -z 0.72            rsvg IGNORES --dpi when the root is unitless -- it maps
#                                  1 user unit to 1 pt whatever you pass (verified: 479.7762
#                                  in gives 479.77 pt out at --dpi 100). The zoom does it
#                                  instead: 72/100, taking user units at 100/inch to points
#                                  at 72/inch. Checked against the generating script's intent
#                                  -- construction_validation_doubles.py asks for
#                                  PANEL_WIDTHS_MM["wide"] = 118.9 mm and this yields 122 mm,
#                                  where the naive conversion gave 169 mm.

set -euo pipefail

in="$1"
out="$2"

# The root width attribute -- first width="..." in the file.
w=$(grep -oE 'width="[^"]*"' "$in" | head -1 | sed -E 's/width="(.*)"/\1/')

case "$w" in
  *pt)
    scale=(--dpi-x 72 --dpi-y 72) ;;   # family 1: PostScript points, 1:1
  ''|*[!0-9.]*)
    echo "svg2pdf: $in -- root width '$w' is neither pt nor a bare number; refusing to guess" >&2
    exit 1 ;;
  *)
    scale=(-z 0.72) ;;                 # family 2: draw.io units at 100/inch -> 72/inch
esac

mkdir -p "$(dirname "$out")"
rsvg-convert "${scale[@]}" -f pdf -o "$out" "$in"
