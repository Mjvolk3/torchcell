#!/usr/bin/env bash
# paper/nature-biotech/check-figures.sh
#
# Codifies the WYSIWYG figure contract (steps 2-3 of the figure workflow):
#   2. SIZE  -- each exported figures/*.pdf must fit Nature's print box
#               (<= 180 x 240 mm, with a small grace for draw.io stroke/rounding),
#               so the draw.io mm/pt sizing maps 1:1 into the document.
#   3. SCALE -- figures must be placed true-size with \tcfig (NO scaling). Any use
#               of \tcfigfit re-scales the PDF and breaks the WYSIWYG font promise.
#
# Step 1 (drawing content inside the 180x240 mm box) is manual and on the author;
# this script verifies that what came out of draw.io can be placed verbatim.
#
# Usage:        bash check-figures.sh
# Exit status:  0 = all figures pass; 1 = at least one violation (gates CI/builds).

set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIGDIR="$DIR/figures"
MAXW=180        # Nature full-width, mm
MAXH=240        # max figure height, mm (relaxed from 170; tall multi-panel figures)
TOL=2           # grace (mm) for the guide-box stroke width / export rounding

# Colour only when writing to a real terminal (keeps CI/piped logs clean).
if [ -t 1 ]; then
  G=$'\033[32m'; R=$'\033[31m'; D=$'\033[2m'; B=$'\033[1m'; Z=$'\033[0m'
else
  G=''; R=''; D=''; B=''; Z=''
fi
OK="${G}✓${Z}"; BAD="${R}✗${Z}"

fail=0

echo "${B}== Figure SIZE check ==${Z}  ${D}limit ${MAXW} x ${MAXH} mm (+${TOL} mm grace)${Z}"
shopt -s nullglob
for pdf in "$FIGDIR"/*.pdf; do
  name=$(basename "$pdf")
  out=$(pdfinfo "$pdf" | awk -v n="$name" -v mw="$MAXW" -v mh="$MAXH" -v tol="$TOL" \
        -v ok="$OK" -v bad="$BAD" -v r="$R" -v d="$D" -v z="$Z" '
    /Page size/ {
      wmm = $3 * 25.4 / 72; hmm = $5 * 25.4 / 72;
      over = (wmm > mw + tol || hmm > mh + tol);
      note = "";
      if (wmm > mw + tol) note = note sprintf("  W +%.0f mm", wmm - mw);
      if (hmm > mh + tol) note = note sprintf("  H +%.0f mm", hmm - mh);
      printf "  %s  %6.1f x %6.1f mm   %s%s%s", (over ? bad : ok), wmm, hmm, n, \
             (over ? r : d), note z;
      exit (over ? 2 : 0);
    }')
  st=$?
  echo "$out"
  [ "$st" -ne 0 ] && fail=1
done

echo
echo "${B}== Figure SCALE check ==${Z}  ${D}true-size \\tcfig only (no \\tcfigfit)${Z}"
hits=$(grep -rn --include='*.tex' '\\tcfigfit' "$DIR/sections" "$DIR"/content.tex 2>/dev/null || true)
if [ -n "$hits" ]; then
  echo "  ${BAD}  \\tcfigfit (scaling) found -- switch to true-size \\tcfig:"
  echo "$hits" | sed "s/^/      ${D}/; s/\$/${Z}/"
  fail=1
else
  echo "  ${OK}  all figures placed true-size"
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "  ${R}${B}✗ FIGURE CHECK FAILED${Z} -- fix in draw.io (pull content inside the 180x240 mm box) or restore \\tcfig."
  exit 1
fi
echo "  ${G}${B}✓ FIGURE CHECK PASSED${Z} -- all figures within the print box and placed true-size (WYSIWYG)."
exit 0
