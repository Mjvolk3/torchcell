#!/bin/bash
# notes/assets/publish/scripts/drawio_vector_svg.sh
#
# draw.io SVG -> a TRUE-VECTOR SVG whose text is outlined paths.
#
# WHY THIS EXISTS
# ---------------
# draw.io writes its labels as HTML `<foreignObject>` elements. Every browser draws
# those, and `rsvg-convert` -- which is what `bib_tex_pdf.sh` uses to embed an SVG as
# vector art in a note PDF -- cannot. The failure is not an error: the figure appears
# with EVERY LABEL TRUNCATED ("Gene tokens...") and a "Text is not SVG - cannot display"
# overlay, and the build exits 0.
#
# The workaround used to be embedding the PNG, which costs the vector.
#
# This script keeps both. draw.io's own PDF export already has the text as real glyphs,
# so round-tripping that PDF through `pdftocairo -svg` outlines them into paths. The
# result has zero foreignObject elements, renders identically in rsvg-convert, a
# browser, and Dendron's preview, and stays zoomable.
#
# Usage:
#   bash notes/assets/publish/scripts/drawio_vector_svg.sh notes/assets/drawio/foo.drawio
#
# Reads  <stem>.drawio.pdf  (produced by the drawio CLI export)
# Writes <stem>.vector.svg  (reference THIS one from a note)

set -euo pipefail

input="${1:?usage: drawio_vector_svg.sh <file.drawio>}"
stem="${input%.drawio}"
pdf="${stem}.drawio.pdf"
out="${stem}.vector.svg"

if [[ ! -f "${pdf}" ]]; then
    echo "error: ${pdf} not found. Export the drawio to PDF first:" >&2
    echo "  xvfb-run -a /tmp/drawio.AppImage '${input}' --no-sandbox --disable-gpu \\" >&2
    echo "    -x -f pdf --crop -o '${pdf}'" >&2
    exit 1
fi

pdftocairo -svg "${pdf}" "${out}"

# A foreignObject that survived means the outlining did not happen and the note would
# silently get a broken figure again. Fail loudly instead.
if grep -q "foreignObject" "${out}"; then
    echo "error: ${out} still contains foreignObject; text was not outlined" >&2
    exit 1
fi

echo "wrote ${out} ($(stat -c%s "${out}") bytes, text outlined)"
