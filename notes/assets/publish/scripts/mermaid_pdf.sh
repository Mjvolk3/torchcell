#!/bin/bash
# Render standalone mermaid diagram(s) from a markdown file to vector PDF.
#
# Output PDF name MATCHES the input markdown file name (Dendron fname) and lands
# in notes/assets/pdf-output/. Example:
#   notes/torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii.md
#   -> notes/assets/pdf-output/torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii.pdf
#
# Uses the GLOBAL mermaid-cli (mmdc, v11+), NOT the pandoc `mermaid-filter` path.
# mermaid v11 renders KaTeX, so $$...$$ math in node/cluster/edge labels typesets
# correctly. mmdc -> headless Chromium -> print-to-PDF gives a true vector PDF
# (crisp, selectable math) suitable for placing into a draw.io figure.
#
# Usage:
#   bash notes/assets/publish/scripts/mermaid_pdf.sh <path/to/file.md> [bg]
#     <path/to/file.md>  markdown file containing one (or more) ```mermaid blocks
#     [bg]               background passed to mmdc -b (default: transparent)
#
# Notes:
#   - mmdc names per-diagram outputs <base>-1.pdf, <base>-2.pdf, ...
#     For a single-diagram file we rename <base>-1.pdf -> <base>.pdf so the PDF
#     name matches the md exactly. Multi-diagram files keep the numeric suffixes.

set -euo pipefail

input_file="$1"
bg="${2:-transparent}"

base="$(basename "${input_file%.md}")"        # e.g. torchcell.models...type-i-ii
outdir="notes/assets/pdf-output"

# --pdfFit fits the page to the diagram (no wasted whitespace around the chart).
mmdc -i "${input_file}" -o "${outdir}/${base}.pdf" -b "${bg}" --pdfFit

# Single-diagram file: collapse the mmdc "-1" suffix to the exact md name.
if [ -f "${outdir}/${base}-1.pdf" ] && [ ! -f "${outdir}/${base}-2.pdf" ]; then
  mv -f "${outdir}/${base}-1.pdf" "${outdir}/${base}.pdf"
else
  echo "Multi-diagram file -> ${outdir}/${base}-N.pdf (no SVG/PNG sidecar made):"
  ls -1 "${outdir}/${base}"-*.pdf
  exit 0
fi

# Outlined vector SVG for draw.io. pdftocairo (cairo) renders all glyphs as PATHS,
# not <text>, so the SVG is self-contained (no font dependency) and survives being
# EMBEDDED AS AN IMAGE in draw.io. Do NOT use draw.io "Import PDF" -- that converts
# the content stream into scattered editable text boxes.
pdftocairo -svg "${outdir}/${base}.pdf" "${outdir}/${base}.svg"

# High-DPI PNG fallback (raster), in case an SVG embed is not wanted.
rsvg-convert -z 4 "${outdir}/${base}.svg" -o "${outdir}/${base}.png" 2>/dev/null || true

echo "Outputs:"
echo "  ${outdir}/${base}.pdf   (vector, fitted)"
echo "  ${outdir}/${base}.svg   (outlined vector -> embed as image in draw.io)"
echo "  ${outdir}/${base}.png   (high-DPI raster fallback)"
