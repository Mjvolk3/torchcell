#!/bin/bash
# Whole-note -> PDF (pandoc + xelatex + citeproc, Nature CSL).
#
# Folds in the assay-note preprocessing that used to live in a one-off
# build_assay_pdf.py:
#   * Sanitize non-ASCII glyphs the default xelatex font lacks (arrows, math
#     operators, etc.) to LaTeX equivalents -- skipping fenced code blocks so
#     shell snippets are left intact.
# This runs into a TEMP markdown that pandoc consumes; the original note is never
# modified. Relative image/bib paths still resolve because pandoc runs from ./notes.
#
# SVG figures are embedded as VECTOR (zoomable) -- pandoc converts each .svg to PDF
# via rsvg-convert (librsvg, /usr/bin/rsvg-convert). We therefore do NOT swap
# .svg->.png anymore; reference the .svg in the note to get crisp zoom, keep .png for
# rasters (photos/overlays). If rsvg-convert is ever missing, install librsvg2-tools.
#
# Dendron wikilinks are rendered by assets/publish/filters/dendron-links.lua. Pandoc has
# no idea what `[[note.path]]` means and prints the brackets verbatim, which puts raw
# markup where a reference belongs. The filter turns each one into the target note's title
# plus its dendron path in typewriter, which is what a reader can act on in print. It runs
# on the parsed AST, so brackets inside math, inline code and code fences are untouched.
#
# assets/publish/filters/keep-tables.lua marks each table that is small enough to sit on
# one page, so it moves down whole instead of splitting away from its caption. It measures
# the table because the header cannot: an unbreakable table taller than a page overflows
# and loses rows without failing the build.
#
# The pandoc `-F mermaid-filter` path is intentionally NOT used (its bundled
# puppeteer times out -- see CLAUDE.md); pre-render mermaid diagrams with
# mermaid_pdf.sh and reference the produced image instead.
#
# CANONICAL STYLE: this build is matched to paper/nature-biotech/editing.pdf.
# The paper's drafting view is the house look for every notes-tex document, so the
# geometry, page size, type size, numbered sections and contents page below are copied
# from editing.tex rather than chosen here:
#   A4, left/right 14 mm, top 15 mm, bottom 22 mm  -> a 182 mm text block, so a true
#   180 mm figure sits flush at 1:1, the same as in the paper.
#   10 pt on 12 pt leading, matching sn-jnl's 10bp/12bp \normalsize.
#   Numbered sections + a contents page, because a note long enough to need a PDF is
#   long enough to need navigation.
# --shift-heading-level-by=-1 is REQUIRED, not cosmetic. A Dendron note carries its title
# in frontmatter and starts the body at `##` (the date-stamped section convention in
# CLAUDE.md), so without the shift pandoc numbers the first real heading `0.1` under a
# phantom section zero. With it, the top level numbers 1, 2, 3 like the paper's sections.
# Do NOT change these to taste. If the paper's editing view changes, change these to
# match it; the point is that a note and the manuscript read as one document family.
#
# Usage (see .vscode/tasks.json): bib_tex_pdf.sh <file.md> <fileDirname> <basenameNoExt>

input_file="$1"
output_dir="$2"
output_filename="$3"
header_includes_path="${output_dir}/assets/publish/tex-templates/header-includes.tex"

echo "Edit notes/assets/publish/tex-templates/header-includes.tex for customizing spacing."

# --- preprocess: Unicode sanitize -> temp markdown (SVGs stay vector) ---
preprocessed="$(mktemp --suffix=.md)"
trap 'rm -f "${preprocessed}"' EXIT
python3 - "${input_file}" "${preprocessed}" <<'PYEOF'
import sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()

# sanitize non-ASCII glyphs the default xelatex font lacks (outside code fences)
repl = {
    "±": r"$\pm$", "×": r"$\times$", "÷": r"$\div$", "·": r"$\cdot$",
    "≈": r"$\approx$", "≃": r"$\simeq$", "≡": r"$\equiv$",
    "≤": r"$\le$", "≥": r"$\ge$", "≠": r"$\ne$",
    "→": r"$\rightarrow$", "↔": r"$\leftrightarrow$", "←": r"$\leftarrow$",
    "∝": r"$\propto$", "√": r"$\surd$", "∞": r"$\infty$",
    "°": r"$^\circ$", "′": r"$'$", "″": r"$''$", "…": r"\ldots{}",
    "—": "---", "–": "--", "⚠": "(!)", "✓": r"$\checkmark$", "✗": "x",
    "ρ": r"$\rho$", "α": r"$\alpha$", "β": r"$\beta$", "μ": r"$\mu$",
    "µ": r"$\mu$", "σ": r"$\sigma$", "Δ": r"$\Delta$", "γ": r"$\gamma$",
}
out, fence = [], False
for line in text.split("\n"):
    if line.lstrip().startswith("```"):
        fence = not fence
        out.append(line)
        continue
    if not fence:
        for k, v in repl.items():
            line = line.replace(k, v)
    out.append(line)
open(dst, "w", encoding="utf-8").write("\n".join(out))
PYEOF

cd ./notes && pandoc \
  --metadata link-citations=true \
  -s "${preprocessed}" \
  -o "${output_dir}/assets/pdf-output/${output_filename}.pdf" \
  --pdf-engine=xelatex \
  --citeproc \
  --bibliography assets/publish/bib/bib.bib \
  --metadata csl=assets/publish/bib/nature.csl \
  -V documentclass=article \
  -V papersize=a4 \
  -V fontsize=10pt \
  -V geometry:'a4paper, left=14mm, right=14mm, top=15mm, bottom=22mm' \
  --number-sections \
  --lua-filter=assets/publish/filters/dendron-links.lua \
  --lua-filter=assets/publish/filters/keep-tables.lua \
  --lua-filter=assets/publish/filters/section-status.lua \
  --shift-heading-level-by=-1 \
  --toc \
  --toc-depth=3 \
  --include-in-header="${header_includes_path}" \
  --strip-comments --dpi=600 && cd ..

output_file_path="${output_dir}/assets/pdf-output/${output_filename}.pdf"
echo "Output file: ${output_file_path}"
