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
# The pandoc `-F mermaid-filter` path is intentionally NOT used (its bundled
# puppeteer times out -- see CLAUDE.md); pre-render mermaid diagrams with
# mermaid_pdf.sh and reference the produced image instead.
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
  -V geometry:'top=2cm, bottom=1.5cm, left=2cm, right=2cm' \
  --include-in-header="${header_includes_path}" \
  --strip-comments --dpi=600 && cd ..

output_file_path="${output_dir}/assets/pdf-output/${output_filename}.pdf"
echo "Output file: ${output_file_path}"
