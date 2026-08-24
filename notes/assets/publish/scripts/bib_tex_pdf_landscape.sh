#!/bin/bash
# Whole-note -> LANDSCAPE PDF (pandoc + xelatex + citeproc, Nature CSL).
#
# Same pipeline as bib_tex_pdf.sh, only the geometry differs. Use it for a note whose
# widest table does not fit the portrait text block -- a bench build sheet, a wide
# comparison matrix -- where portrait wraps every cell mid-token and doubles the page
# count.
#
# Shares bib_tex_pdf.sh's two behaviours:
#   * Unicode sanitize to a TEMP markdown (the original note is never modified), so
#     glyphs the default xelatex font lacks do not silently drop. Fenced code blocks
#     are skipped so shell snippets survive intact.
#   * SVG figures embed as VECTOR via rsvg-convert; .svg is NOT swapped to .png.
#
# The pandoc `-F mermaid-filter` path is intentionally NOT used (its bundled puppeteer
# times out -- see CLAUDE.md); pre-render mermaid diagrams with mermaid_pdf.sh and
# reference the produced image instead.
#
# Usage: bib_tex_pdf_landscape.sh <file.md> <fileDirname> <basenameNoExt>

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
  -V geometry:'landscape,top=2cm, bottom=1.5cm, left=2cm, right=2cm' \
  --include-in-header="${header_includes_path}" \
  --strip-comments --dpi=600 && cd ..

output_file_path="${output_dir}/assets/pdf-output/${output_filename}.pdf"
echo "Output file: ${output_file_path}"
