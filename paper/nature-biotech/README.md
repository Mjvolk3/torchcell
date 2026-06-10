# Nature Biotechnology manuscript (TorchCell CGT)

LaTeX skeleton for the TorchCell Cell Graph Transformer paper, using the official
Springer Nature template (`sn-jnl.cls`) with the Nature Portfolio reference style
(`sn-nature`). The structure mirrors `notes/paper.nature-biotech-cgt-outline.md`.

## Files

Two maintained versions share ONE body so edits never drift:

- `content.tex` -- **the manuscript body you edit** (title, abstract, sections,
  figures, bibliography).
- `preamble.tex` -- shared `\usepackage` block (edit packages once).
- `submission.tex` -- **single-column, official Nature Portfolio format. Upload
  this.** Wrapper: `\documentclass[pdflatex,sn-nature,Numbered]{sn-jnl}` + `\input`s.
- `editing.tex` -- **single-column with print-approximate margins** (~170 mm text
  block), readable for drafting while sizing figures realistically. NOT the
  submission format. In this layout `\textwidth` ~ full-page print figure and
  ~`0.49\textwidth` ~ one print column; add `,iicol` to its documentclass to see
  the true two-column print typeset instead.
- `figure-proto.tex` -- **figure-prototyping sandbox at true Nature print scale**
  (`iicol` geometry = real 160 mm x 216 mm text block). Lay out multi-panel
  figures here at 1:1 with print: `\textwidth` = full figure width (~160 mm),
  `\linewidth` in a 0.47-width panel = one print column (~76 mm), `[p]` float =
  full page. Decoupled from `content.tex`; port the final `figure*` into the body.
- `sn-jnl.cls` / `sn-nature.bst` -- Springer Nature class (v3.1, Dec 2024) + Nature style.
- `sn-article.tex` -- upstream example/manual (reference only; not compiled).
- `references.bib` -- seeded bibliography (keep <= 50 refs).
- `figures/` -- drop exported vector figures here (create as needed).

Build: `tectonic -X compile submission.tex` and/or `tectonic -X compile editing.tex`.
`pdflatex` option = compiles under pdflatex/xelatex/Tectonic; `sn-nature` = Nature
reference style; `Numbered` = superscript numbered citations. Big main figures use
`figure*` (full width / spans both columns) so they don't overflow a column in
`editing.tex`.

## Get this into Overleaf (three options)

**Option A -- Official Overleaf template (recommended start).** Open the Springer
Nature template from the Overleaf gallery
(<https://www.overleaf.com/latex/templates/springer-nature-latex-template/myxmhdsbzkyd>),
"Open as Template", then paste `submission.tex` + `references.bib` over the example and
set the documentclass line above. Always the most current class files.

**Option B -- Upload this folder.** Zip `paper/nature-biotech/` and upload to
Overleaf (New Project -> Upload Project). Self-contained; compiles as-is.

**Option C -- Git sync (keeps repo <-> Overleaf in sync).**
- Overleaf Pro git remote: each project has `git clone https://git.overleaf.com/<id>`;
  push/pull this folder's contents to it.
- Or GitHub sync: link the Overleaf project to a GitHub repo (Menu -> GitHub) and
  point it at this path (or split the manuscript into its own repo).

Figures live under `notes/assets/images/...` in the main repo; export final
vector PDFs into `figures/` here (or copy into the Overleaf project) for inclusion.

## Local builds with Tectonic (offline / CI alternative to Overleaf)

Tectonic is a self-contained XeTeX engine that auto-downloads packages. Our
`pdflatex` documentclass option enables the xelatex-compatible path sn-jnl needs,
and Tectonic runs classic BibTeX for `sn-nature.bst`, so this skeleton should
build with:

```bash
conda install -c conda-forge tectonic   # or: cargo install tectonic
tectonic -X compile submission.tex            # V2 CLI; or: tectonic submission.tex
```

Not yet verified on this machine (tectonic was not installed when the skeleton
was created) -- run the above once to confirm before relying on it for CI.

## "The margins look huge" -- this is correct

`sn-jnl` is a **content-first manuscript template** (Springer Nature's words), and
`sn-jnl` + `sn-nature` is the official style for Nature Portfolio submissions
(incl. Nature Biotechnology). The wide single-column page is intentional -- it is
the *submission/review* format, NOT the published two-column Nature layout, which
Springer typesets only after acceptance. The roomy margins leave space for the
`[referee]` (double-spaced) and `[lineno]` (margin line numbers) review options.
`[iicol]` gives a two-column preview but is rare and not the submission format.
(Confirmed 2026-06-10 via support.springernature.com + the Springer Nature
Overleaf template.)

## Nature Portfolio: embed references, do not upload a .bib

Nature Portfolio wants references **inside the manuscript file**, not a separate
bibliography upload. Workflow: keep `references.bib` for drafting, compile once to
generate `main.bbl`, then before submission paste the `.bbl` contents in place of
`\bibliography{references}` (and remove the `.bib` from the upload). Also drop the
preview-only `\nocite{*}`.

## Pre-submission checklist (from the outline)

- Abstract <= 150 words; main text <= 3,000 words; <= 6 display items; <= 50 refs.
- Reconcile Fig. 3 error-bar provenance (SE/SEM vs SD).
- Greek (tau, rho) renders via Symbol font per nbt guidance.
- Confirm live limits on nature.com/nbt before submitting.
