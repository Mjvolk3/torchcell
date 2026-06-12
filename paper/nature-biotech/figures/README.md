# figures/

Final figure assets for the manuscript. Compose in **draw.io** (diagrams.net;
the "Draw.io Integration" VS Code extension edits `.drawio` files inline).

## Workflow

1. Create `figN_name.drawio` here (e.g. `fig1_overview.drawio`).
2. Lay out panels to Nature print size: **180 mm** full-width or **88 mm** single
   column, max **170 mm** tall (see the outline's figure-prep guidelines). Use
   `figure-proto.tex` to check the full-page fit.
3. Export to vector PDF: draw.io `File -> Export as -> PDF` (crop to drawing,
   uncheck page background) -> save as `figN_name.pdf` here.
4. In `content.tex`, swap the `\figph{...}{...}` placeholder for
   `\includegraphics[width=\textwidth]{figures/figN_name.pdf}` (or `\linewidth`).

## draw.io sizing (design at TRUE size)

`\includegraphics[width=\textwidth]` rescales the exported PDF to 180 mm
regardless of the draw.io canvas, so the only risk is fonts scaling out of the
5-7 pt range. Avoid it by drawing at final size. draw.io default = 100 units/inch
(1 mm = 3.94 units):

| Final | mm | draw.io units |
|-------|------|---------------|
| Full width | 180 mm | ~709 units wide |
| Single column | 88 mm | ~347 units wide |
| Max height | 170 mm | ~669 units tall |

Draw a full-width figure to ~709 units wide, fonts 6-7 pt, then export
crop-to-drawing -> PDF and include at `width=\textwidth` (scale ~1, fonts intact).
Drawing much wider than 709 shrinks your text below 5 pt on import. Aspect ratio
of the drawing sets the final height (must stay <= 170 mm at 180 mm wide).

Figure float *placement* on the page is automatic (LaTeX floats, `[t]/[b]/[p]`);
only panel layout inside the figure is set in draw.io.

## Expected files (placeholders live in content.tex until these exist)

- `fig1_overview.pdf` -- TorchCell overview (R1)
- `fig2_cgt_architecture.pdf` -- CGT architecture (R2)
- `fig3_ggi.pdf` -- trigenic GGI state of the art (R3)
- `fig4_multitask.pdf` -- multitask generalization (R4)
- `fig5_design.pdf` -- strain design + DBTL (R5)
- `figS1_classical_ml.pdf` ... `figS6_inference.pdf` -- Supplementary

Keep panel text sans-serif, 5-7 pt at final size; line weights >= 0.25 pt.
