---
id: s3liobp24uaoz0u70cthr8n
title: Scaling_law_forms
desc: ''
updated: 1791533044906
created: 1791533044906
---


## 2026.10.09 - The two scaling-law forms, drawn to be read

Script: `experiments/041-scaling-laws/scripts/scaling_law_forms.py`. Nothing here is measured on a torchcell model; the panels draw the Kaplan 2020 (arXiv:2001.08361) and Hoffmann 2022 (arXiv:2203.15556) forms at their published constants, plus the Besiroglu 2024 refit (arXiv:2404.10102). Panel i is a synthetic fitting exercise on runs drawn from the Hoffmann form. The typeset explainer is `notes-tex/modeling/scaling-laws/` ([[scaling-laws]]), which reads this figure through `make plots` and the constants table the script writes to `tables/t1-constants.tex`.

Run from the repo root (the GIFs take a few minutes; `--no-gifs` skips them):

```bash
python experiments/041-scaling-laws/scripts/scaling_law_forms.py
```

### Figure 1 (3 x 3)

![](./assets/images/041-scaling-laws/scaling_law_forms.svg)

- **a, b** Kaplan single-axis laws, one exponent published and two others; on log-log axes the exponent is the slope.
- **c** The floor: `E + A N^-alpha` bends toward `E` while `L - E` stays straight. Curvature at the large end is the sign that `E` is near.
- **d** The Chinchilla surface `L(N, D)` with iso-compute diagonals `6ND = C` and the compute-optimal frontier.
- **e, f** Cuts at fixed `D` and fixed `N`: each plateaus at the term the fixed axis leaves behind.
- **g** IsoFLOP curves, a U per budget; the minima trace the frontier.
- **h** `N*` and `D*` against `C` under two fits of the same runs (`a = 0.45` vs `0.51`).
- **i** Fit on fifteen small synthetic runs, hold out the two largest, extrapolate with bootstrap bands: the pure power law (`alpha = 0.15`) misses the held-out runs, the floored fit (`alpha = 0.35`, `E = 1.71` against a generating 0.34, 1.69) lands on them. Numbers in `experiments/041-scaling-laws/results/scaling_law_forms_summary.json`.

### GIFs

Each is one sweep of one quantity with everything else pinned. The title of every frame carries the equation, the constants and their source, and the swept value with what it implies, so a frame reads on its own. Legends name every line, dotted ones included, and hold still from frame to frame.

The frames are typeset by real LaTeX (Computer Modern, `amsmath`) rather than matplotlib's mathtext, at 200 dpi on a 150 mm canvas. matplotlib's Agg backend cannot rasterize `usetex` text without `dvipng`, which GilaHyper lacks, so each frame is written through the PDF backend (which reads the DVI itself) and rasterized with `pdftoppm`. The system TeX Live also omits `type1cm.sty`, which matplotlib's `usetex` preamble loads; it is installed in the user tree at `~/texmf/tex/latex/type1cm/type1cm.sty` (built from the CTAN `type1cm.ins` on 2026.10.09). The static 3x3 figure stays in Arial mathtext per the repo figure standard.

**Data sweep.** `L` vs `N` while `D` rises from 10^7.5 to 10^12.5: the plateau `E + B D^-beta` drops toward `E`, and the N-curve runs into it later and later.

![](./assets/images/041-scaling-laws/scaling_gif_data_sweep.gif)

**Parameter sweep.** `L` vs `D` while `N` rises: the mirror image, plateau `E + A N^-alpha`.

![](./assets/images/041-scaling-laws/scaling_gif_params_sweep.gif)

**Compute sweep.** The IsoFLOP U while `C` rises from 10^18 to 10^23; the minimum traces the frontier, about 0.45 decades of `N*` per decade of `C`.

![](./assets/images/041-scaling-laws/scaling_gif_compute_sweep.gif)

**Exponent sweep.** Kaplan's pure power law with `alpha_N` moving by plus or minus 0.02 about the published 0.076. Every curve is pinned through the same loss at `N = 10^7.5`, the center of the fitted range, which is why the sweep pivots there: the five runs the fit sees stay within a few percent of every curve, which 2% replicate noise cannot rule out, while the forecast at `N = 10^12` moves by about a quarter each way. This is why an interval on the exponent is the result, not the point.

![](./assets/images/041-scaling-laws/scaling_gif_exponent_sweep.gif)

**Floor sweep.** `E` rises from 0 to 2.5 under a fixed `A N^-alpha`: on log-log axes the curve bends where `A N^-alpha = E`, and `L - E` is the same straight line in every frame. This is the diagnostic behind panel c.

![](./assets/images/041-scaling-laws/scaling_gif_floor_sweep.gif)

**Surface sweep.** The joint surface `L(N, D)` with one budget line `6ND = C` sliding across it. The optimum is where the line touches its lowest contour, and the optima lie on the frontier.

![](./assets/images/041-scaling-laws/scaling_gif_surface_sweep.gif)

**Bootstrap.** One resample per frame: the fifteen fitted runs are redrawn with replacement (ring size = times drawn), the floored form is refit, and the refits pile up into the band panel i shows at once. The 90% interval on `alpha` in the title settles as the resamples accumulate.

![](./assets/images/041-scaling-laws/scaling_gif_bootstrap.gif)

**Allocation sweep.** `beta` moves from 0.2 to 0.4 with `alpha` fixed at 0.34: `a = beta / (alpha + beta)` and `b = alpha / (alpha + beta)` move with it, and the `N*` and `D*` lines tilt against the Hoffmann allocation (dashed). A steeper data term tilts a budget toward parameters.

![](./assets/images/041-scaling-laws/scaling_gif_allocation_sweep.gif)

### Counting D: instances, not dimensions

`D` is the number of training examples the optimizer consumed, in the unit the loss averages over (tokens for a language model, records here), times nothing: epochs are a fixed hyperparameter and repeated records count for less than fresh ones (Muennighoff 2023, arXiv:2305.16264). The dimension of a record (features, genes in the genome, sequence length) is not `D`; it changes compute per record and the input width of the model. Full argument in section 3 of the notes-tex document.
