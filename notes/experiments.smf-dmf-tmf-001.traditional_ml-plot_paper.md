---
id: 54oh60gyk9woslaw1gd7rfh
title: Traditional_ml Plot_paper
desc: ''
updated: 1783996988459
created: 1783996988459
---

## 2026.07.13 - One paper-facing composite, drawn from the same CSV as the SI tables

The classical-ML figure and its tables must not disagree, but the exploratory plot scripts each re-derived their numbers from a different cached W&B dump. This script draws every paper-facing panel from the single reconstructable summary CSV that also generates the SI tables, so the figure and Tables S3--S5 are guaranteed to be the same validation-selected configurations. It also carries both studies in one place: fitness (smf-dmf-tmf-001) and gene interaction (002-dmi-tmi).

- **Reads** `experiments/smf-dmf-tmf-001/results/traditional_ml_summary_with_std.csv` and `experiments/002-dmi-tmi/results/traditional_ml_summary_with_std.csv` -- both produced by [[experiments.smf-dmf-tmf-001.traditional_ml-summary_table]]. The two dataset-construction panels additionally read the cached one-hot splits under `DATA_ROOT`. No model is refit.
- **Writes 16 PNGs** (prefix `paper_tradml_`, 300 dpi) to `ASSET_IMAGES_DIR` (`notes/assets/images/`): 12 bar charts `paper_tradml_bar_{fitness,interaction}_{rf,svr}_{1e3,1e4,1e5}.png`, 2 line plots `paper_tradml_progression_{fitness,interaction}.png`, plus `paper_tradml_datadist_fitness.png` (label distribution per split) and `paper_tradml_genecoverage_fitness.png` (deletions per gene). Nothing is overwritten -- the old bar/line charts keep their names.
- **Bars:** one validation-selected best config per encoding; validation solid, test at `alpha=0.45`; black error bar = 5-fold CV s.d. on the validation score (only $10^3$/$10^4$ ran CV, $10^5$ is a single split). Encodings are colored by *family* -- identity / biological / hand-crafted / random -- so the finding ("random matches biological once wide enough; one-hot wins") is legible without reading 17 labels. **Progression:** max-over-encodings test Spearman vs $10^3$--$10^5$, one line per model (RF, SVR).
- **Which figure it feeds:** the SI classical-ML figure, `\label{fig:classical-ml}` (Fig S4) in the classical-ML Supplementary Note of `paper/nature-biotech/sections/backmatter.tex`. That float is still a `\figph` placeholder awaiting composition.
- **Predates the current figure standard** (CLAUDE.md "Figure & Plotting Standards"): free-form `figsize`, PNG with `bbox_inches="tight"`, top/right despined, and a green family color (`#729E5A`) the green-free palette no longer allows. The panels currently composed in `notes/assets/drawio/Fig7-Traditional-ML-justification-of-CGT.drawio.svg` come from the `_palette` true-size SVG scripts instead. Reconcile the two routes before the figure is final.
