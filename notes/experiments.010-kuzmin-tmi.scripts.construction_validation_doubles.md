---
id: sz4ska0lk5agzfrct0qnyig
title: Construction_validation_doubles
desc: ''
updated: 1784597538231
created: 1784579566973
---

## 2026.07.20 - Doubles for BOTH Triple Reconstruction and Assay Validation

Script: `experiments/010-kuzmin-tmi/scripts/construction_validation_doubles.py`
Data:   `experiments/010-kuzmin-tmi/results/construction_validation_doubles.csv`

## 2026.08.11 - Moved to LaTeX

**This note now lives as a typeset document**, because it had outgrown a markdown render --
it wants a table of contents, real cross-references, and its figures placed at true print
size:

- **Source:** `paper/notes/experiments.010-kuzmin-tmi.scripts.construction_validation_doubles.tex`
  (sections under `paper/notes/sections/construction-validation-doubles/`)
- **PDF:** `paper/notes/experiments.010-kuzmin-tmi.scripts.construction_validation_doubles.pdf`
- **Build:** `make -C paper/notes`

It uses the manuscript's editing styling (`sn-jnl`, the `editing.pdf` page geometry), inherited
by symlink and `\input` from `paper/nature-biotech/` rather than copied -- see
`paper/notes/README.md`.

The prose is **not duplicated here**; this stub is the Dendron entry point so `[[...]]` links
and search still resolve. Edit the `.tex`, not this file.

### What the document covers

1. Why the pure set-cover was the wrong objective
2. Two tiers, unioned (13 doubles) -- coverage, validation, and the one novel candidate
3. Single mutants to inoculate (10)
4. There are only 3 significant interactions in the whole panel
5. Cross-dataset check -- Costanzo vs Kuzmin DMF, plus the strain-level ambiguity *within*
   Costanzo (YJR060W `sn154` 0.5900 vs `dma2646` 0.9230)
6. Novel construction candidate -- YPL046C+YPL081W (ELC1+RPS9A)
7. Supplementary table -- all 45 within-10 doubles
8. What was actually built -- 13 of 14; `YKL033W-A x YJR060W` (CBF1) failed to construct

Related: [[experiments.010-kuzmin-tmi.scripts.topk_triples_from_constructed_10]],
[[experiments.010-kuzmin-tmi.scripts.optimized_doubles_setcover_constructed_10]],
[[experiments.010-kuzmin-tmi.scripts.constructed_10_dmf_reference]],
[[experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay]].
