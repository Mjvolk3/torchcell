---
id: vqqcmcquki2jgovdim5nfxj
title: '38'
desc: ''
updated: 1789524342979
created: 1789524342979
---

## 2026.09.15

- [x] `notes-tex/publishing-system`: a typeset note on what `make` builds for the manuscript and for a typeset note, with two mermaid flow diagrams, the target tables, the provenance rules for figures and tables, the two bibliography tiers, and how a build becomes a hashed Zotero version [[publishing-system]]
- [x] 025 additive-baselines brought to the module-document standard: a script-generated W&B run registry with links for all eight runs, an arms section naming what each configuration turns on and the missing learnable-table control, the forward-pass diagram, and the first arm Q test score folded in with a paired bootstrap and a prespecified decision table. GH 1640's epoch 7 checkpoint scores 0.139 on the 46 held-out screens against the additive ridge's 0.185, gap -0.046 [-0.056, -0.036], level with B5; D1 and D2 fail toward the null, D3 (three seeds) not run [[experiments.025-solid-growth.scripts.wandb_run_index_025]], [[experiments.025-solid-growth.scripts.paired_bootstrap_025]], [[experiments.025-solid-growth.scripts.additive_baselines_025_panels]]

## 2026.09.16

- [x] nightly literature sync walks a list of personal Zotero trees (`torchcell` + `thesis`), each mirrored + MinerU'd into the same tc-lit mirror; a root missing from Zotero is reported without hiding the others; cron line updated [[scripts.lit_sync]]
