---
id: vqqcmcquki2jgovdim5nfxj
title: '38'
desc: ''
updated: 1789524342979
created: 1789524342979
---

## 2026.09.14

- [x] `genomes-tier` worktree: a genomes tier with pydantic manifests and a registry as the sole path authority [[user.Mjvolk3.torchcell.tasks.weekly.2026.38.genomes-tier]]
- [x] `live-rebuild` worktree: the pre-rebuild dataset sweep, 36 stale stores rebuilt, and the full live rebuild that swapped in 51 served datasets on 09.17 [[user.Mjvolk3.torchcell.tasks.weekly.2026.38.live-rebuild]]

## 2026.09.15

- [x] `notes-tex/publishing-system`: a typeset note on what `make` builds for the manuscript and for a typeset note, with two mermaid flow diagrams, the target tables, the provenance rules for figures and tables, the two bibliography tiers, and how a build becomes a hashed Zotero version [[publishing-system]]
- [x] 025 additive-baselines brought to the module-document standard: a script-generated W&B run registry with links for all eight runs, an arms section naming what each configuration turns on and the missing learnable-table control, the forward-pass diagram, and the first arm Q test score folded in with a paired bootstrap and a prespecified decision table. GH 1640's epoch 7 checkpoint scores 0.139 on the 46 held-out screens against the additive ridge's 0.185, gap -0.046 [-0.056, -0.036], level with B5; D1 and D2 fail toward the null, D3 (three seeds) not run [[experiments.025-solid-growth.scripts.wandb_run_index_025]], [[experiments.025-solid-growth.scripts.paired_bootstrap_025]], [[experiments.025-solid-growth.scripts.additive_baselines_025_panels]]
- [x] `cooper2010-amino-acid-metabolome` worktree: Cooper 2010 amino-acid pools landed as the 51st dataset [[user.Mjvolk3.torchcell.tasks.weekly.2026.38.cooper2010-amino-acid-metabolome]]

## 2026.09.16

- [x] nightly literature sync walks a list of personal Zotero trees (`torchcell` + `thesis`), each mirrored + MinerU'd into the same tc-lit mirror; a root missing from Zotero is reported without hiding the others; cron line updated [[scripts.lit_sync]]
- [x] 008 figures reworked so each epistasis sign reads at a glance: one ring per sign, positive interactions in blue, a figure of what each null model expects, and the measured chemistry on a metabolic map as Fig. 5a [[experiments.008-xue-ffa.scripts.epistasis_model_intuition_panels]] [[experiments.008-xue-ffa.scripts.ffa_ipath_map]] [[experiments.008-xue-ffa.scripts.ffa_network_overlay_drawio]]

## 2026.09.17

- [x] 008 Fig. 5a redrawn from KEGG's yeast global map (KGML) with Yeast9 overlaid, because iPath3's map is thinner, and three figure standards recorded from the font audit [[experiments.008-xue-ffa.scripts.ffa_kegg_map]] [[paper.nature-biotech.style-guide]]

## 2026.09.18

- [x] `neo4j-browser-style` worktree: a generated Browser stylesheet seeded into the image, and `torchcell` as the served store's default database [[user.Mjvolk3.torchcell.tasks.weekly.2026.38.neo4j-browser-style]]
- [x] 008 review rounds 6 and 7 plus a fact-check: a flux model of the regulator-vs-enzyme question, a perturbation reach analysis, and single-mutant fitness of the ten factors as SI Note 5 [[experiments.008-xue-ffa.scripts.regulator_vs_enzyme_epistasis_model]] [[experiments.008-xue-ffa.scripts.perturbation_reach_analysis]] [[experiments.008-xue-ffa.scripts.tf_single_mutant_fitness]]

## 2026.09.19

- [x] `kg-releases-ops` worktree: versioned knowledge-graph releases with content hashes, a `KgRelease` node and a `make ops` panel [[user.Mjvolk3.torchcell.tasks.weekly.2026.38.kg-releases-ops]]
- [x] `kg-release-archive-fix` worktree: release 1.0 backed up, archived and verified on Taiga [[user.Mjvolk3.torchcell.tasks.weekly.2026.38.kg-release-archive-fix]]
- [x] `ci-green` and `ci-green-2` worktrees: CI red since mid-July traced to import-time `DATA_ROOT` and a missing `rdkit`, experiments 016 to 026 under the lint gate [[user.Mjvolk3.torchcell.tasks.weekly.2026.38.ci-green]] [[user.Mjvolk3.torchcell.tasks.weekly.2026.38.ci-green-2]]

## 2026.09.20

- [x] `datasets-figure-51` worktree: the instances-vs-signal scatter regenerated at 51 datasets [[user.Mjvolk3.torchcell.tasks.weekly.2026.38.datasets-figure-51]]
- [x] Kuzmin 2018/2020 dmf loaders now ingest the double-mutant query strain fitness (172 and 201 records), the one term the published trigenic score needs and both loaders dropped [[torchcell.datasets.scerevisiae.kuzmin2018]] [[torchcell.datasets.scerevisiae.kuzmin2020]]
- [x] Read-time `LabelPolicy`: a build keeps every measurement and each run chooses which label to train on, hashed with the split and seed [[torchcell.data.label_policy]] [[torchcell.data.label_table]]
- [x] 029 evaluates the published asymmetric trigenic identity, now that array/query roles resolve from strain ids; the form matters on Kuzmin 2020 (0.325 against 0.278 symmetric) and not on 2018 (0.511 against 0.504) [[experiments.029-solid-growth-ko.scripts.closure_recompute_asymmetric]]
