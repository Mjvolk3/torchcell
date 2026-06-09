---
id: xe602ztxryw81lzvgu6pi8e
title: '23'
desc: ''
updated: 1780787603754
created: 1780623011872
---

## 2026.06.04

- [x] Audited error-bar provenance for the trigenic τ model-comparison bar chart and documented which whiskers are real vs placeholders (Yeast9 = legit zero; DCell/DANGO = computed SEM from 3 replicates; TorchCell = hardcoded reported SE over a reconstructed array) [[2026.06.04 - Error-Bar Provenance|experiments.010-kuzmin-tmi.scripts.trigenic_tau_model_comparison#20260604---error-bar-provenance-which-whiskers-are-real]]
- [x] Moved the trigenic τ data/provenance note out of `scratch.*` into the script's dendron note and embedded the plot [[trigenic τ model comparison|experiments.010-kuzmin-tmi.scripts.trigenic_tau_model_comparison]]
- [x] Moved the SIMB 2026 conference abstract out of `scratch.*` and added a reciprocal error-bar caveat (flag SEM vs reported SE before submission) [[SIMB 2026 abstract|conference.simb-2026.abstract]]
- [x] Added `iBioFoundry-AI` to the local `torchcell.code-workspace` folders
- [x] Investigated `YLR312C-B` (SGD merged ORF, alias of SPH1/`YLR313C`) — verified **no SPH1 overlap** (~108 bp gap; KO removes SPH1 3′UTR, not its ORF), 0/10 in-panel digenics but interaction-active genome-wide; **recommend swapping** the panel-12 node [[Inference Dataset 3|experiments.010-kuzmin-tmi.inference-dataset-3]]

## 2026.06.06

- [x] Repo cleanup after the 12-gene-panel scramble
- [x] Graduated inference-dataset-2 scratch notes → [[Inference Dataset 2|experiments.010-kuzmin-tmi.inference-dataset-2]] (iterative-fitness goal, statistical framework, why SMF>1.10 was too restrictive)
- [x] Graduated inference-dataset-3 scratch notes → [[Inference Dataset 3|experiments.010-kuzmin-tmi.inference-dataset-3]] (relaxed thresholds + Jonckheere–Terpstra test, final 12-gene panel, `max()`-vs-`mean()` aggregation bug, YLR312C-B swap)
- [x] Fixed plot layout collisions (annotations over lines, legends over text, bar labels off-figure) in `select_12_and_24_genes_top_triples_inference_2.py`, `select_12_and_24_genes_top_triples_inference_3.py`, `generate_triple_combinations_inference_2.py`, `generate_triple_combinations_inference_3.py`, `12_panel_inference_3_fitness_comparison.py` — all 6 figures regenerated + verified
- [x] Added `--plot-only` mode to the 4 heavy scripts (select_* and generate_*): redraws figures from saved results/`generation_summary.txt` in seconds–minutes instead of re-running the ~2 h selection / ~8 h generation
- [x] Relocated stray files: `slurm-797.out` → `experiments/010-kuzmin-tmi/slurm/output/`; ScienceDirect iIsor850 supplements → `DATA_ROOT/data/external/iIsor850-mmc/`; restored gitignored `./torchcell-scratch` symlink to `DATA_ROOT`
- [x] Pushed `.claude/` skill suite + `settings.json` to `origin/main` for cross-machine use
