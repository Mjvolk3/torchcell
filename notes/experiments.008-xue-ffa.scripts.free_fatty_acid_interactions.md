---
id: 5ruob210tfnz91ca38qb9hg
title: Free_fatty_acid_interactions
desc: ''
updated: 1786496512755
created: 1786496512755
---

## 2026.08.11 - Portable Paths So This Can Serve as the 008 Loader Layer

This is the root analysis script of 008: it loads the Xue 2025 raw titer workbook, computes
per-FFA means and SDs from the replicate columns, normalizes every strain to the `+ve Ctrl`
base production strain, parses the letter-coded genotypes into TF gene lists, and fits the
additive and multiplicative epistasis models.

Three absolute `/Users/michaelvolk/...` literals -- the raw workbook, the results directory,
and the matplotlib style file -- meant it only ran on the Mac it was authored on. They now
resolve from `DATA_ROOT`, `EXPERIMENT_ROOT`, and the installed `torchcell` package
respectively. The package-relative style lookup matters specifically because this work runs
from a worktree: a checkout-specific literal would silently reach into the primary tree.

The reason this file in particular had to become portable is that it is now an imported
**library**, not just a script. Both
[[experiments.008-xue-ffa.scripts.ffa_total_titer_trajectories]] and
[[experiments.008-xue-ffa.scripts.ffa_epistatic_path_panels]] import `load_ffa_data`,
`normalize_by_reference`, and `parse_genotype` from here rather than re-deriving the
normalization, so the base-strain reference is defined in exactly one place and the
trajectory figures cannot drift from the interaction analysis.

### Known trap: the Abbreviations sheet header

`load_ffa_data` reads the `Abbreviations` sheet with a default header row, which consumes
FKH1 as the header and silently yields only **9** of the 10 TFs -- strains containing `F`
then carry the bare letter as their gene name instead of `FKH1`. Callers that need all ten
must re-read the sheet with `header=None`; that is what
`ffa_total_titer_trajectories.read_abbreviations` exists for, and it asserts a count of 10
so the failure can never again be silent.
