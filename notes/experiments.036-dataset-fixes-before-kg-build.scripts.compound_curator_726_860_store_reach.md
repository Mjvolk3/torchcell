---
id: ql07sgb12dchofs3vaufd72
title: Compound_curator_726_860_store_reach
desc: ''
updated: 1791698131863
created: 1791698131863
---

## 2026.10.11 - Which dev stores the #726 / #860 rows reach

Scans every mapped dev store (`build_adapter_map(include_private=True)`, yeast included)
with the gap inventory's readers (the `interned` env, plus records for titer datasets) for
compounds named by a curated label, and compares each with what the pinned table returns
now. A store is stale when a held compound's name or InChIKey differs from the table, or
its InChIKey gap now closes. `--classes-file` writes the stale classes as the
`gilahyper_build_dataset_lmdbs_array.slurm` CLASSES_FILE.

Before the rebuild (`results/compound_curator_726_860_store_reach_before.json`): 40 stale
stores of 43 that hold a curated label, 0 unreadable; the three fresh ones hold only labels
whose rows add no identifier (casamino acids in the two Campos 2018 stores, pinene in Niu
2019). After the rebuild (slurm arrays 3809 and 3836, at the branch's pre-rebase
`cb310e339`; `results/compound_curator_726_860_store_reach_after.json`): 0 stale stores of
the 42 that hold a curated label, 0 unreadable. MetabolomeSchastnaya2021Dataset left the
set: its rebuilt store names the compound `fructose`, the existing row's name, which is
not one of the pass's input labels (`D-fructose` was).
