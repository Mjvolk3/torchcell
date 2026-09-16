---
id: terpu7lb4xvxmpncrqgiib1
title: Archive_essentiality_dumps
desc: ''
updated: 1789536669978
created: 1789536669978
---

## 2026.09.16 - Why this script exists: the essentiality dumps collided

`experiments/027-betaxanthin-metabolic/scripts/archive_essentiality_dumps.py` gives every essentiality cell's per-gene prediction dump the name of its result file, which is what `train_bx.py` writes since 2026-09-16.

Until then the dump stem was `<arm>_essentiality_s<seed>.json`, with no architecture in the name. The wave-11 size ladder ran widths 12, 24, 54 and 90 on the pooled arm at seeds 2700 to 2702, so every width wrote the same three files and the last cell to finish won. Measured when the FCL comparison was rerun: `pooled_essentiality_s2700.json` held the width-12 cell (`cell.hidden` 12, written 2026-09-15 23:24) while `fcl_comparison.py` read it as the wave-9 width-90 arm, and its across-seed range moved from 0.578 to 0.793 to 0.786 to 0.801 without any cell having been rerun.

What the archive found (`results/essentiality_dump_archive.json`):

- archived: the three width-12 pooled cells and the three wave-9 flux-biomass cells, whose legacy files still held their own cell.
- lost: the three wave-9 width-90 pooled cells and the three width-24 pooled cells. Their result files keep every metric, curve and the trainer's own pinned AUROC; only the per-gene predictions are gone, so those rungs are rerun rather than reconstructed.

The trainer's assert is the durable fix: `dump_predictions=True` now requires `dump_stem`, the result stem, and `test.dump_path` in every result names the file. `fcl_comparison.py` resolves a dump only through the result that produced it and checks the dump's `cell` against the result's on every identity field, then checks that the dump reproduces the result's own pinned AUROC to 1e-6.
