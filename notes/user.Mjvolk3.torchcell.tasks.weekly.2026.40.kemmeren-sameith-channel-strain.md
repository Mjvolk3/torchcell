---
id: ipyq9xr2n0nvd3zq9eblf71
title: kemmeren-sameith-channel-strain
desc: ''
updated: 1790624142216
created: 1790624142216
---

## 2026.09.28

- [x] Re-checked the two Phase 8 data findings before touching anything (user: "re read the papers and geo"): Kemmeren's channel swap is real by GEO's `label_ch*`/`source_name_ch*` on all 3061 arrays and by the deleted gene's own probes (702 of 705 arrays agree with GEO, 10 with the loader's title rule), but the served log2 sign is right on 1448 of 1484 records because of the negation at the old line 2152 (dev LMDB self ratio negative on 97.0%, median -2.48), which is why the cross-study correlation and the trained models looked fine; 35 records are flipped or cancelled (the "-c", "-d", "[hs199x]" and `yil014c-a` arrays), and the SE and linear fields are off on all. Sameith's four `MATa` pairs are BY4741 by the paper, the SI and GEO alike; all 72 are served as BY4742. [[torchcell.datasets.scerevisiae.kemmeren2014]], [[torchcell.datasets.scerevisiae.sameith2015]]
- [x] Fix on this branch, PR left open by decision ("leave solve on wt"): Kemmeren reads the channels from GEO metadata (`_channel_columns`), takes log2 within each array (the within-array SE is smaller on 76% of entries, median ratio 0.34), and checks itself on the deleted genes' probes at build time (2538 of 2560 arrays depleted, 0.991, on the real pickles); Sameith maps `MATa` to BY4741. Tests flipped: [[tests.torchcell.datasets.scerevisiae.test_kemmeren2014_synthetic]], [[tests.torchcell.datasets.scerevisiae.test_sameith2015_synthetic]]; ruff, mypy (CI form), quality lint, paired check green; 22 tests pass.
- [ ] Before the next KG build (issue `#459`): rebuild the two dev LMDBs from the fixed loaders and re-admit them, by the full rebuild or by a dataset-replacement operation that does not exist yet (retire the dataset's nodes, re-admit, manifest event, release bump). Every store step under slurm.
