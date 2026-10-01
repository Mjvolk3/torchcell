---
id: yd96cbslcubmbv2djy1g3yk
title: Test_019_train_cgt_multitask
desc: ''
updated: 1790872921905
created: 1790872921905
---

## 2026.10.01 - 019 batch sizing and row masks (issue #567)

New file. Loads `experiments/019-simb-multimodal/scripts/train_cgt_multitask.py` by path with `dotenv.load_dotenv` stubbed and calls `MultitaskCGTTask._batch_size` / `_extract_targets_and_masks` unbound on a namespace. Three genotypes, the last a wild type with no perturbation: size 3; fitness and 2-feature morphology masks are [True, False, True]; targets are [0.5, 0, -0.25] and [[1, 2], [0, 0], [3, 4]]. Under the old `max + 1` sizing the mask had 2 rows against a 3-row target.
