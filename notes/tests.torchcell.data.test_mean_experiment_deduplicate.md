---
id: ku9n3uzirp8ckokl3pr0m4t
title: Test_mean_experiment_deduplicate
desc: ''
updated: 1790415101812
created: 1790415101812
---

## 2026.09.26 - Mean deduplication on three in-memory records

No LMDB: the deduplicator is built with a tmp root and only its grouping and merge methods run. Fitness 1, 2, 6 with stds 0.1, 0.2, 0.2 reduce to 3.0 and sqrt(0.03); the mean genotype is a `MeanDeletionPerturbation` with `num_duplicates` 3; dataset names join sorted as `a+b+c`; the reference pools the same way. Grouping ignores gene order and separates experiment types; the streaming `duplicate_key` on `model_dump()` output equals the pydantic `duplicate_check` key. The three dict helpers (`_mean_float_dict`, `_rms_pool_float_dict`, `_sum_int_dict`) have closed-form checks, and the vector family runs end to end through `create_deduplicate_entry` on two `MetaboliteExperiment` duplicates (levels 1 and 3 -> 2.0, se pooled to 0.3535533906, replicates 2 + 3 = 5, the target-id map taken from the first record that carries one). Gene-interaction duplicates: 0.1 and 0.3 -> 0.2 with `graph_level` kept, and the merged p-value is a one-sample t-test of the scores against zero, 2 * t.sf(2, df = 1) = 0.2951672353; the records' own p-values never enter it (`_compute_p_value_for_mean` reads them only for a length check, so a record with p = None makes the merge raise). `Genotype.perturbations` does not admit the abstract `DeletionPerturbation`, so the fixtures use `KanMxDeletionPerturbation`. Phase 3 of [[plan.test-suite-buildout.2026.09.25]]; the vector, p-value and None-p tests came from the Phase 3 audit.
