---
id: 7z0afcrx6337o60q4qmzlya
title: Test_label_table
desc: ''
updated: 1790534914420
created: 1790534914420
---

## 2026.09.27 - label_table on a three-record processed store

`tests/torchcell/data/test_label_policy.py` covers the policy and the success paths of `triple_roles`; this file pins the remaining branches, `label_table_path`, `_finite`, `entries_of_record`, `_rows` and `build_label_table` on a store written by hand in the processed-LMDB shape (key `str(i)`, value the JSON list of experiment plus reference dumps) under `tmp_path`. Under the default `kuzmin-first` policy: record 0 carries a Costanzo 30 C fitness 0.9 and an SGD converted zero, so the zero yields and fitness is 0.9 from `costanzo2016@30` with 2 entries available and 1 combined; record 1 carries Kuzmin 2018 fitness 0.5 over Costanzo 0.7 and a Kuzmin interaction of -0.2; record 2 carries only an interaction, so no fitness. `perturbation_count_index.json` is `{"1": [0, 2], "2": [1]}`. The policy id `b2f26bb6ef86d11a` fixes the cache path `<root>/label_tables/<id>.parquet`. Finding: the cache write tests `indices is not None` after the full-build branch has filled `indices` (label_table.py line 231), so a subset run with the default `cache=True` writes a partial parquet that the next full run reads back as complete, the opposite of what the docstring says. The `len(query) != 2` branch of `triple_roles` (line 86) cannot fire once three genes and exactly one array gene are found. Phase 6 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.10.01 - Unknown experiment type refused

`test_entries_of_record_refuses_an_experiment_type_no_policy_ranks`: a stored "calmorph" entry raises `no label for experiment type 'calmorph'` instead of being read as fitness (issue #527).
