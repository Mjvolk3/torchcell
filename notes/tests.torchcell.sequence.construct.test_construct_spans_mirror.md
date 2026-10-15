---
id: qtepy3tf8duvgdo0nycvs9p
title: Test_construct_spans_mirror
desc: ''
updated: 1792086736141
created: 1792086736141
---

## 2026.10.15 - Spans of parts, asserted against the real maps by name

`tests/torchcell/sequence/construct/test_construct_spans_mirror.py` holds the
measurements the span design was built from, on the 17 mirrored Cachera maps rather than
on a fixture, and is marked `data` so it runs with `--data` and a real `DATA_ROOT`
exported (the root `conftest.py` sets a sentinel root for a plain `pytest` run, so the
real root has to come from the shell).

Five tests, each a number that moved:

| test | what it pins |
|---|---|
| `test_all_seventeen_cachera_maps_index_with_no_blocked_rows` | the deposited index holds 17 contigs, 0 blocked, every name sourced |
| `test_pbtx002_puc_ori_is_666_bp_and_not_the_whole_plasmid` | `pUC\ori` at `11559..128` is 666 bp on a 12,096 bp replicon, with both INSDC spellings |
| `test_the_ampr_cds_of_pbtx013_is_861_bp_on_the_reverse_strand` | the strand case: `complement(4892..858)` keeps the file's part order, 861 bp |
| `test_every_feature_of_every_map_regenerates_its_source_location` | 570 of 570 locations regenerated verbatim |
| `test_the_set_carries_eleven_wraps_and_no_insdc_join` | 11 wraps, 0 joins, and no `join(` or `order(` anywhere in any FEATURES block |

The last one is why the `join` case has no real-map test: the audit read 11 `join`
locations and 11 origin-wrap warnings as two overlapping findings, and they are one.
The files carry no `join` operator at all, so a join is asserted against the fixtures in
`tests/torchcell/sequence/test_plasmid.py`
(`test_a_plain_two_part_join_is_joined_and_not_origin_spanning`) and the absence is
recorded here instead of being papered over.

Run it with:

```bash
PYTHONPATH=$PWD DATA_ROOT=/scratch/projects/torchcell-scratch \
  python -m pytest tests/torchcell/sequence/construct/test_construct_spans_mirror.py --data
```
