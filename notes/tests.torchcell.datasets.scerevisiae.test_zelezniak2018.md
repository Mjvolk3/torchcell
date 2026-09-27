---
id: 7fnyn3mq9q6jrtbluhoxgr6
title: Test_zelezniak2018
desc: ''
updated: 1790482262626
created: 1790482262626
---

## 2026.09.26 - Hermetic build tests for the Zelezniak 2018 proteome and metabolome loaders

Test file: `tests/torchcell/datasets/scerevisiae/test_zelezniak2018.py` (13 tests), covering
[[torchcell.datasets.scerevisiae.zelezniak2018]]. Part of the test-coverage campaign
([[plan.test-suite-buildout.2026.09.25]]); the module sat at 24.8% line coverage before this file.

### Approach

Each test writes a hand-sized Zenodo-shaped TSV under `<tmp_root>/raw/` so PyG skips `download()`,
lets `process()` build the LMDB once, and compares `ds[i]["experiment"]` / `["reference"]` /
`["publication"]` against hand-built pydantic objects (`model_dump()` on both sides). No network,
no `$DATA_ROOT`. The metabolome's `build_metabolite_s_id_map` (which loads Yeast9 through
`YeastGEM`) is monkeypatched to a recording stub returning a fixed `s_NNNN` map, and the stub's
recorded argument is asserted exactly.

### Proteome fixture and expected values

`groupby("ORF")` mean / sample SD / count per strain, SE = SD / sqrt(n):

| KO_ORF | protein | values | mean | SE | n |
|---|---|---|---|---|---|
| WT | YAL001C | 10, 12 | 11.0 | 1.0 | 2 |
| WT | YBR002C | 5, 7, 9 | 7.0 | 2/sqrt(3) | 3 |
| YDR003W (KIN3) | YAL001C | 8, 9 | 8.5 | 0.5 | 2 |
| YDR003W (KIN3) | YBR002C | 6, 6 | 6.0 | 0.0 | 2 |
| YER004W (KIN4) | YAL001C | 1, 3 | 2.0 | 1.0 | 2 |
| YER004W (KIN4) | YBR002C | 20, 24 | 22.0 | 2.0 | 2 |
| bad_ko | YAL001C | 99 | skipped | | |

Asserted: `len == 2`; record 0 is YDR003W and record 1 YER004W (`groupby` sorts KO_ORF);
`preprocess/data.csv == "orf,gene\nYDR003W,KIN3\nYER004W,KIN4\n"`; `gene_set.json ==
["YDR003W", "YER004W"]`; one `experiment_reference_index` entry with `member_indices [0, 1]`;
`build_manifest.json` carries `manifest_schema_version 1`, the root basename as `dataset_name`,
the loader class and module, the hostname, and `surface_modules == ["pydant.py", "schema.py"]`.
Every record is compared by whole `model_dump()` equality, SE dicts included: the fixture
values were chosen so pandas `std() / sqrt(n)` is exactly 0.5, 0.0, 1.0, 2.0 and
`2.0 / math.sqrt(3)`, so no tolerance is used anywhere.

### Metabolome fixture and expected values

Rows pooled across the `dataset` protocol column per (metabolite, strain):

| genotype | metabolite | values | mean | SE | n |
|---|---|---|---|---|---|
| WT | pyr | 2 (protocol 1), 4 (protocol 2) | 3.0 | 1.0 | 2 |
| WT | 3pg;2pg | 10 | 10.0 | NaN | 1 |
| YDR003W | pyr | 1, 3 | 2.0 | 1.0 | 2 |
| YDR003W | atp | 5, 5 | 5.0 | 0.0 | 2 |
| YER004W | 3pg;2pg | 7 | 7.0 | NaN | 1 |

Asserted: the stub is called once with `{"pyr": "C00022", "3pg;2pg": "C00197;C00631", "atp":
"C00002"}`; record 0 (YDR003W) has levels `{pyr 2.0, atp 5.0}` and a reference restricted to
`{pyr}` only (WT never measured atp); `target_metabolite_ids` is the stub map filtered to each
side's keys; record 1 (YER004W) stores `metabolite_level_se None` on both experiment and
reference because the only SE is NaN; two reference-index entries `[[0], [1]]`;
`data.csv == "orf,n_metabolites\nYDR003W,2\nYER004W,1\n"`.

### Findings (pinned as the code behaves)

- A non-systematic `KO_ORF` in the proteome file is silently skipped (only a log count), while a
  non-systematic protein `ORF` raises `RuntimeError("non-systematic protein ORF ids present: N")`.
  The proteome regex has no `Q\d{4}` alternative, so a mitochondrial protein id (`Q0250`) would
  abort a proteome build; the Messner and YeastPhenome loaders accept it.
- A protein or metabolite measured once in a strain stores `float("nan")` as its SE rather than
  omitting the key; the proteome never collapses an all-NaN SE dict to None, the metabolome does.
- In the metabolome, the genotype regex check runs before `build_metabolite_s_id_map` (zero stub
  calls on failure) and the missing-WT check runs after it (one stub call on failure).
