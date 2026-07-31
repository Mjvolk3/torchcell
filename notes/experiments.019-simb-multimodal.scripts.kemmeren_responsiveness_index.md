---
id: 4t4rrf64jl0abodi3rpdim1
title: Kemmeren_responsiveness_index
desc: ''
updated: 1785532038584
created: 1785532038584
---

## 2026.07.31 - Recovering the responsive / non-responsive label the loader computes and throws away

Kemmeren2014 profiled deletion mutants in two GEO series BY DESIGN -- GSE42527 responsive, GSE42526 non-responsive -- and that label is the first thing to check when an identical A0 config scores `val/mean/pearson_per_feature` at 0.1527 / 0.0235 / 0.0661 across three seeds that are three different SPLITS (`seed` feeds `CellDataModule(random_seed=...)`). The label is computed inside the loader and never persisted, so this script recovers it from the mirrored raw artifacts instead, handing downstream analysis a per-strain responsiveness class with no dataset rebuild.

- **Output** `results/kemmeren_responsiveness.json`: **700** responsive strain tokens from GSE42527 and **783** non-responsive from GSE42526, with **0 overlap**. A token appearing in both series would be ambiguous, so the script reports the overlap set rather than silently picking a side.
- **Source is the sha-pinned mirror the loader already downloaded** (`$DATA_ROOT/data/torchcell/microarray_kemmeren2014/raw/*_family.soft.gz`, 421 MB and 390 MB gzipped) -- no network, no re-query. Only `!Sample_title` lines are read; the full SOFT carries every probe measurement. Measured title counts: 1387 arrays -> 700 genes (GSE42527), 1246 arrays -> 783 genes (GSE42526).
- **Why a script and not a loader change.** `torchcell/datasets/scerevisiae/kemmeren2014.py:412` already sets `sample_info["is_responsive_mutant"] = gsm_name in responsive_gsm_names`, but never writes it into the schema, so it is absent from the built LMDB. Persisting it is the right long-term fix (tracked as a GitHub issue) and costs a dataset rebuild plus re-query we did not want mid-round.
- **The parse is fail-loud for a reason.** Titles are `<gene>-del-<replicate>-<dye swap>`, optionally prefixed by a hybridization-batch tag (`[hs1991] abf2-del-1-d`). A `^`-anchored regex without the bracket alternative returned `None` for **1,891 of 2,633** titles, which a naive comprehension dropped silently -- the label set looked clean and complete at 230/148 genes while omitting 72% of the data. The landed version raises `ValueError` on any unparsed title, which is why the counts are 700/783 and not 230/148.

```python
# optional [batch] prefix; no '-' in the class, or the whole sample id is swallowed
_TITLE_GENE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?([A-Za-z0-9]+)")
```

- **Shipped scope is labels only.** The docstring's usage (`--seeds 100 101 102 --base-seed 0`) and its plan to write a stratified `index_seed_<N>.json` + `index_details_seed_<N>.json` into the datamodule cache dir are NOT in the landed `main()`, which takes `--raw-dir` / `--out` and stops after the label JSON. The downstream measurement leaves that unmotivated: [[experiments.019-simb-multimodal.scripts.stratified_responsiveness_eval]] finds the plain random split already balanced to within 3 percentage points of 50/50 responsive in every split and every seed, so there is nothing for stratification to correct.
