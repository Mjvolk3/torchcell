---
id: zbak11dhipydfw104rbfm7o
title: Li2014
desc: ''
updated: 1791625454750
created: 1791625454750
---

## 2026.10.10 - Loader, the synthesis-rate leaf, and the L0-L4 table (#857)

Li GW, Burkhardt D, Gross C, Weissman JS (2014) Cell 157:624-635, doi:10.1016/j.cell.2014.02.033, PMID 24766808. Loader: `torchcell/datasets/ecoli/li2014.py` (`ProteinSynthesisRateLi2014Dataset`). Release inventory and mirror deposit: [[experiments.036-dataset-fixes-before-kg-build.scripts.li2014_release_inventory]]. Duplication and round-trip measurements: [[experiments.036-dataset-fixes-before-kg-build.scripts.li2014_synthesis_rate_checks]].

### Schema decision: a new leaf, not a relaxed turnover class

`ProteinTurnoverPhenotype` has `label_name = "degradation_rate"`, requires a non-empty `degradation_rate`, and admits `synthesis_rate` only on keys that carry one. Making `degradation_rate` optional would let one class carry two primary labels, with `label_name` switching per record, so a consumer of the turnover lane could no longer trust what the label is. The new `ProteinSynthesisRatePhenotype` (in `torchcell/datamodels/schema.py`, block `#857`) is a sibling under `phenotypic feature`:

- `synthesis_rate` is the label; `synthesis_rate_se`, `n_replicates` and `censoring` are optional.
- `rate_unit` (`SynthesisRateUnit`: `molecules_per_generation`, `molecules_per_hour`) types the amount and the time basis; a rate per generation must state `generation_time_minutes`, because the doubling time is what converts it to a rate per hour.
- Graph class `protein synthesis rate phenotype`, a `phenotype member of` source, CellAdapter node method and reference collector.

Schema impact (`scripts/schema_impact_check.py --base origin/main`): 9 symbols added or grown (the leaf, its pair, the unit enum, the three unions, both type maps), **0 breaking**. Nine datasets read stale only through the grown `ExperimentType` union (Caglar 2017 x3, Gupta 2024, Ishii 2007 x4, Schastnaya 2021, De Siqueira 2025 x4, Lim 2022, and the private Volk 2021 and Hoepfner 2014). Their records do not change: every record of 12 of those dev stores, validated through this branch's schema and dumped again, equals the stored dict (0 of 622 records differ; script and results below). The 13th, `IsoprenolTiterDeSiqueira2025Dataset`, was written by another branch's schema (`titer_censoring`), a shared-dev-tree effect that says nothing about this change. The shared stores were not rebuilt on this branch, because a rebuild here would make them stale for `main`; KG 4.0 is a full rebuild.

### What is stored

One `ProteinSynthesisRateExperiment` per medium, wild type (`Genotype(perturbations=[])`), keyed on MG1655 b-numbers, against the MG1655 GenBank assembly. Counted from the built dev store (`$DATA_ROOT/data/torchcell/protein_synthesis_rate_li2014`, built 2026-10-10 with `build_dataset_lmdb --verify`):

| medium | Table S1 column | plain cells | stored keys | generation time (min) |
|---|---|---|---|---|
| complete | MOPS complete | 3,041 | 3,025 | 21.5 |
| minimal | MOPS minimal | 3,362 | 3,346 | 56.3 |
| complete without methionine | MOPS complete without methionine | 2,241 | 2,226 | 26.5 |

3,391 distinct b-numbers over the three records. The reference of every record is the MOPS-complete record, the condition "The results presented in this work are based on". The media are `MOPS_MINIMAL` plus 0.2% glucose, with the supplement as a `composition_deferred` component deferring to Neidhardt 1974 (not mirrored); the methionine dropout is its own named sub-mix rather than a `dropout` edit, since an edit of an unexpanded mixture would claim a composition the record does not hold. `Environment` has no culture-format field, so the 2.8-l flask and 180 rpm are quoted in the loader but not stored.

### Typed refusals (per cell, `preprocess/refused_cells.csv`)

| reason | complete | minimal | without Met |
|---|---|---|---|
| `below_read_count_gate` (`[n]` cell) | 1,054 | 733 | 1,854 |
| `merged_pair_one_value_two_loci` | 2 | 3 | 1 |
| `name_retired_in_assembly` | 6 | 5 | 7 |
| `name_ambiguous_in_assembly` | 5 | 5 | 5 |
| `two_rows_one_locus` (`ecpD`/`yagW`, `mscM`/`yjeP`) | 3 | 3 | 2 |

Every medium balances: stored + refused = 4,095 rows (asserted by `BuildAccounting.check`). The stored counts equal the release inventory's "loadable keys" exactly. The bracket is back-solved, not defined by the source: the main text's "we evaluated 3,041 genes" equals the MOPS-complete plain count, and the build stops if that stops holding.

Measured, not explained: some plain cells are tiny. Plain values of 0 occur once in minimal and three times in the methionine dropout, and plain values of 1 to 5 occur in all three columns (`acrF` 5 in complete, `fdnG` 0 in minimal), although the main text says "The lowest expression rate among these genes correspond to ~10 molecules per generation". Hypothesis (untested): the >128-footprint gate is on read count, so a long gene can pass it at a low per-codon density; the stored values are the released cells verbatim either way.

### Not stored, with typed gaps

- No degradation rate, half-life or turnover: none is released.
- No abundance: "For stable proteins, ki is also the copy number" is an assumption, and the main text calls the rate "an upper bound for the protein levels for the small subset of proteins that are rapidly degraded".
- `n_replicates` and `synthesis_rate_se` are None with `ProvenanceGap(not_reported_by_primary)`: the paper's "error of less than 1.3-fold across biological replicates" is a bound over all genes, and GEO GSE53767 holds pooled tracks (one rich-medium and two minimal-medium ribosome-profiling samples, none for the methionine dropout).

### L0-L4 verification (`verify_build`, dev store, 2026-10-10)

| level | check | result |
|---|---|---|
| L0 | every record validates against `ExperimentType` | pass, 3 records |
| L1 | record count | pass, 3 of 3 |
| L1 | per-medium key counts are the pinned ones | pass, 3,025 / 3,346 / 2,226 |
| L2 | rates finite and non-negative | pass, 8,597 values |
| L2 | generation time is the SI doubling time | pass, 21.5 / 56.3 / 26.5 |
| L3 | assembly pin is MG1655 GenBank | pass, `ecoli_K12_MG1655_ASM584v2` |
| L3 | every quote verbatim in its sha256-pinned mirror file | pass, 16 quotes (13 text, 3 Table S1 header cells) |
| L4 | every stored key is a locus of the pinned assembly | pass, 3,391 keys, 0 outside |

### Duplication by content against served proteome stores

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/li2014_synthesis_rate_checks.py`; results: `experiments/036-dataset-fixes-before-kg-build/results/li2014_synthesis_rate_checks.json`. Schmidt 2016 and Ishii 2007 key on BW25113 tags, carried to b-numbers through a one-to-one ECK synonym (4,402 tags mapped).

| store | records | max shared keys | identical values | Spearman range over (Li medium x record) |
|---|---|---|---|---|
| Schmidt 2016 proteome | 14 | 2,279 | 0 | 0.614 to 0.847 |
| Ishii 2007 proteome | 28 | 57 | 0 | 0.305 to 0.776 |
| Brunk 2016 proteome | 81 | 58 | 0 | 0.447 to 0.855 |
| Gupta 2024 turnover | 13 | 2,657 | 0 | -0.126 to 0.095 |

Verdict: not a duplicate. No shared key carries the same number in any store, and the rank agreement with the abundance stores is what a related but different quantity gives (the best pair, Li minimal against Schmidt record 5, is 0.847 over 1,997 keys), not a re-release.

### Files

- Loader: `torchcell/datasets/ecoli/li2014.py`; tests `tests/torchcell/datasets/ecoli/test_li2014.py`
- Adapter: `torchcell/adapters/li2014_adapter.py`, conf `torchcell/adapters/conf/protein_synthesis_rate_li2014_adapter.yaml`
