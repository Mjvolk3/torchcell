---
id: 41xcusmynt9t7yfdz5jpevq
title: Li2024
desc: ''
updated: 1791613793107
created: 1791613793107
---

## 2026.10.10 - Row 60 settled: D2Cell 2026 is a secondary source, not loaded

**Decision.** Row 60 of the bacterial schedule (D2Cell 2026, Li et al., bioRxiv doi:10.1101/2024.09.09.612023, published in Trends in Biotechnology 2026, citation key `liLeveragingLargeLanguage2024`) is settled as a provenance record on the MCF2Chem 2023 pattern ([[torchcell.datasets.ecoli.cai2023]]). `torchcell/datasets/ecoli/li2024.py` registers no dataset; it holds the retrieval records, the verbatim quotes, the measurement functions and the schema blockers. No loader, adapter or store.

**Route.** The paper is in the literature mirror (`paper.md` sha256 `ec236a1a...10ae8e7`). The data are not, so the deposit went to `$DATA_ROOT/torchcell-raw/liLeveragingLargeLanguage2024/` through `torchcell.literature.retrieve.direct_url` (four files, sha256-pinned, `manifest.json` with retrieval records): the Zenodo 18240770 workbook `Cell_factory_dataset_Qwen_110b.xlsx` (sha256 `ce4f08ed...e5e917915494eecf64adcde`, 11,437,401 bytes) and the three D2Cell-pred E. coli split CSVs at GitHub commit `db5f0a83`.

### What the deposit is, measured

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/d2cell2026_release_inventory.py`; results: `experiments/036-dataset-fixes-before-kg-build/results/d2cell2026_release_inventory.json` and `d2cell2026_ecoli_source_dois.csv`. Every quote in the module is checked as a substring of the pinned `paper.md` by the same script (11 of 11 present).

| Workbook (LLM-extracted database) | Count |
|---|---|
| non-empty rows / columns | 29,007 / 39 |
| E. coli rows | 10,525 |
| distinct source DOIs (E. coli; 2,029 after lowercasing) | 2,030 |
| E. coli rows with a numeric `titer value` | 10,024 |
| distinct `titer unit` strings | 383 (g/L 3,982; mg/L 2,620; mM 631; g/g 545; % 280; mol/mol 242; ...) |
| `parent strain` empty | 3,354 |
| `parent strain` holding a temperature (`37°C` etc.) | 695 |
| no knockout, overexpression or heterologous gene | 1,932 |
| placeholder text in a condition field ("Not specified", "Assumed", "See note", "Same as", "similar") | 2,972 |
| Chinese text in an English field | 108 |
| rows carrying the source sentence a value was read from | 0 (no such column) |

| D2Cell-pred E. coli splits | Count |
|---|---|
| rows (train / valid / test) | 15,822 / 1,977 / 1,978 = 19,777, equal to the paper's stated total |
| `real data == yes` | 1,810 (paper states 8,134 experimental) |
| real rows by label (1 / 0) | 905 / 905 |
| simulated rows by label (1 / 0) | 12,660 / 5,307 |
| rows with a `doi list` | 905; 0 of them real positives |

The 1,810 versus 8,134 gap is measured; why the flag marks fewer rows than the paper's count is not stated in the release and is not explained here.

### Why it is not admitted

1. The values are transcriptions by a language model, not measurements. The paper: "In the RE module, we use the Qwen1.5- 110B-Chat model to extract relationships between entities." and the corpus was "more than 10000 abstracts and 1340 openaccess full texts". No row carries the sentence it was read from, so no value can be quoted against source bytes, and the field misalignment counts above are what an unaudited extraction looks like.
2. The training labels are constructed. "Therefore, we designated gene modifications opposing those enhancing product production in the literature as negative samples." and "Additionally, genes not predicted as targets by FSEOF were included in the negative dataset." A 0 is not a measured null.
3. Required schema fields have no honest value: no parent-strain titer for `ProductTiterExperimentReference.phenotype_reference`, 383 unit strings mixing concentrations, yields, percentages and activities for `titer_unit`, and gene-name strings with no locus, allele or promoter for the genotype.

### Duplication

Of the 2,029 lowercased E. coli source DOIs, 1 is the DOI of a served bacterial dataset (Niu 2019, 10.1016/j.synbio.2019.05.001; one D2Cell row whose titer cell is the sentence "Increased production varies from 7% to 31% depending on the overexpressed gene (e.g., cbpA, tabA, dxs)") and 8 are rows of the bacteria candidate table. Overlap with the served store is therefore negligible by DOI; nothing is subsumed.

### What remains useful

`d2cell2026_ecoli_source_dois.csv` lists the 2,030 source DOIs with their entry counts, a lead list of primary metabolic engineering papers. Any of them would enter only through its own release, quoted from its own bytes.
