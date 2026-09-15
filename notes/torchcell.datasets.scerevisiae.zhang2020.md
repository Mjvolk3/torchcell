---
id: ku4hq9qk1li24owoyoikdrg
title: Zhang2020
desc: ''
updated: 1789437513577
created: 1789437513577
---

## 2026.09.14 - The tryptophan promoter-combination data exists and is now in the raw mirror

**The data-availability statement we had was truncated by OCR.** Our mirrored `paper.md` ends at "available from the correspondi...", which read as author-request-only. The published statement (PMC7519671) continues: genotypes at JBEI ICE, time series at JBEI EDD under the study "Zhang and Petersen, et al. 2019", **and a copy at** <https://github.com/sorpet/Zhang_and_Petersen_et_al_2019>, plus Source Data and four Supplementary Data workbooks. `public-edd.jbei.org` needs a manually approved account, so the GitHub copy is the retrievable source; its README states it is a copy of the same ICE and EDD records.

**Mirrored 2026-09-14** to `$DATA_ROOT/torchcell-raw/zhangCombiningMechanisticMachine2020/data/`, pinned to commit `f95f7d10`, with `manifest.json` carrying per-file sha256 and the retrieval command.

| file | bytes | what it holds |
|---|---|---|
| `ice.csv.zip` | 5,839 | 576 strains: name (SP001...), cured/single-population flags, plate alias, and `strainData`, the 5-slot promoter design |
| `edd.csv.zip` | 2,491,709 | 289,152 measurement rows |
| `recommendations.csv` | 983,509 | ART and EVOLVE recommended promoter combinations |
| `biosensor_val.csv` | 485 | HPLC tryptophan titers for the validation strains |

**Shape of the measurements.** 576 strains x 3 replicates = 1,728 lines, two measurement types (enhanced GFP and optical density), median 81 time points per line per type, 144,576 rows each. The time column spans 13 to 1404 in its recorded units. So the phenotype is a growth-and-fluorescence time series per replicate, not a single scalar; the paper derives a GFP synthesis rate from it and uses that as the tryptophan proxy.

**Shape of the genotype.** `strainData` is five underscore-separated tokens, one per gene in the order PCK1, TAL1, TKL1, CDC19, PFK1, each either a promoter number or `NI` for the native promoter. 8 to 10 distinct tokens appear per slot. 457 of the 576 strains carry distinct designs, so 119 are repeats or controls.

**Why ingestion is not a normal deletion loader.** The perturbation is a promoter replacement plus relocation of all five genes out of their native loci into one cluster at EasyClone site XII-5, so the edit is a cassette integration, not a deletion. The phenotype is a biosensor reporter time series. Both need schema work beyond the deletion-collection pattern. Related: [[paper.north-star.dataset-triage]].
