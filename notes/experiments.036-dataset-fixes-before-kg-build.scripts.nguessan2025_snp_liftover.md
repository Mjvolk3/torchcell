---
id: pfgy5z5z5viqdyt98mi2km8
title: Nguessan2025_snp_liftover
desc: ''
updated: 1791814078710
created: 1791814078710
---

## 2026.10.12 - The Nguyen Ba 2022 SNPs lifted from BY4742_fixed to S288C R64-4-1

`experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_liftover.py` writes the lifted SNP table into the raw mirror (`derived/SNP_list_lifted_S288C_R64-4-1.tsv`, 2,537,055 B, sha256 `18c768d365f7086874f13fa02ffa8ea5c277e71a1ef59ead04b1c3dfb83be211`) and `experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_snp_liftover.json`. The lift is `torchcell.datasets.scerevisiae.nguessan2025.lift_snps`: for each of the 41,594 released SNPs, the 25, 50, 100 and 200 bp flanks on each side (the SNP base excluded) are taken from `BY4742_fixed.fasta` at the released position and searched on both strands of every S288C R64-4-1 chromosome; one exact match lifts the SNP, none is `no_match`, several try the next flank and end as `multiple`, a flank off the chromosome is `edge`. Every SNP is written with its flank length and hit count; none is dropped. Inputs are checked against their pins: `SNP_list.txt` and `BY4742_fixed.fasta` against the raw-mirror manifest, the S288C FASTA (sha256 `dbf065ff...`) through the genomes registry.

Run under slurm from a clean checkout at `2a6c9bcfb` (job 4319, 13 s, 1.4 GB peak RSS; the first run, job 4307 at `e94cecb0b`, wrote the same bytes). The raw-mirror manifest records the table as a derived artifact with a `ProcessingRecord` naming this script, that commit, the flanks and seed, and the three inputs' digests.

| quantity | value |
|---|---|
| lifted | 41,587 of 41,594 (41,472 at the 25 bp flank, 100 at 50, 14 at 100, 1 at 200) |
| `no_match` | 7 (2 on chrI, 1 chrII, 1 chrX, 1 chrXII, 2 chrXIII) |
| `multiple`, `edge` | 0, 0 |
| strand | all 41,587 on the plus strand; none lifted to another chromosome |
| S288C base at the lifted position = BY allele | 41,578 of 41,587; the other 9 read the RM allele (sites where BY4742 and S288C differ) |
| lifted position = released position | 22,129 |
| offset range | chrII -1 to +4,489, chrIII 0 to +10,580, chrV 0 to +1,097, chrXIII -327 to 0, chrXV 0 to +190; chrI and chrVI 0 |

The stored N'Guessan records do not read this table: they keep the released positions (owner, 2026-10-12, [[torchcell.datasets.scerevisiae.nguessan2025]]). The table is for modeling.
