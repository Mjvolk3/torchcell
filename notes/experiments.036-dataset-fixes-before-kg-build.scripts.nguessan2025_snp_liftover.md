---
id: pfgy5z5z5viqdyt98mi2km8
title: Nguessan2025_snp_liftover
desc: ''
updated: 1791814078710
created: 1791814078710
---

## 2026.10.12 - A modeling-time lift of the Nguyen Ba 2022 SNPs to S288C R64-4-1

Not part of the N'Guessan 2025 dataset. The loader stores the Dryad genotypes exactly as released, positions included, and reads nothing this script writes (owner, 2026-10-12, [[torchcell.datasets.scerevisiae.nguessan2025]], #1050). The released positions are `BY4742_fixed.fasta` coordinates ([[experiments.036-dataset-fixes-before-kg-build.scripts.nguessan2025_snp_coordinate_audit]]); a model that needs S288C coordinates can use this lift.

`experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_liftover.py --retrieve` fetches `BY4742_fixed.fasta` from `arturrc/bbqtl_inference` at `2b1c89d6` with `torchcell.literature.retrieve.direct_url`, checks it against sha256 `c2e18c61d0529cf44d5991702f7ee50a3d968b2fa8e97bbc65a35d90b83c6f4e` and writes its `RetrievalRecord` beside it. For each of the 41,594 released SNPs it takes the 25, 50, 100 and 200 bp flanks on each side (the SNP base excluded) from `BY4742_fixed` and searches them on both strands of every S288C R64-4-1 chromosome: one exact match lifts the SNP, none is `no_match`, several try the next flank and end as `multiple`, a flank off the chromosome is `edge`. Every SNP is written with its flank length and hit count. It then applies the verifier's two L4 genome-reference rules to the lifted SNPs. Inputs are checked against their pins (raw-mirror manifest, genomes registry).

Outputs: `$DATA_ROOT/experiments/036-dataset-fixes-before-kg-build/nguessan2025_coordinates/` (`BY4742_fixed.fasta` and its retrieval record; `SNP_list_lifted_S288C_R64-4-1.tsv`, 2,537,055 B, sha256 `18c768d365f7086874f13fa02ffa8ea5c277e71a1ef59ead04b1c3dfb83be211`, and its processing record naming this script at `a53c938f2`, the flanks and seed, and the three inputs' digests) and `experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_snp_liftover.json`. Run under slurm (job 4335, 49 s, 2.5 GB peak RSS). The lift functions are tested in `tests/torchcell/datasets/scerevisiae/test_nguessan2025_snp_liftover.py`, which loads the script by path.

| quantity | value |
|---|---|
| lifted | 41,587 of 41,594 (41,472 at the 25 bp flank, 100 at 50, 14 at 100, 1 at 200) |
| `no_match` | 7 (2 on chrI, 1 chrII, 1 chrX, 1 chrXII, 2 chrXIII) |
| `multiple`, `edge` | 0, 0 |
| strand | all 41,587 on the plus strand; none lifted to another chromosome |
| S288C base at the lifted position = BY allele | 41,578 of 41,587; the other 9 read the RM allele (sites where BY4742 and S288C differ) |
| lifted position = released position | 22,129 |
| offset range | chrII -1 to +4,489, chrIII 0 to +10,580, chrV 0 to +1,097, chrXIII -327 to 0, chrXV 0 to +190; chrI and chrVI 0 |
| L4 `marker_reference` on the lifted SNPs | 41,578 of 41,587 (0.999784), against 0.6740 on the released positions in the store |
| L4 `mat_fixation` on the lifted SNPs | mean P(RM) 0.0438 over 2,951 strains at the lifted SNP nearest S288C MAT (chrIII:201,156, released `snp5020`), within 0.05; 0.1097 on the released positions |
