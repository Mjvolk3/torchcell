---
id: m82lzscvtasrg8h9xbj020g
title: Nguessan2025_snp_coordinate_audit
desc: ''
updated: 1791807289084
created: 1791807289084
---

## 2026.10.12 - Which reference the Nguyen Ba 2022 SNP positions sit on

`experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_coordinate_audit.py` writes `experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_snp_coordinate_audit.json`. Run under slurm (job 4233, 13 s, 1.5 GB peak RSS) from the repo root with the worktree on `PYTHONPATH` and `--by4742 <path>`; `BY4742_fixed.fasta` is not in a mirror, so the script refuses a file that does not hash to `c2e18c61d0529cf44d5991702f7ee50a3d968b2fa8e97bbc65a35d90b83c6f4e` (the retrieval command, a commit-pinned `curl` of `arturrc/bbqtl_inference` @ `2b1c89d6`, is in the docstring).

Why it exists: the N'Guessan 2025 store (slurm job 4199) failed L4 `marker_reference` and `mat_fixation` ([[torchcell.datasets.scerevisiae.nguessan2025]], #1050). Measured over all 41,594 SNPs of the deposited `SNP_list.txt`:

| chromosome | SNPs | BY allele = S288C R64-4-1 base | BY allele = `BY4742_fixed` base |
|---|---|---|---|
| chrI | 995 | 1.0000 | 1.0000 |
| chrII | 3,735 | 0.5371 | 1.0000 |
| chrIII | 558 | 0.4498 | 1.0000 |
| chrIV | 4,035 | 0.7051 | 1.0000 |
| chrV | 1,859 | 0.4325 | 1.0000 |
| chrVI | 1,288 | 1.0000 | 1.0000 |
| chrVII | 3,931 | 0.9479 | 1.0000 |
| chrVIII | 1,638 | 0.4200 | 1.0000 |
| chrIX | 2,078 | 0.4880 | 1.0000 |
| chrX | 2,974 | 0.9499 | 1.0000 |
| chrXI | 3,239 | 0.3844 | 1.0000 |
| chrXII | 4,310 | 0.3691 | 1.0000 |
| chrXIII | 2,945 | 0.9847 | 1.0000 |
| chrXIV | 1,799 | 0.2496 | 1.0000 |
| chrXV | 4,289 | 0.8837 | 1.0000 |
| chrXVI | 1,921 | 0.4185 | 1.0000 |
| all | 41,594 | 27,221 (0.6544) | 41,594 (1.0000) |

The offset that best aligns windows of 15 consecutive SNPs to S288C (search +-6,000 bp, a window resolved at 13 of 15 or better) is piecewise constant: 0 from the left end of every chromosome to a breakpoint, then 1 to 6 bp on most chromosomes (chrIV -1 then -2, chrVIII +2, chrIX -3, chrXI -3, chrXII +1, chrXIV +1 to +6, chrXVI +2 then +3), +190 on chrXV after 726 kb, -327 on chrXIII after 908 kb, +1,097 on chrV after 116 kb, +4,489 on chrII after 471 kb; chrIII after 86 kb (420 SNPs) is not resolved within the range. Along chrIII of the store, mean P(RM) over the 2,951 records is lowest at 187,586 (0.0414) and rises to both sides; the SNP nearest 200,000 (200,179) reads 0.1083.

The README's statement that the dataset is based on `S288C_reference_genome_R64-3-1_20210421` does not hold for these positions: they are `BY4742_fixed` coordinates.
