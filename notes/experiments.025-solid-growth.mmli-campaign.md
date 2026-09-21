---
id: s2aila785erdifcb7mdnyva
title: Mmli Campaign
desc: ''
updated: 1790008362725
created: 1790008362725
---

## 2026.09.21 - The campaign on the mmli node

Typeset document: `notes-tex/025-mmli-campaign/` (`make refresh` refills both tables from W&B and the live queue). Generator: [[experiments.025-solid-growth.scripts.mmli_campaign_plan]].

Two claims. Training on fitness records merged across datasets improves trigenic interaction prediction, and the full-gene composite embedding matches the learnable table where the table is at its best, which is what licenses a transferable representation.

Queue order on mmli, one four-GPU job at a time, 26 jobs, about 547 hours:

1. S3 on the random split, table, seeds 2 and 3 (jobs 2409672, 2409673). Seed 1 scored 0.478 against S0 at 0.439 (n = 3, sd 0.003).
2. Composite on the random split: S0 seeds 1 to 3 (`cgt_s0_r_kl_embfit_035`, jobs 2409705 to 2409707), then S3 seeds 1 to 3 (`cgt_s3_r_kl_embfit_034`, jobs 2409708 to 2409710).
3. Single-region replicates on the disjoint split (jobs 2409035 to 2409039).
4. S4 size-matched control, seed 1 (`cgt_s4_r_kl_fit_039`, job 2409711).
5. Table with joint fitness on the disjoint split, the one-change baseline (`cgt_s0_q_kl_fit_038`, jobs 2409712 to 2409714).
6. S3 strict on the disjoint split, composite and table alternating by seed (`cgt_s3_q_kl_embfit_037`, `cgt_s3_q_kl_fit_036`, jobs 2409715 to 2409720).
7. S4 seeds 2 and 3 (jobs 2409721, 2409722).

S3 strict is S3 minus the 85 closure doubles that are themselves a held-out query pair of the disjoint split ([[experiments.025-solid-growth.scripts.subset_s3_query_strict]]). The 145,208 doubles pairing a held-out query gene with an array gene stay in train.

All new jobs run from the IGB worktree `025-fitness-joint-head-j` at commit 5f549bc63.
