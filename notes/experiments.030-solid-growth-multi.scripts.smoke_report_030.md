---
id: eomd8t2h2z5zdclzxfrgfk8
title: Smoke_report_030
desc: ''
updated: 1790417029711
created: 1790417029711
---

## 2026.09.26 - The three criteria

Reads the token run's and the control run's `SmokeResult` and prints PASS/FAIL per criterion (plan decision 10), with delta the normalized offset (0.3): (1a) mean of pred(synthetic token) minus pred(own token) within 10% of delta, (1b) its sd over rows below delta/5; (2) reading real rows under the synthetic token raises their MSE by 0.5 to 1.5 delta squared; (3) the control's MSE over all rows exceeds the token run's by at least delta squared over 8 (half the delta squared over 4 floor a token-blind predictor pays on the cloned rows). Exit code 0 on PASS, 1 on FAIL; the smoke slurm job exits with it.

## 2026.09.26 - Result: PASS on the third submission, and what the first two measured

Job 2867 (`cgt_030_smoke_tok_000`, 3 x 500 steps, one RTX 6000, 1 h 17 m, W&B run 5xn8w427): pred(twin) minus pred(own) on 2,579 real trigenic rows of 40 validation batches 0.2854 (bound 0.27 to 0.33), sd 0.0528 (bound < 0.06), MSE rise under the twin token 0.0550 (bound 0.045 to 0.135). Trajectory by validation epoch: 0.046, 0.207, 0.282. Beside it, from the same partial model: policy-reduced val Pearson 0.369 (Tmi2018 rows 0.366, Tmi2020 rows 0.551; twins 0.373 and 0.530), cross-token 0.383, fitness 0.824; essentiality AUROC 0.823 released and 0.760 matched under the Costanzo token, identical to three decimals under the SGD token.

Jobs 2863 and 2866 used ONE synthetic token over every source and a paired control. Token run of 2866 (4 epochs): difference 0.79 (trajectory 0.48, 0.88, 0.93, 0.79), sd 0.085, MSE rise 0.119 (the one criterion that passed); control minus token MSE over all rows -0.058 against a floor of +0.0225. The design flaw, not the encoding: the validation triples sit at -0.87 (Kuzmin 2018) and +0.16 (2020) normalized against a training mixture at 0, so one global token bias carries the trigenic sources' residual gap on top of the offset until the encoder absorbs every source mean; and two runs' validation point losses differ by 0.06 from initialization alone, more than the control's floor. Kept: `results/smoke/cgt_030_smoke_tok_000_seed0_job2863_dist_lambda_0.1.json`.
