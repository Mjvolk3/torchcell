# Agent 01: the 019 expression/proteome campaign ledger and its epistemic state

Read-only audit, 2026-09-17. Sources: every `notes/experiments.019-simb-multimodal*.md`,
the config headers of `experiments/019-simb-multimodal/conf/cgt_*.yaml`,
`notes-tex/019-simb-multimodal-expression/sections/*.tex`, the weekly notes
`user.Mjvolk3.torchcell.tasks.weekly.2026.{30,31,35,37}.md`, the memory files `019-*`, and
the committed result JSONs under `experiments/019-simb-multimodal/results/`.

**Three rounds that were run and never read out I read out in this session** from the W&B
full per-epoch history with the campaign's own rule (`roll_max` = max of a centered 5-epoch
rolling mean of `val/expression/pearson_per_feature`, an upward-biased order statistic whose
bias grows with epochs run, and which cancels in a contrast between runs scored the same
way). They are marked **NEW** below and they change three standing conclusions.

---

## 1. Chronological ledger of every expression / proteome round

`pf` = `val/*/pearson_per_feature`. "Scoring rule" is named in every cell because the
campaign's numbers are not comparable without it. Every number in the last two columns is
either from a committed result file, a note, or (marked NEW) computed in this session.

| # | date | project / job | what was varied | n runs | best measured val pf + scoring rule | what the notes CONCLUDED |
|---|---|---|---|--:|---|---|
| 1 | 2026.07.22 | `torchcell_019-simb-multimodal_cgt_multitask`, GH job 1024, `gh_expr_grid_000..287` | hidden {16,32,64} x layers {2,3} x raw/per-gene-zscore x Kemmeren-only/+Sameith x graph_reg {0,1e-3} x 2 hp bundles x 3 seeds, 100 epochs, early stop on `val/loss` patience 20 | 288 planned, 496 runs in project | **0.1184** final-epoch `val/per_gene/pearson_per_gene` (`21d8zyak`, hidden 32, Kemmeren-only, ep 25). NEW: I scanned all 496 run summaries; 0.1184 is the project maximum | Note claims "per-gene Pearson up to 0.48, per-strain up to 0.62 ... expression is NOT intrinsically hard ... capacity reduction 5M -> 116K is the whole story ... target standardization was NOT the lever, retract that bet". **The 0.48/0.62 are the TRAIN metrics** (`train/per_gene/pearson_per_gene` 0.4906, `train/.../pearson_per_strain` 0.6525 on the hidden-64 runs). Never corrected in `experimental-plans.md`. |
| 2 | 2026.07.25 | `_003` decoder x distributional Optuna: GH 1318 (joint), mmli 2312638 (morph), cabbi 2312639 (expr) | decoder {s1_pool, s3_xattn} x dist {point, crps, quantile} x size x lambda; expr arm S0-only | 78 morph + 56 expr trials | expr val median 0.067, train 0.46 (config header of `cgt_embed_005.yaml`) | "expression is at its ~0.09-0.11 noise ceiling, decoder/loss-agnostic" (`cgt_decoder_003.yaml` header). **Ceiling later retracted**, see §3.1. `val/loss` early stopping ended runs at ~22 epochs. |
| 3 | 2026.07.26 | kNN embedding probe, CPU (`knn_embedding_probe.py`) | 17 gene representations, parameter-free neighbor average | n/a | prot_T5 **0.117**, chrom_pathways 0.051, CaLM 0.069, species-LM flanks 0.020/0.036, every NT window 0.005-0.012, random floor 0.033. Swept transformer on the same split: **0.080** | "A parameter-free similarity average beats the transformer; protein yes, DNA no; `one_hot_gene` is undefined, so `learnable_embedding=False`; topology (chrom_pathways 0.100 on morphology) supports widening to nine graphs." |
| 4 | 2026.07.26-27 | `_004` (stopping rule), `_005` embed round (`torchcell_019_embed_v5`) | monitor moved to the ranked metric; node embeddings swept | 81 morph + 56 expr trials read | train 0.66 / val 0.023 morph; dropout 0 -> 0.1 val 0.0215 -> 0.0235; wd x1e4 same; hidden 64 -> 128 val 0.0083 -> 0.0307; graph_reg 0 -> >0 val 0.0139 -> 0.0260 | "Regularization is inert and capacity helps: this is a COLD START problem, not overfitting; attention-to-adjacency is the model's only generalization channel." Optuna marginals, not a controlled contrast. |
| 5 | 2026.07.27 | `_006` nine-graph round, `torchcell_019_expr` | 9 graphs on 9 regularized heads (010's set), hidden {90,180}, L {2,4,6,8} | 133 trials, 0 failures | peak **0.1746** (best-epoch max, not `roll_max`) | "Topology is the channel that matters; 0.109 -> 0.1746." Header also records that the lambda double-application bug made graph_reg effectively OFF in `_005` and earlier, and that `_006`-`_008` silently regularized only 7 of 9 heads (name mismatch). |
| 6 | 2026.07.27-30 | `_007` attribution round, jobs 2317356 (mmli) / 2317357 (cabbi), `torchcell_019_expr_v7` | lr, weight_decay (continuous log-uniform), dropout, dist {crps, quantile, laplace_crps, nll, energy}, energy_rank {0,32}, graph_reg_on {F,T}, graph_reg_layer {[1],[2]}; Halton QMC | ~60 per dist mode | top ~0.1702 (best epoch) | "The five distributional modes are within noise of each other on accuracy (means 0.017-0.028 vs within-mode sd 0.030-0.040) while calibration succeeded. The distributional axis is not the lever." |
| 7 | 2026.07.28-29 | `_008` waves 1-3, `torchcell_019_expr_v8` | decoder families: GEARS cross-gene, bilinear32, Perceiver, response basis, concat, graph propagation h0/h1/h2, free-gene, FiLM | 67 scored runs, mostly 1 seed | all runs 18-300 epochs, `roll_max` | **Voided.** "Not one of 67 runs reached past 300 epochs, against a curve that dips at 200-300 then climbs for thousands. The round contains no valid arm comparison at all." |
| 8 | 2026.07.29 | `_009`/`_010` mask-vs-KL | hard graph attention mask on + `graph_reg_lambda=0` | 8 runs, 1 seed | `D2_mask` 28.0 s/epoch vs 42-49 for the KL arms; memory ~35 vs ~40 GB | Hard masking FROZEN as the standing architecture on **speed, memory and organism transfer only**. Header states plainly: "ACCURACY ... is effectively UNMEASURED" (the scorer resolved `D2_mask` to the string `A0_baseline` and averaged it in). A clean mask-vs-KL A/B "is still owed". |
| 9 | 2026.07.29-30 | wave 4b null sink (`_011`, fixed split) | `N_sink_mm` vs co-launched `R_ref`, seeds 0/1/2/42 | 8 (4 pairs) | paired delta **+0.0024**, sd 0.0090, t(3) 95% CI +/-0.0143, `roll_max` at 300 epochs | "Underpowered, not established null. MDE at 80% power is +0.019 = 7.9x the point estimate; an exact paired sign test floors at p=0.125 at n=4. The fix is epochs, not replicates." |
| 10 | 2026.07.30 | wave 5 (`_011`), H1/H2 | H1 saturation (`H1_ref`/`H1_nodrop`), H2 post-perturbation Perceiver mixing (9 paired reps) | 75 `stage-wave5` runs | `H1_ref` s1 peak 0.2044 @ ep 1597 of 1620; mixing delta -0.0023 +/- 0.0076 | "Training was never converged: all four long runs walltime-killed at 12.6 h still climbing; val peaks at 93-99% of the run. Mixing null, but tested with nothing in the objective requiring it." |
| 11 | 2026.07.30 | **wave 6 pair-rank ladder**, mmli 2324896 etc., `torchcell_019_expr_v8` tag `stage-wave6` | `V_ref` (rank 0), `V_sink` (9), `V_basis16/32/64`, `V_film` (90), `V_hadamard`, `V_hadamard_add`, plus `V_drop2/3`, `V_wd1e4/1e2`. One seed each | 12 | **NEW, computed this session.** All twelve FINISHED at 4,102-4,375 epochs. `roll_max` at the matched budget 4,102: `V_ref` **0.2126**; sink -0.0094; basis16 -0.0054; basis32 -0.0172; **basis64 +0.0177**; film -0.0071; hadamard +0.0039; hadamard_add +0.0061; drop 0.2 -0.0246; drop 0.3 -0.0668; wd 1e-4 -0.0062; wd 1e-2 -0.0084 | **Never read out.** The notes still say these runs are at "105-174 epochs, liveness only" (2026-07-30 snapshot) and the notes-tex says FiLM "has not been [trained at a resolving budget]". See §3.4. |
| 12 | 2026.07.30 | v9 masked-label objective, GH job 1443, `torchcell_019_expr_v9` | 8 mask-schedule arms `M_sched/M_lo/M_hi/M_fine/M_coarse/M_nomix/M_gate_rezero/M_off`, 9,900 epochs, 1 seed each | 8 | `roll_max` 0.1663-0.2382, mean **0.1965 +/- 0.0222**; `M_fine` 0.2382 and its resume lineage 0.2393/0.2430 at 12,229/18,990 | Read as "8 identical-config replicates" until 2026-09-08, then corrected to "8 ARMS". `M_off` (no masked objective) scores 0.2008, inside the spread, so "at k=0 the masked objective has not been shown to help", one draw per arm. |
| 13 | 2026.08.27 | graph prior probe (CPU, `graph_prior_probe.py`) | does graph proximity predict which reporters respond, on all 9 deployed graphs | n/a | AUC 0.4961-0.5057 deployed vs 0.4943-0.5017 degree-rewired; largest excess over control **+0.0046**. Directed: tflink TF->target **0.5508**, regulatory 0.5239; the mask symmetrizes both back to chance | "The graph prior is at chance on all nine. Do NOT build `P_graph`. Free the nine masked heads (`P_free`). Stop symmetrizing the two directed relations." **None of the three was acted on**, see §4. |
| 14 | 2026.08.27 | objective diagnosis (`expression_objective_diagnosis.py`) | loss vs metric on the two longest runs | 2 | `hx8pxdic`: quantile-loss min @ 463, mse min @ 9,175, Pearson max 0.2362 @ 9,674, nmse at peak 1.010, `s/r` 1.95 | "MSE and Pearson improve TOGETHER, late; only the quantile loss turns early. nmse never gets below 1: the model reaches r=0.236 while no better than the mean in squared error. Rescaling by r/s is free and takes nmse to 1-r^2." |
| 15 | 2026.08.31 | objective round, IGB 2368333 / 2368337 / 2368339 | `dist` head at a matched long budget: point, crps, laplace_crps x3 seeds vs the n=8 quantile arms; plus a 19,000-epoch resume | 10 fresh + resume | `roll_max` >= 1,000 epochs: point 6/6 COLLAPSED; crps 4/6 collapsed, live mean 0.1697; laplace_crps 0/6 collapsed, 0.1947; quantile 0.1952 | "The head choice was never load-bearing between quantile and laplace_crps (gap 0.0005 against a detectable 0.042). What the head buys is not collapsing." |
| 16 | 2026.09.06 | metric-aligned Pearson round, cabbi 2378262 / 2378267 / 2378268 | `dist: pearson` (1 - mean per-feature r) and `pearson_mse`; batch 32 packed vs batch 64 solo | 8 | `Q_pearson` b32: all 3 seeds collapse to constant output by epochs 584-680 (peaks 0.133-0.145 at ep 150-240). `Q_pearson_mse` b32: 2 of 3 collapse; survivor `ppc2pyv5` 0.1936 @ 3,193. **`Q_pearson_b64` solo: both survive, 0.1942 and 0.1856**, peaks at 1,926/2,419 | "Training directly on the metric does not beat the quantile head at any budget measured; at batch 32 it is unstable in 5 of 6. Batch size and solo placement are confounded." |
| 17 | 2026.09.06-08 | **v10 factorial**, Delta 21830323, `torchcell_019_expr_v10` | 2^4: embedding {random_1024, prot_T5_all} x trunk {L6h90, L2h45} x readout {MLP, linear} x wd {1e-8, 1e-4}, 2 seeds, matched epochs <= 990 | 32 | `roll_max` main effects: **embedding +0.0682 (t 7.8)**; trunk -0.0120 (t -1.4); readout +0.0072; wd +0.0046. Pooled within-cell sd 0.0246 (0.0122 without one stuck run) | "Embedding CONTENT is the only large factor. Trunk, readout, weight decay null at ~0.02." **Readout row retracted 2026-09-10** (`linear_readout` was never wired into `PerGeneHead`). Neither embedding level is the incumbent's `calm`, so calm-vs-ProtT5 stayed unmeasured. |
| 18 | 2026.09.06-08 | mechanism round, IGB 2369697 / 2371531 | `R_ref`, `R_basis64`, `R_pergene` (GEARS row), `R_pergene_basis64`; 2 seeds; matched epochs <= 4,079 | 8 | `R_ref` 0.1881/0.1693. Paired vs ref: basis64 -0.0135/+0.0303; **pergene +0.0007/+0.0544**; pergene_basis64 +0.0255/+0.0292 | "Pair term alone is a null with sign-disagreeing pairs. Per-gene readout +0.027 against a design resolution of ~0.06: a direction, not a result. Per-gene arms peak early (1,200-2,600) and give part back by 8,500." All four tasks died OOM at 61 GB host RSS. |
| 19 | 2026.09.08-11 | ListMLE round, cabbi 2385807 / gpu 2385808 | Plackett-Luce ranking loss, pure and MSE-anchored, batch 32 and 64 | 6 | pure b32 `roll_max` 0.1770 / 0.1499; b64 0.1529 / 0.1608; both `listmle_mse` collapsed to the mean by epoch ~250 and were killed | "Every survivor is below the incumbent band at 6,000 (0.1917 +/- 0.0177). The MSE anchor kills it; the pure loss learns slowly. lr 3e-4 inherited from pinball may be low for the ranking gradient (hypothesis)." |
| 20 | 2026.09.09-11 | **per-gene replicate stage**, cabbi 2389901 + 2392370, tag `round-pergene-rep` | `RR_ref` vs `RR_pergene`, seeds 2/3/4, 1,000 epochs, one pair per card | 6 | **NEW, computed this session.** `roll_max` at 999: ref 0.1591/0.1690/0.1827; pergene 0.2343/0.1933/0.1994. Paired **+0.0751 / +0.0243 / +0.0167, mean +0.0387, sd 0.0318, 3 of 3 positive** | **Never read out.** The weekly note records the runs completing and nothing else. |
| 21 | 2026.09.10-12 | **v11 input-richness round**, Delta 21948711, `torchcell_019_expr_v11` | `E_calm` / `E_ptt5` / `E_calm_ptt5` / `E_full` = [fudt_upstream, calm, prot_T5_all, fudt_downstream], 3 seeds, 1,400 epochs | 12 | **NEW, computed this session.** `roll_max` at matched 1,396: E_calm 0.0242/0.1765/0.1631 (seed 0 is a plateau run); E_ptt5 0.1512/0.1500/0.1321; E_calm_ptt5 0.1723/0.1662/0.1862; E_full 0.1769/0.1596/0.1824. Paired vs E_calm on the **2 clean seeds**: ptt5 **-0.0288**, calm_ptt5 **+0.0064**, **E_full +0.0012** | **Never read out.** The weekly checkbox "When 21948711 lands: v11 readout script" is still open. v12, v13, v14, v15, v16 all pin `E_full` on the strength of this unread round. See §3.2. |
| 22 | 2026.09.10-13 | v12 readout round, IGB 2392379-2392386, `torchcell_019_expr_v12` | 8 readouts on `E_full` under pinball, 4 seeds, 1,400 epochs, 4 per card, seed-major | 32 | `roll_max` at 1,399, paired vs `H_ref` (0.1740 +/- 0.0076): **H_concat +0.0135 (sd 0.0026, 4/4)**; H_linear +0.0024; H_state -0.0007; H_pergene **-0.0085**; basis64 -0.0209; pergene_basis64 -0.0254; gears -0.0396. At 500 epochs the same contrasts read H_pergene **+0.0319 (4/4)** and H_state **+0.0356 (4/4)** | "H_concat is the one arm resolved over H_ref; adopt it for the next stack." 3 of 32 runs never left the constant plateau, all basis or cross-gene forms. |
| 23 | 2026.09.13- | v13 split round, IGB 2397311, `torchcell_019_expr_v13` | split seeds 0-3 x {H_ref, H_concat} x 2 seeds, plus split 0 at 90/10; 6,000 epochs | 24 | `roll_max` at matched epoch 3,754: split means **0.185 / 0.156 / 0.127 / 0.132** at the earlier read, 0.193/0.164/0.134/0.137 at 2,859; best run 0.2127 (`8i75d8h1`, split 0). **H_concat minus H_ref +0.0056, sd 0.0197, t 0.94, 5 of 11 clean pairs positive.** 90/10 minus 80/10/10 +0.0615 over 4 pairs but +0.027 over the 3 clean pairs | "The partition moves the score by five times the within-partition spread. **The v12 gain of +0.0135 does not replicate across partitions.**" `test_at_best_val` is `None` for all 24 in the committed readout: still no test number. |
| 24 | 2026.09.15- | v14 proteome round, cabbi 2400350, `torchcell_019_prot_v14` | v13's design on Messner 2023 proteome (log2 to HIS3, 1,850-protein union with NaN), 2,000 epochs | 16 | `roll_max` at matched 814: splits **0.114 / 0.083 / 0.112 / 0.085**, within-split sd 0.0045. **P_concat minus P_ref -0.0002 (sd 0.0058, t -0.09, 4/8 positive).** Curves peak at epochs 66-646 (median ~150) and decline | "The partition axis is as large on the proteome. Concat is nothing. The proteome has no slow-feature phase: val Pearson peaks by epoch 100 (mean 0.086) and holds 0.078 to epoch 1,000 while train climbs to 0.61. v14 at 19-24% of the 0.42 genotype ceiling, the same fraction v13 holds of 0.775." |
| 25 | 2026.09.15- | v15 weight-decay round, queued 2401479, `torchcell_019_expr_v15` | wd 1e-8 vs 1e-2 vs 1e-1, split seeds 1-2, 2 seeds, 6,000 epochs | 12 planned; **4 in the committed readout** | `roll_max` at matched 444: ref 0.0918/0.1094; wd1e1 0.0933 (+0.0015 on one pair); wd1e2 0.1103 (+0.0185 on one pair) | Expectation stated before the run: "no gain". Partial, one pair per contrast. |
| 26 | 2026.09.17 | v16 joint round | proteome head + auxiliary expression head, splits 0-2, 2 seeds, 500 epochs | 0 (built, not launched) | n/a | "Hypothesis (untested): the auxiliary label changes the proteome score by less than the within-partition sd of 0.005 at 500 epochs." |

### The three new readouts, with run links

Wave-6 pair-rank ladder (round 11), reference and the two extremes:

https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/sm6efleg

https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/fqk5mbr3

https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/1j0tn0p7

https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/b50f93ju

Per-gene replicate stage (round 20), the seed-2 pair:

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/ob6emcr9

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/1syxbz46

v11 input-richness round (round 21), the E_calm plateau run and the E_full arm:

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/crcbyxju

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/9zgjaaka

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/t8n7rmax

---

## 2. Every standing conclusion, classified

Verdicts: **MEASURED** (effect resolved against its own noise floor), **DIRECTIONAL**
(consistent sign, effect below the design's resolution), **ASSUMED** (never run, run once,
or run only on seed 0 / one partition), **REFUTED**.

### 2.1 Things the campaign currently stands on

| claim | verdict | evidence, effect size, noise floor |
|---|---|---|
| "Per-gene head helps" | **DIRECTIONAL, and now self-contradictory** | Three separate reads. Mechanism round (2 pairs, 4,079 ep, calm input): +0.0007 / +0.0544. Per-gene replicate stage (**NEW**, 3 pairs, 999 ep, calm input): +0.0751 / +0.0243 / +0.0167. Pooled over those 5 pairs on the calm stack: **mean +0.034, sd 0.030, t(4) = 2.5, 5 of 5 positive** (exact sign test p = 0.0625). But v12 on the **E_full** stack at 1,399 epochs, 4 paired seeds: **-0.0085** (2 of 4 positive); the same v12 pairs at 500 epochs read **+0.0319, 4 of 4**. So the mechanism's gain is real early and on the calm input and is gone late and on E_full. Nobody has reconciled the two. The `+0.041 on seed 0` figure in the shared context matches nothing in v12; the nearest real number is the NEW +0.0387 over 3 pairs. |
| "Embedding content helps (+0.068)" | **MEASURED, but it does not say what it is being used to say** | v10, 32 runs, matched <= 990 epochs, t 7.8 against a pooled within-cell sd of 0.0246. The contrast is **ProtT5 against width-matched RANDOM**, so it establishes "a real gene representation beats noise", not "richer is better". Within real representations the effect vanishes: v11 (**NEW**) E_full minus E_calm +0.0012 and E_ptt5 minus E_calm **-0.0288** on the two clean seeds; the linear study over every representation the builder serves finds ESM2 level with ProtT5, every composite within 0.005 of its best member, and the four-embedding stack the model consumes not better than ProtT5 alone under B2 or B3 on either panel. |
| "Pinball beats MSE" | **MEASURED, in the weak form only** | The point/MSE head collapses to a constant 6 of 6 times at a long budget; crps 4 of 6; quantile and laplace_crps 0 of 6. Between the two that survive the gap is 0.0005 against a detectable 0.042. The campaign's own wording is the honest one: "what the head buys is not collapsing", not "pinball is better". |
| "Batch 64 collapses" | **REFUTED, and the shared context has it backwards** | Measured: batch **32** pure-Pearson collapsed at 3 of 3 seeds (epochs 584-680) and MSE-anchored at 2 of 3; both solo **batch-64** runs survived 9,899 epochs at 0.1942 / 0.1856, on the incumbent band. Batch size and solo placement are confounded in that arm, so "why b64 survives" is unmeasured. Nothing in the campaign has measured batch 64 under the pinball loss at all. |
| "Graph reg / the nine-graph mask does X" | **ASSUMED on accuracy, MEASURED on speed, and the underlying prior is REFUTED** | Hard masking was frozen as a standing architecture decision on a measured 1.51x-1.74x speedup and a memory drop; the `_010` header states outright that accuracy is "effectively UNMEASURED" because the scorer folded `D2_mask` into `A0_baseline`, and that "a clean mask-vs-KL A/B at 4 paired seeds is still owed". It has not been run. Meanwhile the probe the mask rests on came back at chance: AUC 0.4961-0.5057 on all nine graphs, largest excess over the degree control +0.0046, and three graphs are silent on 25-71% of deleted genes. The one place signal exists (tflink TF->target 0.5508) is destroyed by the symmetric mask builder. `_006`-`_008` also silently regularized 7 of 9 heads. |
| "Response basis rank 32" | **ASSUMED as a modeling choice; MEASURED only as a property of the DATA** | Measured: residual gene-gene correlation replicates split-half at r = 0.8687 against a permutation null of 8.45e-05, participation-ratio effective rank **32.78**; rank-r output ceilings 0.7265 (r=32), 0.7799 (r=64) against the 0.775 label ceiling. Those are data facts and they are solid. That a rank-32 or rank-64 **head** helps is not measured anywhere: wave 6 (**NEW**) reads basis16 -0.005, basis32 -0.017, basis64 +0.018 at n=1; the mechanism round reads basis64 -0.014/+0.030 at n=2; v12 reads basis64 -0.021 with one of its four seeds a plateau run. |
| "Seed 0 / split 0 is inflated" | **MEASURED, repeatedly and by three independent routes** | (a) before the pin, between-seed sd 0.0444 vs between-arm 0.0058, seed 0 **+0.0893** above the others over 7 fully crossed blocks, Friedman p = 1.7e-4, while fitting TRAIN worse; best non-seed-0 score in 67 usable runs 0.0901. (b) v13, 24 runs: split means 0.193 / 0.164 / 0.134 / 0.137, between-partition sd 0.027 against a pooled within sd 0.009-0.012. (c) the linear baselines with no seed of their own: B2 on ProtT5 over twelve partitions 0.104 +/- 0.018, split 0 the highest validation draw. Also measured NOT to be responsiveness composition (val responsive fraction moves 2.9 points across seeds whose score moves 6x). |
| "Long budgets keep rising" | **DIRECTIONAL, and it is three different claims fused into one** | Measured: the eight v9 arms peak at epochs 2,199 / 3,398 / 3,503 / 3,715 / 4,109 / 8,859 / 9,691 / 9,697, so only 3 of 8 peak late; the mean gain from 1,000 to 9,900 epochs is 0.1609 -> 0.1965 (+0.036) at a spread that grows 0.0099 -> 0.0222. The 19,000-epoch resume bought ~0.005. v13 is still rising at 3,754 epochs. But the proteome does the opposite: v14 peaks by epoch ~100 and declines, and the per-gene readout arms peak at 1,200-2,600 and give part back. So "longer is better" holds for the expression incumbent and is refuted for the proteome and for the per-gene head. |
| "The masked-label objective is worth having" | **MEASURED as imputation, ASSUMED as a genotype-to-expression lever, and with one draw suggesting it costs nothing and buys nothing at k=0** | Oracle: within-study 0.42 / 0.69 / 0.80 at m = 10/100/1000, cross-study 0.22 / 0.37 / 0.48, and **97.5 / 99.2 / 100.6%** of the gain survives removing a kNN genotype predictor, so the channel is orthogonal to the one being scored. At k=0, `M_off` (no schedule at all) scored 0.2008 inside the eight-arm spread: one draw, so a hypothesis. Every round from v9 onward carries the masked machinery and is scored at k=0 where, by this evidence, it does nothing. |
| "H_concat (State SE form) is the readout to adopt" | **MEASURED on split 0, REFUTED across partitions** | v12: +0.0135, sd 0.0026, 4 of 4 seeds, at 1,399 epochs on split 0. v13 at 3,754 epochs over four partitions: **+0.0056, sd 0.0197, t 0.94, 5 of 11 clean pairs positive**; on split 0 itself two of three clean pairs are negative. v14 proteome: -0.0002 over 8 pairs. The adoption decision rests on the one partition that is measured to be the easiest draw. |
| "The perturbation operator has no pair term at \|S\|=1" | **MEASURED, and it is the strongest structural fact in the campaign** | `perturbation_selector_degeneracy.json`: attention weights {1.0}, across-query spread 3.31e-09, output change under re-drawing W_Q/W_K at std 10 exactly 0.0, 16,200 of 32,760 attention weights dead; 95.4% of the build is \|S\|=1. `h_CLS` across-strain sd 0.0 vs 0.973 for `z_S`. Independent of any training run. |
| "The expression label ceiling is 0.775" | **MEASURED** | 82 deletions profiled in both Kemmeren and Sameith; per-gene test-retest r 0.611, ceiling 0.775 by mean-sqrt-rho, 0.862 by the variance decomposition. Note the campaign scores against 0.775 while the cross-study oracle caps a transferable imputation predictor at ~0.48 and cross-study agreement at 0.611. |
| "The model beats the linear baseline by 0.09 (4.2 sigma)" | **REFUTED as stated; the honest number is 0.04-0.05, on the easiest partition** | The 0.093 came from B2 = 0.1040 on a different permutation. On the partitions the model actually trains on, B2 on ProtT5 reads 0.135 on split 0 and the v12 stack reads 0.174-0.188, a margin of 0.04-0.05. On splits 2 and 3 the trained arms lead B3 by **0.01 to 0.02**. On the proteome the margin is 0.00 to 0.04. |
| "nmse never beats the mean" | **MEASURED** | Best `nmse` at the Pearson peak across every live run is 1.005; at the peak `s/r` is 1.95 (v9) and 2.21 (v8). The free rescale by r/s takes nmse to 1-r^2 and has never been applied to any checkpoint. |
| "The proteome is a target of its own" | **MEASURED** | Kemmeren vs Messner per strain 0.036 and per protein 0.075 over 1,350 shared deletions; gene co-variation Spearman 0.31; Zelezniak reproduces Messner at 0.08 per deletion while Messner reproduces itself at 0.80 on the same 87 strains. Proteome ceilings: 0.61 (HIS3 replicate), 0.42 (149 duplicate strains), 0.29 (cross-study). |

### 2.2 Things that are load-bearing and were never measured at all

- **calm vs ProtT5 at the incumbent configuration.** Named as "the cheapest contrast on the table" in `3-next.tex` on 2026-09-09. v11 ran it and was never read; my read gives E_ptt5 minus E_calm = -0.0288 over two clean seeds, i.e. the wrong sign for the campaign's story, and underpowered.
- **A mask-on vs mask-off (or mask vs KL) accuracy ablation.** Owed since `_010`.
- **`P_free`** (free the nine masked heads) and the directed-mask fix. Recommended 2026-08-27, never implemented (`grep -rn "P_free"` returns nothing but this note).
- **A width-matched random filler arm** to separate E_full's 5.85M preprocessor parameters from its content. Named in the v11 config header as the control "if E_full leads".
- **Any held-out TEST number for expression.** `test_at_best_val` is `None` for all 24 v13 runs and all 16 v14 runs in the committed readouts.
- **The weight-decay ladder as designed.** v15 was designed as 12 runs (3 arms x 2 split seeds x 2 init seeds); the W&B project holds **4**, and the committed readout resolves one pair per contrast at epoch 444 of 6,000. No note records what happened to the other eight. Strong L2 at 1e-2 and 1e-1 therefore remains untested at the intended power, exactly as it was before the round.

---

## 3. Contradictions

**3.1 The expression noise ceiling was 0.09-0.11, then 0.775, and the early rounds' design
still carries the old one.** `cgt_decoder_003.yaml`, `decoder-distributional-plan.md` and the
`_003` success criteria all read "expression is at its ~0.09-0.11 ceiling, so it is noise
saturated and decoder/loss agnostic" and used that to justify spending the round on
morphology. The replicate-based ceiling is 0.775 and the old one was measured to be invalid
(an observed 0.109 was 118% of it; its median was 0.0000 with IQR [0,0]; the reported SE
overstated noise ~10x in variance). The retraction lives in memory `019-expr-sweep-006-007`;
`decoder-distributional-plan.md` still says 0.09-0.11 with no correction block.

**3.2 The E_full input was adopted on a round that was never read, and reading it does not
support the adoption.** v12's header says "v11 has not finished ... so E_full is not yet
measured against calm; every contrast in this round is within E_full and stands either way".
That was true at submission. v11 then finished (all 12 runs at 1,399 epochs, 2026-09-10/11),
the weekly checkbox for its readout is still open, and v13, v14, v15 and v16 inherit E_full
through v12. My read: E_full minus E_calm = **+0.0012** over the two clean seeds, E_ptt5
minus E_calm = **-0.0288**. Separately, the linear study says the four-embedding stack is not
a better perturbation representation than ProtT5 alone under either rule on either panel, and
the two species-LM flanks in that stack sit at the random floor by themselves. So the input
the whole current stack runs on has no measured advantage over the single embedding it
replaced, and it costs 16x the preprocessor parameters.

**3.3 The per-gene readout is positive at 5 of 5 pairs on calm and negative on E_full.**
Mechanism round +0.0007/+0.0544 (4,079 ep), per-gene replicate stage **NEW**
+0.0751/+0.0243/+0.0167 (999 ep), v12 H_pergene -0.0085 at 1,399 ep but +0.0319 (4/4) at 500
ep. Two candidate explanations, neither measured: the input stack differs, or the arm's gain
is an early-training effect that the longer budget erases. The campaign dropped the per-gene
readout on the v12 number alone.

**3.4 "The decoder families have never been trained at a resolving budget" is half wrong.**
`0-strand.tex` says "the GEARS-style cross-gene readout, the bilinear term, Perceiver mixing
on its own and FiLM have not been [trained at a resolving budget]", and the round
retrospective describes wave 6 as "105-174 epochs ... a liveness check only". In fact all
twelve wave-6 runs finished at 4,102-4,375 epochs and were never re-read. They are n=1 per
arm, so "not resolving" is right; "not trained" is not. The measured ladder is flat: rank 0 =
0.2126, rank 9 = 0.2033, rank 16 = 0.2072, rank 32 = 0.1954, rank 64 = 0.2304, rank 90 (FiLM)
= 0.2055, rank 90 (Hadamard) = 0.2165. The falsifiable prediction stated in
`multiplicative-perturbation-conditioning.md` ("rank 9 should reproduce the null sink's null
and rank 90 should not") is **not supported** by the one draw available.

**3.5 The wave-6 regularization recommendation is contradicted by the wave-6 runs.** The
design note concluded from paired 1,600-epoch runs that "dropout goes up, not down: arms at
{0.1, 0.2, 0.3}". The arms ran: at 4,102 epochs, dropout 0.2 reads -0.0246 and dropout 0.3
reads -0.0668 against the dropout-0.1 reference, and `V_drop3` ends at numerically zero. One
seed, so not resolved, but the direction is the opposite of the recommendation and it was
never recorded.

**3.6 "Capacity reduction is the whole story" and "per-gene 0.48 is a strong honest number"
were train metrics.** The best validation per-gene Pearson in the entire 496-run first project
is 0.1184. The top runs' TRAIN metrics are 0.4906 per-gene and 0.6525 per-strain, which is
where 0.48 and 0.62 came from. `experimental-plans.md` still carries the claim, and it is the
origin of the "expression is not intrinsically hard" framing.

**3.7 The graph channel is simultaneously "the channel that matters" and at chance.**
`cgt_expr_006.yaml`: "topology is the channel that matters, and 9 graphs on 9 regularized
heads is the configuration that carries it" (0.109 -> 0.1746). `graph_prior_probe.json`: the
proximity prior the mask encodes is at chance on all nine graphs. Both cannot be describing
the same mechanism. The `_006` gain is also confounded: that round changed graph count,
attention-head count, the lambda double-application bug, the per-graph normalization, and
`require_modalities` in one step, and its lambda was measured to have been effectively off
in `_005` and earlier.

**3.8 The shared context carries three inverted or misattributed numbers.** "batch-64 Pearson
collapsed" (it was batch 32 that collapsed; batch 64 survived); "v12 was a readout round
(per-gene head +0.041 on seed 0)" (v12's per-gene at seed 0 and matched budget is -0.0086;
+0.0387 is the unread per-gene replicate round over 3 pairs); "v11 an embedding round
(embedding content +0.068)" (+0.068 is v10's ProtT5-vs-random effect; v11 was never read).

**3.9 "The masked objective is mandatory because mixing needs something requiring it" versus
`M_off` at 0.2008.** The v9 design argues mixing was previously null only because nothing in
the objective required it, and pairs the two. The one arm that isolates the objective from the
parameters, `M_off`, lands inside the eight-arm spread. One draw, but it is the only direct
evidence and it points at "the objective is not paying for itself at k=0".

---

## 4. Tried and abandoned: do not re-run these

1. **Point / MSE head at a long budget.** Collapses to a per-gene constant 6 of 6 times.
2. **Gaussian CRPS head.** Collapses 4 of 6 at a long budget; live mean 0.1697, below quantile.
3. **`nll` and `energy` heads** (`_007`). `nll` was the designed sigma-collapse control; energy's covariance is global and cannot move a point metric.
4. **Training directly on Pearson at batch 32** (pure or MSE-anchored at weight 1.0). Constant-output collapse at 5 of 6 runs; the loss does not see it because it drops constant columns. The MSE anchor is not a reliable guard.
5. **ListMLE, pure and anchored.** Anchored collapses to the mean by epoch ~250 at both seeds (the MSE gradient is ~50x the ranking gradient at init). Pure ListMLE survives but lands 0.03-0.04 below the band at matched epochs. An lr arm was proposed and never run.
6. **`per-gene z-scored target` as an anti-collapse lever.** Retracted 2026-07-22 (top runs used raw log2 ratio). Note that later rounds re-introduced it as a standing default (`standardize_per_feature_target: [per_gene]`) without re-testing it.
7. **`learnable_embedding` / free per-gene table.** Pinned off since `_006`: only 4.8% of validation genes are ever perturbed in training, and `one_hot_gene` is geometrically undefined in the kNN probe (all cosines exactly 0).
8. **Graph propagation arms (`A2_prop`, `A6_prop_sparse`).** Killed on the organism-transfer criterion (they need curated yeast graphs as input features, not as attention structure), and their numbers were non-measurements anyway (<= 300 epochs, and `A2_prop_sparse`'s "+0.0231 best result" was a cross-wave comparison; the in-wave figure is +0.0030 +/- 0.0321).
9. **Zeroing the deleted gene's embedding.** Argued structurally (it changes 1 of 6,607 predictions after the transform, and before the transform it only moves the gene-independent context) and measured at n=1: -0.0010.
10. **The `P_graph` campaign arm** (pair term as a function of network distance). Dropped on the graph-prior probe.
11. **Nadal-Ribelles single-cell as a knockout-expression target.** Cell-to-genotype assignment ~40% pure; cross-batch self-replication 0.043 vs 0.022 null; the deleted gene cannot be read off its own profile.
12. **Packing four or two long runs per card at 60 GB host RSS on IGB.** 4 of 4 packed five-day tasks died OOM at 61 GB after 3.5-4.6 days. Cause still unidentified.
13. **`nt_window_three_prime_5979`** and every Nucleotide Transformer window: at or below the random floor on the deletion side in the probe and in the linear study.

Also: **do not re-run the v11 embedding round, the wave-6 ladder, or the per-gene replicate
stage.** They exist, they are complete, and this document reads them out.

---

## 5. The five most consequential things the campaign has never tested

1. **Whether the nine-graph attention mask helps accuracy at all.** It is in every config
   from `_010` to v16, it was frozen on speed with accuracy explicitly unmeasured, and the
   data-side prior it encodes is at chance on all nine graphs with the one directional signal
   symmetrized away. A mask-on / mask-off / directed-mask contrast at 3 paired seeds is cheap
   and could remove nine constrained heads from the model or justify keeping them. Nothing in
   the campaign's 145-run parameter sweep or its factorials touches this axis.
2. **Whether the perturbation belongs inside the encoder.** Measured structural fact: the
   encoder runs once on the wild-type graph, `h_CLS` is byte-identical across strains, and the
   deletion enters through a one-key cross-attention whose pair term is rank 0. Every arm ever
   run is a different way of applying the perturbation **after** the encoder, and the wave-6
   ladder (**NEW**) says that whole family is flat from rank 0 to rank 90. The three designs
   written down in `3-next.tex` (perturb CLS, edit the token before the encoder, typed
   perturbation tokens) have never been implemented, and the second is the only one that lets
   the graphs carry the perturbation rather than only shaping the wild-type state.
3. **Whether the model beats the linear baseline on a held-out TEST set on any partition.**
   Every headline is a validation `roll_max`, an order statistic selected over thousands of
   epochs, on the partition measured to be the easiest of twelve draws. `test_at_best_val` is
   `None` for all 24 v13 and all 16 v14 runs. On validation, splits 2 and 3 already show the
   trained arms leading B3 by only 0.01-0.02. The campaign has no number that would survive a
   reviewer asking for a test score.
4. **Whether the per-gene metric is measuring the thing that matters.** Measured and then
   dropped: the bottom half of genes by across-strain variance carry 15-17% of the variance
   while contributing half the terms of an unweighted mean over ~6,127 genes, and the top 1%
   carry 15-17%. No round has ever scored a variance-stratified or reliability-weighted
   Pearson, and `variance_stratified_pearson.md` is an empty note. A gain concentrated in the
   responsive genes and a gain spread over the flat ones are indistinguishable in every number
   the campaign has produced.
5. **Whether 1,244 training strains is the binding constraint.** The 90/10 arm is the only
   probe and it is three clean pairs at +0.027 (the linear baselines move +0.002 to +0.011 for
   the same 10%). Nobody has run a training-set-size ladder (for example 25 / 50 / 75 / 100%
   of train at a fixed budget and partition), which is the one measurement that separates
   "the architecture is wrong" from "there is not enough data", and which would price every
   remaining architecture round. Related and also untested: whether the 4,476-strain proteome,
   which has 2.9x the strains, shows a different data-size slope.

Honorable mentions, each untested and cheap: the free r/s rescale on an existing checkpoint
(changes no correlation, takes nmse from 1.010 to 0.944); a learning-rate arm for ListMLE;
the width-matched random filler arm for E_full; and the 6 near-constant CalMorph features
that two configs disagree about.

---

## Appendix: every W&B project, counted

Pulled this session (`wandb.Api`, entity `zhao-group`). `best_final` is the best
**final-epoch** value of `val/*/pearson_per_feature` in the project, which is deliberately
NOT the `roll_max` the rounds are scored on; it is here only to place each project.

| project | runs | best final-epoch pf | max epoch reached | what it is |
|---|--:|--:|--:|---|
| `torchcell_019-simb-multimodal_cgt_multitask` | 496 | 0.1184 | 26 | the 288-cell sniff grid, round 1 |
| `torchcell_019_expr_v2` | 120 | 0.0999 | 49 | pre-campaign |
| `torchcell_019_expr_v3` | 60 | 0.0792 | 29 | `_003` decoder x distributional, expr control |
| `torchcell_019_expr_v5` | 35 | 0.0532 | 151 | `_005` node-embedding round |
| `torchcell_019_expr_v6` | 158 | 0.1562 | 199 | `_006` nine-graph round (peak 0.1746) |
| `torchcell_019_expr_v7` | 295 | 0.1567 | 199 | `_007` attribution round (peak 0.1702) |
| `torchcell_019_expr` | 26 | 0.0687 | 42 | stray runs under the `_006`/`_007` project name |
| `torchcell_019_expr_v8` | 163 | 0.2231 | 4,375 | `_008`-`_012`: waves 1-6, incl. the unread wave-6 ladder |
| `torchcell_019_expr_v9` | 81 | 0.2342 | 18,999 | masked objective, objective round, Pearson, ListMLE, per-gene replicates |
| `torchcell_019_expr_v10` | 68 | 0.1621 | 1,399 | the 2^4 factorial |
| `torchcell_019_expr_v11` | 12 | 0.1698 | 1,399 | input richness, **never read out** |
| `torchcell_019_expr_v12` | 42 | 0.2072 | 1,399 | readout round (32 scored + 10 smoke runs) |
| `torchcell_019_expr_v13` | 24 | 0.2429 | 4,117 | split round, running |
| `torchcell_019_prot_v14` | 16 | 0.0962 | 1,185 | proteome round, running |
| `torchcell_019_expr_v15` | **4** | 0.0952 | 451 | weight-decay round: **12 runs were designed and 4 exist**; the committed readout carries one pair per contrast. Whether the other 8 were never launched or died is not recorded anywhere I found |

`torchcell_019_expr_v4` does not exist.

## Summary (600 words)

The campaign is 26 rounds deep and its epistemic state is worse than its record suggests, for
three reasons: several rounds were run and never read, several conclusions rest on the single
validation draw that is measured to be the easiest of twelve, and a few headline numbers are
train metrics or inverted in the shared context.

**Three completed rounds had never been read out.** I read them from W&B this session, with
the campaign's own `roll_max` rule at a matched budget. (a) The **v11 input-richness round**
(Delta 21948711, 12 runs, finished 1,399 epochs) says `E_full` minus `E_calm` is **+0.0012**
over the two clean seeds and `E_ptt5` minus `E_calm` is **-0.0288**. Every round since v12,
including v13, v14, v15 and the queued v16, pins `E_full` as the input on the strength of that
unread round. It has no measured advantage over the single `calm` embedding it replaced and it
costs 16x the preprocessor parameters. The linear embedding study independently says the same:
the four-embedding stack is not a better perturbation representation than ProtT5 alone, and its
two species-LM flanks sit at the random floor. (b) The **wave-6 pair-rank ladder** (12 arms,
one seed, 4,102-4,375 epochs) is flat: rank 0 = 0.2126, rank 9 = 0.2033, rank 32 = 0.1954,
rank 64 = 0.2304, rank 90 FiLM = 0.2055, rank 90 Hadamard = 0.2165. The notes describe these
runs as a 174-epoch liveness check and say FiLM was never trained at a long budget; it was.
At n=1 nothing is resolved, but there is no rank response to see, which is the falsification
test the multiplicative-conditioning note itself proposed. (c) The **per-gene replicate stage**
(3 pairs, 999 epochs) gives per-gene minus reference = **+0.039, 3 of 3 positive**.

**That third number creates the sharpest live contradiction.** Pooled with the mechanism round,
the per-gene readout is positive at 5 of 5 pairs on the `calm` stack (mean +0.034, sd 0.030),
while v12 on `E_full` reads -0.0085 at 1,399 epochs and +0.0319 at 500. The campaign dropped
the per-gene readout on the v12 number alone, and adopted `H_concat` on a +0.0135 that v13 then
failed to replicate across partitions (+0.0056, t 0.94, 5 of 11 pairs positive) and v14 read as
exactly zero on the proteome.

**What is genuinely measured** is a short list: the operator has no pair term at one deletion
(exact, structural); the label ceiling is 0.775 and the proteome's is 0.40-0.70; split 0 is
inflated by about +0.09 and the partition moves the score five times the replicate spread;
the point and CRPS heads collapse and quantile does not; embedding content beats width-matched
random by +0.068; no run beats the per-gene mean on squared error; and the masked-conditioning
signal is measured to be 97-100% orthogonal to the genotype channel the model is scored on.

**What is assumed** is more load-bearing: that the nine-graph mask helps accuracy (frozen on
speed, accuracy explicitly unmeasured, and the prior it encodes is at chance on all nine
graphs with the one directional signal symmetrized away); that a richer input helps; that the
per-gene head does not; that long budgets keep paying (3 of 8 arms peak late, the proteome
peaks by epoch 100). And there is still **no held-out test number** on either panel.

The five things never tested, in priority order: the mask ablation; the perturbation inside
the encoder; a test score; a variance-stratified metric; and a training-set-size ladder, which
is the only measurement that separates "the architecture is wrong" from "1,244 strains is not
enough" and should price every remaining architecture round.

Full path to this report:

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/agent-01-ledger.md
