# Shared brief for the joint proteome + expression review (2026-09-27)

## The question

Can joint training of the cell graph transformer (CGT) on the knockout PROTEOME (Messner
2023, `fig3_proteome` partition, log2 strain/HIS3 over ~1,850 proteins, NaN where not
quantified) and the knockout EXPRESSION (Kemmeren 2014 / Sameith 2015 style deletion
transcriptomes, `expression_log2_ratio`) be shown to be SYNERGISTIC for each label's OWN
prediction: joint >= single-head on the proteome head AND on the expression head, with a
pre-registered paired statistic, on held-out strains? The principal investigator wants a
provable statement within TWO WEEKS. Compute: IGB `gpu` partition (A40 48 GB, at most THREE
of our jobs at once, `--array=..%3`, walltime unlimited) and IGB `cabbi` (RTX 6000 Ada
48 GB, one to three cards free at a time, EVERY job <= 5 days). Nothing on `mmli`.

## Where things are

- Repo worktree (read from here, NEVER edit repo files, NEVER run git):
  `/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective`
- Python: `~/miniconda3/envs/torchcell/bin/python` (W&B API works; `wandb.Api()`, entity
  `zhao-group`). Run from the worktree root. Reading W&B and local files is fine; do NOT
  launch training, do NOT touch IGB (no ssh), do NOT add anything to Zotero.
- Experiment dir: `experiments/019-simb-multimodal/` (conf/, scripts/, results/).
  Training script: `scripts/train_cgt_multitask.py`. Arm definitions:
  `scripts/gh_expr_008_arm.sh`. Launcher: `scripts/igb_expr_wave5.slurm`. Readout:
  `scripts/v13_split_readout.py` (`--round v13|v14|v15|v16|v16_expr|v17|v18`, JSON in
  results/). Views: `scripts/wandb_v13_report.py`.
- Notes-tex documents (LaTeX, `sections/*.tex`): `notes-tex/019-simb-multimodal/`,
  `notes-tex/019-simb-multimodal-expression/`, `notes-tex/028-knockout-expression-metabolic/`.
- Dendron notes: `notes/experiments.019-simb-multimodal.*.md` (retrospectives, plans, the
  expression-fit review, proteome-expression EDA, conf notes), `notes/user.Mjvolk3.torchcell.tasks.weekly.2026.3[5-7].md`.
- W&B projects: `torchcell_019_prot_v14` (proteome split round), `torchcell_019_prot_v16`
  (JOINT round: arms J_ref proteome-only, J_expr expression-only, J_joint both; 500 epochs
  done; a continuation to 1,200 epochs is running as NEW runs whose config
  `wandb.resumed_from` names the source run), `torchcell_019_expr_v13` (expression split
  round), `_v15` (weight decay), `_v17` (perturbation locality), `_v18` (mask off, context
  row), older `_v7`..`_v12`, `torchcell_019_expr_morph*`, `torchcell_019_morph*`.
  Metric keys: `val/proteome/pearson_per_feature`, `val/expression/pearson_per_feature`,
  `val/loss`, `test/<pheno>/pearson_per_feature` (at best-val checkpoint).

## What is already known (measured; cite the source when you reuse it)

- v16 at epoch 499, 6 pairs per contrast (`results/v16_joint_readout.json`,
  `v16_joint_expr_readout.json`): joint minus proteome-only on the proteome -0.008 (t -1.4);
  four of six expression-only runs COLLAPSED to exactly 0 val Pearson (peaks < 0.03 by
  epoch 39 to 388), all six joint expression heads trained (0.08 to 0.12) and were still
  rising at 499 (+0.003 to +0.023 per 100 epochs). Continuation to 1,200 running (IGB 2410399).
- Collapse (a run's val Pearson decaying to ~0 and staying there) also hit 2 of 36 in v17
  (`tnv3dbe4`, `sylu3gsw`), 1 of 24 in v13 (`825on260`), 0 of 36 in v18 so far. One
  collapsed pair flipped v17's 12-pair call (+0.020 on 11 pairs, +0.012 on 12).
- Levers tested at this trunk and NOT adopted: readout concat (v13, v14), weight decay
  (v15), self-indicator and 2-hop propagation (v17: +0.012, CI through 0), mask off
  (v18: -0.012), per-gene context row (v18: +0.005). Partition (split seed) moves the score
  0.13 to 0.19; every arm effect is a few hundredths.
- Ceilings: best expression ~0.21 on split 0 vs linear ProtT5 ridge 0.13; proteome ~0.11 vs
  0.07 (`results/expression_baselines_split*`, `baselines_split_fig3_proteome*`).
- Throughput: v16 joint 500 epochs in ~30 h at 3 runs/card (A40); v17 1,200 epochs in
  ~24 h at 3/card (A40); v18 1,200 epochs in 19 to 31 h at 3/card (cabbi).

## Rules for your report

- EVIDENCE DISCIPLINE: every claim traces to a file, run id, or script you actually read or
  ran; label hypotheses "Hypothesis (untested):". Partial runs are partial. Name the
  statistic. Distinguish "not measured" from "measured null".
- W&B run links one per line: `https://wandb.ai/zhao-group/<project>/runs/<id>`.
- Style: no em-dashes, American spelling.
- Write your report as Markdown to the OUTPUT path given in your task, nowhere else. Do not
  edit anything under the worktree. Do not run git. Do not ssh anywhere.
- End with a section `## Proposed experiments` : each experiment with arms, partitions x
  seeds, epochs, runs per card, estimated wall per task, partition (gpu %3 / cabbi <= 5 d),
  the pre-registered statistic and decision rule, and what result would let the PI say
  "joint training helps provably". Rank them by evidence value per GPU-day within 14 days.
