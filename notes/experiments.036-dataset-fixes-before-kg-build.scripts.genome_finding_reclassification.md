---
id: gklpldc7j11u1mxnfamy2kz
title: Genome_finding_reclassification
desc: ''
updated: 1791817907577
created: 1791817907577
---

## 2026.10.12 - Which stores flip from FAIL to PASS

Judges each mapped store under two rules: every rule gates (before 2026-10-12) and only `gate` rules gate (owner, "the genome is separate.. it doesn't gate anything"; [[torchcell.verification.report]]). `--reports` reads the `preprocess/verification_report*.json` already on the dev tree (results file `results/genome_finding_reclassification_reports.json`); `--sweep <json>` reads the lane files of a read-only rerun through [[experiments.036-dataset-fixes-before-kg-build.scripts.gh_genome_finding_rerun]]. A store passing every rule cannot fail when fewer rules gate, so only an old FAIL can flip. Writes `results/genome_finding_reclassification.json`; the tables are in [[database.kg-4-pre-build-verification]].
