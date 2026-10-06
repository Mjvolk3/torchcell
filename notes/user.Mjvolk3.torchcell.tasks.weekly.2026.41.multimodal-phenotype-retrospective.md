---
id: cb8uzgpvwgepvoqna9i7n5s
title: multimodal-phenotype-retrospective
desc: ''
updated: 1791255604174
created: 1791255604174
---


## 2026.10.05

- [x] Ten independent reviews of the expression, proteome and morphology campaign: no trained model beats the graph-keyed neighbor baseline on matched strains, several headline numbers were not valid comparisons, and the plan to the deadline was rewritten as five parallel tracks sized to the Delta allocation [[experiments.019-simb-multimodal.expression-strand-retrospective]]
- [x] v19 joint round read out at 11 of 12 partitions as a null, written into the expression document as a dated checkpoint section [[experiments.019-simb-multimodal.scripts.joint_checkpoint_readout]]
- [x] v20 conditioned round (one modality revealed, the other predicted, permuted controls) launched on the two free cabbi cards; pending tasks resubmitted with persistent workers [[experiments.019-simb-multimodal.scripts.gh_expr_008_arm]]
- [x] Linear and nearest-neighbor baselines for every gene representation on all twelve split seeds, restricted to the joint rounds' strains, running on IGB CPU inside the container [[experiments.019-simb-multimodal.scripts.igb_expression_baselines_split]]
- [x] Figure 3 gate document with draw.io mockups at Nature print size (two framings plus two SI figures), status marks per panel, every value from committed results [[experiments.019-simb-multimodal.scripts.fig3_mockup_panels]]
- [x] The abstract's expression (0.543) and morphology (0.619) correlations withdrawn from the manuscript and marked incorrect at every place the repo reports them [[conference.simb-2026.abstract]]
- [x] Perturbation operator written out in the paper's notation (softmax, the sigmoid of Fig 1f, the null sink, what a gene can choose at one deletion) as a notes-tex document published to Zotero; the null sink has four attempts, one at 4,184 epochs [[experiments.019-simb-multimodal.multiplicative-perturbation-conditioning]]
- [x] Single-deletion triangle completed with the expression-against-morphology side [[experiments.019-simb-multimodal.scripts.expression_morphology_covariation]]
- [ ] Bring the twelve-seed baseline results back from the IGB `019-baselines` worktree and redraw the embedding panel over all split seeds [[experiments.019-simb-multimodal.scripts.igb_expression_baselines_split]] #high
- [ ] Resync v20 with `--recheck` as tasks finish and rerun the readout; delete the four empty canary runs from the v20 W&B project once approved [[experiments.019-simb-multimodal.scripts.joint_checkpoint_readout]]
- [ ] Build the cached-encoder trainer (encoder output cached, operator and heads train) and canary it on the free IGB gpu card before any Delta submission #high
- [ ] Author decisions: Figure 3 as a benchmark; Fig 1f or the Methods as the named operator (touches the trigenic model); when Delta submissions start and on which account #high
- [ ] Fold the scratch operator note into the notes-tex document and trash it [[scratch.2026.10.05.perturbation-operator-fig1f-vs-softmax]]
- [ ] Launch discussion: Delta `bbub` 1,000-epoch arrays (calm vs ProtT5; R_ref vs R_pergene, 3 seeds each), one batch-32 solo Pearson seed on cabbi when the batch-64 tasks end
- [ ] CPU probe of stacked embeddings (ProtT5 + calm / codon frequency / chrom pathways) and a reporter-side promoter probe before spending cards on the embedding arm
- [ ] When 21948711 lands: `v11` readout script (paired within node), fold into the expression document's readouts and next sections; width-matched random filler arm if E_full leads
- [ ] When v12 lands (~2026-09-12): readout script paired within seed against `H_ref` at matched budget, then the document's readouts and next sections; compare `H_ref` against v11's E_full at 1,400
- [ ] scYeast (fanScYeastBiologicalknowledgeguidedFoundation2027) datasets not in torchcell: Jackson 2020/2023 scRNA, Wang 2022 aging, Su 2023 stress, IDEA (Hackett 2020), Messner growth rates, McManus ribosome occupancy, Martin-Perez half-lives; lowest effort = Messner growth rate (same mirrored SI)
- [ ] Zelezniak 2018 metabolome vs Messner protein / Kemmeren mRNA goes to the metabolic strand (026/027), not 028
- [ ] Xue FFA: 7 of the 10 TF deletions have Kemmeren profiles (and O'Duibhir growth), Messner has 10 of 13 genes; idea: predict expression for the 175 combinatorial strains from a Kemmeren-trained model and test the predictions against FFA titers [[torchcell.datasets.scerevisiae.xue2025]]
- [ ] expression-correlate candidates vs the triage table: Muenzner 2024 is row 53 (not in the top 10); Hughes 2000, Hu 2007, IDEA (Hackett 2020), Albert 2018, McManus 2014, Martin-Perez 2017, Sun 2013 are NOT in the 79-row table (Hughes 2000 appears only in the SPELL list of [[torchcell.sgd-expression-studies]]); recorded as candidate rows in [[paper.north-star.dataset-triage]]
- [ ] Next for tryptophan: schema for a promoter-replacement-plus-relocation perturbation and a biosensor time-series phenotype, then loader + adapter
- [ ] score the 40 pulled checkpoints on GilaHyper (val must reproduce W&B, test dumped), then the v13 and v14 test readouts with the gene-disjoint subset
- [ ] v17 wave 2 and v18 reads once they reach epoch 1,000; both sides of v16 at the loss-minimum checkpoint once a checkpoint home is chosen
- [ ] readout stitching: `v13_split_readout.py --round v16` and `v16_expr` must concatenate a continued run's history onto its source run by `wandb.resumed_from` before the 1,200-epoch read; the view script should group the two segments under one arm

## 2026.10.06

- [x] **profile of the expression CGT epoch** ([[experiments.019-simb-multimodal.scripts.gh_profile_cgt_expr]], GilaHyper jobs 3302 to 3313): a sample costs about 50 ms to produce (LMDB, JSON, pydantic, processor), 57 s of a 151 s epoch at zero workers; the model step is 8 s per epoch at batch 32 and is paced per step because the encoder's self-attention over 6,607 genes (70 percent of CUDA time) runs once per step whatever the batch; the eval-mode train pass respawns its loader workers on every call (about 30 s)
- [x] fast code path: batched perturbation operator (40 percent off the step, equivalence test on eight cases), `pooled_perturbed`, `MaterializedSplit` (`+data_module.materialize=true`), `trainer.profiler` pass-through
- [x] **v22 wave 1 launched** ([[experiments.019-simb-multimodal.scripts.gh_expr_v22_fast]], job 3315, 16 runs, 1,200 epochs): F_ref, F_b128 (lr 6e-4), F_b128lr4 (lr 1.2e-3), F_l4w180 on split seeds 0 to 3, one card per split seed
- [ ] read v22 at the registered window; fix the eval-train pass loader respawn; decide the Delta round from v22 and v21
