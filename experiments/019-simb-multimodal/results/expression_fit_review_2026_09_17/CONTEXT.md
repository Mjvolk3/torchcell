# Shared context for the 019 expression-fit review (2026-09-17)

## Where things are
- Worktree (READ ONLY for you): /home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective
- Run python as: PYTHONPATH=<worktree> ~/miniconda3/envs/torchcell/bin/python  (from the worktree root). W&B API works (`wandb.Api()`, entity `zhao-group`).
- DATA_ROOT on this machine (GilaHyper): /scratch/projects/torchcell-scratch  (datasets under data/torchcell/fig3_core and fig3_proteome; experiments/019-simb-multimodal/results/ has JSON results; val-predictions/ and test-predictions/ hold per-gene dumps of best-val checkpoints)
- Experiment folder: experiments/019-simb-multimodal/{conf,scripts,results}
- Notes: notes/experiments.019-simb-multimodal*.md (76 notes), notes-tex/019-simb-multimodal-expression/ (LaTeX doc with sections incl. 2-readouts.tex), notes-tex/028-knockout-expression-metabolic/, notes/user.Mjvolk3.torchcell.tasks.weekly.2026.*.md (weekly logs)
- Memory notes (short summaries of past conclusions): /home/michaelvolk/.claude/projects/-home-michaelvolk-Documents-projects-torchcell/memory/*.md (019-*, 026-*, 027-*, 028-*, expression-*, nadal-*, flux-*)
- Model: torchcell/models/equivariant_cell_graph_transformer.py ; trainer: experiments/019-simb-multimodal/scripts/train_cgt_multitask.py ; losses: torchcell/losses/distributional.py ; datamodule: torchcell/datamodules/cell.py ; metabolic/flux layer: torchcell/models/cell_graph_transformer_metabolism.py and whatever it imports (026/027 work, experiments/026-*, 027-*)
- W&B projects (entity zhao-group): torchcell_019_expr_v2 ... v13, torchcell_019_expr_v15, torchcell_019_prot_v14, torchcell_019_expr (early), torchcell_019-simb-multimodal_cgt_multitask (earliest). Runs are tagged by arm; groups = arm tags.

## State of the campaign (measured unless marked)
- Task: predict the knockout transcriptome (Kemmeren 2014 + Sameith microarray log2 ratios vs WT, ~6,000 genes x ~1,500 strains, dataset fig3_core) and the knockout proteome (Messner 2023 yeast5k, log2 to HIS3, 1,850 proteins x 4,476 strains, fig3_proteome) from the deletion genotype, with an equivariant cell graph transformer (CGT) over the gene graph, per-gene head, pinball/quantile loss, and a masked-reveal objective (a fraction of the target genes is revealed as input, the rest predicted; k = number of revealed genes; the leaderboard metric is val/<pheno>/pearson_per_feature at k0 = nothing revealed).
- Expression: best val per-feature Pearson ~0.19 to 0.20 on split seed 0 (one 155-strain validation draw that sits +0.089 above other split seeds), v13 across split seeds 0-3 (matched epoch 3,754): 0.197/0.165/0.141/0.139; still rising at epoch ~4,000 of 6,000 (36 epochs/h, A40, 4 runs/card). Train Pearson 0.74. Val loss rises +12% from its minimum while val Pearson keeps rising (uncalibration: pred_sd_ratio, coverage_50 0.32, NMSE 1.085 > 1). Reproducibility ceiling of the expression label ~0.775 (replicate-based), low-rank ceilings from lowrank_output_ceiling.json. Baselines: bilinear ridge on gene embeddings (B2) 0.104 and kNN on ProtT5 embeddings (B3); model 0.1965 vs B2 0.1040 on seed 0.
- Proteome (v14): val Pearson peaks at epoch ~100-200 at 0.08-0.13 and holds; train Pearson 0.08 -> 0.61; label ceiling 0.42 (duplicate strains), 0.61 (HIS3 replicate); kNN baseline 0.073. Test at best-val checkpoint just measured: P_ref s0 0.130 val.
- v15 (weight decay ladder on expression) is running on cabbi; v12 was a readout round (per-gene head +0.041 on seed 0), v11 an embedding round (embedding content +0.068), v10 a grid, v9 a mask round, _006/_007 earlier sweeps. Objective round: pinball vs Pearson loss. Batch 32; batch-64 Pearson collapsed in one round (see memory 019-readouts-2026-09-08).
- Queue plan already decided (do NOT re-litigate): 1) v16 joint proteome+expression heads (built, awaiting launch), 2) proteome trunk size x weight decay ladder at 500 epochs, 3) reliability-weighted per-protein loss. Your job is what comes AFTER those, and whether any of the premises behind the whole campaign are wrong.

## The user's questions (verbatim, dictated)
"question model architecture... question everything... come up with a plan for what to do next after the 1-3 runs. The question is how to improve expression fit.. are we suffering from slow runs bc masking actually needs some warm up to fit properly? would the metabolic module help here? are genes allowed to interact enough? etc. etc. GO deep"

## Rules (hard)
- READ ONLY on the repo: no edits, no git commands that change state (git log/show/diff are fine), no writes under notes/ or experiments/. Write your report ONLY to the file named in your task.
- No training runs, no GPU jobs, no sbatch, no ssh to other clusters. Small CPU analyses with the python interpreter above are fine (W&B history pulls, reading JSON/LMDB, numpy). Keep anything heavy under a few minutes.
- Never add anything to Zotero.
- Evidence discipline: every claim traces to a file, run id, or number you actually read. Label anything unmeasured as "Hypothesis (untested):". A max over epochs is a biased order statistic; say so when you use one.
- W&B evidence: give run ids AND full URLs of the form https://wandb.ai/zhao-group/<project>/runs/<id>, each URL alone on its own line.
- Prose: no em-dashes, American spelling.
- Reason at maximum depth. Prefer reading the actual code and the actual numbers over summaries. When a note's conclusion and the data disagree, say so.
