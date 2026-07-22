---
id: n605e42sy6d1gov8or11uw7
title: Experimental Plans
desc: ''
updated: 1784755927010
created: 1784755927010
---

Experimental program for studying the **Cell Graph Transformer (CGT, model class H)**
on multimodal phenotype prediction — how it is affected by **params × dataset size ×
modality × dataset type**, and which modalities we need for engineering. The immediate
vehicle is Fig-3 expression (YKO), but the framing generalizes.

## Links (top of note)

- **W&B project:** <https://wandb.ai/zhao-group/torchcell_019-simb-multimodal_cgt_multitask>
- Prior overfit run (per-strain 0.41 vs per-gene ~0.03 diagnostic): <https://wandb.ai/zhao-group/torchcell_019-simb-multimodal_cgt_multitask/runs/3rllp896>
- Roadmap: [[plan.simb-2026-multimodal-cgt.2026.07.21]] · Fig-3 build: [[experiments.019-simb-multimodal.scripts.query_fig3]] · sniff-sweep results log: [[experiments.019-simb-multimodal.fig3-expression-experiments]]

## Relevant files (full paths)

- Experiment root: `/home/michaelvolk/Documents/projects/torchcell/experiments/019-simb-multimodal/`
  - Harness: `scripts/train_cgt_multitask.py` (Hydra + Lightning + wandb; heads, masked loss, per-strain/per-gene Pearson, per-gene target standardization, EarlyStopping, scale+param logging)
  - Grid: `scripts/generate_expr_grid.py` → `conf/gh_expr_grid_000..287.yaml` + `results/grid_manifest.json`
  - Launchers: `scripts/gh_cgt_multitask_array.slurm` (GilaHyper single-GPU array, `%4`); `scripts/gh_cgt_multitask.slurm` (4-GPU DDP, superseded for sniffing); IGB/Delta configs `conf/{igb_mmli,igb_cabbi,delta}_train_cgt_multitask_000.yaml`
  - Builds: `queries/fig3_build.cql` · `fig3_core.cql` · `fig6_build.cql`; `scripts/query_fig3.py` / `query_fig6.py`
  - Outputs: `slurm/output/019-expr-grid_<jobid>_<task>.out`; `results/*.json`
- Model (current): `/home/michaelvolk/Documents/projects/torchcell/torchcell/models/equivariant_cell_graph_transformer.py`
- Model (planned main): `torchcell/models/cell_graph_transformer.py` (fork — see §Main CGT file)
- Data pipeline: `torchcell/data/{neo4j_cell,graph_processor,genotype_aggregate,mean_experiment_deduplicate}.py`; `torchcell/datamodules/{cell,perturbation_subset}.py`
- Embeddings: `experiments/embeddings/{compute_esm2_embeddings,compute_isolate_embeddings}.py`; `torchcell/datasets/{esm2,fungal_up_down_transformer,node_embedding_builder}.py`

## Style + compute portability (design invariant)

Every experiment is **wandb + Hydra config + sweep + SLURM**, so it ports across compute
with only header + activation changes. Workflow to scale up: `rsync`/`git sync` code +
configs (dataset LMDBs live under `$DATA_ROOT`, hardlink/rsync separately), `ssh` in, run a
smoke, launch the SLURM array. Targets, all already scaffolded:

- **GilaHyper** (now): single-GPU job array `--array=0-N%4` (`-p main`, `--gres=gpu:1`, plain
  `python`, no DDP) — best for many small sniff runs. Multi-GPU DDP works but see the
  launch gotchas in memory `gilahyper-torchrun-ddp-launch`.
- **IGB BioCluster** `mmli` / `cabbi` partitions (scale-up): Singularity `rockylinux_9.sif`;
  `conf/igb_{mmli,cabbi}_*` + matching slurm. `cabbi` = bigger mem.
- **UIUC Delta** `bbub` (scale-up): apptainer + `/projects/bbub` binds, `gpuA40x4`/`A100`,
  `--account=bbub-delta-gpu`; `conf/delta_*` + slurm.

**Everything logs its scale axes** (added to `train_cgt_multitask.py`): `total_param_count`,
`trainable_param_count`, `n_{train,val,test}_supervised`, `dataset_type`/`composition`,
`active_head`, `hidden_channels`, `num_layers`, `num_heads`, `target_standardized`,
`graph_reg_lambda`, `lr`, `dropout`, `weight_decay`, `seed`. This is the axis data for the
model-class study — outcome (per-strain & per-gene Pearson, val loss) vs scale.

## Research questions

1. **Which modalities does CGT need?** How much does jointly training on modality X (fitness,
   morphology, proteome, metabolome, natural-isolate expression) improve prediction of
   modality Y — and can we predict expensive states (proteome) from cheap ones (fitness /
   a little expression)? Directly informs which measurements to fund for engineering.
2. **How is class-H affected by scale?** Params × train-set size × modality × dataset type →
   outcome. Small-model-first, because on ~1.6k genotypes a 5M-param CGT overfits and
   collapses to the per-gene mean (below).
3. **What decoder makes F̂(P) ≈ F(P) as a distribution**, not a point map (below)?

## Metric definitions (both reported, always)

- **per-strain Pearson** — for each genotype, correlate predicted vs actual vector across
  features; average over genotypes. "Did we reproduce this strain's phenotype *vector*?"
  Dominated by the shared mean KO response → higher (~0.41 seen). The number the ~0.5
  placeholder implicitly targeted.
- **per-gene Pearson** — for each gene/feature, correlate predicted vs actual across
  strains; average over features. "Did we predict *which strains* move gene X?" The hard,
  honest, strain-specific number → low (~0.03 seen). Point-MSE collapses to the per-gene
  mean (MSE-optimal), which scores 0 here — the failure the standardized-target lever
  attacks.

## Current: Fig-3 expression sniff sweep (GilaHyper, ~16 h / ~64 GPU-h)

`sbatch --array=0-287%4 experiments/019-simb-multimodal/scripts/gh_cgt_multitask_array.slurm`
(job 1024). 288 = hidden{16,32,64}×layers{2,3}×target{raw, per-gene-standardized}×
dataset{Kemmeren-only, +Sameith}×graph_reg{0,0.001} × 2 hyperparam profiles × 3 seeds.
Learnable gene embeddings only (correct for YKO — sequence is constant across KO strains,
so it carries no signal; sequence embeddings are for *natural isolates*). Results log +
analysis: [[experiments.019-simb-multimodal.fig3-expression-experiments]].
**Confounds tracked:** Kemmeren↔Sameith overlapping single-KO genes are mean-merged across
platforms by the deduplicator; the Kemmeren-only vs +Sameith arm isolates that effect.

## Decoder research directions (F̂(P) ≈ F(P) as a distribution)

The point-MSE per-gene head collapses to the mean. We want the decoder to **model the
conditional distribution of the phenotype given the perturbation**, exploiting gene–gene
co-regulation structure. Three directions, sequenced quick→ambitious:

### D1 — Distributional readout (quick)

Replace the per-gene point head with a distribution + NLL loss:

- **Gaussian NLL**: predict (μ_g, σ_g) per gene. (Caveat: can still mean-collapse with large
  σ; regularize/​share σ.)
- **Mixture of Gaussians (MoG, K≈3)**: per gene predict {π_k, μ_k, σ_k}, train NLL. Captures
  the bimodal "affected vs ~unchanged" structure of KO expression that a single point/Gaussian
  cannot. Small change to the head + loss; a good first test of "does distribution modeling
  beat point-MSE here?"

### D2 — Masked generative decoder (CellGPT/scGPT-like — the flagship)

Frame phenotype prediction as **masked modeling over the gene set**, conditioned on the
perturbation embedding. This is the direction to build on the new main CGT file.

- **Training (teacher forcing):** given a strain's phenotype vector, randomly MASK a subset
  of genes; predict the masked genes from (perturbation embedding, gene-identity embeddings,
  the *observed* values of unmasked genes). Loss on masked genes only. Because prediction of
  gene *i* conditions on the observed values of other genes, the model learns
  P(gene_i | other genes, P) — i.e. it **uses co-regulation**, exactly the structure point-MSE
  throws away (and the reason MSE mean-collapses).
- **Inference — iterative unmasking (MaskGIT-style, k ≤ 3):** start with all target genes
  masked; predict all; keep the most-confident predictions, unmask them (condition on them),
  re-predict the rest; repeat up to k=3 until fully unmasked. A k-step refinement schedule.
- **Why it fits our goals:** (i) it is inherently *distributional* (an any-order/masked model
  defines a joint over the vector); (ii) it natively handles **partial data** — some genes
  measured, some not → mask the unmeasured; (iii) it extends across **modalities** — mask
  across fitness/expression/proteome and ask "how much does observing modality X reduce
  uncertainty in Y", which is *precisely* the "which modalities do we need" question and the
  "predict expensive from cheap" thesis, as one mechanism.
- This is the "CellGPT-like decoder with random masking on genes without data, then iterative
  unmasking, up to k=3 steps until all data available" the author described.

### D3 — Teacher-forcing / conditional framing (unifies D1–D2)

At inference, condition on any known sub-vector (some genes, fitness, partial proteome) and
predict the rest — the masked decoder supports this directly. This is the operational form of
"predict expensive states from cheaper measured states," and gives a clean modality-necessity
probe (uncertainty reduction per added modality).

**Recommended sequencing:** D1 (MoG) as a fast test on the current head → if distribution
modeling helps, invest in D2 (masked decoder) on the main CGT file. D3 is the evaluation lens
used throughout.

## Main CGT file (fork from equivariant)

Create `torchcell/models/cell_graph_transformer.py` as the canonical CGT, ripped from
`equivariant_cell_graph_transformer.py` (cell encoder + graph-regularized attention +
perturbation operator + heads), decoupled from the legacy file so the decoder variants
(D1–D3) and multi-modal heads are built cleanly on one model class. The 019 harness switches
its import to the new file once it lands. Keep the equivariant file until parity is verified.

## Proteome strand (Messner) — mostly CPU/EDA first

Messner proteome (genome-wide KO) rebuild is pending in the served DB. Before spending GPU on
joint training: (a) EDA — correlate Messner proteome vs Kemmeren expression on shared
genes/strains; (b) linear proteome→expression map (baseline). Expectation is weak on both;
that itself is a result. Then (c) joint train: few YKO expression + a little proteome → does
proteome help expression (and vice-versa)? This is a D2/D3 use-case (mask across modalities).

## Sequencing (near-term)

1. **Now:** Fig-3 expression sniff sweep (job 1024) → read the manifest-joined W&B table;
   the key questions it answers: does small-model + per-gene-standardized target lift the
   per-gene number off the floor; does Kemmeren-only beat the cross-platform-merged +Sameith;
   does attention reg help.
2. Fork the main CGT file; add D1 (MoG) head.
3. Build D2 masked decoder; port the same sniff sweep to it.
4. Proteome EDA (CPU) → conditional D3 modality-necessity probes once Messner is served.
5. Scale-up winners to IGB `cabbi`/`mmli` and Delta `bbub`.

## 2026.07.22 - Initial plan

Created alongside launching the 288-run expression sniff sweep (job 1024). Prior 4-GPU-DDP
run (`3rllp896`) overfit at 5M params / 600 epochs (no early stopping) and mean-collapsed;
this program replaces it with small-model single-GPU sniffing + honest dual metrics +
distribution-modeling decoder research.

## 2026.07.22b - Early sweep result + program expansion

### Sweep 1024 early result (CORRECTS earlier hypotheses)

First 13/288 tasks (all the hidden=16 corner): **per-gene Pearson up to 0.48, per-strain up
to 0.62** (task 1024_5: 116K params, hidden 16, 2 layers, Kemmeren-only, aggressive profile
lr 1e-3 / dropout 0 / wd 1e-4, **target_standardized=FALSE**, graph_reg negligible). Robust
across seeds (0.44–0.48).

- **Expression is NOT intrinsically hard.** The mean-collapse was pure over-parameterization
  (5M params on ~1.3k samples). Capacity reduction (5M→116K) is the whole story.
- **Target standardization was NOT the lever** (top runs use raw log2-ratio). Retract that bet.
- **Attention reg negligible** here (0.44466 with/without) at this small size.
- Per-strain 0.62 > the 0.543 placeholder; per-gene 0.48 is a strong honest number. Full
  manifest-joined leaderboard (params × dataset × standardization × reg) once the grid fills.

### IGB is OFFLINE → Optuna path (not the wandb launcher)

IGB compute nodes have no internet, so the wandb online launcher fails. Mirror the repo's
existing offline pattern (prior art: `experiments/003-fit-int/conf/igb_optuna-*.yaml`,
`experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi.py`):

- **Hydra `OptunaSweeper` + Submitit SLURM launcher**, ONE config = base 019 config +
  `defaults: [..., override hydra/sweeper: optuna, override hydra/launcher: submitit_slurm]`.
- Search space `params:` are DOTTED PATHS into the same Hydra config `wandb.init(config=)`
  logs (the "compatible symbols") — e.g. `model.hidden_channels: choice(16,32,64)`,
  `regression_task.optimizer.lr: tag(log, interval(1e-4,1e-3))`.
- **Offline coordination:** `storage: sqlite:////<scratch>/optuna_019_expr.db` on the cluster
  FS; parallel SLURM workers share one study via `study_name` + `storage` (no internet).
  Multi-objective `direction: [minimize, maximize]` = (val MSE, val per-gene Pearson).
- Harness change: add `is_optuna` detection (`cfg.hydra.sweeper._target_.endswith("OptunaSweeper")`)
  - per-trial job_id from `SLURM_ARRAY_*`; run `WANDB_MODE=offline`; later `wandb sync` from an
  internet machine (`experiments/extra/m1_wandb_sync.sh` pattern).
- TODO: `conf/igb_{cabbi,mmli}_optuna-expr-*.yaml` + submitit launcher slurm. cabbi = 4×48GB
  (A6000/A40-class, like GilaHyper); mmli = 4×A100 for scale. GilaHyper stays wandb-online.

### More metrics (diagnose, don't just score) — mirror the GI trainer

Add to the 019 harness (source: `torchcell/trainers/int_transformer_cell.py`):

- **Spearman** (per-gene + per-strain) — expression ORDERING; add torchmetrics `SpearmanCorrCoef`
  (the GI trainer has none — only scipy for degree bias). Plus **MSE / RMSE** explicit metrics.
- **Attention diagnostics** on diagnostic epochs: row **entropy**, **effective rank** exp(H),
  **top-k mass concentration** (k=5/10/50), avg-max-row-weight (collapse), **column entropy**
  (sink). Graph-recovery: **edge-mass alignment** (attn mass on true edges), recall@degree,
  precision@k, **degree↔attention Spearman**. Plus residual-update-ratio, cls_token_norm, and
  the graph-reg loss breakdown (point/dist/graph_reg + normalized shares).
- **Graph-reg placement rule:** regularize MIDDLE layers (documented: layer 4 of 6–8), each
  graph→distinct head; test with/without as an ablation axis. (Currently 019 uses layers 1–4
  on the small model; revisit once models grow.)

### Splits (be intentional; datasets are small)

- **Start with ONE smart split** (current: seed-42 CellDataModule split). Document its exact
  ratio + strategy.
- **Then replicate over splits** for PROMISING models only (small n → split variance is real;
  ~3–5 splits to get an error bar on the winners). Don't pay split-replication cost across the
  whole sniff grid — reserve it for the leaderboard top-k.
- Open: whether a gene-disjoint / held-out-perturbation split is more honest than random-strain
  (a KO strain's identity IS its gene; random split can leak related genes). Decide before the
  final numbers.

### Deduplicator decision (start reasonable)

Kemmeren↔Sameith overlapping single-KO genes are mean-merged across platforms. Since they
CORRELATE, **mean-merge is a reasonable default to start**; alternatives (dominant dataset /
median) are minor. Keep the Kemmeren-only vs +Sameith arm to measure the effect; document the
choice, don't over-engineer now.

### Sameith double-KO expression sub-study (few n=72)

Question: **does single-KO expression help predict double-KO expression?** Constraint: only 72
doubles. Framing is a **distributional / multi-vector matching problem** — the model must output
a full expression vector, so the baseline choice depends on the representation. Plan:

- Simple baselines that output a vector: linear model / random-forest-per-gene, and a
  "compose singles" baseline (predict double from its two single-KO expression vectors, e.g.
  additive / min / learned combine). The comparison is only meaningful once the representation
  - baseline are pinned — pin them first.
- If it works → (ambitious) extrapolate to ALL double-KO genotypes; then interpretation of
  double-KO expression, which pairs with double-KO **morphology** interpretation.

### Morphology as its own isolated prediction

Fold morphology in with the SAME small-model technique, predicted **alone** (isolated head).
Before preprocessing: **review the CalMorph SI / per-feature variance** — only a subset of the
501 (281 base) features carry across-strain variability; the 3 degenerate ones are already
dropped ([[experiments.019-simb-multimodal.scripts.calmorph_variance_analysis]]), but confirm
which of the 278 actually vary and how to weight/normalize (Yeo–Johnson already wired).

### Decoders — do real D2, not just D1

Author wants to SENSE whether D2 (masked scGPT-like) works, not only D1. Plan a **bounded D2
prototype** on the small-model harness: mask a random gene subset, predict from perturbation +
observed genes, loss on masked; k≤3 iterative unmask at inference; compare per-gene/per-strain

- Spearman vs the point-MSE head. Keep it time-boxed (sense-check, not a full build) before
committing to the main-CGT-file version.

### Modeling goal (framing)

Ultimate goal is to **represent states** (a virtual-cell latent); the expression + morphology
correlation targets are the PRACTICAL proxy we optimize toward now.
