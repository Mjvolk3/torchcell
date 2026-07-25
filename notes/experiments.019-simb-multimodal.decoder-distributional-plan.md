---
id: 337vkaiulqllqmt2u0tta6n
title: Decoder Distributional Plan
desc: ''
updated: 1784958241910
created: 1784958241910
---

Decoder architecture taxonomy, distributional-loss options, and the concrete Optuna experiment
plan for the 019 multimodal Cell Graph Transformer (CGT). Notation matches the manuscript
(`paper/nature-biotech/sections/methods.tex`, Table `tab:notation`). This note is the
self-contained spec for a `/uber-implement` + `/enqueue-merge` pass. Base branch:
`feat/igb-mmli-optuna-morph`.

## Terms & acronyms

- **CGT** — Cell Graph Transformer (this model).
- **ENC / PERT / DEC** — the three stages: encode / perturb / decode, implemented as $F_\theta$ /
  $\mathcal{T}_\psi$ / $\mathcal{R}_\phi$.
- **CLS token** — a learnable "whole-cell" summary token prepended to the gene tokens (BERT-style);
  it attends to every gene in ENC, so its output $h_{\mathrm{CLS}}$ is a *learned pool*.
- **MLP** — multilayer perceptron; **LN** — layer norm; **FFN** — feed-forward network.
- **per-feature pearson** (a.k.a. the *difficult* / honest metric) — for a vector output, correlate
  EACH output dimension across strains, then mean over dimensions. Destroyed by mean-collapse.
- **per-instance / per-strain pearson** (the *easy* metric) — for each strain, correlate its
  predicted vector across dimensions; survives mean-collapse and is scale-dominated (do not rank on it).
- **mean-collapse** — under MSE the optimal prediction is the conditional mean, so a weakly-supervised
  model hedges every strain to the per-feature mean → per-feature pearson $\approx 0$.
- **CalMorph** — Ohya-lab image pipeline producing the 278 morphology features.
- **PCA** — principal component analysis; **effective rank** — participation ratio $(\sum\lambda)^2/\sum\lambda^2$.
- **DETR** — DEtection TRansformer (Carion 2020): a fixed set of learned *object queries* cross-attend
  to encoder features, each emitting one output. **Perceiver** (Jaegle 2021): learned latent queries
  cross-attend to a large input. Both are the template for S3.
- **MVC** — "Masked Value prediction for Cell-embedding" (scGPT): read each output from a per-output
  embedding × cell-vector product; the template for S2.
- **NLL** — negative log-likelihood; **β-NLL** — β-weighted NLL (Seitzer 2022), stabilizes σ.
- **CRPS** — Continuous Ranked Probability Score, a *proper scoring rule* (a loss minimized in
  expectation only by the true distribution) for probabilistic forecasts; value is in units of $y$.
- **pinball / quantile loss** — $\rho_\tau(u)=\max(\tau u,(\tau-1)u)$; trains predicted quantiles.
- **KDE** — kernel density estimation (the past approach whose *bandwidth* we want to avoid tuning).
- **TPE** — Tree-structured Parzen Estimator (Optuna's default sampler).

## Notation (from `tab:notation`)

$G$ cell graph, $\varepsilon$ environment, $p=\{(e_t,\tau_t,m_t)\}_{t=1}^M$ perturbation set,
$\mathcal{Y}$ phenotype space ($y$ observed, $\hat y$ predicted), $d$ embedding dim, $N$ genes,
$F$ output dimensions of a task ($F=N$ expression, $F=278$ morphology, $F=1$ fitness).

$$\hat f_\theta(G,\varepsilon,p)=\mathcal{R}_\phi\big(\mathcal{T}_\psi(F_\theta(G),p),\,\varepsilon\big),\qquad H=F_\theta(G)=(h_{\mathrm{CLS}},h_1,\dots,h_N),\ \ h_i\in\mathbb{R}^d.$$

$$H_{\mathrm{pert}}=\mathcal{T}_\psi(H,p)=(h_{\mathrm{CLS}}^{\mathrm{pert}},h_1^{\mathrm{pert}},\dots,h_N^{\mathrm{pert}}).$$

## The decoder factorizes into two orthogonal axes

- **Structural** ($\mathcal{R}_\phi$'s form, S0–S4): *where* each output reads its info → input support.
- **Distributional** (the head's output + loss): a point $\hat y_k$ vs a distribution; sets the
  mean-collapse incentive. Applies on top of ANY structural form → it is a **column**, not a fourth row.

## Structural forms

**S0 — per-token direct (expression; BUILT).** Outputs map 1-to-1 to gene tokens:
$$\hat y_{\mathrm{expr},i}=\operatorname{MLP}\!\big(h_i^{\mathrm{pert}}\big),\quad i=1,\dots,N.$$
Each output reads its OWN token → full per-output support. Expression is *not* decoder-limited (it is
at its noise ceiling ~0.11).

**S1 — pooled MLP (morphology / fitness; BUILT).** No 1-1 map → collapse to one vector then fan out:
$$u=\operatorname{pool}(H_{\mathrm{pert}}),\qquad \hat y_k=w_k^{\!\top}\sigma(W_1u),$$
where $\hat y_k$ is component $k$ of $\hat y\in\mathbb{R}^F$ and $w_k$ is row $k$ of $W_{\mathrm{out}}$
(a free learned weight vector). Pool = $h_{\mathrm{CLS}}^{\mathrm{pert}}$, mean $\tfrac1N\sum_i
h_i^{\mathrm{pert}}$, or their concat. **Bottleneck:** all $F$ outputs read the SAME $u$; a mean over
$\sim$6000 genes dilutes a few-gene perturbation → the strain signal is lost before readout. Current
config `use_gene_pool: true` → $u=[h_{\mathrm{CLS}}^{\mathrm{pert}}\Vert\text{mean}]$. **010** used this
form for trigenic interaction, pooling over ONLY the perturbed genes:
$\hat y_{\mathrm{int}}=\operatorname{MLP}([h_{\mathrm{CLS}}^{\mathrm{pert}}\Vert\tfrac{1}{|p|}\sum_{g_j\in p}h_j^{\mathrm{pert}}])$.

**S2 — bilinear / feature-embedding (MVC-style; DEFERRED).** Give each output a learned embedding
$q_k\in\mathbb{R}^d$, shared $W$: $\hat y_k=q_k^{\!\top}Wu$. Difference from S1: $W_{\mathrm{out}}$ is
$F$ *free* rows; $q_k^{\!\top}W$ is a *factored* per-output weight. For a FIXED $F$, S1 $\supseteq$ S2 in
expressiveness → **S2 buys nothing for morphology-alone.** Its value is scaling: a new label = add a
$q_k$ (multi-phenotype masking, Fig 5). Add only when adding phenotypes.

**S3 — cross-attention (DETR/Perceiver; the fix; BUILD).** Each output is a learned query $q_k$ that
cross-attends to the FULL token set:
$$A=\operatorname{softmax}\!\Big(\tfrac{(QW_Q)(H_{\mathrm{pert}}W_K)^\top}{\sqrt{d_k}}\Big),\quad C=A\,(H_{\mathrm{pert}}W_V),\quad \hat y=\operatorname{DistHead}(C).$$
$Q\in\mathbb{R}^{F\times d}$ (learned), keys/values over $\{h_{\mathrm{CLS}}^{\mathrm{pert}},h_1^{\mathrm{pert}},\dots,h_N^{\mathrm{pert}}\}$.
Now output $k$ reads a DIFFERENT weighted mix $c_k$ → per-output support recovered without a 1-1 map;
feature $k$ can attend to the perturbed genes instead of averaging them away.

**S4 — feature tokens in the encoder (DEFERRED).** Prepend $F$ learnable feature-*slot* tokens (NOT
label values — parameters like CLS) so they contextualize through ENC+PERT, then read them out. Most
expressive (features see genes and each other) but $F\times$ longer sequence. Try only if S3 wins but
the residual is feature↔feature structure.

## Expression ↔ morphology asymmetry (principled)

- **Expression:** outputs ARE tokens → **S0**, genes only, no queries.
- **Morphology:** the 278 features are NOT tokens → **S3** queries attend into $\{$all genes $+$ CLS$\}$
  (CLS kept as a key so a feature can also read the global summary).

## Distributional axis (trimmed to 3: `point`, `crps`, `quantile`)

Per feature $k$ on standardized targets; the metric uses `.point()` so ranking is loss-agnostic.
$\Phi,\varphi$ = standard-normal CDF/PDF.

- **`point`** — head $\to\hat y_k$; loss $=\tfrac1F\sum_k(\hat y_k-y_k)^2$ (MSE). Baseline; *rewards*
  mean-collapse (optimal $\hat y_k=\mathbb{E}[y_k]$). `.point()` $=\hat y_k$.
- **`crps`** (Gaussian, closed form) — head $\to(\hat\mu_k,\hat\sigma_k)$, $z=(y_k-\hat\mu_k)/\hat\sigma_k$;
  $$\mathcal L=\sum_k\hat\sigma_k\big[z\,(2\Phi(z)-1)+2\varphi(z)-\tfrac{1}{\sqrt\pi}\big].$$
  Proper scoring rule, in $y$-units, robust (no $1/\sigma^2$ blow-up). Gives calibrated intervals /
  $P(y>t)$. `.point()` $=\hat\mu_k$. Student-t is a later toggle.
- **`quantile`** (distribution-free) — head $\to\hat q_{\tau,k}$ for $\tau\in\{0.05,\dots,0.95\}$ (K=19,
  evenly spaced — a *robust* choice, no kernel/bandwidth); loss $=\sum_{\tau,k}\rho_\tau(y_k-\hat q_{\tau,k})$.
  Empirical CRPS $\approx\tfrac1K\sum_\tau$ pinball. `.point()` = median ($\tau{=}0.5$).

**Dropped `beta_nll`** — CRPS supersedes it on robustness (β-NLL was a patch for NLL's $\sigma$-collapse;
CRPS avoids it natively). These three span: no-distribution / parametric-distributional /
distribution-free — and all avoid the binning + KDE-bandwidth tuning of past attempts (they model the
conditional distribution in continuous space, per-instance, not a discretized or kernel-smoothed marginal).

## Sizing finding (drives the design)

PCA of the 278 CalMorph outputs (z-scored): **effective rank 11.7**; 90% variance in 34 PCs, 95% in 51.
Consequences: (1) $d{=}96$ over-represents a ~12-dim output → morph is NOT capacity/embedding-limited
(scaling $d$ won't help — matches `_002`); (2) the wall is **pool-dilution** → S3 is the fix; (3) the
output is low-rank → a factored readout is natural; (4) features co-vary → a **low-rank-cov Gaussian**
$\Sigma=D+LL^\top$ (R≈12) is the natural future distributional upgrade.

Ceilings (committed scripts, sha-pinned SCMD/deleteome mirrors): morphology per-feature ceiling
**0.61** (`morphology_noise_ceiling.py`), expression **~0.09–0.11** (`expression_noise_ceiling.py`).
Observed: morph 0.04 (6.5% of ceiling → decoder-limited), expr 0.11 (~saturated).

## Experiment spec

### Rank metric = the difficult pearson (per-feature, across strains)

Optuna ranks trials on the PEAK (BestMetricTracker `_max`) of `val/<phenotype>/pearson_per_feature`.

| arm | Optuna objective |
|---|---|
| morph | `val/morphology/pearson_per_feature_max` |
| expr | `val/expression/pearson_per_feature_max` |
| joint | multi-obj `(val/expression/…_max, val/morphology/…_max)` (Pareto) |

### Table A — Optuna search space

| axis | values | notes |
|---|---|---|
| `decoder` | {`s1_pool`, `s3_xattn`} | expr always S0; S2/S4 deferred |
| `dist` | {`point`, `crps`, `quantile`} | one pluggable `DistHead` |
| `hidden_channels` | {64, 96, 128} | morph not d-limited; kept for joint/expr |
| `num_transformer_layers` | {2, 4} | |
| `graph_reg_lambda` | {0, 3e-4, 1e-3, 3e-3} | |
| `target_norm` | {`zscore`, `yeo_johnson`} | **always standardize** (no `raw`) |
| `hp_profile` | {baseline, aggressive} | lr/dropout/wd bundle, unchanged |

### Table B — the three jobs (→ Wed 2026-07-29 10:00 CST, requeue)

| job | machine · part | GPU | condition · heads | study `_003` | wandb project | grid |
|---|---|--:|---|---|---|---|
| **morph** | IGB **mmli** | 4 | morph · `[morphology]` | `morph_003` | `torchcell_019_morph_v3` (offline) | full A |
| **joint** | **GH** main | 4 | expr_morph · `[expression,morphology]`, standardize-both | `expr_morph_003` | `torchcell_019_expr_morph_v3` (online) | {s1,s3}×{point,crps}×size×λ |
| **expr-ctrl** | IGB **cabbi** | 2 | expr · `[expression]` (S0 fixed) | `expr_003` | `torchcell_019_expr_v3` (offline) | {point,crps,quantile}×size×λ |

cabbi allocation kept on one node (partition rule). Data staged (fig3_core, split-identical, sha `a572f3eb…`).

### Table C — files to create / edit

| file | change |
|---|---|
| `torchcell/models/equivariant_cell_graph_transformer.py` | add `CrossAttnHead` (S3); add output-mode (`point`/`gaussian`/`quantile`) so head width = F / 2F / KF; head factory keyed on (`decoder`,`dist`) |
| `torchcell/losses/distributional.py` **(new)** | `gaussian_crps` (closed form), `pinball`; `DistHead` interface `params → .loss(y,mask) / .point()` |
| `experiments/019-simb-multimodal/scripts/train_cgt_multitask.py` | (1) logging refactor → `val/<phenotype>/pearson_per_feature` (primary) + `pearson_per_instance` (diagnostic, on z-scored feats), phenotype namespaces via head→phenotype map; (2) always-standardize default; (3) wire `decoder`+`dist` into head build + loss select; (4) `.point()` feeds the metric so ranking is loss-agnostic |
| `experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py` | add `decoder`,`dist` categoricals; rank on per-feature peak; `_003` studies/db; project `torchcell_019_<cond>_v3` |
| `experiments/019-simb-multimodal/conf/*.yaml` | `multitask.decoder`, `multitask.dist`, standardize defaults; morph output_dim 278 |
| `experiments/019-simb-multimodal/scripts/mmli_morph_decoder_003.slurm` **(new)** | mmli 4-GPU, morph, requeue |
| `experiments/019-simb-multimodal/scripts/gh_joint_decoder_003.slurm` **(new)** | GH 4-GPU, expr_morph, online, requeue |
| `experiments/019-simb-multimodal/scripts/cabbi_expr_decoder_003.slurm` **(new)** | cabbi 2-GPU, expr, requeue |
| `experiments/019-simb-multimodal/scripts/requeue_until.sh` **(new)** | deadline-guarded self-resubmit |
| `experiments/019-simb-multimodal/scripts/{morphology,expression}_noise_ceiling.py` | **commit** (ceilings 0.61 / ~0.09) |

### S3 module + requeue

- **S3** `CrossAttnHead`: $F$ learned queries $Q$, multi-head cross-attn over the $N{+}1$ tokens (eqn
  above), optional FFN on $C$, then `DistHead`. Morph $F{=}278$; expression stays S0.
- **Requeue:** `#SBATCH --deadline=2026-07-29T10:00:00` + `--signal=B:USR1@180`; trap runs
  `[ "$(date +%s)" -lt "$DEADLINE_EPOCH" ] && sbatch "$0"`. Optuna SQLite persists → workers reattach,
  the study grows until the deadline. Fresh `_003` studies (grid changed the categorical space).

### Success criteria

- **morph:** does `s3_xattn` and/or a distributional loss lift `pearson_per_feature` above 0.04 toward
  0.61? s3 wins ⇒ decoder-limited confirmed → invest (S4 / low-rank-cov). Nothing wins ⇒ genotype→morph
  may be unlearnable from these instances (revisit data).
- **joint:** does standardize-both restore joint-expr to ~0.11 (kills the negative transfer)? does s3
  lift morph-in-joint?
- **expr-ctrl:** stays ~0.11 across decoders/losses ⇒ confirms expression is noise-saturated,
  decoder/loss-agnostic.

### Launch (after build + a `--fast_dev_run` smoke test per head×loss)

```
# IGB — sbatch only on the login node (NO compute on login nodes):
ssh mjvolk3@biologin.igb.illinois.edu 'sbatch <repo>/experiments/019-simb-multimodal/scripts/mmli_morph_decoder_003.slurm'
ssh mjvolk3@biologin.igb.illinois.edu 'sbatch <repo>/experiments/019-simb-multimodal/scripts/cabbi_expr_decoder_003.slurm'
# GH — from the worktree root so SLURM_SUBMIT_DIR uses worktree code:
sbatch experiments/019-simb-multimodal/scripts/gh_joint_decoder_003.slurm
```

Sync IGB offline runs with `wandb_sync_agent_dirs` (filter to the `_003` job ids). Land via
`/enqueue-merge`. Cluster rules: [[cluster-rules-igb-delta]]; prior run: [[019-controlled-multitask-v2-sweep]].

## 2026.07.25 - Implementation landed (commit `d07037b1`)

Built on `feat/igb-mmli-optuna-morph`. Deviations from the spec above are flagged **[CHANGED]**.

### What shipped

| file | state |
|---|---|
| `torchcell/losses/distributional.py` | NEW — `gaussian_crps`, `pinball`, `DistHead` (`.point()` / `.loss(params,y,mask)` / `.param_dim`), `make_dist_head`, `dist_param_dim` |
| `tests/torchcell/losses/test_distributional.py` | NEW — 17 tests |
| `torchcell/models/equivariant_cell_graph_transformer.py` | `CrossAttnHead` (S3); `param_dim` on `GlobalHead`/`PerGeneHead`; head factory on (`decoder`,`dist`); `MaskedMultitaskLoss(dist_heads=...)` |
| `experiments/019-simb-multimodal/scripts/train_cgt_multitask.py` | phenotype-namespaced metrics; `DistHead` wiring; always-standardize guard; `decoder`/`dist` in scale-meta |
| `experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py` | `decoder`/`dist` axes; `_003` studies; `_v3` projects; per-feature peak objective |
| `experiments/019-simb-multimodal/conf/cgt_decoder_003.yaml` | NEW base config (all three arms) |
| `.../scripts/{mmli_morph,gh_joint,cabbi_expr}_decoder_003.slurm` + `requeue_until.sh` | NEW |

### Design decisions

- **Param layout `[B,F]` / `[B,F,P]`.** `output_dim` stays the FEATURE count (278); `param_dim`
  (1/2/19) only widens the head's final projection. This keeps the Part A `output_dim ==
  feat_dim` sanity check valid and leaves the `per_gene` `index_select` gather untouched.
- **`.point()` feeds the metric** → ranking is loss-agnostic across `point`/`crps`/`quantile`.
- **`pearson_per_instance` is computed on NORMALIZED features**, `pearson_per_feature` on raw
  units. Per-instance correlates *across* features, so on raw multi-scale CalMorph values it
  would be dominated by the largest-magnitude features rather than measuring profile shape.
- **Backward compatible**: a `heads_config` without `decoder`/`param_dim` builds the old
  `GlobalHead`, emits `[B,F]`, and uses the plain MSE path.
- **S3 cost is ~1.5x S1** at d=32 (17.5k vs 11.3k params); the readout is a *shared*
  `Linear(d→P)`, so parameter count is independent of F and the per-feature specificity lives
  in the 278 learned queries. A win is therefore evidence for pool-dilution, not capacity.

### [CHANGED] vs the spec

1. **`quantile` is excluded from the JOINT arm** (morph/expr sweep all three). K=19 params over
   both a 6127-gene and a 278-feature head is a large readout for the shared 4-GPU node.
2. **`decoder` is not suggested for the `expr` arm** — expression is S0 by construction, so
   sweeping it would add a phantom dimension that splits the TPE search space without changing
   the model.
3. **Always-standardize is an OPT-IN guard** (`multitask.require_standardized_targets`, on in
   `cgt_decoder_003`, off elsewhere) — several legacy configs legitimately run an
   un-normalized head, and a global default would have broken them.
4. **`drop_features` = `[A113_A, D203, D205]`** (matching `_002` via `delta_joint_expr_morph_000`)
   so `_003` targets the SAME 278 features and stays comparable. **OPEN ISSUE:** the train-split
   stats find **six** near-constant features (`A113_A, A113_A1B, A113_C, C123_C, D203, D205`),
   and the repo's configs split them into two different 3-subsets (`train_cgt_multitask.yaml` /
   `gh_cgt_multitask_*` drop the other three). Both give 278. Worth reconciling before the
   paper figure; dropping all six would give 275 and break comparability with `_002`.

### Fixes forced by the refactor

- `_extract_targets_and_masks` sized the target buffer from the head output, so any
  distributional run crashed (`[B,F,P]` vs a decoded `[F]` row). Targets are now sized from
  `.point()`. **Only the real-data path caught this** — the synthetic dry-run built its targets
  from `.point()` and passed.
- The metric rename broke three live references: `optuna_morph_sweep.py`'s objective (would have
  pruned every trial) and the early-stopping monitors in `gh_expr_optuna_000.yaml` /
  `mmli_morph_optuna_000.yaml`. All updated.

### Verification

`22 passed` (17 new distributional + 5 pre-existing model), `mypy --strict` + `ruff` clean, and
**12/12 real-data `fast_dev_run` combinations** train end-to-end: morph {s1_pool,s3_xattn} ×
{point,crps,quantile}, morph s3+crps under z-score, expr S0 × 3, joint × 2 — plus the
always-standardize guard correctly failing fast on an un-standardized head. CRPS is validated
against the Monte-Carlo energy form (`E|X-y| - ½E|X-X'|`) to ~1e-4 and shown proper.

### Launch (NOT yet submitted)

```bash
# IGB — sbatch from a LOGIN node only (no compute on login nodes)
ssh mjvolk3@biologin.igb.illinois.edu 'sbatch /home/a-m/mjvolk3/projects/torchcell/experiments/019-simb-multimodal/scripts/mmli_morph_decoder_003.slurm'
ssh mjvolk3@biologin.igb.illinois.edu 'sbatch /home/a-m/mjvolk3/projects/torchcell/experiments/019-simb-multimodal/scripts/cabbi_expr_decoder_003.slurm'
# GilaHyper — from the worktree root so SLURM_SUBMIT_DIR picks up worktree code
cd ~/Documents/projects/torchcell.worktrees/feat/igb-mmli-optuna-morph && \
  sbatch experiments/019-simb-multimodal/scripts/gh_joint_decoder_003.slurm
```

IGB requires the branch to be synced to `/home/a-m/mjvolk3/projects/torchcell` first. Sync IGB
offline runs with `wandb_sync_agent_dirs` (filter to the `_003` job ids). Land via `/enqueue-merge`.
