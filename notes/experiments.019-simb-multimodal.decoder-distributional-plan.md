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
