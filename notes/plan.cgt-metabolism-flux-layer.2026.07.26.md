---
id: 29l9q6cz0vo64bwt6nkw4mt
title: CGT-Metabolism Flux Layer — model math and the Flux Cone Learning comparison
desc: 'Track B model specification (gene-centric entities, soft catalysis, amortized flux sampler, full objective) and the Merzbacher 2025 FCL comparison it is built for'
updated: 1785093686225
created: 1785093686225
---

## 2026.07.26 — Why this note exists

The model math for the metabolism layer lives in **two untracked scratch notes** and in no commit:

- `scratch.2026.07.25.010919-adding-metabolism` §2026.07.25b/c — the first pass.
- `scratch.2026.07.25.172340-adding-metabolism-explainer` — the **authoritative** version: derivations
  from definitions up (Parts 0–7) plus the Q1–Q15 read-through, and a consolidated decisions table
  that **supersedes the first pass on the central architectural choice**.

Where they disagree, the explainer wins. The most important reversal: **the enzyme node layer is
dropped.** The first pass proposed +1,065 enzyme nodes so $k_{\mathrm{cat}}$ and MW would have
somewhere to live; the explainer's Part 3 shows that is the wrong place to put an *annotation*, and
that all three things enzyme nodes were buying survive as functions of gene nodes.

It is also **re-framed**, because the priority changed. Track B was written as "after SIMB." It is
not: the headline comparison is **Merzbacher 2025, *Flux Cone Learning***, whose method is the
un-amortized version of exactly this flux layer. Comparing a gene-token model carrying no metabolic
structure against a flux-cone method is a category error — we would win or lose for reasons
unrelated to the claim. The math below is the critical path, not an appendix.

Companions: [[plan.cgt-metabolism.2026.07.25]] (Track A execution, run logs, noise ceilings) ·
[[plan.simb-2026-multimodal-cgt.2026.07.21]] (WS8, WS16 precursor pools) ·
[[torchcell.models.cell_graph_transformer_metabolism]] (Track A as built).

**Two goals, in priority order:**

1. **Betaxanthin — beat Flux Cone Learning on its own task**, then report what it structurally
   cannot: regression, and the ~81 % of the screen outside the GEM.
2. **β-carotene — extrapolate to double mutants.** We hold double-mutant *fitness*; β-carotene
   production is measured only in **single** deletions (Ozaydin 2013 is the haploid BY4741 deletion
   collection, 4,406 rows in our build). Doubles are therefore an inverse-design extrapolation, and
   §"Why FCL cannot follow" is the argument for why an amortized model is the only route there.

## Settled decisions (consolidated — these override the first-pass note)

| # | decision | resolution |
| --- | --- | --- |
| 1 | **enzyme nodes** | **DROP** — gene-centric nodes + soft catalysis $\Pi$ |
| 2 | exactness budget | the **box**, now *dynamic* in $(\varepsilon,p,\mathrm{seq})$ |
| 3 | growth objective | **drop the prior**; supervise $v_{\mathrm{bio}}$ with fitness |
| 4 | flux head | **stochastic → amortized flux sampler**; report against FVA width |
| 5 | $K_M$ | **required *if and only if* promiscuity is enabled** — it is what makes promiscuous flux cost something. Sourced from the Open Enzyme Database / BRENDA, **not** from ecYeastGEM (GECKO carries no $K_M$) |
| 6 | $S$ source | Yeast9 now; Yeast-MetaTwin a live option — decide, do not drift |
| 7 | media | port the corrected iBioFoundry media into `torchcell/metabolism/` |
| — | $S$ vs $\rho$ | **hard chemistry, soft biology** — $S$ never softens; $\rho$'s zeros mean *untested* |
| — | promiscuity | $\Pi=\Pi^{\mathrm{GPR}}+\Delta\Pi$, $\Delta\Pi\ge0$, $\ell_1$-penalized — annotation is a **floor** |
| — | thermodynamics | per-metabolite $\mu_i$; loop-freedom falls out, no integer variables |

## The comparison target — Merzbacher 2025, verified against the paper

`merzbacherAccuratePredictionGene2025` — Merzbacher, Mac Aodha & Oyarzún, *"Accurate prediction of
gene deletion phenotypes with Flux Cone Learning,"* Nat Commun 2025, doi
`10.1038/s41467-025-63436-9`. OCR'd at `$DATA_ROOT/torchcell-library/merzbacherAccuratePredictionGene2025/`.
Numbers below are quoted from `paper.md`, not from a summary.

**The method.** For each single-gene deletion, zero the GPR-implied flux bounds in the wild-type GEM,
then run **OptGPSampler** (artificial-centering hit-and-run MCMC) over the resulting flux cone. Every
sample from a deletion's cone inherits that deletion's label. A shallow sklearn model
(HistGradientBoosting / LinearSVC / LogReg / RandomForest) trains at the *sample* level; sample-wise
predictions are averaged into a gene-level call. **That is explicit MCMC flux sampling plus shallow
ML — the un-amortized version of the stochastic head below.**

**Their yeast numbers, exactly as reported:**

- Yeast9: **1,159 single-gene deletions**, step size $k=5000$, **124 samples/cone** → **143,716
  samples × 4,130 fluxes**, total **4.43 GB**.
- Betaxanthin: data **is Cachera 2023**, "averaged across four nonclonal cultures." The screen has
  **4,223 deletions**, of which **N = 811** are Yeast9 metabolic genes — those 811 are the whole
  dataset. Classes **17.1 / 67.2 / 15.7 %** = **138 low / 545 medium / 128 high**, thresholds set
  *"qualitatively to label 67 % of samples as medium."*
- **Regression attempted and abandoned** — *"this proved challenging with the limited number of
  knockouts at the high and low ends."*
- **Headline: 69.8 % 3-class accuracy against a 67.2 % majority rate — +2.6 points.** High-producer
  accuracy 11.4–23.8 % baseline, best 29.5 % after rebalancing. **No correlation or AUC reported.**
- Split: a single class-stratified random 80/20 at gene level, held constant across models. **No CV,
  no seed, no gene- or pathway-disjoint holdout, split not released.**
- Deep models and PCA features did not help, which they attribute to fluxes being linearly correlated
  through $Sv=0$.
- Code/data: Zenodo `10.5281/zenodo.15518666`.

### Are they just predicting "medium"? No — and Table S6 is what proves it

The obvious suspicion, given 69.8 % against a 67.2 % majority, is that the classifier collapsed onto
the majority class. **It did not, and the paper does clarify this — in the SI, not the main text.**

**Table S4** confirms the class breakdown exactly: **N = 138 low, N = 545 medium, N = 128 high**,
summing to 811, so the majority rate is $545/811 = 67.20\,\%$ — matching their stated figure.

**Table S6** gives per-model **high-producer accuracy**, and a pure majority predictor would score
exactly **0 %** there:

| model | baseline | resampled | balanced | both |
| --- | --: | --: | --: | --: |
| HistGradientBoosting | 11.4 | 13.6 | 14.1 | 14.6 |
| LinearSVC | **23.8** | 24.0 | 27.2 | 26.8 |
| LogisticRegression | 23.3 | 26.2 | **29.5** | 28.9 |
| RandomForest | 18.1 | 11.7 | 19.1 | 18.4 |

So there is **real minority-class signal**: the model recovers 11–30 % of high producers, which
majority-prediction cannot do at all. **The correct criticism is not "they predicted medium" — it is
that 3-class accuracy is the wrong metric to report.** Decompose it on a stratified $n = 162$
(20 % of 811 → ~28 low / ~109 medium / ~25 high):

- majority-only → 108.9 correct = 67.2 %, with **0** high producers found
- best reported → 113.1 correct = 69.8 %, i.e. **+4.2 deletions**
- at 23.8 % high accuracy the model finds ~6 high producers — so it **gains ~6 on the minority class
  and gives back ~2 on low+medium**, netting the +4

That is a genuine but small effect, and the overall-accuracy number **hides it in both directions**:
it understates that they learned something about high producers, and it overstates how usable the
model is, because ~24 % high-producer accuracy means **missing three out of every four high
producers** — which is precisely the class a metabolic engineer cares about.

**This is the opening, and it is a methodological one rather than a "we beat their number" one.**
The right target is not 3-class accuracy. It is high-producer recall/precision and ranking — and we
can report what they cannot: **regression** (our noise ceiling is $r = 0.914$, reliability 0.836), so
Spearman and top-$k$ enrichment over the full ~4,735, which is the quantity that actually drives
strain design.

**Caveat on their $N$, unchanged:** Fig. 4b says $N=659$, Table S6 says $N=649$, and 20 % of 811 is
**162** — while 80 % of 811 is 649, so the SI most likely reports the *training* size. Their split is
also unreleased. **An exact head-to-head requires re-running their Zenodo code**; budget for it,
because the alternative is comparing against a number we cannot reproduce.

**Comparison protocol, in order:**

1. **Reproduce their setting** — intersect our Cachera build with Yeast9, verify against their 811
   (we get **906**; reconcile and document the 95-gene difference), apply their 3-class binning,
   class-stratified 80/20. Report accuracy and per-class accuracy.
2. **Beat the bar on their own turf** — same 811 genes, same classes.
3. **Report what they cannot** — regression on the full ~4,735 (they abandoned regression; our noise
   ceiling is $r=0.914$, reliability 0.836, median 15 replicates, so regression is *available* to us),
   and the ~3,900 non-metabolic deletions their cone has no representation for.
4. **Fix the evaluation** — their split is random, so related genes leak across the boundary. Report
   a gene-/pathway-disjoint split alongside and note the difference.

## Why FCL cannot follow us to double mutants

The decisive argument, and it lands exactly on goal 2. FCL needs **one independent MCMC chain per
genotype**, each run to mixing time. Scaling their own reported cost:

| | genotypes | samples @124/cone | data (scaled from their 4.43 GB) |
| --- | --: | --: | --: |
| Yeast9 singles (what they did) | 1,159 | 143,716 | **4.43 GB** |
| Yeast9 metabolic **pairs** $\binom{1161}{2}$ | **673,380** | 83,499,120 | **≈ 2.6 TB** |
| full-screen pairs $\binom{4735}{2}$ | 11,207,745 | — | no cone exists for non-GEM genes |

581× the data, and **673,380 separate random walks**. Their own Methods call flux sampling
"computationally costly because it requires running a random walk on a high-dimensional flux space
that needs to reach mixing time." The cost is per-genotype and does not amortize.

**An amortized model pays it once.** The perturbation operator consumes a *set* of deleted genes;
$\lvert p\rvert=1 \to 2$ is the same forward pass. That asymmetry — not accuracy on 811 single
deletions — is the actual contribution, and it is what makes the β-carotene double-mutant case
reachable at all. Stated positively: **an LP has no use for extra data; an amortized solver does.**

## The strongest available framing: an amortized flux sampler

If $v$ satisfies mass balance, the box and capacity, it is a point in the feasible polytope
$\mathcal{P}$ — a feasible flux distribution in the FBA sense. Whether it is a *sample* is a design
choice, and the choice is settled (decision 4):

- **Deterministic head** → $v$ is one point *selected* by the objective's implicit preference
  (parsimony + data fit). A selection, not a sample.
- **Stochastic head** → draw $z\sim q_\phi(z\mid H_{\mathrm{pert}})$, then $v=\mathrm{box}(z)$ is a
  genuine random variable supported on $\mathcal{P}$. **That is an amortized flux sampler**: a
  per-reaction flux distribution on every forward pass.

**Non-identifiability stops being a weakness to apologize for and becomes the object we report.**
Classical sampling (ACHR/OptGP in cobra — i.e. Merzbacher's method) samples $\mathcal{P}$ per
condition by MCMC, ignores data, and costs minutes per genotype. Ours is data-conditioned and
amortized. It yields a clean, publishable evaluation, per reaction:

$$\text{information gained from data}\;=\;\underbrace{\text{width}_{\mathrm{FVA}}(j)}_{\text{constraints alone}}\;-\;\underbrace{\text{width}_{\text{model posterior}}(j)}_{\text{constraints}+\text{data}}$$

If our interval is narrower than FVA's, the phenotype data added information; if equal, it did not.
Restrict to the FVA-licensed reactions (width $\le 1$; $n=2{,}818$ at $f=0.9$). This reuses the
decoder note's distributional machinery (CRPS / quantile heads) — same code, different head.

*Caveat to state plainly:* classical sampling targets a **uniform** distribution over $\mathcal{P}$;
ours targets whatever the data and priors induce. Different object, not a drop-in replacement.

## Stage 0 — the entity set (gene-centric; no enzyme layer)

Notation follows the manuscript contract (`paper/nature-biotech/sections/methods.tex`, Table
`tab:notation`), which already reserves $S,v$ and $\rho$. Constraint terms are $C_\bullet$ with
weights $\nu_\bullet$, extending Eq. (18).

| symbol | meaning | value |
| --- | --- | --: |
| $N$ | gene nodes | 6,607 |
| $m,\ r$ | metabolites, reactions | 2,806 · 4,131 |
| $\Pi\in[0,1]^{N\times r}$ | **soft gene→reaction catalysis** (replaces enzyme nodes) | — |
| $\gamma_g\in[0,1]$ | functional availability of gene $g$ under $p$ | — |
| $c_u,\ c_j$ | availability of catalytic unit $u$ / reaction $j$ | — |
| $E_g,\ \mathrm{MW}_g,\ k_{\mathrm{cat},gj},\ K_{M}$ | per-gene-product abundance, MW; **edge** kinetics on $(g,j)$ | — |
| $P_{\mathrm{avail}}$ | metabolic protein budget | scalar |
| $\omega_i(v)$ | turnover of metabolite $i$ (WS16 readout) | — |
| $\mu_i$ | chemical potential of metabolite $i$ ($\approx\ln$ concentration, up to affine) | — |
| $\mathcal{M}^{\dagger}$ | non-redundant balance rows | 2,593 |

Collisions deliberately avoided: $\alpha^{(\ell,a)}$ and $\beta_{i,t}$ are attention maps, so gene
availability is $\gamma$, **not** $\alpha$; $\tau_t$ is the perturbation *type*, so metabolite
turnover is $\omega$, **not** $\tau$; $\mu_i$ is a chemical potential, **not** growth rate (growth is
$v_{\mathrm{bio}}$); $E$ is the edge set of $G=(N,E)$ while $E_g$ is a scalar abundance.

$$H_{\mathrm{pert}}=\mathcal{T}_\psi\big(F_\theta(G),p\big)=\Big(h_{\mathrm{CLS}},\ \underbrace{h^{\mathrm{g}}_1,\dots,h^{\mathrm{g}}_{N}}_{\text{genes, }6607},\ \underbrace{h^{\mathrm{m}}_1,\dots,h^{\mathrm{m}}_{m}}_{\text{metabolites, }2806}\Big),\qquad h_\bullet\in\mathbb{R}^{d}.$$

**≈9.4 k tokens**, not the ~10.5 k the first pass proposed. Reactions are absent on purpose: a
reaction is a *process*, so its flux is a relation-level readout, not a node embedding.

### Why no enzyme nodes — hard chemistry, soft biology

Two objects were conflated in the first pass:

| object | encodes | epistemic status | treatment |
| --- | --- | --- | --- |
| $S$ (stoichiometry) | conservation of mass | **physical law**; a zero means *does not participate* | **HARD.** Never soften. |
| $\rho$ (GPR / catalysis) | which gene product catalyzes which reaction | **annotation**; a zero usually means *untested* | **SOFT.** Prior, not mask. |

**Promiscuity is the metabolic instance of "zeros mean untested"** — the same principle the
manuscript already argues for gene–gene graphs ("*a hard operator would commit to those uncertain
zeros, whereas the KL prior only pulls attention toward the known edges and lets the data overrule
it*"). If enzymes are nodes whose identity and count come from GPR's 1,065 AND-terms, the annotation
is baked into the **node set** — the most rigid part of the architecture — and representing a
promiscuous activity means inventing nodes at inference time, i.e. the fixed-$N$ problem (WS-NS1) all
over again. With genes as nodes and catalysis soft, promiscuity is just edge mass where the
annotation has none, and it becomes a **publishable output** rather than a caveat.

**All three things enzyme nodes were buying survive as functions of gene nodes:**

- **Complexes** (104 of 1,065 units are multi-gene) → the softmin below. No node required.
- **$k_{\mathrm{cat}}$** is a property of a *(catalyst, reaction) pair*, not of a catalyst — the same
  enzyme has different $k_{\mathrm{cat}}$ on different substrates, which *is* promiscuity. It belongs
  on the **edge** $(g,j)$, never on a node. MW is genuinely per-gene-product.
- **Protein budget** $\sum_g \mathrm{MW}_g E_g \le P_{\mathrm{avail}}$ sums over gene products —
  which is what proteomics measures anyway (Zelezniak measures 726 *proteins*).

**The one real cost:** GECKO's formulation is enzyme-centric, so we are no longer a drop-in
re-implementation and cannot validate line-by-line. Mitigation: keep a frozen enzyme-centric mode for
a one-time numerical agreement check against ecYeastGEM on wild type, then run gene-centric.

### Catalysis as an additive soft relation

$$\Pi=\underbrace{\Pi^{\mathrm{GPR}}}_{\text{hard, fixed, a FLOOR}}+\underbrace{\Delta\Pi}_{\text{soft, learned},\ \Delta\Pi\ge0},\qquad \text{penalty}\ \ \nu_{\mathrm{prom}}\lVert\Delta\Pi\rVert_1 .$$

Because $\Delta\Pi\ge0$, capacity on an annotated reaction can never be *removed* — the annotation is
a floor, not a suggestion, which is what stops mass drifting off known reactions. Promiscuity is
exactly $\Delta\Pi$: readable, rankable, reportable, with magnitude set by one prior. Start
$\nu_{\mathrm{prom}}$ large (conservative) and anneal down. This is the same cost structure as the
multiple attention heads per graph type we already run — one head bound to the annotation, one free —
with the advantage that promiscuity is **localized to an identifiable head** rather than smeared.

## Stage 1 — flux head, with the box exact and dynamic

Reaction $j$ reads its stoichiometric neighbourhood, the $\Pi$-weighted gene tokens that catalyze it,
and a learned reaction embedding $\mathbf{r}_j$:

$$z_j=\operatorname{MLP}_v\Big(\big[\,\textstyle\sum_g \Pi_{gj}h^{\mathrm{g}}_g\ \big\Vert\ \tfrac{1}{\sum_i\lvert S_{ij}\rvert}\sum_i \lvert S_{ij}\rvert\,h^{\mathrm{m}}_i\ \big\Vert\ \mathbf{r}_j\,\big]\Big),\qquad j=1,\dots,r.$$

Bounds are imposed **by construction, not by penalty**:

$$\boxed{\;v_j=v^{\ell}_j(\varepsilon,p,\mathrm{seq})+\big(v^{u}_j(\varepsilon,p,\mathrm{seq})-v^{\ell}_j(\varepsilon,p,\mathrm{seq})\big)\,\sigma(z_j)\;}$$

One line, four jobs, free: it enforces the box exactly; it enforces **directionality** for the 2,463
irreversible reactions ($v^\ell_j=0\Rightarrow v_j\ge0$); it folds **medium, deletion and capacity**
into the parameterization (decision 2 — the box is dynamic, not a constant); and it therefore
contributes **no term to $\mathcal{L}$**. Use $\tanh$ scaled to $[v^\ell,v^u]$ where
$\lvert v^u\rvert=1000$ so gradients do not saturate. For the stochastic head, $z_j$ is drawn from
$q_\phi$ rather than emitted deterministically — the box applies identically.

## Stage 2 — gene → capacity: where a deletion actually bites

Gene availability is **hard-set** by the perturbation for deletions, learned otherwise:

$$\gamma_g=\begin{cases}0,&g\ \text{deleted in}\ p\quad(\text{from the perturbation, not learned}),\\[2pt] \sigma\!\big(w^{\!\top}h^{\mathrm{g}}_g\big),&\text{otherwise (dosage, alleles, over-expression)}.\end{cases}$$

Per catalytic unit $u$ (an AND-term of genes $C_u$) and per reaction $j$ — complexes take a **min**
(the scarcest subunit limits), isozymes **add** (capacities sum):

$$c_u=\operatorname{softmin}_{g\in C_u}\gamma_g=-\tfrac1\beta\log\!\sum_{g\in C_u}e^{-\beta\gamma_g},\qquad c_j=\min\Big(1,\ \sum_{u\,:\,j\in\rho(u)}c_u\Big).$$

**Read the chain: $\gamma_g\to c_u\to c_j\to$ capacity $\to v_j$.** A deletion moves flux only if the
gene appears in $\rho$. Yeast9 holds 1,161 of our 6,607 genes; in the pigment screens only 906/4,735
(betaxanthin) and 811/4,474 (β-carotene) deleted ORFs are in it — so **~81 % of screened deletions
have no path through this chain at all** and act only through learned gene–gene coupling in
$F_\theta$. That is the coverage ceiling, stated mechanically. **The difference from FCL: those genes
still have a representation and still make a prediction.** For FCL they have no cone and are absent.

## Stage 3 — constraint residuals (the physics terms)

**(a) Mass balance, scale-relative.** Penalizing $\sum_i [Sv]_i^2$ is wrong because metabolites
operate at wildly different scales: a residual of $0.01$ is negligible on a metabolite carrying flux
$10$ and nonsense on one carrying $0.01$. Divide by throughput:

$$\omega_i(v)=\tfrac12\sum_{j=1}^{r}\big\lvert S_{ij}v_j\big\rvert\qquad\text{(half the total in+out flux — the turnover of }i)$$

$$C_{\mathrm{bal}}=\frac{1}{\lvert\mathcal{M}^{\dagger}\rvert}\sum_{i\in\mathcal{M}^{\dagger}}\left(\frac{[Sv]_i}{\operatorname{sg}\!\big(\omega_i(v)\big)+\epsilon}\right)^{2}.$$

Two details that are not cosmetic. $\mathcal{M}^{\dagger}$ is the $\operatorname{rank}(S)=2{,}593$
independent rows — the other 213 are linear combinations (conserved moieties, dead ends) whose
residual is already determined, so penalizing them adds noise, not information. And
$\operatorname{sg}(\cdot)$ is a **stop-gradient**: without it the cheapest way to shrink the ratio is
to shrink $\omega_i$, i.e. carry no flux anywhere. **A normalizer must not be an optimization
target.** $\omega_i$ does double duty — this denominator *and* the WS16 precursor-pool readout.

**(b) Capacity and budget, gene-centric.** Enzyme demand is a **function of $v$**, so $E$ is never
predicted:

$$E_g(v)=\sum_{j\,:\,\Pi_{gj}>0}\frac{\sqrt{v_j^{2}+\delta}}{k_{\mathrm{cat},gj}},\qquad P(v)=\sum_{g=1}^{N}\mathrm{MW}_g\,E_g(v).$$

($\sqrt{v^2+\delta}$ is a smooth $\lvert v\rvert$; the true absolute value is non-differentiable at
$0$, where 88 % of fluxes sit.) Both constraints are **one-sided hinges**:

$$C_{\mathrm{cap}}=\frac1N\sum_g\Big[\operatorname{relu}\big(E_g(v)-c_g\bar E_g\big)\Big]^2,\qquad C_{\mathrm{bud}}=\Big[\operatorname{relu}\big(P(v)-P_{\mathrm{avail}}\big)\Big]^2 .$$

**No reward for approaching the budget.** With 427 free internal reactions, "use up your budget" is
maximized by **futile cycles** — an internal loop burns enzyme, produces nothing, and scores well —
and it fights the parsimony prior that is the main defence against non-uniqueness. ecFBA never
rewards protein usage. Report $P(v)/P_{\mathrm{avail}}$ as a **diagnostic** (proteome- vs
substrate-limited, Domenzain's choline/putrescine distinction) instead.

**(c) Thermodynamics — one scalar per metabolite kills the loops.**

$$\mu_i=u^{\!\top}h^{\mathrm{m}}_i,\qquad \Delta_j(\mu)=\sum_{i=1}^{m}S_{ij}\,\mu_i,\qquad C_{\mathrm{th}}=\frac1r\sum_{j=1}^{r}\operatorname{relu}\big(v_j\Delta_j(\mu)+\epsilon\big).$$

This enforces $v_j\Delta_j\le0$ ("flux runs downhill"). Worked on the two-reaction toy: with
$\Delta_2=\mu_B-\mu_A$ and $\Delta_4=-\Delta_2$, requiring both $v_2\Delta_2\le0$ and
$v_4\Delta_4\le0$ under a running pathway $v_2>0$ forces $\Delta_2<0$, hence $v_4\le0$; but reaction
4 is irreversible so $v_4\ge0$, therefore $\boxed{v_4=0}$ — the futile cycle is eliminated with **no
integer variables**. The general reason: $\Delta$ is a *potential difference*, so it sums to zero
around any cycle and no cycle can run downhill everywhere. That is precisely what makes loopless-FBA
a mixed-integer program and makes this version cheap. It also gives the metabolite embedding a
physical coordinate, $\mu_i\approx\ln c_i$ up to affine.

## Stage 4 — data terms

$$\mathcal{L}_y=\sum_t w_t\sum_{b=1}^{B}m^{(b)}_t\,\ell_t\big(\hat y^{(b)}_t,y^{(b)}_t\big)\qquad\text{(Eq. 17, unchanged)}$$

Three heads read the flux layer rather than the token stack:

- **Fitness ⇒ biomass flux — the most abundant flux observation we have, and the bridge to goal 2.**
  Fitness *is* relative growth rate, so
  $\hat y_{\mathrm{fit}}=v_{\mathrm{bio}}(p)/v_{\mathrm{bio}}(\varnothing)$ supervises one coordinate
  of $v$ directly. We hold $\sim\!10^{7}$ fitness records — **orders of magnitude more flux-relevant
  data than any $^{13}$C-MFA study** — **and the double-mutant fitness records supervise
  $v_{\mathrm{bio}}$ at $\lvert p\rvert=2$.** That is what licenses evaluating the flux layer on
  doubles before any double-mutant production label exists. Cache the wild-type pass once per step.
- **Precursor pools (WS16)** $=\omega_i(v)$ at 13 metabolites, compared in **log fold-change vs wild
  type**, matching Domenzain's reported quantity, never absolute units.
- **Proteome ⇒ $E_g$** on Zelezniak/Messner strains: $\ell(\log E_g(v),\log E_g^{\mathrm{obs}})$ over
  726 measured proteins. **ecFBA structurally cannot do this check** — its $E_i$ are optimizer
  choices with nothing to compare against. We can.

## Stage 5 — identifiability

Essentially **zero** reactions are pinned by the constraints alone, so without these the flux vector
is arbitrary within a 1,538-dimensional polytope. The parsimony prior is the main lever:

$$C_{\mathrm{par}}=\frac1r\sum_j\sqrt{v_j^2+\delta}\qquad\text{(pFBA prior)}.$$

**The growth objective is dropped as a prior** (decision 3): fitness supervises $v_{\mathrm{bio}}$
directly from $\sim\!10^{7}$ records and, unlike an optimality assumption, is not blind to
non-metabolic genes. Where a near-optimality term is retained it is only because the FVA widths that
make any flux interpretable were computed *at 90 % of optimum*; state that as the reason, not
"cells maximize growth."

Three further mitigations for non-identifiability, cheapest first: supervise **pFBA** targets not
FBA; supervise only **measurable projections** (exchange fluxes, growth, the 13 precursor pools);
supervise the objective value, not the vector. The identifiability caveat is already in the
manuscript's Supplementary Note — **cite it, do not re-derive it.**

## The full objective

$$\boxed{\ \mathcal{L}\;=\;\underbrace{\mathcal{L}_y}_{\text{Eq. 17}}\;+\;\underbrace{\sum_{k=1}^{K}\lambda_k\Omega_k}_{\text{Eq. 18 graph priors}}\;+\;\underbrace{\nu_{\mathrm{bal}}C_{\mathrm{bal}}+\nu_{\mathrm{cap}}C_{\mathrm{cap}}+\nu_{\mathrm{bud}}C_{\mathrm{bud}}+\nu_{\mathrm{th}}C_{\mathrm{th}}}_{\text{physics}}\;+\;\underbrace{\nu_{\mathrm{par}}C_{\mathrm{par}}+\nu_{\mathrm{prom}}\lVert\Delta\Pi\rVert_1}_{\text{identifiability + promiscuity}}\ }$$

**There is no $C_{\mathrm{box}}$ term** — the box is enforced by the parameterization, so it cannot
appear in the objective. The attention regularizer extends to metabolism by **support only**:

$$\tilde{A}^{(\mathrm{met})}=\operatorname{rownorm}\big(\lvert S\rvert^{\!\top}\lvert S\rvert>0\big).$$

$\lvert S\rvert$, not $S$ — attention rows are non-negative and sum to one, so only the *pattern* of
$S$ is representable as an attention prior; the *signed values* live in $C_{\mathrm{bal}}$ and
$\Delta_j$. **$S$ therefore appears twice in different roles**: pattern as prior, values as the
mass-balance operator. Not an either/or.

## The exactness budget

| constraint | how enforced | cost |
| --- | --- | --- |
| $v^{\ell}\le v\le v^{u}$, directionality, medium, capacity | **exact**, dynamic reparameterization | free |
| $Sv=0$ | **soft**, $C_{\mathrm{bal}}$ | one weight |
| capacity, budget | **soft**, one-sided hinge | two weights |
| $v_j\Delta_j\le0$ (loops) | **soft**, learned potential | one weight |

The alternative spends exactness on mass balance instead: $v=\mathcal{N}z$ with
$\mathcal{N}\in\mathbb{R}^{4131\times1538}$, $S\mathcal{N}=0$, giving $Sv=0$ identically and a
$2.7\times$ narrower head — **but a null-space projection does not preserve a box**, so bounds would
revert to a penalty and directionality would no longer be guaranteed. Decoupled constraints go to the
parameterization; **coupled** ones ($Sv=0$, the shared budget) go to the objective.

## Reporting: feasibility is a measurement, not a hope

Every term is also a per-sample diagnostic. Logging them is what lets us *say* a predicted vector is
a flux vector rather than assert it — and it is what FCL cannot report, since its samples are
feasible by construction but carry no model-predicted flux:

$$\text{feas}_{\mathrm{bal}}=\operatorname{median}_i\frac{\lvert[Sv]_i\rvert}{\omega_i(v)},\qquad \text{feas}_{\mathrm{bud}}=\frac{P(v)}{P_{\mathrm{avail}}},\qquad \text{feas}_{\mathrm{th}}=\frac1r\sum_j\mathbb{1}\big[v_j\Delta_j>0\big].$$

Report alongside every flux-derived claim, restricted to the reactions the FVA mask licenses.

## $k_{\mathrm{cat}}$ and $K_M$ — different jobs, different sources

**They are not the same kind of parameter and they are not required for the same reason.** Both are
edge features on $(g,j)$ — the prediction input is a *(protein sequence, substrate)* pair, which is
itself an argument for the edge and against an enzyme node.

| | what it is | where it enters | why required |
| --- | --- | --- | --- |
| $k_{\mathrm{cat}}$ | max turnover, h$^{-1}$ | capacity $\lvert v_j\rvert\le k_{\mathrm{cat},gj}E_g$ | **unconditionally** — the only source of magnitude in the whole model; **4,129 of 4,131** Yeast9 bounds carry none |
| $K_M$ | half-saturation, mM | saturation $\eta^{\mathrm{sat}}$ inside the *box*, not a loss term | **only because we allow promiscuity** — see below |

**$k_{\mathrm{cat}}$ tightens the polytope; $K_M$ by itself does not.** Adding capacity constraints
is what makes $\mathcal{F}_{\mathrm{ec}}\subseteq\mathcal{F}_{\mathrm{FBA}}$ — it removes flux
distributions that are stoichiometrically fine but need more protein than exists. $K_M$ belongs to a
rate law $v=k_{\mathrm{cat}}E\,c/(K_M+c)$ that needs a **concentration** $c$; introducing it adds a
parameter *and* a free variable, so on its own it **enlarges** the model's freedom. That is the trap:
$K_M$ looks like extra constraint and is actually extra slack.

**What promotes it to required.** Wu 2026 (`wuSystematicallyExploringYeast2026`, *Nature Catalysis*)
measured underground vs known reactions across three independent predictors: median $K_M$ is **~2×
higher** (0.25 vs 0.11 mM Boost_KM; 0.21 vs 0.11 UniKP; 0.25 vs 0.07 EITLEM) while
**$k_{\mathrm{cat}}$ distributions are indistinguishable** (DLKcat 5.52 vs 5.28 s⁻¹; TurNuP 10.09 vs
11.01). Underground metabolism is *"dominated by variations in $K_m$ and not $k_{cat}$."* So if
promiscuous edges carry the same $k_{\mathrm{cat}}$ as native ones, a **$k_{\mathrm{cat}}$-only model
gives promiscuous flux away for free** — $\Delta\Pi$ routes at full native capacity and only the
$\ell_1$ prior resists. Affinity is what limits promiscuous flux in the cell:

$$\lvert v_j\rvert\le k_{\mathrm{cat},gj}\,E_g\,\eta^{\mathrm{sat}}_j,\qquad \eta^{\mathrm{sat}}_j=\prod_{i\in\mathrm{sub}(j)}\frac{c_i}{K_{M,ij}+c_i},\qquad c_i=e^{\,u^{\!\top}h^{\mathrm{m}}_i}.$$

**The requirement is therefore contingent, and that is worth stating plainly: drop promiscuity and
$K_M$ becomes optional again.** It also needs no new machinery — the concentration is the
$\mu_i\approx\ln c_i$ already introduced for thermodynamics, so one quantity satisfies both, and
$\eta^{\mathrm{sat}}$ folds into the dynamic box rather than adding a loss weight.

**Where the numbers actually come from — and ecYeastGEM is NOT a source of $K_M$.** GECKO's
formulation uses only $k_{\mathrm{cat}}$, so mirroring ecYeastGEM yields
$k_{\mathrm{cat}}$/MW/$P_{\mathrm{avail}}$ and **no $K_M$ at all**. Sourcing order, published before
predicted, with a per-value provenance tag so we can ablate "published only" vs "published +
predicted":

| parameter | 1. published | 2. database | 3. predicted (gaps only) |
| --- | --- | --- | --- |
| $k_{\mathrm{cat}}$ | **ecYeastGEM** (GECKO-curated) — not yet mirrored | Open Enzyme Database (`yuanOpenEnzymeDatabase2026`, **already in our mirror**) | DLKcat · TurNuP · KcatNet |
| $K_M$ | **none — GECKO has no $K_M$** | Open Enzyme Database; BRENDA / SABIO-RK | Boost_KM · UniKP · EITLEM |

**Promiscuous edges have no published $k_{\mathrm{cat}}$ or $K_M$ by definition**, so if we want the
model to *use* a promiscuous activity, prediction is the only possible source. Promiscuity and
kinetic-parameter prediction are the same workstream, not two.

**Carry uncertainty rather than point values.** A predicted $k_{\mathrm{cat}}$ that is too low
forbids feasible flux; too high is vacuous. Push the predictor's posterior quantile into the box:

$$\bar v^u_j \;=\; \min\Big(v^u_j(\varepsilon),\ c_j(p)\cdot \textstyle\sum_g \Pi_{gj}\,\hat k^{(q)}_{\mathrm{cat},gj}\,\bar E_g\,\eta^{\mathrm{sat}}_j\Big)$$

$q$ is a single interpretable conservatism dial: $q=0.1$ → the model may only use capacity it is
confident exists (**start here**); $q=0.5$ → the usual ecFBA point estimate; $q\sim\mathrm{Uniform}$
resampled per forward pass → **the box itself becomes stochastic**, which composes with the
amortized-sampler framing and propagates parameter uncertainty into the predictive interval. That
last one is the principled version and the one to aim for.

**Activation policy:** enable $\eta^{\mathrm{sat}}$ only on metabolites with anchored concentrations
and set $\eta^{\mathrm{sat}}=1$ elsewhere. We are partly anchored already — Mülleder gives
**absolute intracellular mM for 19 amino acids across 4,678 strains**, Zelezniak relative pools for
~50 metabolites.

## GECKO: comparison, not parity

We are not trying to be ecYeast, but we must be able to compare. Parity would mean reproducing
GECKO's enzyme-centric formulation, which conflicts with the gene-centric choice. Comparison needs
only a shared interface — four comparisons of increasing interest:

1. **Wild-type flux agreement** — our $v$ on WT vs ecYeastGEM pFBA, correlated over FVA-licensed
   reactions only (width $\le1$; $n=2{,}818$ at $f=0.9$).
2. **Deletion growth** — our predicted fitness vs ecFBA growth ratio on the 1,161 Yeast9 genes.
3. **Enzyme allocation** — our $E_g$ vs measured proteome. **ecFBA structurally cannot do this.**
4. **The comparison that actually matters** — on the ~81 % of screened deletions outside Yeast9,
   ecFBA (and FCL) predict *no effect at all*. We predict something. That is not a tie-break; it is a
   category difference, and it is where the architecture earns its keep.

## Sizing (measured) — `YeastGEM()` v9.0.2

| quantity | value |
| --- | --: |
| `S` shape (metabolites × reactions) | 2,806 × 4,131 |
| `S` nonzeros | 15,567 (**0.134 %** dense) |
| `rank(S)` | 2,593 |
| **nullity = R − rank** | **1,538** |
| reversible (`lb<0<ub`) · irreversible (`lb ≥ 0`) | 1,668 · 2,463 |
| boundary/exchange-like (1 metabolite) | 274 |
| reactions carrying a GPR | 2,709 / 4,131 |
| GPRs with `or` · `and` · both | 591 · 209 · 50 |
| catalytic units (AND-terms) | 1,065 (104 multi-gene) — **used as a function, not as nodes** |
| **genes in Yeast9** | **1,161** — of 6,607 gene nodes (**17.6 %**) |

## The coverage ceiling, and the ablation it converts into

- betaxanthin (Cachera): **906 of 4,735** deleted ORFs are Yeast9 metabolic genes (**19.1 %**)
- β-carotene (Ozaydin): **811 of 4,474** (**18.1 %**)

**~81 % of the deletions we want to rank have no direct handle on the flux layer.** That bounds what
an enzyme-constrained layer can contribute and converts into a cheap ablation: **does the flux layer
improve ranking on the metabolic ~19 % more than on the non-metabolic ~81 %?** If yes, that asymmetry
*is* the mechanistic-grounding result. If no, the constraint layer is decoration. Measure it.

Merzbacher's 811 is the β-carotene overlap number *and* their betaxanthin subset size, because both
derive from the same Yeast9 gene set. The metabolic subset is the natural head-to-head arena; the
non-metabolic complement is the arena they cannot enter.

## Pathway addition is the same blocker as WS-NS1

Adding the crt pathway means new metabolites (phytoene, lycopene, β-carotene), new reactions, and new
columns in `S`. The model is fixed-$N$ with no rows for new entities — **this is exactly WS-NS1**, so
"accommodate adding pathways" and "additive perturbation operator" are one structural change, not
two. Near-term trick avoiding full variable-$N$: **pre-extend the entity universe** with the union of
cassette metabolites/reactions across our pigment datasets and **mask them off for strains lacking
the cassette**. Note Merzbacher sidesteps this entirely by never representing the pathway — which is
why their model has no mechanistic route to the product at all.

## Gradient hazards (each is a real failure mode)

1. **Term scales differ wildly** — $C_{\mathrm{bal}}$ is dimensionless by construction,
   $C_{\mathrm{bud}}$ is in $\mathrm{g\,gDW^{-1}}$ squared, $\mathcal{L}_y$ is whatever the phenotype
   is. This is exactly the failure the 019 joint runs hit (un-normalized morphology MSE swamping
   expression). **Normalize every term to dimensionless before weighting**, then tune $\nu$.
2. **$\lvert v\rvert$ at $v=0$** — smooth it; 88 % of reactions sit exactly there.
3. **$\operatorname{softmin}$ temperature $\beta$** — too soft and a complex behaves like a mean (a
   deletion stops being lethal); too hard and gradients vanish to all but one subunit. Anneal.
4. **$\omega_i$ in a denominator** — stop-gradient, or the model carries no flux.
5. **The wild-type pass** for $v_{\mathrm{bio}}(\varnothing)$ is per-step, not per-sample — cache it.
6. **$\nu_{\mathrm{prom}}$ annealing** — start conservative; a cheap $\Delta\Pi$ early lets the model
   invent catalysis instead of learning biology.

## Build order

1. **Mirror ecYeastGEM** ($k_{\mathrm{cat}}$, MW, $P_{\mathrm{avail}}$ — **not $K_M$; GECKO has
   none**) with sha256 provenance — **blocking; nothing below works without parameters.** `grep` for
   `ecYeast|GECKO|kcat` over `torchcell/` returns nothing today: we hold plain yeast-GEM 9.0.2 via
   cobra (`torchcell/metabolism/yeast_GEM.py`) and no turnover numbers, no MWs, no protein pool.
   Acquire first: Sánchez 2017 (GECKO), Domenzain 2022 (GECKO 2.0 — where ecYeastGEM's
   $k_{\mathrm{cat}}$/$P_{\mathrm{avail}}$ come from), Elsemman 2022 (**already in Zotero, just
   uncollected — one-line fix**), Chen/Li/Nielsen 2022.
   **1b. $K_M$ comes from a different source entirely** — Open Enzyme Database
   (`yuanOpenEnzymeDatabase2026`, already mirrored) and BRENDA/SABIO-RK, then Boost_KM/UniKP/EITLEM
   for gaps. This is only blocking **if promiscuity is enabled**; without $\Delta\Pi$ the model runs
   on $k_{\mathrm{cat}}$ alone with $\eta^{\mathrm{sat}}=1$.
2. **Load the thermodynamics we already have** — `data/databases/model_metDeltaG.csv` holds 2,389
   real $\Delta_fG'^\circ$ values (85.1 % of 2,806 metabolites); the SBML drops them.
3. **Fix the metabolic incidence defects** — same code path.
   `_build_metabolic_incidence` (`equivariant_cell_graph_transformer.py:1150`) still reads
   `.edge_index` (`:1170`, `:1173`) and assumes `rmr_ei[0]` = metabolite / `rmr_ei[1]` = reaction
   (`:1200-1201`), while `cell_data.py:169` emits `.hyperedge_index` and the bipartite processor
   (`cell_data.py:456+`) emits `("reaction","rmr","metabolite")` transposed. Never caught because the
   test fixture fabricates the expected names — **the fiction is what hid the defect.** Replace it
   with a real `to_cell_data` fixture.
4. **Metabolite nodes** in the encoder; $\Pi$ as a soft gene→reaction relation.
5. **Flux head:** dynamic-box reparameterization, stochastic $q_\phi$, soft $C_{\mathrm{bal}}$ with
   rank-deficient rows down-weighted; log feasibility.
6. **Capacity layer:** $E_g$ derived from $v$; over-budget hinge only.
7. **Validate:** $E_g$ vs Zelezniak proteome; $v$ vs pFBA; precursor pools vs WS16; **posterior width
   vs FVA width.**
8. **Compare:** the four-step Merzbacher protocol.
9. **Ablate:** flux-layer benefit on metabolic (19 %) vs non-metabolic (81 %) deletions.

## Open — the β-carotene double-mutant route

Fit on double-mutant **fitness**, then apply to double-mutant **production**. Two things need pinning
down before this is a plan rather than an intention.

- **Which double-mutant fitness data — still to pin down.** Costanzo/Kuzmin DMF/TMI is in the
  database at ~$10^7$ records and would supervise $v_{\mathrm{bio}}$ at $\lvert p\rvert=2$ directly.
  If instead this means newly collected in-house data, it is not in the repo yet —
  `experiments/019-echo-crispr-array` currently holds a **single**-strain colony assay
  (`reference_smf_12panel.csv` is Costanzo/Kuzmin *single*-mutant fitness).
- **No in-house β-carotene data has been added yet** (author, 2026.07.26), and the published
  β-carotene labels are singles only — Ozaydin 2013 is the haploid BY4741 deletion collection. So
  **there is no double-mutant β-carotene label anywhere**, and the production step is a genuine
  extrapolation whose evaluation must be designed *before* training rather than after.
- **The amortized-sampler framing supplies a label-free evaluation, and this is why it matters
  here.** $\text{width}_{\mathrm{FVA}}(j)-\text{width}_{\text{posterior}}(j)$ is computable on a
  double-deletion cone **with no production label at all**, so we can demonstrate the model gains
  information on doubles before any β-carotene measurement on doubles exists. The sequence is:
  (1) fit $v_{\mathrm{bio}}$ on double-mutant fitness; (2) show posterior narrowing on doubles vs
  FVA; (3) predict β-carotene on doubles as a *ranked* output; (4) let that ranking select which
  doubles are worth measuring in-house. Step 4 is the inverse-design payoff and it makes the missing
  data a deliverable rather than a blocker.
