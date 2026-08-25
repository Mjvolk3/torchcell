---
id: 0ffzc4o0pvn2k9bm6afe81d
title: Distributional
desc: ''
updated: 1785532045851
created: 1785532045851
---

## 2026.07.27 - The energy score: what it is, why it fits here, what it costs

Written for the `_007` dist axis, before `energy` shipped: propriety from first principles, what
`X` actually is, how to choose `m` against `k`, and PIT. This is the derivation behind the mode;
the 2026.07.31 section below records what the shipped module does. One reconciliation between
them: the per-row masking in section 8 is the design as written, and the `feature_mask` gap that
section reports is exactly where that design stops holding for the v9 masked objective.

### 0. What is a "scoring rule", and what makes one "proper"?

**Scoring rule.** A function $S(F,\mathbf{y})$ that grades a *probabilistic forecast* $F$
against the outcome $\mathbf{y}$ that actually happened. It "scores" the forecast the way
you'd score a bet after seeing the result. Ours are **negatively oriented** -- lower is
better, so they double as loss functions.

The subtlety: the forecast is a whole distribution, the outcome is a single point. Many
ways to compare them are *gameable*.

**Expected score.** If outcomes really come from $G$, the score you expect by reporting
$F$ is

$$S(F,G)\;=\;\mathbb{E}_{\mathbf{Y}\sim G}\big[S(F,\mathbf{Y})\big].$$

**Proper** means honesty is optimal:

$$S(G,G)\;\le\;S(F,G)\qquad\text{for every }F.$$

**Strictly proper** means honesty is *uniquely* optimal -- equality only when $F=G$. This
is the property that makes a scoring rule usable as a training loss: the model cannot
lower its loss by reporting anything other than its true belief.

#### Why impropriety is a real hazard, not a technicality

Our own `point` (MSE) head is the example. MSE grades only the predicted mean, so it is
proper **for the mean functional** and says *nothing whatsoever* about spread -- a model
can report any $\sigma$ it likes at zero cost. There is no incentive to be honest about
uncertainty because uncertainty is not scored.

Gaussian NLL fails differently: it *is* proper, but its $1/\sigma^2$ term lets the model
lower the loss on hard points by inflating $\sigma$, which simultaneously shrinks those
points' gradient on $\mu$ (§7). Proper, but badly conditioned.

#### Why the energy score's second term must be there

$$\mathrm{ES}_\beta(F,\mathbf{y})=\underbrace{\mathbb{E}_F\lVert \mathbf{X}-\mathbf{y}\rVert_2^{\beta}}_{\text{(a) be close to the outcome}}-\underbrace{\tfrac12\,\mathbb{E}_F\lVert \mathbf{X}-\mathbf{X}'\rVert_2^{\beta}}_{\text{(b) be spread out}}$$

Term (a) alone would be **improper**: it is minimized by collapsing $F$ to a point mass
(any spread only adds expected distance), so a model scored on (a) would always report
zero uncertainty. Term (b) is a *reward* for self-spread -- it pays the forecast for
disagreeing with itself. The two are balanced so that the optimum sits exactly at the
true distribution: too narrow loses more on (b) than it gains on (a), too wide the
reverse. **That balance is what "proper" buys.**

Gneiting & Raftery (2007) prove $\mathrm{ES}_\beta$ is strictly proper for
$\beta\in(0,2)$ on distributions with finite $\beta$-th moment. The endpoint is
instructive -- at $\beta=2$ the norms expand and everything but the mean cancels:

$$\mathrm{ES}_2(F,\mathbf{y})=\lVert\boldsymbol\mu_F-\mathbf{y}\rVert_2^2+\text{const},$$

i.e. it degenerates into MSE-on-the-mean and stops being strictly proper. **Use
$\beta=1$.**

### 1. Setup -- what our current losses actually score

A strain $s$ deletes one gene and we predict a **vector** $\mathbf{y}_s\in\mathbb{R}^{F}$,
$F = 6{,}169$ reporter genes. Both current probabilistic heads score that vector **one
coordinate at a time**:

$$
\mathcal{L}_{\text{CRPS}}=\sum_{f=1}^{F}\mathrm{CRPS}\!\left(F_{s,f},\,y_{s,f}\right),
\qquad
\mathcal{L}_{\text{pinball}}=\sum_{f=1}^{F}\sum_{k=1}^{K}\rho_{\tau_k}\!\left(y_{s,f}-\hat q_{s,f,k}\right)
$$

Every term touches exactly one $f$. So both are **sums of marginal scores**.

Critically: *summed CRPS is proper for each **marginal** separately, but NOT for the
joint.* It is minimized by getting all $F$ marginals right, and **any** joint carrying
those marginals attains the same minimum. Two models with identical per-gene marginals -- one that knows genes $i$ and $j$ co-respond, one that treats them as independent -- receive
**exactly the same loss**.

That is a real gap for us, because our two metrics live on two different axes:

| metric | axis | what it needs |
|---|---|---|
| `pearson_per_feature` | across strains, per gene | marginals |
| `pearson_per_instance` | across genes, per strain | **the joint** |

**Nothing we currently train optimizes `pearson_per_instance`.** It has been reported all
along as a diagnostic while no loss ever targeted it.

### 2. The energy score

$$
\boxed{\;
\mathrm{ES}_\beta(F,\mathbf{y})
=\mathbb{E}_F\lVert \mathbf{X}-\mathbf{y}\rVert_2^{\beta}
-\tfrac12\,\mathbb{E}_F\lVert \mathbf{X}-\mathbf{X}'\rVert_2^{\beta},
\qquad \mathbf{X},\mathbf{X}'\overset{iid}{\sim}F,\;\beta=1
\;}
$$

**It is exactly the multivariate generalization of CRPS.** In one dimension
$\lVert\cdot\rVert_2=|\cdot|$ and

$$\mathrm{ES}_1(F,y)=\mathbb{E}|X-y|-\tfrac12\mathbb{E}|X-X'|=\mathrm{CRPS}(F,y).$$

We already depend on this identity: `tests/torchcell/losses/test_distributional.py`
validates `gaussian_crps` against Monte-Carlo $\mathbb{E}|X-y|-\tfrac12\mathbb{E}|X-X'|$.
ES is that same expression with $|\cdot|\to\lVert\cdot\rVert_2$ -- which makes the existing
test the natural oracle for the new implementation (§8).

**Where the coupling comes from.** $\lVert\mathbf{X}-\mathbf{y}\rVert_2=\sqrt{\sum_f
(X_f-y_f)^2}$ does **not** decompose over $f$ -- the square root binds the coordinates.
Replace $\lVert\cdot\rVert_2$ with $\sum_f|\cdot|$ and you recover summed CRPS and lose all
joint sensitivity. The square root *is* the mechanism.

### 3. What `X` is -- NOT an embedding

$\mathbf{X}$ is a **sample of the prediction**: one draw from the predictive distribution
over the target, living in $\mathbb{R}^{F}$ -- the same shape and units as $\mathbf{y}$
(log2 ratios for 6,169 reporter genes). $\mathbf{X}-\mathbf{y}$ is plain elementwise
subtraction of two 6,169-vectors; $\lVert\cdot\rVert_2$ then collapses the genes to one
scalar per (sample, strain).

The thing that *is* embedding-like is $\mathbf{V}$ (§4): the low-rank factor matrix. It
carries one $k$-dimensional vector **per gene**, and

$$\mathrm{Cov}(y_i,y_j)=\mathbf{V}_i\!\cdot\!\mathbf{V}_j\quad(i\neq j),$$

so genes whose $\mathbf{V}$ rows align are predicted to co-respond. $\mathbf{V}$ is a
**learned co-response embedding whose inner products are predicted covariances** -- an
interpretable output in its own right, comparable against the graph adjacencies.

### 4. The head -- cheaper than it first looks

Full covariance over $F=6{,}169$ is $3.8\times10^{7}$ entries per strain: impossible. Use
**low-rank plus diagonal**:

$$
\boldsymbol\Sigma = \mathbf{D}+\mathbf{V}\mathbf{V}^{\!\top},\qquad
\mathbf{D}=\mathrm{diag}(\sigma_1^2,\dots,\sigma_F^2),\quad
\mathbf{V}\in\mathbb{R}^{F\times k},\;k\approx4\text{-}16
$$

$\boldsymbol\Sigma$ is never formed. Sampling is

$$\mathbf{x}=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\varepsilon+\mathbf{V}\mathbf{z},
\qquad\boldsymbol\varepsilon\sim\mathcal N(0,I_F),\;\mathbf{z}\sim\mathcal N(0,I_k)$$

Cost $O(Fk)$ per sample. `param_dim` goes from 2 ($\mu,\sigma$) to $2+k$ per feature -- at
$k=8$ that is **10 per gene, versus the quantile head's K=19**. The energy head is
*smaller* than one we already run.

**Design choice -- global vs per-strain $\mathbf{V}$.** A per-strain $\mathbf{V}$ is
$F\times k$ *outputs* per row (expensive, conditional dependence). A **global** learned
$\mathbf{V}$ is $F\times k$ *parameters* total (~49 k at $k=8$) and says "there is one
gene-gene co-response structure." Start global: far cheaper, directly interpretable, and
it is the hypothesis we actually want to test.

### 5. Estimator, and how to pick `m` vs `k`

$$
\widehat{\mathrm{ES}}
=\frac1m\sum_{i=1}^{m}\lVert\mathbf{x}*i-\mathbf{y}\rVert_2
-\frac{1}{2m(m-1)}\sum*{i\neq j}\lVert\mathbf{x}_i-\mathbf{x}_j\rVert_2
$$

The $m(m-1)$ denominator (not $m^2$) excludes the zero self-distances and keeps the
estimator **unbiased**. Gradients flow through the samples by reparameterization.

```python
# mu [B,F]   sigma [B,F]   V [F,k] (global)   y [B,F]
eps = torch.randn(B, m, F)
z   = torch.randn(B, m, k)
X   = mu[:,None] + sigma[:,None]*eps + z @ V.T          # [B,m,F]

term1 = (X - y[:,None]).norm(dim=-1).mean(1)            # E‖X−y‖    → [B]
D     = torch.cdist(X, X)                               # ‖xi−xj‖   → [B,m,m]
term2 = D.sum((1,2)) / (2*m*(m-1))                      # ½E‖X−X'‖  → [B]
es    = term1 - term2
```

Use `cdist`; the naive `X[:,:,None]-X[:,None]` is $[B,m,m,F]$ = 197 M floats (~790 MB) at
$B{=}32,m{=}10,F{=}6169$.

#### `m` and `k` answer different questions -- this is the key point

- **$k$ decides whether dependence is *representable*.** At $k=0$ the predictive
  distribution is diagonal; ES still couples coordinates *in the score*, but the family
  cannot express correlation, so there is nothing for the coupling to learn.
- **$m$ only controls *estimator variance*.** The estimator is unbiased for any $m\ge2$;
  $m$ affects the noise of the gradient, not what can be learned.

So **to conclude whether modeling the joint helps, vary $k$, not $m$** -- the ablation is
$k\in\{0,8\}$ at fixed loss and fixed everything else. Set $m$ merely large enough that it
is not the bottleneck:

| | value | reason |
|---|---|---|
| $m_{\text{train}}$ | 16-32 | reparameterized gradients also average over batch and steps |
| $m_{\text{eval}}$ | 256+ | no gradients needed, so precision is nearly free -- and the *comparison* must not be limited by estimator noise |

Decoupling train/eval $m$ is what makes the $k=0$ vs $k=8$ verdict trustworthy.

### 6. What it would tell us that CRPS/pinball cannot

1. Whether `pearson_per_instance` is trainable at all, or structurally capped.
2. Whether gene-gene co-response is learnable from ~1,236 training strains, or whether
   $F\gg n$ makes the joint hopeless.
3. **Whether the graph channel helps the *joint* more than the marginals.** The graph
   prior asserts that neighbors co-respond -- and every loss we have run is blind to
   co-response. It is possible graphs look weak on `per_feature` precisely because that
   metric cannot see their contribution. $\mathbf{V}\mathbf{V}^{\!\top}$ can be compared
   directly against the adjacencies.

### 7. PIT and coverage -- the shared calibration metric

Every probabilistic head emits per-label uncertainty and **none of it is currently
scored**: `gaussian` gives $(\mu,\sigma)$ per gene per strain, `quantile` gives 19
quantiles, `energy` gives samples. One diagnostic covers all three.

**PIT = Probability Integral Transform.** Push the observation through its own predicted
CDF:

$$\mathrm{PIT}_{s,f}=\hat F_{s,f}\!\left(y_{s,f}\right)\in[0,1]$$

**Rosenblatt (1952):** if $\hat F$ is the true predictive distribution and is continuous,
then $\mathrm{PIT}\sim\mathrm{Uniform}(0,1)$. So histogram the PIT values over all
(strain, gene) pairs and read the shape:

| PIT histogram | meaning |
|---|---|
| **flat** | calibrated |
| **U-shaped** (mass at 0 and 1) | intervals too NARROW -- overconfident, $\sigma$ too small |
| **∩-shaped** (hump at 0.5) | intervals too WIDE -- underconfident, $\sigma$ too large |
| **skewed / shifted** | biased mean, not a spread problem |

Per head: `gaussian` → $\Phi\!\big((y-\mu)/\sigma\big)$; `quantile` → interpolate the
empirical CDF through the 19 knots; `energy` → $\frac1m\sum_i\mathbb 1[x_{i,f}\le y_f]$.

**Coverage** is the readable summary: for a nominal central interval at level $\alpha$,

$$\text{coverage}_\alpha=\frac{1}{|S|F}\sum_{s,f}\mathbb 1\!\left[y_{s,f}\in\hat I^{\alpha}_{s,f}\right],
\qquad\text{calibrated}\Rightarrow\text{coverage}_\alpha\approx\alpha.$$

**This is also the direct test of the Gaussian-NLL $\sigma$-collapse** we adopted from
Seitzer 2022 *without ever running it here* (verified: no config and no code history ever
used `nll`/`beta_nll`). Collapse inflates $\sigma$, so it shows up unmistakably as the
∩-shaped PIT and coverage far ABOVE nominal -- 80 % intervals containing ~95 %. Adding an
`nll` arm converts an inherited literature assumption into a measurement.

### 8. Implementation checks

- **Oracle:** at $F=1$, ES must equal `gaussian_crps` to within Monte-Carlo error. Reuses
  the existing CRPS test as ground truth.
- **Unbiasedness:** $\widehat{\mathrm{ES}}$ at $m=2$ and $m=512$ must agree in
  expectation; only the variance may differ.
- **Propriety, empirically:** score a fixed sample against a family of candidate $F$ with
  varying $\sigma$; the minimum must land on the generating $\sigma$, not at 0 or $\infty$.
- **Masks are per-ROW** (`masks: {head: bool [B]}`), not per-feature, so every supervised
  strain has all $F$ coordinates present. No ragged norms, no per-strain dimension
  normalization -- masking is just a row filter before the norm.

### 9. Not to forget

- $\beta=1$ throughout; $\beta\to2$ degenerates to MSE-on-the-mean (§0).
- The $k=0$ vs $k>0$ contrast is the experiment; $m$ is only a precision knob.

## 2026.07.31 - Make "score only the genes still hidden" one reduction that every distributional mode obeys

The round's teacher-forced masked-label objective (v9) hands the model `m` true gene values and asks it to predict the rest. That measures nothing unless the loss ignores the revealed genes: a revealed gene is model INPUT at that step, so scoring it rewards copying input to output -- train loss collapses, train Pearson inflates, and none of it transfers to validation, where nothing is revealed. `masked_mean` is that restriction, and it lives here rather than in the training script so the point path and the five probabilistic modes reduce through ONE definition instead of five copies that can drift apart silently (commit `73224a55`, +49/-5).

- **Why this module is distributional at all.** Under MSE the optimal point prediction is the conditional mean, so `point` actively *rewards* mean-collapse -- exactly the failure the expression task keeps hitting. The five probabilistic modes replace it with proper scoring rules in y-units: `gaussian` / `laplace` closed-form CRPS (the Laplace twin isolates whether a quantile head's edge is the distributional loss or merely a median point estimate), `quantile` pinball over K=19 taus, `energy` the multivariate generalization with a global low-rank `V [F, k]`, and `nll_gaussian` deliberately UNPATCHED as the negative control carrying the `1/sigma^2` sigma-collapse pathology. `pit_values` / `coverage` / `pit_ks` are the one diagnostic that puts all of them on the same `[0, 1]` scale, since the losses themselves are not comparable to each other. Default `rank=32` is not a guess: `residual_covariance_diagnostic.json` puts the participation-ratio effective rank of the reproducible residual structure at **32.78** (cumulative variance 37.6% at k=8, 59.1% at k=32).
- **What the reduction has to satisfy.** `feature_mask` is `[B, F]` (True = score this entry) while `elem` may be `[B, F]` (point) or `[B, F, K]` (quantile), so the mask broadcasts along trailing axes -- a quantile knot is scored exactly when its gene is. An empty selection returns a graph-connected zero rather than a NaN, so DDP find-unused stays happy.

```python
def loss(self, params, target, mask=None, feature_mask=None) -> torch.Tensor:
    if mask is not None:                      # row mask [B] as before
        params, target = params[mask], target[mask]
        if feature_mask is not None:
            feature_mask = feature_mask[mask]  # kept aligned with the rows
    if self.mode == "point":
        return masked_mean((params - target) ** 2, feature_mask)
    ...
```

- **`feature_mask=None` is bit-identical to the previous reduction**, which is the whole reason wave-6 arms already in flight were not silently re-baselined. `verify_masked_objective.json` C2 (quantile head, B=6, F=40, K=19): implicit `0.54242224`, explicit `None` `0.54242224`, all-True mask `0.54242224`. Note `masked_mean` on an all-True mask is a *weighted* sum/denominator while `None` short-circuits to `elem.mean()`; the contract asserts they agree, it is not assumed.
- **A revealed gene contributes exactly zero gradient** -- the copying failure in its most direct form. C3: `|grad|` hidden `0.4941`, revealed `0.000e+00`.
- **The point path reuses the same function, it does not reimplement it.** `MultitaskLoss._elementwise_masked` in `torchcell/models/equivariant_cell_graph_transformer.py` imports `masked_mean` *inside* the function -- a deferred import on purpose, so that models module keeps carrying no runtime dependency on `torchcell.losses` (whose `__init__` pulls in unrelated losses) while still sharing one definition.
- **Known gap, from reading the code, not measured:** `energy` mode ignores `feature_mask`. `energy_score` collapses all F features into one per-row scalar through the Euclidean norm, so there is no per-feature term left to select; a v9 arm run with `dist=energy` would score revealed genes. Likewise `DistHead.pit` takes only the row mask, so calibration diagnostics still pool every feature. No v9 arm has been run with `dist=energy`, so this has cost nothing yet.
- **Why the exactness mattered before spending GPU.** The CPU oracle (`masked_conditioning_oracle.json`, 1,482 strains x 6,169 genes, seed 0, 5 draws) bounds what ridge-from-revealed-genes alone can reach: val Pearson **0.408** (m=10), **0.676** (m=100), **0.793** (m=1000) -- at m=1000 the gene-gene signal by itself exceeds the 0.7746 replicate-based genotype ceiling (`expression_ceiling_replicate.json`). With numbers that large available for free from the labels, a leaky reduction would have produced a headline result that meant nothing; only the `k=0` column, where nothing is revealed, is comparable to prior arms.
