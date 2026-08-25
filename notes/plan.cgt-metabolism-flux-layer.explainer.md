---
id: 7jmfai8z0y1knora4lmo9qz
title: CGT Metabolism Flux Layer Explainer
desc: ''
updated: 1787671494848
created: 1787671494848
---

## 2026.07.25 -- Explainer: adding metabolism, from definitions up

Long-form companion to the objective sketched in [[plan.cgt-metabolism.2026.07.25]], which was
too compressed to follow. Same content, every intermediate step shown; the decisions it reaches
are consolidated in [[plan.cgt-metabolism-flux-layer.2026.07.26]], which supersedes this note
wherever the two disagree. **Terminology is the manuscript's**
(`paper/nature-biotech/sections/methods.tex`, Table `tab:notation` and Eqs. 15-18); where I
extend it I say so explicitly.

Answers four questions raised on reading the compressed version:

1. How is the Stage-1 constraint enforced, and why is it *not* a term in the objective?
2. What does "identifiability" mean here?
3. Do we want enzyme nodes at all, given that promiscuity argues for looser connections?
4. How do predicted $k_{\mathrm{cat}}$ / $K_M$ enter, and do they actually shrink the polytope?

---

### Part 0 -- What the manuscript already fixes

From Table `tab:notation`, unchanged:

| symbol | meaning |
| --- | --- |
| $\mathcal{U},\ \mathcal{I}$ | encoded entities; labeled instances |
| $\mathcal{Y},\ \mathcal{P}$ | phenotype space ($y$ observed, $\hat y$ predicted); perturbation space |
| $N$ | gene nodes, $N=6607$ |
| $G=(N,E)$, $A^{(k)}$, $\tilde A^{(k)}$ | cell graph; relation-$k$ adjacency; its row-normalization |
| $H=(h_{\mathrm{CLS}},h_1,\dots,h_N)$ | cell representation, $h_i\in\mathbb{R}^d$ |
| $\alpha^{(\ell,a)}$ | encoder self-attention map (rows sum to 1) |
| $F_\theta$ (ENC), $\mathcal{T}_\psi$ (PERT), $\mathcal{R}_\phi$ (DEC) | encoder, perturbation operator, decoder |
| $p=\{(e_t,\tau_t,m_t)\}_{t=1}^{M}$, $\varepsilon$ | perturbation set (entity, type, magnitude); environment |
| **$S,\ v$** | **stoichiometric matrix $\in\mathbb{R}^{m\times r}$, flux; $Sv=0$** |
| **$\rho$** | **gene-reaction association (GPR), gene $\mapsto 2^{[r]}$** |
| $\Theta,\ \mathcal{L},\ D$ | parameters $(\theta,\psi,\phi)$, loss, data law |

and the pipeline $\hat f_\theta(G,\varepsilon,p)=\mathcal{R}_\phi\big(\mathcal{T}_\psi(F_\theta(G),p),\varepsilon\big)$.

The manuscript's current objective is Eqs. (17)-(18):

$$\mathcal{L}_y=\sum_{t}w_t\sum_{b=1}^{B}m^{(b)}_t\,\ell_t\big(\hat y^{(b)}_t,y^{(b)}_t\big),\qquad \Omega_k=\sum_{i\in I_k}D_{\mathrm{KL}}\!\Big(\tilde A^{(k)}_{i,:}\,\Big\|\,\alpha^{(\ell_k,a_k)}_{i,:}\Big),\qquad \mathcal{L}=\mathcal{L}_y+\sum_{k=1}^{K}\lambda_k\Omega_k.$$

$w_t$ = task weights, $m^{(b)}_t\in\{0,1\}$ = "is task $t$ labeled for instance $b$", $\lambda_k$ =
graph-prior weights. **Everything metabolism adds is a third group of terms bolted onto Eq. (18).**

> **A commitment we would be changing.** Methods currently states: *"Only gene-gene graphs are
> regularized… Metabolism is bipartite and has no $N\times N$ adjacency, so it enters as a
> representation annotation and is never used as an attention prior."* Adding metabolite entities
> gives metabolism an adjacency over the extended entity set, so that sentence's *reason*
> dissolves -- but the *decision* is still a decision. See Part 6.

### Part 0.1 -- New symbols (extensions, flagged)

| symbol | meaning | shape / value |
| --- | --- | --- |
| $m,\ r$ | metabolites, reactions | 2,806 · 4,131 |
| $S$ | stoichiometry, $S_{ij}$ = moles of metabolite $i$ per unit flux of reaction $j$ | $2806\times4131$, 15,567 nonzeros |
| $v$ | flux vector, units mmol gDW$^{-1}$ h$^{-1}$ | $\mathbb{R}^{4131}$ |
| $v^{\ell},v^{u}$ | lower/upper flux bounds from Yeast9 | $\mathbb{R}^{4131}$ each |
| $\gamma_g\in[0,1]$ | functional availability of gene $g$ under $p$ | scalar per gene |
| $E_i$ | enzyme abundance, mmol gDW$^{-1}$ | scalar |
| $k_{\mathrm{cat},ij}$ | turnover number of enzyme $i$ on reaction $j$, h$^{-1}$ | scalar |
| $\mathrm{MW}_i$, $P_{\mathrm{avail}}$ | molecular weight (g mmol$^{-1}$); protein budget (g gDW$^{-1}$) | scalars |
| $\omega_i(v)$ | turnover of metabolite $i$ | scalar per metabolite |
| $\mu_i$ | chemical potential of metabolite $i$ ($\approx\ln c_i$ up to affine) | scalar per metabolite |
| $\Delta_j(\mu)$ | thermodynamic driving force of reaction $j$ | scalar per reaction |

Collisions avoided against `tab:notation`: $\alpha$ and $\beta$ are attention, so gene availability
is $\gamma$; $\tau_t$ is perturbation *type*, so turnover is $\omega$; $\mu_i$ is a chemical
potential, **not** growth rate (growth is the biomass flux $v_{\mathrm{bio}}$); $w_t$ stays task
weights, so constraint weights get a new letter $\nu$.

---

### Part 1 -- The two ways to enforce a constraint (this answers Q1)

**This is the part that did not come through, and the compressed note's wording made it worse: I
called the Stage-1 constraint an "equality." It is not.** There are two different constraints in
this system and they are enforced in two completely different ways.

| | constraint | type | how enforced |
| --- | --- | --- | --- |
| Stage 1 | $v^{\ell}_j\le v_j\le v^{u}_j$ | **inequality** (a box) | **by construction** -- no loss term |
| Stage 3a | $Sv=0$ | **equality** | **by penalty** -- a loss term |

#### 1.1 Enforcement by construction

Ask: *can the model even represent a violation?* If the answer is no, there is nothing to penalize.

Let $z_j\in\mathbb{R}$ be the raw (unbounded) output of the flux head for reaction $j$, and let
$\sigma$ be the logistic function

$$\sigma(z)=\frac{1}{1+e^{-z}},\qquad \text{range}\ \ \sigma:\mathbb{R}\to(0,1).$$

Now define

$$v_j\;=\;v^{\ell}_j+\big(v^{u}_j-v^{\ell}_j\big)\,\sigma(z_j).$$

Step through it. Since $0<\sigma(z_j)<1$ for **every** real $z_j$, and $v^u_j-v^\ell_j\ge0$:

$$0<\big(v^u_j-v^\ell_j\big)\sigma(z_j)<v^u_j-v^\ell_j \;\Longrightarrow\; v^\ell_j<v_j<v^u_j .$$

There is **no value of $z_j$ -- not $10^{6}$, not $-10^{6}$ -- that puts $v_j$ outside the box.** The
constraint is not "encouraged"; it is unreachable to violate. Hence no term in $\mathcal{L}$.

Numbers, with $v^\ell_j=0$, $v^u_j=1000$:

| $z_j$ | $\sigma(z_j)$ | $v_j$ |
| ---: | ---: | ---: |
| $-6$ | 0.00247 | 2.47 |
| $0$ | 0.5 | 500 |
| $+6$ | 0.99753 | 997.5 |
| $+50$ | $\approx1-2\!\times\!10^{-22}$ | 999.9999… |

**Free bonus -- directionality.** Yeast9 marks 2,463 reactions irreversible by setting
$v^\ell_j=0$. The same line then gives $v_j>0$ always, so the model can never run those reactions
backwards. That is the part of thermodynamics we get for nothing (Part 5 covers the part we do not).

*Practical note:* with $v^u=1000$ the logistic saturates and gradients vanish, so in practice use
$v_j=v^\ell_j+(v^u_j-v^\ell_j)\cdot\tfrac12(1+\tanh(z_j/s))$ with a scale $s$, or clip $v^u$ to a
physiological range. Same guarantee, better gradients.

#### 1.2 Enforcement by penalty (the contrast)

If instead we let the head emit $v_j$ directly, $v_j=\operatorname{MLP}(\cdot)$, then $v_j=5000$ is
representable, so we would need

$$C_{\mathrm{box}}=\sum_j\Big[\operatorname{relu}\big(v_j-v^u_j\big)\Big]^2+\Big[\operatorname{relu}\big(v^\ell_j-v_j\big)\Big]^2$$

with a weight $\nu_{\mathrm{box}}$ to tune, and the constraint would hold only *approximately* and
only *on average over the training distribution*. Strictly worse. That contrast is the whole point:
**spend "enforcement by construction" wherever you can, because it removes a hyperparameter and a
failure mode simultaneously.**

#### 1.3 Why $Sv=0$ cannot get the same treatment (easily)

$Sv=0$ is $m=2806$ coupled equations across all $r=4131$ coordinates. Unlike the box, you cannot
satisfy it coordinate-by-coordinate. Two options:

**(a) Penalty (recommended).** Add a term measuring how far $Sv$ is from $0$ -- Part 4.

**(b) By construction, via the null space.** Every $v$ with $Sv=0$ lies in $\ker S$. Compute a
basis $\mathcal{N}\in\mathbb{R}^{4131\times1538}$ once with $S\mathcal{N}=0$ (1,538 because
$\dim\ker S=r-\operatorname{rank}S=4131-2593$), and set $v=\mathcal{N}z$ with $z\in\mathbb{R}^{1538}$.
Then $Sv=S\mathcal{N}z=0\cdot z=0$ identically, and the head is $2.7\times$ narrower.

**But (a) and (b) are mutually exclusive with the box.** $\mathcal{N}z$ is a linear map; nothing
constrains its output to $[v^\ell,v^u]$, and projecting back into the box destroys $Sv=0$. So you
get to make **exactly one** of {box, mass balance} exact. Recommendation stands: take the box,
because it is free *and* it carries directionality for 2,463 reactions, and let $Sv=0$ be soft but
**measured and reported** (Part 7).

---

### Part 2 -- Identifiability, from zero (this answers Q2)

#### 2.1 The definition

A quantity is **identifiable** if the data determine it uniquely. It is **non-identifiable** if two
or more different values are equally consistent with everything you can observe. Non-identifiability
is not noise and not model error -- **more data of the same kind does not fix it.** It means the
question, as posed, has more than one right answer.

The manuscript already states our exact case, in Supplementary Note 4:

> *"The caveat is identifiability. The trigenic case is clean because $\tau$ is a function of
> observed fitnesses, whereas metabolic constraints $Sv=0$ involve fluxes $v$ that are largely
> unobserved, so the constraint is not a function of measured phenotypes alone."*

Unpacked: for trigenic interaction we can *compute* $\tau$ from things we measured, so there is one
answer. For flux we cannot -- we measure phenotypes, and many different $v$ produce the same
phenotypes.

#### 2.2 A worked example small enough to check by hand

Two metabolites $A,B$; four reactions:

- $j=1$: $\varnothing\to A$ (uptake)
- $j=2$: $A\to B$
- $j=3$: $B\to\varnothing$ (secretion)
- $j=4$: $B\to A$ (a back-reaction, catalyzed by a different enzyme)

Rows of $S$ are metabolites, columns reactions:

$$S=\begin{pmatrix} +1 & -1 & 0 & +1\\ 0 & +1 & -1 & -1 \end{pmatrix}\begin{matrix}\ \leftarrow A\\ \ \leftarrow B\end{matrix}$$

$Sv=0$ writes out as

$$\text{(A)}\quad v_1-v_2+v_4=0,\qquad\qquad \text{(B)}\quad v_2-v_3-v_4=0.$$

Suppose we *measure* everything an experiment can see from outside the cell: uptake $v_1=1$ and
secretion $v_3=1$. Substitute:

$$\text{(A)}\ \ 1-v_2+v_4=0\ \Longrightarrow\ v_2=1+v_4,\qquad \text{(B)}\ \ v_2-1-v_4=0\ \Longrightarrow\ v_2=1+v_4 .$$

**Both equations collapse to the same one.** So $v_4$ is completely free:

| $v_4$ | $v_2$ | $Sv=0$? | uptake | secretion | distinguishable by measurement? |
| ---: | ---: | --- | ---: | ---: | --- |
| 0 | 1 | ✓ | 1 | 1 | -- |
| 10 | 11 | ✓ | 1 | 1 | **no** |
| 1000 | 1001 | ✓ | 1 | 1 | **no** |

Every row is a perfectly valid flux vector. Nothing observable separates them. **That is
non-identifiability**, and the free direction $v_2,v_4$ moving together is precisely a **futile
cycle** -- $A\to B\to A$ spinning, consuming enzyme, achieving nothing.

#### 2.3 Why this matters at Yeast9 scale -- the measurement

The toy has one free direction. Yeast9, measured (§2026.07.25b of the companion note): running FVA
at 90 % of optimal growth over all 4,131 reactions,

- **exactly 1 reaction is pinned at a nonzero value** -- and it is `r_4046`, a *hardcoded* maintenance
  flux, not something the constraints derived;
- 1,285 more are pinned at **zero** (blocked -- no information);
- **427 reactions can take any value in $\pm1000$**, and all 427 are internal (non-boundary): the
  toy's $v_4$, 427 times over;
- 1,532 sit in a usable middle band (interval width $\le1$ against total flux $\approx100$).

**So per-reaction flux is not identifiable for this organism -- and that is true of FBA itself, not
just of a neural network.** It is why the field reports turnover *fold-changes* (Domenzain's 12
precursors) rather than absolute fluxes.

#### 2.4 What the "identifiability terms" in the objective are for

They are not generic regularization. They **select one point** out of a set the data cannot
distinguish, using a stated principle:

- **Parsimony** $\;\sum_j|v_j|$ -- "the cell does not run pointless cycles." In the toy this picks
  $v_4=0$, $v_2=1$: minimize $|v_2|+|v_4|=|1+v_4|+|v_4|$, which is smallest at $v_4=0$. This is the
  pFBA principle, and it is the main lever.
- **Near-optimality** $\;-v_{\mathrm{bio}}$ -- included because the FVA widths above were computed
  *at 90 % of optimum*. Drop the growth requirement and the polytope is strictly bigger, so
  identifiability is strictly worse. It is a prior for tractability, not a claim that cells optimize.

Both are **choices we are making explicit**, which is the honest framing: we are not discovering
the flux, we are reporting the parsimonious near-optimal flux consistent with the measured
phenotypes.

---

### Part 3 -- Enzymes: nodes, or genes plus a soft GPR? (this answers Q3)

#### 3.1 The proposal on the table

Your position: nodes are **gene-central-dogma** entities (one node per gene, standing for
DNA → RNA → protein), and getting from gene to enzyme is *"a matter of bringing GPR to bear."*
Motivation: **promiscuity** -- enzymes act on more substrates than annotation records -- so we want
**looser** connections, not a rigid enzyme partition.

**I think you are right, and promiscuity is the argument that settles it.** Here is why, and what it
costs.

#### 3.2 The key separation: hard chemistry, soft biology

Two different objects were conflated in the compressed note:

| object | what it encodes | epistemic status | treatment |
| --- | --- | --- | --- |
| $S$ (stoichiometry) | conservation of mass: 1 glucose + 1 ATP → … | **physical law**; a zero means *this species does not participate* | **HARD.** Never soften. |
| $\rho$ (GPR / catalysis) | *which gene product catalyzes which reaction* | **annotation**; a zero usually means *untested*, not *impossible* | **SOFT.** Prior, not mask. |

This is exactly the manuscript's own stated rationale for soft graph priors, transplanted:

> *"An edge with $A^{(k)}_{ij}=1$ carries experimental evidence and is trusted, but $A^{(k)}_{ij}=0$
> usually means the pair has not been tested rather than that no interaction exists. A hard operator
> would commit to those uncertain zeros, whereas the KL prior only pulls attention toward the known
> edges and lets the data overrule it."*

**Promiscuity is the metabolic instance of "zeros mean untested."** So it is not a new principle we
need to invent -- it is the principle the paper already argues for, applied to $\rho$.

#### 3.3 Consequence: a fixed enzyme node set is the wrong place to put the annotation

If enzymes are nodes whose *identity and count* come from GPR (the 1,065 AND-terms), then the
annotation is baked into the **node set** -- the most rigid part of the architecture. Representing a
promiscuous activity then means inventing nodes at inference time, which is the fixed-$N$ problem
(WS-NS1) all over again.

If instead genes are the nodes and catalysis is a **soft relation**, promiscuity is just *edge mass
the model puts where the annotation has none* -- no new entities, nothing structural. And it becomes
**reportable**: the manuscript already names this benefit for gene-gene graphs -- *"a head that keeps
strong weight on a non-edge … flags a candidate interaction that the networks have not yet
recorded."* The metabolic version of that sentence is **a predicted promiscuous activity**, which is
a publishable output, not a caveat.

#### 3.4 What we lose, and how to get it back without nodes

Three things enzyme nodes were buying. All three survive as *functions of gene nodes*:

**(a) Complexes (104 of the 1,065 units are multi-gene).** A heterodimer's abundance is limited by
its scarcest subunit. With gene availability $\gamma_g$, define per catalytic unit $u$ (an AND-term
of genes $C_u$) and per reaction $j$:

$$c_u=\operatorname{softmin}_{g\in C_u}\gamma_g=-\tfrac1\beta\log\!\sum_{g\in C_u}e^{-\beta\gamma_g},\qquad\quad c_j=\min\Big(1,\ \sum_{u\,:\,j\in\rho(u)}c_u\Big).$$

Complexes take the min (AND), isozymes add (OR). **No node is required -- this is a function
evaluated on gene embeddings.**

**(b) $k_{\mathrm{cat}}$ and $\mathrm{MW}$.** $k_{\mathrm{cat}}$ is a property of a *(catalyst,
reaction)* **pair**, not of a catalyst alone -- the same enzyme has different $k_{\mathrm{cat}}$ on
different substrates, which *is* promiscuity. So it belongs on the **edge** $(g,j)$, never on a node.
$\mathrm{MW}$ genuinely is per-gene-product (sum over subunits for a complex) -- also fine.

**(c) Protein budget.** $\sum_g \mathrm{MW}_g E_g\le P_{\mathrm{avail}}$ sums over gene products,
which is what proteomics measures anyway. Arguably *more* natural gene-centric: Zelezniak measures
726 **proteins**, i.e. gene products, not complexes.

#### 3.5 Recommendation

**Gene-centric nodes + metabolite nodes; catalysis as a soft prior; no enzyme node layer.**

- Entities: $N=6607$ genes + $m=2806$ metabolites $\approx 9.4$k tokens (down from the ~10.5k the
  compressed note proposed -- the 1,065 enzyme nodes are dropped).
- $\rho$ becomes a **soft gene→reaction incidence** $\Pi\in[0,1]^{N\times r}$, initialized/pulled
  toward the GPR annotation, free to place mass elsewhere = promiscuity.
- $k_{\mathrm{cat}}$, $K_M$ ride as **edge features on $(g,j)$**.
- Complexes/isozymes are the min/sum aggregation above.

**The one real cost:** GECKO's own formulation is enzyme-centric, so we are no longer a drop-in
re-implementation of ecYeastGEM and cannot validate line-by-line against it. Mitigation: keep a
frozen enzyme-centric mode for a one-time numerical agreement check against ecYeastGEM on wild type,
then run gene-centric.

---

### Part 4 -- The objective, term by term with the algebra shown

Notation for constraint terms: $C_\bullet$ with weight $\nu_\bullet$, extending Eq. (18).

#### 4.1 Gene availability -- how a deletion reaches the flux

$$\gamma_g=\begin{cases}0,&\text{$g$ deleted in $p$}\quad(\text{hard-set from the perturbation, not learned}),\\[4pt] \sigma\big(w^{\!\top}h_g\big),&\text{otherwise (dosage, alleles, over-expression)}.\end{cases}$$

Then $c_j$ from §3.4(a). **Read the chain: $\gamma_g\to c_u\to c_j\to$ capacity $\to v_j$.** A
deletion moves flux *only if the gene appears in $\rho$*. Yeast9 contains 1,161 of our 6,607 genes,
and in the pigment screens only 906/4,735 (betaxanthin) and 811/4,474 (β-carotene) deleted ORFs are
in it -- so **~81 % of screened deletions have no path through this chain at all** and act only
through learned gene-gene coupling in $F_\theta$. That is the coverage ceiling, stated mechanically.

#### 4.2 Mass balance -- and why the denominator

Raw residual: $[Sv]_i=\sum_j S_{ij}v_j$ for metabolite $i$. Penalizing $\sum_i[Sv]_i^2$ is wrong
because metabolites operate at wildly different scales: a residual of $0.01$ is negligible on a
metabolite carrying flux $10$ and total nonsense on one carrying $0.01$. So divide by throughput:

$$\omega_i(v)=\tfrac12\sum_{j=1}^{r}\big|S_{ij}v_j\big| \qquad\text{(half the total in+out flux -- i.e. the turnover of $i$)}$$

$$C_{\mathrm{bal}}=\frac{1}{|\mathcal{M}^{\dagger}|}\sum_{i\in\mathcal{M}^{\dagger}}\left(\frac{[Sv]_i}{\operatorname{sg}\big(\omega_i(v)\big)+\epsilon}\right)^{2}.$$

Two details that are not cosmetic:

- $\mathcal{M}^{\dagger}$ = the $\operatorname{rank}(S)=2593$ independent rows. The other
  $2806-2593=213$ are linear combinations of those (conserved moieties, dead ends); their residual is
  already determined, so penalizing them adds noise, not information.
- $\operatorname{sg}(\cdot)$ = **stop-gradient**. Without it the cheapest way to shrink the ratio is
  to shrink $\omega_i$ -- i.e. carry no flux anywhere. A normalizer must not be an optimization target.

**$\omega_i$ does double duty:** it is this denominator *and* it is the WS16 precursor-pool readout
(precursor turnover = $\omega_i$ at 13 specific $i$).

#### 4.3 Capacity and budget

Enzyme demand is a **function of $v$**, so $E$ is never predicted:

$$E_g(v)=\sum_{j\,:\,\Pi_{gj}>0}\frac{\sqrt{v_j^{2}+\delta}}{k_{\mathrm{cat},gj}},\qquad\quad P(v)=\sum_{g=1}^{N}\mathrm{MW}_g\,E_g(v).$$

($\sqrt{v^2+\delta}$ is a smooth $|v|$; the true absolute value is non-differentiable at $0$, which
is where 88 % of fluxes sit.) Both constraints are **one-sided hinges**:

$$C_{\mathrm{cap}}=\frac1N\sum_g\Big[\operatorname{relu}\big(E_g(v)-c_g\bar E_g\big)\Big]^2,\qquad C_{\mathrm{bud}}=\Big[\operatorname{relu}\big(P(v)-P_{\mathrm{avail}}\big)\Big]^2 .$$

**No reward for approaching the budget** -- with 427 free internal reactions, "use up your budget" is
maximized by futile cycles. Report $P(v)/P_{\mathrm{avail}}$ as a *diagnostic* (proteome-limited vs
substrate-limited, Domenzain's choline/putrescine distinction) instead.

#### 4.4 Thermodynamics -- one scalar per metabolite kills the loops

Read a potential off each metabolite token and form the driving force:

$$\mu_i=u^{\!\top}h^{\mathrm{m}}_i,\qquad \Delta_j(\mu)=\sum_{i=1}^{m}S_{ij}\,\mu_i,\qquad C_{\mathrm{th}}=\frac1r\sum_{j=1}^{r}\operatorname{relu}\big(v_j\Delta_j(\mu)+\epsilon\big).$$

This enforces $v_j\Delta_j\le0$ ("flux runs downhill"). **Worked on the toy of §2.2**, to show it
kills the cycle exactly. Let $\Delta_2=\mu_B-\mu_A$ (reaction $A\to B$) and
$\Delta_4=\mu_A-\mu_B=-\Delta_2$ (reaction $B\to A$). Require both conditions:

$$v_2\Delta_2\le0\quad\text{and}\quad v_4\Delta_4\le0\ \Longleftrightarrow\ -v_4\Delta_2\le0 .$$

Suppose the pathway runs, $v_2>0$. Then the first gives $\Delta_2<0$. Substituting into the second:
$-v_4\Delta_2\le0$ with $\Delta_2<0$ requires $v_4\le0$. But reaction 4 is irreversible, so
$v_4\ge0$. Therefore

$$\boxed{v_4=0}$$ -- the futile cycle is eliminated, with no integer variables. The general reason: $\Delta$ is a
*potential difference*, so it sums to zero around any cycle, and no cycle can run downhill
everywhere. This is what makes loopless-FBA a mixed-integer program and makes this version cheap.

**And it gives the metabolite embedding a physical coordinate**, $\mu_i\approx\ln c_i$ up to affine -- which Part 5 then uses.

#### 4.5 Data terms -- including the one we have most of

$$\mathcal{L}_y=\sum_t w_t\sum_{b=1}^{B}m^{(b)}_t\,\ell_t\big(\hat y^{(b)}_t,y^{(b)}_t\big)\qquad\text{(Eq. 17, unchanged)}$$

Three heads read the flux layer rather than the token stack:

- **Fitness $\Rightarrow$ biomass flux.** Fitness *is* relative growth rate, so
  $\hat y_{\mathrm{fit}}=v_{\mathrm{bio}}(p)/v_{\mathrm{bio}}(\varnothing)$ supervises one coordinate
  of $v$ directly. We hold $\sim\!10^{7}$ fitness records -- **orders of magnitude more
  flux-relevant data than any $^{13}$C-MFA study.** Cache the wild-type pass once per step.
- **Precursor pools (WS16)** = $\omega_i(v)$ at 13 metabolites, compared in **log fold-change vs
  wild type**, matching Domenzain's reported quantity.
- **Proteome $\Rightarrow$ $E_g$** on Zelezniak/Messner strains: $\ell(\log E_g(v),\log E_g^{\mathrm{obs}})$
  over 726 measured proteins. ecFBA structurally cannot do this check; we can.

#### 4.6 The whole thing

$$\boxed{\ \mathcal{L}\;=\;\underbrace{\mathcal{L}_y}_{\text{Eq.\ 17}}\;+\;\underbrace{\sum_{k=1}^{K}\lambda_k\Omega_k}_{\text{Eq.\ 18 graph priors}}\;+\;\underbrace{\nu_{\mathrm{bal}}C_{\mathrm{bal}}+\nu_{\mathrm{cap}}C_{\mathrm{cap}}+\nu_{\mathrm{bud}}C_{\mathrm{bud}}+\nu_{\mathrm{th}}C_{\mathrm{th}}}_{\text{physics (new)}}\;+\;\underbrace{\nu_{\mathrm{par}}\tfrac1r\!\sum_j\!\sqrt{v_j^2+\delta}\;-\;\nu_{\mathrm{opt}}v_{\mathrm{bio}}}_{\text{identifiability (new, Part 2.4)}}\ }$$

**There is no $C_{\mathrm{box}}$ term.** That is Part 1: the box is enforced by the
parameterization, so it cannot appear in the objective.

---

### Part 5 -- $k_{\mathrm{cat}}$ and $K_M$ (this answers Q4)

#### 5.1 They are not the same kind of parameter

| | what it is | where it enters | does predicting it shrink the polytope? |
| --- | --- | --- | --- |
| $k_{\mathrm{cat}}$ | max turnover, h$^{-1}$ | capacity $\lvert v_j\rvert\le k_{\mathrm{cat},gj}E_g$ | **Yes, directly** |
| $K_M$ | half-saturation concentration, mM | *kinetics only* -- saturation $\eta^{\mathrm{sat}}$ | **No, not by itself** |

**$k_{\mathrm{cat}}$ tightens.** Adding capacity constraints is exactly what makes
$\mathcal{F}_{\mathrm{ec}}\subseteq\mathcal{F}_{\mathrm{FBA}}$ -- it removes flux distributions that
are stoichiometrically fine but need more protein than exists. Predicting $k_{\mathrm{cat}}$ before
training and **freezing it** is therefore a genuine, direct reduction of the polytope, and it is
precisely the ecFBA move.

**$K_M$ does not, by itself.** An enzyme-constrained model uses only the max-capacity part; $K_M$
belongs to the rate law $v=k_{\mathrm{cat}}E\frac{c}{K_M+c}$, which needs a **concentration** $c$.
Concentrations are not observed, so introducing $K_M$ introduces both a parameter *and* a free
variable -- it can *enlarge* the model's freedom, not shrink it. **This is the trap: $K_M$ looks like
extra constraint and is actually extra slack.**

#### 5.2 …unless concentrations are anchored -- and ours partly are

We already introduced $\mu_i\approx\ln c_i$ for thermodynamics (§4.4), so $c_i=e^{\mu_i}$ is
available. Then

$$\eta^{\mathrm{sat}}_j=\prod_{i\in\mathrm{sub}(j)}\frac{c_i}{K_{M,ij}+c_i},\qquad \lvert v_j\rvert\le k_{\mathrm{cat},gj}\,E_g\,\eta^{\mathrm{sat}}_j\,\eta^{\mathrm{thermo}}_j,\qquad \eta^{\mathrm{thermo}}_j=1-e^{\Delta_j/RT}.$$

This only *constrains* if $\mu$ is anchored to something measured. **It partly is:** Mülleder gives
**absolute intracellular concentrations in mM for 19 amino acids** across 4,678 strains, and
Zelezniak gives relative pool sizes for ~50 metabolites. So $K_M$ becomes usable **exactly on the
metabolites where we have metabolome data** -- which is also the natural bridge to the kinetic rung.

#### 5.3 Recommended handling

1. **Published before predicted.** Mirror ecYeastGEM's $k_{\mathrm{cat}}$/$\mathrm{MW}$/$P_{\mathrm{avail}}$
   with sha256 provenance. Nothing here runs without them, and none of it is currently in the repo
   (`grep ecYeast|GECKO|kcat` over `torchcell/` is empty).
2. **Predict only the gaps** (KcatNet / Wu 2026, WS-NS2) -- and note that **promiscuous edges have no
   published $k_{\mathrm{cat}}$ by definition**, so if we want the model to *use* a promiscuous
   activity, prediction is the only possible source. Promiscuity and $k_{\mathrm{cat}}$ prediction
   are the same workstream.
3. **Carry uncertainty.** A predicted $k_{\mathrm{cat}}$ that is too low forbids feasible flux; too
   high is vacuous. Use a slack proportional to predictor uncertainty:
   $\lvert v_j\rvert\le(1+\kappa\hat s_{gj})k_{\mathrm{cat},gj}E_g$, and ablate $\kappa$.
4. **$K_M$: predict now, activate later** -- wire the interface, enable $\eta^{\mathrm{sat}}$ only on
   metabolites with measured concentrations, and treat the rest as $\eta^{\mathrm{sat}}=1$.
5. **Prediction input is a *(protein sequence, substrate)* pair**, which is another argument for
   §3.5: the natural home for these parameters is the $(g,j)$ edge, not an enzyme node.

---

### Part 6 -- The one manuscript commitment this would change

Methods currently says, verbatim: *"Only gene-gene graphs are regularized… Metabolism is bipartite
and has no $N\times N$ adjacency, so it enters as a representation annotation and is never used as
an attention prior."* Two coherent options:

- **(A) Keep the commitment.** Metabolism enters only through the new $C_\bullet$ physics terms and
  the metabolite tokens. $\Omega_k$ stays gene-gene. **No Methods edit.** Recommended for now -- it is
  also the roadmap's Design Decision 4.
- **(B) Extend $\Omega_k$ to metabolism.** With metabolite nodes there *is* an adjacency over the
  extended entity set, so the stated reason no longer holds and we could align a head to
  $\operatorname{rownorm}(|S|^{\!\top}|S|>0)$. Note it must be $|S|$, not $S$ -- attention rows are
  non-negative and sum to 1, while $S$ is signed. **Requires rewriting that Methods sentence**, and
  the section is author-approved, so it needs an explicit go-ahead.

The compressed note proposed (B) without flagging the conflict. That was my error; (A) is the
default until you decide otherwise.

---

### Part 7 -- Reporting: feasibility is a measurement, not a hope

Because the constraints are soft, "is this a valid flux vector" must be *answered per sample*, not
assumed:

$$\mathrm{feas}_{\mathrm{bal}}=\operatorname{median}_i\frac{\big|[Sv]_i\big|}{\omega_i(v)},\qquad \mathrm{feas}_{\mathrm{bud}}=\frac{P(v)}{P_{\mathrm{avail}}},\qquad \mathrm{feas}_{\mathrm{th}}=\frac1r\sum_j\mathbb{1}\big[v_j\Delta_j>0\big].$$

And every flux-derived biological claim gets restricted to the reactions the **FVA mask** licenses
(the ~1,532 with interval width $\le1$), with the 427 unconstrained loop carriers excluded by name.

### Gradient hazards (each is a real failure mode)

1. **Term scales differ by orders of magnitude** -- $C_{\mathrm{bal}}$ is dimensionless,
   $C_{\mathrm{bud}}$ is (g gDW$^{-1}$)$^2$, $\mathcal{L}_y$ is phenotype units. This is the failure
   the 019 joint runs already hit. **Normalize each term to dimensionless before weighting.**
2. **$|v|$ at $v=0$** -- smooth it; 88 % of reactions sit exactly there.
3. **$\operatorname{softmin}$ temperature $\beta$** -- too soft and a complex behaves like a mean (a
   lethal deletion stops being lethal); too hard and gradients reach only one subunit. Anneal.
4. **$\omega_i$ in a denominator** -- stop-gradient, else "carry no flux" wins.
5. **Wild-type forward pass** for $v_{\mathrm{bio}}(\varnothing)$ -- per step, not per sample.

### Open decisions

1. **Enzyme nodes: drop them?** Recommend yes (Part 3) -- gene-centric + soft $\rho$, which is what
   makes promiscuity representable at all. Cost: no line-by-line ecYeastGEM parity.
2. **Exactness budget: box or null-space?** Recommend box (Part 1.3).
3. **Methods commitment (A) or (B)?** Recommend (A) for now (Part 6).
4. **$K_M$ scope** -- wire the interface, activate only where concentrations are measured (Part 5.3).
   → **REVISED in Part 12 below: Wu 2026 shows $K_M$ is the parameter that carries promiscuity,
   which promotes it from "defer" to "load-bearing."**

---

## 2026.07.25b -- Answers to the read-through questions (all verified)

Every number below is measured, not assumed. Sources: `YeastGEM()` v9.0.2 loaded from
`data/torchcell/yeast-GEM/yeast-GEM-9.0.2/`, cobra FVA runs saved to the session scratchpad, and
the OCR'd mirror at `$DATA_ROOT/torchcell-library/`.

### Q1 -- Are $v^{\ell}, v^{u}$ arbitrary? Mostly yes, and that is load-bearing

I searched the SBML, the annotations, and the yeast-GEM curation scripts. The result is stark.

**The entire model has FIVE distinct bound values.** `yeast-GEM.xml` `<listOfParameters>`:

| parameter | value | used by |
| --- | ---: | --- |
| `FB1N1000` | $-1000$ | 1,667 lower bounds |
| `FB2N1` | $-1$ | **1** lower bound (`r_1714`, D-glucose exchange) |
| `FB3N0` | $0$ | 2,462 lower, 7 upper |
| `FB4N1` | $0.7$ | **1** lower + **1** upper (`r_4046`, NGAM -- a fixed equality) |
| `FB5N1000` | $+1000$ | 4,123 upper bounds |

No $\pm\infty$ anywhere. **Exactly 2 of 4,131 reactions carry a bound that is not
$\{0,\pm1000\}$** -- i.e. **4,129 reactions have bounds that encode only on/off and direction, and
no capacity whatsoever.**

- `r_1714` glucose exchange, $v^\ell=-1$ -- an arbitrary **normalization**, not a measured uptake
  rate. It sets growth linearly: $\mu=0.0858$ at $-1$, $0.8877$ at $-10$, $1.7786$ at $-20$.
- `r_4046` NGAM $=0.7$ -- the one genuinely fitted physiological constant, from chemostat data
  (`code/otherChanges/fitGAM.m` reading `data/physiology/chemostatData_VanHoek1998.tsv`).

**Provenance is a curation script, not an annotation.** No bound-related key exists in
`r.annotation` (which carries `sbo`, `ec-code`, `kegg.pathway`, `pubmed`, `metanetx`, `bigg`, …)
or `r.notes`. The bounds are byte-for-byte the output of
`data/torchcell/yeast-GEM/yeast-GEM-9.0.2/code/modelCuration/minimal_Y6.m`, whose header cites
Sánchez et al., *PLoS Comput Biol* (doi 10.1371/journal.pcbi.1004530) for the minimal medium.

**Verdict: directions are principled; magnitudes are conventional placeholders.** Reversibility is
curated and EC/KEGG-backed (2,455 irreversible, 1,668 reversible, 7 hard-blocked). $\pm1000$ is
COBRA's stand-in for infinity.

**Why this matters more than it looks.** The box is doing exactly two jobs -- **direction** and
**medium** -- and *no* capacity job. So **all capacity information must come from
$k_{\mathrm{cat}}\cdot E$.** The enzyme layer is not a refinement on top of a partly-constrained
model; it is the *only* source of magnitude constraints. That is the strongest argument yet for
prioritizing the ecYeastGEM parameter mirror.

**And the medium is 16 numbers.** All 16 exchanges with a nonzero lower bound *are* the medium
definition: ammonium, H+, iron(2+), oxygen, phosphate, potassium, sodium, sulphate, water,
chloride, Cu²⁺, Mn²⁺, Zn²⁺, Mg²⁺, Ca²⁺ (all at $-1000$, i.e. unlimited) plus glucose at $-1$.

### Q1b -- Symbol table with provenance and role (as requested)

| symbol | shape | **where the value comes from** | **role in the model** |
| --- | --- | --- | --- |
| $S$ | $2806\times4131$, 15,567 nz | Yeast9 SBML -- **chemistry, hard** | operator in $C_{\mathrm{bal}}$; support $\lvert S\rvert$ optionally an attention prior |
| $v^{\ell},v^{u}$ | $\mathbb{R}^{4131}$ each | `minimal_Y6.m`; **4,129/4,131 are placeholders**, 2 informative | defines the box → direction + medium, **not** capacity |
| medium | 16 exchange $v^\ell$ | `minimal_Y6.m`; our own media → **not yet ported** (Q10) | how $\varepsilon$ enters the box |
| $\rho$ / $\Pi$ | $N\times r$ | Yeast9 GPR -- **annotation, soft** | gene→reaction catalysis; promiscuity = mass beyond $\rho$ |
| $\gamma_g$ | $\mathbb{R}^{N}$ | **0 for deletions (from $p$)**, else learned from $h_g$ | how a KO reaches flux |
| $k_{\mathrm{cat},gj}$ | per $(g,j)$ | **NOT in repo.** ecYeastGEM (missing) → Open Enzyme DB (captured) → predicted (DLKcat/TurNuP/KcatNet) | capacity: $\lvert v_j\rvert\le k_{\mathrm{cat}}E_g$ -- the *only* magnitude constraint |
| $K_{M,gj}$ | per $(g,j)$ | predicted (Boost_KM / UniKP / EITLEM per Wu 2026) | saturation $\eta^{\mathrm{sat}}$ -- **the parameter that distinguishes promiscuous from native (Q12)** |
| $\mathrm{MW}_g$ | $\mathbb{R}^{N}$ | protein sequence (computable now) | protein budget |
| $P_{\mathrm{avail}}$ | scalar | ecYeastGEM / literature -- **missing** | budget hinge |
| $\Delta_fG'^\circ_i$ | $\mathbb{R}^{2806}$ | **`…/yeast-GEM-9.0.2/data/databases/model_metDeltaG.csv` -- WE HAVE IT** (Q5) | standard part of $\mu_i$ |
| $\ln c_i$ | $\mathbb{R}^{2806}$ | learned; anchored by Mülleder mM (19 AAs) | concentration part of $\mu_i$; feeds $\eta^{\mathrm{sat}}$ |
| $\mu_i$ | $\mathbb{R}^{2806}$ | $=\Delta_fG'^\circ_i+RT\ln c_i$ | thermodynamic potential |
| $\Delta_j$ | $\mathbb{R}^{4131}$ | $=\sum_i S_{ij}\mu_i$ -- **derived, never supplied** | driving force; loop elimination |
| $E_g$ | $\mathbb{R}^{N}$ | $=\sum_j\lvert v_j\rvert/k_{\mathrm{cat},gj}$ -- **derived from $v$** | budget; validated vs measured proteome |
| $\omega_i$ | $\mathbb{R}^{2806}$ | $=\frac12\sum_j\lvert S_{ij}v_j\rvert$ -- derived | balance denominator **and** WS16 precursor readout |

### Q2 -- "$v^{\ell},v^{u}$ can change depending on sequence -- how does that connect?"

You've spotted the right generalization, and it makes the design *simpler*, not harder. **The box
does not have to be static.** Recompute it every forward pass as a function of environment,
perturbation, and sequence:

$$\bar v^{u}_j(\varepsilon,p,\mathrm{seq})=\min\Big(\underbrace{v^{u}_j(\varepsilon)}_{\text{medium + direction}},\ \underbrace{c_j(p)}_{\text{GPR availability}}\cdot\underbrace{\textstyle\sum_g \Pi_{gj}\,k_{\mathrm{cat},gj}(\mathrm{seq}_g)\,\bar E_g}_{\text{catalytic capacity}}\Big)$$

and symmetrically for $\bar v^{\ell}_j$ on reversible reactions. The sigmoid then maps into
$[\bar v^\ell,\bar v^u]$ exactly as before, so **everything stays exact-by-construction**. Three
payoffs:

1. **Gene deletion becomes exact.** If $g$ is the sole catalyst and $\gamma_g=0$, then $c_j=0$, so
   $\bar v^u_j=\bar v^\ell_j=0$ -- the reaction is off by construction, reproducing FBA's
   single-gene-deletion semantics with no penalty term.
2. **Sequence enters through $k_{\mathrm{cat}}(\mathrm{seq}_g)$** -- which is exactly your
   species-transferability lever (Q11): a predictor that reads protein sequence carries to any
   organism.
3. **The capacity constraint leaves the objective.** $C_{\mathrm{cap}}$ disappears; one fewer
   weight to tune.

**The general principle this exposes:** *decoupled constraints go in the parameterization; coupled
constraints go in the objective.* Per-reaction caps (direction, medium, capacity) are decoupled →
box. $Sv=0$ and the shared budget $\sum_g\mathrm{MW}_gE_g\le P_{\mathrm{avail}}$ couple many
coordinates → penalties. That is the whole allocation, and it is decidable by inspection.

*Gradient note:* when $c_j\to0$ the box collapses to a point and $\partial v_j/\partial z_j\to0$.
That is correct (a dead reaction has no flux to tune) but means no learning signal on that
reaction for that strain -- expected, not a bug.

### Q3 -- What FVA is, and what the growth requirement actually buys (measured)

**Flux Variability Analysis** answers, per reaction: *given the constraints, how much can this flux
vary without breaking anything?* For reaction $j$ it solves two LPs -- minimize $v_j$, then maximize
$v_j$ -- subject to $Sv=0$, the bounds, and optionally a floor on growth
($v_{\mathrm{bio}}\ge f\cdot v^{\max}_{\mathrm{bio}}$). The **width** $=\max-\min$ is the answer:

- width $\approx0$ → the flux is **pinned**; you may interpret it.
- width huge → the flux is **free**; any value is equally consistent, so interpreting it is meaningless.

That is identifiability, computed per reaction, before any model exists.

**The growth floor $f$ is why FVA needs a choice, and here is what it buys** (both runs over all
4,131 reactions, saved as `yeast9_fva_frac0.csv` / `yeast9_fva_frac90.csv`):

| | $f=0$ (no growth requirement) | $f=0.9$ (growth ≥ 90 % of max) |
| --- | ---: | ---: |
| median width | **1.077** | **0.1176** |
| reactions with width $\le1$ | 2,049 | **2,818** |
| reactions with width $\le10$ | 2,863 | **3,671** |
| effectively unconstrained (width $>10^3$) | 428 | **427** |

**Requiring growth narrows 2,316 reactions -- 56.1 % of the network -- and improves the median width
9-fold.** That is the entire content of the sentence you got stuck on: identifiability of the
*other* fluxes is conditional on the cell being told to grow.

**And notice what it does not fix: the loops.** 428 → 427. The unconstrained reactions are
indifferent to growth, because a futile cycle neither helps nor hurts biomass. So the two levers
do different jobs: **growth narrows the bulk; parsimony and the thermodynamic potential kill the
loops.** Neither substitutes for the other.

**Resolution of "do we keep the growth objective?" -- your instinct is right, and it goes further.**
You said FBA growth-optimization is blind to genes outside metabolism, and that it's fine if
$v_{\mathrm{bio}}$ can be affected by genes elsewhere. In our model it can: $v_{\mathrm{bio}}$ is a
predicted coordinate read from $H_{\mathrm{pert}}$, which every one of the 6,607 gene tokens
influences through $F_\theta$. So the blindness is FBA's, not ours.

Which means **we should drop $\mathcal{L}_{\mathrm{opt}}=-v_{\mathrm{bio}}$ entirely and let
fitness do the job.** The near-optimality prior is a *substitute for missing data*; we are not
missing it -- we hold ~$10^7$ fitness measurements covering essentially every deletion, metabolic or
not. Pinning $v_{\mathrm{bio}}(p)=y_{\mathrm{fit}}\cdot v_{\mathrm{bio}}(\varnothing)$ is a
**stronger** conditioning than $v_{\mathrm{bio}}\ge0.9\,v^{\max}$ -- a point rather than a
half-space, at the *measured* value, per strain -- so identifiability is at least as good as the
$f=0.9$ column above. Keep $-v_{\mathrm{bio}}$ only as a fallback for unlabeled strains.

*Modeling caveat to record:* SGA fitness is colony size on selection medium, a proxy for growth
rate, not a chemostat $\mu$. The proportionality constant is an assumption.

### Q4 -- Are the rank numbers real? Yes, measured

$\operatorname{rank}(S)=2593$ and $\dim\ker S=4131-2593=1538$ are computed, not illustrative -- `np.linalg.matrix_rank` on the dense $2806\times4131$ stoichiometric matrix. One honesty caveat:
that is a *numerical* rank at NumPy's default SVD tolerance. It should be re-confirmed with an
explicit tolerance sweep before the number goes in a paper, since near-dependent rows could flip
it by a few.

### Q5 -- Chemical potentials: we already have them, they are just not loaded

Direct answer to "can we look up chemical potential, or only per-reaction Gibbs energy?" -- **both,
and they ship inside yeast-GEM 9.0.2**:

| file | rows | usable values | coverage |
| --- | ---: | ---: | ---: |
| `…/yeast-GEM-9.0.2/data/databases/model_metDeltaG.csv` | 2,806 | **2,389** real $\Delta_fG'^\circ$ (kJ/mol, range $-1769.75$ … $1882.32$) | **85.1 % of metabolites** |
| `…/data/databases/model_rxnDeltaG.csv` | 4,131 | 3,210 real $\Delta_rG'^\circ$ | 77.7 % of reactions |

The rest are sentinel `10000000` (= unknown) or `NaN`. **Why we never noticed:** `YeastGEM.model`
reads the *SBML*, and SBML is the one export that drops the ΔG fields -- `grep -c deltaG
yeast-GEM.xml` returns 0. The `.yml` and `.mat` exports carry them (6,766 `- deltaG:` entries in
the YAML). Picking them up is a two-column CSV read; the ids already match `m.metabolites`.
`equilibrator_api` is **not** installed and is not needed.

**So $\mu_i$ decomposes into a known part and a learned part:**

$$\mu_i=\underbrace{\Delta_fG'^\circ_i}_{\text{data, 2{,}389 of 2{,}806}}+\;RT\underbrace{\ln c_i}_{\text{learned}},\qquad \Delta_j=\sum_i S_{ij}\mu_i=\Delta_rG'^\circ_j+RT\sum_i S_{ij}\ln c_i .$$

### Q6 -- "Where does $u$ come from?"

$u\in\mathbb{R}^{d}$ was just a learned linear readout -- a probe on the metabolite embedding, no
different from any head's weight matrix. That was an unsatisfying answer, and Q5 makes it a better
one: **with the decomposition above, the learned part only has to produce $\ln c_i$**, i.e.

$$\mu_i=\Delta_fG'^\circ_i+RT\big(u^{\!\top}h^{\mathrm{m}}_i\big),$$

where the standard formation energy is *data* and $u^\top h^m_i$ is a **log-concentration**. Two
benefits: it is far better conditioned (log-concentrations span a few units; formation energies
span thousands of kJ/mol), and it is **directly checkable** against Mülleder's absolute mM for 19
amino acids. For the 417 metabolites without a $\Delta_fG'^\circ$, either learn the whole $\mu_i$
or mask them out of $C_{\mathrm{th}}$ -- recommend masking, and report the coverage.

### Q7 -- "Is the output a flux sample?" Yes, and this is the strongest framing available

If $v$ satisfies mass balance, the box, and capacity, then $v$ is a point in the feasible polytope
$\mathcal{P}$ -- so it is *a feasible flux distribution* in the FBA sense. But "sample" implies a
distribution, and that depends on a design choice:

- **Deterministic head** → $v$ is one point *selected* by the objective's implicit preference
  (parsimony + data fit). A selection, not a sample.
- **Stochastic head** → draw $z\sim q_\phi(z\mid H_{\mathrm{pert}})$, then $v=\mathrm{box}(z)$ is a
  genuine random variable supported on $\mathcal{P}$. **That is an amortized flux sampler**, and it
  delivers exactly what you wanted: a per-reaction flux distribution on every forward pass.

**Why this is the right framing.** Non-identifiability stops being a weakness to apologize for and
becomes *the object we report*. Classical flux sampling (ACHR/OptGP in cobra) samples $\mathcal{P}$
per condition by MCMC, ignores data, and costs minutes per genotype. Ours is data-conditioned and
amortized to one forward pass. And it yields a clean evaluation:

$$\text{information gained from data}\;=\;\underbrace{\text{width}_{\mathrm{FVA}}(j)}_{\text{constraints alone}}\;-\;\underbrace{\text{width}_{\text{model posterior}}(j)}_{\text{constraints}+\text{data}}$$

per reaction. If our interval is narrower than FVA's, the phenotype data added information; if
equal, it did not. That is a publishable figure and an honest one. It also reuses the decoder
note's distributional machinery (CRPS / quantile heads) -- same code, different head.

*Caveat:* classical sampling targets a **uniform** distribution over $\mathcal{P}$; ours targets
whatever the data and priors induce. Different object, not a drop-in replacement -- say so.

### Q8 -- Hard **and** soft in parallel, and how to stop mass leaving known reactions

You proposed running hard and soft attention as parallel paths, and separately worried that a fully
soft $\Pi$ lets mass drift off annotated reactions entirely. **Those are the same idea and they
solve each other.** Make the soft path strictly additive:

$$\Pi=\underbrace{\Pi^{\mathrm{GPR}}}_{\text{hard, fixed, a FLOOR}}+\underbrace{\Delta\Pi}_{\text{soft, learned, }\Delta\Pi\ge0},\qquad \text{penalty }\ \nu_{\mathrm{prom}}\lVert\Delta\Pi\rVert_1 .$$

Because $\Delta\Pi\ge0$, capacity on an annotated reaction can never be *removed* -- the annotation
is a floor, not a suggestion. Promiscuity is exactly $\Delta\Pi$: readable, rankable, reportable,
and its overall magnitude is a single tunable prior $\nu_{\mathrm{prom}}$. Set
$\nu_{\mathrm{prom}}$ large at the start (your "be conservative initially") and anneal it down.

**Is the parallel path wasteful?** No -- it is the pattern we already run: multiple attention heads
per graph type. One head bound to the annotation and one free head is the same cost structure, and
it has the advantage that promiscuity is *localized to an identifiable head* instead of smeared
across the model.

### Q9 -- GECKO comparison, not parity (expanding, per your note)

Agreed with your framing: *we are not trying to be ecYeast, but we must be able to compare.* Parity
would mean reproducing GECKO's enzyme-centric formulation, which conflicts with the gene-centric
choice. Comparison needs only a shared interface. Concretely, four comparisons of increasing
interest:

1. **Wild-type flux agreement.** Our $v$ on WT vs ecYeastGEM pFBA, correlated over the
   FVA-licensed reactions only (width $\le1$; $n=2{,}818$ at $f=0.9$).
2. **Deletion growth.** Our predicted fitness vs ecFBA growth ratio, on the 1,161 Yeast9 genes.
3. **Enzyme allocation.** Our $E_g$ vs measured proteome -- **ecFBA structurally cannot do this**,
   because its $E_i$ are optimizer choices with nothing to check them against.
4. **The comparison that actually matters.** On the ~81 % of screened deletions outside Yeast9,
   ecFBA predicts *no effect at all*. We predict something. That is not a tie-break -- it is a
   category difference, and it is where the architecture earns its keep.

**Reading list (checked against the mirror + Zotero).** Captured and OCR'd locally:
`domenzainComputationalBiologyPredicts2025` (the PNAS 103-chemicals application -- the one you
already have), `wuSystematicallyExploringYeast2026`, `yuanOpenEnzymeDatabase2026` (**the Open
Enzyme Database you mentioned -- it is already in the mirror**), `longEnzymeEngineeringDatabase2026`,
`bordbarConstraintbasedModelsPredict2014`.

**Start with these four, none of which we hold** -- add to the Zotero `paper` collection then
`lit_sync`:

| priority | paper | why |
| --- | --- | --- |
| 1 | **Sánchez et al. 2017, *Mol Syst Biol*** -- original GECKO | the formulation itself; also the source of Yeast9's minimal medium (Q1) |
| 2 | **Domenzain et al. 2022, *Nat Commun*** -- GECKO 2.0 / ecModels catalog | where ecYeastGEM's $k_{\mathrm{cat}}$/$P_{\mathrm{avail}}$ actually come from |
| 3 | **Elsemman et al. 2022, *Nat Commun*** -- compartment-specific proteome constraints | **already in Zotero, just has no collection** -- one-line fix, cheapest win |
| 4 | **Chen, Li & Nielsen 2022, *FEMS Yeast Res*** -- the review | in `notes/assets/bib/bib.bib` only |

Also missing and worth having for Q11: DLKcat, TurNuP (in bib only), KcatNet, UniKP, EITLEM.

### Q10 -- Media: yes, we ported them, and the good version is not here

You remembered correctly, and the state is worse than "we have it":

- **`experiments/007-kuzmin-tm/scripts/setup_media_conditions.py`** does implement COBRA media -- `reset_media()`, `setup_minimal_media()` (16 exchanges, glucose $-10$), `setup_ynb_media()`
  (+9 vitamins at 5 % of glucose), `setup_ypd_media()` (+20 amino acids at 5 %). Sourced from
  Suthers 2020 iIsor850.
- **But**: a single WIP commit (`c64744df`), a hardcoded absolute path to a model *outside the
  repo* (`/home/michaelvolk/Documents/yeast-GEM/model/yeast-GEM.xml`), no config, not in the
  `torchcell/` package. Nothing in `torchcell/` sets a single exchange bound -- `torchcell/metabolism/yeast_GEM.py` never touches `lower_bound`/`upper_bound` at all.
- **The corrected formulations are in the external `iBioFoundry-AI` repo**
  (`experiments/case-study/s-cerevsisiae-beta-carotene-knockout/`) and were **never ported here**.
  Our own note flags the 007 versions as wrong: *"the iIsor850 media formulations are slightly off…
  We have since fixed the formulation in the beta-carotene case study"*
  ([[experiments.007-kuzmin-tm.FBA-interaction-experiments]]).
- The hook already exists: `torchcell/datamodels/media.py:20` says the exchange-bound mapping
  *"lives in a future cobra/AMICI adapter, NOT in these wet-lab records."*

**Action:** port the corrected iBioFoundry media into `torchcell/metabolism/media.py` as the
`Media` → exchange-bound adapter. This *is* WS6 ($\varepsilon$-conditioning): our datasets span
SC, SC-URA, SM, and YPD, and each is a different $v^\ell$ on the 16-plus exchange reactions. With
Q2's dynamic box, media conditioning becomes structural rather than a learned embedding -- strictly
better, because it is the actual mechanism.

### Q11 -- New model file, and species transferability

Agreed on a new file. Recommended split, because it is what makes transfer possible:

- **`torchcell/models/cell_graph_transformer.py`** -- the canonical CGT fork already planned in
  [[experiments.019-simb-multimodal.experimental-plans]].
- **`torchcell/metabolism/constraints.py`** *(new)* -- the constraint layer as **pure functions of a
  GEM**: box construction, $C_{\mathrm{bal}}$, capacity, budget, $\mu$/$\Delta$. No yeast constants
  in the model file.

**What is and is not species-specific** -- this is the transferability argument, and it is better
than expected:

| ingredient | transfers? | why |
| --- | --- | --- |
| $\Delta_fG'^\circ$ | **fully** | chemistry is species-independent |
| $k_{\mathrm{cat}}$, $K_M$ | **fully** | predicted from protein sequence -- your point exactly |
| $\mathrm{MW}$ | **fully** | computed from sequence |
| $S$, $\rho$ | needs a GEM | draft GEMs are automatable (CarveMe/RAVEN) |
| $P_{\mathrm{avail}}$ | one scalar | literature or proteome-derived |

So the only genuinely organism-specific input is **a GEM plus one scalar**. Everything with a
learned component is sequence-driven, which is precisely why sequence-based $k_{\mathrm{cat}}$
prediction is worth the investment.

### Q12 -- $K_M$ is promoted: Wu 2026 says promiscuity is a $K_M$ effect, not a $k_{\mathrm{cat}}$ effect

**This reverses my Part 5 advice, and it is the most consequential finding of this round.**
`wuSystematicallyExploringYeast2026` (Wu, …, Kerkhoven, Chen, Nielsen, Li -- *Nature Catalysis*,
doi 10.1038/s41929-026-01523-w) measured exactly the "squishiness" you wanted quantified:

**How much of metabolism is unannotated:**

- **93 % of the yeast reaction network has never been systematically modelled** -- 4,131 known
  (Yeast9) vs **55,734 predicted underground reactions**.
- Yeast9 covers **~7 % of the known yeast metabolome** (14,882 of 16,042 YMDB metabolites are
  absent); Yeast-MetaTwin raises coverage 7 % → 92 %.
- **~53 % of modelled metabolic enzymes are generalists** -- of 1,422 kinetically analysable
  enzymes, 611 known Yeast9 enzymes participate in underground reactions vs 550 specialists.
- Per-EC-class exploration: isomerases (EC 5) **8.5 %** explored; ligases (EC 6), the best, **48.9 %**.
- **52 % of underground reactions re-link metabolites Yeast9 already contains** -- the sharpest
  possible statement of "an absent edge means untested, not impossible."
- Yeast9 **overstates fragility 2×**: it predicts loss of amino-acid synthesis at 20 % pathway-gene
  knockout; with underground reactions it is 40 %.
- Adding them improves the GEM itself: growth Pearson **0.50 → 0.61**, essentiality 88 % → 90 %,
  synthetic lethality 77 % → 84 %.

**The kinetic finding that changes our design:**

> Median $K_M$ underground vs known is **~2× higher** across three independent predictors (0.25 vs
> 0.11 mM Boost_KM; 0.21 vs 0.11 UniKP; 0.25 vs 0.07 EITLEM), while **$k_{\mathrm{cat}}$
> distributions are indistinguishable** (DLKcat 5.52 vs 5.28 s⁻¹; TurNuP 10.09 vs 11.01; …).
> Underground metabolism is *"dominated by variations in $K_m$ and not $k_{cat}$."*

**Consequence.** Our capacity bound uses only $k_{\mathrm{cat}}$. If promiscuous edges have the same
$k_{\mathrm{cat}}$ as native ones, then **a $k_{\mathrm{cat}}$-only model gives promiscuous flux
away for free** -- the model can route through $\Delta\Pi$ at native capacity with no penalty beyond
the $\ell_1$ term. The thing that actually limits promiscuous flux *in the cell* is affinity, i.e.
$K_M$, i.e. the saturation factor. So:

$$\lvert v_j\rvert\le k_{\mathrm{cat},gj}\,E_g\,\eta^{\mathrm{sat}}_j,\qquad \eta^{\mathrm{sat}}_j=\prod_{i\in\mathrm{sub}(j)}\frac{c_i}{K_{M,ij}+c_i},\qquad c_i=e^{\,u^{\!\top}h^{\mathrm{m}}_i}.$$

$K_M$ moves from "defer until concentrations are anchored" to **"required if we allow promiscuity
at all."** And it folds into the dynamic box (Q2) rather than adding a loss term. The concentration
it needs is the $\ln c_i$ we already introduced for thermodynamics (Q5/Q6) -- so the two
requirements are satisfied by one quantity.

*Also worth noting:* Wu validated wet-lab (predicted geraniol→geranial; Sfa1p and Adh6p confirmed
from 10 ranked candidates) and guarded against hallucinated reactions (95 % of rules regenerate
their source; 79 % of reactions / 85 % of enzyme annotations recovered against UniProt, not just
Yeast-GEM). Code: `github.com/LiLabTsinghua/Yeast-MetaTwin`, Zenodo 10.5281/zenodo.13911783.
**Yeast-MetaTwin itself (16,244 metabolites / 59,865 reactions) is a candidate replacement for
Yeast9 as our $S$** -- a much larger, mostly-predicted network. Flag as a decision, not a default.

### Q13 -- $k_{\mathrm{cat}}$/$K_M$ as distributions, and being conservative early

You asked for distributions rather than point values, since in-vivo values scatter. That fits
cleanly and interacts with the box:

$$\bar v^u_j \;=\; \min\Big(v^u_j(\varepsilon),\ c_j(p)\cdot \sum_g \Pi_{gj}\,\hat k^{(q)}_{\mathrm{cat},gj}\,\bar E_g\,\eta^{\mathrm{sat}}_j\Big)$$

where $\hat k^{(q)}$ is the **$q$-th quantile** of the predictor's posterior. $q$ is a single
interpretable conservatism dial:

- $q=0.1$ → conservative: the model may only use capacity it is confident exists. **Start here**,
  per your instinct.
- $q=0.5$ → median, the usual ecFBA-style point estimate.
- **$q\sim\mathrm{Uniform}$, resampled per forward pass** → the box itself becomes stochastic, which
  composes with the flux-sampling framing of Q7 and propagates parameter uncertainty into the
  predictive interval. This is the principled version and I would aim for it.

Sourcing order stays: **ecYeastGEM published values → Open Enzyme Database
(`yuanOpenEnzymeDatabase2026`, already mirrored) → predicted**, with a per-value provenance tag so
we can ablate "published only" vs "published + predicted."

### Q14 -- Merzbacher 2025 is the betaxanthin baseline, and the bar is low

`merzbacherAccuratePredictionGene2025` is captured and OCR'd at
`$DATA_ROOT/torchcell-library/merzbacherAccuratePredictionGene2025/`. **Merzbacher, Mac Aodha &
Oyarzún, "Accurate prediction of gene deletion phenotypes with Flux Cone Learning," *Nature
Communications* 2025, doi 10.1038/s41467-025-63436-9.**

**Their method is a striking near-miss of ours.** Flux Cone Learning = OptGPSampler hit-and-run
**sampling of the deletion-specific flux cone**, producing (deletions × samples × reactions)
features; a scikit-learn model is trained with the deletion's fitness label broadcast to every
sample; sample-level predictions are averaged to a gene-level call. **That is explicit MCMC flux
sampling plus shallow ML -- the un-amortized version of Q7.** Our proposal replaces the MCMC with a
learned, genotype-conditioned sampler that also sees the 81 % of genes their cone cannot represent.

**Their betaxanthin task, exactly:**

- Data **is Cachera 2023**, confirmed: *"betaxanthin autofluorescence readouts for N = 811 yeast
  deletions were taken from Cachera et al., averaged across four cultures."* The screen has 4,223
  deletions, of which **811 are metabolic genes present in Yeast9** -- those 811 are the whole
  dataset.
- **Reconciliation with our build:** we hold **4,735** deletions with **906** in Yeast9 (19.1 %);
  they hold 4,223 with 811 (19.2 %). **The ratio matches exactly** -- the gap is filtering, not a
  different dataset. Matching them requires intersecting down; exceeding them means using the other
  ~3,900.
- **Regression was attempted and abandoned** -- *"this proved challenging with the limited number of
  knockouts at the high and low ends"* -- and reframed as **3-class classification**
  (low/medium/high) with thresholds set *"qualitatively to label 67 % of samples as medium."*
  Classes: 138 low / 545 medium / 128 high.
- **Split: a single class-stratified random 80/20 at gene level, held constant across models. No
  cross-validation, no seed reported, no gene- or pathway-disjoint holdout, and the split is not
  released.**
- **Headline: 69.8 % 3-class accuracy -- against a 67.2 % majority-class rate.** That is **+2.6
  points over predicting "medium" for everything.** High-producer accuracy is 11.4-23.8 % baseline,
  best 29.5 % after rebalancing. **No correlation or AUC is reported for betaxanthin at all.**
- Deep models and PCA features did not help, which they attribute to fluxes being linearly
  correlated through $Sv=0$.
- Code/data: Zenodo 10.5281/zenodo.15518666. **Reported test-set size is internally inconsistent**
  (Fig. 4b says $N=659$, Table S6 says $N=649$; 20 % of 811 is 162) -- do not cite either without
  checking their code.

**How to run the comparison** (in order):

1. **Reproduce their setting**: intersect our Cachera build with Yeast9 genes, verify we recover
   811 (we will get 906 -- reconcile the 95-gene difference and document it), apply their 3-class
   binning, and run a class-stratified 80/20. Report accuracy and per-class accuracy.
2. **Beat the bar on their own turf** -- same 811 genes, same classes.
3. **Then report what they cannot**: regression on the full ~4,735 (they abandoned regression), and
   the ~3,900 non-metabolic deletions their flux cone has no representation for.
4. **Fix the evaluation** while we are at it: their split is random, so related genes leak across
   the boundary. Report a gene-/pathway-disjoint split alongside, and note the difference.

**Since their split is unreleased and their $N$ is inconsistent, an exact head-to-head requires
re-running their Zenodo code.** Worth budgeting; the alternative is comparing to a number we cannot
reproduce.

### Q15 -- The metabolic-engineering superset build

Inventory done against the canonical 49-dataset table and the adapter map (35 of 49 wired).

**Free additions to `fig6_build.cql` -- wired, queryable, zero new engineering (4):**

| dataset | n | why |
| --- | ---: | --- |
| `OrganicAcidYoshida2012Dataset` | 17 | the only real organic-acid **titer** vector we hold |
| `FattyAcidXue2025Dataset` | 176 | FFA titers on a production chassis; only multi-KO-stacked design data |
| `ProteomeMessner2023Dataset` | 4,699 | enables the enzyme-capacity arm; co-locates with Mülleder |
| `ProteomeZelezniak2018Dataset` | 97 | **1:1 paired with the Zelezniak metabolome** -- the only strains with both proteome and precursor pools |

That makes **11 wired production/metabolite/proteome datasets, ~19.2 k records** -- trivially
buildable. Worth adding as covariates in the same build: Smf fitness (growth normalization),
Kemmeren (deletion→expression bridge), Caudal (isolate axis).

**The real expansion is blocked on two missing adapter families.** 13 further ME-relevant datasets
(~8.2 M records: Hoepfner, Hillenmeyer het/hom, Wildenhain, Vanacloig, Costanzo 2021 condition-SGA,
Mota, Auesukaree, Smith 2006 β-oxidation, Smith 2016 CRISPRi, Mormino, Lian MAGIC, Nadal-Ribelles)
are **loader-only** because:

- **no adapter exists for any `EnvironmentResponsePhenotype` dataset** (37 adapter YAMLs, none
  env-chemgen), and
- **no adapter exists for CRISPRa/i/d perturbations** -- so the entire "up-regulate a flux node"
  modality is unreachable from Cypher, which matters a lot for strain design.

**Also flagged:** `fig6_build.cql`'s filter requires ≥1 deletion with every deletion in
`$gene_set`, which **silently drops addition-only and CRISPR-only strains** -- a superset build must
relax it or those datasets contribute zero rows. And carry forward the privacy/provenance caveats:
Lopez isobutanol and Xue FFA are DOI-less in-house (fine internally, not presentable as sourced);
Lian MAGIC is reprocessed from SRA and portable only with its mirror.

### Consolidated open decisions after this round

| # | decision | recommendation |
| --- | --- | --- |
| 1 | enzyme nodes | **drop** -- gene-centric + soft $\Pi$ (Part 3, Q8) |
| 2 | exactness budget | **box**, now dynamic in $(\varepsilon,p,\mathrm{seq})$ (Q2) |
| 3 | growth objective | **drop the prior; supervise $v_{\mathrm{bio}}$ with fitness** (Q3) |
| 4 | flux head | **stochastic → amortized flux sampler**; report vs FVA width (Q7) |
| 5 | $K_M$ | **required, not deferred** -- it is what makes promiscuity cost something (Q12) |
| 6 | $S$ source | Yeast9 now; **Yeast-MetaTwin is a live option** (Q12) -- decide, do not drift |
| 7 | media | **port the corrected iBioFoundry media** into `torchcell/metabolism/` (Q10) |
| 8 | Methods attention-prior commitment | you are not married to it -- revisit once the flux head is distributional (Q7) |
