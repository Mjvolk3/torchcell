---
id: 6tmj5bvkrdp0e7yz9tfvyiq
title: Metabolic module research plan, week of 2026-09-14
desc: 'Five-agent review of the 026/027 metabolic module, the ranked experiments, and a one-week schedule with contingencies'
updated: 1789371851147
created: 1789371851147
---

## 2026.09.14 - Review, ranking and the week plan

Five review agents (physics, parameters, targets, baselines, literature and generality) read
the 026/027 code, notes, result files and the mirrored papers on 2026-09-14 and each returned
a ranked experiment list. Their reports are in the session scratchpad
(`metabolic-review/agent-{1..5}-*.md`); this note merges them into one plan. Every number
below is cited to a job id, a result file or a probe script named in those reports; a claim
without a measurement is labeled a hypothesis.

Standing rules the plan obeys: the hard physics constraints are hard; the Gibbs dissipation
limit (Niebel 2019) is a held-out readout, never a training term; parameters from predictors
carry distributions; the module must port to a second organism, so nothing yeast-specific is
allowed into the layer; every run goes through slurm, and CPU jobs carry `--nice=1000` with at
most 8 cores and 24 GB so the database rebuild campaign (jobs 1847-1897) keeps precedence.

### What the review measured, ranked by how much it changes the plan

1. **The proposal, not the projection, is why feasibility never moved.** From a random box
   point, K = 20 to 5,000 alternations (von Neumann or Dykstra) leave the active-row balance
   ratio at 2.0 and only empty the network (1,808 to 557 active rows). From a proposal within
   1 percent of a feasible flux, the existing K = 100 alternation reaches max 0.0012 and p95
   0.00038 with 82 active rows (`probe_projrate.py`, `probe_warmstart.py`). A real pFBA flux
   on this GEM has 83 active rows and 10.7 percent of reactions carrying flux; the boxforce
   cells carry 422 to 810 rows and 28 to 32 percent, so the density term pushed away from the
   reference (`probe_fba_rule.py`; jobs 1767/1781/1784). Consequence: the three queued Dykstra
   cells cannot pass the criterion they were queued for; job 1898 was cancelled at 02:40.
2. **The thermodynamic table cannot support a real growth flux.** Under every one of the five
   thermo routes, 24 to 48 flux-carrying reactions cannot be made downhill by any concentration
   in the physiological window, with single-reaction violations up to 313 kJ/mol, and the set
   includes ATP synthase (r_0226), phosphoglycerate kinase (r_0892) and 2-oxoglutarate
   dehydrogenase (r_0831) (`probe_tfa_union.py`). The proton-translocation term that would
   price the proton-motive force exists in `transport.py` and has never been switched on;
   1,468 multi-compartment reactions are priced as if it were free. The GEM itself has 212
   elementally unbalanced internal reactions and 46 with a charge imbalance (`probe_massbal.py`).
3. **Nothing in the loss requires the flux to do anything.** Biomass enters the module only as
   a second-law exemption; a zero flux satisfies every penalty (agent 5, gap table row 13).
   Every wave-7 and wave-8 cell also ran the physics ramp at 0.0, so balance, parsimony and
   the prior on the free per-reaction energy offset were all off (driver log 19:10 on 09-12).
4. **The two headline readouts never touch the flux.** `thermo_log_c` is a sigmoid of the
   pooled context alone (`flux_layer.py:2006-2030`), and both the amino-acid concentration
   readout (+0.0145 over pooled, n = 3) and the betaxanthin precursor readout (+0.0399 over
   the scalar head, n = 3) consume only that tensor. The is-the-module-on control (`ctxconc`)
   was built and never submitted; the betaxanthin equivalent does not exist (agent 3, section 0).
5. **Betaxanthin is not a discriminating target.** A correctly assembled constraint-based
   baseline (Yeast9 + the repo's cassette on YPD, 628 pinned genes, 84 s of CPU) reads
   Spearman +0.047 with permutation p 0.24; FCL's own published number on the same genes is
   +0.039 against a null SE of 0.040 (z = 0.98). Every effect claimed in waves 6 to 8 sits
   below the minimum detectable effect of the design that measured it (MDE 0.073 at n = 3 for
   betaxanthin, 0.047 for amino acids) (agent 4, sections 0 and 5).
6. **The kcat tiers are a provenance ordering, not an accuracy ordering.** The same enzyme
   and substrate reproduces across independent papers at a pooled sd of 0.50 decades (492
   groups, 1,157 papers, Open Enzyme Database); the admitted ensemble's error against the 58
   measured yeast pairs is 0.47 decades; the homolog tier is the widest at 0.73. Any kcat
   distribution must be 0.5 to 0.8 decades wide at every tier. Under the capacity bound
   exactly one reaction binds at the wild-type optimum, ATP synthase, a 17-subunit complex
   whose predictors disagree by 0.99 decades and where our aggregation takes the minimum over
   subunits while GECKO takes the maximum (agent 2, section 1.2).
7. **The benchmark that makes the module interpretable is already on disk.** FCL's yeast
   essentiality split and the E. coli iML1515 ground truth with its FBA baseline sit under
   `$DATA_ROOT/data/merzbacher2025_fcl/`. Essentiality is where FBA scores 93.5 percent and
   FCL 95 percent, and it ports to a second organism unchanged (agent 5, section 3).
8. **Isobutanol is the best-conditioned target of the three and has never been trained on.**
   The biosensor reads the 2-isopropylmalate pool, which is a Yeast9 metabolite (s_0162,
   s_0164); the product has a native exchange (r_1866) so no cassette is needed; test-retest
   across the two screens is Pearson 0.915 on 222 genes, giving a label reliability of
   roughly 0.6 to 0.8; 581 of the 639 pinned test genes are present; the dataset is built and
   served (agent 3, sections 1c and 2).

Corrections recorded: the wave-7 Dykstra bench measured the median rule only, so the earlier
claim that Dykstra was "feasible and non-zero at init" was overstated; the thermodynamic flux
paper the reports build on is Thermo-Flux (`smithThermofluxGenerationAnalysis2026`), not the
Yeast-MetaTwin paper; the all-rows max balance rule is mis-specified because a real FBA flux
fails it (0.098), so the turnover-thresholded rule at 1e-3 is the only one called Genesis from
here on, reported beside the pFBA reference row.

### Ranked experiments

Ranked by information per compute hour with gates first. "CPU" jobs run under slurm at nice
1000. A 200-epoch GPU cell is about 3.3 h; an 80-epoch cell about 1.4 h. Owners are the four
builder agents launched 02:40 (baselines, kinetics, targets, physics).

| rank | experiment | resource | gate or control | pre-registered read | success | kill |
|---|---|---|---|---|---|---|
| 0a | proposal distance of a trained head (job 1784 checkpoint) | CPU, 1 h | none | rel = norm(v_pre - P v_pre) / norm(v_pre), against the warm-start table | rel < 0.1 makes E1 a formality | rel > 1 makes E1 the main event |
| 0b | pFBA reference row in the feasibility audit; retire the all-rows max rule | CPU | none | every table gains a scale | done when committed | none |
| 0c | constraint-based baselines, three targets, proper strain and medium | CPU, minutes each | wild-type-only readout scores exactly 0 | Spearman with permutation p and bootstrap CI, top-10/25/50, tie fraction | any CI excluding 0 with tie fraction below 0.9 | CI spans 0: publish the null, do not tune |
| 0d | linear baselines mirroring the 019 expression family | CPU, half a day | B0 scores exactly 0, NMSE in (0.9, 1.15) | test Spearman and top-50 at the val-selected setting | a CI excluding 0 sets the bar by gene identity | all inside the null: report and stop |
| 0e | FCL three-class scoring of the 027 dumps | CPU, 2 h | the 020 numbers reproduce bit for bit through the adapter | MCC, recall, fraction called medium, top-50 | high-producer recall CI clears 0.295 | fraction medium near 0.95: we reproduced FCL's failure mode |
| 0f | FVA under correlated kcat draws, aggregation and abundance settings | CPU, under 1 h wall | sigma = 0 reproduces `fva_wildtype.csv` | fraction of reactions moving width 2x, growth spread, rho | a ranked sensitive-reaction list | under 5 percent move: the distributional program is a non-problem here |
| 0g | E2 gate: TFA LP with the 47 offenders exempted | CPU, minutes | none | L1 slack after removal | below about 10 kJ | slack stays: run minimum-cardinality removal |
| 0h | E3 gate: transport term on for the three tabulated compartments | CPU, minutes | none | violated count; whether r_0226 leaves the list | count falls below 47 | no change: correctness fix only, no GPU |
| 1 | `ctxconc` x3, the is-the-module-on control for amino acids | GPU, 10 h | conc 0.0648/0.0663/0.0562, pooled 0.0500/0.0425/0.0511 | paired ctxconc minus conc, ctxconc minus pooled | ctxconc minus pooled under 0.005: the module carries the gain | abs(ctxconc minus conc) under 0.010: the module is off for this target |
| 2 | warm-started residual head x3 (v_pre = v_ref + alpha N z, K = 100 boxforce) | GPU, 10 h | 0a; init check must sit near max 0.0012, p95 0.0004, 82 rows | Genesis on the top 100; active rows and density moving toward 83 and 0.107; Spearman within the seed sd of the control | Genesis at or above 0.95 on 2 of 3 seeds | not met on any seed: go to 5 and 6 |
| 3 | isobutanol three-way (ibpool, ibscalar, ibprec) x3 | GPU, 30 h | none; ibpool seeds set the noise floor | paired ibprec minus ibpool, ibscalar minus ibpool; external check on the 224 re-screened strains | ibprec minus ibpool above 0.02 on all seeds | all arms within 0.01: the biosensor reads chromatin state, not pathway flux |
| 4 | `bxctxprec` x3, the betaxanthin context-only control | GPU, 10 h | bxprec 0.1404/0.1377/0.1159 | abs(bxctxprec minus bxprec) | above 0.012: the layer does the work | below 0.012: the wave-8 result is a parameterization result |
| 5 | exempt the thermodynamically impossible reactions x3 | GPU, 10 h | 0g | active-row max and density on the top 100 | max below 2.0 | unchanged: the table is not the blocker |
| 6 | transport term on x3 | GPU, 10 h | 0h | same as 5, plus r_0226 sign | same | same |
| 7 | parsimony and active balance outside the ramp | GPU, 13 h | density weight 0 control | density toward 0.107, rows toward 83, p95 | density below 0.20 | sparsity does not come: the sigmoid magnitude cannot be made sparse, build the L0 gate |
| 8 | `nodiss` seeds 2701 and 2702 | GPU, 7 h | concver 0.0580/0.0725/0.0554; job 1842 is seed 2700 | paired difference; g_diss of the unpenalized arm against 3,700 J/gDW/h | under the limit unasked | none; this makes dissipation reportable at all |
| 9 | held-out amino-acid columns (supervise 13, score 6) x3 plus context control x3 | GPU, 20 h | unsupervised baseline -0.004 and +0.007 | held-out mean Spearman, flux-coupled vs context-only | above 0.02 and above context-only by 0.010 | both near 0: the readout is 19 independent sigmoids |
| 10 | medium applied before the layer (bxprec on YPD, conc on minimal) x3 each | GPU, 20 h | the amino-acid arm must not move | asymmetric: amino acids within 0.005, betaxanthin beyond its 0.026 paired sd | betaxanthin moves | no movement: retract the wrong-medium explanation |
| 11 | gene essentiality as the endpoint, rung 0 (pooled, flux_off, flux_capacity, flux_shuffled) | GPU, 12 h | FCL yeast AUROC and FBA baseline; majority class | AUROC on the class-stratified 20 percent | flux arms beat pooled on 3 of 3 and beat shuffled | below majority or equal to shuffled: the kinetic tables are decoration on this endpoint |
| 12 | sample the kcat bound per forward pass x3 x2 | GPU, 9 h | 0f must not kill; off arm bit-identical | capacity-active fraction, fraction pinned on a bound, Spearman as secondary | pinned fraction falls 10 points with Spearman inside 0.05 | active fraction goes to 0: the draw dissolved the constraint |
| 13 | learned per-enzyme kcat residual (ProtT5) with the held-out-measured check | GPU, 15 h | control B driven by the perturbed embedding | median error on 20 held-out measured units | below the ensemble's 0.308 decades and not on control B | indistinguishable from control B: a slack variable, drop it |
| 14 | proton and charge balance (Thermo-Flux step 5, charge pseudo-metabolite row) | 1-2 days code, then GPU | initialization read first | balance residual decomposed by protons, charge, rest | max-rule ratio falls below 2.0 | unchanged: the defect is in the parameterization |
| 15 | iML1515 portability, rung 1 (thermo-free) | 2 engineer days plus 4 GPU h | recomputed FBA at the 0.41 cutoff must reproduce 93.5 percent | accuracy on the held-out 300 genes, 5 repeats | within 2 points of our FBA | below majority: the architecture does not transfer |
| 16 | shared concentration latent, amino acids plus isobutanol | GPU, 20 h | ranks 1 and 3 both positive | joint minus isobutanol-only, paired | direction check only at n = 3 | the prior joint run measured -0.020 (n = 4); report as underpowered |
| 17 | power runs: betaxanthin parity to n = 12 (79 h) or amino acids to n = 7 (46 h) | GPU | only after rank 1 decides which contrast is real | paired CI on Spearman with top-50 as co-primary | CI excludes 0 or lies inside 0.02 | a CI wider than 0.02 that spans 0: the endpoint cannot decide it |

Stopped by this plan: the three Dykstra cells; the K-draw ladder (closed null twice at n = 3,
and FCL's own ablation shows 10 samples per cone match FBA); raising the density weight; the
all-rows max rule; further seeds on betaxanthin Spearman contrasts until rank 1 and rank 4
say the module is on.

### The week

Four GPUs are shared with the rebuild campaign, whose 120 GB memory requests block backfill of
a 60 GB GPU cell while the node's free memory is under 120 GB, so the GPU lane may sit idle for
hours at a time; the CPU lane is nice 1000 and simply waits. Order within the GPU lane is the
rank order above.

- **Day 1 (Mon 09-14).** CPU: 0a to 0h, all submitted by the builders. GPU: `ctxconc` s2700
  (job 1899, queued) then s2701 and s2702; warm-start s2700 once its init check passes; ibpool
  and ibprec s2700; `nodiss` s2701. Code landing: width-1 guard, isobutanol head, bxctxprec,
  held-out mask, `--apply-media`, warm-start head, reference row, probes as scripts, sigma
  table, p_avail record.
- **Day 2 (Tue).** Read 0a to 0h and rank 1. GPU: warm-start seeds 2701 and 2702 if seed
  2700 is alive at epoch 20; the rest of the isobutanol three-way; bxctxprec x3; ranks 5 and
  6 only if their gates pass; `nodiss` s2702.
- **Day 3 (Wed).** Read ranks 1 to 4. GPU: rank 7 (parsimony) and rank 9 (held-out columns)
  if rank 1 says the module is on; rank 10 (media) on whichever target survived; rank 12 if
  0f did not kill it. Write the day-3 note with every number and its job id.
- **Days 4 to 5 (Thu, Fri).** Rank 11 (essentiality rung 0), rank 13 (kcat residual), and
  the first half of rank 14 (charge row at initialization). Start rank 15's data steps on
  CPU: iML1515 to the mirror with provenance, the E. coli UniProt table, M9 medium.
- **Days 6 to 7 (Sat, Sun).** Rank 15 rung 1 on GPU; rank 16 if ranks 1 and 3 were both
  positive; rank 17 on the one contrast the week showed is worth powering. Close with a
  ranked table of every pre-registered read and its outcome.

### Contingencies

- **Rank 1 says the module is off for amino acids** (abs difference under 0.010): retire the
  +0.0145 claim in the wave-7 note; every concentration readout from then on must be
  flux-coupled (the flux-force channel with the concentration readout, no code, 10 GPU h) or
  it is not a module result; skip the amino-acid power run; rank 9 runs only in its
  flux-coupled form.
- **Rank 1 says the module is on** (ctxconc minus pooled under 0.005): promote rank 9 and the
  amino-acid power run (n = 7) ahead of the betaxanthin work.
- **0a reads rel above 1**: rank 2 is the main event and gets seeds before anything else in
  the GPU lane. If rank 2 passes Genesis but the Spearman collapses below the control by more
  than the seed sd (0.076), the head is pinned to the reference flux: read the across-genotype
  flux sd, and if it is near zero ramp alpha over training and rerun one seed.
- **Rank 2 fails Genesis on every seed**: go to ranks 5 and 6, whose gates were run on day 1.
  If 0g's slack does not clear even under minimum-cardinality removal, the infeasibility is
  distributed and the next lever is the exact metric projection (dual ascent in float64 on the
  2,593 independent rows, a project of 400 to 600 lines), not another penalty weight.
- **0c finds signal in a constraint-based readout** (a CI excluding 0 with ties below 0.9):
  the constraint-based baseline enters the paper's baseline table, and the isobutanol flux
  arm's product column has a meaning; add the pooled-enzyme constraint to the baseline.
  **0c is null on all three targets**: publish the null; the module has to beat gene identity
  (0d), not a flux model.
- **0d finds a ridge on protein-language-model embeddings beats the CGT pooled arm**: the
  paper states that the bar is set by gene identity; every module contrast is then read
  against the ridge, not against pooled.
- **Rank 3, ibscalar wins**: the first evidence anywhere that a flux readout beats a
  concentration readout; run 3 more seeds before believing it.
- **0f kills the kcat distribution** (under 5 percent of reactions move): drop ranks 12 and
  13, spend the budget on the pool constraint and the promiscuity overcount instead, and
  report the CORAL result reproducing in yeast.
- **GPU starvation persists**: the CPU lane (0a to 0h, all baselines, the kcat sweep) is the
  whole of day 1 anyway; if by day 2 no GPU cell has started, ask whether the rebuild jobs
  may take a lower priority, which is reversible.
- **A builder breaks an import while cells run**: the freeze wrapper protects submitted
  cells; a shell smoke does not. Every builder was told to leave the module importable after
  every edit and to run the tests after every block.

### Evaluation rules for the week

- Primary statistic unchanged: pinned test Spearman at the epoch the centered 5-epoch moving
  average of validation Pearson selects, never a max over epochs. Co-primary from now on:
  top-50 precision on high producers against the 0.156 base rate, the best-resolving of the
  five statistics measured on the wave-7 pairs (agent 4, section 5a).
- Every contrast names both members and states whether its sd is paired.
- Feasibility is the turnover-thresholded rule at 1e-3 on the top 100, reported beside the
  pFBA reference row and split by reaction class (exchange and demand, transport, internal).
- Dissipation is reported only from the `nodiss` arm; every anchored number before
  2026-09-14 is labeled "trained against".
- Nothing enters a note or the paper unless a committed script under `experiments/` wrote it.

### 2026.09.14 evening - first reads against the plan

- **Rank 1 decided: the module is off for the amino-acid concentration readout.** The
  context-only control `ctxconc` (jobs 1899/1930/1931, runs p0qg8du0/57tyvuib/l2f1xpts) reads
  0.0545 / 0.0601 / 0.0670 against the flux-coupled `conc` cells' 0.0648 / 0.0663 / 0.0562 at
  the same seeds: paired conc minus ctxconc +0.0103 / +0.0062 / -0.0108, mean +0.0019, sd
  0.0113, inside the kill rule's 0.010. Against pooled the control reads +0.0127. The +0.0145
  gain recorded in the wave-7 note is the sigmoid-window parameterization on the pooled
  context. The plan's contingency applies: no amino-acid power run, and every concentration
  readout from here is flux-coupled (the warm start, or the flux-force channel with the
  readout) or it is not a module result.
- **Rank 4, one seed:** `bxctxprec` s2700 (job 1932, run oi8k8rno) reads 0.0794 against
  `bxprec` 0.1404 and pooled 0.1007 at the same seed. A 0.061 gap is above the 0.012 paired
  sd, in the direction "the layer does the work on betaxanthin", at n = 1; seeds 2701 and
  2702 are queued (job 1943 and its sibling).
- **Rank 3, first seed of the isobutanol three-way:** `ibpool` s2700 0.1278 (job 1926, run
  6z2ovude, selected at epoch 21) and `ibprec` s2700 0.0762 (job 1928, run s4z0ur7o, selected
  at epoch 13) on 504 pinned genes. Both selected very early; the plateau diagnostic
  (`val_pred_sd_ratio` at the selected epoch) is read before either number is interpreted.
  `ibscalar` s2700 and the s2701 pair are queued (jobs 1939, 1940, 1942).
- **Rank 8, one seed:** `nodiss` s2700 (job 1842) 0.0682 against `concver` 0.0580, so the
  dissipation penalty off is not worse at this seed.
- **Physics gates (jobs 1904 to 1938).** A trained head's proposal is 99 percent discarded
  by the projection (rel 0.990). The 47-reaction exemption takes the TFA slack to 0.000. The
  transport term fixes ATP synthase's sign and raises the violated count, so it gets no GPU.
  With the nominal abundance path, the exemption, the TFA-consistent concentration
  initialization, the normalized availability fold (an intact 17-subunit complex read 0.646
  under the shipped softmin) and a two-sigma kcat ceiling, the warm-started head reads at
  initialization max 0.0306 / p95 0.0019 / 81.8 active rows / density 0.177 / proposal
  distance 0.018 against the pFBA reference's 82 rows; the residual 3x on the maximum is
  per-genotype availability spread. The cells are written; they run once the kcat ceiling
  widening is reachable through the parameters module.
- **Rank 0d decided: the bar is gene identity.** A cosine k-nearest-neighbor average over
  protein-language-model embeddings of the deleted gene (k = 25, validation-selected, no
  fitted parameter) reads pinned betaxanthin Spearman +0.1446 (n = 629, permutation p under
  0.0005, bootstrap CI +0.061 to +0.224, top-50 precision 0.50 against 0.157) against the
  CGT pooled arm's 0.1148 and FCL's 0.039 (job 1909). On isobutanol the same ladder reads
  +0.198 (CI +0.120 to +0.274, n = 581; job 1912), so any isobutanol arm has that to clear.
  On amino acids the ladder and the model sit in one band (+0.045 vs pooled 0.040; job 1911).
- **Rank 0c decided: the constraint-based baseline is a measured null** on betaxanthin (every
  CI spans zero, ties 0.85 to 0.98; job 1901) and amino acids (+0.004; job 1902). The one
  readout with a CI excluding zero is isobutanol's mitochondrial 2-isopropylmalate capacity,
  +0.120 (p 0.005, n = 581) with a tie fraction of 0.932, just above the rule's 0.9 (job
  1903's log; job 1941 rewrites the JSON). Two defects surfaced: the Cachera loader records
  SC for a YPD screen, and the YPD stand-in lacks uracil and adenine, so the ura3 marker
  alone zeroes growth; SC is the primary condition. The shipped FVA reference is against
  the wrong medium (licensed set down 22.7 percent on YPD; job 1905).
- **Rank 0e decided:** every 027 arm reproduces FCL's failure mode (93 to 100 percent of
  genes called medium); best MCC 0.157 on the pooled arm (job 1908).
- **Rank 0f decided: the kcat distribution is not a non-problem.** Under correlated draws
  (rho 0.22) 69 percent of reactions move their FVA width by more than 2x and growth spreads
  165 percent of its median; the shipped minimum-over-subunits aggregation costs 13.7 percent
  of growth where GECKO's maximum costs nothing; ATP synthase is the top bottleneck (jobs
  1915 to 1918). The "1.5e-7 box width" figure does not reproduce (median 0.093; job 1921).
  The parameters module now carries `complex_aggregation` and `kcat_ceiling_sigma_k`.

### 2026.09.14 night - the warm start, the reference, and a defect in the genotype path

- **Rank 2 (job 1946, run 6s8r3aov) never left the plateau and lost feasibility as it
  trained.** Training-batch p95 started at 0.007 and density at 0.175, but the active-row max
  was about 1.0 from epoch 0 on real genotypes (the init probe's 0.03 was on four wild types;
  on 32 real genotypes the same head reads 0.70, job 1958), the learned residual scale grew
  from 0.010 to 0.087 and the balance max hit 2.0 at epoch 45; validation Pearson flat at
  -0.05, spread ratio under 0.012, data loss flat at 0.527. Dissipation stayed at 1,300 to
  1,570 J/gDW/h with the penalty off, the first held-out dissipation readout.
- **A per-genotype pFBA reference (job 1957) exposed a defect upstream of the module.**
  Every betaxanthin genotype carries YBR249C (ARO4) and YPR060C (ARO7) in its perturbed
  gene set, so every knockout reference had growth 0.0 on the shipped bounds. The Cachera
  loader records those two as GeneAddition (the cassette's feedback-resistant alleles), but
  the graph processor collects every perturbation's gene regardless of type, so the flux
  layer's availability gate has treated ARO4 and ARO7 as deleted in every betaxanthin flux
  cell since wave 5, shutting chorismate mutase in strains that overproduce through it. The
  backbone removes both nodes from every Cachera cell graph too, a constant across strains.
  Fix in flight: an additive deletion-only index on the batch, routed to the gate and the
  reference lookup; the reference table keyed on deletion-only sets, on the dataset's medium.
- **Rank 4 at two seeds: +0.061 then +0.003.** The betaxanthin context control matched the
  flux-coupled readout at seed 2701 (0.135 vs 0.138; job 1943); seed 2702 queued (job 1959).
- **Essentiality (rank 11) corrected before it runs.** The 93.5 and 95 percent figures are
  E. coli; for yeast FCL reports AUROC 0.73 against FBA 0.69 on 223 test genes with a
  majority rate of 0.861. The released split's essential column is inverted (1.0 =
  nonessential); the loader fixes it and asserts the class counts. The dataset needs a
  graph query with the boolean kept as a boolean; head, biomass readout and tests landed.
- **The amino-acid target's ceiling is measured** (`mulleder_retest_reliability.py`): 194
  strains measured twice give retest Pearson 0.43 and Spearman 0.20, reliability 0.42; the
  QC sample disagrees with itself by 0.03 to 0.05 log10 while two cultures disagree by 0.16
  to 0.27. Models at 0.02 to 0.08 are far below the ceiling; the target is weak because most
  deletions move pools by less than one culture's noise. Amino-acid GPU work stops here.

### 2026.09.15 - What the warm-start line established, and where it stops

- **Rank 2 is the first cell to meet the honest rule on a real genotype, and it meets it on
  46 of the top 100.** Job 1946, audited from its selected checkpoint at epoch 3 (job 1985,
  `audit_wave9_warmstart_s2700.json`): on the top 100 the box and the model second law are
  exact on 100 of 100, and the turnover-thresholded active-row rule at 1e-3 passes on 46
  (p95 half on 81); at the 1e-2 threshold the max passes on 88 and the p95 on 90. The
  per-strain median active-row maximum is 0.0106 over 81 active rows, against the pFBA
  reference's 82 rows; every boxforce cell before it read 0.00 under this rule. The trainer's
  own block reports the all-rows maximum at 2.0, the statistic the review retired as solver
  noise (the pFBA reference reads 0.098 on it), which is what an earlier draft of this
  addendum mistook for the ceiling. What the trajectory shows (run 6s8r3aov): the priced
  balance term closed the gap within three epochs, then the data term grew the residual
  scale from 0.010 to 0.087 and the worst strain's maximum went to 2.0 by epoch 45. The
  cell is a constant predictor for betaxanthin (pinned -0.028), ran under the old gate and
  the shipped bounds, and is one seed. Hypothesis (untested): freezing the residual scale at
  its initial value holds feasibility for the whole run; that is the next cell.
- **The 50x between the probe and the real batch is the medium and the gene universe, not
  the genotypes.** The 2x2 (jobs 1974 and 1980) crosses shipped bounds against YPD and wild
  types against 32 real deletion genotypes with the gate fixed: wild type to deletions moves
  the active-row maximum by 1.00x on both media; YPD moves the p95 by 20.5x; and a third
  axis nobody had varied, initializing the layer's heads over the transformer's 6,607 gene
  tokens rather than the GEM's 1,161 genes, moves it by 10.6x. Re-solving the exemption list
  and the concentration witness on YPD against the YPD reference (job 1975) closes the sign
  axis (forbidden reference reactions 73 to 4) and leaves the balance at the ceiling; what
  remains is capacity, because the YPD reference has norm 103 against 20 on the shipped
  bounds and ATP synthase is 3.4x over its ceiling again. Best real-genotype cell: shipped
  bounds, maximum 0.75 and p95 0.024, short by 7.5x and 2.4x. No cell is ready.
- **A per-genotype reference does not carry the genotype, by arithmetic.** Of the 4,930
  deleted genes, 3,991 are not in the GEM; of the 939 that are, 201 move the pFBA optimum on
  YPD (221 on the shipped bounds), and on a random batch of 32 that is 2 references that
  differ from the wild type (job 1971). MOMA is not a different reference: it sits 0.006 of
  the wild-type norm from the wild type and 0.39 from the pFBA knockout, and 11 of 28
  "moved" pFBA references moved only by LP degeneracy (job 1982). A flux reference that
  varies with the genotype has to come from somewhere other than a knockout optimum.
- **What this leaves for the feasibility program.** The proposal-not-projection diagnosis
  holds on wild types and is untested on real genotypes at the reference's own medium; the
  next lever is the capacity ceiling at YPD (a 2-sigma kcat widening was enough on the
  shipped bounds and is 3.4x short on YPD), and the honest statement for the paper is that
  no head has produced a balanced, sign-consistent, non-zero flux on a real deletion
  genotype. The isobutanol pooled cells escape the plateau and every flux readout on that
  target does not, so the endpoint that can show the module working is essentiality, where
  the flux itself (growth) is the label.

### Housekeeping the review surfaced

- `TC_LIT_URL` and `TC_LIT_API_KEY` are absent from the worktree `.env` on GilaHyper; the
  server is local (`http://localhost:8723`), so the two lines should be added.
- The mirror holds two Zotero items for the Merzbacher paper with byte-identical text; flagged,
  not fixed.
- `BETAXANTHIN_PRECURSORS` in 026 still lists s_0188 (2-phospho-D-glycerate) as an aromatic
  intermediate; the scalar head that "costs -0.023" reads that list.
- The isobutanol screen LMDB is mid-rebuild (job 1894); anything reading it records which copy.

## 2026.09.15 - Correction: the isobutanol label is not well conditioned genome-wide

Item 8's "test-retest Pearson 0.915, reliability 0.6 to 0.8" was the pooled first-screen vs
re-screen correlation over the 225 strong hits, which spans two blocks a decade apart and is
inflated by selection on extremes. Measured by `isobutanol_retest_reliability.py`
(`results/isobutanol_retest_reliability.json`), on log10 fold change:

| read | n | Pearson | Spearman |
|---|---:|---:|---:|
| first screen, ORFs on two or more rows (unselected) | 190 | 0.106 | 0.309 |
| same, dropping 4 pairs disagreeing by more than 1 decade | 186 | 0.289 | 0.350 |
| first screen vs re-screen, pooled hits | 225 | 0.922 | 0.704 |
| same, up block only (FC >= 2) | 64 | 0.414 | 0.276 |
| same, down block only (FC <= 0.5) | 161 | 0.681 | 0.261 |

- Single-measurement reliability genome-wide: 0.11 by sd, 0.50 by MAD; Pearson ceiling
  0.33 / 0.71. The pinned Spearman is scored genome-wide, so the unselected row applies.
- Direction is reproducible: 99.6 percent of re-screened hits keep their side of FC 1.
- The triplicate re-screen is tight: median sd 0.037 log10 against a genome-wide spread of
  0.216, so the first screen's disagreement is larger than culture-to-culture noise.
  Hypothesis (untested): plate or position effects in the one-shot screen, or duplicate rows
  that are distinct YKO isolates (YBL071W-A sits in both re-screen blocks).
- Consequence for rank 3: kNN +0.198 and ibpool s2700 +0.128 sit against a Spearman retest of
  about 0.31, not against a near-perfect label. The re-screen's 225 strains are the only
  replicated readout, and they are held out.

## 2026.09.15 - Ceilings for every metabolic target, and what they reorder

`target_reliability_table.py` estimates each target's reliability from its own repeats and
writes `results/target_reliability.json` plus the notes-tex table
`tables/target_reliability.tex` (Table 15 of the 027 document).

| target | repeats | reliability of the stored value | Pearson ceiling | one measurement |
|---|---:|---:|---:|---:|
| betaxanthin | 4,722 genes, median 15 colonies | 0.84 | 0.91 | 0.28 (ceiling 0.53) |
| amino acids (19) | 194 ORFs in two batches | 0.42 | 0.62 | same |
| isobutanol | 190 ORFs on two or more screen rows | 0.11 | 0.33 | same |

- **Betaxanthin's label is the best of the three**, because the release averages a median of
  15 colonies per gene over screen replicates 1, 2, 4 and 6. The flat betaxanthin result is
  therefore not label noise: kNN reads +0.145 and the module's arms near zero against a label
  that supports 0.91 Pearson. Caveat: colonies are assumed independent within a gene, and a
  shared plate or position effect would inflate the estimate.
- **This reverses item 8 of the ranking.** Isobutanol was called the best-conditioned target
  on the strength of a selection-inflated 0.915; genome-wide it is the weakest of the three.
- Rank 3's kill rule is unchanged, but its interpretation is: an isobutanol arm reading
  +0.13 is already near half of the 0.31 retest Spearman, while a betaxanthin arm reading
  near zero has no label-noise excuse.

## 2026.09.15 - All six metabolite targets, and the carotenoid screen

The generator now covers every metabolite dataset in the build, not only the trained three,
and two details of the betaxanthin estimator were fixed: the control row (`gene == "0"`,
87 colonies, sd 1.20) is excluded, and the within-gene variance is pooled with `n - 1`
weights rather than averaged. The paper's own three-colony rule sets the inclusion cut.

| target | head | estimator | repeats | features | reliability | ceiling |
|---|---|---|---:|---:|---:|---:|
| betaxanthin | yes | variance components | 4,696 | 1 | 0.84 | 0.92 |
| amino acids | yes | retest | 194 | 19 | 0.42 | 0.62 |
| isobutanol | yes | retest | 190 | 1 | 0.11 | 0.33 |
| carotenoid (Ozaydin) | no | one-way ICC | 129 | 1 | 0.11 | 0.33 |
| metabolome (Zelezniak) | no | one-way ICC | 18 | 43 | 0.62 | 0.79 |
| lipids (da Silveira) | no | SSMD back-solve | 130 | 138 | 0.82 | 0.91 |

- **Cross-check on betaxanthin**: Cachera reports a trial-to-trial correlation of 0.765, and
  Spearman-Brown over four combined screens implies 0.93, a ceiling of 0.96. The variance
  split reads 0.84 and 0.92, so the two routes agree that the label is strong.
- **The carotenoid screen is a weak label**, 0.11 ICC on a 6-level visual score. Its second
  round of transformations re-screened only strains scoring 2.5 or higher, so it cannot
  improve the estimate. A carotenoid head would be scored against a ceiling of 0.33.
- **The metabolome and the lipids are the context sets worth a head**, at 0.79 and 0.91,
  with the caveats that the Zelezniak replicates exist only on the re-grown variable subset
  and the lipid number is a back-solve rather than a measured repeat.
- Two loader gaps found while surveying, neither blocking: the Ozaydin loader discards the
  workbook sheet holding the second round of transformations, and the da Silveira loader
  hardcodes two replicates with no standard error while the SSMD sheet carries a recoverable
  dispersion for most cells.

## 2026.09.15 - Which runs satisfy plain FBA feasibility, with no thermodynamics

Scanned all 31 audit files in `results/` for the two halves of ordinary FBA feasibility:
bounds (`box_violation_frac_eq_0`, which includes the capacity bound) and steady state
(`balance_active_max_ratio_1e-03 <= 1e-2`, the turnover-thresholded rule calibrated against
pFBA). No cell satisfies both halves on the top-100 strains. Each half has a family that
satisfies it outright.

| family | cells | bounds | steady state, all 4,930 | steady state, top 100 | active reactions |
|---|---|---|---|---|---|
| `flux_projection` (wave 6, seeds 2700-2702) | 3 | 0 percent | 100 percent | 100 percent | 88 percent |
| warm start `boxforce` (job 1946, epoch 3) | 1 | 100 percent | 98 percent | 46 percent | 17 percent |
| every other audited cell | 24 | 100 percent | 0 percent | 0 percent | varies |
| pFBA reference on the same GEM | -- | exact | exact | exact | 10.3 percent |

- **The projection arm solves steady state and nothing else.** Its balance residual is
  essentially exact (median ratio 0.0, active-row max 1.1e-5, tighter than the pFBA
  reference reads on the plain max rule), because a null-space projection enforces Sv = 0 by
  construction. It pays for it everywhere else: half of all reactions sit outside their
  bounds (median box violation 0.4994, and 0.598 on the top 100) and 88 percent of reactions
  carry flux against pFBA's 10.3, so it is a dense out-of-bounds circulation rather than a
  flux a cell could run.
- **The box family solves bounds and approximates steady state.** Bounds and capacity hold
  exactly by the sigmoid parameterization in every cell. Only the warm-started cell then
  gets the balance residual down, and only for 3 epochs.
- **On the top 100 the binding constraint is mass balance, not thermodynamics.** Job 1946
  reads 100 percent on the second law under the layer's own potentials there, so its Genesis
  fraction of 46 percent is entirely the balance half. At standard state no cell passes,
  which is the learned concentration term doing the work.

## 2026.09.15 - Freezing the constraint scale nearly closes Genesis, and the medium decides it

Jobs 1986 and 1987 ran the warm-started boxforce arm with the constraint scale frozen, the
lever the 09-15 note proposed after job 1946 lost feasibility as its scale grew. Audits 1994
and 1995, under the turnover-thresholded rule at 1e-3.

| cell | medium | select ep | Genesis, all 4,930 | Genesis, top 100 | Genesis, pinned test top 100 | active rows | active reactions |
|---|---|---:|---:|---:|---:|---:|---:|
| 1946 shipped, scale free | shipped | 3 | 0.98 | 0.46 | 0.64 | 81 | 17.1 percent |
| 1986 shipped, scale frozen | shipped | 7 | 0.98 | **0.88** | 0.74 | 81 | 17.9 percent |
| 1987 YPD, scale frozen | YPD | 4 | 0.01 | 0.10 | 0.05 | 207 | 23.9 percent |
| pFBA reference | shipped | -- | exact | exact | exact | 82 | 10.3 percent |

- **Freezing the scale is the lever.** Top-100 Genesis moves 0.46 to 0.88 at matched seed,
  medium and arm, with the box exact throughout and the second law at 1.00 under the layer's
  own potentials. The criterion is 0.95 on 2 of 3 seeds, so this is close and not met; seeds
  2701 and 2702 are jobs 1999 and 2000.
- **YPD destroys it.** The same cell on the dataset's own medium reads 0.10 on the top 100,
  because open uptake gives a denser flux: 207 active balance rows against 81, and 23.9
  percent of reactions carrying flux against pFBA's 10.3. Feasibility on the shipped bounds
  is therefore partly an artifact of a closed medium, and the honest version of the Genesis
  claim has to name which medium it was met on.
- **The predictor is a constant in both.** Pinned Spearman 0.027 (shipped) and 0.053 (YPD),
  so the top 100 is a near-arbitrary subset rather than the strains the model believes in.
  A feasibility fraction measured on a collapsed ranking is weaker evidence than the same
  fraction under a predictor that ranks, and the wave-10 cells have to be read that way.
- Job 2001 raises the active-row balance weight from 0.5 to 2.0 at seed 2700 to test whether
  the remaining 12 points are reachable by pricing alone.

### Correction, same day: the 0.88 does not replicate, and the selector picks the epoch

Audits 2010, 2011 and 2012 put the other two seeds and the balance-weight variant through
the same rule. The frozen-scale result is one seed at one early epoch, not a property of the
arm.

| cell | weight | select ep | Genesis top 100 | active balance rows | active reactions |
|---|---:|---:|---:|---:|---:|
| 1986 s2700 | 0.5 | 7 | 0.88 | 81 | 17.9 percent |
| 1999 s2701 | 0.5 | 149 | 0.00 | 9 | 12.7 percent |
| 2000 s2702 | 0.5 | 51 | 0.00 | 9 | 12.7 percent |
| 2001 s2700 | 2.0 | 27 | 0.00 | 84 | 25.0 percent |
| pFBA reference | -- | -- | exact | 82 | 10.3 percent |

- **The two seeds that selected late emptied the network.** Nine balance rows turn over above
  the threshold against the reference's 82, so those cells are not balancing a flux, they are
  carrying almost none. The earlier reading that freezing the scale is the lever was
  measured at one seed and is withdrawn.
- **Epoch and seed are confounded in every one of these numbers**, because the audit scores
  the checkpoint the selector kept and the selector runs on the validation Pearson of a
  near-constant predictor, which is noise. Job 1986 was audited at epoch 7 only because its
  selector happened to land there.
- **More balance pricing made it worse, not better.** At weight 2.0 the flux got denser,
  25.0 percent of reactions against 17.9, and Genesis went to 0.00. Hypothesis (untested):
  the penalty is a ratio and raising turnover inflates its denominator, so pricing it harder
  buys density rather than balance.
- Jobs 2016 to 2018 rerun the arm at a 10-epoch budget on three seeds, which forces selection
  inside the early window and makes the audit a statement about an epoch we chose rather than
  one the selector wandered into.

## 2026.09.15 - The penalties are what collapse the predictor, with a dose-response

Every readout that goes through the flux layer collapses and no readout on the pooled context
does. Measured at each cell's selected epoch, the validation prediction spread ratio is 0.37
to 0.46 on the three context-precursor cells and 0.0000 to 0.06 on every flux cell. The cause
is not the layer's presence. It is the price put on the physics, and the wave-5 cells already
carried the dose-response.

| constraint scale | cells | prediction spread ratio | pinned Spearman |
|---:|---|---|---|
| 0.0 | wave5 g7, six seeds | 0.16 to 0.40 | 0.023 to 0.142 |
| 0.025 | wave5 g10 | 0.084 | 0.045 |
| 0.1 | wave5 g1, g6 | 0.0012 and 0.0000008 | 0.020 and -0.016 |
| n/a (no layer) | wave5 g8 pooled, three seeds | 0.34 to 0.50 | 0.072 to 0.169 |

The same ordering holds for the other physics terms. The frozen-scale warm-start cells run
the constraint scale at 0 but carry an active-row balance weight of 0.5 and a parsimony weight
of 1e-3, and they collapse; at balance weight 2.0 the spread ratio reads 0.0000 and Genesis
drops to 0.00.

**What the collapse looks like inside the layer** (`probe_flux_genotype_spread.py`, job 2029,
the frozen warm-start cell at seed 2700, 933 genotypes): the per-reaction standard deviation
of the flux ACROSS genotypes, divided by the layer's own flux scale of 100, reads a median of
2.5e-7, a p90 of 3.1e-5 and a maximum of 6.5e-3. Every strain is being assigned nearly the
same flux vector, so the head is starved rather than lazy, and no loss, spread or schedule can
recover a signal that is not in its input.

Correction to the first reading of this probe: the statistic that divides a reaction's spread
by its OWN mean reported a median of 6.7, which is noise on near-zero fluxes and not signal.
The scale-normalized version above is the headline, and the script now reports it as such.

**The ladder now running** varies only the active-row balance weight on one arm and one seed:
0.0 (job 2026), 0.1 (2027), 0.5 (job 1986, already read) and 2.0 (job 2001, already read),
plus a cell with every flux penalty off (2030) that also gives the probe a checkpoint for the
penalty-free configuration, which the wave-5 cells predate. Each gets the feasibility audit
and the spread probe, so the output is the tradeoff curve between a flux that satisfies the
physics and a model that predicts anything.

## 2026.09.15 - Essentiality at three seeds: a direction against the module, and the pooled arm collapses too

All six cells of rank 11 have landed (jobs 2008, 2009, 2021 to 2024; released FCL split, 195
pinned test genes, 31 essential, 80 epochs, selection on validation). Paired by seed with
`essentiality_contrast.py`, which writes `results/essentiality_contrast.json`:

| seed | pooled AUROC | flux AUROC | flux minus pooled | collapsed |
|---:|---:|---:|---:|---|
| 2700 | 0.7927 | 0.5398 | -0.2529 | flux (spread ratio 0.0005) |
| 2701 | 0.7882 | 0.7579 | -0.0303 | neither (2.60 and 0.79) |
| 2702 | 0.5784 | 0.5115 | -0.0669 | BOTH (0.0000 and 0.0001) |

Mean difference -0.117 with sd 0.119, 3 of 3 negative (p = 0.25 under the null). On the one
seed where neither arm collapsed the gap is -0.030, inside the seed-to-seed spread. So the
finding is a direction against the module and not a size: the first seed pair's -0.25 was the
one pairing where only the flux arm collapsed.

Three cells clear the published FCL test AUROC of 0.73: pooled 2700, pooled 2701, and flux
2701. No cell of either arm recalls a single essential gene at the 0.5 threshold except the
two live pooled cells (0.32 and 0.26), which is why accuracy sits at the majority rate 0.841
everywhere else.

**Correction to the section above.** "No readout on the pooled context collapses" was true
of the betaxanthin context-precursor cells and is false on essentiality: pooled seed 2702
collapsed with a spread ratio of 0.0000 at its selected epoch 28. The flux arm collapses more
often (2 of 3 against 1 of 3), and the penalty dose-response on betaxanthin stands, but
collapse on this endpoint is not a property of the layer alone. Hypothesis (untested): the
binary target with 16 percent positives gives the selector a flat validation surface early,
and whichever arm lands on the constant first stays there.

W&B, one group per arm:
`https://wandb.ai/zhao-group/torchcell_027_bxfx/groups/w9esspool` and
`https://wandb.ai/zhao-group/torchcell_027_bxfx/groups/w9essflux`.
