---
id: hkcml1pl3ia0i5172ptddeq
title: next-strains-to-construct
desc: ''
updated: 1786931101296
created: 1786931101296
---

## 2026.08.23 - Capped build: 45 strains for the first trigenic round

**45 new strains: 25 doubles + 20 triples**, chosen by the `capped` strategy
(cap 6 on zero-trigenic-data genes, 20 triples).
Bench-facing list: [[experiments.W019-echo-crispr-array.build-list]]. All tables from
`experiments/W019-echo-crispr-array/scripts/triple_design_rank_sampling.py`
(`results/triple_design_rank_sampling_*.csv`, `results/triple_build_construction_check.csv`).

Supersedes the 2026.08.16 `balanced` list (36 strains), which put 15 of 21 triples on
genes the model has no training data for. See "Why capped" below.

### The finding that drives the design

The model was trained on **trigenic interactions only**. `queries/001_small_build.cql` has
every dataset block commented out except `TmiKuzmin2018Dataset` and `TmiKuzmin2020Dataset`
-- no SMF, no DMF, no DMI, no essentiality. Kuzmin record counts for the panel, counted
per gene over the LMDBs (`torchcell/datasets` builds under `$DATA_ROOT/data/torchcell/`):

**Only the trigenic column is training data.** It comes from `tmi_kuzmin2018` (91,111
records) + `tmi_kuzmin2020` (301,798), the two datasets the query pulls; every record in
both is a 3-KO. The digenic column comes from `dmi_kuzmin2018` + `dmi_kuzmin2020`
(1,043,196 records, all 2-KO) and is shown only to establish that a gene's absence is
absence from Kuzmin altogether, not an artifact of the trigenic subset. Those digenic
records were **not** in this model's training set.

| gene | trigenic | digenic | verdict |
|---|---|---|---|
| YLL012W (YEH1) | 1129 | 1140 | well supported |
| YBR203W (COS111) | 526 | 1147 | well supported |
| YPL046C (ELC1) | 493 | 1133 | well supported |
| YJR060W (CBF1) | 319 | 828 | well supported |
| YGL087C (MMS2) | 319 | 828 | well supported |
| YPL081W (RPS9A) | 10 | 21 | thin |
| YDR057W (YOS9) | 10 | 21 | thin |
| YLR104W (LCL2) | 9 | 20 | thin |
| YKL033W-A | 8 | 19 | thin |
| **YER079W** | **0** | **0** | **none -- extrapolation** |
| **YLR312C-B** | **0** | **0** | **none -- extrapolation** |

Both zero-data genes *do* have Costanzo data, and it was simply never in this model's
training set. `YER079W` has 4 SMF records over 2 distinct strains and **5,232** digenic
partners, 386 of them significant; `YLR312C-B` has 2 SMF records over 1 strain and
**3,544** partners, 99 significant. An earlier version of this line attached "~3,544
digenic partners" to both genes; it belongs to `YLR312C-B` alone and is exact, not
approximate. The "4 SMF" is not a distinguishing figure either, since every panel gene
except `YLR312C-B` has exactly 4. And node embeddings
are learned, not pre-computed, so those two genes carry embeddings that never received a
gradient from a labeled example containing them.

**They then take ranks 1--11 of the 39 targets**, the entire top of the ranking; the first
clean target is rank 12. Predicted interaction tracks *absence* of training data: mean
0.4531 with no zero-data gene, 0.5300 with one, 0.6676 with two (Spearman 0.634, p =
1.5e-05). The model is most confident exactly where it has no evidence.

### Model calibration -- corrected 2026.08.24

The val $r$ / val RMSE pairs below are the **last-epoch** val metrics in each run's W&B
summary (epoch 63, 63, 48). They are not the metrics of the checkpoint that produced the
predictions. Every prediction in this note comes from one checkpoint, `c7671wgj`
best-pearson epoch 24 (`.../c7671wgj-best-pearson-epoch=24-val/gene_interaction/Pearson=0.4619.ckpt`),
re-evaluated on the val split by slurm job 761
(`experiments/010-kuzmin-tmi/slurm/output/010-eval-m02_761.out`).

| ckpt | run | val $r$ last epoch | val RMSE last epoch | val $r$ ckpt used | val RMSE ckpt used |
|---|---|---|---|---|---|
| M01 | `lzs9pcj3` | 0.4428 | 0.0580 | 0.4520 | 0.0570 |
| M02 | `yv4r30bi` | 0.4312 | 0.0591 | 0.4472 | 0.0571 |
| M03 | `c7671wgj` | 0.4572 | 0.0567 | 0.4619 | 0.0562 |

The label population is the 010 build `001-small-build`: 376,732 records, split
301,386 / 37,673 / 37,673 at seed 42. Over the val split $\sigma_y=0.0629$, mean
$-0.0091$; over the whole build $\sigma_y=0.0633$, mean $-0.0080$, which is exactly what
all three runs logged as `normalization/gene_interaction/std` and `.../mean`. An earlier
version of this note used $\sigma_y=0.0535$, mean $-0.048$. Those are the statistics of
`tmi_kuzmin2018` alone (n = 91,111), 23% of the training records, so every calibration
number derived from them was wrong.

The exact identity is
$\mathrm{MSE}=(\mu_p-\mu_y)^2+\sigma_y^2+\sigma_p^2-2r\sigma_y\sigma_p$. With the correct
$\sigma_y$ it does **not** determine $\sigma_{pred}$: for the checkpoint used it admits
$\sigma_{pred}\in[0.0220,0.0361]$, a calibration slope
$a=r\sigma_y/\sigma_{pred}\in[0.805,1.321]$, and a mean bias bounded by
$|\mu_p-\mu_y|\le0.0071$. The model is therefore **under-dispersed**, not over-dispersed,
and the admissible transform runs from a mild shrink to a mild expansion. The previously
stated 2.3x over-dispersion and 0.43 slope lie outside that interval. Under every
admissible calibration the raw 0.4092--0.7114 in-basis predictions land in 0.33--0.94, so
all 39 clear Kuzmin's intermediate (0.08) and stringent (0.12) magnitude cuts, not 15 of 39
as previously stated.

**The "very stringent 0.20" cut does not exist in the form it was used here.** Kuzmin's SI
gives $|\text{score}|>0.08$ and $|\text{score}|>0.12$ as magnitude thresholds (Fig. S5B),
but 0.2 appears only as a **negative** cut-off, $\tau<-0.2$ (Fig. S15). There is no
magnitude form of it in the SI or the paper. Every prediction here is positive, so the
earlier "above very stringent for 15 of 39" applied a negative-tail threshold to
positive-tail values and should not be restated in either direction.

Measured directly on the full inference output for that checkpoint (465,735,532 rows,
`$DATA_ROOT/data/torchcell/experiments/010-kuzmin-tmi/inference_3/inferred/...Pearson=0.4619.parquet`),
the prediction distribution has mean $-0.0164$ and SD $0.0232$. The SD is 0.368 of
$\sigma_y$, which is the under-dispersion measured rather than inferred. The mean sits
$0.11\,\sigma_y$ below the val label mean, but that is the inference population, not val,
so it is not the val bias; the val bias stays bounded by the 0.0071 above.

**The calibration is being read far outside the range it was fitted on.** Against a
prediction SD of 0.0232, the 39 in-basis targets sit **18.4 to 31.4 SD above the prediction
mean**. No slope estimated on the bulk of the val
distribution constrains a point that far into the tail, in either direction. That, rather
than the value of the slope, is the reason a calibrated number should not be quoted for
these targets. It compounds with the separate problem that `YER079W` and `YLR312C-B` are
outside the training distribution entirely.

W&B group `compute-3-3-2059267_f83395661f9a37aec8c2fbb4b577c1f5bcca6e00e6f64f2772f606e02c6de5e3`
(rank-0 run `vjfp4d83`) is a **failed** run: val $r=-0.0314$, train $r=0.0007$, val RMSE
0.0629, equal to $\sigma_y$. It is not the source of these predictions.

### What blocks the round -- our error bar, and how far it has to move

$\tau_{abc}=f_{abc}-f_{ab}f_c-f_{ac}f_b-f_{bc}f_a+2f_af_bf_c$ carries seven measured
terms, and they are multiplied rather than summed, so each term's SE enters with a
coefficient set by the other fitnesses: $\partial\tau/\partial f_{ab}=-f_c$ and
$\partial\tau/\partial f_a=2f_bf_c-f_{bc}$. All seven coefficients are $\pm1$ only when
every fitness equals 1, which is the single point where $\mathrm{SE}(\tau)=\sqrt{7}\,s$.
Two multipliers matter and they are different quantities. **Measured**, over the four
run-4 gene triples whose three doubles and three singles were all measured:
**2.98--3.94, mean 3.44** (`run4_tau_multiplier.csv`). **Expected for a well-behaved
deletion panel** at a typical fitness $f=0.85$ under a multiplicative null: **2.18**. The
measured value is the larger because the run-4 singles score above the on-plate WT and the
doubles sit well below the product of their singles. Everything below uses the measured
3.44. The seven terms also do not share one SE: run-4 singles carry 2.0x the bootstrap SE
of the doubles, and singles carry the largest coefficients.

**The run-4 fitness scale is itself suspect, which is why the measured multiplier is the
larger one.** All three run-4 plates fail the frac-above-WT gate that
`between_day_variance.py` applies: 0.40 / 0.32 / 0.24 of deletions score above WT against
a limit of 0.20. `run4_plate_qc.csv` passes all three, but on WT CV alone. This is the same
wild-type reference problem the run-4 handoff records, and it means the fitness LEVELS,
and therefore the multiplier, are less trustworthy than the SPREADS the variance
components below are built from.

Per-strain precision, re-derived from run 4 (25 strains x 3 plates, 75 strain-plate cells,
`run4_strain_scores_by_plate.csv`): colony SD $\sigma_{colony}=0.185$, plate SD
$\sigma_{plate}=0.131$ once the colony share is removed from the across-plate spread, and
day SD $\sigma_{day}=0.020$ (`between_day_variance.csv`, 1 degree of freedom, indicative).

$$s(P,c)=\sqrt{\sigma_{day}^2+\sigma_{plate}^2/P+\sigma_{colony}^2/(Pc)}$$

**The structure this is measured on is ONE pick per strain.** Every strain in runs 2 to 4
was plated from a single picked colony, so the colony-pick term is not estimable and the
formula sets it to zero. At one pick and this round's layout of 5 wells per strain,
$s=0.092$ at 3 plates and $0.066$ at 6, and both are LOWER bounds on the honest error.
Any two-pick figure is conditional on an unmeasured $\sigma_{pick}$ and is a hypothesis
until the pick-replication round measures it, so none is quoted here. Measured
cross-checks at one pick: the run-4 bootstrap SE over 3 plates has rms 0.067 across 25
strains, and the committed run-3 benchmark puts the median honest per-strain SE at 0.057
(`run3_precision_benchmark.csv`, n = 12).

**Callable means $|\tau|>1.96\,\mathrm{SE}(\tau)$, the two-sided 95% convention.** At the
measured multiplier the smallest callable value is **0.617 at 3 plates and 0.446 at 6
plates** (`run4_tau_precision.csv`). Against the raw predictions, **10 of 39** in-basis
targets sit above the 3-plate floor and **25 of 39** above the 6-plate floor. Calibrated
at the derived slope $a=r\,\sigma_y/\sigma_{pred}=1.254$ those counts become 18 and 39,
but that pair is calibration-dependent: these targets sit 18 to 31 prediction-SDs above
the prediction mean, so the calibration is being read far outside the range it was fitted
on and no calibrated point estimate is defensible. Sweeping the slope alone across the
interval the calibration identity admits, 0.805 to 1.321, the count above the 3-plate
floor runs 0 to 22 and above the 6-plate floor 12 to 39. What survives every reading is
that callability turns on plate count. Three plates leaves most targets out of reach; six
plates brings most or all of them inside. An earlier version of this section concluded
that none of the 39 was callable. That rested on $\sqrt{7}$ in place of the measured
multiplier and on a label SD since shown to be wrong, and it is not supported.

**Computing tau per plate and averaging, rather than from strain means, changes little.**
The plate-common offset is shared by all seven terms within a plate, so it enters with the
sum of the tau coefficients rather than their root sum of squares, and it survives WT
normalization at only 15% of the plate variance. The per-plate floors are 0.598 and 0.433
against 0.617 and 0.446, a 3% gap, so the choice of route is closed rather than an open
modeling question. The strain-mean route is the default here, on the argument in
`next_round_layout.epsilon_se` that the common plate factor is already removed by
normalizing to on-plate wild type; every number quoted above uses it. The 3% does still
move the 6-plate raw count from 25 to 31, because the predictions cluster near that floor.

Kuzmin's own median $\mathrm{SE}(\tau)$, back-solved from $\tau$ and $p$ in
`tmi_kuzmin2018/2020`, is **0.0785**. Their released $p$ column is one-sided; reading it
as two-sided gave 0.031 and understated their error bar by 2.5x. It is an SE on tau, not
on a strain. Ours is $\mathrm{SE}(\tau)=0.31$ at 3 plates and 0.23 at 6, so we sit at
4.0x and 2.9x their error. Reaching per-strain SE 0.04 takes about **20 plates**, 6.6x the
current 3, and leaves $\mathrm{SE}(\tau)=0.138$, 1.75x theirs. Matching them outright
needs per-strain SE 0.023, which sits above the one-day $\sigma_{day}$ floor and is
therefore reachable within a single day, at about 175 plates, **58x** the current 3. This
is a pick and plate problem, not a model problem.

### Why capped

All six strategies at their natural size:

| | tri | new dbl | **construct** | measure | wells | gene range | YLR312C-B | zero-data tri | one wave | mean pred |
|---|---|---|---|---|---|---|---|---|---|---|
| rank | 20 | 20 | 40 | 62 | 5 | 3--13 | 10 | 17 | yes | 0.6005 |
| count | 20 | 15 | 35 | 56 | 6 | 0--11 | 8 | 15 | yes | 0.5221 |
| balanced | 21 | 15 | 36 | 57 | 6 | 0--10 | 9 | 15 | yes | 0.5329 |
| uniform | 20 | 22 | 42 | 65 | 5 | 4--7 | 7 | 11 | yes | 0.5403 |
| **capped** | **20** | **25** | **45** | **65** | 5 | 2--8 | **6** | **6** | yes | 0.5300 |
| no_ylr | 20 | 20 | 40 | 59 | 6 | 0--11 | 0 | 9 | **no** | 0.4743 |

`no_ylr` removes YLR312C-B only. It still places 9 of its 20 triples on YER079W, so it is
not a zero-extrapolation design, and it is the one strategy that selects a triple with no
built parent (rank 26), so its 40 strains do not build in a single wave. The `zero-data
tri` column previously read `0` for `no_ylr` and `--` for `count` and `uniform`, which made
the cheapest design look like the safest one.

Capped is the only design that holds zero-data exposure to a chosen level -- **6 of 20**,
against 15 for `balanced` and 17 for `rank` -- while still covering all eleven genes. It is
the most expensive of the 20-triple designs (45 strains vs 40), because constraining the
flagged genes forces it down the ranking. That is the trade being bought: fewer results
resting on extrapolation.

**Two constraints shape the size.** Only **16 of 39** targets are clean, and only **14 of
those 16 have an existing double to build from** -- ranks 32 and 37 have none. Three
targets in all have no built parent, ranks 26, 32 and 37; rank 26 is flagged rather than
clean. All three contain YLR104W (LCL2), which has zero built doubles because it sat
outside the TEN gene set the original set-cover ran on. Cap 6 is therefore the smallest cap
that reaches 20 parallel triples: 6 flagged plus all 14 constructible clean targets. A cap
of 5 tops out at 19.

The original set-cover guarantee is intact: **31 of 31** within-TEN targets have a built
parent. The three that do not are among the 8 targets YLR104W added when the basis was
corrected to eleven genes.

![Which triples each design samples](assets/images/W019-echo-crispr-array/triple_design_rank_sampling.png)

![Gene participation per design](assets/images/W019-echo-crispr-array/triple_design_gene_frequency.png)

### The 20 triples

`*` = involves YER079W or YLR312C-B.

| rank | pred | triple | new dbl | |
|---|---|---|---|---|
| 1 | 0.7114 | YKL033W-A + YLR312C-B + YPL081W | 2 | * |
| 2 | 0.7109 | YER079W + YLL012W + YLR312C-B | 2 | * |
| 3 | 0.7012 | YLR104W + YLR312C-B + YPL081W | 2 | * |
| 4 | 0.6987 | YBR203W + YLR312C-B + YPL081W | 2 | * |
| 5 | 0.6841 | YER079W + YLR104W + YLR312C-B | 2 | * |
| 6 | 0.6831 | YGL087C + YLR312C-B + YPL081W | 2 | * |
| 12 | 0.5679 | YBR203W + YDR057W + YPL081W | 2 |  |
| 13 | 0.5415 | YDR057W + YKL033W-A + YPL081W | 2 |  |
| 16 | 0.5195 | YDR057W + YGL087C + YPL081W | 1 |  |
| 21 | 0.4644 | YJR060W + YLL012W + YPL046C | 1 |  |
| 24 | 0.4529 | YBR203W + YLL012W + YPL046C | 1 |  |
| 25 | 0.4492 | YDR057W + YKL033W-A + YLL012W | 2 |  |
| 27 | 0.4436 | YGL087C + YLL012W + YPL046C | 2 |  |
| 28 | 0.4385 | YBR203W + YLR104W + YPL046C | 2 |  |
| 29 | 0.4370 | YBR203W + YKL033W-A + YPL046C | 2 |  |
| 31 | 0.4331 | YBR203W + YJR060W + YPL046C | 1 |  |
| 33 | 0.4226 | YJR060W + YLR104W + YPL046C | 2 |  |
| 34 | 0.4204 | YBR203W + YGL087C + YPL046C | 2 |  |
| 38 | 0.4109 | YBR203W + YDR057W + YLL012W | 2 |  |
| 39 | 0.4092 | YDR057W + YGL087C + YLL012W | 1 |  |

### A strong digenic sits inside rank 5

`YER079W x YLR104W` is the largest-magnitude significant digenic interaction anywhere in
the eleven-gene basis: Costanzo 2016 scores it at $\varepsilon=-0.5672$, $P=1.832\text{e-}04$,
double-mutant fitness 0.5050 against an expected $1.0387\times1.0322=1.0722$. Significance
is Costanzo's own intermediate threshold, $P<0.05$ and $|\varepsilon|>0.08$, applied per
directional screen over all 104 directional rows covering 53 of the 55 in-panel pairs
(`results/published_source_inpanel_digenics.csv`). The next largest significant in-panel
$|\varepsilon|$ is 0.1295, so this one is 4.4x the runner-up. Only three other in-panel
pairs are significant: `YDR057W x YGL087C` $(+0.0979, P=9.89\text{e-}04)$,
`YER079W x YPL081W` $(-0.1295, P=0.0352)$ and `YJR060W x YKL033W-A` $(-0.0820,
P=6.99\text{e-}03)$, the last being the pair that failed to construct.

**The reciprocal screen does not reproduce it.** Costanzo stores query-to-array and
array-to-query separately, and the two rows disagree by a factor of 26: query
`YER079W_sn3579` x array `YLR104W_dma3202` gives $\varepsilon=-0.5672$, $P=1.832\text{e-}04$
with a double-mutant fitness SD of 0.3249, while query `YLR104W_sn3294` x array
`YER079W_dma1369` gives $\varepsilon=-0.0217$, $P=0.3636$ with an SD of 0.0611. Both are
negative, so under the reciprocal rule of the SGA scoring reference the lower-$P$ row is
the one retained, which is the $-0.5672$. But that row's own SD is 0.32 on a fitness of
0.505, so the call rests on the noisier of the two directions. Treat the magnitude as
unsettled and the sign as the reliable part.

**What it means for rank 5.** Rank 5 is `YER079W + YLR104W + YLR312C-B`, one of the two
selected triples containing both zero-trigenic-data genes. Two of its three genes already
interact strongly and negatively at the digenic level, which puts it in Kuzmin's
**modified** class rather than the novel one: the trigenic model subtracts
$\varepsilon_{i,k}f_j+\varepsilon_{j,k}f_i$ precisely so that a known digenic does not
present as a trigenic interaction, so a large $\tau$ here would have to survive removing a
$-0.57$ term. **This weakens the case for rank 5**, on three separate grounds: the
interaction it would report is not novel, the double is predicted to grow at about half
wild-type so the triple lands near the low-fitness end where our assay noise is worst, and
none of it is visible in the model's inputs, since the training query pulls trigenic
records only and `YER079W` has zero Kuzmin digenic records as well. It does not make rank 5
unbuildable, and the double `YER079W + YLR104W` is a worthwhile measurement in its own
right as a check on the two disagreeing Costanzo directions. It does mean rank 5 should not
be read as a clean trigenic prediction.

### The 25 new doubles

| double | serves triples (rank) | |
|---|---|---|
| YBR203W + YDR057W | 12, 38 |  |
| YBR203W + YLL012W | 24, 38 |  |
| YBR203W + YPL081W | 4, 12 |  |
| YDR057W + YKL033W-A | 13, 25 |  |
| YGL087C + YLL012W | 27, 39 |  |
| YGL087C + YPL046C | 27, 34 |  |
| YGL087C + YPL081W | 6, 16 |  |
| YKL033W-A + YPL081W | 1, 13 |  |
| YLR104W + YLR312C-B | 3, 5 | * |
| YLR104W + YPL046C | 28, 33 |  |
| YBR203W + YGL087C | 34 |  |
| YBR203W + YJR060W | 31 |  |
| YBR203W + YKL033W-A | 29 |  |
| YBR203W + YLR104W | 28 |  |
| YBR203W + YLR312C-B | 4 | * |
| YER079W + YLL012W | 2 | * |
| YER079W + YLR104W | 5 | * |
| YGL087C + YLR312C-B | 6 | * |
| YJR060W + YLL012W | 21 |  |
| YJR060W + YLR104W | 33 |  |
| YKL033W-A + YLL012W | 25 |  |
| YKL033W-A + YLR312C-B | 1 | * |
| YKL033W-A + YPL046C | 29 |  |
| YLL012W + YLR312C-B | 2 | * |
| YLR104W + YPL081W | 3 |  |

### Gene participation

YBR203W 8 · YPL046C 8 · YLL012W 7 · YPL081W 7 · YDR057W 6 · YLR312C-B 6 · YGL087C 5 · YLR104W 4 · YKL033W-A 4 · YJR060W 3 · YER079W 2

All eleven genes covered, range 2--8. Contrast `balanced`, which left two genes at zero
and put 10 triples on YER079W.

### Plate

| | count | note |
|---|---|---|
| singles | 11 | all built |
| doubles | 33 | 8 existing + **25 new** |
| triples | 20 | **all new** |
| WT (BY4741) | 1 | |
| **on plate** | **65** | 5 wells/strain at one 384 layout |
| **to construct** | **45** | 25 doubles + 20 triples |

65 tubes at one pick, 130 at two. The 5 wells/strain figure covers the 64 non-WT strains;
with the 6 orientation blanks that is 326 of 384 wells, leaving 58 for WT. Two
destination-plate designs restore ~11 wells/strain and add the 6-plate replication gain.

**One starter culture per strain.** At one pick, K = 1 for all 65 strains, so the
`sigma_culture` floor described in `next_round_layout.py` applies and stays unmeasurable.
Run 4's epsilon was unreportable because its WT denominator rested on a single colony, so
the WT well count is the number to settle before ordering.

### Construction feasibility -- verified

| check | result |
|---|---|
| 11 required singles built | **yes** |
| 25 new doubles, both parent singles built | **25 / 25** |
| any new double = blocked `YKL033W-A x YJR060W` | **no** |
| triples with zero already-built parent | **none** |

**All 20 build in one parallel wave.** Targets with no existing parent double are excluded
from the `capped` selection by construction. **15 of 20 have only one parent route**
(T01--T15) -- start those first.

### The contingency

If YER079W and YLR312C-B are dropped, **14 triples survive and 17 of the 25 new doubles are
still needed** -- a **31-strain** core on a **47-strain plate** (9 singles, 23 doubles, 14
triples, WT). The six flagged triples are the top of the ranking
and the seven flagged doubles feed only those, so the flagged portion is cleanly separable
and the core can start immediately.

The one loose end: 18 of the 25 new doubles contain neither flagged gene, but
`YLR104W + YPL081W` serves only rank 3 (`YLR104W + YLR312C-B + YPL081W`). Dropping the two
genes leaves it feeding no surviving tau, so 32 strains are clean by composition and 31 are
needed. It remains a usable digenic measurement either way.

### Excluded

`YKL033W-A x YJR060W` failed to construct (reported at the 2026-08-11 handover; attempt
date never recorded) and is excluded everywhere. It blocks **0** of the 39 targets.

### Related

- [[experiments.W019-echo-crispr-array.build-list]]
- [[experiments.W019-echo-crispr-array.run4-handoff]]
- [[experiments.010-kuzmin-tmi.scripts.construction_validation_doubles]]
- [[experiments.010-kuzmin-tmi.inference-dataset-3]]
