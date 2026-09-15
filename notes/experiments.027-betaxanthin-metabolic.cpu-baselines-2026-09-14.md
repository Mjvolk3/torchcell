---
id: r2yx33xe9ci6new91yhk096
title: Cpu Baselines 2026 09 14
desc: ''
updated: 1789373101532
created: 1789373101532
---

## 2026.09.14 - CPU baselines for the 027 metabolic targets

Four CPU experiments from the metabolic review's E1 to E4, all run under slurm on GilaHyper
while the database rebuild held the machine, all niced to 1000 with at most 8 cores and 24G.
Every number below names the job that produced it and the result file it was read from.

Scripts, all new except the last:

- `experiments/027-betaxanthin-metabolic/scripts/cbm_baseline_bx.py` (+ `cbm_scoring.py`)
- `experiments/027-betaxanthin-metabolic/scripts/linear_baselines_bx.py`
- `experiments/027-betaxanthin-metabolic/scripts/score_fcl_027.py`
- `experiments/026-metabolism-flux/scripts/fva_reference.py` (given `--media` / `--cassette`)

### The headline

**A parameter-free nearest-neighbor baseline on gene embeddings beats every 027 arm on the
betaxanthin endpoint, and the constraint-based baseline carries no rank information at all.**

| baseline | pinned test Spearman | n | evidence |
|---|---:|---:|---|
| B3 cosine kNN, `prot_T5_all`, k = 25 | **+0.1446** | 629 | perm-p < 0.0005, boot CI [+0.061, +0.224], job 1909 |
| B3 cosine kNN, `calm`, k = 25 | +0.1504 | 629 | perm-p < 0.0005, CI [+0.072, +0.228], job 1909 |
| CGT `pooled`, 200 epochs, 3 seeds | +0.1148 sd 0.0135 | 629 | wave-7 dumps, review section 5b |
| B2 ridge, `calm` | +0.0881 | 629 | perm-p 0.033, CI [+0.010, +0.164], job 1909 |
| FBA, best readout (`btx_at_90pct_reference_growth`, YPD, no markers) | +0.0468 | 598 | perm-p 0.262, CI [-0.033, +0.127], job 1901 |
| FCL RandomForest (Merzbacher 2025) | +0.0391 | 629 | `conf/base.yaml:36` |
| B0 train mean | 0 by construction | 629 | per-feature Pearson +0.000000000, NMSE 1.0000, job 1909 |

`prot_T5_all` is the cell validation selects for B3 (val per-feature Pearson +0.2483 against
`calm`'s +0.2302), so +0.1446 is the reported number and +0.1504 is the larger one the
selection rule did not pick. On top-50 precision the same cell reads **0.50** against a base
rate of 0.157, which is 3.2x chance and above the CGT pooled arm's 0.433 sd 0.037.

### E1. Constraint-based baseline (jobs 1901 betaxanthin, 1902 amino acids, 1941 isobutanol)

Yeast-GEM 9.0.2, GLPK via optlang, cobra 0.30.0. Wild-type growth on YPD-approx with the
Cachera cassette reproduces at **0.535202 /h**. Results in
`experiments/027-betaxanthin-metabolic/results/cbm_baseline_{betaxanthin,amino_acids,isobutanol}.json`.

**The strain background is in the genotype.** Cachera's screen is the BY4741 knockout
collection, so every strain carries `his3D1 leu2D0 met15D0 ura3D0` before the screened gene
is deleted. All four markers resolve in the GEM (HIS3 YOR202W, LEU2 YCL018W, MET15/MET17
YLR303W, URA3 YEL021W) and are applied through cobra's GPR-aware knockout.

**A medium finding that forced a second condition.** Cachera grew the screen on YPD + G418;
`torchcell/datasets/scerevisiae/cachera2023.py:321` emits `Media(name="SC", state="solid")`,
which does not match the paper. The GEM stand-in for YPD (`YPD_APPROX_FBA`, Suthers' "YNB
plus 20 amino acids") carries no uracil and no adenine, because yeast extract is never
modeled. MEASURED (job 1901): under that bound vector the `ura3D0` marker alone takes growth
from 0.535202 to **0.000000 /h**, so the entire BY4741 background is dead in silico while the
bench strain grows. `SC_FBA` is the same bound vector plus those two nucleobases and gives a
background growth of **0.484628 /h**. Both media are run and both are reported; `sc_markers`
is the primary condition because it is the only one whose background strain is alive.

Betaxanthin, 628 of the 629 pinned test genes with a finite measurement are in the GEM
(job 1901, 20 min wall, 1.50 s/genotype on the primary condition):

| condition | readout | rho | perm-p | boot 95 % CI | ties | top-10/25/50 |
|---|---|---:|---:|---|---:|---|
| ypd_none | KO growth | +0.0258 | 0.495 | [-0.055, +0.107] | 0.876 | 0.20/0.20/0.16 |
| ypd_none | btx at 90 % of reference growth | +0.0468 | 0.262 | [-0.033, +0.127] | 0.921 | 0.20/0.16/0.14 |
| ypd_none | btx at 90 % of own growth | -0.0840 | 0.035 | [-0.162, -0.005] | 0.869 | 0.10/0.04/0.06 |
| ypd_none | shadow price L-tyrosine (s_1051) | -0.0052 | 0.903 | [-0.082, +0.075] | 0.879 | 0.10/0.24/0.28 |
| sc_markers | KO growth | -0.0673 | 0.094 | [-0.152, +0.016] | 0.959 | 0.20/0.12/0.12 |
| sc_markers | btx at 90 % of reference growth | -0.0600 | 0.149 | [-0.143, +0.021] | 0.927 | 0.20/0.16/0.14 |
| sc_markers | btx at 90 % of own growth | +0.0230 | 0.569 | [-0.061, +0.107] | 0.889 | 0.20/0.16/0.14 |
| sc_markers | pFBA total flux | -0.0500 | 0.211 | [-0.130, +0.032] | 0.908 | 0.40/0.28/0.16 |
| sc_markers | FVA width, sum over the 20 `DM_btx_*` | +0.0702 | 0.078 | [-0.010, +0.149] | 0.855 | 0.10/0.16/0.14 |
| sc_markers | FVA width, max over the demands | +0.0354 | 0.379 | [-0.045, +0.118] | 0.897 | 0.10/0.16/0.10 |

The base rate is 0.156 to 0.157 in every row. The `ypd_markers` condition is reported in the
JSON and is degenerate on every growth-derived readout, as the dead background requires.

**The E1 kill rule fires.** Every readout's 95 % CI spans zero except `btx at 90 % of own
growth` on `ypd_none`, whose tie fraction is 0.869; every other readout ties above 0.85 and
most above 0.9. The one CI that excludes zero points the wrong way (-0.084): the deletions
the model says have the most headroom measured lowest. That is one uncorrected test out of
ten, it is on the non-primary medium, and it replicates the review's prototype exactly
(-0.0840, perm-p 0.035), so it is worth naming and not worth explaining. **The constraint-based
baseline is a measured null on this screen. Do not tune the growth fraction, the medium or the
objective in search of a positive; that is a garden of forking paths on 628 genes.**

Amino acids (job 1902, 15 min wall): Mulleder's prototrophic collection on synthetic minimal
medium (`FBA_MEDIA["SM"]`, wild-type growth 0.313781 /h), no marker deletions, readout = the
maximum drain of each of the 19 measured amino acids at 90 % of reference growth. Mean
per-feature Spearman **+0.0036** over 19 columns, n = 540 per column, tie fraction 0.950 to
0.957 in every column. Two columns reach p < 0.05 uncorrected and in opposite directions
(leucine +0.0962, phenylalanine -0.0893), which is what 19 tests at alpha 0.05 produce. The
kill rule fires here too.

Isobutanol: BY4741 `ura3D0` with the Leu3p reporter at HIS3, on SC, objective `r_1866` (the
isobutanol exchange yeast-GEM 9.0.2 already carries, so no cassette is needed), scored against
log10 of the released fold change on 581 pinned genes, all of them in the GEM.

**Read from job 1941, which reproduced job 1903 exactly.** Job 1903 completed the primary
`sc_markers` condition in 8.3 min, then stalled inside the marker-free `sc_none` condition and
was killed at the 3 h wall before writing its JSON. Job 1941 reran the primary condition alone
(`--conditions sc_markers`, 534 s wall, 0.920 s/genotype) and every readout below matches job
1903's log to four decimal places, which is what a deterministic LP with a fixed permutation
seed should do. Wild-type growth 0.542782 /h, background 0.484628 /h. The Lopez LMDB read is
the copy job 1894 rebuilt, mtime 2026-09-14 02:29:03, 4,554 entries and 4,554 screen genes.

| readout | rho | perm-p | boot 95 % CI | ties | top-50 |
|---|---:|---:|---|---:|---:|
| KO growth | -0.0150 | 0.721 | [-0.102, +0.067] | 0.957 | 0.16 |
| isobutanol at 90 % of reference growth | +0.0680 | 0.110 | [-0.014, +0.153] | 0.932 | 0.16 |
| isobutanol at 90 % of own growth | +0.0784 | 0.056 | [-0.007, +0.162] | 0.892 | 0.22 |
| 2-isopropylmalate, cytosolic (s_0162) | +0.0738 | 0.080 | [-0.011, +0.156] | 0.930 | 0.28 |
| 2-isopropylmalate, mitochondrial (s_0164) | **+0.1201** | **0.005** | **[+0.039, +0.198]** | 0.932 | 0.24 |
| pFBA total flux | +0.0084 | 0.840 | [-0.074, +0.094] | 0.916 | 0.14 |

Base rate 0.157.

The mitochondrial 2-isopropylmalate capacity is the only constraint-based readout in this
whole set whose CI excludes zero on its primary condition. It does **not** meet E1's success
rule, which requires the CI to exclude zero **with a tie fraction below 0.9**, and 0.932 of
genotypes sit at the modal value. It is also the readout the biosensor's chemistry points at:
the Leu3p reporter responds to alpha-ketoisovalerate through 2-isopropylmalate, not to
isobutanol. **Hypothesis (untested):** the mitochondrial compartment is where the biosensor's
ligand pool is set, and a replication on a pre-registered second gene set would decide it. It
is one test among six on this target and is not a result yet.

### E2. The FVA reference was computed on the wrong medium (job 1905, 4 min)

`fva_reference.py` optimized `YeastGEM().model` with no medium and no cassette. Result in
`experiments/026-metabolism-flux/results/fva_census_by_condition.json`.

**Control passed:** the `default` condition reproduces the shipped
`026-metabolism-flux/results/fva_census.json` on all 8 fields, so the differences below are
the flag's and not a drifted model's.

| condition | wild-type growth /h | reactions | FVA width <= 1 | median width |
|---|---:|---:|---:|---:|
| default (SBML bounds, no cassette) | 0.085844 | 4131 | 2818 | 0.1176 |
| YPD-approx, no cassette | 0.535202 | 4131 | 2178 (-22.7 %) | 0.6166 |
| YPD-approx + cassette | 0.535202 | 4177 | 2224 (-21.1 %) | 0.5729 |

**E2 success rule fires:** the licensed set differs by 21 to 23 %, far beyond the 5 %
threshold. Every `width_FVA - width_posterior` number on record was computed against a medium
and a model the 027 cells do not live in, and has to be restated against the YPD + cassette
reference. `[[experiments.026-metabolism-flux.scripts.fva_reference]]` needs the correction.

### E3. Linear baselines on the 027 pinned split (jobs 1909, 1911, 1912)

019's B0 to B3 ladder over four embedding sets, fit-split standardization, 019's `ok` mask,
validation-selected hyperparameters, test reported on the pinned genes. Results in
`experiments/027-betaxanthin-metabolic/results/linear_baselines_{betaxanthin,amino_acids,isobutanol}.json`.

**Two controls passed on every target.** B0, the train mean, scores per-feature Pearson
+0.000000000 with NMSE 1.0000 on both val and test. And the pinned test targets this script
reads from the LMDB match the wave-7 dumps exactly, on 629 genes for betaxanthin and 564 for
the amino acids, so baseline and model are scoring the same numbers on the same gene set.

Betaxanthin (job 1909, 34 s; train 3,494 / val 270 / test 905, of which 629 pinned):

| model | embedding (val-selected) | setting | val pf | test-pinned rho | perm-p | boot CI | top-50 |
|---|---|---|---:|---:|---:|---|---:|
| B3 kNN | **prot_T5_all** | k = 25 | +0.2483 | **+0.1446** | < 0.0005 | [+0.061, +0.224] | **0.50** |
| B3 kNN | calm | k = 25 | +0.2302 | +0.1504 | < 0.0005 | [+0.072, +0.228] | 0.32 |
| B3 kNN | normalized_chrom_pathways | k = 25 | +0.0221 | +0.1253 | 0.002 | [+0.044, +0.203] | 0.24 |
| B3 kNN | species_lm_5p_3p | k = 25 | +0.1072 | +0.0444 | 0.257 | [-0.031, +0.123] | 0.18 |
| B2 ridge | **calm** | ridge 1e-3 | +0.1766 | +0.0881 | 0.033 | [+0.010, +0.164] | 0.24 |
| B2 ridge | prot_T5_all | ridge 100 | +0.1448 | +0.0822 | 0.036 | [+0.003, +0.159] | 0.34 |
| B2 ridge | normalized_chrom_pathways | ridge 1e-3 | +0.1170 | +0.1536 | < 0.0005 | [+0.079, +0.230] | 0.28 |

**E3 success rule fires, and it reframes the endpoint.** Six of eight val-selected linear
cells have a CI excluding zero, and the best of them exceeds the CGT pooled arm. The bar the
metabolic module has to clear is set by **gene identity**, not by flux: a cosine average over
the 25 nearest deleted genes in protein-language-model space, with no fitted parameters at
all, orders the Cachera screen better than a 200-epoch transformer with a stoichiometric
layer. The paper's baseline table has to carry this row.

Amino acids (job 1911, 5 min; 564 pinned genotypes, mean per-feature Spearman over 19
columns): B2 on `calm` at rank 19 reads **+0.0450** and B3 on `prot_T5_all` at k = 10 reads
**+0.0249**, against the CGT `pooled` arm's 0.0397 sd 0.0124 and `flux_projection_anchored`'s
0.0553 sd 0.0027 (n = 3, review section 5b). The non-selected `normalized_chrom_pathways`
cells read higher (B2 +0.0655, B3 +0.0993) and were not selected on validation, so they are
noted and not reported as the result. On this target the linear ladder and the model sit in
the same band, which is a weaker statement than the betaxanthin one and is stated as such.

Isobutanol (job 1912, 34 s). The build the 027 arms train on carries no isobutanol column;
`fig6_build` does, and MEASURED, 4,244 of its records hold a width-20 `metabolite_level` (the
Mulleder 19 plus isobutanol, merged by the deletion-keyed aggregator) and 189 hold an
isobutanol-only record. Train 3,356 / val 300 / test 900, of which 581 pinned. B3 on
`normalized_chrom_pathways` at k = 25, the val-selected cell (val pf +0.2490), reads
**+0.1982** on the pinned test genes, perm-p < 0.0005, CI [+0.120, +0.274]. No 027 arm
predicts isobutanol, so this is a baseline with nothing beside it yet; it is what a future arm
would have to clear.

Note the two isobutanol scripts read the target on different scales: `cbm_baseline_bx.py`
takes log10 of the released fold change and `linear_baselines_bx.py` takes the stored fold
change. log10 is monotone, so every Spearman and every top-k number is identical; only the
NMSE and the Pearson differ, and B1's NMSE of 5.52 on val is a statement about the fold change
not being centered at zero.

### E4. The 027 dumps under Merzbacher's three-class protocol (job 1908, 9 s)

`score_fcl_027.py` adapts the dumps and imports 020's scoring functions untouched. Result in
`experiments/027-betaxanthin-metabolic/results/fcl_head_to_head_027.json`.

**Control passed before any 027 number was computed:** the 10 dumps that
`020-cachera-betaxanthin/results/merzbacher_head_to_head.json` names reproduce it exactly on
every aggregated metric. `$DATA_ROOT/test-predictions` has since gained four gilahyper dumps
that carry a betaxanthin head and no grid setting tag, which `setting_label` refuses by
design; they postdate the stored result and are excluded so the control compares one run set
to itself rather than two.

15 dumps, 639 genes scored per dump, mean +/- SEM over seeds, train-pool scale:

| arm | n | MCC | accuracy | high-producer recall | fraction called medium | top-50 |
|---|---:|---:|---:|---:|---:|---:|
| pooled, 200 ep | 3 | **0.157 +/- 0.034** | 0.681 +/- 0.006 | **0.123 +/- 0.037** | 0.934 +/- 0.010 | **0.433 +/- 0.037** |
| flux_projection_anchored, 200 ep | 3 | 0.095 +/- 0.057 | 0.681 +/- 0.005 | 0.057 +/- 0.035 | 0.989 +/- 0.006 | 0.353 +/- 0.067 |
| flux_capacity, 80 ep | 3 | 0.073 +/- 0.039 | 0.678 +/- 0.002 | 0.037 +/- 0.027 | 0.991 +/- 0.008 | 0.340 +/- 0.031 |
| flux_projection_anchored_fluxforce, 80 ep | 3 | 0.000 +/- 0.000 | 0.674 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.340 +/- 0.058 |
| flux_fluxforce_boxforce, 80 ep | 2 | 0.000 +/- 0.000 | 0.674 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.140 +/- 0.000 |
| FCL RandomForestClassifier_Resampled | - | 0.205 | 0.700 | 0.180 | 0.948 | not reported |

Rank-matched (predictions cut at the train pool's own class shares, so calibration is set
aside and only the ordering is charged): `pooled` MCC 0.096 +/- 0.012, accuracy 0.608 +/- 0.005,
high-producer recall 0.177 +/- 0.007, fraction called medium 0.790 in every arm by construction.

**E4 kill rule fires.** Every arm calls 93 to 100 % of genes medium, against FCL's 94.8 %. We
reproduced their failure mode rather than beat it, and no arm's high-producer recall comes near
0.295, Merzbacher's ceiling over all sixteen of their model-by-rebalancing cells. The three
flux arms that read exactly 0.000 on MCC called every single gene medium.

**A correction to the review's bar.** The review quotes FCL as "0.232 / 0.700 / 0.18 / 0.948",
but those numbers come from two different cells of their table. Read from
`020-cachera-betaxanthin/results/merzbacher_baseline_analysis.json`:
`RandomForestClassifier` has MCC 0.2321 with high-producer accuracy 0.1813, while
`RandomForestClassifier_Resampled`, the model whose gene-level 0.700 / 0.18 / 0.948 the review
quotes and which 020's protocol names `THEIR_BEST`, has MCC **0.2052**. The comparison above
uses 0.2052, which is the MCC of the model the other three numbers belong to.

### Open items

- **One LP in the isobutanol marker-free condition does not return.** Job 1903 completed
  `sc_markers` in 8.3 min and then stalled inside `sc_none` at genotype 300 of 581, emitting no
  further progress line before the 3 h wall. It is not the FVA stage: FVA runs only on the
  betaxanthin target and only on the primary condition, so the isobutanol `sc_none` loop
  contains nothing but the growth LP and three `_maximize` calls. The rerun (job 1941) is
  narrowed to the primary condition with `--conditions sc_markers`, so the marker-free control
  for isobutanol is missing. Diagnosing the stall needs the specific genotype, which the
  progress lines print only every 100, so the next attempt should log per genotype or shard the
  condition over gene blocks.
- **E8 is off the table for now.** The pooled-enzyme constraint was contingent on E1 producing
  a readout whose CI excludes zero on its primary condition with an honest tie fraction.
  It did not. Tightening a model that carries no signal is not a route to signal.
- **E1's contingency was already taken.** Per-demand FVA width was computed for the primary
  betaxanthin condition and reads +0.0702, perm-p 0.078, CI [-0.010, +0.149], tie fraction
  0.855, which is the lowest tie fraction of any betaxanthin readout and still not a result.
