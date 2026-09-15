---
id: bkne8woxztj5x399rxtn0e5
title: Wave9 Isobutanol
desc: ''
updated: 1789373250336
created: 1789373250336
---

## 2026.09.14 - Wave 9 design: isobutanol, the two missing controls, held-out columns, and the medium

Wave 9 adds one new TARGET and three ABLATIONS that the module's existing results cannot be
read without. It was specified by the targets-and-readouts review of 2026-09-14 (sections 3,
4 and 6, experiments E2, E3, E4 and E6) and implemented in
`experiments/027-betaxanthin-metabolic/scripts/train_bx.py` and
`experiments/026-metabolism-flux/scripts/train_flux.py`.

Everything below that is a number was measured today, with the script or the file it came
from named. Anything not measured is labeled as a hypothesis.

### The width-1 collision guard, and why it was not enough

026's `extract_targets` separates two heads that share a phenotype label by the WIDTH of
their value group, because the phenotype dict keys are dropped in collation. Two width-1
`metabolite_level` groups on one genotype therefore both match a width-1 head and are
averaged into one target by `torch.stack(matched).mean`. That is a silent correctness bug,
not a nuisance, so a width-1 head that meets such a genotype is now REFUSED, with the group
sizes in the message.

The guard is scoped to width 1 and to the head that would do the averaging, and the scoping
is load-bearing rather than cautious. Measured before it was written:

- `fig6_pigment_transfer` (4,930 genotypes) carries group-size multisets (1, 19) on 4,432
  genotypes, (19,) on 246, (1,) on 237 and () on 15. No genotype has two width-1 groups, so
  it cannot fire on anything already run.
- `fig6_build` (13,948 genotypes) carries exactly ONE colliding genotype, **record 3363**,
  where two Cachera strains (the ARO4/YBR249C deletion and the ARO7/YPR060C deletion, each
  inside the betaxanthin cassette) aggregate to one deletion key and contribute two width-1
  betaxanthin values, +0.2533 and -0.5194.

A guard that fired on that genotype for EVERY head, which is how it was first written, would
have killed every isobutanol cell on this build at its first batch, for a collision the
isobutanol head does not read. Averaging several groups of the same WIDER width is left
alone: that is the documented behavior for genuine replicate experiments.

Six tests cover it in `tests/torchcell/metabolism/test_extract_targets_width_collision.py`,
including the three decodes it must leave unchanged.

**The guard is necessary and not sufficient, and that is the finding that reshaped this
block.** On `fig6_build`, `MeanExperimentDeduplicator` has already MERGED the Lopez screen
into the Mulleder record wherever both were measured, so the isobutanol fold change is not a
width-1 group at all. Measured over all 13,948 genotypes, the `metabolite_level` group
widths are:

| width | genotypes | what it is |
|---:|---:|---|
| 1 | 4,858 | 4,669 betaxanthin, 180 isobutanol screen only, 8 screen + re-screen, 1 re-screen only |
| 19 | 304 | Mulleder amino acids only |
| 20 | 4,244 | the 19 amino acids PLUS isobutanol, key-sorted |
| 33 to 214 | 442 | blocks that also carry da Silveira lipids or the Zelezniak metabolome |

A width-1 isobutanol head would have matched 4,858 groups of which 4,669, or 96 percent, are
betaxanthin colony color. The guard would not have fired on a single one of them, because
each of those genotypes carries exactly one width-1 group. **Width is an identity only while
no two groups collide on it; it is never a statement about WHICH dataset a group came from.**

### How the isobutanol target is actually decoded

The head reads COLUMN 8 of the width-20 block. The column index is derived at import time as
`sorted(AA_KEYS + ["isobutanol"]).index("isobutanol")`, so it follows the graph processor's
own key sort rather than a hardcoded number. That decode is identity-based: no other
combination of datasets in this build produces width 20.

- Supervised blocks: 4,244 genotypes, every one of them carrying an `isobutanol` entry.
- Fold change in those blocks: minimum 0.0386, maximum 7.474, so log10 is well defined.
- The target is **log10 of the fold change**. The distribution is multiplicative (log10 sd
  0.213 over the screen), so a Gaussian CRPS is well posed there and not in linear FC, and
  Spearman is invariant to the change, which keeps the primary endpoint comparable with
  every other 027 arm.

### The validated re-screen is held back, by gene

The 224-strain n=3 re-screen is the external direction check. On this build its value is not
separable from the screen's: the deduplicator averaged the two into one `isobutanol` entry on
208 of the 4,244 width-20 blocks. "Include only the screen" is therefore not available, and
the faithful realization is to mask every genotype deleting a gene that panel measured out of
the loss, the validation curve and the test score. `isobutanol_heldout_flags` derives that set
from the dataset's own `dataset_name_index`, so no second copy of the Lopez gene list exists
anywhere.

Measured on `fig6_build`, and confirmed by the smoke run: **224 records / 224 genes masked,
1.6 percent of the dataset, 0 absent from the cell graph.**

What is left to score on, measured by intersecting the width-20 gene set with Merzbacher's
own split file:

| set | genes |
|---:|---|
| any isobutanol record | 4,556 |
| in the width-20 block | 4,244 |
| pinned TEST, any isobutanol | 581 |
| pinned TEST, width-20 | 527 |
| **pinned TEST, width-20, not held back** | **504** |
| `train_val_pool`, width-20, not held back | 3,353 |
| in neither of Merzbacher's lists | 183 |

The pinned split itself resolves BETTER on this build than on the pigment one: **639 of 639
genes present** (against 629 there), giving 1,801 records pinned to test, 12.9 percent of
13,948. That is the smoke run's own `[pinned-split]` line.

### Which copy of the data was read

This matters because both Lopez LMDBs are mid-campaign.

- **The training build is `$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig6_build`,
  whose `processed/` directory is dated 2026-07-22.** It is opened read-only and is not
  rebuilt. Its `lmdb`, `dataset_name_index.json` and `phenotype_label_index.json` all carry
  that date. Nothing in wave 9 reads the source Lopez LMDBs during training.
- The SOURCE builds `$DATA_ROOT/data/torchcell/isobutanol_screen_lopez2024/processed` and
  `.../isobutanol_validated_lopez2024/processed` were **rebuilt on 2026-09-14 at 02:29 and
  02:23** as part of another session's 36-dataset campaign, and neither now carries a
  `build_manifest.json`, a `verification_report.json` or an
  `experiment_reference_index.json`. The validated build holds 224 entries.
- Only the external-check ANALYSIS reads the validated source build, and it does so after the
  cells finish, which is why that join lives in
  `analyze_wave9_ib_external.py` rather than in the trainer: a 3-hour GPU cell must not die at
  its last step because a dataset build moved.

### The anchors, with GEM ids, read from yeast-GEM 9.0.2 today

The dissertation names the ligand verbatim (mirrored `paper.md`, line 425):

> This biosensor works by expressing green fluorescent protein (GFP) proportionally to the
> amount α- isopropylmalate (α-IPM) in the cell, which is a precursor of leucine and a
> byproduct in the isobutanol pathway ... As more α-IPM accumulates, it binds to Leu3p, both
> of which activate transcription of GFP which is under control of the Leu1 promoter.

So the measured quantity is the 2-isopropylmalate POOL, not a keto-acid flux. Verified
against `yeast-GEM.xml` (ids, names and compartments read directly):

| label | anchor id | other compartments | GEM name |
|---|---|---|---|
| `alpha_isopropylmalate` | `s_0162` [c] | `s_0164` [m]; `s_0163` [e] dropped as non-cell | 2-isopropylmalate |
| `keto_isovalerate` | `s_0232` [c] | `s_0233` [m] | 3-methyl-2-oxobutanoate |
| `valine` | `s_1056` [c] | `s_1058` [m]; `s_1057` [e] dropped | L-valine |

As with `BETAXANTHIN_PRECURSOR_ANCHORS`, the id NAMES the chemical and
`compartment_species_for` derives the rest of its compartments from the GEM at run time, so
the readout cannot drift from the model the layer was built on. The **one-anchor primary** is
alpha-isopropylmalate alone, because that is literally what Leu3p binds; the three-anchor set
is the secondary.

The product needs no cassette, which is what lets the flux arm read a real column:
isobutanol is `s_0932` [c] / `s_0933` [e] / `s_0934` [m] with secretion exchange **`r_1866`**
("isobutanol exchange", `s_0933 -->`). Measured today: `s_0933` is touched by exactly two
reactions, the transport `r_1868` and that exchange, so its TURNOVER is the secretion flux;
`r_1866` is also column 1307 of the 4,131-wide dense term. `ISOBUTANOL_PRECURSORS` therefore
names six intracellular precursor species and the three isobutanol species.

### The medium

`--apply-media {none,dataset}` writes the head's dataset medium onto the GEM's exchange bounds
through `media_to_bounds(FBA_MEDIA[name], model).apply(model)` BEFORE the stoichiometric
tensors are taken, and records the wild-type growth rate before and after in
`flux_meta.media`. `none` is the default, so every earlier cell builds a bit-identical layer.

Measured on yeast-GEM 9.0.2 today, every component of every recipe resolving:

| medium | components resolved | exchanges opened | wild-type growth before -> after (1/h) |
|---|---:|---:|---|
| shipped bounds | -- | -- | 0.0858 |
| SM (Mulleder) | 25 / 25 | 25 | 0.0858 -> 0.3138 |
| YPD-approx (Cachera) | 45 / 45 | 45 | 0.0858 -> 0.5352 |
| SC (Montano Lopez) | 47 / 47 | 47 | 0.0858 -> 0.5428 |

**SM is not a no-op**, which the review's E6 expected it to be: the shipped boundary is
TIGHTER than the recipe's uptake policy, so the amino-acid arm on SM is a real arm and not
only a wiring check. That is a correction to the plan, recorded before any cell runs.

**SC carries no provenance and no concentrations.** Its composition is the standard
Difco/Sigma formulation named in `torchcell/datamodels/media.py`; only the uptake magnitudes
are sourced (Suthers). Every SC-bounded result file says so in
`flux_meta.media.provenance`, which begins with the word UNSOURCED. YPD-approx is Suthers'
defined stand-in and carries his quote; peptone is not modeled by us or by him.

### Held-out amino-acid columns

`--aa-supervise-columns` takes the columns the LOSS is charged on. Wave 9 supervises 13 and
holds out **valine, tyrosine, glutamate, leucine, isoleucine, methionine** -- the first three
are the multi-target anchors, since valine is one transaminase step from the biosensor's pool,
tyrosine is the betaxanthin cassette's substrate, and glutamate is the amino donor for both
transaminations.

One deliberate departure from the review's wording, stated because it changes what the number
means. The review asked for the mask on the loss only. The mask is ALSO applied to the
epoch-selection metric, because selection reads validation Pearson and selecting the epoch by
a held-out column's validation Pearson would be choosing the checkpoint by the very quantity
the arm asks about. The TEST scorer is unchanged and still reports all 19 per column; the two
groups are summarized in a new `aa_column_split` block of the result file.

### Pre-registered reads and kill rules, verbatim from the review

**E3, the isobutanol three-way.** *Pre-registered read:* `ibprec - ibpool` and
`ibscalar - ibpool`, paired by seed; plus the external check on the 224 independent
re-screened strains (fraction of the predicted top/bottom 30 that reproduced at FC>=2 /
FC<=0.5, against the empirical ceilings 59/60 and 150/150). *Success/kill:* `ibprec - ibpool`
> +0.02 with all three seeds positive -> the module transfers to a second, better-conditioned
product target and the concentration mechanism is not betaxanthin-specific. All three arms
within 0.01 -> the target is not sensitive to the module and the honest report is that the
biosensor reads chromatin/mitochondrial state (the paper's own top hits are histone modifiers:
spt10Δ, dep1Δ, sap30Δ, gcn5Δ, swi4Δ, swi6Δ) rather than pathway flux. *Contingency if opposite
(`ibscalar` wins):* that would be the first evidence that a flux readout beats a concentration
readout anywhere; re-run at 3 more seeds before believing it, because the flux head has lost on
every target so far.

**E4, the betaxanthin context-only control.** *Success/kill:* |bxctxprec - bxprec| < 0.012 (the
prec-scalar sd) -> the betaxanthin "mechanism" result is a parameterization result and the
wave-8 conclusion needs restating in the note. *Contingency if opposite:* bxctxprec near bxpool
-> the layer really is doing the work on betaxanthin, which would be the first such evidence and
would justify the joint design (E7).

**E2, the held-out amino-acid columns.** *Pre-registered read:* mean Spearman over the 6
held-out columns, flux-coupled vs context-only vs unsupervised. *Success/kill:* flux-coupled
held-out > 0.02 AND > context-only by more than 0.010 -> the latent is a coupled state, and that
is a physics result a pooled backbone structurally cannot produce. Both ~0 -> the readout is 19
independent sigmoids; retire the "learned concentration" framing and move the budget to E6/E3.
*Contingency if opposite:* if the context-only arm ALSO transfers, the transfer is coming from
the shared backbone, not the layer; report it as such.

**E6, the medium.** *Pre-registered read:* the asymmetric prediction is the point -- amino acids
within 0.005 of their current values, betaxanthin moving by more than its 0.026 paired sd in
either direction. *Success/kill:* if the amino-acid arm MOVES, the wiring changed something it
should not have and the cell is invalid, not informative. If betaxanthin does not move, the
wrong-medium explanation for the betaxanthin null is falsified and should be retracted from the
memory that states it.

The controls these are read against, all already measured and none re-derived here: the
amino-acid arms `wave7_aa_conc_e200_s270{0,1,2}` 0.0648 / 0.0663 / 0.0562 and
`wave7_aa_pooled_e200_s270{0,1,2}` 0.0500 / 0.0425 / 0.0511; the betaxanthin arms
`wave8_bxprec` 0.1404 / 0.1377 / 0.1159, `wave8_bxpool` 0.1007 / 0.1159 / 0.1277 and
`wave8_bxscalar` 0.1027 / 0.0845 / 0.0871; the unsupervised concentration baseline
-0.0041 and +0.0071 from the turnover cells 1766 and 1772. The isobutanol noise floor is
UNKNOWN until the three `ibpool` seeds establish it, and no contrast on that target may be
read before they do.

### The 24 cells

All at 200 epochs, CRPS, the 019 graph regularizer, 60 GB / 4 CPU / 1 GPU / 10 h, submitted
through `freeze_and_submit.sh` so the code is snapshotted per job.

| block | cells | arm | what it changes |
|---|---|---|---|
| B2 | `gh_wave9_ibpool_e200_s270{0,1,2}` | pooled | no module; establishes the target's seed noise floor |
| B2 | `gh_wave9_ibscalar_e200_s270{0,1,2}` | flux_projection_anchored | 026's flux scalar head on the isobutanol precursors and product |
| B2 | `gh_wave9_ibprec_e200_s270{0,1,2}` | flux_projection_anchored | one-anchor alpha-IPM concentration readout |
| B3 | `gh_wave9_bxctxprec_e200_s270{0,1,2}` | pooled | the betaxanthin precursor head with no flux layer |
| B4 | `gh_wave9_aa_holdout_conc_e200_s270{0,1,2}` | flux_projection_anchored | 13 of 19 columns supervised, flux-coupled |
| B4 | `gh_wave9_aa_holdout_ctx_e200_s270{0,1,2}` | pooled | the same mask, context-only |
| B5 | `gh_wave9_bxprec_ypd_e200_s270{0,1,2}` | flux_projection_anchored | YPD bounds under the wave-8 bxprec recipe |
| B5 | `gh_wave9_aa_conc_sm_e200_s270{0,1,2}` | flux_projection_anchored | SM bounds under the wave-7 aa conc recipe |

The three isobutanol arms share `--kcat-tier predicted --thermo-source verified
--constraint-warmup-epochs 20 --constraint-scale 0.1 --conc-upper-m 1.0`, copied from
`gh_wave8_aa_nodiss_e200_s2700.slurm`, so the three-way contrast is on the readout alone. The
betaxanthin and amino-acid cells keep `union_external`, matching the wave-8 and wave-7 cells
they are paired against seed for seed.

### Verification, and the two corrections it forced

**Tests.** `tests/torchcell/metabolism/test_extract_targets_width_collision.py` (6) and
`tests/torchcell/metabolism/test_isobutanol_head.py` (11) are new; the whole metabolism
suite is 251 tests and passes.

**CPU smoke, one epoch, the exact `ibpool` command.** Run on the shell with
`NUM_WORKERS=2`, `--smoke --no-checkpoint`. Measured:

- dataset 13,948 aggregated genotypes; pinned split **639/639 genes present** giving 1,801
  records pinned to test, 12.9 percent; held-out panel **224 records / 224 genes, 0 absent
  from the cell graph**.
- epoch 0: `loss 1.4917 data 0.5538 val_loss 0.5564 val_r +0.0791 sd_ratio 0.0009`,
  reproduced bit-identically across two runs.
- test block: **`n` = 504 on the pinned genes** and 789 over the whole test split, which is
  exactly the count the static intersection predicted.
- `isobutanol_external_check`: 224 records, 30 top, 30 bottom, all 224 predictions dumped.
- feasibility, calibration and the concentration audit correctly report unavailable on a
  layer-free scalar arm.

At one epoch the head is still on the mean plateau (`sd_ratio 0.0009`, predicted log10 FC
spanning 0.0131 to 0.0142 across the whole held-out panel), so the smoke's own test
Spearman of -0.0498 is the correlation of a constant plus jitter and is **not** a result.

**`ibprec` construction and step check**, on the real dataset and the real flux layer
(`scratchpad/w9/check_ibprec.py`, two training batches plus one test batch): readout
`FluxPrecursorConcentrationReadout` on head `isobutanol`, 934,024 parameters, whole-cell
mixture `{alpha_isopropylmalate: [c, m]}` over `s_0162` at fractional volume 0.65 and
`s_0164` at 0.05, window [1e-07, 1e+00] M; batches decoded 45 of 128 observed with log10 FC
in [-0.516, +0.438]; loss 1.7206 with `thermo_log_c` of shape (128, 2806); the backward
pass through `constraint_loss` and the optimizer step both ran.

**Correction 1, found by the smoke and already applied.** The first smoke died in
`_score_test` on the unscoped guard, at record 3363, after completing epoch 0. That is the
collision described above, and it is why the guard is scoped to the head that would average.

**Correction 2, a pre-existing defect fixed in passing.** The per-gene test dump named its
file `{arm}_s{seed}.json`, so wave 8's `bxprec` and `bxscalar` ran on the SAME base arm and
the same seed and the second overwrote the first; `ibprec` and `ibscalar` would have done
the same. The stem now appends the readout when it is not that head's default, which leaves
every file the existing Merzbacher head-to-head scripts read under the name it has.

**Note on `flux_meta.media`.** It is written by `build_flux_layer`, which returns early on a
`pooled` arm, so a layer-free cell carries no `media` key at all rather than
`{"applied": false}`. Every flux arm records it either way.

### Submitted

| job | cell |
|---:|---|
| **1926** | `gh_wave9_ibpool_e200_s2700.slurm` |
| **1928** | `gh_wave9_ibprec_e200_s2700.slurm` |

Both went through `freeze_and_submit.sh`, so each runs from its own frozen snapshot under
`$DATA_ROOT/job-snapshots/`. The other 22 cells are written and verified but NOT submitted.

### 2026.09.14 - Wave 9 seed 2700 diagnosis: ibpool overfits, ibprec never escapes

Jobs 1926 (`ibpool`, W&B `6z2ovude`) and 1928 (`ibprec`, W&B `s4z0ur7o`), both 200 epochs.
Read from the two result JSONs in `results/`, the two checkpoints in
`$DATA_ROOT/checkpoints-027/`, and the split cache.

#### The trajectories

`ibpool`, first 40 epochs then the last 20:

| ep | data loss | val loss | val Pearson | sd ratio |
|---:|---:|---:|---:|---:|
| 0 | 0.5526 | 0.5543 | -0.0491 | 0.0004 |
| 10 | 0.5355 | 0.5531 | -0.0685 | 0.0055 |
| 16 | 0.5376 | 0.5502 | +0.0929 | 0.0960 |
| 19 | 0.5450 | -- | +0.1137 | 0.1137 |
| 26 | 0.5409 | 0.5504 | +0.0789 | 0.0646 |
| 38 | 0.5321 | 0.5562 | +0.0546 | 0.0859 |
| 180 | 0.3925 | 0.7270 | -0.0431 | 0.6367 |
| 199 | 0.3709 | 0.7120 | -0.0435 | 0.5571 |

`ibprec`:

| ep | data loss | val loss | val Pearson | sd ratio |
|---:|---:|---:|---:|---:|
| 0 | 0.6555 | 0.6228 | +0.0618 | 0.00000 |
| 10 | 0.5413 | 0.5542 | +0.0292 | 0.00029 |
| 14 | 0.5397 | 0.5535 | +0.1316 | 0.00037 |
| 40 | 0.5532 | -- | -- | 0.00224 |
| 100 | 0.5436 | -- | -- | 0.00009 |
| 180 | 0.5399 | 0.5531 | -0.0089 | 0.00000 |
| 199 | 0.5386 | 0.5534 | +0.0615 | 0.00009 |

**The collapse-abort did not fire and could not have.** Both cells carry
`abort_on_collapse: false` in their serialized record: I never passed `--abort-on-collapse`.
The `COLLAPSE_PATIENCE_EPOCHS=200` the cells export sets the PATIENCE, not the switch, and at
200 it could not fire inside a 200-epoch budget even if the switch were on.

#### `ibpool` is not broken, it is over-budgeted

Its spread leaves the plateau at epoch 10 and is 0.096 by epoch 16. The selected epoch 21
scores **pinned Spearman 0.1278 on 504 genes**, inside the wave-8 betaxanthin pooled range
of 0.1007 / 0.1159 / 0.1277. What follows is 175 epochs of textbook overfitting: train data
loss 0.553 to 0.371 while validation loss climbs 0.554 to 0.712. The negative last-5
validation mean is that tail, not the selected epoch, and the scoring rule already excludes
it. Nothing here needs a code fix.

#### `ibprec` learned nothing at all, and three suspects are refuted

Its train data loss sits between 0.538 and 0.548 for 195 consecutive epochs. For contrast,
wave-8 `bxprec` is the same readout class on the same layer and window and DID escape: spread
0.0002 at epoch 20, 0.0228 at epoch 40, 0.0661 at 60, 0.3521 at 80, selected epoch 109, data
loss 0.528 to 0.421. So `ibprec` is not a slow escaper, it is flat.

**Refuted: the target scale.** The log10 fold change IS standardized on the train pool.
`run_cell` accumulates the train-split target (train_bx.py L3887-3899), takes `nanmean` and
the centered `nanmean` sqrt as mean and sd, and the loss is charged on `z = (t - mean) / sd`
(L3980 in training, L4576 in the held-out loss). The transform into log10 happens once, in
`targets_in_space` (L1440, L1488-1494). The checkpoints record what was used: **mean 0.0124,
sd 0.1831**, and `ibpool` and `ibprec` carry the SAME pair. One escaped and one did not, so
the scale cannot be the discriminator, and a CRPS sigma head initialized for unit variance is
seeing unit variance.

**Refuted: loss masking.** Train split 9,733 records, of which 2,981 carry a supervised
isobutanol target: 30.6 percent, about 39 per 128-row batch. Thin next to betaxanthin's 95
percent, but not starvation, and `ibpool` learned from the identical mask.

**Refuted: constraint balance.** The constraint-to-data ratio is 0.05 for `ibprec` at epochs
20, 100 and 199, and also 0.05 for `bxprec` at the same three epochs.

**The head is not dead either.** From the `ibprec` checkpoint, the anchor rows of
`flux_layer.log_c_from_context.weight` have norms 0.844 (`s_0162`) and 0.847 (`s_0164`)
against a global row-norm median of 0.583; `bxprec`'s five anchors run 0.50 to 0.74. The
readout affine learned a non-zero weight, 0.679 for the mean. So the weights grew and the
prediction still did not move, which means the POOL the readout reads is constant.

#### What is thin, and it matters for selection

The validation split carries only **266 supervised isobutanol measurements** (1,203 records,
22.1 percent). The null width of a Pearson on n = 266 is about 0.061, so `ibprec`'s selected
epoch 13 at validation Pearson 0.1035, taken while its prediction spread was 0.0009, was
chosen from a curve whose noise is comparable to the value. Its reported pinned 0.0762 is
jitter on a constant, exactly as the coordinator read it. Betaxanthin's validation split
carries roughly 465 measurements and the amino acids carry 19 columns, so this target's
selection is the noisiest of the three.

#### Remaining hypothesis, labeled

Hypothesis (untested): a SINGLE anchor is too narrow a channel. The whole
genotype-to-phenotype map has to pass through one scalar that is simultaneously constrained
by the second law and the dissipation limit across 2,806 metabolites, while `bxprec` has a
five-dimensional bottleneck. The only other structural difference is the reaction-energy
table, `verified` against `bxprec`'s `union_external`.

#### Fix applied: instrumentation, not a mechanism guess

Both changes are additive and leave every head's computation bit-identical.

1. `result["target_standardization"]` now records the head, the target space, and the
   train-split mean and sd actually used. It lived only in the checkpoint, which is why
   answering "was it standardized?" needed a 2 GB file.
2. `result["precursor_concentration"]` now records, for every precursor-concentration arm,
   the across-genotype sd of the whole-cell log10 concentration at each anchor plus the
   fraction of predictions pinned at the window ceiling and floor, from one pass over the
   test split. A flat pool and a dead affine are different numbers there; wave 9 could not
   tell them apart without rebuilding the model from the checkpoint.

#### Cells written, NOT submitted

`gh_wave9b_ib{pool,scalar,prec}_e200_s2700.slurm`. All three take
`--thermo-source union_external` (matching the wave-8 betaxanthin three-way exactly) and
`--abort-on-collapse` with `COLLAPSE_PATIENCE_EPOCHS=100`; `ibprec` additionally takes
`--ib-anchors three`. The patience is measured rather than guessed: the latest escape onset
among arms that recovered is epoch 40 (`bxprec`), so 100 is 2.5x the observed worst case and
would have stopped `ibprec` at epoch 100 while sparing every arm that escaped.

`ibprec` moves two flags at once, so a 9b `ibprec` that escapes will not say which of them
did it; separating them needs two more single-flag cells. That is a repair attempt, and it is
labeled as one.

#### What the new audit already says, at initialization

Run on the real dataset and the real layer under the wave-9b configuration
(`union_external`, three anchors), untrained, over 256 test records:

| anchor | sd of log10 c (decades) | median log10 c | range |
|---|---:|---:|---|
| alpha_isopropylmalate | 0.0715 | -4.504 | [-4.708, -4.215] |
| keto_isovalerate | 0.0865 | -2.741 | [-3.134, -2.207] |
| valine | 0.0637 | -4.826 | [-4.947, -4.459] |

`frac_at_ceiling` and `frac_at_floor` are both **0.0000**.

Two things follow, and they reorder the remaining hypotheses. The sigmoid is **not**
saturated at initialization, so window saturation is not the mechanism. And the pool **does**
vary across genotypes at initialization, by about 0.07 decades, while the trained `ibprec`
implies a spread near 1.5e-4 decades (its prediction spread of 1e-4 standardized units
divided by the learned affine weight 0.679). So training did not fail to develop a spread, it
**destroyed one that was there at the start**, by roughly 500x.

Hypothesis (untested), and it is now the leading one: the thermodynamic and dissipation terms
drive `log c` toward a genotype-independent configuration, and with only two supervised
concentration rows out of 2,806 the data term cannot resist it. The three-anchor arm triples
the supervised rows to six, which is what wave 9b tests; a cleaner test of the same idea would
hold the anchors at one and drop the constraint scale to 0.
