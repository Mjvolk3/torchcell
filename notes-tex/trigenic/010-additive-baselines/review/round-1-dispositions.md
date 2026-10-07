# Round 1 review dispositions

Comments pulled from `010-additive-baselines_2026-09-01-22-04-06_32f02891.pdf`
with `notes-tex/common/zotero_comments.py`. Keyed by the stable Zotero
annotation id. Four of the eleven found real errors in the document, and three
of those four were errors of fact rather than of emphasis.

## [1] `JSDU4J67` -- "the nine interaction graphs never enter the forward pass"

**Reviewer:** they are used for regularizing the graphs.

**Correct, and the wording was wrong twice over.** The Kullback-Leibler penalty
is computed inside the same `forward()` call as the prediction, from the same
pre-dropout attention tensor, and nothing detaches it, so its gradient reaches
the layer-1 projections, layer 0 and the gene embedding table. Abstract and
Section 3.3 now say the graphs act on the parameters during training and not on
the function from checkpoint to prediction.

## [2] `PVKGJRQC` -- "0.79 to 0.82, so the non-additive content is largely not reproducible"

**Reviewer:** how is that not reproducible, and do we even change the seed.

**Both halves correct.** The comparison of 0.79 to 0.82 against 0.73 to 0.76 was
not sound, because both correlations are dominated by the additive part every
model reproduces. Replaced with a direct measurement: residualize each
prediction on B1, then correlate. The non-additive component is 42 to 47 percent
of prediction variance, agrees across two runs at 0.53 to 0.58, and tracks the
residual label at 0.23 to 0.25. On the seed question, the training script fixes
the seed at 42; M01 and M02 differ only in run-to-run nondeterminism and M03
also changes the scheduler cycle. The document now says so and stops calling
them seeds.

## [3] `SMNS9CQL` -- "predictions do not agree between inference runs of one checkpoint"

**Reviewer:** the model is deterministic, you are probably using it incorrectly,
verify the datasets and that you are not mixing indices.

**Correct on every count, and it was indices.** Re-scoring stored triples
through the validated direct-scoring path reproduces each run exactly, at
Pearson 0.999998 and mean absolute difference around 2e-5, so nothing was
stochastic. The genome gene set went from 6,607 to 6,579 between runs; the 28
missing mitochondrial ORFs sort first, so every index shifted by exactly 28.
The stored records are byte-identical across the three builds, so schema drift
was not the cause. New Section 7.3.

## [4] `RWJLWZ8X` -- "setting that penalty's weight to zero cannot change any prediction"

**Reviewer:** but we do enforce it, and `val_edge_recovery_summary/recall` shows
we recover edges.

**Correct.** The sentence was true only for inference on fixed weights and read
as though the penalty did nothing. The penalty was 99.99 percent of the training
loss (`norm_weighted_graph_reg` 0.99986), and the regularized heads recover
their graphs at 0.729 to 1.000 neighbor recall. New Section 6.3 reports both,
plus the matched training-time ablation.

## [5] `ZKRKZBVX` -- "both were fixed in July 2026, after these runs", with W&B links

**Reviewer:** those look to me like best val models.

**Both defect claims withdrawn as false for these runs.** The suffixed edge
names (`physical_interaction`) were introduced 2026-06-27, six months after the
December 2025 runs, and the runs log `val_edge_recovery/physical_L1_H0`, a key
the trainer emits only for a matched graph, so all nine heads were regularized.
The squared-lambda defect belongs to the 019 configuration; the 010 script reads
the global scale from a different key whose value is 1.0. The real defect is the
opposite of what was claimed: `len(A.nonzero()[0])` on a dense torch tensor is
2, not the edge count, so the intended division by mean degree multiplies by
367. On the linked groups, they are `cabbi_006`, 30-epoch replicates of the M03
recipe; one reaches 0.4639 against M03's 0.4619, which is within run-to-run
spread, and its checkpoint is not mirrored locally.

## [6] `S4JAFRGL` -- where do the models come from

**Reviewer:** be clear why they are sourced, we need citations, URLs for now.

**Added.** New Section 4.1 with a provenance table: B1 and B5 come from the
Visani preprint, B2, B3 and B4 were built here for the Kuzmin screen
confounder. Both DOIs given as URLs.

## [7] `JNL5NU2V` -- "six things have to match, five do"

**Reviewer:** state them explicitly and say which do and do not.

**Added.** Table in Section 5 naming all six with a verdict on each. The one
mismatch is label normalization.

## [8] `PEUMJZTP` -- "capacity, not the graphs, closes most of the remaining gap"

**Reviewer:** how can you explain this when the attention patterns are actually
mapping to graphs.

**Retracted.** The inference-time ablation cannot support the claim. Section 6.3
adds the training-time evidence: the penalty dominated the loss, the heads
recover the graphs, and the matched pair without the penalty fails to train
(validation Pearson -0.007, -0.031, one diverged, against 0.464, 0.456, 0.456
with it). The document now says how much of the 0.03 is the graphs is not
separable from these runs, and names the degree-matched random-graph control
that would separate it.

## [9] `W6RU6BEL` -- "this looks like potentially a horrible problem"

**It is, and it is now confirmed.** See [3]. The panel-12 and panel-24
selections are invalid rather than uncertain, and Section 7.3 enumerates what
they contaminate and what they do not. A guard now raises when the cell graph's
gene count differs from the checkpoint's `gene_num`.

## [10] `67RW4R6H` -- "more noise for high triples, can we really expect more overlap"

**Fair, and the framing was wrong.** Added the overlap sweep: between two
checkpoints it rises monotonically from 0.10 to 0.30 at K=10 through 0.39 to
0.47 at K=100 to 0.91 at K=10,000. The document now says the tail figure is what
should be expected at an extreme quantile, and is a reason to ensemble rather
than evidence of a defect.

## [11] `ZLMC3D3B` -- modernize 010, and version the data

**Done for the migration, scoped for the versioning.** All 376,732 records now
migrate onto the current schema and validate. The document had this wrong in
detail: it is not one break but two (the ontology refactor and the newly
required `Media.is_synthetic`), the replacement vocabulary includes
`mean_deletion`, and no record in this build is a natMX deletion. The migration
is a key-preserving copy, because every published artifact is position-keyed and
a rebuild reorders. On versioning, Section 8.3 argues a commit hash cannot
answer it, using [3] as the proof: two runs at the same commit scored different
genes because the genome moved underneath them. Three steps named, none of the
general ones built.

## What changed that no comment asked for

The query-pair-disjoint split, which was next step 2 of the previous round. It
is the largest result in the revision: the additive null falls from 0.400 to
0.127 +/- 0.033 across five query-pair-grouped folds, and the nonlinear baseline
falls below the additive one.
