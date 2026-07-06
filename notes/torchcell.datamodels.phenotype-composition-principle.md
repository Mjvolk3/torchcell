---
id: nyk3h6v5e7xputjfqm6q2x1
title: Phenotype Composition Principle
desc: ''
updated: 1783309480372
created: 1783309480372
---

## 2026.07.05 - The Deciding Principle (one phenotype vs many)

Motivated by `[[torchcell.datasets.scerevisiae.ozaydin2013]]`: its
`VisualScorePhenotype` conflated a metabolite-proxy colony COLOR score with petite /
slow-growth / QC booleans parsed from the same Comment column. For phenotype
CONVERSION (e.g. visual color score -> estimated absorbance -> carotenoid
concentration) the color signal is convertible but petiteness is NOT -- it is a
different biological axis and a confounder. Same discipline is wanted for pretraining
(Mulleder amino-acid metabolome, expression, morphology): a model must see the pure
measured signal separately from separable confounders.

### The rule

> **One phenotype record per (dataset x measurement modality).** Within a modality,
> keep the assay's native output shape -- SCALAR if the assay emits a scalar
> (fitness, interaction, colony color score), VECTOR if the assay is inherently
> multivariate (morphology). Promote a quantity to a SEPARATE phenotype record only
> when it (a) comes from a different modality / biological axis, or (b) must be held
> out as a CONFOUNDER when converting or modeling the primary signal, or (c) is
> independently interpretable/convertible on its own. A quantity that is neither
> measured signal nor separable-confounder -- a pure data-VALIDITY flag -- is NOT a
> phenotype; it is QC/exclusion metadata.

**Key discriminator: dimensionality != modality.** Morphology is high-dimensional but
single-modality (one imaging pipeline, same cells, one normalization, interpreted as a
joint signature) -> ONE vector phenotype. Ozaydin's Comment column is low-dimensional
but multi-modality (color assay + respiration observation + fitness observation) ->
SPLITS. Operational test: "would a conversion/embedding of the primary signal need to
hold this quantity out?" yes -> separate; no -> same record.

### Applied

- **Kuzmin / Costanzo** (`GeneInteractionPhenotype`, `FitnessPhenotype`): one scalar
  phenotype each. One modality, one output. Nothing to split -- these DO NOT churn.
- **Morphology** (`CalMorphPhenotype`, Ohya): one VECTOR phenotype (~500 features are
  one imaging modality, not 500 phenotypes).
- **Ozaydin** (`VisualScorePhenotype`): `visual_score` = primary; petite/tiny/
  slow_growth = separate (respiration/fitness axis + confounders of the color->
  carotenoid conversion); qc_failure/het_diploid = validity metadata (not phenotype).
- **Mulleder** amino acids: one MODALITY (LC-SRM), 18 analytes -> naturally ONE
  MetabolitePhenotype whose `metabolite_level` dict holds all 18 (vector-by-key),
  NOT 18 separate records. Same "vector within one modality" case as morphology.

### Three-tier vocabulary (locked 2026.07.05)

Because Ozaydin's secondaries are PARSED FROM FREE TEXT (qualitative observations, not
a quantified petite-frequency assay), do NOT force them into the full `Phenotype`
class. Three tiers:

1. **Primary measured phenotype** -- quantified; the conversion/embedding target
   (color score; fitness; morphology vector; amino-acid levels).
2. **Secondary qualitative annotation** -- real biology, only qualitatively noted
   (Ozaydin petite/tiny/slow_growth). Lighter than a `Phenotype`; separable; never
   fabricated into a fake quantity.
3. **Validity / QC metadata** -- not biology (qc_failure, het_diploid). Drives
   exclusion, lives off the phenotype.

### Open (design WITH the conversion/pretraining framework -- do not churn twice)

- Container: does `Experiment.phenotype` go plural (list of phenotype records) or does
  a `CompositePhenotype` wrap primary + secondaries? Settle when the conversion
  framework lands; most datasets stay single so this is not urgent.
- Where the secondary-annotation tier physically lives (a typed sub-model on the
  primary phenotype vs a sibling field on `Experiment`).
- Interaction with `[[torchcell.datamodels.gene-addition-perturbation-design]]`
  (both are "pure measured signal + separable context" discipline).
