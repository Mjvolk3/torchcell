---
id: j9ct0792j681cw5qwvdrfrr
title: Audit_ploidy_representation
desc: ''
updated: 1790484357395
created: 1790484357395
---

## 2026.09.26 - The schema already handles heterozygosity, the tensor layer throws it away

Audited for the five-dataset representation design, because pooling Hillenmeyer HET and
Hoepfner HIP with the homozygous and haploid datasets requires representing a gene at half
dose. Verified against the built stores, not only read from source.

**The schema needs no extension.** Heterozygosity is carried in two parts. A genome-wide
baseline sits on `ReferenceGenome.ploidy`, and a per-locus deviation sits on
`EngineeredCopyNumberPerturbation` with `copy_number` and `reference_copy_number`. That
extension was made for exactly this case; the Hoepfner loader docstring names HIP and HOP as
its motivating use.

| dataset | ploidy | perturbation stored |
|---|---|---|
| Vanacloig 2022 | haploid | 4 deletions, barcoded KanMX query plus 3 marker and NatMX background |
| Hillenmeyer HOM | diploid | `KanMxDeletionPerturbation`, state absent |
| Hillenmeyer HET | diploid | `EngineeredCopyNumberPerturbation`, state present, 1 of 2, KanMX |
| Hoepfner HOP | diploid | `KanMxDeletionPerturbation`, state absent |
| Hoepfner HIP | diploid | `EngineeredCopyNumberPerturbation`, state present, 1 of 2, KanMX |
| Wildenhain 2015 | haploid | one `BarcodedKanMxDeletionPerturbation`, state absent |

Heterozygous and homozygous records are distinguishable from the stored record alone on three
redundant signals: the perturbation class, the `state` field, and the presence of the
copy-number fields. Hoepfner holds both arms in one store separated only by class.

**The gap is entirely in the tensor layer, and it is total.** Verified by direct grep, both
returning nothing: `copy_number` appears nowhere in `torchcell/data/`, and neither does
`ploidy`. The perturbation processor builds its input from a single field:

```python
perturbed_names = {
    p.systematic_gene_name
    for item in data
    for p in cast(Experiment, item["experiment"]).genotype.perturbations
}
```

So a Hillenmeyer HET record and a HOM record for the same gene produce byte-identical model
input. Both contribute one index to `perturbation_indices` and one True to a boolean mask. The
diploid baseline, the surviving wild-type copy and the `state="present"` flag are all discarded.
Pooling the two arms today would hand the model identical inputs carrying different labels.

**All four modalities already share one perturbation representation**, which is what makes a
unified design tractable. Trigenic interactions, expression and metabolic flux all consume the
same variable-length set of perturbed gene node indices plus its batch assignment. A triple is
three indices rather than one, and the processor has no cardinality logic. Everything
downstream of that is head choice, not perturbation encoding.

**Recommended minimal change, a judgment and not yet implemented.** Keep the schema. Add one
continuous per-gene channel beside `perturbation_indices`, a functional dose equal to
`copy_number / reference_copy_number` where those fields exist, 0.0 for an absence leaf and 1.0
for an unperturbed gene. It is a strict generalization of today's boolean, since every current
dataset emits 0.0 and reproduces current behavior exactly. Normalizing by the reference copies
rather than using raw copies is what makes haploid and diploid comparable in one model. Call it
a functional dose rather than a copy number, because that channel is also the only honest place
a DAmP allele or a CRISPRi knockdown can go later.

**What a copy-number vector would lose.**

- **Cassette identity.** Vanacloig's background carries three different markers and the
  heterozygous arms record KanMX on the affected allele. A KanMX replacement is a deletion and
  an insertion of a heterologous gene that is not in the S288C universe, so no vector over
  S288C genes can hold it.
- **Barcode and collection.** These are the readout identity in a pooled Bar-seq assay and the
  only discriminator between two collections holding the same ORF deletion.
- **Allele classes off the dosage axis.** Not a loss for these five, but a hard limit on the
  unified design, because 006 and 010 draw on Costanzo, which uses temperature-sensitive, DAmP
  and suppressor alleles. A ts allele sits at full DNA copy number and is functionally
  compromised, so a copy-number vector would score it 1 or 2 and be wrong. CRISPRi has the same
  problem by construction.
- **Perturbation cardinality as a strain property.** Vanacloig contributes 4 indices per record
  against 1 for the other four datasets. Hypothesis, untested: the constant 3-gene
  drug-sensitized background is a dataset-identity shortcut the model can key on, so an
  ablation that drops the background indices is worth running.

**One verified risk for the metabolic module.** `FluxLayer.gene_availability` hard-zeros a
deleted gene's slot in the genome-scale model with `keep[rows, cols] = 0.0`. A heterozygous
deletion is not a zero there. Admitting HET or HIP data to the metabolic module without
changing that line asserts a full knockout of a gene that still has one working copy.

Result file: `experiments/031-env-chemgen-inhibitor-tolerance/results/ploidy_audit.csv`
