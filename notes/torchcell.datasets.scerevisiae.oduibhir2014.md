---
id: pa621rbrdwtsfs5m902pkga
title: Oduibhir2014
desc: ''
updated: 1783885375641
created: 1783885375642
---

## 2026.07.12 - O'Duibhir 2014 growth-rate (fitness) build

`SmfODuibhir2014Dataset` (`torchcell/datasets/scerevisiae/oduibhir2014.py`, REPLACING a
broken Mac-path stub). O'Duibhir et al. 2014 (Holstege), *Mol Syst Biol* 10:732,
doi:10.15252/msb.20145172, PMID 24952590. Per-deletion relative growth rates →
`FitnessPhenotype`. **L0-L4 verified, 1,312 records.**

- **Classified as FITNESS, not expression** (user hypothesis confirmed): the only new
  per-strain data is Dataset S2 = relative doubling times; the paper's "expression"
  datasets are the Kemmeren 2014 compendium + PCA transforms (100% redundant with the
  existing Kemmeren microarray dataset).
- **Source (open-access SI, mirror + hash-pin):** Dataset S2 `data set 2.txt` (sha256
  `37ef19ee…`) from the EMBO/EuropePMC SI (PMC4265054 — the task's PMCID was wrong).
  Cols: ORF, commonName, `log2relT`, similarity. n=1,312.
- **Fitness direction (empirically reconciled, NOT from the docstring):** the schema field
  says `wt/ko`, but the BUILT Costanzo SMF stores sick mutants BELOW 1 (e.g. ded1-f144c
  = 0.114) → real convention is ko/wt. So `fitness = 2^(-log2relT)` (slow grower <1). Verified:
  PAF1Δ=0.573, min 0.255, max 1.183, WT ref=1.0. (Schema docstring is misleading — flag for fix.)
- n_samples=2 (biological duplicates, sourced); no per-strain SD released → uncertainty None.
  Genotype KanMxDeletionPerturbation; ReferenceGenome BY4741; SC liquid 30 C (Kemmeren setup).
- Adds a new fitness verifier `torchcell/verification/fitness.py` (L3 `reference_one`) +
  `FITNESS_DATASETS` / `run_fitness` in runners.
- FLAG: mating type (BY4741 vs BY4742) not resolvable per strain from S2 → BY4741 representative.
