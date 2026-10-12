---
id: bfl3irwxk8p3wfk1k14mq4p
title: Protein
desc: ''
updated: 1783563814904
created: 1783563814904
---

## 2026.07.08 - Protein verifier: gate WS9 proteome records beyond what the schema encodes

This verifier exists because `ProteinAbundancePhenotype` datasets (WS9 Zelezniak SWATH-MS kinase knockouts) carry invariants the schema validator alone cannot express -- one record per knocked-out ORF, an absolute WT reference (not a centered 0), and a single measurement_type across the set. The schema already enforces non-empty abundances, matching abundance/replicate keys, and non-negative SE, so L0 subsumes those; this module adds the L1-L3 checks that make the built LMDB trustworthy for training.

- L3 `reference_finite` is the key distinction from the centered-score families: protein abundance is ABSOLUTE, so the WT reference must be finite and key-matched to the experiment, NOT identically zero.
- L1 `orf_uniqueness` guards against a knockout appearing in multiple records; L3 `measurement_type_consistent` refuses to silently mix assays.
- Exposes `protein_gene_set` (the knocked-out ORF union) as the L4 overlap key; the cross-source containment check against the deletion collection is asserted by [[torchcell.verification.runners]], not here.
- Built from the shared checks in [[torchcell.verification.levels]] and reported via [[torchcell.verification.report]].

## 2026.10.11 - Natural-isolate gates (Muenzner 2024)

- `verify_natural_isolate_proteome` and `verify_natural_isolate_turnover` serve stores whose records are natural isolates, each a perturbation set of thousands of leaves off S288C. L1 is one record per isolate (`strain_uniqueness`, shared with the RNA-seq family) plus `genotype_strain_consistency` (every leaf names one isolate); the ORF-uniqueness rule above does not apply.
- L3 `shared_reference`: one population reference (the paper's euploid median) serves every record; the proteome gate requires it to cover every measured key, the turnover gate requires the uncovered cells to equal the build ledger's count (939 for Muenzner 2024). The turnover gate re-derives `half_life = ln 2 / degradation_rate` (tolerance 1e-9 relative).
- L4: protein keys against the S288C R64 ORF + RNA set, the isolate overlap with the built Caudal 2024 store (an exact expected count), and `caudal_genotype_agreement` (sampled shared isolates must carry identical perturbation lists in both stores).
- L0 validates against `EXPERIMENT_TYPE_MAP[experiment_type]`, not the `ExperimentType` union: the undiscriminated union tries every experiment class and every perturbation leaf, measured 21.8 s against 0.67 s for one 4,768-perturbation record.
- Entry point: `torchcell.datasets.scerevisiae.muenzner2024.verify_build`, also what `build_dataset_lmdb --verify` and the `natural_isolates` entry of [[torchcell.verification.runners]] `PROTEIN_DATASETS` call. Results: [[torchcell.datasets.scerevisiae.muenzner2024]].

## 2026.10.12 - An S288C-derived column in a natural-isolate store (#1045 row 18)

`natural_isolate_of(record)`: the shared `strain_id` of the perturbations, or, for a record with no perturbation off S288C, the background its own reference genome names (Muenzner 2024's FY4); None when neither says. `strain_uniqueness` (through `rnaseq._l1_strain_uniqueness(strain_of=...)`), `genotype_strain_consistency` (which counts such records as `n_s288c_derived`) and the Caudal overlap read isolates through it. A record with no perturbation on the S288C reference still fails.
