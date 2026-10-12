---
id: 2gugq9uiuohip7sjnejornx
title: Potel2018_phosphosite_checks
desc: ''
updated: 1791817500275
created: 1791817500275
---

## 2026.10.12 - What the script measures (#921)

`experiments/036-dataset-fixes-before-kg-build/scripts/potel2018_phosphosite_checks.py --base origin/main`, run under slurm, writes `experiments/036-dataset-fixes-before-kg-build/results/potel2018_phosphosite_checks.json`:

1. **Proteome-family fingerprints.** The contract fingerprint of `Phenotype`, `ProteinAbundancePhenotype`, `ProteinFoldChangePhenotype`, `ProteinTurnoverPhenotype`, `ProteinSynthesisRatePhenotype`, their experiment pairs, `Environment`, `Genotype` and `AssemblyReferenceGenome`, at the base and in the tree. The phosphosite leaf is a sibling, so each is expected to be unchanged.
2. **Stale-store round trips.** Every record of each dev store that `schema_impact` names as impacted through the five unions and type maps the leaf grows (`ExperimentType`, `ExperimentReferenceType`, `PhenotypeType`, `EXPERIMENT_TYPE_MAP`, `EXPERIMENT_REFERENCE_TYPE_MAP`) is validated through the branch's schema and re-dumped; a dump that differs from the stored dict would mean the additive leaf changed a stored serialization. A shared dev store pickled by another branch's schema is reported as not readable under this branch, not compared (the Li 2014 checks' rule; the first run, slurm 4400, stopped on one such store, `NaturalLossOfFunctionCause`).

Status at the time of writing: queued as slurm job 4419; no result is claimed here. The W3110 vocabulary carried in the same PR has its own record and is not measured by this script.
