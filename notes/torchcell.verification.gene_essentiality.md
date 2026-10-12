---
id: 91rxc45ismg7jj9lvrgxynv
title: Gene_essentiality
desc: ''
updated: 1791686815992
created: 1791686815992
---

## 2026.10.10 - Gene-essentiality verifier (#889)

Single-pass battery for `gene_essentiality_sgd`: `structural`, `reference_structural`, `count`, `single_gene`, `one_record_per_gene_and_publication` (a record identical down to its publication is a duplicate; two papers for one gene are two records), `essential_label`, `released_annotations` (stored (gene, PubMed) multiset against the SGD per-gene JSON release, pinned by `released.sgd_json_digest`), `reference_viable`, caller rows, shared rules. `essential_without_viable_deletion_result` is the cross-dataset L4 row the runner adds against `smf_costanzo2016`.

## 2026.10.11 - Module removed: SGD essentiality is a provenance record

`Environment.media` stays required (owner, [[torchcell.datamodels.schema]]) and no SGD annotation states a medium the schema can carry, so `GeneEssentialitySgdDataset` left the served graph and its source is a provenance record ([[torchcell.datasets.scerevisiae.sgd]]). There is no gene-essentiality store left to verify: `torchcell/verification/gene_essentiality.py` and `runners.run_gene_essentiality` are removed, with `GENE_ESSENTIALITY_DATASETS`, `viable_deletion_genes` and their tests. `run_all` no longer calls the family, and the registry-root pin in `test_runners.py` reads 56 rather than 59 (this family's one root and the two SynLethDB roots left with it).

The checks the verifier made are not lost, they moved to where the evidence now lives: the allele rule, the condition inventory and the SGD-versus-Costanzo disagreement split are measured functions and constants of the provenance record module, pinned by its own tests against the pinned dump. The 7 unconditional disagreements stay unexplained and are recorded as `UNCONDITIONAL_DISAGREEMENTS`.

## 2026.10.12 - The verifier is back with its dataset; "no condition" reads the medium's name

`GeneEssentialitySgdDataset` stays (owner decision, [[torchcell.datasets.scerevisiae.sgd]]), so this module and `run_gene_essentiality` are restored. `states_no_condition` now reads `environment["media"]["name"] is None`, since the medium is a `Media` whose labels are typed gaps rather than `None`: an unnamed medium with no temperature and no edit is unconditional. SGD's stateless "YPD with 2% glucose" (YIL051C, MMF1) names a medium, so that one annotation is now conditional, as SGD records it; it is not one of the 23 SGD-versus-Costanzo disagreements.

## 2026.10.12 - The cross-release contradiction reports, never gates

`no_viable_deletion_in_<store>` judges this store's essentiality annotations against another paper's deletion strains through the shared gene key, which is a gene-set judgment across releases and not this store's fidelity to its own bytes, so it is `report`-class ([[torchcell.verification.report]], owner 2026-10-12: "the genome is separate.. it doesn't gate anything"). Its computation is unchanged. On the dev tree's current report it does not hold for `gene_essentiality_sgd`: 7 of 1,119 unconditionally essential genes are viable full-deletion strains of `smf_costanzo2016` (YHR072W-A, YHR099W, YJL012C, YLR293C, YNL138W-A, YOR167C, YOR182C); the store's verdict now reads its gate rows. `released_annotations` stays a gate.

## 2026.10.12 - An `evidence` row (#924)

`GeneEssentialityPhenotype` carries a required `GeneEssentialityEvidence` ([[torchcell.datamodels.schema]]), so the family gains one L3 row, `evidence` (`evidence_result`), placed after `reference_viable`: every experiment and its reference carry one evidence type, the same on both; a `curated_null_phenotype` experiment states its SGD experiment type (`curated_experiment_type`), and a reference never does, since no call was made on it. Its details count the (experiment, reference) type pairs and the experiment types. The rule reads only the stored dicts, so it holds for any store of the family; SGD is its one member today.
