---
id: 7ano39yc7adq4oiv3aftgym
title: Compound_curator_audit_1045
desc: ''
updated: 1791809367971
created: 1791809367971
---

## 2026.10.12 - Four compound rows for Price 2018's MOPS rich recipe (#1045 row 16)

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_audit_1045.py`. Input list: `torchcell/datamodels/compound_identity_inputs/audit_1045_media.txt`. Results: `experiments/036-dataset-fixes-before-kg-build/results/compound_curator_audit_1045.json`.

`MOPS_RICH_DEFINED_NOCARBON_PRICE2018` (Price 2018 Supplementary Table 18, `si3.xlsx` sha256 `e5dbf3d5c97cfc12f49d7fd83f84bc95c16cbe963309a561fff20f442788b879`, rows 302-350) names 45 single reagents. 41 already resolved to a row with an InChIKey; these four did not, and were curated from PubChem by name with the committed curator and merged with the #726/#860 pass's additive `merge` (loaded from `compound_curator_726_860.py`, not reimplemented):

| label (canonical) | PubChem CID | InChIKey | ChEBI |
|---|---:|---|---|
| potassium hydroxide | 14797 | KWYUFKZDYYNOTN-UHFFFAOYSA-M | CHEBI:32035 |
| guanine | 135398634 | UYTPUPDQBNUYGX-UHFFFAOYSA-N | CHEBI:16235 |
| L-aspartic acid potassium salt | 23672742 | TXXVQZSTAVIHFD-DKWTVANSSA-M | none (several or none on PubChem) |
| L-cysteine hydrochloride monohydrate | 23462 | QIJRTFXNRTXDIP-JIZZDEOASA-N | CHEBI:91248 |

`L-aspartic acid potassium salt` is the label Table S18 prints; PubChem's name index returns the monopotassium salt (CID 23672742) first, and every monopotassium record shares the InChIKey above, so the stoichiometry the label leaves open is the one the index adopts.

Table: 5,723 to 5,727 records; `compound_identity_table_diff.py --old <origin/main table>` reports 4 keys added, 0 removed, 0 renamed, 0 lost, 0 identifier changes. `_TABLE_SHA256` re-pinned to `3042ca39de93cd381e1ef2b07494c3c575e75a05de820e757c8cb43487c09307`. No loader passes any of the four labels to the resolver today (searched the loaders and the committed bacterial gap inventory), so no other store's compounds change.
