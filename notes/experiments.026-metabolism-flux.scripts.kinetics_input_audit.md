---
id: q8a9k31f0en4040rzte2lzb
title: Kinetics_input_audit
desc: ''
updated: 1788407587370
created: 1788407587370
---

## 2026.09.02 - The kinetics bottleneck is the weights, not the inputs

The k_cat gap has been carried as a single number, 4.0 percent, since
[[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]] first
measured it. That number is the coverage of the Open Enzyme Database against the model's
3,728 catalytic units, and it has been read as the reason a sequence predictor is
required. It does not say how much of the model a predictor could actually reach, and
those are different quantities.

### Both predictor inputs were already on disk

yeast-GEM 9.0.2 ships them in `data/databases/`, and neither needs the network. This was
not obvious, and the sequence column in particular was sitting in the same table the
molecular weights are already read from by [[torchcell.metabolism.parameters]].

| file | contents | rows |
| --- | --- | --- |
| `swissprot.tsv` | uniprot, name, gene_id, ec_code, MW, **sequence** | 18,362 gene tokens carrying a sequence |
| `smilesDB.tsv` | metabolite name to SMILES | 894 names |

The `gene_id` column holds whitespace-separated synonyms with the standard name first
where one exists, as in `RMA1 YKL132C`. Indexing only the first token loses every gene
whose standard name precedes its systematic name, which is most of the annotated ones, so
every token is indexed.

### Measured coverage

| quantity | value |
| --- | --- |
| GEM genes with a protein sequence | **1,161 / 1,161 (100 %)** |
| catalytic units where every member gene resolves | 3,728 / 3,728 |
| catalytic units with at least one substrate SMILES | 3,262 / 3,728 |
| **units ready for a (sequence, substrate SMILES) predictor** | **3,262 / 3,728 (87.5 %)** |
| units with a measured k_cat, Open Enzyme Database | 148 / 3,728 (4.0 %) |
| predictor input rows | 6,829 |

A complex counts as ready only when every subunit resolves. Its turnover is the minimum
over its members, so a missing member makes that minimum undefined rather than smaller.
Substrates are the metabolites a reaction consumes, so products are never candidates.

The loss is entirely in the SMILES join, 894 named compounds against 2,806 metabolites,
and not in the sequences. Metabolites also carry ChEBI on 2,402 of 2,806 and MetaNetX on
2,251, so the 466 unready units are reachable through a second identifier route rather
than being structurally absent.

### What this changes

The predictors could take k_cat coverage from 4.0 percent to 87.5 percent, a factor of
22, and the inference itself is small: 6,829 pairs, minutes on one GPU for any of these
models. The bottleneck was never the scale of the inference or the availability of the
inputs. It is the model weights, and each predictor needs a wrapper satisfying the
`KcatPredictor` protocol that [[torchcell.metabolism.parameters]] already declares.

All six Wu Figure 3 repositories are reachable. Sizes and last push, checked 2026.09.02:
DLKcat 47 MB (2023.07.04), UniKP 24 MB (2025.12.30), TurNuP 3 MB (2023.09.21), Boost_KM
121 MB (2022.06.20), EITLEM-Kinetics 14 MB (2025.11.24), DeepEnzyme 30 MB (2024.09.12).

Which one to run first is not settled here. Wu reports no accuracy numbers for any of the
eight, the ranking would have to come from Supplementary Table 8, and that table is not in
the mirror. Picking one on impression would be inventing a ranking.
