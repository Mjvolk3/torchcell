# Thermodynamic curation closure

Built 2026-09-11T23:33:33.653441+00:00 by `experiments/026-metabolism-flux/scripts/thermo_curation_closure_summary.py`.

## Coverage before and after

| quantity | before | after |
|---|---:|---:|
| reactions priced, all 4131 | 3779 | 3801 |
| fraction, all reactions | 0.9148 | 0.9201 |
| reactions priced, chemical 3847 | not reported | 3535 |
| fraction, chemical only | not reported | 0.9189 |
| metabolites priced | 2561 | 2559 |
| non-chemical reactions, outside the denominator | 0 | 284 |
| reactions priced through cancellation | 0 | 22 |

## Ceiling ladder after, over the chemical reactions

| rung | reactions priced | fraction | flux carrying | fraction |
|---|---:|---:|---:|---:|
| now | 3535 | 0.9189 | 2724 | 0.9575 |
| structures and pka ladders resolved | 3545 | 0.9215 | 2732 | 0.9603 |
| everything chemically possible resolved | 3556 | 0.9244 | 2735 | 0.9613 |

## One row per curated record

| slug | kind | admission | species | decomposition | priced | closed by the record | closed by cancellation | what it would take |
|---|---|---|---:|---|---:|---:|---:|---|
| cytochrome_c | carrier_potential | measured | 2 | not attempted | 2 | 7 | 0 | nothing; every species this record covers is priced |
| mannan | polymer_convention | tabulated | 2 | refused | 2 | 3 | 0 | nothing; every species this record covers is priced |
| heme_a | structure | tabulated | 3 | refused | 0 | 1 | 0 | a group decomposition, or a measured formation energy; not more retrieval |
| heme_o | structure | tabulated | 1 | refused | 0 | 1 | 0 | a group decomposition, or a measured formation energy; not more retrieval |
| chitobiosyl_pp_dolichol | polymer_convention | tabulated | 1 | decomposes | 0 | 0 | 0 | a microspecies whose groups component contribution has seen: the structure decomposes and one of its groups is in the null space |
| cytochrome_b5 | carrier_potential | provisional | 2 | not attempted | 0 | 0 | 0 | a measured or tabulated source; the record is reported and never priced at admission=provisional |
| ethylnitronate | structure | tabulated | 1 | decomposes | 0 | 0 | 0 | a group decomposition, or a measured formation energy; not more retrieval |
| ferroheme_b | pka_ladder | provisional | 1 | refused | 0 | 0 | 0 | a measured or tabulated source; the record is reported and never priced at admission=provisional |
| glycogen | polymer_convention | tabulated | 2 | refused | 2 | 0 | 0 | nothing; every species this record covers is priced |
| gpi_intermediate | formula_correction | provisional | 1 | not attempted | 0 | 0 | 0 | a measured or tabulated source; the record is reported and never priced at admission=provisional |
| methionine_s_oxide | not_found | none | 3 | refused | 0 | 0 | 0 | a measured or tabulated source; the record is reported and never priced at admission=none |
| nitrophenyl_phosphate | structure | tabulated | 2 | mixed | 0 | 0 | 0 | a group decomposition, or a measured formation energy; not more retrieval |
| pectin | polymer_convention | tabulated | 1 | refused | 0 | 0 | 0 | a group decomposition, or a measured formation energy; not more retrieval |
| porphyrin | not_found | none | 2 | refused | 0 | 0 | 1 | a measured or tabulated source; the record is reported and never priced at admission=none |
| siroheme | structure | tabulated | 1 | refused | 0 | 0 | 0 | a group decomposition, or a measured formation energy; not more retrieval |
| triphosphate | pka_ladder | measured | 2 | refused | 0 | 0 | 1 | a group decomposition, or a measured formation energy; not more retrieval |

## What each record holds

| slug | value | sources | source |
|---|---|---:|---|
| cytochrome_c | E'0 272.0 mV vs SHE, n=1, Saccharomyces cerevisiae | 10 | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6177310/ |
| mannan | repeat unit C6H10O5, curated dGf'0 s_1107 -248.7, s_1108 -254.5 | 11 | https://www.ebi.ac.uk/chebi/backend/api/public/compound/CHEBI%3A27912/ |
| heme_a | SMILES [H]C(=O)C1=C(CCC(=O)O)C2=[N]3C1=Cc1c(C)c([C@@H](O)CC/C=C(\C)CC/C=C(\C)CCC=C(C)C)c4[n]1[Fe]31[N]3=C(C=c5c(C)c(CCC(=O)[O-])c([n]51)=C2)C(C=C)=C(C)C3=C4, formula C49H55FeN4O6 | 7 | https://www.ebi.ac.uk/chebi/backend/api/public/compound/61715/ |
| heme_o | SMILES C=CC1=C(C)C2=[N+]3C1=Cc1c(C)c(CCC(=O)[O-])c4[n]1[Fe-2]31[n]3c(c(C)c([C@@H](O)CC/C=C(\C)CC/C=C(\C)CCC=C(C)C)c3=C2)=CC2=[N+]1C(=C4)C(CCC(=O)[O-])=C2C, formula C49H56FeN4O5 | 7 | https://www.ebi.ac.uk/chebi/backend/api/public/compound/60530/ |
| chitobiosyl_pp_dolichol | repeat unit C36H64N2O17P2, no linkage stated | 14 | https://www.ebi.ac.uk/chebi/backend/api/public/compound/12427/ |
| cytochrome_b5 | E'0 5.5 mV vs SHE, n=1, Bos taurus | 10 | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4484759/ |
| ethylnitronate | SMILES CCN(=O)=O, formula C2H5NO2 | 11 | https://www.ebi.ac.uk/chebi/backend/api/public/compound/16268/ |
| ferroheme_b | pKa ladder [4.75, 4.15] | 9 | https://pubmed.ncbi.nlm.nih.gov/9666570/ |
| glycogen | repeat unit C6H10O5, curated dGf'0 s_0773 -244.3, s_0774 -302.3 | 2 | none |
| gpi_intermediate | formula -> C55H100N3O39P3R2 | 17 | file:///scratch/projects/torchcell-scratch/data/torchcell/yeast-GEM/yeast-GEM-9.0.2/model/yeast-GEM.xml |
| methionine_s_oxide | nothing found | 10 | https://pmc.ncbi.nlm.nih.gov/articles/PMC3766491/ |
| nitrophenyl_phosphate | per metabolite: s_3860=a structure, s_3859=none | 7 | file:///home/michaelvolk/.cache/equilibrator/group_definitions.csv |
| pectin | repeat unit C6H7O6, alpha-(1->4) glycosidic, the homogalactu | 18 | file:///scratch/projects/torchcell-scratch/data/torchcell/yeast-GEM/yeast-GEM-9.0.2/model/yeast-GEM.xml |
| porphyrin | nothing found | 9 | https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/66868/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,SMILES,ConnectivitySMILES,InChI,InChIKey,Charge,IUPACName,Title/JSON |
| siroheme | SMILES C[C@@]1(CC(=O)[O-])C2=CC3=[N+]4C(=Cc5c(CCC(=O)[O-])c(CC(=O)[O-])c6[n]5[Fe-2]45[N]2C(=CC2=[N+]5C(=C6)[C@@H](CCC(=O)[O-])[C@]2(C)CC(=O)[O-])[C@H]1CCC(=O)[O-])C(CCC(=O)[O-])=C3CC(=O)[O-], formula C42H36FeN4O16 | 11 | https://www.ebi.ac.uk/chebi/backend/api/public/compound/60052/ |
| triphosphate | pKa ladder [7.77, 5.4] | 10 | https://data.nist.gov/od/ds/mds2-2124/NBS_Tables_with_ErratumValues.csv |

## Reactions still open

330 reactions carry no standard energy. The full list, with the metabolites blocking each, is in `thermo_curation_closure.json`.

| reason | reactions |
|---|---:|
| blocked_by_metabolite | 312 |
| non_chemical | 18 |

## The accession route had no composition check

Of 2414 accession-priced metabolites, 2087 match the GEM's formula, 12 differ by whole C5H8 repeat units (offsets {'+1': 12}, family consistent True), and 8 were priced from a different molecule.

| metabolite | name | GEM formula | structure formula | shipped dGf'0 |
|---|---|---|---|---:|
| s_0773 | glycogen | C6H10O5 | C24H42O21 | -1127.2 |
| s_0774 | glycogen | C6H10O5 | C24H42O21 | -1370.9 |
| s_1578 | zymosterol intermediate 1c | C28H43O3 | C29H46O2 | 1257.2 |
| s_2920 | 3-oxo-cis-hexadec-9-enoyl-CoA | C37H62N7O18P3S | C37H60N7O17P3S | -575.1 |
| s_2923 | 3-oxo-cis-octadec-9-enoyl-CoA | C39H66N7O18P3S | C39H64N7O17P3S | -395.1 |
| s_4207 | trans-4-hydroxy-L-proline | C5H9NO3 | C5H4N4O4 | -119.0 |
| s_4295 | dimethylallyl phosphate | C5H9O4P | C8H8O | -512.4 |
| s_4313 | ITP | C10H11N4O14P3 | C4H4O4 | -2531.7 |

## The conserved-moiety route

| reaction | outcome | dGr'0 | sigma | why |
|---|---|---:|---:|---|
| r_0365 | untrusted_structure |  |  | a participant has no structure this route can trust |
| r_0436 | untrusted_structure |  |  | a participant has no structure this route can trust |
| r_0530 | priced | -440.54 | 3.17 |  |
| r_0531 | untrusted_structure |  |  | a participant has no structure this route can trust |
| r_0999 | moiety_does_not_cancel |  |  | the atoms no group covers do not match across the two sides: {'C': -2.0, 'Fe': 1.0, 'N': 3.0} |
| r_4159 | untrusted_structure |  |  | a participant has no structure this route can trust |
| r_4186 | moiety_does_not_cancel |  |  | the atoms no group covers do not match across the two sides: {'O': 1.0} |
| r_4197 | untrusted_structure |  |  | a participant has no structure this route can trust |
| r_4202 | moiety_does_not_cancel |  |  | the atoms no group covers do not match across the two sides: {'O': 1.0} |
| r_4220 | moiety_does_not_cancel |  |  | the atoms no group covers do not match across the two sides: {'O': -4.0, 'P': -1.0} |
| r_4273 | group_in_null_space |  |  | the group difference has an infinite-variance component; groups no training compound carries: ['-NO2 [H0 Z0 Mg0]'] |
| r_4403 | moiety_does_not_cancel |  |  | the atoms no group covers do not match across the two sides: {'O': -8.0, 'P': -3.0} |

## Carrier reactions the potential route refused

| reaction | why |
|---|---|
| r_0365 | untrusted_structure: a participant has no structure this route can trust |
| r_0436 | untrusted_structure: a participant has no structure this route can trust |
| r_0531 | untrusted_structure: a participant has no structure this route can trust |
| r_0999 | moiety_does_not_cancel: the atoms no group covers do not match across the two sides: {'C': -2.0, 'Fe': 1.0, 'N': 3.0} |
| r_4159 | untrusted_structure: a participant has no structure this route can trust |
| r_4186 | moiety_does_not_cancel: the atoms no group covers do not match across the two sides: {'O': 1.0} |
| r_4197 | untrusted_structure: a participant has no structure this route can trust |
| r_4202 | moiety_does_not_cancel: the atoms no group covers do not match across the two sides: {'O': 1.0} |
| r_4220 | moiety_does_not_cancel: the atoms no group covers do not match across the two sides: {'O': -4.0, 'P': -1.0} |
| r_4273 | group_in_null_space: the group difference has an infinite-variance component; groups no training compound carries: ['-NO2 [H0 Z0 Mg0]'] |
| r_4403 | moiety_does_not_cancel: the atoms no group covers do not match across the two sides: {'O': -8.0, 'P': -3.0} |

Elementally or charge imbalanced chemical reactions, diagnostic only: 205.
