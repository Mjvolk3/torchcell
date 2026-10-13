---
id: klxaoiryoft2oncvm0ordrq
title: Baba2006_growth
desc: ''
updated: 1791822584883
created: 1791822584883
---

## 2026.10.12 - Table 3's OD600 growth arm, loaded as released

Baba et al. 2006 (Mol Syst Biol 2:2006.0008, doi:10.1038/msb4100050, `babaConstructionEscherichiaColi2006`), issue #927, PR #1070. `GrowthYieldBaba2006Dataset`, slug `growth_yield_baba2006`, 11,735 records. The sibling [[torchcell.datasets.ecoli.baba2006]] loads the same table's construction outcomes.

**What is stored.** One `BacterialEnvironmentResponseExperiment` per released numeric cell of Table 3's column `8. cell growth (OD600)` (`msb4100050-s5.xls`, sha256 `a1a9c30b6cd703ff835f2809b420afe38b1c139865c9af87566a410669a30d2e`), `MeasurementType.optical_density` (added in this PR: an end-point absorbance, absolute, unnormalized), `AssayType.liquid_od_growth`. Per condition: LB 22 h 3,912, MOPS 24 h 3,912, MOPS 48 h 3,911.

**Identity as released.** `systematic_gene_name` is Table 3's MG1655 `b number`, stored as released in the `ecoli_k12_mg1655_bnumber` namespace with no crosswalk to BW25113; `perturbed_gene_name` is the released `gene`; `construction.strain_accession` is the Keio `JW id`. The JW id is what keeps apart the two strains Table 3 lists for each of b0621 (dcuC JW0613, JW0616), b2094 (gatA JW2078, JW2081) and b2218 (rcsC JW5917, JW5920).

**Conditions, from the Methods** (`paper/PMC1681482.1.txt`, sha256 `6e39bc2f...`): "Mutants were tested for growth in 200-μl LB medium with kanamycin as a rich medium in 96-well microplates ... and incubated for 22 h at 37°C without shaking." and "... from LB into 200-μl 0.4% glucose MOPS medium with 2 mM Pi (Wanner, 1994) and kanamycin as minimal medium and incubated for 24 and 48 h at 37°C without shaking." Kanamycin is the paper's stated 30 μg/ml. The rest of the MOPS composition is deferred by the paper to Wanner 1994, which is not mirrored. `aerobicity` carries the field's default; the static plates' oxygen regime is not stated and the schema has no gap for it.

**The reference.** BW25113 on the same medium without kanamycin (it carries no marker; the paper grows it "with or without antibiotics"), with its OD600 a `ProvenanceGap`: Table 3 has no BW25113 row, and the Results compare to wild type ("The vast majority showed no differences from wild type.") without releasing that reading. The verifier counts the three gapped references against a declared oracle.

**Not stored** (`preprocess/not_stored.json`): 73 obtained rows (219 readings) whose ECK, gene and b-number cells are all `none`, identified only by a JW id and W3110 coordinates; none of the 73 JW ids is on the pinned BW25113 annotation, and every bacterial gene-perturbation leaf keys on a locus tag, so there is no carrier (owner question in PR #1070). And rcsC JW5920's MOPS 48 h cell, released as `NT` and defined nowhere.

**Uncertainty.** One reading per mutant per condition; no per-value replicate count or dispersion is released (Figure 6's grey band is 2 x s.d. of the whole population), so the uncertainty fields and `n_samples` are gaps.

**Follow-up for the paper note.** Table 3 is also the Keio strain-to-gene table Campos 2018 lacked when it dropped 412 JW-labelled strains (#927).

### L0-L4, dev store built and verified under slurm (job 4387, task 2, source snapshot `0d8bc1f43`)

`$DATA_ROOT/data/torchcell/growth_yield_baba2006`: 11,735 records, 3 references. Verdict PASS.

| level | rule | result |
|---|---|---|
| L0 | structural | 11,735 records validate |
| L1 | count | 11,735 observed, 11,735 expected |
| L1 | pair_uniqueness | 11,735 unique (study, strain, condition); the JW id keeps the two strains of b0621, b2094, b2218 apart |
| L1 | provenance_gaps | 54,763 typed gaps over 11,735 records |
| L1 | canonical_gene_names | 3,909 systematic names, one spelling each |
| L2 | value_fidelity | 11,735 values finite |
| L2 | value_and_identity_fidelity_to_table_3 | 11,735 of 11,735 equal their released cell, b-number and gene |
| L3 | measurement_type_consistent | `optical_density` only |
| L3 | reference_zero (absolute branch) | 11,735 of 11,735 references a typed gap, as declared |
| L3 | environment_perturbed | 7,823 records on the modal (MOPS) medium carry no edit, as declared |
| L3 | media_membership | 11,735 on a medium deriving from a library medium (2 distinct media) |
| L3 | provenance_audit | 9 of 9 quotes verbatim in `paper/PMC1681482.1.txt` |
| L4 | released_b_numbers_on_mg1655_annotation | 3,832 of 3,909 released b-numbers are MG1655 gene rows, 77 are not; a finding, never a gate (the records keep the released numbers) |

## 2026.10.12 - Rebuilt on main after the rebase (slurm 4425)

The dev store was rebuilt from the rebased tree of PR #1070 (snapshot `accd94cfb`, slurm 4425, `--retire-existing --verify`) and verifies PASS with the same counts and the same L0-L4 rows as job 4387 above; the build manifest now records that tree's schema closure.
