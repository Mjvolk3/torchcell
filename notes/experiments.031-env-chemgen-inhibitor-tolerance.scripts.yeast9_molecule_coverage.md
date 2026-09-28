---
id: vp59n2zvpwbzpkjtqak93gh
title: Yeast9_molecule_coverage
desc: ''
updated: 1790485736407
created: 1790485736407
---

## 2026.09.27 - Yeast9 coverage: the encoders are fine, the structures are not

Measured to decide whether Yeast9 metabolites can share the molecule embedding space with the
dosed inhibitors and media components. Model is yeast-GEM 9.0.2.

**Structure inventory.** 2,806 compartment-specific metabolite entries collapse to 1,378
distinct species. Only **894 of 1,378, 64.9%, carry a SMILES**, and every one of those comes
from a single source, the release's own name-keyed `smilesDB.tsv`. **Zero species carry an
InChI**, so every InChIKey used here is RDKit-derived from the shipped SMILES rather than
sourced. The repo's own compound identity table adds nothing: it matches 100 species and all
100 were already in the release table.

Two quality problems in those 894, both load-bearing for a shared space. Every SMILES is
**stereochemically flat**, with 621 of 894 carrying at least one unassigned stereocenter, so L
and D forms collapse and 894 species yield only 834 distinct keys. And **36 of 894 do not encode
the species the model declares**, judged by heavy-atom counts against the model's own formula.
Two are outright wrong molecules: tetrahydrofolate is given tetrahydrofuran's SMILES, and a
halide is given hypoxanthine's.

**The encoders are not the bottleneck.** Eleven of the twelve embed all 894 with zero failures.
Uni-Mol reaches 836 of 894, its 58 refusals being 11 monatomic ions with no conformer and 47
planar or linear molecules plus the inositol-phosphoceramide sphingolipids and long polyprenyl
diphosphates. The RDKit descriptor block emits NaN on 18 rows, so a caller must impute before
taking any distance.

**The 484 species with no SMILES split into two very different halves.**

- **167 have no single structure and never will**: 44 tRNA species, 24 generic R-group species,
  23 pooled lipid classes, 22 protein species, 22 KEGG glycan accessions, 16 acyl-carrier-protein
  thioesters, 11 pseudo-metabolites such as biomass and protein, 5 with no formula. These need a
  non-molecular representation regardless of curation effort.
- **317 have a definite structure that is simply absent from the release table**, dominated by
  258 acyl-resolved lipids plus 56 ordinary molecules such as the enoyl-CoA series, fructose
  1,6-bisphosphate and dolichol. 295 of the 484 carry a ChEBI, KEGG or MetaNetX identifier, so
  the lookup route exists.

**Reaction-level ceiling for anything structure-based: 2,156 of 4,131 reactions, 52.2%, have a
SMILES for every participant.** 80% have at least one. The gap is the lipid and biomass pools.

**Property predictors are mostly not implemented here.** Runnable in this worktree: kcat and KM
from the Open Enzyme Database, covering 148 of 3,728 catalytic units at 3.97%, keyed on UniProt
accession rather than structure; and a Gibbs lookup from the model's shipped CSVs covering 85.1%
of metabolites and 77.7% of reactions with no live prediction and no uncertainty. Not implemented
at all: eQuilibrator, which appears only in docstrings saying it is absent, every pKa predictor,
and CatPred. A fully-run version of all of this exists on the branch
`feat/kinetics-equilibrator-datasets`, snapshotted under `$DATA_ROOT/job-snapshots/`, where
eQuilibrator-backed pricing reaches 3,801 of 4,131 reactions at 92%. None of those scripts are in
this worktree.

**The overlap question, recomputed against the full five-dataset compound set.** The agent's
first pass used a stale 343-key archive while a rewrite was in flight; against the current 5,385
dosed compounds the numbers are larger and the conclusion is more favorable.

| against | their n | exact key shared | skeleton shared |
|---|---|---|---|
| dosed compounds, five datasets | 5,385 | 49 | 103 |
| media components | 37 | 5 | 30 |

So **103 Yeast9 metabolites are dosed as exogenous compounds somewhere in the five datasets**,
12.4% of the structure-bearing metabolites and 2.0% of the dosed panel. They are central
cofactors and metabolites, NADPH, NAD, amino acids, spermidine, riboflavin, nicotinamide,
ascorbate, ergosterol, lanosterol, farnesol. That is a real bridge rather than the near-disjoint
picture the stale count suggested, because it means the data contains cases of dosing a
metabolite the metabolic model already represents.

**The media axis is nearly a subset of the metabolite axis**, 30 of 37 components, which is
expected since media components are metabolites by construction and is the strongest argument
for one shared space.

**One join rule follows from the stereo problem.** Yeast9 keys are stereo-flat and the curated
compound keys carry stereo, so any join between the metabolite and inhibitor axes must be on the
connectivity skeleton with the stereo ambiguity recorded. At least one skeleton match is
spurious: a beta-glucan matches glucose only because its SMILES is the monomer.

Result files: `results/yeast9_structure_inventory.csv`, `results/yeast9_structure_summary.csv`,
`results/yeast9_failure_classes.csv`, `results/yeast9_encoder_coverage.csv`,
`results/yeast9_overlap_summary.csv`, `results/yeast9_embeddings/`
