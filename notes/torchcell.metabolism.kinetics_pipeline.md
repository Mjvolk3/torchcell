---
id: 5z64m8jf2eca4qxpt2jj456
title: Kinetics_pipeline
desc: ''
updated: 1789100320065
created: 1789100320065
---

## 2026.09.10 - From a GEM plus a proteome to per-unit k_cat and K_M

The portable path. Takes a GEM, a proteome and a set of mirrors and emits per-catalytic-unit
k_cat and K_M with a tier, the predictor set, provenance and a blocker enum for every unit it
cannot resolve. Nothing in it names an organism, a gene, a metabolite or a predictor.

Four stages. **Sequence**: gene to accession and protein sequence from a proteome the caller
supplies. **Structure**: substrate metabolite to SMILES through a name table, a direct
annotation id, a cross-reference to a current id, and externally supplied structures, in that
order, each accepted only when RDKit's formula for the SMILES equals the GEM's formula with
both neutralized. That test is the whole game: a wrong SMILES is worse than no SMILES,
because a predictor returns a confident number for the wrong molecule and nothing downstream
can tell. It also rejects every R-group placeholder for the right reason, since `R` never
equals `*`. **Measured**: an optional first tier keyed by `MeasuredKey`, either the accession
or EC number plus substrate plus organism; the three measured tiers are tagged apart because
a measurement of this protein, of a same-organism enzyme doing the same chemistry, and of a
homolog in another organism are three different claims. **Predicted**: the median in log10
over the predictors a within-class band study admits, which is a computed stage so another
organism re-decides the set from its own statistics.

`resolve_kinetics(config)` runs all four and returns one record with the tables, the tier
counts, the blocker counts and the sha256 of every file read. CLI:
`python -m torchcell.metabolism.kinetics_pipeline --config <json> --out <json>`.

Test: `tests/torchcell/metabolism/test_kinetics_pipeline.py`, a four-reaction synthetic GEM
with synthetic mirrors, one reaction per blocker class. Yeast test case:
[[experiments.026-metabolism-flux.scripts.kinetics_pipeline_yeast]]. Full accounting:
[[experiments.026-metabolism-flux.kinetic-coverage-closure]].

## 2026.09.11 - A second database stage

`KineticsPipelineConfig` gained `homolog_key` and `homolog_mirror_dir`, so the four-tier
cascade the flux layer runs is expressible here too: an accession-keyed, substrate-matched
measurement of this organism's protein, then a cross-organism EC-plus-substrate measurement,
then predictions, then the constant. The primary stage always wins where it has an answer.
The organism default has one definition shared with
`torchcell.metabolism.parameters._organism_default`: the median of the values the database
tiers placed in the table, per catalytic unit.
