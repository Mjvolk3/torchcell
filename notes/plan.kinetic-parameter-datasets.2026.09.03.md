---
id: ant6rfb35dtsoy8wlm6nlf2
title: Kinetic Parameter Datasets
desc: 'One module per kcat/Km inference model, mirroring the node-embedding dataset pattern, so model design stays something we experiment over'
updated: 1788412469488
created: 1788412469488
---

## 2026.09.03 - One module per predictor, on the node-embedding pattern

The requirement is flexibility rather than a single pipeline: which kinetic predictor
feeds the flux layer is a design choice to experiment over, so no predictor may be wired
in as the one true source. The sequence-embedding datasets already solve this exact shape
and the kinetic predictors should copy it rather than invent a second convention.

### The pattern being copied

| existing piece | role |
| --- | --- |
| `torchcell/data/embedding.py`, `BaseEmbeddingDataset(InMemoryDataset, ABC)` | abstract `initialize_model()` and `process()`, plus `__add__` so embeddings compose |
| `torchcell/datasets/esm2.py`, `nucleotide_transformer.py`, `fungal_up_down_transformer.py`, `one_hot_gene.py`, `codon_frequency.py` | one module per model, each with a registry of named variants |
| `torchcell/datasets/node_embedding_builder.py`, `NodeEmbeddingBuilder.EMBEDDING_CONFIGS` | name to class, so a config string selects a dataset |

A node embedding is materialized once per gene, cached, and assigned into the model at
construction. A kinetic parameter is the same kind of object: precomputed, cached, and
assigned, never called inside a forward pass.

### The one structural difference, and it drives everything

**A node embedding is keyed by GENE. A kinetic parameter is keyed by a PAIR.** A turnover
number is a property of one enzyme acting on one substrate, so the key is
`(uniprot, substrate_id, predictor, parameter)`. Collapsing to one row per gene would
silently pick a substrate, and collapsing to one row per catalytic unit would silently
pick a subunit.

Consequences that fall out of the key rather than being separate decisions:

- **Caching is on the ORF, permanently.** A well-defined systematic name plus a substrate
  id plus a predictor version is a stable key forever, so nothing is recomputed across
  runs. This is the author's standing instruction and it is what the pair key makes
  expressible.
- **Aggregation to a catalytic unit stays in the resolver, not the dataset.** The unit
  takes the minimum over its member genes, matching the availability softmin one level up.
  The dataset stores what was predicted; `resolve_kcat_table` decides what the layer sees.
- **Two parameters, not one.** RealKcat, UniKP and EITLEM emit both `k_cat` and `K_M`, so
  `parameter` is part of the key rather than two parallel stores. The `KcatPredictor`
  protocol in [[torchcell.metabolism.parameters]] already advertises this through
  `emits_km`.

### Proposed shape

```
torchcell/data/kinetics.py                  BaseKineticsDataset(ABC)
torchcell/datasets/kinetics/dlkcat.py       sequence + substrate SMILES
torchcell/datasets/kinetics/unikp.py        sequence + substrate SMILES, emits Km
torchcell/datasets/kinetics/eitlem.py       sequence + substrate SMILES, emits Km
torchcell/datasets/kinetics/turnup.py       sequence + full REACTION SMILES
torchcell/datasets/kinetics/deepenzyme.py   sequence + AlphaFold STRUCTURE + SMILES
torchcell/datasets/kinetics/boost_km.py     sequence + reactant SMILES, Km only
torchcell/datasets/kinetics_builder.py      KINETICS_CONFIGS, name -> class
```

The protocol needs widening to carry an optional reaction SMILES and an optional structure
path, since TurNuP and DeepEnzyme do not fit the two-argument signature. That is a
protocol change, not a per-model special case.

### Inputs are ready, so this is unblocked

Measured in [[experiments.026-metabolism-flux.scripts.kinetics_input_audit]] and mirrored
by [[experiments.026-metabolism-flux.scripts.fetch_kinetics_assets]]:

| input | coverage |
| --- | --- |
| protein sequence | 1,161 / 1,161 GEM genes |
| substrate SMILES | 3,552 / 3,728 catalytic units, 95.3 % |
| protein 3D structure | 1,161 / 1,161 accessions, 0 absent |

7,456 predictor input rows. Inference is minutes on one GPU per model, so all six can be
run and their outputs compared against each other. That comparison is the validity test,
and it removes the dependency on a published ranking we do not hold.

### What the 176 unready units are blocked by

86 distinct metabolites, and they are not one problem:

| category | count | can a SMILES predictor use it |
| --- | --- | --- |
| combinatorial lipid species, e.g. `phosphatidylcholine (1-16:0, 2-16:1)`, monolysocardiolipin variants | 68 | yes in principle; the SMILES is derivable from the acyl-chain notation, no database indexes that name |
| acyl-CoA isomers, e.g. `trans-2,cis-5-dodecadienoyl-CoA` | 8 | yes in principle, same naming problem |
| macromolecules and conjugates: `generic protein`, cytochrome c, `3-hydroxybutanoyl-ACP`, `Arg-tRNA(Arg)` | ~5 | **no, ever.** Not small molecules |
| plain name-join misses carrying ChEBI: `D-arabinose`, `D-fructose 1,6-bisphosphate` | ~4 | yes, a ChEBI lookup closes them |

The macromolecule row is the reason a coverage gate has to distinguish "missing but
obtainable" from "missing and unobtainable". Filed as issue #309.

### Open, and deliberately not decided here

Which predictor runs first. Wu reports no accuracy numbers for any of the eight, the
ranking lives in a Supplementary Table not in the mirror, and picking on impression would
be inventing one. Running all six and comparing sidesteps it.
