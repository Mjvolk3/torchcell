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

### SMILES provenance has to be tiered, not a single column

A SMILES read out of a database is a fact. A SMILES assembled from a name is an
assumption, and the two must never share a column. Three tiers:

| tier | meaning | source |
| --- | --- | --- |
| `sourced` | read from a database keyed by name or id | `smilesDB.tsv`, MetaNetX `chem_prop.tsv` |
| `derived` | **assumed.** constructed from the metabolite name plus a resolved sibling | acyl-chain substitution, see below |
| `unrepresentable` | not a small molecule; no SMILES exists in principle | `generic protein`, cytochrome c, ACP- and tRNA-conjugates |

`unrepresentable` is a declaration, not a gap. It must reach the coverage gate (issue #309)
as a permanent incompatibility so those units are never silently filled with
`FALLBACK_KCAT_PER_S`.

### Deriving the lipid SMILES is tractable, and here is why

Two measurements make the 68 lipid names a bounded problem rather than an open one:

- **Only four acyl chains appear across every blocked lipid**: `16:0`, `16:1`, `18:0`,
  `18:1`, with counts 28, 64, 11 and 53. Those are palmitoyl, palmitoleoyl, stearoyl and
  oleoyl, the standard yeast set.
- **Several head groups already have RESOLVED siblings** carrying other chain
  combinations: diglyceride 44, phosphatidyl-L-serine 30, phosphatidylcholine 20,
  phosphatidylethanolamine 6, 1-acyl-sn-glycerol 3-phosphate 5.

So the missing combination is obtained by substituting an acyl chain into a real sibling
structure rather than by hand-writing a template. `rdkit` installs cleanly on this Python
(2026.3.5, cp313 wheel) and is what makes the substitution structural instead of textual.

**The assumption being made, stated once:** the name gives chain length and degree of
unsaturation but NOT double-bond position or geometry. `16:1` and `18:1` are taken as
Δ9-cis, which is what yeast predominantly makes. That is why these rows are `derived`.

Head groups with zero resolved siblings (phosphatidylglycerol, monolysocardiolipin,
CDP-diacylglycerol) need a from-scratch template and are a heavier assumption. Do those
second, and separately.

### Predictor readiness, measured 2026.09.03

All six cloned to `$DATA_ROOT/data/enzyme_kinetics/predictors/` (6.2 GB total).

| predictor | size | ships weights | hardcodes cuda | notes |
| --- | --- | --- | --- | --- |
| Boost_KM (`KM_prediction`) | 5.7 GB | 18 files | **0** | CPU-clean as-is; the obvious first one |
| UniKP | 68 MB | 2 files | 2 files | clean deps; also pulls ProtT5 from HuggingFace at runtime |
| TurNuP (`kcat_prediction`) | 11 MB | none | -- | weights need a separate download |
| DLKcat | 158 MB | none | -- | no requirements file; weights need a download |
| EITLEM-Kinetics | 133 MB | none | -- | weights need a download |
| DeepEnzyme | 181 MB | 2 files | 4 files | requires `apex` and pinned old numpy/rdkit; hardest |

**CPU is the right call and the repos support it.** Boost_KM has no CUDA references at
all. UniKP and DeepEnzyme touch `.cuda()` in a handful of files, which is a small patch to
a device argument. Running on CPU means all six can be brought up in parallel without
competing with a GPU sweep, and inference on 7,456 pairs is small enough that CPU is not
a constraint.

### The macromolecule row examined, and it is smaller and more fixable than first stated

These are **not bad annotations**. Each reaction genuinely has a macromolecule
participant, and that is the biology rather than a shortcut:

| reaction | what it does | why a macromolecule is correct |
| --- | --- | --- |
| `r_4239` arginyltransferase | moves Arg from Arg-tRNA onto a protein N-terminus | the substrate IS a protein; post-translational modification has no single substrate |
| `r_4152` cytochrome c heme lyase | attaches heme to apocytochrome c | protein in, protein out |
| `r_2145` / `r_2148` fatty acid synthase | reduces and dehydrates a growing acyl chain | the chain is covalently tethered to acyl carrier protein by design |
| `r_4325` Fe-S cluster assembly | forms a scaffold-desulfurase complex | a protein-protein complex |

**But yeast-GEM already performs exactly the substitution in question, inconsistently.**
`ferricytochrome c` and `ferrocytochrome c` both carry a SMILES in `smilesDB.tsv`, and
that SMILES is the **heme b macrocycle with `[Fe+2]`**, 130 characters, the prosthetic
group rather than the protein. The same file gives `Cytochrome c` nothing. That is an
annotation gap with the precedent sitting two rows away, not a chemical impossibility.
`r_4152` makes it starker: `ferroheme b` and `Apocytochrome c` both resolve, so two of
its three participants already have structures and only the holo form does not.

Revised verdict per species, with the units each blocks:

| species | units | substitutable | with what |
| --- | --- | --- | --- |
| `Cytochrome c` | 2 | **yes**, and the model does it for the siblings | the heme prosthetic group |
| `3-hydroxybutanoyl-ACP` | 1 | yes, chemically sound | the acyl-CoA analog; ACP and CoA share the phosphopantetheine thioester arm |
| `Arg-tRNA(Arg)` | 1 | partially | aminoacyl-adenosine, since the 3'-terminal ester is the reactive part |
| `generic protein` | 1 | **no** | deliberately generic, no structure exists |
| Fe-S scaffold and desulfurase | 2 | **no** | protein-protein complex |

No CoA analog of the ACP species is named in the model, so that proxy has to be
constructed rather than looked up, which puts it firmly in `derived`.

**Correction to the earlier framing.** Genuinely unrepresentable is roughly **3 units of
3,728**, not the five metabolites previously implied to matter. The macromolecules
together block under ten units. The 176 figure is overwhelmingly lipids, so the lipid
derivation is where essentially all of the remaining coverage is, and the macromolecule
tier matters for correctness of the gate rather than for coverage.

### Open, and deliberately not decided here

Which predictor runs first. Wu reports no accuracy numbers for any of the eight, the
ranking lives in a Supplementary Table not in the mirror, and picking on impression would
be inventing one. Running all six and comparing sidesteps it.
