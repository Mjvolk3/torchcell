---
id: 0ej8nrbigdad0o2azas226s
title: Cell_graph_transformer_metabolism
desc: ''
updated: 1785040422141
created: 1785040422142
---

## 2026.07.25 - CGT-Metabolism model class (Track A heads)

`torchcell/models/cell_graph_transformer_metabolism.py`; tests
`tests/torchcell/models/test_cell_graph_transformer_metabolism.py` (7 passing).

The model class the metabolism work of [[plan.cgt-metabolism.2026.07.25]] is built inside.
Track A activates only the readout heads; no flux layer.

### Reuse by inheritance, not transcription

`CellGraphTransformerMetabolism` SUBCLASSES
[[torchcell.models.equivariant_cell_graph_transformer]]'s `CellGraphTransformer` and
delegates the whole forward to it, so the encoder, the graph-regularized attention and the
equivariant PERT operator are literally the parent's implementation. A transcription of
those ~900 lines could drift; an inherited one cannot.

What inheritance does NOT give for free is **initialization parity**. Parameters created in
a subclass `__init__` consume the global RNG stream, so a head constructed in the wrong
place shifts every encoder weight at a given seed and silently invalidates any comparison
against a Fig-3 run. Every metabolism head is therefore built strictly AFTER
`super().__init__()` returns, and the test asserts:

- with heads disabled, parent and fork have **bit-identical `state_dict()`** and
  bit-identical `predictions` / `h_CLS` / `H_genes` / `H_genes_pert` / `graph_reg_loss`
  (`atol=0, rtol=0`) at seed 17;
- with all three heads ACTIVE, every inherited parameter still matches the parent's, and
  `H_genes_pert` is unchanged.

### The three heads

Declared through `heads_config` with an explicit `kind`; there is no default, because
guessing scalar-vs-vector is exactly the mistake that lets a scalar target broadcast across
a vector head.

| head | class | shape | target |
| --- | --- | --- | --- |
| `betaxanthin` | `ProductScalarHead` | `[B, 1]` | Cachera CRI-SPA fluorescence (centred at 0) |
| `beta_carotene` | `ProductScalarHead` | `[B, 1]` | Ozaydin colony-colour ordinal, -5..+5 |
| `mulleder19` | `MetabolomeVectorHead` | `[B, 19]` | Mulleder amino acids, mM, fixed key-sorted columns |

`ProductScalarHead.output_dim` is fixed at 1 by construction and `MetabolomeVectorHead`
rejects `output_dim < 2`, so the scalar/vector distinction cannot be violated by config.
The three units are mutually incomparable, so they must not share a head; the test asserts
their parameter sets are pairwise disjoint.

`mulleder19` needs **no Yeast9 metabolite alignment**: the 19 columns are dense in every
record and fixed, which is what lets Track A skip the `target_metabolite_ids` /`col_idx`
gap entirely.

At hidden 24 x 3 layers the whole model is **191,541 parameters**, of which 158,568 are the
6,607 x 24 learnable gene embedding.
