---
id: mga5gv6ksx504m08dqq18f7
title: Test_neo4j_preprocessed_cell_full_masks
desc: ''
updated: 1790979684025
created: 1790979684026
---

## 2026.10.01 - Full-mask store written by the 006 writer

Test file: `tests/torchcell/data/test_neo4j_preprocessed_cell_full_masks.py`, target [[torchcell.data.neo4j_preprocessed_cell_full_masks]].

### Fixture

The same five-gene build as [[tests.torchcell.data.test_neo4j_preprocessed_cell]]. Records are written by the real writer, `extract_full_masks` from `experiments/006-kuzmin-tmi/scripts/preprocess_lazy_dataset_full_masks.py`, imported with `importlib` while `dotenv.load_dotenv` and `logging.basicConfig` are stubbed, applied to the live `LazySubgraphRepresentation` item exactly as that script's loop does.

### Expected values

- Writer layout for genotype {1}: top keys `gene, node_masks, reaction, metabolite, edge_masks`; every mask uint8; gene `pert_mask` `[0,1,0,0,0]`, `x_pert` None.
- Loaded masks are bool and equal the hand-derived sets (same table as the compact note); node `pert_mask` is exactly `~mask`.
- Each item equals the live Lazy item except reaction `node_ids` and GPR `num_edges`, and equals the compact variant's item for the same build.
- `get(3)` raises `IndexError("Sample 3 not found in LMDB")` (the compact variant returns None).

### Findings

- Same two field differences as the compact store (`neo4j_preprocessed_cell_full_masks.py:204-207, 215`).
- Without a source, `get` fails with a bare `TypeError` on `None["gene"]` (line 177) instead of a refusal.
- Pointing the loader at a compact store only logs `Expected storage_type 'full_masks', got 'unknown'`; the first read then fails with `AttributeError: 'dict' object has no attribute 'to'`.
- No source fingerprint: a one-record source is served the old three-record store.

Lines 233-235 and 254-256 (non-`pert_mask` keys under reaction and metabolite) are left uncovered: the writer never stores such keys.

## 2026.10.05 - Audit 2 corrections

- Line cite corrected: live reaction ids come from `graph_processor.py:1724`.
- Reach: the two field differences are latent (see [[tests.torchcell.data.test_neo4j_preprocessed_cell]]); the store is read by 006 config 077; the no-fingerprint finding reaches 077; the storage-type finding reaches 077 only if it is pointed at a compact root; the no-source TypeError is latent (the script always passes a source).
- Hermeticity: `_load_writer` now stubs `load_dotenv` and `basicConfig` only around `exec_module`, then rebinds every module that captured the stub (dcell, yeast_GEM, sgd, kemmeren2014, sameith2015) to the real `load_dotenv`. `test_writer_import_leaves_no_stubbed_load_dotenv` checks this and fails (on `torchcell.models.dcell`) when the restore is removed, run in a fresh process.
- The storage-type mismatch match is anchored to the full message.
