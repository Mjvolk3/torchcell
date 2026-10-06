---
id: qtgxxuzgbjjk0kmxtn3gqvt
title: Test_esm2
desc: ''
updated: 1790780543404
created: 1790780543404
---

## 2026.09.30 - Phase 18: the ESM2 dataset on a faked backbone

New file, five tests (9 cases), 25 to 88 percent alone. The shared `embedding_genome` conftest fixture holds three real `SCerevisiaeGene` objects on a 6,100 nt chromosome from `random.Random(1809)`; the `Esm2` class is monkeypatched to a stand-in whose `embed` is a cumulative sum of [len, #M, #K, #*] (`MKPG*` gives [5, 6, 7, 8]) so no weights are ever loaded. Pinned: the backbone name per dataset name, which genes are embedded versus zeroed per exclusion variant, the stored value on the exact sequence, caching (a second construction never builds the backbone), chunked saves equal to a single chunk, the refusal for an unknown name.

Findings: `ds["<gene>"]` returns a scalar slice of the wrong gene because the squeeze at line 165 makes each embedding 1-D and the collate flattens them (YAL002C gets YAL001W's second component); the `_no_dubious_uncharacterized` variant lists the classes in lowercase (29-32) so it never matches SGD's values and excludes nothing; an unknown name raises a bare `KeyError` at 102 rather than the base class's `ValueError`.

## 2026.09.30 - Findings retired (issue #543)

Retired: flat collate and wrong-gene lookup, lowercase exclusion list, bare `KeyError`. Now asserted: every stored row is `[1, 4]` float32 and the collate is `[3, 4]`; `ds["<gene>"]` and `ds[i]` return the same exact row; `CellDataset.create_embedding_graph` gives each gene's exact `[4]` vector; `_no_dubious_uncharacterized` zeroes YAL002C and YAL003W and embeds only YAL001W; an unknown name is the base `ValueError` with the full valid list; `model_name=None` builds no backbone and writes only PyG's markers.

## 2026.10.06 - Live embedding-graph consumer

`torchcell/datasets/cell.py` retired with the legacy cluster ([[torchcell.legacy-retirement.2026.10.06]]), so the node-feature check in `test_lookup_by_gene_id_and_by_index_return_the_same_row` now calls the live `torchcell.data.neo4j_cell.create_embedding_graph`. That function min-max normalizes each feature in place first, so the pinned nodes are YAL001W [1, 1, 0.5, 0.5], YAL002C [0, 0, 0, 0], YAL003W [1, 1, 1, 1]. It runs on a fresh `Esm2Dataset` over the same store: on the dataset whose three items were already read, PyG served its cached items and the nodes kept the raw rows (observed in the first run of the rewrite).
