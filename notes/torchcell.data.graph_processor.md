---
id: px3gopqs0en6i4dhea8obmu
title: Graph_processor
desc: ''
updated: 1746906455219
created: 1746906455219
---

## 2026.09.30 - Incidence cache tied to its graph, one w_growth source, uniform keys (issue #527)

Previous behavior, pinned by Phase 14 of the test campaign:

- `IncidenceSubgraphRepresentation` and `LazySubgraphRepresentation` built the incidence cache once and returned early on every later call, whatever graph arrived. A second graph through the same processor read the first graph's edge positions and emitted node index -1.
- `SubgraphRepresentation` and `IncidenceSubgraphRepresentation` returned the stored `cell_graph["reaction"].w_growth` subset to the kept reactions, while `LazySubgraphRepresentation` returned a vector recomputed from a `reaction.subsystem` attribute. `to_cell_data` stores `w_growth` (1 for a Growth-subsystem reaction) and writes no `subsystem` attribute, so on every real cell graph Lazy returned all zeros. A tensor `subsystem` compared 0-d tensors with the string "Growth" and never marked anything.
- `Unperturbed` appended `label_statistic_name` unconditionally and raised `TypeError` on a phenotype with none (VisualScorePhenotype).
- `DCellGraphProcessor` wrote `perturbation_indices_batch` only when some perturbed gene was a node.
- `NeighborSubgraphRepresentation._add_phenotype_data` wrote nothing when no label value existed.

Fix:

- The cache records what it was built from: the gene count and the gene-gene `edge_index` tensors, the only inputs it depends on (`_gene_edge_indices`, `_incidence_source_matches`). Every call checks identity per edge type (O(1)); a tensor that is not the recorded object is compared by content and, if equal, the record is re-pointed so the next call is O(1) again. A different graph REBUILDS the cache rather than raising: the cache is a pure function of the graph, and one processor may legitimately serve several graphs. An in-place mutation of the recorded tensor is not detected (same object).
- `w_growth` has one source in all three masking processors: the stored tensor, subset to the kept reactions (Lazy keeps every reaction and returns the stored tensor by reference). The subsystem-derived computation, the only reader of `reaction.subsystem`, is removed, which retires the tensor-subsystem branch.
- `Unperturbed` appends the statistic field only when the class names one.
- DCell always writes `perturbation_indices_batch` (an empty long tensor when no perturbed gene is a node). Superseded below: this does not collate.
- The neighbor processor writes the [NaN] placeholder at type 0, sample 0, plus `phenotype_types`, like the subgraph processors.

Evidence: `test_a_cached_processor_rebuilds_its_cache_for_a_different_graph`, `test_the_cache_is_rebuilt_for_same_size_graphs_with_different_edges`, `test_every_masking_processor_returns_the_stored_w_growth`, `test_a_tensor_subsystem_does_not_touch_the_stored_w_growth`, `test_unperturbed_skips_a_phenotype_without_a_statistic`, `test_dcell_with_a_gene_outside_the_graph_writes_an_empty_batch_vector`, `test_neighbor_phenotypes_write_the_shared_placeholder` in [[tests.torchcell.data.test_graph_processor]], and `test_lazy_marks_invalid_reactions_but_keeps_all_of_them` in [[tests.torchcell.data.test_graph_processor_subgraph]]. The `label_policy.py` items of issue #527 are out of scope here and remain open.

## 2026.09.30 - DCell batch vector removed (issue #527 review)

The empty `perturbation_indices_batch` written in the previous section does not collate: PyG 2.8.0 `HeteroData.__inc__` computes `int(value.max()) + 1` for any key containing "batch", and `Batch.from_data_list` raises `max(): Expected reduction dim to be specified for input.numel() == 0` when one sample's tensor is empty. The DCell model never reads the per-sample key; the experiments collate with `follow_batch=["perturbation_indices"]`, which creates `perturbation_indices_batch` in the batch and handles an empty sample. The processor therefore writes no `perturbation_indices_batch` at all, and the record-order loop that built it (whose order also mispaired with the node-order `perturbation_indices`) is removed. Evidence: `test_dcell_with_a_gene_outside_the_graph_collates_through_follow_batch` collates an inside sample (YAL002W, node 1) with an outside one (YBR001C) and gets `perturbation_indices` [1] with batch [0].

## 2026.10.23 - Released gene symbols resolved at read time (`released_name_resolver`)

Since PR #1085 (#1045) a loader stores a deleted gene under the name its source released, and a source that names the gene by symbol puts that symbol in `systematic_gene_name` on `ReleasedIdentifierPerturbation`: 3,592 of Cachera 2023's 4,759 strains, and the non-systematic ORF column entries of Mulleder 2016, Messner 2023, Zelezniak 2018 and Nadal-Ribelles 2025. `Perturbation` matched that field exactly against `cell_graph["gene"].node_ids`, so such a record came out with no perturbed gene and no error (found by the 046 store build, jobs 6154 to 6156).

`Perturbation(released_name_resolver=ReleasedNameResolver(genome))` resolves every deletion-kind perturbation (a type ending in `deletion`, or `released_identifier`) whose name is not a gene node before the match, through `torchcell.datasets.gene_alias_resolution.ReleasedNameResolver`: strict (live name or a single candidate ORF), then the SGD standard-name rule for an ambiguous alias with exactly one standard-name owner, else a refusal. A refused perturbation stays unmapped and is listed with its reason on `processed_graph["gene"].unresolved_perturbations` (a list of strings, set only when a resolver is given) so a trainer can exclude the record explicitly. A gene addition or an allele is never resolved; a name that is already a node passes through. Without a resolver the output is unchanged, attribute for attribute. The resolver is typed through the `ReleasedNameResolverLike` protocol so this module does not import the datasets package. Over Cachera's symbols the 046 build measured 3,511 strict, 75 by the standard-name rule and 6 refusals (`AAD6`, `CRS5`, `FLO8`, `YLR287-A` not in R64; `PPA1`, `FEN1` ambiguous). Tests: `tests/torchcell/data/test_graph_processor_released_names.py`. Record of the finding: [[experiments.046-betaxanthin-omics-cotarget]].
