---
id: sz8r7xtbyhl1r8oy5p6lpw6
title: Hetero_cell_bipartite_dango_gi
desc: ''
updated: 1748996654673
created: 1748977996389
---
## 2025.06.03 - Data Updated

`torchcell/scratch/load_batch_005.py`

```python
dataset_hetero.cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
  },
  metabolite={
    num_nodes=2806,
    node_ids=[2806],
  },
  reaction={
    num_nodes=7122,
    node_ids=[7122],
    w_growth=[7122],
  },
  (gene, physical, gene)={
    edge_index=[2, 144211],
    num_edges=144211,
  },
  (gene, regulatory, gene)={
    edge_index=[2, 44310],
    num_edges=44310,
  },
  (reaction, rmr, metabolite)={
    hyperedge_index=[2, 26325],
    stoichiometry=[26325],
    num_edges=26325,
    reaction_to_genes=dict(len=4881),
    reaction_to_genes_indices=dict(len=4881),
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5450],
    num_edges=4881,
  }
)
dataset_hetero[0]
HeteroData(
  gene={
    node_ids=[6604],
    num_nodes=6604,
    ids_pert=[3],
    perturbation_indices=[3],
    x=[6604, 0],
    x_pert=[3, 0],
    phenotype_values=[1],
    phenotype_type_indices=[1],
    phenotype_sample_indices=[1],
    phenotype_types=[1],
    phenotype_stat_values=[1],
    phenotype_stat_type_indices=[1],
    phenotype_stat_sample_indices=[1],
    phenotype_stat_types=[1],
    pert_mask=[6607],
  },
  reaction={
    num_nodes=7122,
    node_ids=[7122],
    w_growth=[7122],
    pert_mask=[7122],
  },
  metabolite={
    node_ids=[2806],
    num_nodes=2806,
    pert_mask=[2806],
  },
  (gene, physical, gene)={
    edge_index=[2, 144100],
    num_edges=144100,
    pert_mask=[144211],
  },
  (gene, regulatory, gene)={
    edge_index=[2, 44289],
    num_edges=44289,
    pert_mask=[44310],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5450],
    num_edges=5450,
    pert_mask=[5450],
  },
  (reaction, rmr, metabolite)={
    hyperedge_index=[2, 26325],
    stoichiometry=[26325],
    num_edges=26325,
  }
)
batch_hetero
HeteroDataBatch(
  gene={
    node_ids=[2],
    num_nodes=13208,
    ids_pert=[2],
    perturbation_indices=[6],
    perturbation_indices_batch=[6],
    perturbation_indices_ptr=[3],
    x=[13208, 0],
    x_pert=[6, 0],
    phenotype_values=[2],
    phenotype_type_indices=[2],
    phenotype_sample_indices=[2],
    phenotype_types=[2],
    phenotype_stat_values=[2],
    phenotype_stat_type_indices=[2],
    phenotype_stat_sample_indices=[2],
    phenotype_stat_types=[2],
    pert_mask=[13214],
    batch=[13208],
    ptr=[3],
  },
  reaction={
    num_nodes=14243,
    node_ids=[2],
    w_growth=[14243],
    pert_mask=[14244],
    batch=[14243],
    ptr=[3],
  },
  metabolite={
    node_ids=[2],
    num_nodes=5612,
    pert_mask=[5612],
    batch=[5612],
    ptr=[3],
  },
  (gene, physical, gene)={
    edge_index=[2, 288178],
    num_edges=[2],
    pert_mask=[288422],
  },
  (gene, regulatory, gene)={
    edge_index=[2, 88574],
    num_edges=[2],
    pert_mask=[88620],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 10899],
    num_edges=[2],
    pert_mask=[10900],
  },
  (reaction, rmr, metabolite)={
    hyperedge_index=[2, 52645],
    stoichiometry=[52645],
    num_edges=[2],
  }
)
```

## 2025.06.03 - Detailed View of Data For Indexing Based on Phenotype Type

For printing

```python
dataset_hetero[0]['gene'].phenotype_values
dataset_hetero[0]['gene'].phenotype_type_indices
dataset_hetero[0]['gene'].phenotype_sample_indices
dataset_hetero[0]['gene'].phenotype_types
dataset_hetero[0]['gene'].phenotype_stat_values
dataset_hetero[0]['gene'].phenotype_stat_type_indices
dataset_hetero[0]['gene'].phenotype_stat_sample_indices
dataset_hetero[0]['gene'].phenotype_stat_types
```

```python
dataset_hetero[0]['gene'].phenotype_values
tensor([-0.0588])
dataset_hetero[0]['gene'].phenotype_type_indices
tensor([0])
dataset_hetero[0]['gene'].phenotype_sample_indices
tensor([0])
dataset_hetero[0]['gene'].phenotype_types
['gene_interaction']
dataset_hetero[0]['gene'].phenotype_stat_values
tensor([0.1059])
dataset_hetero[0]['gene'].phenotype_stat_type_indices
tensor([0])
dataset_hetero[0]['gene'].phenotype_stat_sample_indices
tensor([0])
dataset_hetero[0]['gene'].phenotype_stat_types
['gene_interaction_p_value']
```

## 2026.09.29 - Findings from the Phase 10 tests (not fixed)

Pinned in [[tests.torchcell.models.test_hetero_cell_bipartite_dango_gi]]: the conv wrapper's PyG `LayerNorm` runs in graph mode without a batch vector (lines 654 to 655), so an eval-mode prediction depends on the other samples in its batch (norm "batch" removes the dependence); `PreProcessor` shares one norm module across layers (lines 572 to 582); every activation but "relu" is SiLU (line 571) and so is the wrapper's `activation=None` (line 645); a one-layer GIN outputs `gin_hidden_dim` (lines 703 to 706); the GATv2 init branch is dead (lines 877 to 893); an unknown `combination_method` fails only at the end of the first forward (lines 847 to 855, 1112); local scores go to sample 0 when neither pointer field is present (lines 1019 to 1051) and to the wrong samples when the last sample has no perturbations (lines 1043 to 1050); a batch without `pert_mask` raises `IndexError` (lines 916 to 919). Coverage 0% to 39.6%; `main()` is untested.

## 2026.09.30 - Fix: Wasserstein loss keys, finiteness checks, activation, aggregation dropout, seeding (issue #540)

Previous behavior and fix, with the tests in [[tests.torchcell.models.test_hetero_cell_bipartite_dango_gi]] that now assert each contract:

- **Loss keys.** The training loop read `loss_dict["weighted_dist"]` and `loss_dict["dist_loss"]` for every composite loss, but `MleWassSupCR` emits `weighted_wasserstein` and `wasserstein_loss`, so `loss: mle_wass_supcr` (the shipped `hetero_cell_bipartite_dango_gi.yaml`) raised `KeyError` in epoch 1. The loop now picks the distribution-term name per loss (`wasserstein` for `MleWassSupCR`, `dist` for `ICLoss` and `MleDistSupCR`), prints `wasserstein=` for the Wasserstein loss, and stores the value under the generic `weighted_dist` / `dist_loss` history keys the plots read. The loss keys themselves are unchanged. Test: `test_main_with_the_shipped_wasserstein_loss_prints_its_own_components` (the epoch-1 line equals a fresh `MleWassSupCR` on a `_tiny(seed=0)` forward).
- **Finiteness checks.** All 15 stage checks were `torch.isnan(x).any()`, so +/-inf passed and a single +inf head under concat returned inf with no error. They are now `not torch.isfinite(x).all()` with the same `RuntimeError`, message `NaN or inf detected in <stage>`. Test: `test_an_infinite_parameter_is_named_by_the_first_stage_it_reaches`.
- **GIN width.** A GIN `nn.Sequential` with no `nn.Linear` raised `UnboundLocalError`; it now raises `ValueError("AttentionConvWrapper cannot infer the output width of GINConv: its nn.Sequential contains no nn.Linear")`. Test: `test_wrapper_around_a_gin_mlp_with_no_linear_is_refused_by_name`.
- **Activation.** Every name but "relu" built SiLU (the 006 configs ask for "gelu"). `get_activation` now maps the name through `torchcell.models.act.act_register` (as `deep_set`, `early_cell_diffpool_dense` and `cell_latent_perturbation_*` do) and refuses an unregistered name with a `ValueError` listing the registered ones. `AttentionConvWrapper(activation=None)` is now Identity, matching `norm=None`. Note: a model trained before this fix under `activation: gelu` actually ran SiLU, so its checkpoint loads into the same shapes but computes differently.
- **Aggregation dropout.** `graph_aggregation_config.dropout` was overwritten by the model dropout; the model dropout is now only the default and the config's own value wins. Test: `test_shipped_006_encoder_flags_reach_the_modules_they_name`.
- **Seeding.** `main` never seeded. It now calls `L.seed_everything(int(cfg.get("seed", 42)), workers=True)` (the `experiments/025-solid-growth` pattern; 42 matches the constant in `experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi.py`), and the 006 config carries `seed: 42`. Test: `test_main_at_lr_zero_plots_on_schedule_and_reports_the_untrained_model` (caller RNG at 123, config seed 0, metrics equal `_tiny(seed=0)`).
- **Final components figure.** Only "icloss" saved `loss_components_evolution`; all three composite losses now do, titled by the loss label.
- **Left open.** `graph_aggregation_config.aggregation_norm` is still read by nothing here (only the `_lazy` variant builds a norm from it); wiring it adds a norm module to the aggregators and changes the parameter count and architecture, a design change rather than a one-liner.

## 2026.10.01 - aggregation_norm is refused instead of ignored

- **Previous behavior.** `graph_aggregation_config.aggregation_norm` was passed into `HeteroConvAggregator`'s config dict and read by nothing, so any value built the same module and every run of this module trained with no aggregation norm (issue #540, last open item).
- **Fix.** `aggregation_norm` is now an explicit argument of `HeteroConvAggregator` (default None), and `GeneInteractionDango` pops it out of a copy of `graph_aggregation_config` (absent means None) and forwards it. None builds exactly the module of 8774914fe (same parameter count, same state_dict keys). Any other value raises `AggregationNormNotImplementedError` (a `ValueError`) whose message names the value and points at the `_lazy` variant, whose `pairwise_interaction` aggregator builds a norm. Refusal rather than an implementation: this module's `PairwiseGraphAggregation` is a different architecture from the `_lazy` one (no identity option, fixed-width MLPs), so a norm here would match neither module's history and would change parameter counts and checkpoints.
- **Configs.** Composing every config in `experiments/006-kuzmin-tmi/conf/` with hydra (164 files, 163 parseable; no other experiment sets the key), 29 composed configs carry `aggregation_norm: "layer"` on main: 23 set it literally and 6 inherit it through `defaults:`. Only two reach this module's defaults:
  - `hetero_cell_bipartite_dango_gi.yaml` is the hydra default of this module's `main` and of `experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi.py`. Its method has been "sum" since 2025-10-23, before the key was added on 2025-10-31 (8b68e55d5), so neither module ever built a norm from it. It is now `null` with a comment saying so.
  - `hetero_cell_bipartite_dango_gi_mle_test.yaml` inherits that default, so its composed value flips from "layer" to null with this change. It is "sum" and has no launcher, and the build is unchanged: composed and built on a two-gene stand-in multigraph with the config's graph names, both this module and `_lazy` give 932,064 parameters and 344 state_dict keys on main and on the change (the review measured 157,323 and 113 with its own harness; the point, equal before and after, holds in both).
  - The other 27 feed only the `_lazy` model and are unchanged: 22 literal (20 `pairwise_interaction`, where `_lazy` builds the LayerNorm; `gh_080` and `gh_profile_v5` are "sum", where `_lazy` ignores it) and 5 inheriting from `gh_085`/`gh_086` (`gh_085_dataloader_profile`, `gh_086`, `gh_086_dataloader`, `gh_086_model` under `_lazy.py` launchers or as their base, and `neighbor_subgraph_gh_087_dataloader` with no launcher).
- **Pre-existing defects seen on main, not fixed here.** `hetero_cell_bipartite_dango_gi_040.yaml` is invalid YAML (hydra raises `ScannerError`; line 134) although `gh_hetero_cell_bipartite_dango_gi-ddp_040.slurm` launches it. The `gh_hetero_cell_bipartite_dango_gi_lazy-ddp_083.slurm` script path is truncated (`.../scripts/o`), so which script ran `gh_083` cannot be read from the repo.
- **Evidence.** Tests `test_null_aggregation_norm_builds_exactly_the_module_main_built`, `test_a_non_null_aggregation_norm_is_refused_by_name` ("layer", "batch", "none" and the falsy "", False, 0; a truthiness mutant of the check fails 9 cases), `test_main_passes_a_layer_aggregation_norm_through_to_the_refusal`, `test_main_with_a_null_aggregation_norm_builds_the_main_pairwise_model`.
