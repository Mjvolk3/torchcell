---
id: 4fkcflbfsazgd891005btds
title: Hetero_cell_nsa_retry
desc: ''
updated: 1759878653404
created: 1759878651283
---
`torchcell/experiments/006-kuzmin-tmi/conf/hetero_cell_nsa_retry.yaml`

```yaml
defaults:
  - default
  - _self_

wandb:
  project: torchcell_006-kuzmin-tmi_hetero_cell_nsa_retry_test
  tags: []

cell_dataset:
  # graphs: [physical] # runs
  # graphs: [physical, regulatory] # runs
  # graphs: [physical, regulatory, tflink] # runs
  # graphs: [physical, regulatory, tflink, string12_0_neighborhood] # runs
  # graphs: [physical, regulatory, tflink, string12_0_neighborhood, string12_0_fusion] # runs
  # graphs: [physical, regulatory, tflink, string12_0_neighborhood, string12_0_fusion, string12_0_cooccurence]
  graphs:
    [
      physical,
      regulatory,
      tflink,
      string12_0_neighborhood,
      string12_0_fusion,
      string12_0_cooccurence,
      string12_0_coexpression,
      string12_0_experimental,
      string12_0_database,
    ]
  node_embeddings:
    - learnable
  learnable_embedding_input_channels: 64
  incidence_graphs: [metabolism_bipartite]

profiler:
  is_pytorch: true

# Transform configuration
transforms:
  use_transforms: true
  forward_transform:
    normalization:
      gene_interaction:
        strategy: "standard" # Options: standard, minmax, robust

data_module:
  is_perturbation_subset: false
  perturbation_subset_size: 1000 #2.5e4
  batch_size: 2 #32 #4 #16
  num_workers: 1 #2
  pin_memory: true
  prefetch: false
  prefetch_factor: 2

trainer:
  max_epochs: 300
  strategy: ddp #ddp #ddp_find_unused_parameters_true #auto # ddp
  num_nodes: 1
  accelerator: gpu
  devices: 4
  overfit_batches: 0
  precision: "bf16-mixed" # "32-true", "16-mixed", "bf16-mixed"

# Compilation settings for PyTorch 2.0+
compile_mode: null # null, "default", "reduce-overhead", "max-autotune"

# ------ Placeholders for hydra optuna
heads: 8 # placeholder
norms: "layer" # placeholder
# ------

model:
  gene_num: 6607
  reaction_num: 7122
  metabolite_num: 2806
  hidden_channels: 64
  out_channels: 1 # Only gene interaction
  attention_pattern: ["M", "S"] # NSA attention pattern
  norm: ${norms}
  activation: "relu"
  dropout: 0.0 # 0.1
  heads: ${heads}
  prediction_head_config:
    hidden_channels: 64
    head_num_layers: 4
    dropout: 0.0
    activation: "relu"
    head_norm: ${norms}

regression_task:
  loss: mle_dist_supcr # Options: logcosh, icloss, mle_dist_supcr
  is_weighted_phenotype_loss: false
  # Lambda weights for loss components
  lambda_mse: 1.0
  lambda_dist: 100.0 # 0.1 # 0.1 #0.1 #0.1
  lambda_supcr: 0.0 #0.001 #0.001 #0.001 #0.001

  # MleDistSupCR specific configuration
  loss_config:
    # Component-specific parameters
    dist_bandwidth: 2.0
    supcr_temperature: 0.1

    # Buffer configuration
    use_buffer: true
    buffer_size: 256
    min_samples_for_dist: 64
    min_samples_for_supcr: 64

    # DDP configuration
    use_ddp_gather: true
    gather_interval: 1

    # Adaptive weighting
    use_adaptive_weighting: true

    # Temperature scheduling
    use_temp_scheduling: true
    init_temperature: 1.0
    final_temperature: 0.1
    temp_schedule: "exponential"

    # Training duration (for temperature scheduling)
    max_epochs: 300
  optimizer:
    type: "AdamW"
    lr: 5e-4
    weight_decay: 1e-12
  lr_scheduler: null
  clip_grad_norm: true
  clip_grad_norm_max_norm: 10.0
  grad_accumulation_schedule: null #{0: 16}  # Accumulate for 16 steps
  plot_sample_ceiling: 10000
  plot_every_n_epochs: 2
```

```python
(torchcell) michaelvolk@gilahyper torchcell % python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/hetero_cell_nsa_retry.py
Starting HeteroCellNSA Training 🔫
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_hetero_cell_nsa_retry_test', 'tags': []}, 'cell_dataset': {'graphs': ['physical', 'regulatory', 'tflink', 'string12_0_neighborhood', 'string12_0_fusion', 'string12_0_cooccurence', 'string12_0_coexpression', 'string12_0_experimental', 'string12_0_database'], 'node_embeddings': ['learnable'], 'learnable_embedding_input_channels': 64, 'incidence_graphs': ['metabolism_bipartite']}, 'profiler': {'is_pytorch': True}, 'transforms': {'use_transforms': True, 'forward_transform': {'normalization': {'gene_interaction': {'strategy': 'standard'}}}}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 2, 'num_workers': 1, 'pin_memory': True, 'prefetch': False, 'prefetch_factor': 2}, 'trainer': {'max_epochs': 300, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': None, 'heads': 8, 'norms': 'layer', 'model': {'gene_num': 6607, 'reaction_num': 7122, 'metabolite_num': 2806, 'hidden_channels': 64, 'out_channels': 1, 'attention_pattern': ['M', 'S'], 'norm': 'layer', 'activation': 'relu', 'dropout': 0.0, 'heads': 8, 'prediction_head_config': {'hidden_channels': 64, 'head_num_layers': 4, 'dropout': 0.0, 'activation': 'relu', 'head_norm': 'layer'}}, 'regression_task': {'loss': 'mle_dist_supcr', 'is_weighted_phenotype_loss': False, 'lambda_mse': 1.0, 'lambda_dist': 100.0, 'lambda_supcr': 0.0, 'loss_config': {'dist_bandwidth': 2.0, 'supcr_temperature': 0.1, 'use_buffer': True, 'buffer_size': 256, 'min_samples_for_dist': 64, 'min_samples_for_supcr': 64, 'use_ddp_gather': True, 'gather_interval': 1, 'use_adaptive_weighting': True, 'use_temp_scheduling': True, 'init_temperature': 1.0, 'final_temperature': 0.1, 'temp_schedule': 'exponential', 'max_epochs': 300}, 'optimizer': {'type': 'AdamW', 'lr': 0.0005, 'weight_decay': 1e-12}, 'lr_scheduler': None, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 2}}
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.22.0
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-2a7bbb39-c8d0-4857-88f0-bcfdee6006d5_68d9662fae0f6f7634b7cef4c74aef7a60db175f540047c2f2fb130eb4e966d4/wandb/run-20250919_235347-66sskluj
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-2a7bbb39-c8d0-4857-88f0-bcfdee6006d5_68d9662fae0f6f7634b7cef4c74aef7a60db175f540047c2f2fb130eb4e966d4
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_hetero_cell_nsa_retry_test
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_hetero_cell_nsa_retry_test/runs/66sskluj
[2025-09-19 23:53:53,903][cobra.core.model][INFO] - The current solver interface glpk doesn't support setting the optimality tolerance.
/home/michaelvolk/Documents/projects/torchcell/experiments

Initializing transforms from configuration...
Added normalization transform for: ['gene_interaction']
Normalization parameters for gene_interaction:
  mean: -0.009887
  std: 0.062055
  min: -1.081600
  max: 1.128043
  q25: -0.037695
  q75: 0.020609
  strategy: standard
Transforms initialized successfully
Applied mandatory dense mask transformation for NSA M blocks
  Dense config: genes=6607, reactions=7122, metabolites=2806
Dataset Length: 332313
[2025-09-19 23:53:56,673][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-09-19 23:53:56,714][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-09-19 23:53:56,784][__main__][INFO] - cuda
Instantiating model (2025-09-19-23-53-56)
Parameter counts: {'gene_embedding': 422848, 'reaction_embedding': 455808, 'metabolite_embedding': 179584, 'preprocessor': 8448, 'nsa_layer': 704088, 'layer_norms': 384, 'global_aggregator': 6273, 'prediction_head': 12929, 'total': 1790362}
Creating RegressionTask (2025-09-19-23-53-56)
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/utilities/parsing.py:208: Attribute 'loss_func' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['loss_func'])`.
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-09-19-23-53-57)
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/fabric/plugins/environments/slurm.py:204: The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python /home/michaelvolk/Documents/projects/torchcell/exper ...
Using bfloat16 Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Starting HeteroCellNSA Training 🔫
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_hetero_cell_nsa_retry_test', 'tags': []}, 'cell_dataset': {'graphs': ['physical', 'regulatory', 'tflink', 'string12_0_neighborhood', 'string12_0_fusion', 'string12_0_cooccurence', 'string12_0_coexpression', 'string12_0_experimental', 'string12_0_database'], 'node_embeddings': ['learnable'], 'learnable_embedding_input_channels': 64, 'incidence_graphs': ['metabolism_bipartite']}, 'profiler': {'is_pytorch': True}, 'transforms': {'use_transforms': True, 'forward_transform': {'normalization': {'gene_interaction': {'strategy': 'standard'}}}}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 2, 'num_workers': 1, 'pin_memory': True, 'prefetch': False, 'prefetch_factor': 2}, 'trainer': {'max_epochs': 300, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': None, 'heads': 8, 'norms': 'layer', 'model': {'gene_num': 6607, 'reaction_num': 7122, 'metabolite_num': 2806, 'hidden_channels': 64, 'out_channels': 1, 'attention_pattern': ['M', 'S'], 'norm': 'layer', 'activation': 'relu', 'dropout': 0.0, 'heads': 8, 'prediction_head_config': {'hidden_channels': 64, 'head_num_layers': 4, 'dropout': 0.0, 'activation': 'relu', 'head_norm': 'layer'}}, 'regression_task': {'loss': 'mle_dist_supcr', 'is_weighted_phenotype_loss': False, 'lambda_mse': 1.0, 'lambda_dist': 100.0, 'lambda_supcr': 0.0, 'loss_config': {'dist_bandwidth': 2.0, 'supcr_temperature': 0.1, 'use_buffer': True, 'buffer_size': 256, 'min_samples_for_dist': 64, 'min_samples_for_supcr': 64, 'use_ddp_gather': True, 'gather_interval': 1, 'use_adaptive_weighting': True, 'use_temp_scheduling': True, 'init_temperature': 1.0, 'final_temperature': 0.1, 'temp_schedule': 'exponential', 'max_epochs': 300}, 'optimizer': {'type': 'AdamW', 'lr': 0.0005, 'weight_decay': 1e-12}, 'lr_scheduler': None, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 2}}
Starting HeteroCellNSA Training 🔫
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_hetero_cell_nsa_retry_test', 'tags': []}, 'cell_dataset': {'graphs': ['physical', 'regulatory', 'tflink', 'string12_0_neighborhood', 'string12_0_fusion', 'string12_0_cooccurence', 'string12_0_coexpression', 'string12_0_experimental', 'string12_0_database'], 'node_embeddings': ['learnable'], 'learnable_embedding_input_channels': 64, 'incidence_graphs': ['metabolism_bipartite']}, 'profiler': {'is_pytorch': True}, 'transforms': {'use_transforms': True, 'forward_transform': {'normalization': {'gene_interaction': {'strategy': 'standard'}}}}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 2, 'num_workers': 1, 'pin_memory': True, 'prefetch': False, 'prefetch_factor': 2}, 'trainer': {'max_epochs': 300, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': None, 'heads': 8, 'norms': 'layer', 'model': {'gene_num': 6607, 'reaction_num': 7122, 'metabolite_num': 2806, 'hidden_channels': 64, 'out_channels': 1, 'attention_pattern': ['M', 'S'], 'norm': 'layer', 'activation': 'relu', 'dropout': 0.0, 'heads': 8, 'prediction_head_config': {'hidden_channels': 64, 'head_num_layers': 4, 'dropout': 0.0, 'activation': 'relu', 'head_norm': 'layer'}}, 'regression_task': {'loss': 'mle_dist_supcr', 'is_weighted_phenotype_loss': False, 'lambda_mse': 1.0, 'lambda_dist': 100.0, 'lambda_supcr': 0.0, 'loss_config': {'dist_bandwidth': 2.0, 'supcr_temperature': 0.1, 'use_buffer': True, 'buffer_size': 256, 'min_samples_for_dist': 64, 'min_samples_for_supcr': 64, 'use_ddp_gather': True, 'gather_interval': 1, 'use_adaptive_weighting': True, 'use_temp_scheduling': True, 'init_temperature': 1.0, 'final_temperature': 0.1, 'temp_schedule': 'exponential', 'max_epochs': 300}, 'optimizer': {'type': 'AdamW', 'lr': 0.0005, 'weight_decay': 1e-12}, 'lr_scheduler': None, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 2}}
Starting HeteroCellNSA Training 🔫
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_hetero_cell_nsa_retry_test', 'tags': []}, 'cell_dataset': {'graphs': ['physical', 'regulatory', 'tflink', 'string12_0_neighborhood', 'string12_0_fusion', 'string12_0_cooccurence', 'string12_0_coexpression', 'string12_0_experimental', 'string12_0_database'], 'node_embeddings': ['learnable'], 'learnable_embedding_input_channels': 64, 'incidence_graphs': ['metabolism_bipartite']}, 'profiler': {'is_pytorch': True}, 'transforms': {'use_transforms': True, 'forward_transform': {'normalization': {'gene_interaction': {'strategy': 'standard'}}}}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 2, 'num_workers': 1, 'pin_memory': True, 'prefetch': False, 'prefetch_factor': 2}, 'trainer': {'max_epochs': 300, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': None, 'heads': 8, 'norms': 'layer', 'model': {'gene_num': 6607, 'reaction_num': 7122, 'metabolite_num': 2806, 'hidden_channels': 64, 'out_channels': 1, 'attention_pattern': ['M', 'S'], 'norm': 'layer', 'activation': 'relu', 'dropout': 0.0, 'heads': 8, 'prediction_head_config': {'hidden_channels': 64, 'head_num_layers': 4, 'dropout': 0.0, 'activation': 'relu', 'head_norm': 'layer'}}, 'regression_task': {'loss': 'mle_dist_supcr', 'is_weighted_phenotype_loss': False, 'lambda_mse': 1.0, 'lambda_dist': 100.0, 'lambda_supcr': 0.0, 'loss_config': {'dist_bandwidth': 2.0, 'supcr_temperature': 0.1, 'use_buffer': True, 'buffer_size': 256, 'min_samples_for_dist': 64, 'min_samples_for_supcr': 64, 'use_ddp_gather': True, 'gather_interval': 1, 'use_adaptive_weighting': True, 'use_temp_scheduling': True, 'init_temperature': 1.0, 'final_temperature': 0.1, 'temp_schedule': 'exponential', 'max_epochs': 300}, 'optimizer': {'type': 'AdamW', 'lr': 0.0005, 'weight_decay': 1e-12}, 'lr_scheduler': None, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 2}}
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: ⢿ Waiting for wandb.init()...
wandb: ⣻ Waiting for wandb.init()...
wandb: Tracking run with wandb version 0.22.0
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-e56fec76-ce34-43a2-9262-03745c117637_68d9662fae0f6f7634b7cef4c74aef7a60db175f540047c2f2fb130eb4e966d4/wandb/run-20250919_235419-j5t5pywy
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-e56fec76-ce34-43a2-9262-03745c117637_68d9662fae0f6f7634b7cef4c74aef7a60db175f540047c2f2fb130eb4e966d4
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_hetero_cell_nsa_retry_test
wandb: Tracking run with wandb version 0.22.0
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-ea80956f-cd1e-40c5-a2be-6204dae3558b_68d9662fae0f6f7634b7cef4c74aef7a60db175f540047c2f2fb130eb4e966d4/wandb/run-20250919_235419-5qx3zmj7
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-ea80956f-cd1e-40c5-a2be-6204dae3558b_68d9662fae0f6f7634b7cef4c74aef7a60db175f540047c2f2fb130eb4e966d4
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_hetero_cell_nsa_retry_test
wandb: Tracking run with wandb version 0.22.0
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-2c39cded-9876-456a-85b9-c38960ad678d_68d9662fae0f6f7634b7cef4c74aef7a60db175f540047c2f2fb130eb4e966d4/wandb/run-20250919_235419-gm8f8lhs
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-2c39cded-9876-456a-85b9-c38960ad678d_68d9662fae0f6f7634b7cef4c74aef7a60db175f540047c2f2fb130eb4e966d4
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_hetero_cell_nsa_retry_test
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_hetero_cell_nsa_retry_test/runs/gm8f8lhs
[2025-09-19 23:54:25,616][cobra.core.model][INFO] - The current solver interface glpk doesn't support setting the optimality tolerance.
[2025-09-19 23:54:25,655][cobra.core.model][INFO] - The current solver interface glpk doesn't support setting the optimality tolerance.
[2025-09-19 23:54:25,750][cobra.core.model][INFO] - The current solver interface glpk doesn't support setting the optimality tolerance.
/home/michaelvolk/Documents/projects/torchcell/experiments

Initializing transforms from configuration...
/home/michaelvolk/Documents/projects/torchcell/experiments

Initializing transforms from configuration...
/home/michaelvolk/Documents/projects/torchcell/experiments

Initializing transforms from configuration...
Added normalization transform for: ['gene_interaction']
Normalization parameters for gene_interaction:
  mean: -0.009887
  std: 0.062055
  min: -1.081600
  max: 1.128043
  q25: -0.037695
  q75: 0.020609
  strategy: standard
Transforms initialized successfully
Applied mandatory dense mask transformation for NSA M blocks
  Dense config: genes=6607, reactions=7122, metabolites=2806
Dataset Length: 332313
[2025-09-19 23:54:28,460][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
Added normalization transform for: ['gene_interaction']
Normalization parameters for gene_interaction:
  mean: -0.009887
  std: 0.062055
  min: -1.081600
  max: 1.128043
  q25: -0.037695
  q75: 0.020609
  strategy: standard
Transforms initialized successfully
Applied mandatory dense mask transformation for NSA M blocks
  Dense config: genes=6607, reactions=7122, metabolites=2806
Dataset Length: 332313
[2025-09-19 23:54:28,489][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-09-19 23:54:28,503][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-09-19 23:54:28,531][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-09-19 23:54:28,580][__main__][INFO] - cuda
Instantiating model (2025-09-19-23-54-28)
Added normalization transform for: ['gene_interaction']
Normalization parameters for gene_interaction:
  mean: -0.009887
  std: 0.062055
  min: -1.081600
  max: 1.128043
  q25: -0.037695
  q75: 0.020609
  strategy: standard
Transforms initialized successfully
Applied mandatory dense mask transformation for NSA M blocks
  Dense config: genes=6607, reactions=7122, metabolites=2806
Dataset Length: 332313
[2025-09-19 23:54:28,585][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-09-19 23:54:28,614][__main__][INFO] - cuda
Instantiating model (2025-09-19-23-54-28)
[2025-09-19 23:54:28,628][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-09-19 23:54:28,707][__main__][INFO] - cuda
Instantiating model (2025-09-19-23-54-28)
Parameter counts: {'gene_embedding': 422848, 'reaction_embedding': 455808, 'metabolite_embedding': 179584, 'preprocessor': 8448, 'nsa_layer': 704088, 'layer_norms': 384, 'global_aggregator': 6273, 'prediction_head': 12929, 'total': 1790362}
Creating RegressionTask (2025-09-19-23-54-28)
Parameter counts: {'gene_embedding': 422848, 'reaction_embedding': 455808, 'metabolite_embedding': 179584, 'preprocessor': 8448, 'nsa_layer': 704088, 'layer_norms': 384, 'global_aggregator': 6273, 'prediction_head': 12929, 'total': 1790362}
Creating RegressionTask (2025-09-19-23-54-28)
Parameter counts: {'gene_embedding': 422848, 'reaction_embedding': 455808, 'metabolite_embedding': 179584, 'preprocessor': 8448, 'nsa_layer': 704088, 'layer_norms': 384, 'global_aggregator': 6273, 'prediction_head': 12929, 'total': 1790362}
Creating RegressionTask (2025-09-19-23-54-28)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-09-19-23-54-28)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-09-19-23-54-28)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-09-19-23-54-28)
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/4
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/4
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/4
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 4 processes
----------------------------------------------------------------------------------------------------

/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loggers/wandb.py:396: There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`.
LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3]

  | Name                      | Type             | Params | Mode 
-----------------------------------------------------------------------
0 | model                     | HeteroCellNSA    | 1.8 M  | train
1 | loss_func                 | MleDistSupCR     | 0      | train
2 | train_metrics             | MetricCollection | 0      | train
3 | train_transformed_metrics | MetricCollection | 0      | train
4 | val_metrics               | MetricCollection | 0      | train
5 | val_transformed_metrics   | MetricCollection | 0      | train
6 | test_metrics              | MetricCollection | 0      | train
7 | test_transformed_metrics  | MetricCollection | 0      | train
-----------------------------------------------------------------------
1.8 M     Trainable params
0         Non-trainable params
1.8 M     Total params
7.161     Total estimated model params size (MB)
511       Modules in train mode
0         Modules in eval mode
Sanity Checking: |                                                                                                    | 0/? [00:00<?, ?it/s]/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=31` in the `DataLoader` to improve performance.
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
Sanity Checking DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████| 2/2 [00:07<00:00,  0.25it/s]/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using torch.float32).
  warnings.warn(*args, **kwargs)  # noqa: B028
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=31` in the `DataLoader` to improve performance.
Epoch 0:  55%|██████████████████████████████████████▏                               | 18115/33232 [36:13:08<30:13:29,  0.14it/s, v_num=kluj]
```
