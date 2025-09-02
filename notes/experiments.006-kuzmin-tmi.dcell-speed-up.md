---
id: oif1nkriaxkqtrpus3kevoj
title: Dcell Speed Up
desc: ''
updated: 1756769129302
created: 1756769121419
---
## Before Duplicate Forward

```python
(torchcell) michaelvolk@gilahyper torchcell % python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: Using wandb-core as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: Tracking run with wandb version 0.18.3
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
Using Gene Ontology hierarchy
[2025-08-26 13:07:02,356][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-26 13:07:04,156][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
[2025-08-26 13:07:06,514][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-26 13:07:08,430][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-26 13:07:08,470][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-26 13:07:08,539][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-26-13-07-08)
Instantiating DCell
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-26-13-07-12)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-26-13-07-12)
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/fabric/plugins/environments/slurm.py:204: The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python /home/michaelvolk/Documents/projects/torchcell/exper ...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: Tracking run with wandb version 0.18.3
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Tracking run with wandb version 0.18.3
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Tracking run with wandb version 0.18.3
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
[2025-08-26 13:07:29,071][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-26 13:07:29,105][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-26 13:07:29,146][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-26 13:07:30,905][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
[2025-08-26 13:07:30,960][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
[2025-08-26 13:07:30,985][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
After redundant filter: 5637
[2025-08-26 13:07:33,268][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
[2025-08-26 13:07:33,369][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
[2025-08-26 13:07:33,378][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-26 13:07:35,249][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-26 13:07:35,293][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-26 13:07:35,350][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-26 13:07:35,372][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-26-13-07-35)
Instantiating DCell
[2025-08-26 13:07:35,395][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-26 13:07:35,406][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-26 13:07:35,451][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-26 13:07:35,472][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-26-13-07-35)
Instantiating DCell
[2025-08-26 13:07:35,527][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-26-13-07-35)
Instantiating DCell
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-26-13-07-43)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-26-13-07-43)
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-26-13-07-43)
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-26-13-07-43)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-26-13-07-44)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-26-13-07-44)
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
[2025-08-26 13:07:46,260][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-26 13:07:46,261][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-26 13:07:46,266][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-26 13:07:46,275][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-26 13:07:46,327][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-26 13:07:46,327][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-26 13:07:46,327][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-26 13:07:46,328][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-26 13:07:46,334][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-26 13:07:46,334][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-26 13:07:46,341][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-26 13:07:46,341][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-26 13:07:46,353][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-26 13:07:46,354][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-26 13:07:46,362][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-26 13:07:46,368][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/core/optimizer.py:316: The lr scheduler dict contains the key(s) ['monitor'], but the keys will be ignored. You need to call `lr_scheduler.step()` manually in manual optimization.

  | Name                      | Type             | Params | Mode 
-----------------------------------------------------------------------
0 | model                     | DCell            | 20.8 M | train
1 | loss_func                 | DCellLoss        | 0      | train
2 | train_metrics             | MetricCollection | 0      | train
3 | train_transformed_metrics | MetricCollection | 0      | train
4 | val_metrics               | MetricCollection | 0      | train
5 | val_transformed_metrics   | MetricCollection | 0      | train
6 | test_metrics              | MetricCollection | 0      | train
7 | test_transformed_metrics  | MetricCollection | 0      | train
-----------------------------------------------------------------------
20.8 M    Trainable params
0         Non-trainable params
20.8 M    Total params
83.062    Total estimated model params size (MB)
13754     Modules in train mode
0         Modules in eval mode
Sanity Checking DataLoader 0: 100%|██████████████████████████████████████████████████████████████████████| 2/2 [03:32<00:00,  0.01it/s]/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using torch.float32).
  warnings.warn(*args, **kwargs)  # noqa: B028
Epoch 0:   0%|                                                                                                 | 0/260 [00:00<?, ?it/s]Epoch 0:   3%|██▎                                                                        | 8/260 [17:04<8:57:59,  0.01it/s, v_num=2ch5]
```

## After Removing Duplicate Forward

```python
(base) michaelvolk@gilahyper torchcell % conda activate torchcell                                            14:26
(torchcell) michaelvolk@gilahyper torchcell % python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: Using wandb-core as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-6048893f-e9f9-4d6c-bcb9-7ce32d5c6d07_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60/wandb/run-20250826_142638-p13bazry
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-6048893f-e9f9-4d6c-bcb9-7ce32d5c6d07_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/p13bazry
Using Gene Ontology hierarchy
[2025-08-26 14:26:43,985][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-26 14:26:45,780][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
[2025-08-26 14:26:48,108][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-26 14:26:50,031][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-26 14:26:50,074][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-26 14:26:50,143][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-26-14-26-50)
Instantiating DCell
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-26-14-26-53)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-26-14-26-53)
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/fabric/plugins/environments/slurm.py:204: The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python /home/michaelvolk/Documents/projects/torchcell/exper ...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-1adbf4e7-f69e-4cb1-b1ef-4ac76e8a7960_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60/wandb/run-20250826_142710-hoohdmmb
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-1adbf4e7-f69e-4cb1-b1ef-4ac76e8a7960_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/hoohdmmb
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-62f0c99f-8730-4035-9d3b-3190dd09f392_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60/wandb/run-20250826_142710-1v8euv6l
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-62f0c99f-8730-4035-9d3b-3190dd09f392_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/1v8euv6l
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-f3cc7fb9-b3c5-47b2-8804-65387a9e4517_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60/wandb/run-20250826_142710-49k4lgd9
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-f3cc7fb9-b3c5-47b2-8804-65387a9e4517_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/49k4lgd9
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
[2025-08-26 14:27:15,648][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-26 14:27:15,722][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-26 14:27:15,748][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-26 14:27:17,436][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
[2025-08-26 14:27:17,538][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
[2025-08-26 14:27:17,555][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
After redundant filter: 5637
[2025-08-26 14:27:19,787][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
[2025-08-26 14:27:19,868][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
[2025-08-26 14:27:19,892][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-26 14:27:21,757][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-26 14:27:21,801][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-26 14:27:21,840][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-26 14:27:21,869][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-26 14:27:21,879][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-26-14-27-21)
Instantiating DCell
[2025-08-26 14:27:21,884][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-26 14:27:21,913][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-26 14:27:21,956][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-26-14-27-21)
Instantiating DCell
[2025-08-26 14:27:21,983][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-26-14-27-21)
Instantiating DCell
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-26-14-27-30)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-26-14-27-30)
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/4
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-26-14-27-30)
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-26-14-27-30)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-26-14-27-30)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-26-14-27-30)
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/4
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/4
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 4 processes
----------------------------------------------------------------------------------------------------

/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loggers/wandb.py:396: There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
[2025-08-26 14:27:33,420][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-26 14:27:33,423][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-26 14:27:33,424][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-26 14:27:33,428][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-26 14:27:33,489][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-26 14:27:33,489][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-26 14:27:33,491][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-26 14:27:33,491][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-26 14:27:33,493][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-26 14:27:33,493][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-26 14:27:33,500][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-26 14:27:33,500][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-26 14:27:33,516][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-26 14:27:33,517][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-26 14:27:33,521][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-26 14:27:33,528][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/core/optimizer.py:316: The lr scheduler dict contains the key(s) ['monitor'], but the keys will be ignored. You need to call `lr_scheduler.step()` manually in manual optimization.

  | Name                      | Type             | Params | Mode 
-----------------------------------------------------------------------
0 | model                     | DCell            | 20.8 M | train
1 | loss_func                 | DCellLoss        | 0      | train
2 | train_metrics             | MetricCollection | 0      | train
3 | train_transformed_metrics | MetricCollection | 0      | train
4 | val_metrics               | MetricCollection | 0      | train
5 | val_transformed_metrics   | MetricCollection | 0      | train
6 | test_metrics              | MetricCollection | 0      | train
7 | test_transformed_metrics  | MetricCollection | 0      | train
-----------------------------------------------------------------------
20.8 M    Trainable params
0         Non-trainable params
20.8 M    Total params
83.062    Total estimated model params size (MB)
13754     Modules in train mode
0         Modules in eval mode
Sanity Checking DataLoader 0:  50%|█████████████████████████                         | 1/2 [00:53<00:53,  0.02it/s]Sanity Checking DataLoader 0: 100%|██████████████████████████████████████████████████| 2/2 [01:49<00:00,  0.02it/s]/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using torch.float32).
  warnings.warn(*args, **kwargs)  # noqa: B028
Epoch 0:   3%|█▋                                                     | 8/260 [10:54<5:43:23,  0.01it/s, v_num=azry]
```

## After Caching GO Strata

```python
(torchcell) michaelvolk@gilahyper torchcell % python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: Using wandb-core as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-79f7d0b0-3146-4a94-b512-3ddb97dd651a_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60/wandb/run-20250827_170851-s9n932dc
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-79f7d0b0-3146-4a94-b512-3ddb97dd651a_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/s9n932dc
Using Gene Ontology hierarchy
[2025-08-27 17:08:56,341][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-27 17:08:58,141][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
[2025-08-27 17:09:00,474][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-27 17:09:02,433][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-27 17:09:02,474][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-27 17:09:02,545][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-27-17-09-02)
Instantiating DCell
Pre-computing gene indices for 2745 GO terms...
Pre-computed indices in 0.19 seconds
  Total rows per sample: 60737
  Average genes per GO term: 22.1
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-27-17-09-06)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-27-17-09-06)
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/fabric/plugins/environments/slurm.py:204: The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python /home/michaelvolk/Documents/projects/torchcell/exper ...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-9fe39d50-dca7-420d-845e-7bc49ff4441a_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60/wandb/run-20250827_170922-hasx2n41
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-9fe39d50-dca7-420d-845e-7bc49ff4441a_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/hasx2n41
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-2914e5dd-ea6f-470d-bb8f-9d9988c7bb8a_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60/wandb/run-20250827_170922-8ssamtpm
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-2914e5dd-ea6f-470d-bb8f-9d9988c7bb8a_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/8ssamtpm
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-ad11a76c-4407-4997-a170-78fe81137e4d_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60/wandb/run-20250827_170922-rk5yp412
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-ad11a76c-4407-4997-a170-78fe81137e4d_65062819ae19486d940a4c87a777686dc5821c62a3b8f712c6771f27bd4a2a60
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/rk5yp412
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
[2025-08-27 17:09:28,250][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-27 17:09:28,265][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-27 17:09:28,397][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-27 17:09:30,078][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
[2025-08-27 17:09:30,087][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
After redundant filter: 5637
[2025-08-27 17:09:30,211][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
[2025-08-27 17:09:32,459][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
[2025-08-27 17:09:32,469][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
[2025-08-27 17:09:32,601][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-27 17:09:34,444][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-27 17:09:34,462][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-27 17:09:34,489][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-27 17:09:34,506][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-27 17:09:34,563][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-27-17-09-34)
Instantiating DCell
[2025-08-27 17:09:34,577][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-27-17-09-34)
Instantiating DCell
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-27 17:09:34,631][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-27 17:09:34,675][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-27 17:09:34,744][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-27-17-09-34)
Instantiating DCell
Pre-computing gene indices for 2745 GO terms...
Pre-computing gene indices for 2745 GO terms...
Pre-computing gene indices for 2745 GO terms...
Pre-computed indices in 1.12 seconds
  Total rows per sample: 60737
  Average genes per GO term: 22.1
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Pre-computed indices in 1.13 seconds
  Total rows per sample: 60737
  Average genes per GO term: 22.1
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Pre-computed indices in 1.15 seconds
  Total rows per sample: 60737
  Average genes per GO term: 22.1
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-27-17-09-43)
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-27-17-09-43)
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-27-17-09-43)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-27-17-09-43)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-27-17-09-43)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-27-17-09-43)
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/4
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/4
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/4
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 4 processes
----------------------------------------------------------------------------------------------------

/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loggers/wandb.py:396: There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
[2025-08-27 17:09:46,636][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-27 17:09:46,643][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-27 17:09:46,646][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-27 17:09:46,650][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-27 17:09:46,705][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-27 17:09:46,705][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-27 17:09:46,710][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-27 17:09:46,710][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-27 17:09:46,713][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-27 17:09:46,713][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-27 17:09:46,716][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-27 17:09:46,717][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-27 17:09:46,730][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-27 17:09:46,738][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-27 17:09:46,738][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-27 17:09:46,742][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/core/optimizer.py:316: The lr scheduler dict contains the key(s) ['monitor'], but the keys will be ignored. You need to call `lr_scheduler.step()` manually in manual optimization.

  | Name                      | Type             | Params | Mode 
-----------------------------------------------------------------------
0 | model                     | DCell            | 20.8 M | train
1 | loss_func                 | DCellLoss        | 0      | train
2 | train_metrics             | MetricCollection | 0      | train
3 | train_transformed_metrics | MetricCollection | 0      | train
4 | val_metrics               | MetricCollection | 0      | train
5 | val_transformed_metrics   | MetricCollection | 0      | train
6 | test_metrics              | MetricCollection | 0      | train
7 | test_transformed_metrics  | MetricCollection | 0      | train
-----------------------------------------------------------------------
20.8 M    Trainable params
0         Non-trainable params
20.8 M    Total params
83.062    Total estimated model params size (MB)
13754     Modules in train mode
0         Modules in eval mode
Sanity Checking DataLoader 0: 100%|█████████████████████████████| 2/2 [00:30<00:00,  0.06it/s]/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using torch.float32).
  warnings.warn(*args, **kwargs)  # noqa: B028
Epoch 0:  25%|████████▍                        | 66/260 [25:25<1:14:44,  0.04it/s, v_num=32dc]
```

## 16-Mixed Precision

```python
(torchcell) michaelvolk@gilahyper torchcell % python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': '16-mixed'}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: Using wandb-core as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-84ef1472-0b0c-415c-9ccf-f8135f8eba28_ccec53c5514f909299e6929981840c83ecfff4b7df9a1fd04093ecf5f687ff65/wandb/run-20250827_174328-cs5nhina
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-84ef1472-0b0c-415c-9ccf-f8135f8eba28_ccec53c5514f909299e6929981840c83ecfff4b7df9a1fd04093ecf5f687ff65
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/cs5nhina
Using Gene Ontology hierarchy
[2025-08-27 17:43:33,991][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-27 17:43:35,825][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
[2025-08-27 17:43:38,201][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-27 17:43:40,161][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-27 17:43:40,206][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-27 17:43:40,282][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-27-17-43-40)
Instantiating DCell
Pre-computing gene indices for 2745 GO terms...
Pre-computed indices in 0.19 seconds
  Total rows per sample: 60737
  Average genes per GO term: 22.1
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-27-17-43-43)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-27-17-43-43)
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/fabric/plugins/environments/slurm.py:204: The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python /home/michaelvolk/Documents/projects/torchcell/exper ...
Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': '16-mixed'}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': '16-mixed'}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': '16-mixed'}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-ab1e8ec1-2ebf-499a-b660-bc575340b4b5_ccec53c5514f909299e6929981840c83ecfff4b7df9a1fd04093ecf5f687ff65/wandb/run-20250827_174400-rkn5xf69
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-ab1e8ec1-2ebf-499a-b660-bc575340b4b5_ccec53c5514f909299e6929981840c83ecfff4b7df9a1fd04093ecf5f687ff65
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/rkn5xf69
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-3f25250a-3c05-4ff5-9bb9-bdc02b54299c_ccec53c5514f909299e6929981840c83ecfff4b7df9a1fd04093ecf5f687ff65/wandb/run-20250827_174400-oojgnh6b
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-3f25250a-3c05-4ff5-9bb9-bdc02b54299c_ccec53c5514f909299e6929981840c83ecfff4b7df9a1fd04093ecf5f687ff65
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/oojgnh6b
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-3bed042e-a3bc-4c1e-926c-2180824bb5ea_ccec53c5514f909299e6929981840c83ecfff4b7df9a1fd04093ecf5f687ff65/wandb/run-20250827_174400-vfdq67ft
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-3bed042e-a3bc-4c1e-926c-2180824bb5ea_ccec53c5514f909299e6929981840c83ecfff4b7df9a1fd04093ecf5f687ff65
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/vfdq67ft
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
[2025-08-27 17:44:06,523][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-27 17:44:06,612][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-27 17:44:06,645][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-27 17:44:08,346][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
[2025-08-27 17:44:08,458][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
[2025-08-27 17:44:08,490][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
After redundant filter: 5637
[2025-08-27 17:44:10,746][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
[2025-08-27 17:44:10,855][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
[2025-08-27 17:44:10,893][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-27 17:44:12,731][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-27 17:44:12,774][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-27 17:44:12,846][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-27-17-44-12)
Instantiating DCell
[2025-08-27 17:44:12,860][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-27 17:44:12,894][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-27 17:44:12,904][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-27 17:44:12,939][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-27 17:44:12,982][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-27-17-44-12)
Instantiating DCell
[2025-08-27 17:44:13,015][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-27-17-44-13)
Instantiating DCell
Pre-computing gene indices for 2745 GO terms...
Pre-computing gene indices for 2745 GO terms...
Pre-computing gene indices for 2745 GO terms...
Pre-computed indices in 0.92 seconds
  Total rows per sample: 60737
  Average genes per GO term: 22.1
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Pre-computed indices in 1.15 seconds
  Total rows per sample: 60737
  Average genes per GO term: 22.1
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Pre-computed indices in 1.16 seconds
  Total rows per sample: 60737
  Average genes per GO term: 22.1
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-27-17-44-21)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-27-17-44-21)
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/4
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-27-17-44-22)
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-27-17-44-22)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-27-17-44-22)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-27-17-44-22)
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/4
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/4
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 4 processes
----------------------------------------------------------------------------------------------------

/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loggers/wandb.py:396: There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
[2025-08-27 17:44:24,945][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-27 17:44:24,946][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-27 17:44:24,949][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-27 17:44:24,954][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-27 17:44:25,016][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-27 17:44:25,016][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-27 17:44:25,018][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-27 17:44:25,019][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-27 17:44:25,019][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-27 17:44:25,019][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-27 17:44:25,022][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-27 17:44:25,022][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-27 17:44:25,044][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-27 17:44:25,047][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-27 17:44:25,047][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-27 17:44:25,052][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/core/optimizer.py:316: The lr scheduler dict contains the key(s) ['monitor'], but the keys will be ignored. You need to call `lr_scheduler.step()` manually in manual optimization.

  | Name                      | Type             | Params | Mode 
-----------------------------------------------------------------------
0 | model                     | DCell            | 20.8 M | train
1 | loss_func                 | DCellLoss        | 0      | train
2 | train_metrics             | MetricCollection | 0      | train
3 | train_transformed_metrics | MetricCollection | 0      | train
4 | val_metrics               | MetricCollection | 0      | train
5 | val_transformed_metrics   | MetricCollection | 0      | train
6 | test_metrics              | MetricCollection | 0      | train
7 | test_transformed_metrics  | MetricCollection | 0      | train
-----------------------------------------------------------------------
20.8 M    Trainable params
0         Non-trainable params
20.8 M    Total params
83.062    Total estimated model params size (MB)
13754     Modules in train mode
0         Modules in eval mode
Sanity Checking DataLoader 0: 100%|█████████████████████████████| 2/2 [00:30<00:00,  0.06it/s]/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using torch.float32).
  warnings.warn(*args, **kwargs)  # noqa: B028
Epoch 0:  28%|█████████▍                       | 74/260 [27:46<1:09:48,  0.04it/s, v_num=hina]
```

## BF16-Mixed Precision

```python
(torchcell) michaelvolk@gilahyper torchcell % python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: Using wandb-core as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-7976e4da-85b0-4a29-97d6-d72ffd5f14f0_4fe8e9aa750bcf5a6d524fb14b3cd5288fce48279c76540423894b25aa9c8394/wandb/run-20250827_182049-1s5g2x9u
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-7976e4da-85b0-4a29-97d6-d72ffd5f14f0_4fe8e9aa750bcf5a6d524fb14b3cd5288fce48279c76540423894b25aa9c8394
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/1s5g2x9u
Using Gene Ontology hierarchy
[2025-08-27 18:20:55,055][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-27 18:20:56,860][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
[2025-08-27 18:20:59,255][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-27 18:21:01,203][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-27 18:21:01,244][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-27 18:21:01,321][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-27-18-21-01)
Instantiating DCell
Pre-computing gene indices for 2745 GO terms...
Pre-computed indices in 0.19 seconds
  Total rows per sample: 60737
  Average genes per GO term: 22.1
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-27-18-21-04)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-27-18-21-04)
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/fabric/plugins/environments/slurm.py:204: The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python /home/michaelvolk/Documents/projects/torchcell/exper ...
Using bfloat16 Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-8590ef0f-3325-4eb2-a5f0-0af367b98ebd_4fe8e9aa750bcf5a6d524fb14b3cd5288fce48279c76540423894b25aa9c8394/wandb/run-20250827_182121-kzfhkvc1
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-8590ef0f-3325-4eb2-a5f0-0af367b98ebd_4fe8e9aa750bcf5a6d524fb14b3cd5288fce48279c76540423894b25aa9c8394
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/kzfhkvc1
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-9fc5dca6-0bca-43c1-8a76-42d3318889ca_4fe8e9aa750bcf5a6d524fb14b3cd5288fce48279c76540423894b25aa9c8394/wandb/run-20250827_182121-7akxzduw
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-9fc5dca6-0bca-43c1-8a76-42d3318889ca_4fe8e9aa750bcf5a6d524fb14b3cd5288fce48279c76540423894b25aa9c8394
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/7akxzduw
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-61b98677-71d5-40c3-a622-e688e4c5c8dc_4fe8e9aa750bcf5a6d524fb14b3cd5288fce48279c76540423894b25aa9c8394/wandb/run-20250827_182121-6krsptb4
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-61b98677-71d5-40c3-a622-e688e4c5c8dc_4fe8e9aa750bcf5a6d524fb14b3cd5288fce48279c76540423894b25aa9c8394
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/6krsptb4
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
[2025-08-27 18:21:27,024][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-27 18:21:27,138][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-27 18:21:27,148][torchcell.graph.graph][INFO] - Filtering result: 122 GO terms and 1961 IGI gene annotations removed
After IGI filter: 5752
[2025-08-27 18:21:28,815][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
[2025-08-27 18:21:28,966][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
[2025-08-27 18:21:28,979][torchcell.graph.graph][INFO] - Filtering result: 115 redundant GO terms removed
After redundant filter: 5637
After redundant filter: 5637
[2025-08-27 18:21:31,178][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
[2025-08-27 18:21:31,334][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
[2025-08-27 18:21:31,377][torchcell.graph.graph][INFO] - Filtering result: 2892 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
After containment filter (min_genes=4): 2745
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-27 18:21:33,150][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-27 18:21:33,191][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-27 18:21:33,260][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-27-18-21-33)
Instantiating DCell
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-27 18:21:33,316][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
Computed 14 strata for 2745 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 639 terms
  Stratum 3: 540 terms
  Stratum 4: 367 terms
  ... and 9 more strata
[2025-08-27 18:21:33,362][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-27 18:21:33,367][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-27 18:21:33,408][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-27 18:21:33,435][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-27-18-21-33)
Instantiating DCell
[2025-08-27 18:21:33,481][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-27-18-21-33)
Instantiating DCell
Pre-computing gene indices for 2745 GO terms...
Pre-computing gene indices for 2745 GO terms...
Pre-computing gene indices for 2745 GO terms...
Pre-computed indices in 0.95 seconds
  Total rows per sample: 60737
  Average genes per GO term: 22.1
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Pre-computed indices in 1.14 seconds
  Total rows per sample: 60737
  Average genes per GO term: 22.1
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Pre-computed indices in 1.15 seconds
  Total rows per sample: 60737
  Average genes per GO term: 22.1
DCell model initialized:
  GO terms: 2745
  Genes: 6607
  Strata: 14 (max: 13)
  Total subsystems: 2745
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-27-18-21-42)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-27-18-21-42)
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/4
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-27-18-21-42)
Parameter counts: {'subsystems': 20699534, 'dcell_linear': 65974, 'dcell': 20765508, 'total': 20765508, 'num_go_terms': 2745, 'num_subsystems': 2745}
Creating RegressionTask (2025-08-27-18-21-42)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-27-18-21-42)
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-27-18-21-42)
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/4
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/4
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 4 processes
----------------------------------------------------------------------------------------------------

/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loggers/wandb.py:396: There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
[2025-08-27 18:21:45,251][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-27 18:21:45,251][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-27 18:21:45,255][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-27 18:21:45,255][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-27 18:21:45,319][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-27 18:21:45,319][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-27 18:21:45,320][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-27 18:21:45,320][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-27 18:21:45,320][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-27 18:21:45,320][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-27 18:21:45,324][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-27 18:21:45,324][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-27 18:21:45,346][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-27 18:21:45,346][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-27 18:21:45,347][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
[2025-08-27 18:21:45,352][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,765,509
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/core/optimizer.py:316: The lr scheduler dict contains the key(s) ['monitor'], but the keys will be ignored. You need to call `lr_scheduler.step()` manually in manual optimization.

  | Name                      | Type             | Params | Mode 
-----------------------------------------------------------------------
0 | model                     | DCell            | 20.8 M | train
1 | loss_func                 | DCellLoss        | 0      | train
2 | train_metrics             | MetricCollection | 0      | train
3 | train_transformed_metrics | MetricCollection | 0      | train
4 | val_metrics               | MetricCollection | 0      | train
5 | val_transformed_metrics   | MetricCollection | 0      | train
6 | test_metrics              | MetricCollection | 0      | train
7 | test_transformed_metrics  | MetricCollection | 0      | train
-----------------------------------------------------------------------
20.8 M    Trainable params
0         Non-trainable params
20.8 M    Total params
83.062    Total estimated model params size (MB)
13754     Modules in train mode
0         Modules in eval mode
Sanity Checking DataLoader 0: 100%|█████████████████████████████| 2/2 [00:30<00:00,  0.06it/s]/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using torch.float32).
  warnings.warn(*args, **kwargs)  # noqa: B028
Epoch 0:  92%|█████████████████████████████▍  | 239/260 [1:21:39<07:10,  0.05it/s, v_num=2x9u]
```

## BF16-Mixed Precision Regress

```python
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': None, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': None, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': None, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': None, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
[2025-08-29 10:44:50,318][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 10:44:50,328][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 10:44:50,446][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 10:44:50,526][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 10:44:52,345][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
[2025-08-29 10:44:52,348][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
After redundant filter: 5435
[2025-08-29 10:44:52,470][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
[2025-08-29 10:44:52,591][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
[2025-08-29 10:44:54,959][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
[2025-08-29 10:44:54,989][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
[2025-08-29 10:44:55,097][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
[2025-08-29 10:44:55,175][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 10:44:57,093][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 10:44:57,135][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-29 10:44:57,154][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-29 10:44:57,198][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 10:44:57,210][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-10-44-57)
Instantiating DCell
[2025-08-29 10:44:57,251][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-29 10:44:57,273][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-10-44-57)
Instantiating DCell
[2025-08-29 10:44:57,295][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 10:44:57,353][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-29 10:44:57,369][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-10-44-57)
Instantiating DCell
[2025-08-29 10:44:57,400][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-29 10:44:57,476][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-10-44-57)
Instantiating DCell
Pre-computing gene indices for 2655 GO terms...
Pre-computing gene indices for 2655 GO terms...
Pre-computing gene indices for 2655 GO terms...
Pre-computing gene indices for 2655 GO terms...
Pre-computed indices in 1.43 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Pre-computed indices in 1.43 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Pre-computed indices in 1.48 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Pre-computed indices in 1.48 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-10-45-08)
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-10-45-08)
devices: 4
SLURM_JOB_NUM_NODES: 1
SLURM_NNODES: 1
SLURM_NPROCS: 1
Starting training (2025-08-29-10-45-08)
devices: 4
SLURM_JOB_NUM_NODES: 1
SLURM_NNODES: 1
SLURM_NPROCS: 1
Starting training (2025-08-29-10-45-08)
gilahyper:212652:212652 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:212652:212652 [0] NCCL INFO Bootstrap: Using enp37s0f0:192.168.1.15<0>
gilahyper:212652:212652 [0] NCCL INFO cudaDriverVersion 12080
gilahyper:212652:212652 [0] NCCL INFO NCCL version 2.27.3+cuda12.9
gilahyper:212652:212652 [0] NCCL INFO Comm config Blocking set to 1
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-10-45-08)
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-10-45-08)
gilahyper:212653:212653 [1] NCCL INFO cudaDriverVersion 12080
gilahyper:212653:212653 [1] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:212653:212653 [1] NCCL INFO Bootstrap: Using enp37s0f0:192.168.1.15<0>
gilahyper:212653:212653 [1] NCCL INFO NCCL version 2.27.3+cuda12.9
devices: 4
SLURM_JOB_NUM_NODES: 1
SLURM_NNODES: 1
SLURM_NPROCS: 1
Starting training (2025-08-29-10-45-08)
gilahyper:212653:212653 [1] NCCL INFO Comm config Blocking set to 1
gilahyper:212652:214161 [0] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. 
gilahyper:212652:214161 [0] NCCL INFO Failed to open libibverbs.so[.1]
gilahyper:212652:214161 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:212652:214161 [0] NCCL INFO NET/Socket : Using [0]enp37s0f0:192.168.1.15<0> [1]veth46046cc:fe80::c81e:2aff:fe27:6ad0%veth46046cc<0> [2]veth449c143:fe80::4475:85ff:fe90:2681%veth449c143<0>
gilahyper:212652:214161 [0] NCCL INFO Initialized NET plugin Socket
gilahyper:212652:214161 [0] NCCL INFO Assigned NET plugin Socket to comm
gilahyper:212652:214161 [0] NCCL INFO Using network Socket
gilahyper:212652:214161 [0] NCCL INFO ncclCommInitRankConfig comm 0x1cc2ab30 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 1000 commId 0x87fefd9cae059a89 - Init START
devices: 4
SLURM_JOB_NUM_NODES: 1
SLURM_NNODES: 1
SLURM_NPROCS: 1
Starting training (2025-08-29-10-45-08)
gilahyper:212653:214164 [1] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. 
gilahyper:212653:214164 [1] NCCL INFO Failed to open libibverbs.so[.1]
gilahyper:212653:214164 [1] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:212653:214164 [1] NCCL INFO NET/Socket : Using [0]enp37s0f0:192.168.1.15<0> [1]veth46046cc:fe80::c81e:2aff:fe27:6ad0%veth46046cc<0> [2]veth449c143:fe80::4475:85ff:fe90:2681%veth449c143<0>
gilahyper:212653:214164 [1] NCCL INFO Initialized NET plugin Socket
gilahyper:212653:214164 [1] NCCL INFO Assigned NET plugin Socket to comm
gilahyper:212653:214164 [1] NCCL INFO Using network Socket
gilahyper:212653:214164 [1] NCCL INFO ncclCommInitRankConfig comm 0x1e364560 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 2b000 commId 0x87fefd9cae059a89 - Init START
gilahyper:212654:212654 [2] NCCL INFO cudaDriverVersion 12080
gilahyper:212655:212655 [3] NCCL INFO cudaDriverVersion 12080
gilahyper:212654:212654 [2] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:212654:212654 [2] NCCL INFO Bootstrap: Using enp37s0f0:192.168.1.15<0>
gilahyper:212654:212654 [2] NCCL INFO NCCL version 2.27.3+cuda12.9
gilahyper:212655:212655 [3] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:212655:212655 [3] NCCL INFO Bootstrap: Using enp37s0f0:192.168.1.15<0>
gilahyper:212655:212655 [3] NCCL INFO NCCL version 2.27.3+cuda12.9
gilahyper:212654:212654 [2] NCCL INFO Comm config Blocking set to 1
gilahyper:212655:212655 [3] NCCL INFO Comm config Blocking set to 1
gilahyper:212655:214174 [3] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. 
gilahyper:212655:214174 [3] NCCL INFO Failed to open libibverbs.so[.1]
gilahyper:212655:214174 [3] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:212655:214174 [3] NCCL INFO NET/Socket : Using [0]enp37s0f0:192.168.1.15<0> [1]veth46046cc:fe80::c81e:2aff:fe27:6ad0%veth46046cc<0> [2]veth449c143:fe80::4475:85ff:fe90:2681%veth449c143<0>
gilahyper:212655:214174 [3] NCCL INFO Initialized NET plugin Socket
gilahyper:212655:214174 [3] NCCL INFO Assigned NET plugin Socket to comm
gilahyper:212655:214174 [3] NCCL INFO Using network Socket
gilahyper:212654:214173 [2] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. 
gilahyper:212655:214174 [3] NCCL INFO ncclCommInitRankConfig comm 0x1db6b500 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 61000 commId 0x87fefd9cae059a89 - Init START
gilahyper:212654:214173 [2] NCCL INFO Failed to open libibverbs.so[.1]
gilahyper:212654:214173 [2] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:212654:214173 [2] NCCL INFO NET/Socket : Using [0]enp37s0f0:192.168.1.15<0> [1]veth46046cc:fe80::c81e:2aff:fe27:6ad0%veth46046cc<0> [2]veth449c143:fe80::4475:85ff:fe90:2681%veth449c143<0>
gilahyper:212654:214173 [2] NCCL INFO Initialized NET plugin Socket
gilahyper:212654:214173 [2] NCCL INFO Assigned NET plugin Socket to comm
gilahyper:212654:214173 [2] NCCL INFO Using network Socket
gilahyper:212652:214161 [0] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
gilahyper:212654:214173 [2] NCCL INFO ncclCommInitRankConfig comm 0x1a8dd050 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 41000 commId 0x87fefd9cae059a89 - Init START
gilahyper:212653:214164 [1] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
gilahyper:212655:214174 [3] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
gilahyper:212654:214173 [2] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
gilahyper:212653:214164 [1] NCCL INFO Bootstrap timings total 0.140013 (create 0.000036, send 0.000150, recv 0.139316, ring 0.000170, delay 0.000001)
gilahyper:212652:214161 [0] NCCL INFO Bootstrap timings total 0.286448 (create 0.000034, send 0.000188, recv 0.146630, ring 0.001405, delay 0.000000)
gilahyper:212655:214174 [3] NCCL INFO Bootstrap timings total 0.002213 (create 0.000035, send 0.000141, recv 0.000207, ring 0.000181, delay 0.000000)
gilahyper:212654:214173 [2] NCCL INFO Bootstrap timings total 0.000783 (create 0.000028, send 0.000099, recv 0.000208, ring 0.000072, delay 0.000000)
gilahyper:212652:214161 [0] NCCL INFO NVLS multicast support is not available on dev 0 (NVLS_NCHANNELS 0)
gilahyper:212654:214173 [2] NCCL INFO NVLS multicast support is not available on dev 2 (NVLS_NCHANNELS 0)
gilahyper:212655:214174 [3] NCCL INFO NVLS multicast support is not available on dev 3 (NVLS_NCHANNELS 0)
gilahyper:212653:214164 [1] NCCL INFO NVLS multicast support is not available on dev 1 (NVLS_NCHANNELS 0)
gilahyper:212655:214174 [3] NCCL INFO comm 0x1db6b500 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0
gilahyper:212654:214173 [2] NCCL INFO comm 0x1a8dd050 rank 2 nRanks 4 nNodes 1 localRanks 4 localRank 2 MNNVL 0
gilahyper:212652:214161 [0] NCCL INFO comm 0x1cc2ab30 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
gilahyper:212653:214164 [1] NCCL INFO comm 0x1e364560 rank 1 nRanks 4 nNodes 1 localRanks 4 localRank 1 MNNVL 0
gilahyper:212655:214174 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] -1/-1/-1->3->2 [2] -1/-1/-1->3->2 [3] -1/-1/-1->3->2
gilahyper:212654:214173 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1 [2] 3/-1/-1->2->1 [3] 3/-1/-1->2->1
gilahyper:212652:214161 [0] NCCL INFO Channel 00/04 : 0 1 2 3
gilahyper:212655:214174 [3] NCCL INFO P2P Chunksize set to 131072
gilahyper:212654:214173 [2] NCCL INFO P2P Chunksize set to 131072
gilahyper:212653:214164 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0 [2] 2/-1/-1->1->0 [3] 2/-1/-1->1->0
gilahyper:212652:214161 [0] NCCL INFO Channel 01/04 : 0 1 2 3
gilahyper:212653:214164 [1] NCCL INFO P2P Chunksize set to 131072
gilahyper:212652:214161 [0] NCCL INFO Channel 02/04 : 0 1 2 3
gilahyper:212652:214161 [0] NCCL INFO Channel 03/04 : 0 1 2 3
gilahyper:212652:214161 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1 [2] 1/-1/-1->0->-1 [3] 1/-1/-1->0->-1
gilahyper:212652:214161 [0] NCCL INFO P2P Chunksize set to 131072
gilahyper:212653:214164 [1] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
gilahyper:212655:214174 [3] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
gilahyper:212654:214173 [2] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
gilahyper:212652:214161 [0] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
gilahyper:212652:214161 [0] NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0
gilahyper:212652:214182 [0] NCCL INFO [Proxy Service] Device 0 CPU core 65
gilahyper:212655:214183 [3] NCCL INFO [Proxy Service UDS] Device 3 CPU core 66
gilahyper:212655:214179 [3] NCCL INFO [Proxy Service] Device 3 CPU core 71
gilahyper:212652:214185 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 4
gilahyper:212654:214181 [2] NCCL INFO [Proxy Service] Device 2 CPU core 67
gilahyper:212654:214184 [2] NCCL INFO [Proxy Service UDS] Device 2 CPU core 64
gilahyper:212653:214180 [1] NCCL INFO [Proxy Service] Device 1 CPU core 65
gilahyper:212653:214186 [1] NCCL INFO [Proxy Service UDS] Device 1 CPU core 71
gilahyper:212653:214164 [1] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
gilahyper:212654:214173 [2] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
gilahyper:212653:214164 [1] NCCL INFO 4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels, 2 p2p channels per peer
gilahyper:212654:214173 [2] NCCL INFO 4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels, 2 p2p channels per peer
gilahyper:212655:214174 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
gilahyper:212655:214174 [3] NCCL INFO 4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels, 2 p2p channels per peer
gilahyper:212652:214161 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
gilahyper:212652:214161 [0] NCCL INFO 4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels, 2 p2p channels per peer
gilahyper:212652:214161 [0] NCCL INFO CC Off, workFifoBytes 1048576
gilahyper:212652:214161 [0] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
gilahyper:212655:214174 [3] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
gilahyper:212653:214164 [1] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
gilahyper:212652:214161 [0] NCCL INFO ncclCommInitRankConfig comm 0x1cc2ab30 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 1000 commId 0x87fefd9cae059a89 - Init COMPLETE
gilahyper:212653:214164 [1] NCCL INFO ncclCommInitRankConfig comm 0x1e364560 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 2b000 commId 0x87fefd9cae059a89 - Init COMPLETE
gilahyper:212655:214174 [3] NCCL INFO ncclCommInitRankConfig comm 0x1db6b500 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 61000 commId 0x87fefd9cae059a89 - Init COMPLETE
gilahyper:212652:214161 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 4 total 0.39 (kernels 0.09, alloc 0.00, bootstrap 0.29, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
gilahyper:212653:214164 [1] NCCL INFO Init timings - ncclCommInitRankConfig: rank 1 nranks 4 total 0.31 (kernels 0.15, alloc 0.00, bootstrap 0.14, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
gilahyper:212655:214174 [3] NCCL INFO Init timings - ncclCommInitRankConfig: rank 3 nranks 4 total 0.11 (kernels 0.09, alloc 0.00, bootstrap 0.00, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
gilahyper:212654:214173 [2] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
gilahyper:212654:214173 [2] NCCL INFO ncclCommInitRankConfig comm 0x1a8dd050 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 41000 commId 0x87fefd9cae059a89 - Init COMPLETE
gilahyper:212654:214173 [2] NCCL INFO Init timings - ncclCommInitRankConfig: rank 2 nranks 4 total 0.11 (kernels 0.09, alloc 0.00, bootstrap 0.00, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
gilahyper:212653:214187 [1] NCCL INFO Channel 00 : 1[1] -> 2[2] via SHM/direct/direct
gilahyper:212655:214189 [3] NCCL INFO Channel 00 : 3[3] -> 0[0] via SHM/direct/direct
gilahyper:212654:214188 [2] NCCL INFO Channel 00 : 2[2] -> 3[3] via SHM/direct/direct
gilahyper:212652:214190 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
gilahyper:212655:214189 [3] NCCL INFO Channel 01 : 3[3] -> 0[0] via SHM/direct/direct
gilahyper:212652:214190 [0] NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
gilahyper:212654:214188 [2] NCCL INFO Channel 01 : 2[2] -> 3[3] via SHM/direct/direct
gilahyper:212653:214187 [1] NCCL INFO Channel 01 : 1[1] -> 2[2] via SHM/direct/direct
gilahyper:212653:214187 [1] NCCL INFO Channel 02 : 1[1] -> 2[2] via SHM/direct/direct
gilahyper:212652:214190 [0] NCCL INFO Channel 02 : 0[0] -> 1[1] via SHM/direct/direct
gilahyper:212655:214189 [3] NCCL INFO Channel 02 : 3[3] -> 0[0] via SHM/direct/direct
gilahyper:212654:214188 [2] NCCL INFO Channel 02 : 2[2] -> 3[3] via SHM/direct/direct
gilahyper:212653:214187 [1] NCCL INFO Channel 03 : 1[1] -> 2[2] via SHM/direct/direct
gilahyper:212655:214189 [3] NCCL INFO Channel 03 : 3[3] -> 0[0] via SHM/direct/direct
gilahyper:212652:214190 [0] NCCL INFO Channel 03 : 0[0] -> 1[1] via SHM/direct/direct
gilahyper:212654:214188 [2] NCCL INFO Channel 03 : 2[2] -> 3[3] via SHM/direct/direct
gilahyper:212652:214190 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
gilahyper:212654:214188 [2] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
gilahyper:212653:214187 [1] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
gilahyper:212655:214189 [3] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
[2025-08-29 10:45:10,687][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 10:45:10,693][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 10:45:10,693][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 10:45:10,704][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 10:45:10,750][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 10:45:10,750][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 10:45:10,757][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 10:45:10,757][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 10:45:10,757][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 10:45:10,757][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 10:45:10,768][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 10:45:10,768][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 10:45:10,777][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
[2025-08-29 10:45:10,785][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
[2025-08-29 10:45:10,786][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
[2025-08-29 10:45:10,796][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038

Sanity Checking: |          | 0/? [00:00<?, ?it/s]
Sanity Checking:   0%|          | 0/2 [00:00<?, ?it/s]
Sanity Checking DataLoader 0:   0%|          | 0/2 [00:00<?, ?it/s]
Sanity Checking DataLoader 0:  50%|█████     | 1/2 [01:07<01:07,  0.01it/s]
Sanity Checking DataLoader 0: 100%|██████████| 2/2 [02:24<00:00,  0.01it/s]
                                                                           

Training: |          | 0/? [00:00<?, ?it/s]
Training:   0%|          | 0/260 [00:00<?, ?it/s]
Epoch 0:   0%|          | 0/260 [00:00<?, ?it/s] 
Epoch 0:   0%|          | 1/260 [10:35<45:43:02,  0.00it/s]
Epoch 0:   0%|          | 1/260 [10:35<45:43:02,  0.00it/s, v_num=2wlj]
Epoch 0:   1%|          | 2/260 [12:08<26:05:13,  0.00it/s, v_num=2wlj]
Epoch 0:   1%|          | 2/260 [12:08<26:05:13,  0.00it/s, v_num=2wlj]
Epoch 0:   1%|          | 3/260 [13:44<19:36:36,  0.00it/s, v_num=2wlj]
Epoch 0:   1%|          | 3/260 [13:44<19:36:36,  0.00it/s, v_num=2wlj]
Epoch 0:   2%|▏         | 4/260 [15:18<16:19:26,  0.00it/s, v_num=2wlj]
Epoch 0:   2%|▏         | 4/260 [15:18<16:19:26,  0.00it/s, v_num=2wlj]
Epoch 0:   2%|▏         | 5/260 [16:45<14:15:05,  0.00it/s, v_num=2wlj]
Epoch 0:   2%|▏         | 5/260 [16:45<14:15:05,  0.00it/s, v_num=2wlj]
Epoch 0:   2%|▏         | 6/260 [18:00<12:42:25,  0.01it/s, v_num=2wlj]
Epoch 0:   2%|▏         | 6/260 [18:00<12:42:25,  0.01it/s, v_num=2wlj]
Epoch 0:   3%|▎         | 7/260 [19:25<11:42:16,  0.01it/s, v_num=2wlj]
Epoch 0:   3%|▎         | 7/260 [19:25<11:42:16,  0.01it/s, v_num=2wlj]
Epoch 0:   3%|▎         | 8/260 [21:08<11:05:47,  0.01it/s, v_num=2wlj]
Epoch 0:   3%|▎         | 8/260 [21:08<11:05:47,  0.01it/s, v_num=2wlj]
Epoch 0:   3%|▎         | 9/260 [22:43<10:33:57,  0.01it/s, v_num=2wlj]
Epoch 0:   3%|▎         | 9/260 [22:43<10:33:58,  0.01it/s, v_num=2wlj]
Epoch 0:   4%|▍         | 10/260 [24:07<10:03:04,  0.01it/s, v_num=2wlj]
Epoch 0:   4%|▍         | 10/260 [24:07<10:03:04,  0.01it/s, v_num=2wlj]
Epoch 0:   4%|▍         | 11/260 [25:11<9:30:09,  0.01it/s, v_num=2wlj] 
Epoch 0:   4%|▍         | 11/260 [25:11<9:30:09,  0.01it/s, v_num=2wlj]
Epoch 0:   5%|▍         | 12/260 [26:10<9:00:54,  0.01it/s, v_num=2wlj]
Epoch 0:   5%|▍         | 12/260 [26:10<9:00:54,  0.01it/s, v_num=2wlj]
Epoch 0:   5%|▌         | 13/260 [27:14<8:37:28,  0.01it/s, v_num=2wlj]
Epoch 0:   5%|▌         | 13/260 [27:14<8:37:28,  0.01it/s, v_num=2wlj]
Epoch 0:   5%|▌         | 14/260 [28:31<8:21:15,  0.01it/s, v_num=2wlj]
Epoch 0:   5%|▌         | 14/260 [28:31<8:21:15,  0.01it/s, v_num=2wlj]
Epoch 0:   6%|▌         | 15/260 [29:47<8:06:37,  0.01it/s, v_num=2wlj]
Epoch 0:   6%|▌         | 15/260 [29:47<8:06:37,  0.01it/s, 
```

## BF16-Mixed Precison - 12 workers

```python
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 12, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': 'default', 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 12, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': 'default', 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 12, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': 'default', 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 256, 'num_workers': 12, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': 'default', 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
[2025-08-29 11:34:38,240][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 11:34:38,357][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 11:34:38,552][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 11:34:38,593][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 11:34:40,234][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
[2025-08-29 11:34:40,354][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
[2025-08-29 11:34:40,598][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
[2025-08-29 11:34:40,654][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
[2025-08-29 11:34:42,815][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
[2025-08-29 11:34:42,932][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
[2025-08-29 11:34:43,188][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
[2025-08-29 11:34:43,224][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 11:34:44,945][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 11:34:44,985][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-29 11:34:45,032][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-29 11:34:45,060][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-11-34-45)
Instantiating DCell
[2025-08-29 11:34:45,078][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-29 11:34:45,147][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-11-34-45)
Instantiating DCell
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 11:34:45,299][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 11:34:45,341][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-29 11:34:45,349][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-29 11:34:45,391][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-29 11:34:45,413][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-11-34-45)
Instantiating DCell
[2025-08-29 11:34:45,465][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-11-34-45)
Instantiating DCell
Pre-computing gene indices for 2655 GO terms...
Pre-computing gene indices for 2655 GO terms...
Pre-computing gene indices for 2655 GO terms...
Pre-computing gene indices for 2655 GO terms...
Pre-computed indices in 1.24 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Pre-computed indices in 1.32 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Pre-computed indices in 1.47 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Pre-computed indices in 1.47 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-11-34-55)
Attempting torch.compile optimization (PyTorch 2.8.0+cu128)...
  Mode: default, Dynamic: True
  Set precompilation timeout to 2 hours
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-11-34-55)
Attempting torch.compile optimization (PyTorch 2.8.0+cu128)...
  Mode: default, Dynamic: True
  Set precompilation timeout to 2 hours
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-11-34-56)
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-11-34-56)
Attempting torch.compile optimization (PyTorch 2.8.0+cu128)...
  Mode: default, Dynamic: True
  Set precompilation timeout to 2 hours
Attempting torch.compile optimization (PyTorch 2.8.0+cu128)...
  Mode: default, Dynamic: True
  Set precompilation timeout to 2 hours
✓ Successfully compiled model with torch.compile (mode=default, dynamic=True)
  Expected speedup: 2-3x for forward/backward passes
devices: 4
SLURM_JOB_NUM_NODES: 1
SLURM_NNODES: 1
SLURM_NPROCS: 1
Starting training (2025-08-29-11-34-56)
✓ Successfully compiled model with torch.compile (mode=default, dynamic=True)
  Expected speedup: 2-3x for forward/backward passes
devices: 4
SLURM_JOB_NUM_NODES: 1
SLURM_NNODES: 1
SLURM_NPROCS: 1
Starting training (2025-08-29-11-34-57)
✓ Successfully compiled model with torch.compile (mode=default, dynamic=True)
  Expected speedup: 2-3x for forward/backward passes
devices: 4
SLURM_JOB_NUM_NODES: 1
SLURM_NNODES: 1
SLURM_NPROCS: 1
Starting training (2025-08-29-11-34-57)
✓ Successfully compiled model with torch.compile (mode=default, dynamic=True)
  Expected speedup: 2-3x for forward/backward passes
devices: 4
SLURM_JOB_NUM_NODES: 1
SLURM_NNODES: 1
SLURM_NPROCS: 1
Starting training (2025-08-29-11-34-57)
gilahyper:255605:255605 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:255605:255605 [0] NCCL INFO Bootstrap: Using enp37s0f0:192.168.1.15<0>
gilahyper:255605:255605 [0] NCCL INFO cudaDriverVersion 12080
gilahyper:255605:255605 [0] NCCL INFO NCCL version 2.27.3+cuda12.9
gilahyper:255605:255605 [0] NCCL INFO Comm config Blocking set to 1
gilahyper:255606:255606 [1] NCCL INFO cudaDriverVersion 12080
gilahyper:255606:255606 [1] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:255606:255606 [1] NCCL INFO Bootstrap: Using enp37s0f0:192.168.1.15<0>
gilahyper:255606:255606 [1] NCCL INFO NCCL version 2.27.3+cuda12.9
gilahyper:255606:255606 [1] NCCL INFO Comm config Blocking set to 1
gilahyper:255607:255607 [2] NCCL INFO cudaDriverVersion 12080
gilahyper:255607:255607 [2] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:255607:255607 [2] NCCL INFO Bootstrap: Using enp37s0f0:192.168.1.15<0>
gilahyper:255607:255607 [2] NCCL INFO NCCL version 2.27.3+cuda12.9
gilahyper:255607:255607 [2] NCCL INFO Comm config Blocking set to 1
gilahyper:255605:256766 [0] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. 
gilahyper:255605:256766 [0] NCCL INFO Failed to open libibverbs.so[.1]
gilahyper:255605:256766 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:255605:256766 [0] NCCL INFO NET/Socket : Using [0]enp37s0f0:192.168.1.15<0> [1]veth46046cc:fe80::c81e:2aff:fe27:6ad0%veth46046cc<0> [2]veth449c143:fe80::4475:85ff:fe90:2681%veth449c143<0>
gilahyper:255605:256766 [0] NCCL INFO Initialized NET plugin Socket
gilahyper:255605:256766 [0] NCCL INFO Assigned NET plugin Socket to comm
gilahyper:255605:256766 [0] NCCL INFO Using network Socket
gilahyper:255605:256766 [0] NCCL INFO ncclCommInitRankConfig comm 0x1b8dc6e0 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 1000 commId 0xf0ddb20a292e801e - Init START
gilahyper:255608:255608 [3] NCCL INFO cudaDriverVersion 12080
gilahyper:255608:255608 [3] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:255608:255608 [3] NCCL INFO Bootstrap: Using enp37s0f0:192.168.1.15<0>
gilahyper:255608:255608 [3] NCCL INFO NCCL version 2.27.3+cuda12.9
gilahyper:255608:255608 [3] NCCL INFO Comm config Blocking set to 1
gilahyper:255606:256768 [1] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. 
gilahyper:255606:256768 [1] NCCL INFO Failed to open libibverbs.so[.1]
gilahyper:255606:256768 [1] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:255606:256768 [1] NCCL INFO NET/Socket : Using [0]enp37s0f0:192.168.1.15<0> [1]veth46046cc:fe80::c81e:2aff:fe27:6ad0%veth46046cc<0> [2]veth449c143:fe80::4475:85ff:fe90:2681%veth449c143<0>
gilahyper:255606:256768 [1] NCCL INFO Initialized NET plugin Socket
gilahyper:255606:256768 [1] NCCL INFO Assigned NET plugin Socket to comm
gilahyper:255606:256768 [1] NCCL INFO Using network Socket
gilahyper:255606:256768 [1] NCCL INFO ncclCommInitRankConfig comm 0x1bc4b290 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 2b000 commId 0xf0ddb20a292e801e - Init START
gilahyper:255607:256769 [2] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. 
gilahyper:255607:256769 [2] NCCL INFO Failed to open libibverbs.so[.1]
gilahyper:255607:256769 [2] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:255607:256769 [2] NCCL INFO NET/Socket : Using [0]enp37s0f0:192.168.1.15<0> [1]veth46046cc:fe80::c81e:2aff:fe27:6ad0%veth46046cc<0> [2]veth449c143:fe80::4475:85ff:fe90:2681%veth449c143<0>
gilahyper:255607:256769 [2] NCCL INFO Initialized NET plugin Socket
gilahyper:255607:256769 [2] NCCL INFO Assigned NET plugin Socket to comm
gilahyper:255607:256769 [2] NCCL INFO Using network Socket
gilahyper:255607:256769 [2] NCCL INFO ncclCommInitRankConfig comm 0x1a7d65a0 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 41000 commId 0xf0ddb20a292e801e - Init START
gilahyper:255606:256768 [1] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
gilahyper:255608:256772 [3] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. 
gilahyper:255608:256772 [3] NCCL INFO Failed to open libibverbs.so[.1]
gilahyper:255608:256772 [3] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
gilahyper:255608:256772 [3] NCCL INFO NET/Socket : Using [0]enp37s0f0:192.168.1.15<0> [1]veth46046cc:fe80::c81e:2aff:fe27:6ad0%veth46046cc<0> [2]veth449c143:fe80::4475:85ff:fe90:2681%veth449c143<0>
gilahyper:255608:256772 [3] NCCL INFO Initialized NET plugin Socket
gilahyper:255608:256772 [3] NCCL INFO Assigned NET plugin Socket to comm
gilahyper:255608:256772 [3] NCCL INFO Using network Socket
gilahyper:255608:256772 [3] NCCL INFO ncclCommInitRankConfig comm 0x1ba657b0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 61000 commId 0xf0ddb20a292e801e - Init START
gilahyper:255608:256772 [3] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
gilahyper:255607:256769 [2] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
gilahyper:255605:256766 [0] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
gilahyper:255608:256772 [3] NCCL INFO Bootstrap timings total 0.000970 (create 0.000035, send 0.000163, recv 0.000316, ring 0.000128, delay 0.000000)
gilahyper:255605:256766 [0] NCCL INFO Bootstrap timings total 0.105694 (create 0.000055, send 0.000263, recv 0.026814, ring 0.000102, delay 0.000001)
gilahyper:255607:256769 [2] NCCL INFO Bootstrap timings total 0.062779 (create 0.000041, send 0.000180, recv 0.062074, ring 0.000086, delay 0.000000)
gilahyper:255606:256768 [1] NCCL INFO Bootstrap timings total 0.080614 (create 0.000036, send 0.000171, recv 0.018117, ring 0.061799, delay 0.000000)
gilahyper:255608:256772 [3] NCCL INFO NVLS multicast support is not available on dev 3 (NVLS_NCHANNELS 0)
gilahyper:255607:256769 [2] NCCL INFO NVLS multicast support is not available on dev 2 (NVLS_NCHANNELS 0)
gilahyper:255606:256768 [1] NCCL INFO NVLS multicast support is not available on dev 1 (NVLS_NCHANNELS 0)
gilahyper:255605:256766 [0] NCCL INFO NVLS multicast support is not available on dev 0 (NVLS_NCHANNELS 0)
gilahyper:255606:256768 [1] NCCL INFO comm 0x1bc4b290 rank 1 nRanks 4 nNodes 1 localRanks 4 localRank 1 MNNVL 0
gilahyper:255608:256772 [3] NCCL INFO comm 0x1ba657b0 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0
gilahyper:255605:256766 [0] NCCL INFO comm 0x1b8dc6e0 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
gilahyper:255607:256769 [2] NCCL INFO comm 0x1a7d65a0 rank 2 nRanks 4 nNodes 1 localRanks 4 localRank 2 MNNVL 0
gilahyper:255608:256772 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] -1/-1/-1->3->2 [2] -1/-1/-1->3->2 [3] -1/-1/-1->3->2
gilahyper:255606:256768 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0 [2] 2/-1/-1->1->0 [3] 2/-1/-1->1->0
gilahyper:255605:256766 [0] NCCL INFO Channel 00/04 : 0 1 2 3
gilahyper:255608:256772 [3] NCCL INFO P2P Chunksize set to 131072
gilahyper:255607:256769 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1 [2] 3/-1/-1->2->1 [3] 3/-1/-1->2->1
gilahyper:255606:256768 [1] NCCL INFO P2P Chunksize set to 131072
gilahyper:255605:256766 [0] NCCL INFO Channel 01/04 : 0 1 2 3
gilahyper:255607:256769 [2] NCCL INFO P2P Chunksize set to 131072
gilahyper:255605:256766 [0] NCCL INFO Channel 02/04 : 0 1 2 3
gilahyper:255605:256766 [0] NCCL INFO Channel 03/04 : 0 1 2 3
gilahyper:255605:256766 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1 [2] 1/-1/-1->0->-1 [3] 1/-1/-1->0->-1
gilahyper:255605:256766 [0] NCCL INFO P2P Chunksize set to 131072
gilahyper:255608:256772 [3] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
gilahyper:255607:256769 [2] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
gilahyper:255606:256768 [1] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
gilahyper:255605:256766 [0] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
gilahyper:255605:256766 [0] NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0
gilahyper:255607:256813 [2] NCCL INFO [Proxy Service] Device 2 CPU core 7
gilahyper:255607:256814 [2] NCCL INFO [Proxy Service UDS] Device 2 CPU core 66
gilahyper:255606:256816 [1] NCCL INFO [Proxy Service] Device 1 CPU core 66
gilahyper:255608:256815 [3] NCCL INFO [Proxy Service] Device 3 CPU core 4
gilahyper:255608:256818 [3] NCCL INFO [Proxy Service UDS] Device 3 CPU core 70
gilahyper:255605:256812 [0] NCCL INFO [Proxy Service] Device 0 CPU core 65
gilahyper:255606:256817 [1] NCCL INFO [Proxy Service UDS] Device 1 CPU core 7
gilahyper:255605:256819 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 65
gilahyper:255606:256768 [1] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
gilahyper:255606:256768 [1] NCCL INFO 4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels, 2 p2p channels per peer
gilahyper:255605:256766 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
gilahyper:255605:256766 [0] NCCL INFO 4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels, 2 p2p channels per peer
gilahyper:255607:256769 [2] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
gilahyper:255607:256769 [2] NCCL INFO 4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels, 2 p2p channels per peer
gilahyper:255605:256766 [0] NCCL INFO CC Off, workFifoBytes 1048576
gilahyper:255608:256772 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
gilahyper:255608:256772 [3] NCCL INFO 4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels, 2 p2p channels per peer
gilahyper:255606:256768 [1] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
gilahyper:255608:256772 [3] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
gilahyper:255606:256768 [1] NCCL INFO ncclCommInitRankConfig comm 0x1bc4b290 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 2b000 commId 0xf0ddb20a292e801e - Init COMPLETE
gilahyper:255605:256766 [0] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
gilahyper:255607:256769 [2] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
gilahyper:255608:256772 [3] NCCL INFO ncclCommInitRankConfig comm 0x1ba657b0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 61000 commId 0xf0ddb20a292e801e - Init COMPLETE
gilahyper:255606:256768 [1] NCCL INFO Init timings - ncclCommInitRankConfig: rank 1 nranks 4 total 0.21 (kernels 0.10, alloc 0.00, bootstrap 0.08, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
gilahyper:255605:256766 [0] NCCL INFO ncclCommInitRankConfig comm 0x1b8dc6e0 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 1000 commId 0xf0ddb20a292e801e - Init COMPLETE
gilahyper:255607:256769 [2] NCCL INFO ncclCommInitRankConfig comm 0x1a7d65a0 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 41000 commId 0xf0ddb20a292e801e - Init COMPLETE
gilahyper:255608:256772 [3] NCCL INFO Init timings - ncclCommInitRankConfig: rank 3 nranks 4 total 0.11 (kernels 0.08, alloc 0.00, bootstrap 0.00, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
gilahyper:255605:256766 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 4 total 0.24 (kernels 0.10, alloc 0.01, bootstrap 0.11, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
gilahyper:255607:256769 [2] NCCL INFO Init timings - ncclCommInitRankConfig: rank 2 nranks 4 total 0.20 (kernels 0.11, alloc 0.00, bootstrap 0.06, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
gilahyper:255607:256820 [2] NCCL INFO Channel 00 : 2[2] -> 3[3] via SHM/direct/direct
gilahyper:255605:256822 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
gilahyper:255608:256821 [3] NCCL INFO Channel 00 : 3[3] -> 0[0] via SHM/direct/direct
gilahyper:255606:256823 [1] NCCL INFO Channel 00 : 1[1] -> 2[2] via SHM/direct/direct
gilahyper:255607:256820 [2] NCCL INFO Channel 01 : 2[2] -> 3[3] via SHM/direct/direct
gilahyper:255605:256822 [0] NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
gilahyper:255608:256821 [3] NCCL INFO Channel 01 : 3[3] -> 0[0] via SHM/direct/direct
gilahyper:255606:256823 [1] NCCL INFO Channel 01 : 1[1] -> 2[2] via SHM/direct/direct
gilahyper:255607:256820 [2] NCCL INFO Channel 02 : 2[2] -> 3[3] via SHM/direct/direct
gilahyper:255608:256821 [3] NCCL INFO Channel 02 : 3[3] -> 0[0] via SHM/direct/direct
gilahyper:255605:256822 [0] NCCL INFO Channel 02 : 0[0] -> 1[1] via SHM/direct/direct
gilahyper:255606:256823 [1] NCCL INFO Channel 02 : 1[1] -> 2[2] via SHM/direct/direct
gilahyper:255605:256822 [0] NCCL INFO Channel 03 : 0[0] -> 1[1] via SHM/direct/direct
gilahyper:255607:256820 [2] NCCL INFO Channel 03 : 2[2] -> 3[3] via SHM/direct/direct
gilahyper:255608:256821 [3] NCCL INFO Channel 03 : 3[3] -> 0[0] via SHM/direct/direct
gilahyper:255606:256823 [1] NCCL INFO Channel 03 : 1[1] -> 2[2] via SHM/direct/direct
gilahyper:255608:256821 [3] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
gilahyper:255605:256822 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
gilahyper:255607:256820 [2] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
gilahyper:255606:256823 [1] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
[2025-08-29 11:35:00,490][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 11:35:00,494][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 11:35:00,498][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 11:35:00,504][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 11:35:00,608][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 11:35:00,609][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 11:35:00,614][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 11:35:00,615][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 11:35:00,615][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 11:35:00,616][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 11:35:00,627][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 11:35:00,628][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 11:35:00,639][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
[2025-08-29 11:35:00,644][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
[2025-08-29 11:35:00,645][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
[2025-08-29 11:35:00,657][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038

Sanity Checking: |          | 0/? [00:00<?, ?it/s]
Sanity Checking:   0%|          | 0/2 [00:00<?, ?it/s]
Sanity Checking DataLoader 0:   0%|          | 0/2 [00:00<?, ?it/s]
Sanity Checking DataLoader 0:  50%|█████     | 1/2 [05:17<05:17,  0.00it/s]
Sanity Checking DataLoader 0: 100%|██████████| 2/2 [07:21<00:00,  0.00it/s]
                                                                           

Training: |          | 0/? [00:00<?, ?it/s]
Training:   0%|          | 0/260 [00:00<?, ?it/s]
Epoch 0:   0%|          | 0/260 [00:00<?, ?it/s] 
Epoch 0:   0%|          | 1/260 [13:23<57:48:21,  0.00it/s]
Epoch 0:   0%|          | 1/260 [13:23<57:48:24,  0.00it/s, v_num=8eh5]
Epoch 0:   1%|          | 2/260 [15:43<33:49:29,  0.00it/s, v_num=8eh5]
Epoch 0:   1%|          | 2/260 [15:43<33:49:29,  0.00it/s, v_num=8eh5]
Epoch 0:   1%|          | 3/260 [17:59<25:40:44,  0.00it/s, v_num=8eh5]
Epoch 0:   1%|          | 3/260 [17:59<25:40:45,  0.00it/s, v_num=8eh5]
Epoch 0:   2%|▏         | 4/260 [20:13<21:34:12,  0.00it/s, v_num=8eh5]
Epoch 0:   2%|▏         | 4/260 [20:13<21:34:12,  0.00it/s, v_num=8eh5]
Epoch 0:   2%|▏         | 5/260 [22:24<19:03:06,  0.00it/s, v_num=8eh5]
Epoch 0:   2%|▏         | 5/260 [22:24<19:03:06,  0.00it/s, v_num=8eh5]
Epoch 0:   2%|▏         | 6/260 [23:38<16:40:50,  0.00it/s, v_num=8eh5]
Epoch 0:   2%|▏         | 6/260 [23:38<16:40:50,  0.00it/s, v_num=8eh5]
Epoch 0:   3%|▎         | 7/260 [25:05<15:07:01,  0.00it/s, v_num=8eh5]
Epoch 0:   3%|▎         | 7/260 [25:05<15:07:01,  0.00it/s, v_num=8eh5]
Epoch 0:   3%|▎         | 8/260 [26:44<14:02:36,  0.00it/s, v_num=8eh5]
Epoch 0:   3%|▎         | 8/260 [26:44<14:02:36,  0.00it/s, v_num=8eh5]
Epoch 0:   3%|▎         | 9/260 [28:30<13:15:17,  0.01it/s, v_num=8eh5]
Epoch 0:   3%|▎         | 9/260 [28:30<13:15:17,  0.01it/s, v_num=8eh5]
Epoch 0:   4%|▍         | 10/260 [30:16<12:36:46,  0.01it/s, v_num=8eh5]
Epoch 0:   4%|▍         | 10/260 [30:16<12:36:46,  0.01it/s, v_num=8eh5]
Epoch 0:   4%|▍         | 11/260 [31:24<11:50:57,  0.01it/s, v_num=8eh5]
Epoch 0:   4%|▍         | 11/260 [31:24<11:50:57,  0.01it/s, v_num=8eh5]
Epoch 0:   5%|▍         | 12/260 [32:31<11:12:19,  0.01it/s, v_num=8eh5]
Epoch 0:   5%|▍         | 12/260 [32:31<11:12:19,  0.01it/s, v_num=8eh5]
Epoch 0:   5%|▌         | 13/260 [33:39<10:39:28,  0.01it/s, v_num=8eh5]
Epoch 0:   5%|▌         | 13/260 [33:39<10:39:28,  0.01it/s, v_num=8eh5]
Epoch 0:   5%|▌         | 14/260 [34:44<10:10:26,  0.01it/s, v_num=8eh5]
Epoch 0:   5%|▌         | 14/260 [34:44<10:10:26,  0.01it/s, v_num=8eh5]
Epoch 0:   6%|▌         | 15/260 [35:49<9:45:09,  0.01it/s, v_num=8eh5] 
Epoch 0:   6%|▌         | 15/260 [35:49<9:45:09,  0.01it/s, v_num=8eh5]
Epoch 0:   6%|▌         | 16/260 [36:50<9:21:48,  0.01it/s, v_num=8eh5]
Epoch 0:   6%|▌         | 16/260 [36:50<9:21:48,  0.01it/s, v_num=8eh5]
Epoch 0:   7%|▋         | 17/260 [37:55<9:02:00,  0.01it/s, v_num=8eh5]
Epoch 0:   7%|▋         | 17/260 [37:55<9:02:00,  0.01it/s, v_num=8eh5]
Epoch 0:   7%|▋         | 18/260 [39:00<8:44:31,  0.01it/s, v_num=8eh5]
Epoch 0:   7%|▋         | 18/260 [39:00<8:44:31,  0.01it/s, v_num=8eh5]
Epoch 0:   7%|▋         | 19/260 [40:02<8:27:55,  0.01it/s, v_num=8eh5]
Epoch 0:   7%|▋         | 19/260 [40:02<8:27:55,  0.01it/s, v_num=8eh5]
Epoch 0:   8%|▊         | 20/260 [41:12<8:14:24,  0.01it/s, v_num=8eh5]
Epoch 0:   8%|▊         | 20/260 [41:12<8:14:24,  0.01it/s, v_num=8eh5]
Epoch 0:   8%|▊         | 21/260 [42:12<8:00:19,  0.01it/s, v_num=8eh5]
Epoch 0:   8%|▊         | 21/260 [42:12<8:00:20,  0.01it/s, v_num=8eh5]
Epoch 0:   8%|▊         | 22/260 [43:27<7:50:11,  0.01it/s, v_num=8eh5]
Epoch 0:   8%|▊         | 22/260 [43:27<7:50:11,  0.01it/s, v_num=8eh5]
Epoch 0:   9%|▉         | 23/260 [44:24<7:37:34,  0.01it/s, v_num=8eh5]
Epoch 0:   9%|▉         | 23/260 [44:24<7:37:34,  0.01it/s, v_num=8eh5]
Epoch 0:   9%|▉         | 24/260 [45:21<7:26:04,  0.01it/s, v_num=8eh5]
Epoch 0:   9%|▉         | 24/260 [45:21<7:26:04,  0.01it/s, v_num=8eh5]
Epoch 0:  10%|▉         | 25/260 [46:24<7:16:15,  0.01it/s, v_num=8eh5]
Epoch 0:  10%|▉         | 25/260 [46:24<7:16:15,  0.01it/s, v_num=8eh5]
Epoch 0:  10%|█         | 26/260 [47:26<7:06:56,  0.01it/s, v_num=8eh5]
Epoch 0:  10%|█         | 26/260 [47:26<7:06:56,  0.01it/s, v_num=8eh5]
Epoch 0:  10%|█         | 27/260 [48:31<6:58:44,  0.01it/s, v_num=8eh5]
Epoch 0:  10%|█         | 27/260 [48:31<6:58:44,  0.01it/s, v_num=8eh5]
Epoch 0:  11%|█         | 28/260 [49:37<6:51:06,  0.01it/s, v_num=8eh5]
Epoch 0:  11%|█         | 28/260 [49:37<6:51:06,  0.01it/s, v_num=8eh5]
Epoch 0:  11%|█         | 29/260 [50:39<6:43:30,  0.01it/s, v_num=8eh5]
Epoch 0:  11%|█         | 29/260 [50:39<6:43:30,  0.01it/s, v_num=8eh5]
Epoch 0:  12%|█▏        | 30/260 [51:41<6:36:16,  0.01it/s, v_num=8eh5]
Epoch 0:  12%|█▏        | 30/260 [51:41<6:36:16,  0.01it/s, v_num=8eh5]
Epoch 0:  12%|█▏        | 31/260 [52:44<6:29:35,  0.01it/s, v_num=8eh5]
Epoch 0:  12%|█▏        | 31/260 [52:44<6:29:35,  0.01it/s, v_num=8eh5]
Epoch 0:  12%|█▏        | 32/260 [53:51<6:23:44,  0.01it/s, v_num=8eh5]
Epoch 0:  12%|█▏        | 32/260 [53:51<6:23:44,  0.01it/s, v_num=8eh5]
Epoch 0:  13%|█▎        | 33/260 [54:56<6:17:53,  0.01it/s, v_num=8eh5]
Epoch 0:  13%|█▎        | 33/260 [54:56<6:17:53,  0.01it/s, v_num=8eh5]
Epoch 0:  13%|█▎        | 34/260 [56:07<6:13:03,  0.01it/s, v_num=8eh5]
Epoch 0:  13%|█▎        | 34/260 [56:07<6:13:03,  0.01it/s, v_num=8eh5]
```

## BF16-Mixed Precison - 8 Workers - Compile - batch size 500

```python
(torchcell) michaelvolk@gilahyper torchcell % python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 500, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': 'default', 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: Using wandb-core as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-e6b53f38-2ee6-47cf-8d01-4116e2154f0f_e1644690a617cde81fccfb694eb7897c5f20e282f0317998a4ec9051a0b56afd/wandb/run-20250829_163956-ggktoaxp
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-e6b53f38-2ee6-47cf-8d01-4116e2154f0f_e1644690a617cde81fccfb694eb7897c5f20e282f0317998a4ec9051a0b56afd
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/ggktoaxp
Using Gene Ontology hierarchy
[2025-08-29 16:40:01,945][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 16:40:03,750][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
[2025-08-29 16:40:06,200][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 16:40:08,402][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-29 16:40:08,445][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-29 16:40:08,520][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-16-40-08)
Instantiating DCell
Pre-computing gene indices for 2655 GO terms...
Pre-computed indices in 0.18 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-16-40-11)
Attempting torch.compile optimization (PyTorch 2.8.0+cu128)...
  Mode: default, Dynamic: True
  Set precompilation timeout to 2 hours
✓ Successfully compiled model with torch.compile (mode=default, dynamic=True)
  Expected speedup: 2-3x for forward/backward passes
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-29-16-40-13)
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/fabric/plugins/environments/slurm.py:204: The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python /home/michaelvolk/Documents/projects/torchcell/exper ...
Using bfloat16 Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 500, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': 'default', 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 500, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': 'default', 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 500, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': 'default', 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-ec1b198b-a47e-4eff-82b3-04258d87c600_e1644690a617cde81fccfb694eb7897c5f20e282f0317998a4ec9051a0b56afd/wandb/run-20250829_164029-vdh8olfm
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-ec1b198b-a47e-4eff-82b3-04258d87c600_e1644690a617cde81fccfb694eb7897c5f20e282f0317998a4ec9051a0b56afd
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/vdh8olfm
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-fd0ed48d-71ae-4ee0-908b-36a5796b2520_e1644690a617cde81fccfb694eb7897c5f20e282f0317998a4ec9051a0b56afd/wandb/run-20250829_164030-mjy9z2xl
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-fd0ed48d-71ae-4ee0-908b-36a5796b2520_e1644690a617cde81fccfb694eb7897c5f20e282f0317998a4ec9051a0b56afd
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/mjy9z2xl
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-978a0c70-d676-4968-aa2d-4be296cee147_e1644690a617cde81fccfb694eb7897c5f20e282f0317998a4ec9051a0b56afd/wandb/run-20250829_164030-z3iu0m5d
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-978a0c70-d676-4968-aa2d-4be296cee147_e1644690a617cde81fccfb694eb7897c5f20e282f0317998a4ec9051a0b56afd
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/z3iu0m5d
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
[2025-08-29 16:40:35,436][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 16:40:35,571][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 16:40:35,666][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 16:40:37,278][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
[2025-08-29 16:40:37,396][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
[2025-08-29 16:40:37,503][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
[2025-08-29 16:40:39,714][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
[2025-08-29 16:40:39,863][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
[2025-08-29 16:40:39,935][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 16:40:41,756][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-29 16:40:41,799][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 16:40:41,872][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-16-40-41)
Instantiating DCell
[2025-08-29 16:40:41,912][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 16:40:41,959][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-29 16:40:41,999][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-29 16:40:42,035][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-16-40-42)
Instantiating DCell
[2025-08-29 16:40:42,044][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-29 16:40:42,120][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-16-40-42)
Instantiating DCell
Pre-computing gene indices for 2655 GO terms...
Pre-computing gene indices for 2655 GO terms...
Pre-computing gene indices for 2655 GO terms...
Pre-computed indices in 0.89 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Pre-computed indices in 1.09 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Pre-computed indices in 1.10 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-16-40-50)
Attempting torch.compile optimization (PyTorch 2.8.0+cu128)...
  Mode: default, Dynamic: True
  Set precompilation timeout to 2 hours
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-16-40-50)
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-16-40-50)
Attempting torch.compile optimization (PyTorch 2.8.0+cu128)...
  Mode: default, Dynamic: True
  Set precompilation timeout to 2 hours
Attempting torch.compile optimization (PyTorch 2.8.0+cu128)...
  Mode: default, Dynamic: True
  Set precompilation timeout to 2 hours
✓ Successfully compiled model with torch.compile (mode=default, dynamic=True)
  Expected speedup: 2-3x for forward/backward passes
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-29-16-40-51)
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/4
✓ Successfully compiled model with torch.compile (mode=default, dynamic=True)
  Expected speedup: 2-3x for forward/backward passes
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-29-16-40-52)
✓ Successfully compiled model with torch.compile (mode=default, dynamic=True)
  Expected speedup: 2-3x for forward/backward passes
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-29-16-40-52)
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/4
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/4
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 4 processes
----------------------------------------------------------------------------------------------------

/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loggers/wandb.py:396: There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`.
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
[2025-08-29 16:40:54,388][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 16:40:54,390][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 16:40:54,393][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 16:40:54,409][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 16:40:54,512][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 16:40:54,512][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 16:40:54,525][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 16:40:54,525][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 16:40:54,526][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 16:40:54,526][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 16:40:54,534][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 16:40:54,535][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 16:40:54,538][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
[2025-08-29 16:40:54,551][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
[2025-08-29 16:40:54,552][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
[2025-08-29 16:40:54,560][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/core/optimizer.py:316: The lr scheduler dict contains the key(s) ['monitor'], but the keys will be ignored. You need to call `lr_scheduler.step()` manually in manual optimization.

  | Name                      | Type             | Params | Mode 
-----------------------------------------------------------------------
0 | model                     | OptimizedModule  | 20.6 M | train
1 | loss_func                 | DCellLoss        | 0      | train
2 | train_metrics             | MetricCollection | 0      | train
3 | train_transformed_metrics | MetricCollection | 0      | train
4 | val_metrics               | MetricCollection | 0      | train
5 | val_transformed_metrics   | MetricCollection | 0      | train
6 | test_metrics              | MetricCollection | 0      | train
7 | test_transformed_metrics  | MetricCollection | 0      | train
-----------------------------------------------------------------------
20.6 M    Trainable params
0         Non-trainable params
20.6 M    Total params
82.452    Total estimated model params size (MB)
13305     Modules in train mode
0         Modules in eval mode
Sanity Checking DataLoader 0:   0%|                                                                                                                                                                                                                                                    | 0/2 [00:00<?, ?it/s][rank1]:W0829 16:47:33.766000 2014105 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break from `Tensor.item()`, consider setting:
[rank1]:W0829 16:47:33.766000 2014105 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     torch._dynamo.config.capture_scalar_outputs = True
[rank1]:W0829 16:47:33.766000 2014105 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] or:
[rank1]:W0829 16:47:33.766000 2014105 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
[rank1]:W0829 16:47:33.766000 2014105 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] to include these operations in the captured graph.
[rank1]:W0829 16:47:33.766000 2014105 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank1]:W0829 16:47:33.766000 2014105 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break: from user code at:
[rank1]:W0829 16:47:33.766000 2014105 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]   File "/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py", line 292, in forward
[rank1]:W0829 16:47:33.766000 2014105 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     term_idx_int = int(term_idx)
[rank1]:W0829 16:47:33.766000 2014105 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank1]:W0829 16:47:33.766000 2014105 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank2]:W0829 16:47:33.772000 2014106 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break from `Tensor.item()`, consider setting:
[rank2]:W0829 16:47:33.772000 2014106 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     torch._dynamo.config.capture_scalar_outputs = True
[rank2]:W0829 16:47:33.772000 2014106 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] or:
[rank2]:W0829 16:47:33.772000 2014106 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
[rank2]:W0829 16:47:33.772000 2014106 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] to include these operations in the captured graph.
[rank2]:W0829 16:47:33.772000 2014106 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank2]:W0829 16:47:33.772000 2014106 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break: from user code at:
[rank2]:W0829 16:47:33.772000 2014106 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]   File "/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py", line 292, in forward
[rank2]:W0829 16:47:33.772000 2014106 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     term_idx_int = int(term_idx)
[rank2]:W0829 16:47:33.772000 2014106 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank2]:W0829 16:47:33.772000 2014106 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank0]:W0829 16:47:33.815000 2011461 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break from `Tensor.item()`, consider setting:
[rank0]:W0829 16:47:33.815000 2011461 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     torch._dynamo.config.capture_scalar_outputs = True
[rank0]:W0829 16:47:33.815000 2011461 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] or:
[rank0]:W0829 16:47:33.815000 2011461 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
[rank0]:W0829 16:47:33.815000 2011461 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] to include these operations in the captured graph.
[rank0]:W0829 16:47:33.815000 2011461 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank0]:W0829 16:47:33.815000 2011461 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break: from user code at:
[rank0]:W0829 16:47:33.815000 2011461 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]   File "/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py", line 292, in forward
[rank0]:W0829 16:47:33.815000 2011461 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     term_idx_int = int(term_idx)
[rank0]:W0829 16:47:33.815000 2011461 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank0]:W0829 16:47:33.815000 2011461 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank3]:W0829 16:47:35.885000 2014107 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break from `Tensor.item()`, consider setting:
[rank3]:W0829 16:47:35.885000 2014107 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     torch._dynamo.config.capture_scalar_outputs = True
[rank3]:W0829 16:47:35.885000 2014107 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] or:
[rank3]:W0829 16:47:35.885000 2014107 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
[rank3]:W0829 16:47:35.885000 2014107 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] to include these operations in the captured graph.
[rank3]:W0829 16:47:35.885000 2014107 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank3]:W0829 16:47:35.885000 2014107 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break: from user code at:
[rank3]:W0829 16:47:35.885000 2014107 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]   File "/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py", line 292, in forward
[rank3]:W0829 16:47:35.885000 2014107 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     term_idx_int = int(term_idx)
[rank3]:W0829 16:47:35.885000 2014107 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank3]:W0829 16:47:35.885000 2014107 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank2]:W0829 16:47:48.404000 2014106 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] torch._dynamo hit config.recompile_limit (8)
[rank2]:W0829 16:47:48.404000 2014106 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    function: '_prepare_term_input' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:346)
[rank2]:W0829 16:47:48.404000 2014106 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    last reason: 2/7: term_idx == 1045  # children = self.parent_to_children.get(term_idx, [])  # Documents/projects/torchcell/torchcell/models/dcell.py:358 in _prepare_term_input (_dynamo/variables/tensor.py:1410 in evaluate_expr)
[rank2]:W0829 16:47:48.404000 2014106 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank2]:W0829 16:47:48.404000 2014106 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank2]:W0829 16:47:48.548000 2014106 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] torch._dynamo hit config.recompile_limit (8)
[rank2]:W0829 16:47:48.548000 2014106 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    function: 'forward' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:391)
[rank2]:W0829 16:47:48.548000 2014106 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    last reason: 8/4: tensor 'x' size mismatch at index 1. expected 4, actual 21
[rank2]:W0829 16:47:48.548000 2014106 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank2]:W0829 16:47:48.548000 2014106 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank0]:W0829 16:47:48.562000 2011461 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] torch._dynamo hit config.recompile_limit (8)
[rank0]:W0829 16:47:48.562000 2011461 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    function: '_prepare_term_input' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:346)
[rank0]:W0829 16:47:48.562000 2011461 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    last reason: 2/7: term_idx == 1045  # children = self.parent_to_children.get(term_idx, [])  # Documents/projects/torchcell/torchcell/models/dcell.py:358 in _prepare_term_input (_dynamo/variables/tensor.py:1410 in evaluate_expr)
[rank0]:W0829 16:47:48.562000 2011461 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank0]:W0829 16:47:48.562000 2011461 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank1]:W0829 16:47:48.590000 2014105 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] torch._dynamo hit config.recompile_limit (8)
[rank1]:W0829 16:47:48.590000 2014105 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    function: '_prepare_term_input' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:346)
[rank1]:W0829 16:47:48.590000 2014105 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    last reason: 2/7: term_idx == 1045  # children = self.parent_to_children.get(term_idx, [])  # Documents/projects/torchcell/torchcell/models/dcell.py:358 in _prepare_term_input (_dynamo/variables/tensor.py:1410 in evaluate_expr)
[rank1]:W0829 16:47:48.590000 2014105 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank1]:W0829 16:47:48.590000 2014105 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank0]:W0829 16:47:48.712000 2011461 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] torch._dynamo hit config.recompile_limit (8)
[rank0]:W0829 16:47:48.712000 2011461 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    function: 'forward' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:391)
[rank0]:W0829 16:47:48.712000 2011461 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    last reason: 8/4: tensor 'x' size mismatch at index 1. expected 4, actual 21
[rank0]:W0829 16:47:48.712000 2011461 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank0]:W0829 16:47:48.712000 2011461 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank1]:W0829 16:47:48.738000 2014105 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] torch._dynamo hit config.recompile_limit (8)
[rank1]:W0829 16:47:48.738000 2014105 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    function: 'forward' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:391)
[rank1]:W0829 16:47:48.738000 2014105 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    last reason: 8/4: tensor 'x' size mismatch at index 1. expected 4, actual 21
[rank1]:W0829 16:47:48.738000 2014105 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank1]:W0829 16:47:48.738000 2014105 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank3]:W0829 16:47:50.521000 2014107 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] torch._dynamo hit config.recompile_limit (8)
[rank3]:W0829 16:47:50.521000 2014107 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    function: '_prepare_term_input' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:346)
[rank3]:W0829 16:47:50.521000 2014107 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    last reason: 2/7: term_idx == 1045  # children = self.parent_to_children.get(term_idx, [])  # Documents/projects/torchcell/torchcell/models/dcell.py:358 in _prepare_term_input (_dynamo/variables/tensor.py:1410 in evaluate_expr)
[rank3]:W0829 16:47:50.521000 2014107 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank3]:W0829 16:47:50.521000 2014107 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank3]:W0829 16:47:50.666000 2014107 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] torch._dynamo hit config.recompile_limit (8)
[rank3]:W0829 16:47:50.666000 2014107 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    function: 'forward' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:391)
[rank3]:W0829 16:47:50.666000 2014107 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    last reason: 8/4: tensor 'x' size mismatch at index 1. expected 4, actual 21
[rank3]:W0829 16:47:50.666000 2014107 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank3]:W0829 16:47:50.666000 2014107 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
Sanity Checking DataLoader 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [01:16<00:00,  0.03it/s]/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using torch.float32).
  warnings.warn(*args, **kwargs)  # noqa: B028
Epoch 0:   1%|▍                                                              | 1/133 [08:47<19:20:16,  0.00it/s]                                                                                                                Epoch 0:  31%|███████████████▋                                   | 41/133 [33:35<1:15:23,  0.02it/s, v_num=oaxp]
```

## BF16-Mixed Precison - 8 Workers - Compile - batch size 600

- 600 is probably max we should do for batch size... Might be able to push to 650 or 700. Think 700 would crash on 45 GB.

```python
(torchcell) michaelvolk@gilahyper torchcell % python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 600, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': 'default', 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: Using wandb-core as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-4653e24b-285c-4a6c-8405-a1606e2b58a0_172ef6bb46a7cf7b6ed67bf6ef21659d2050e69ebbbcc28a1b50526d37539e6d/wandb/run-20250829_173644-d2msnvdg
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-4653e24b-285c-4a6c-8405-a1606e2b58a0_172ef6bb46a7cf7b6ed67bf6ef21659d2050e69ebbbcc28a1b50526d37539e6d
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/d2msnvdg
Using Gene Ontology hierarchy
[2025-08-29 17:36:49,841][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 17:36:51,716][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
[2025-08-29 17:36:54,068][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 17:36:56,146][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-29 17:36:56,192][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-29 17:36:56,267][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-17-36-56)
Instantiating DCell
Pre-computing gene indices for 2655 GO terms...
Pre-computed indices in 0.19 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-17-36-59)
Attempting torch.compile optimization (PyTorch 2.8.0+cu128)...
  Mode: default, Dynamic: True
  Set precompilation timeout to 2 hours
✓ Successfully compiled model with torch.compile (mode=default, dynamic=True)
  Expected speedup: 2-3x for forward/backward passes
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-29-17-37-01)
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/fabric/plugins/environments/slurm.py:204: The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python /home/michaelvolk/Documents/projects/torchcell/exper ...
Using bfloat16 Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 600, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': 'default', 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 600, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': 'default', 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_006-kuzmin-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': False, 'perturbation_subset_size': 1000, 'batch_size': 600, 'num_workers': 8, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'go_gene_strata_state']}, 'trainer': {'max_epochs': 1000, 'strategy': 'ddp', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 4, 'overfit_batches': 0, 'precision': 'bf16-mixed'}, 'compile_mode': 'default', 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': False}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 0.001, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-01a50731-eda5-4b3d-ab15-8a60b8300cf3_172ef6bb46a7cf7b6ed67bf6ef21659d2050e69ebbbcc28a1b50526d37539e6d/wandb/run-20250829_173718-5y5gefa6
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-01a50731-eda5-4b3d-ab15-8a60b8300cf3_172ef6bb46a7cf7b6ed67bf6ef21659d2050e69ebbbcc28a1b50526d37539e6d
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/5y5gefa6
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-ca1d4c1b-c052-47b4-b8db-ce9f84406681_172ef6bb46a7cf7b6ed67bf6ef21659d2050e69ebbbcc28a1b50526d37539e6d/wandb/run-20250829_173718-qp7hgt0w
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-ca1d4c1b-c052-47b4-b8db-ce9f84406681_172ef6bb46a7cf7b6ed67bf6ef21659d2050e69ebbbcc28a1b50526d37539e6d
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/qp7hgt0w
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /scratch/projects/torchcell/wandb-experiments/gilahyper-e1223174-98c8-4089-b8de-1e5bf0af8d18_172ef6bb46a7cf7b6ed67bf6ef21659d2050e69ebbbcc28a1b50526d37539e6d/wandb/run-20250829_173718-896tqzz3
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run run_gilahyper-e1223174-98c8-4089-b8de-1e5bf0af8d18_172ef6bb46a7cf7b6ed67bf6ef21659d2050e69ebbbcc28a1b50526d37539e6d
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_006-kuzmin-tmi_dcell/runs/896tqzz3
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
Using Gene Ontology hierarchy
[2025-08-29 17:37:23,703][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 17:37:23,791][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 17:37:23,795][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-08-29 17:37:25,554][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
[2025-08-29 17:37:25,668][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
[2025-08-29 17:37:25,677][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
After redundant filter: 5435
[2025-08-29 17:37:27,977][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
[2025-08-29 17:37:28,103][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
[2025-08-29 17:37:28,108][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
After containment filter (min_genes=4): 2655
/home/michaelvolk/Documents/projects/torchcell/experiments
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 17:37:30,046][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-29 17:37:30,089][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-29 17:37:30,165][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-17-37-30)
Instantiating DCell
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
Computed 13 strata for 2655 GO terms
  Stratum 0: 1 terms
  Stratum 1: 3 terms
  Stratum 2: 630 terms
  Stratum 3: 530 terms
  Stratum 4: 392 terms
  ... and 8 more strata
[2025-08-29 17:37:30,245][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-29 17:37:30,293][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-29 17:37:30,297][torchcell.datamodules.cell][INFO] - Loading index from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-08-29 17:37:30,341][torchcell.datamodules.cell][INFO] - Loading index details from /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/data_module_cache/index_details_seed_42.json
[2025-08-29 17:37:30,370][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-17-37-30)
Instantiating DCell
[2025-08-29 17:37:30,420][__main__][INFO] - Using device: cuda
Instantiating model (2025-08-29-17-37-30)
Instantiating DCell
Pre-computing gene indices for 2655 GO terms...
Pre-computing gene indices for 2655 GO terms...
Pre-computing gene indices for 2655 GO terms...
Pre-computed indices in 0.82 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Pre-computed indices in 1.09 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Pre-computed indices in 1.10 seconds
  Total rows per sample: 59986
  Average genes per GO term: 22.6
DCell model initialized:
  GO terms: 2655
  Genes: 6607
  Strata: 13 (max: 12)
  Total subsystems: 2655
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-17-37-38)
Attempting torch.compile optimization (PyTorch 2.8.0+cu128)...
  Mode: default, Dynamic: True
  Set precompilation timeout to 2 hours
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-17-37-39)
Parameter counts: {'subsystems': 20548953, 'dcell_linear': 64084, 'dcell': 20613037, 'total': 20613037, 'num_go_terms': 2655, 'num_subsystems': 2655}
Creating RegressionTask (2025-08-29-17-37-39)
Attempting torch.compile optimization (PyTorch 2.8.0+cu128)...
  Mode: default, Dynamic: True
  Set precompilation timeout to 2 hours
Attempting torch.compile optimization (PyTorch 2.8.0+cu128)...
  Mode: default, Dynamic: True
  Set precompilation timeout to 2 hours
✓ Successfully compiled model with torch.compile (mode=default, dynamic=True)
  Expected speedup: 2-3x for forward/backward passes
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-29-17-37-40)
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/4
✓ Successfully compiled model with torch.compile (mode=default, dynamic=True)
  Expected speedup: 2-3x for forward/backward passes
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-29-17-37-40)
✓ Successfully compiled model with torch.compile (mode=default, dynamic=True)
  Expected speedup: 2-3x for forward/backward passes
devices: 4
SLURM_JOB_NUM_NODES: None
SLURM_NNODES: None
SLURM_NPROCS: None
Starting training (2025-08-29-17-37-40)
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/4
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/4
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 4 processes
----------------------------------------------------------------------------------------------------

/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loggers/wandb.py:396: There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
[2025-08-29 17:37:42,744][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 17:37:42,746][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 17:37:42,750][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 17:37:42,756][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
[2025-08-29 17:37:42,865][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 17:37:42,865][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 17:37:42,865][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 17:37:42,865][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 17:37:42,872][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 17:37:42,872][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 17:37:42,885][torchcell.trainers.int_dcell][WARNING] - Error during model initialization: 'NoneType' object has no attribute 'max'
[2025-08-29 17:37:42,885][torchcell.trainers.int_dcell][INFO] - Added dummy parameter to allow optimizer creation
[2025-08-29 17:37:42,891][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
[2025-08-29 17:37:42,892][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
[2025-08-29 17:37:42,898][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
[2025-08-29 17:37:42,912][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,613,038
/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/core/optimizer.py:316: The lr scheduler dict contains the key(s) ['monitor'], but the keys will be ignored. You need to call `lr_scheduler.step()` manually in manual optimization.

  | Name                      | Type             | Params | Mode 
-----------------------------------------------------------------------
0 | model                     | OptimizedModule  | 20.6 M | train
1 | loss_func                 | DCellLoss        | 0      | train
2 | train_metrics             | MetricCollection | 0      | train
3 | train_transformed_metrics | MetricCollection | 0      | train
4 | val_metrics               | MetricCollection | 0      | train
5 | val_transformed_metrics   | MetricCollection | 0      | train
6 | test_metrics              | MetricCollection | 0      | train
7 | test_transformed_metrics  | MetricCollection | 0      | train
-----------------------------------------------------------------------
20.6 M    Trainable params
0         Non-trainable params
20.6 M    Total params
82.452    Total estimated model params size (MB)
13305     Modules in train mode
0         Modules in eval mode
Sanity Checking DataLoader 0:   0%|                                                       | 0/2 [00:00<?, ?it/s][rank0]:W0829 17:45:26.292000 2505741 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break from `Tensor.item()`, consider setting:
[rank0]:W0829 17:45:26.292000 2505741 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     torch._dynamo.config.capture_scalar_outputs = True
[rank0]:W0829 17:45:26.292000 2505741 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] or:
[rank0]:W0829 17:45:26.292000 2505741 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
[rank0]:W0829 17:45:26.292000 2505741 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] to include these operations in the captured graph.
[rank0]:W0829 17:45:26.292000 2505741 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank0]:W0829 17:45:26.292000 2505741 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break: from user code at:
[rank0]:W0829 17:45:26.292000 2505741 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]   File "/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py", line 292, in forward
[rank0]:W0829 17:45:26.292000 2505741 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     term_idx_int = int(term_idx)
[rank0]:W0829 17:45:26.292000 2505741 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank0]:W0829 17:45:26.292000 2505741 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank1]:W0829 17:45:32.119000 2510864 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break from `Tensor.item()`, consider setting:
[rank1]:W0829 17:45:32.119000 2510864 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     torch._dynamo.config.capture_scalar_outputs = True
[rank1]:W0829 17:45:32.119000 2510864 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] or:
[rank1]:W0829 17:45:32.119000 2510864 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
[rank1]:W0829 17:45:32.119000 2510864 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] to include these operations in the captured graph.
[rank1]:W0829 17:45:32.119000 2510864 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank1]:W0829 17:45:32.119000 2510864 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break: from user code at:
[rank1]:W0829 17:45:32.119000 2510864 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]   File "/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py", line 292, in forward
[rank1]:W0829 17:45:32.119000 2510864 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     term_idx_int = int(term_idx)
[rank1]:W0829 17:45:32.119000 2510864 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank1]:W0829 17:45:32.119000 2510864 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank2]:W0829 17:45:32.151000 2510865 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break from `Tensor.item()`, consider setting:
[rank2]:W0829 17:45:32.151000 2510865 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     torch._dynamo.config.capture_scalar_outputs = True
[rank2]:W0829 17:45:32.151000 2510865 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] or:
[rank2]:W0829 17:45:32.151000 2510865 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
[rank2]:W0829 17:45:32.151000 2510865 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] to include these operations in the captured graph.
[rank2]:W0829 17:45:32.151000 2510865 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank2]:W0829 17:45:32.151000 2510865 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break: from user code at:
[rank2]:W0829 17:45:32.151000 2510865 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]   File "/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py", line 292, in forward
[rank2]:W0829 17:45:32.151000 2510865 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     term_idx_int = int(term_idx)
[rank2]:W0829 17:45:32.151000 2510865 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank2]:W0829 17:45:32.151000 2510865 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank3]:W0829 17:45:32.450000 2510867 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break from `Tensor.item()`, consider setting:
[rank3]:W0829 17:45:32.450000 2510867 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     torch._dynamo.config.capture_scalar_outputs = True
[rank3]:W0829 17:45:32.450000 2510867 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] or:
[rank3]:W0829 17:45:32.450000 2510867 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
[rank3]:W0829 17:45:32.450000 2510867 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] to include these operations in the captured graph.
[rank3]:W0829 17:45:32.450000 2510867 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank3]:W0829 17:45:32.450000 2510867 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] Graph break: from user code at:
[rank3]:W0829 17:45:32.450000 2510867 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]   File "/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py", line 292, in forward
[rank3]:W0829 17:45:32.450000 2510867 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1]     term_idx_int = int(term_idx)
[rank3]:W0829 17:45:32.450000 2510867 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank3]:W0829 17:45:32.450000 2510867 site-packages/torch/_dynamo/variables/tensor.py:1047] [0/1] 
[rank0]:W0829 17:45:39.230000 2505741 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] torch._dynamo hit config.recompile_limit (8)
[rank0]:W0829 17:45:39.230000 2505741 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    function: '_prepare_term_input' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:346)
[rank0]:W0829 17:45:39.230000 2505741 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    last reason: 2/7: term_idx == 1045  # children = self.parent_to_children.get(term_idx, [])  # Documents/projects/torchcell/torchcell/models/dcell.py:358 in _prepare_term_input (_dynamo/variables/tensor.py:1410 in evaluate_expr)
[rank0]:W0829 17:45:39.230000 2505741 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank0]:W0829 17:45:39.230000 2505741 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank0]:W0829 17:45:39.388000 2505741 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] torch._dynamo hit config.recompile_limit (8)
[rank0]:W0829 17:45:39.388000 2505741 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    function: 'forward' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:391)
[rank0]:W0829 17:45:39.388000 2505741 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    last reason: 8/4: tensor 'x' size mismatch at index 1. expected 4, actual 21
[rank0]:W0829 17:45:39.388000 2505741 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank0]:W0829 17:45:39.388000 2505741 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank2]:W0829 17:45:45.155000 2510865 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] torch._dynamo hit config.recompile_limit (8)
[rank2]:W0829 17:45:45.155000 2510865 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    function: '_prepare_term_input' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:346)
[rank2]:W0829 17:45:45.155000 2510865 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    last reason: 2/7: term_idx == 1045  # children = self.parent_to_children.get(term_idx, [])  # Documents/projects/torchcell/torchcell/models/dcell.py:358 in _prepare_term_input (_dynamo/variables/tensor.py:1410 in evaluate_expr)
[rank2]:W0829 17:45:45.155000 2510865 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank2]:W0829 17:45:45.155000 2510865 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank1]:W0829 17:45:45.252000 2510864 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] torch._dynamo hit config.recompile_limit (8)
[rank1]:W0829 17:45:45.252000 2510864 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    function: '_prepare_term_input' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:346)
[rank1]:W0829 17:45:45.252000 2510864 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    last reason: 2/7: term_idx == 1045  # children = self.parent_to_children.get(term_idx, [])  # Documents/projects/torchcell/torchcell/models/dcell.py:358 in _prepare_term_input (_dynamo/variables/tensor.py:1410 in evaluate_expr)
[rank1]:W0829 17:45:45.252000 2510864 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank1]:W0829 17:45:45.252000 2510864 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank2]:W0829 17:45:45.314000 2510865 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] torch._dynamo hit config.recompile_limit (8)
[rank2]:W0829 17:45:45.314000 2510865 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    function: 'forward' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:391)
[rank2]:W0829 17:45:45.314000 2510865 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    last reason: 8/4: tensor 'x' size mismatch at index 1. expected 4, actual 21
[rank2]:W0829 17:45:45.314000 2510865 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank2]:W0829 17:45:45.314000 2510865 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank3]:W0829 17:45:45.361000 2510867 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] torch._dynamo hit config.recompile_limit (8)
[rank3]:W0829 17:45:45.361000 2510867 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    function: '_prepare_term_input' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:346)
[rank3]:W0829 17:45:45.361000 2510867 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8]    last reason: 2/7: term_idx == 1045  # children = self.parent_to_children.get(term_idx, [])  # Documents/projects/torchcell/torchcell/models/dcell.py:358 in _prepare_term_input (_dynamo/variables/tensor.py:1410 in evaluate_expr)
[rank3]:W0829 17:45:45.361000 2510867 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank3]:W0829 17:45:45.361000 2510867 site-packages/torch/_dynamo/convert_frame.py:1016] [2/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank1]:W0829 17:45:45.402000 2510864 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] torch._dynamo hit config.recompile_limit (8)
[rank1]:W0829 17:45:45.402000 2510864 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    function: 'forward' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:391)
[rank1]:W0829 17:45:45.402000 2510864 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    last reason: 8/4: tensor 'x' size mismatch at index 1. expected 4, actual 21
[rank1]:W0829 17:45:45.402000 2510864 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank1]:W0829 17:45:45.402000 2510864 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank3]:W0829 17:45:45.510000 2510867 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] torch._dynamo hit config.recompile_limit (8)
[rank3]:W0829 17:45:45.510000 2510867 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    function: 'forward' (/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py:391)
[rank3]:W0829 17:45:45.510000 2510867 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8]    last reason: 8/4: tensor 'x' size mismatch at index 1. expected 4, actual 21
[rank3]:W0829 17:45:45.510000 2510867 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank3]:W0829 17:45:45.510000 2510867 site-packages/torch/_dynamo/convert_frame.py:1016] [8/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
Sanity Checking DataLoader 0: 100%|███████████████████████████████████████████████| 2/2 [01:23<00:00,  0.02it/s]/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using torch.float32).
  warnings.warn(*args, **kwargs)  # noqa: B028
Epoch 0:  31%|███████████████▌                                   | 34/111 [33:40<1:16:16,  0.02it/s, v_num=nvdg]
```

Will finish in `1:48`.
