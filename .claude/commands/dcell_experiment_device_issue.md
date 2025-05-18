## Goal

We are writing the experimental pipeline for `Dcell`. It is very similar to the pipeline for the `Dango` model. We want to keep it similar to the Dango pipeline as possible. We want to record all of the same metrics, use the same code structure, etc.

## Relevant Files

These are the relevant Dango files

/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/dango.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/trainers/int_dango.py
/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/conf/dango_kuzmin2018_tmi.yaml

## Files We Need to Write

/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/dcell.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/trainers/int_dcell.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/losses/dcell.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/scratch/load_batch_005.py

This configuration file should probably be okay. We might need to make some tweaks to it.

/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/conf/dcell_kuzmin2018_tmi.yaml

## Update

The pipeline now runs.

## Task

Not all data is on device... We are trying to run on gpu now. Help me make sure everything is on correct device.

***

This was our last attempt

```python
(torchcell) mjvolk3@compute-5-7 ~/projects/torchcell $ python /home/a-m/mjvolk3/scratch/torchcell/experiments/005-kuzmin2018-tmi/scripts/dcell.py --config-name dcell_kuzmin2018_tmi
/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: /lib64/libm.so.6: version `GLIBC_2.29' not found (required by /home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: /lib64/libm.so.6: version `GLIBC_2.29' not found (required by /home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
Starting DCell Training 🧬
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'project': 'torchcell_005-kuzmin2018-tmi_dcell', 'tags': []}, 'cell_dataset': {'graphs': None, 'incidence_graphs': 'go', 'node_embeddings': None}, 'profiler': {'is_pytorch': True}, 'data_module': {'is_perturbation_subset': True, 'perturbation_subset_size': 100, 'batch_size': 2, 'num_workers': 2, 'pin_memory': False, 'prefetch': False, 'follow_batch': ['perturbation_indices', 'mutant_state']}, 'trainer': {'max_epochs': 500, 'strategy': 'auto', 'num_nodes': 1, 'accelerator': 'gpu', 'devices': 1, 'overfit_batches': 0}, 'model': {'checkpoint_path': None, 'subsystem_output_min': 20, 'subsystem_output_max_mult': 0.3, 'output_size': 1, 'norm_type': 'batch', 'norm_before_act': False, 'subsystem_num_layers': 1, 'activation': 'tanh', 'init_range': 0.001, 'learnable_embedding_dim': None}, 'regression_task': {'loss': 'dcell', 'dcell_loss': {'alpha': 0.3, 'use_auxiliary_losses': True}, 'is_weighted_phenotype_loss': False, 'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 1e-06}, 'lr_scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 3, 'threshold': '1e -4', 'threshold_mode': 'rel', 'cooldown': 2, 'min_lr': 1e-09, 'eps': 1e-10}, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10.0, 'grad_accumulation_schedule': None, 'plot_sample_ceiling': 10000, 'plot_every_n_epochs': 20}}
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Tracking run with wandb version 0.19.6
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
/home/a-m/mjvolk3/scratch/torchcell/data/go/go.obo: fmt(1.2) rel(2024-11-03) 43,983 Terms
Using Gene Ontology hierarchy
[2025-05-15 23:43:10,489][torchcell.graph.graph][INFO] - Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
[2025-05-15 23:43:12,706][torchcell.graph.graph][INFO] - Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
[2025-05-15 23:43:15,612][torchcell.graph.graph][INFO] - Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter (min_genes=4): 2655
/home/a-m/mjvolk3/projects/torchcell/experiments
[2025-05-15 23:43:15,874][torchcell.datamodules.cell][INFO] - Loading index from /home/a-m/mjvolk3/scratch/torchcell/data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-05-15 23:43:15,914][torchcell.datamodules.cell][INFO] - Loading index details from /home/a-m/mjvolk3/scratch/torchcell/data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build/data_module_cache/index_details_seed_42.json
Setting up PerturbationSubsetDataModule...
Loading cached index files...
Creating subset datasets...
Setup complete.
[2025-05-15 23:43:16,026][__main__][INFO] - Using device: cuda
Instantiating model (2025-05-15-23-43-16)
Instantiating DCellModel
Parameter counts: {'dcell': 0, 'subsystems': 0, 'total': 0}
Creating RegressionTask (2025-05-15-23-43-16)
devices: 1
SLURM_JOB_NUM_NODES: 1
SLURM_NNODES: 1
SLURM_NPROCS: 1
Starting training (2025-05-15-23-43-16)
/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/fabric/plugins/environments/slurm.py:204: The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python /home/a-m/mjvolk3/scratch/torchcell/experiments/005- ...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loggers/wandb.py:397: There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`.
Setting up PerturbationSubsetDataModule...
Creating subset datasets...
Setup complete.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[2025-05-15 23:43:16,311][torchcell.trainers.int_dcell][INFO] - Setting up optimizer and initializing DCellModel parameters
Found 1527 leaf nodes out of 2655 total nodes
Created 2655 subsystems out of 2655 nodes
Total parameters in subsystems: 20,472,243
Created 2655 subsystems from GO graph with 2655 nodes
[2025-05-15 23:43:17,879][torchcell.trainers.int_dcell][INFO] - Model parameters initialized successfully
[2025-05-15 23:43:17,910][torchcell.trainers.int_dcell][INFO] - Total trainable parameters: 20,536,209
/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/core/optimizer.py:317: The lr scheduler dict contains the key(s) ['monitor'], but the keys will be ignored. You need to call `lr_scheduler.step()` manually in manual optimization.

  | Name                      | Type             | Params | Mode 
-----------------------------------------------------------------------
0 | model                     | DCellModel       | 20.5 M | train
1 | loss_func                 | DCellLoss        | 0      | train
2 | train_metrics             | MetricCollection | 0      | train
3 | train_transformed_metrics | MetricCollection | 0      | train
4 | val_metrics               | MetricCollection | 0      | train
5 | val_transformed_metrics   | MetricCollection | 0      | train
6 | test_metrics              | MetricCollection | 0      | train
7 | test_transformed_metrics  | MetricCollection | 0      | train
-----------------------------------------------------------------------
20.5 M    Trainable params
0         Non-trainable params
20.5 M    Total params
82.145    Total estimated model params size (MB)
15962     Modules in train mode
0         Modules in eval mode
Sanity Checking: |                                                                      | 0/? [00:00<?, ?it/s]/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/utils/data/dataloader.py:617: UserWarning: This DataLoader will create 2 worker processes in total. Our suggested max number of worker in current system is 1, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:420: Consider setting `persistent_workers=True` in 'val_dataloader' to speed up the dataloader worker initialization.
/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: /lib64/libm.so.6: version `GLIBC_2.29' not found (required by /home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: /lib64/libm.so.6: version `GLIBC_2.29' not found (required by /home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: /lib64/libm.so.6: version `GLIBC_2.29' not found (required by /home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: /lib64/libm.so.6: version `GLIBC_2.29' not found (required by /home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
Sanity Checking DataLoader 0:   0%|                                                     | 0/2 [00:00<?, ?it/s]Error executing job with overrides: []
Traceback (most recent call last):
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/metric.py", line 550, in wrapped_func
    update(*args, **kwargs)
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/regression/pearson.py", line 148, in update
    self.mean_x, self.mean_y, self.var_x, self.var_y, self.corr_xy, self.n_total = _pearson_corrcoef_update(
                                                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/functional/regression/pearson.py", line 72, in _pearson_corrcoef_update
    corr_xy += ((preds - mx_new) * (target - mean_y)).sum(0)
                                    ~~~~~~~^~~~~~~~
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/a-m/mjvolk3/scratch/torchcell/experiments/005-kuzmin2018-tmi/scripts/dcell.py", line 465, in main
    trainer.fit(model=task, datamodule=data_module)
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 539, in fit
    call._call_and_handle_interrupt(
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 47, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 575, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 982, in _run
    results = self._run_stage()
              ^^^^^^^^^^^^^^^^^
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1024, in _run_stage
    self._run_sanity_check()
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1053, in _run_sanity_check
    val_loop.run()
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loops/utilities.py", line 179, in _decorator
    return loop_run(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loops/evaluation_loop.py", line 144, in run
    self._evaluation_step(batch, batch_idx, dataloader_idx, dataloader_iter)
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loops/evaluation_loop.py", line 433, in _evaluation_step
    output = call._call_strategy_hook(trainer, hook_name, *step_args)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 323, in _call_strategy_hook
    output = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/strategies/strategy.py", line 412, in validation_step
    return self.lightning_module.validation_step(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/a-m/mjvolk3/projects/torchcell/torchcell/trainers/int_dcell.py", line 300, in validation_step
    loss, _, _ = self._shared_step(batch, batch_idx, "val")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/a-m/mjvolk3/projects/torchcell/torchcell/trainers/int_dcell.py", line 205, in _shared_step
    transformed_metrics.update(
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/collections.py", line 256, in update
    m.update(*args, **m_kwargs)
  File "/home/a-m/mjvolk3/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/metric.py", line 553, in wrapped_func
    raise RuntimeError(
RuntimeError: Encountered different devices in metric calculation (see stacktrace for details). This could be due to the metric class not being on the same device as input. Instead of `metric=PearsonCorrCoef(...)` try to do `metric=PearsonCorrCoef(...).to(device)` where device corresponds to the device of the input.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
wandb: 
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1796117_bbc9812daa8fcab9134c907bf11d1e1f11cb47f1f60b4bbaf568efebf1877daf/wandb/offline-run-20250515_234254-cffpbetg
wandb: Find logs at: ../../scratch/torchcell/wandb-experiments/compute-5-7-1796117_bbc9812daa8fcab9134c907bf11d1e1f11cb47f1f60b4bbaf568efebf1877daf/wandb/offline-run-20250515_234254-cffpbetg/logs
(torchcell) mjvolk3@compute-5-7 ~/projects/torchcell $    
```

Think very hard. We want to fix this ASAP don't do anything too fancy let's just get everything on the correct device.
