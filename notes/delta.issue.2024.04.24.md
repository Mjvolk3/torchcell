---
id: rrjd7hwy5715tjslg825qcj
title: '24'
desc: ''
updated: 1715730703443
created: 1715730670132
---

Interactive sessions are too shosrt to be able to run anything meaningful.

```bash
 *  Executing task in folder torchcell: srun --account=bbub-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --gpus-per-node=1 --tasks=1 --tasks-per-node=1 --cpus-per-task=1 --mem=62g --pty bash 

srun: job 3479819 queued and waiting for resources
srun: job 3479819 has been allocated resources
Wed Apr 24 11:12:19 CDT 2024 - Starting to source .bashrc
Wed Apr 24 11:12:19 CDT 2024 - Sourcing global definitions...
Wed Apr 24 11:12:20 CDT 2024 - Global definitions sourced.
Wed Apr 24 11:12:20 CDT 2024 - Setting up user-specific environment...
Wed Apr 24 11:12:20 CDT 2024 - User-specific environment set.
Wed Apr 24 11:12:20 CDT 2024 - Initializing Conda...
Wed Apr 24 11:14:49 CDT 2024 - Conda initialized.
Wed Apr 24 11:14:49 CDT 2024 - .bashrc sourced successfully.
(base) [mjvolk3@gpub001 torchcell]$ conda activate torchcell
(torchcell) [mjvolk3@gpub001 torchcell]$ cd /scratch/bbub/mjvolk3/torchcell
(torchcell) [mjvolk3@gpub001 torchcell]$ wandb sweep  experiments/smf-dmf-tmf-001/conf/deep_set-sweep_15.yaml

wandb: Using wandb-core version 0.17.0b10 as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: Creating sweep from: experiments/smf-dmf-tmf-001/conf/deep_set-sweep_15.yaml
wandb: Creating sweep with ID: k0yw7xgw
wandb: View sweep at: https://wandb.ai/zhao-group/torchcell_smf-dmf-tmf-001-15/sweeps/k0yw7xgw
wandb: Run sweep agent with: wandb agent zhao-group/torchcell_smf-dmf-tmf-001-15/k0yw7xgw
(torchcell) [mjvolk3@gpub001 torchcell]$ 
(torchcell) [mjvolk3@gpub001 torchcell]$ wandb agent zhao-group/torchcell_smf-dmf-tmf-001-15/k0yw7xgw
wandb: Using wandb-core version 0.17.0b10 as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: Starting wandb agent 🕵️
2024-04-24 11:25:37,336 - wandb.wandb_agent - INFO - Running runs: []
2024-04-24 11:25:37,594 - wandb.wandb_agent - INFO - Agent received command: run
2024-04-24 11:25:37,594 - wandb.wandb_agent - INFO - Agent starting run with config:
        cell_dataset: {'graphs': None, 'max_size': 1000, 'node_embeddings': ['nt_window_five_prime_1003']}
        data_module: {'batch_size': 8, 'num_workers': 6, 'pin_memory': True}
        models: {'graph': {'activation': 'gelu', 'hidden_channels': 512, 'norm': 'layer', 'num_node_layers': 3, 'num_set_layers': 3, 'out_channels': 32, 'skip_node': True, 'skip_set': True}, 'pred_head': {'activation': None, 'dropout_prob': 0, 'hidden_channels': 0, 'norm': None, 'num_layers': 1, 'out_channels': 1, 'output_activation': None}}
        regression_task: {'alpha': 0, 'boxplot_every_n_epochs': 5, 'clip_grad_norm': False, 'clip_grad_norm_max_norm': 1, 'learning_rate': 1e-06, 'loss': 'mse', 'target': 'fitness', 'weight_decay': 1e-05}
        trainer: {'accelerator': 'gpu', 'max_epochs': 50, 'strategy': 'auto'}
2024-04-24 11:25:37,639 - wandb.wandb_agent - INFO - About to run command: /usr/bin/env python experiments/smf-dmf-tmf-001/deep_set.py
2024-04-24 11:25:42,648 - wandb.wandb_agent - INFO - Running runs: ['zc80uc9a']
slurmstepd: error: *** STEP 3479819.0 ON gpub001 CANCELLED AT 2024-04-24T11:42:17 DUE TO TIME LIMIT ***
srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
srun: error: Timed out waiting for job step to complete
 *  Terminal will be reused by tasks, press any key to close it. 
```
