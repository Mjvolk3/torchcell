---
id: dhxcnez3o7iphm3s8qi4gbn
title: Neo4j_query_raw
desc: ''
updated: 1723490056267
created: 1716490548011
---
## 2024.05.23 - Issue with Hyperparameter Sweep

If not set to `readonly=True` we get this error.

```bash
wandb: Starting wandb agent 🕵️
2024-05-23 13:04:19,665 - wandb.wandb_agent - INFO - Running runs: []
2024-05-23 13:04:20,123 - wandb.wandb_agent - INFO - Agent received command: run
2024-05-23 13:04:20,123 - wandb.wandb_agent - INFO - Agent starting run with config:
 cell_dataset: {'graphs': None, 'max_size': 1000, 'node_embeddings': ['nt_window_5979']}
 data_module: {'batch_size': 8, 'num_workers': 6, 'pin_memory': True}
 models: {'graph': {'activation': 'gelu', 'hidden_channels': 512, 'norm': 'layer', 'num_node_layers': 4, 'num_set_layers': 2, 'out_channels': 64, 'skip_node': True, 'skip_set': True}, 'pred_head': {'activation': None, 'dropout_prob': 0, 'hidden_channels': 0, 'norm': None, 'num_layers': 1, 'out_channels': 1, 'output_activation': None}}
 regression_task: {'alpha': 0, 'boxplot_every_n_epochs': 5, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 1, 'learning_rate': 1e-06, 'loss': 'mse', 'target': 'fitness', 'weight_decay': 0}
 trainer: {'accelerator': 'gpu', 'max_epochs': 50, 'strategy': 'auto'}
2024-05-23 13:04:20,128 - wandb.wandb_agent - INFO - About to run command: /usr/bin/env python experiments/smf-dmf-tmf-001/deep_set.py
2024-05-23 13:04:25,138 - wandb.wandb_agent - INFO - Running runs: ['49txm81b']
wandb: Currently logged in as: mjvolk3 (zhao-group). Use `wandb login --relogin` to force relogin
wandb: WARNING Ignored wandb.init() arg project when running a sweep.
wandb: wandb version 0.17.0 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.16.0
wandb: Run data is saved locally in /scratch/bbub/mjvolk3/torchcell/wandb-experiments/3693862/wandb/run-20240523_130534-49txm81b
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run rose-sweep-6
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_smf-dmf-tmf-001_deep_set_1e04_00
wandb: 🧹 View sweep at https://wandb.ai/zhao-group/torchcell_smf-dmf-tmf-001_deep_set_1e04_00/sweeps/pk6ek5mc
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_smf-dmf-tmf-001_deep_set_1e04_00/runs/49txm81b
Processing...
Done!
Starting Deep Set 🌋
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'mode': 'offline', 'project': 'torchcell_test', 'tags': []}, 'cell_dataset': {'graphs': None, 'node_embeddings': ['codon_frequency'], 'max_size': 1000.0}, 'data_module': {'batch_size': 16, 'num_workers': 6, 'pin_memory': True}, 'trainer': {'max_epochs': 10, 'strategy': 'auto', 'accelerator': 'gpu'}, 'models': {'graph': {'in_channels': None, 'hidden_channels': 128, 'out_channels': 32, 'num_node_layers': 0, 'num_set_layers': 3, 'norm': 'batch', 'activation': 'gelu', 'skip_node': True, 'skip_set': True}, 'pred_head': {'hidden_channels': 0, 'out_channels': 1, 'num_layers': 1, 'dropout_prob': 0.0, 'norm': None, 'activation': None, 'output_activation': None}}, 'regression_task': {'target': 'fitness', 'boxplot_every_n_epochs': 1, 'learning_rate': 0.01, 'weight_decay': 1e-05, 'loss': 'mse', 'alpha': 0.01, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10}}
data/go/go.obo: fmt(1.2) rel(2023-07-27) 46,356 Terms
/scratch/bbub/mjvolk3/torchcell/data/scerevisiae/nucleotide_transformer_embedding/processed/nt_window_5979.pt
=============
node.embeddings
{'nt_window_5979_max': NucleotideTransformerDataset(6607)}
=============
-------------------------
dataset_root:/scratch/bbub/mjvolk3/torchcell/data/torchcell/experiments/smf-dmf-tmf_1e03
-------------------------
================
raw root_dir: /scratch/bbub/mjvolk3/torchcell/data/torchcell/experiments/smf-dmf-tmf_1e03
================
Error executing job with overrides: []
Traceback (most recent call last):
  File "/scratch/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/deep_set.py", line 252, in main
    cell_dataset = Neo4jCellDataset(
                   ^^^^^^^^^^^^^^^^^
  File "/projects/bbub/mjvolk3/torchcell/torchcell/data/neo4j_cell.py", line 264, in __init__
    self.raw_db = self.load_raw(uri, username, password, root, query, self.genome)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/projects/bbub/mjvolk3/torchcell/torchcell/data/neo4j_cell.py", line 326, in load_raw
    raw_db = Neo4jQueryRaw(
             ^^^^^^^^^^^^^^
  File "<attrs generated init torchcell.data.neo4j_query_raw.Neo4jQueryRaw>", line 19, in __init__
    self.__attrs_post_init__()
  File "/projects/bbub/mjvolk3/torchcell/torchcell/data/neo4j_query_raw.py", line 147, in __attrs_post_init__
    self.env = lmdb.open(self.lmdb_dir, map_size=int(1e12))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
lmdb.InvalidParameterError: mdb_txn_begin: Invalid argument
```

This indicates something wrong with accessing `lmdb`. I believe it to be readonly. This should allow multiple processes to access.

```python
self.env = lmdb.open(self.lmdb_dir, map_size=int(1e12), readonly=True)
```

## 2024.08.12 - Neo4jQueryRaw is Immutable

```python
neo4j_db[0] = None
Traceback (most recent call last):
  File "<string>", line 1, in <module>
TypeError: 'Neo4jQueryRaw' object does not support item assignment
```
