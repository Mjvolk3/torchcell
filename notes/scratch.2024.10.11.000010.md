---
id: cfujj5egaesf1go1fompohj
title: '000010'
desc: ''
updated: 1728625329326
created: 1728622812638
---
No prefetch

```yaml
defaults:
  - default
  - _self_

wandb:
  # mode: offline # disabled, offline, online
  project: torchcell_test
  tags: []

cell_dataset:
  graphs: null # [physical], [regulatory], [physical, regulatory], []
  node_embeddings: [codon_frequency]
  # [one_hot_gene], [codon_frequency], [calm], [fudt_downstream], [fudt_upstream], [prot_T5_all], [prot_T5_no_dubious], [nt_window_5979], [nt_window_three_prime_300], [nt_window_five_prime_1003], [esm2_t33_650M_UR50D_all], [esm2_t33_650M_UR50D_no_dubious], [normalized_chrom_pathways], [random_1000], [random_100], [random_10], [random_1]

data_module:
  is_perturbation_subset: true
  perturbation_subset_size: 1e4 #7e3
  batch_size: 64
  num_workers: 12
  pin_memory: true
  prefetch: false # FLAG

trainer:
  max_epochs: 30
  strategy: auto # ddp, auto
  accelerator: gpu

models:
  graph:
    hidden_channels: 32
    out_channels: 16
    num_node_layers: 3 #0
    num_set_layers: 3
    norm: batch
    activation: relu #gelu
    skip_node: true
    skip_set: true
    aggregation: sum
  pred_head:
    hidden_channels: 0
    out_channels: 2
    num_layers: 1
    dropout_prob: 0.0
    norm: null
    activation: null
    output_activation: null

regression_task:
  boxplot_every_n_epochs: 1
  learning_rate: 1e-4
  weight_decay: 1e-5
  clip_grad_norm: true
  clip_grad_norm_max_norm: 10
```

```bash
  | Name          | Type            | Params | Mode 
----------------------------------------------------------
0 | model         | ModuleDict      | 6.1 K  | train
1 | loss          | CombinedMSELoss | 0      | train
2 | train_metrics | ModuleDict      | 0      | train
3 | val_metrics   | ModuleDict      | 0      | train
4 | test_metrics  | ModuleDict      | 0      | train
----------------------------------------------------------
6.1 K     Trainable params
0         Non-trainable params
6.1 K     Total params
0.025     Total estimated model params size (MB)
81        Modules in train mode
0         Modules in eval mode
Sanity Checking DataLoader 0:   0%|                                                                                        | 0/2 [00:00<?, ?it/s]/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using torch.float32).
  warnings.warn(*args, **kwargs)  # noqa: B028
Epoch 0:  63%|██████████████████████████████████████████████████████▌                               | 80/126 [31:19<18:00,  0.04it/s, v_num=ehyc]
```

With prefetch ... we see some minor speed improvements.

```yaml
defaults:
  - default
  - _self_

wandb:
  # mode: offline # disabled, offline, online
  project: torchcell_test
  tags: []

cell_dataset:
  graphs: null # [physical], [regulatory], [physical, regulatory], []
  node_embeddings: [codon_frequency]
  # [one_hot_gene], [codon_frequency], [calm], [fudt_downstream], [fudt_upstream], [prot_T5_all], [prot_T5_no_dubious], [nt_window_5979], [nt_window_three_prime_300], [nt_window_five_prime_1003], [esm2_t33_650M_UR50D_all], [esm2_t33_650M_UR50D_no_dubious], [normalized_chrom_pathways], [random_1000], [random_100], [random_10], [random_1]

data_module:
  is_perturbation_subset: true
  perturbation_subset_size: 1e4 #7e3
  batch_size: 64
  num_workers: 12
  pin_memory: true
  prefetch: true # FLAG

trainer:
  max_epochs: 30
  strategy: auto # ddp, auto
  accelerator: gpu

models:
  graph:
    hidden_channels: 32
    out_channels: 16
    num_node_layers: 3 #0
    num_set_layers: 3
    norm: batch
    activation: relu #gelu
    skip_node: true
    skip_set: true
    aggregation: sum
  pred_head:
    hidden_channels: 0
    out_channels: 2
    num_layers: 1
    dropout_prob: 0.0
    norm: null
    activation: null
    output_activation: null

regression_task:
  boxplot_every_n_epochs: 1
  learning_rate: 1e-4
  weight_decay: 1e-5
  clip_grad_norm: true
  clip_grad_norm_max_norm: 10
```

```bash
  | Name          | Type            | Params | Mode 3
----------------------------------------------------------
0 | model         | ModuleDict      | 6.1 K  | train
1 | loss          | CombinedMSELoss | 0      | train
2 | train_metrics | ModuleDict      | 0      | train
3 | val_metrics   | ModuleDict      | 0      | train
4 | test_metrics  | ModuleDict      | 0      | train
----------------------------------------------------------
6.1 K     Trainable params
0         Non-trainable params
6.1 K     Total params
0.025     Total estimated model params size (MB)
81        Modules in train mode
0         Modules in eval mode
Sanity Checking DataLoader 0:   0%|                                                                                        | 0/2 [00:00<?, ?it/s]/home/michaelvolk/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using torch.float32).
  warnings.warn(*args, **kwargs)  # noqa: B028
Epoch 0:  67%|█████████████████████████████████████████████████████████▎                            | 84/126 [30:29<15:14,  0.05it/s, v_num=49cx]
```
