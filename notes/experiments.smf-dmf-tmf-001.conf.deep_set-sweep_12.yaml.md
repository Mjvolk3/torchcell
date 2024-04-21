---
id: fzqimwm4c2wq1b6p5necffl
title: Yaml
desc: ''
updated: 1713549067643
created: 1713549016069
---

## 2024.04.19 - Failed Run

My guess is that this is a memory error in data loader. We could try to reduce batch size to fix this.

[wandb log](https://wandb.ai/zhao-group/torchcell_smf-dmf-tmf-001-12/runs/t6vpwzuf/logs?nw=nwusermjvolk3)

``` bash
 1 data/go/go.obo: fmt(1.2) rel(2024-01-17) 45,869 Terms
 2 /projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/nn/init.py:412: UserWarning: Initializing zero-element tensors is a no-op   warnings.warn("Initializing zero-element tensors is a no-op")
 3 /projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer. For large datasets, this may lead to large memory footprint.   warnings.warn(*args, **kwargs)  # noqa: B028
 4 [2024-04-18 16:08:48,148][__main__][INFO] - cuda
 5 GPU available: True (cuda), used: True
 6 TPU available: False, using: 0 TPU cores
 7 IPU available: False, using: 0 IPUs
 8 HPU available: False, using: 0 HPUs
 9 You are using a CUDA device ('NVIDIA A40') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
10 /projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loggers/wandb.py:389: There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`.
11 [2024-04-18 16:10:27,074][torchcell.datamodules.cell][INFO] - Loading cached indices from /scratch/bbub/mjvolk3/torchcell/data/torchcell/experiments/smf-dmf-tmf_1e03/data_module_cache/cached_indices.json
12 /projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/callbacks/model_checkpoint.py:639: Checkpoint directory models/checkpoints/3440131_1375bdeca6260d8e20387a5348e4332c2593e6e5c9a7f75ff747b9ce7adb6aad exists and is not empty.
13 LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [1]
14    | Name          | Type             | Params --------------------------------------------------- 0 | model         | ModuleDict       | 51.1 K 1 | loss          | MSEListMLELoss   | 0      2 | loss_node     | MSELoss          | 0      3 | train_metrics | MetricCollection | 0      4 | val_metrics   | MetricCollection | 0      5 | test_metrics  | MetricCollection | 0      6 | pearson_corr  | PearsonCorrCoef  | 0      7 | spearman_corr | SpearmanCorrCoef | 0      --------------------------------------------------- 51.1 K    Trainable params 0         Non-trainable params 51.1 K    Total params 0.205     Total estimated model params size (MB)
15 SLURM auto-requeueing enabled. Setting signal handlers.
16  Sanity Checking: |          | 0/? [00:00<?, ?it/s]
17  Sanity Checking:   0%|          | 0/2 [00:00<?, ?it/s]
18  Sanity Checking DataLoader 0:   0%|          | 0/2 [00:00<?, ?it/s]
19 /projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/nn/modules/linear.py:114: UserWarning: An output with one or more elements was resized since it had shape [210459, 256], which does not match the required output shape [256]. This behavior is deprecated, and in a future PyTorch release outputs will not be resized unless they have zero elements. You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0). (Triggered internally at ../aten/src/ATen/native/Resize.cpp:28.)   return F.linear(input, self.weight, self.bias)
20 Error executing job with overrides: []
21 Traceback (most recent call last):
22 
23   File "/projects/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/deep_set.py", line 331, in main     trainer.fit(task, data_module)
24 
25   File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 544, in fit     call._call_and_handle_interrupt(
26 
27   File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 44, in _call_and_handle_interrupt     return trainer_fn(*args, **kwargs)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
28 
29   File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 580, in _fit_impl     self._run(model, ckpt_path=ckpt_path)
30 
31   File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 989, in _run     results = self._run_stage()               ^^^^^^^^^^^^^^^^^
32 
33   File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1033, in _run_stage     self._run_sanity_check()
34 
35   File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1062, in _run_sanity_check     val_loop.run()
36 
37   File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loops/utilities.py", line 182, in _decorator     return loop_run(self, *args, **kwargs)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
38 
39   File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loops/evaluation_loop.py", line 134, in run     self._evaluation_step(batch, batch_idx, dataloader_idx, dataloader_iter)
40 
41   File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/loops/evaluation_loop.py", line 391, in _evaluation_step     output = call._call_strategy_hook(trainer, hook_name, *step_args)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
42 
43   File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 309, in _call_strategy_hook     output = fn(*args, **kwargs)              ^^^^^^^^^^^^^^^^^^^
44 
45   File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/lightning/pytorch/strategies/strategy.py", line 403, in validation_step     return self.lightning_module.validation_step(*args, **kwargs)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
46 
47   File "/projects/bbub/mjvolk3/torchcell/torchcell/trainers/neo_regression.py", line 207, in validation_step     y_hat = self(x, batch_vector)             ^^^^^^^^^^^^^^^^^^^^^
48 
49   File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl     return self._call_impl(*args, **kwargs)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
50 
51   File "/projects/bbub/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl     return forward_call(*args, **kwargs)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
52 
53   File "/projects/bbub/mjvolk3/torchcell/torchcell/trainers/neo_regression.py", line 144, in forward     x_nodes, x_set = self.model["main"](x, batch)                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```
