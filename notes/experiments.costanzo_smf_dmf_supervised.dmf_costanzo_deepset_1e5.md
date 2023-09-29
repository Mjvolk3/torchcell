---
id: elr05cfappvsttrp4t5spgn
title: Dmf_costanzo_deepset_1e5
desc: ''
updated: 1695957002875
created: 1695956733137
---
## Deep Set Model Only Works with DDP Find Unused

Error indicating that some parameters are not used in producing loss. I believe this is due to the fact that we currenlty skip `batch_norm` for `wt` instances, since there is only a single instance. This would mean that these weights are not used. This could possibly be fixed by passing a duplicate `wt` through the model and then only computing the loss on one instance.

```bash
It looks like your LightningModule has parameters that were not used in producing the loss returned by training_step. If this is intentional, you must enable the detection of unused parameters in DDP, either by setting the string value `strategy='ddp_find_unused_parameters_true'` or by setting the flag in the strategy with `strategy=DDPStrategy(find_unused_parameters=True)`.
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by
making sure all `forward` function outputs participate in calculating loss.
If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
Parameter indices which did not receive grad for rank 0: 10 11 14 15
 In addition, you can set the environment variable TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL to print out information about which particular parameters did not receive gradient on this rank as part of this error

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
It looks like your LightningModule has parameters that were not used in producing the loss returned by training_step. If this is intentional, you must enable the detection of unused parameters in DDP, either by setting the string value `strategy='ddp_find_unused_parameters_true'` or by setting the flag in the strategy with `strateg...
```

```python
if wandb.config.trainer["strategy"] == "ddp_find_unused":
    from pytorch_lightning.strategies import DDPStrategy
    strategy = DDPStrategy(find_unused_parameters=True)
```
