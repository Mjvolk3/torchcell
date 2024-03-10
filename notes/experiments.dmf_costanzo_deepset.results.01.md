---
id: fbh3c3ck36i1hde2wnh35hs
title: dmf_costanzo_deepset.results.01
desc: ''
updated: 1696882919877
created: 1696712795774
---
## Experimental Summary

- This model was only trained on downstream CRE.
- We can achieve a decent correlation and loss fit on `FungalCRE` model. This model uses batch norm and is trained on `x_pert`. When we use `layer norm` we get seem to always get prediction of the mean fitness value.
- [Wandb Log](https://wandb.ai/zhao-group/torchcell/groups/2482163_782109b6f2cffa508d6a1628f03256f0a43419a5628b43a0731e2daeea0c9e13/workspace?workspace=user-mjvolk3)

![](./assets/drawio/model-gene-removal.drawio.png)

## Box Plot of True Fitness vs Predicted Fitness

![](./assets/images/experiments.dmf_costanzo_deepset.results.01.md.box-plot-of-true-fitness-vs-predicted-fitness-2482163_782109b6f2cffa508d6a1628f03256f0a43419a5628b43a0731e2daeea0c9e13.png)

This is after 2 epochs on `1e6` Costanzo.

## Loss and Correlations Show Promise but Noisy

![](./assets/images/experiments.dmf_costanzo_deepset.results.01.md.loss-and-correlations-show-promise-but-noisy-2482163_782109b6f2cffa508d6a1628f03256f0a43419a5628b43a0731e2daeea0c9e13.png)

## DDP Training - Box Plot

- This model was only trained on downstream CRE. This was a mistake, but still a useful to know that we can
- [Wandb Log](https://wandb.ai/zhao-group/torchcell/groups/2485154_3a9c9fca115f0281903cb1ce7b7b251e435c463d1c7785fae1be751d32040c4b/workspace?workspace=user-mjvolk3)

![](./assets/images/experiments.dmf_costanzo_deepset.results.01.md.box-plots-2485154_3a9c9fca115f0281903cb1ce7b7b251e435c463d1c7785fae1be751d32040c4b.png)

## DDP Training - Correlations and Loss

![](./assets/images/experiments.dmf_costanzo_deepset.results.01.md.correlations-and-loss-2485154_3a9c9fca115f0281903cb1ce7b7b251e435c463d1c7785fae1be751d32040c4b.png)

![](./assets/images/experiments.dmf_costanzo_deepset.results.01.md.variance-explained-2485154_3a9c9fca115f0281903cb1ce7b7b251e435c463d1c7785fae1be751d32040c4b.png)

If we look at $R^2 = 0.6^2 = 0.36$, so the embeddings of the downstream DNA sequences that are deleted from the genome can explain $36\%$ of the variance in growth.

## Deep Set FungalCRE Gene Removal

- Models Trained on `downstream` and `downstream + upstream`.

[WandB Report deep-set-fungal-cre-gene-removal](https://wandb.ai/zhao-group/torchcell/reports/Deep-Set-FungalCRE-Removed-Genes--Vmlldzo1NjI2MTc2)

![](./assets/images/experiments.dmf_costanzo_deepset.results.01.md.deep-set-fungal-cre-gene-removal.png)
