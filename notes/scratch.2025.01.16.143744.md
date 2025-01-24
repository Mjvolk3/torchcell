---
id: 2rcppmbfquguhrkxhx5i48w
title: '143744'
desc: ''
updated: 1737063034054
created: 1737059866392
---


| loss                           | train/fitness/Pearson | val/fitness/Pearson | train/fitness/RMSE | val/fitness/RMSE | embedding           |
|:-------------------------------|:----------------------|:--------------------|:-------------------|:-----------------|:--------------------|
| Soft Label Classification      | 0.182                 | 0.087               | 0.368              | 0.259            | learnable_embedding |
| Ordinal Classification         | -0.021                | -0.021              | 0.213              | 0.245            | learnable_embedding |
| Ordinal Entropy Regularization | **0.715**             | **0.581**           | **0.146**          | **0.216**        | learnable_embedding |
| Multiple Quantile Loss         | **0.700**             | **0.394**           | 0.151              | 0.283            | learnable_embedding |
| Dist Loss                      | 0.0374                | 0.013               | 0.282              | 0.253            | learnable_embedding |
| MSE                            | 0.327                 | -0.048              | 0.198              | 0.291            | codon_frequency*    |
| Logcosh                        | 0.000                 | -0.083              | 0.209              | 0.245            | codon_frequency*    |
$*$ - Only tried one embedding type

Should redo ordinal classification. Pearson metric was incorrect for these runs. For best copied val to train. I think Val/Pearson is wrong but train might be ok. Difficult to tell.
