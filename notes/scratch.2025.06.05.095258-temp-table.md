---
id: 8odkjbrt90jn76waqil7bcn
title: 095258 Temp Table
desc: ''
updated: 1749522076488
created: 1749135184177
---
342,350

| Model               | Training Strategy       | loss              | Data                   | Train Pearson | Val Pearson | Status |
|:--------------------|:------------------------|:------------------|:-----------------------|:--------------|:------------|:-------|
| Dango               | PreThenPost             | logcosh           | String 9.1             | 0.510         | 0.402       | x      |
| Dango               | LinearUntilUniform      | logcosh           | String 9.1             | 0.523         | 0.412       | x      |
| Dango               | LinearUntilFlipped      | logcosh           | String 9.1             | 0.511         | 0.402       | x      |
| Dango               | PreThenPost             | logcosh           | String 11.0            | 0.521         | 0.407       | x      |
| Dango               | LinearUntilUniform      | logcosh           | String 11.0            | 0.524         | 0.413       | x      |
| Dango               | LinearUntilFlipped      | logcosh           | String 11.0            | 0.509         | 0.401       | x      |
| Dango               | PreThenPost             | logcosh           | String 12.0            | 0.522         | 0.410       | x      |
| Dango               | LinearUntilUniform      | logcosh           | String 12.0            | 0.520         | 0.417       | x      |
| Dango               | LinearUntilFlipped      | logcosh           | String 12.0            | 0.519         | 0.409       | x      |
| DCell               | containment(n=4)        | graph recon + mse | GO                     |               |             | ~5%    |
| TC_all_graphs       | embed_diff + dango_head | logcosh           | S12.0 + sgd + tf + met | 0.019         | 0.021       | x      |
| TC_all_graphs       | embed_diff + dango_head | icloss            | S12.0 + sgd + tf + met | **0.542**     | **0.431**   | ~60%   |
| TC_all_graphs + fit | embed_diff              |                   | S12.0 + sgd + tf + met |               |             |        |
| TC_all_graphs + fit | embed_diff + dango_head |                   | S12.0 + sgd + tf + met |               |             |        |

- Don't reuse values for publication table, just placeholders for now.