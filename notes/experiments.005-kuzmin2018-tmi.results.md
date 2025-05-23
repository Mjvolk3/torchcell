---
id: w8rypmcerm9cotkmoux04hd
title: Results
desc: ''
updated: 1748033820589
created: 1747352284597
---


| Model                   | Training Strategy       | Data                                | Train Pearson | Val Pearson | Status |
|:------------------------|:------------------------|:------------------------------------|:--------------|:------------|:-------|
| Dango                   | PreThenPost             | String 9.1                          | 0.5264        | 0.4185      | x      |
| Dango                   | LinearUntilUniform      | String 9.1                          | 0.5160        | 0.4196      | x      |
| Dango                   | LinearUntilFlipped      | String 9.1                          | 0.5306        | **0.4214**  | x      |
| Dango                   | PreThenPost             | String 11.0                         | 0.5523        | 0.4137      | x      |
| Dango                   | LinearUntilUniform      | String 11.0                         | 0.5099        | 0.4194      | x      |
| Dango                   | LinearUntilFlipped      | String 11.0                         |               |             | x      |
| Dango                   | PreThenPost             | String 12.0                         | 0.5365        | **0.4209**  | x      |
| Dango                   | LinearUntilUniform      | String 12.0                         | 0.5255        | **0.4202**  | x      |
| Dango                   | LinearUntilFlipped      | String 12.0                         | 0.4768        | 0.4146      | x      |
| DCell                   | containment(n=4)        | GO                                  |               |             | issue  |
| TC_all_graphs           | embed_diff              | Str12.0 + Phys + Reg + TFlink + Met |               |             | x      |
| TC_all_graphs + fitness | embed_diff              | Str12.0 + Phys + Reg + TFlink + Met |               |             | debug  |
| TC_all_graphs + fitness | embed_diff + dango_head | Str12.0 + Phys + Reg + TFlink + Met |               |             | debug  |

Pearson values were read off of charts according to the best pearson value since we have model saved saved at val best pearson. It is probably better to do this programmatically but we just want a sense of the trends.

There are seems to be no benefit of updating to String 12.0