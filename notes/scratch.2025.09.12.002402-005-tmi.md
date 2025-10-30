---
id: 1cr5uqi12356mawlnfnmht1
title: 002402 005 Tmi
desc: ''
updated: 1757657917297
created: 1757654646328
---
| Model         | Training Strategy       | Data                                | Train Pearson | Val Pearson |
|:--------------|:------------------------|:------------------------------------|:--------------|:------------|
| Dango         | PreThenPost             | String 9.1                          | 0.5264        | 0.4185      |
| Dango         | LinearUntilUniform      | String 9.1                          | 0.5160        | 0.4196      |
| Dango         | LinearUntilFlipped      | String 9.1                          | 0.5306        | 0.4214      |
| Dango         | PreThenPost             | String 11.0                         | 0.5523        | 0.4137      |
| Dango         | LinearUntilUniform      | String 11.0                         | 0.5099        | 0.4194      |
| Dango         | LinearUntilFlipped      | String 11.0                         | 0.5480        | 0.4201      |
| Dango         | PreThenPost             | String 12.0                         | 0.5365        | 0.4209      |
| Dango         | LinearUntilUniform      | String 12.0                         | 0.5255        | 0.4202      |
| Dango         | LinearUntilFlipped      | String 12.0                         | 0.4768        | 0.4146      |
| DCell         | containment(n=4)        | GO                                  | 0.4147        | 0.2588      |
| TC_all_graphs | embed_diff + dango_head | Str12.0 + Phys + Reg + TFlink + Met | 0.5246        | **0.4385**  |
