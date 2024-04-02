---
id: k94p3jgmz1mrr355fsg3mvx
title: Docker_v_m1_study_001
desc: ''
updated: 1711575763902
created: 1711575194606
---
[wandb_log](https://wandb.ai/zhao-group/tcdb/workspace?nw=nwusermjvolk3)

## Obvious Difference Between Using Docker and Just M1

![](./assets/images/tags.wandb.tcdb.docker_v_m1_study_001.md.docker_v_m1_study_001-cpus_dont_engage_in_docker.png)

## M1 Charts

Finished

![](./assets/images/tags.docker_v_m1_study_001.md.m1-no-docker-completion-times.png)

![](./assets/images/tags.docker_v_m1_study_001.md.m1-no-docker-system-utilization.png)

## Docker M1 Charts

Crashed

![](./assets/images/tags.docker_v_m1_study_001.md.m1-with-docker-completion-times.png)

![](./assets/images/tags.docker_v_m1_study_001.md.m1-with-docker-system-utilization.png)

We don't finish the last part of the job, the `DmfCostanzo2016` write edges due to what looks like an #OOM from looking at the system memory utilization plots.

## Conclusions

My best explanation that there is overhead with docker or that there is some translation layer for ARM... We are using ARM build so there shouldn't be any emulation...
