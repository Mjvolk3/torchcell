---
id: bcq2kblmm21r0aqb8uvx1td
title: '13'
desc: ''
updated: 1715731429102
created: 1710347696344
---
## 2024.03.13 11:35 - Troubleshooting

- On latest docker build we have an #OOM ![](./assets/images/user.Mjvolk3.torchcell.tasks.md.latest-docker-build-oom.png), we are adjusting memory giving it now all 32 GB and only 1 GB swap. Should fail fast with low swap. This doesn't solve no cpu issue.
- Although we set the number of cpus in docker desktop, this is the number of cpus available to the daemon. It does not specify the number of cpus available to a a given container. For that we can pass the cpus flag.

```bash
docker run --cpus 10
```

## 2024.05.14 - Build Summary

```bash
IMPORT DONE in 7m 32s 617ms. 
Imported:
  49722079 nodes
  107306004 relationships
  249192120 properties
Peak memory usage: 1.554GiB
```
