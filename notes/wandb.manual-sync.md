---
id: y8oq40dcoz8lblgisqd11k7
title: Manual Sync
desc: ''
updated: 1715730612799
created: 1715730599462
---
```bash
(torchcell) mjvolk3@dt-login01 wandb % pwd                                                                          6:52
/scratch/bbub/mjvolk3/torchcell/wandb-experiments/3487237/wandb
(torchcell) mjvolk3@dt-login01 wandb % for d in $(ls -t -d */); do wandb sync $d; done    
```
