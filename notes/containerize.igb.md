---
id: kha3o8zabbygfd74tcfcfrc
title: Igb
desc: ''
updated: 1742647396555
created: 1742642852686
---
## 2025.03.22 - IGB Singularity Container

Get rocky linux image. We use Rocky Linux as it is essentially open source CentOS. And open source is obviously king 👑.

Apptainer used to be called singularity. We don't have updated Apptainer on IGB.

```bash
module load singularity
```

```bash
singularity pull docker://docker.io/library/rockylinux:9
```

We don't have to bind anything additionally on `igb` since everything is in default bound `$HOME`.

```bash
singularity shell --nv rockylinux_9.sif
```

We can source the same env used on machine. This is nice because we don't need to reinstall software and now we can avoid outdated `GLIBC` error.

```bash
source $HOME/miniconda3/bin/activate
conda activate torchcell
```
