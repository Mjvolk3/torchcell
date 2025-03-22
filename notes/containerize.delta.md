---
id: tafdm1c44guiurxk4mpkzfp
title: Delta
desc: ''
updated: 1742648592591
created: 1742642857701
---
## 2025.03.22 - Delta Apptainer Container

Apptainer is already installed on `delta`.

Get rocky linux image. We use Rocky Linux as it is essentially open source CentOS. And open source is obviously king 👑.

```bash
apptainer pull docker://docker.io/library/rockylinux:9
```

Start shell. Home is automatically bound. We bind project and data dirs.

```bash
apptainer shell \
    --nv \
    --bind /projects/bbub:/projects/bbub \
    --bind /scratch/bbub/mjvolk3:/scratch/bbub/mjvolk3 \
    --bind /work:/work \
    rockylinux_9.sif
```

We can source the same env used on machine. This is nice because we don't need to reinstall software and now we can avoid outdated `GLIBC` error.

```bash
source /projects/bbub/miniconda3/bin/activate
conda activate torchcell
```

Then run script.
