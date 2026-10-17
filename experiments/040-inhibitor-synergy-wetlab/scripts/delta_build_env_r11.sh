#!/bin/bash
# experiments/040-inhibitor-synergy-wetlab/scripts/delta_build_env_r11.sh
# Run on a Delta login node with conf/delta_env_r11_requirements.txt copied to
# /scratch/bbub/mjvolk3/envs/requirements-r11.txt.
# Rebuild of /work/hdd/bbub/miniconda3/envs/torchcell, package for package, after
# 5,863 of its files were left unreadable by Lustre target 106 (2026-10-16). The 33
# conda packages at their exact builds from Anaconda main, then 263 pip packages at
# their exact versions with --no-deps (the old env's dist-info list). torchcell itself
# is NOT installed: the 040 jobs import it from the tree through PYTHONPATH.
set -euo pipefail
ROOT=/scratch/bbub/mjvolk3/envs
ENV=$ROOT/torchcell-r11
mkdir -p $ROOT/bin $ROOT/pipcache $ROOT/mamba
cd $ROOT
echo "[$(date -Is)] micromamba"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $ROOT bin/micromamba
export MAMBA_ROOT_PREFIX=$ROOT/mamba
echo "[$(date -Is)] conda layer"
$ROOT/bin/micromamba create -y -p $ENV --override-channels -c pkgs/main _libgcc_mutex=0.1=main _openmp_mutex=5.1=52_gnu bzip2=1.0.8=h5eee18b_6 ca-certificates=2026.7.16=h06a4308_0 ld_impl_linux-64=2.44=h9e0c5a2_3 libexpat=2.8.2=h7354ed3_1 libffi=3.4.8=h06d3fd0_3 libgcc=15.2.0=h69a1729_8 libgcc-ng=15.2.0=h166f726_8 libmpdec=4.0.1=h47b2149_0 libstdcxx=15.2.0=h39759b7_8 libuuid=1.41.5=h5eee18b_0 libxcb=1.17.0=h9b100fa_0 libzlib=1.3.2=h47b2149_0 ncurses=6.5=h7934f7d_0 openssl=3.5.7=h1b28b03_0 packaging=26.0=py313h06a4308_0 pip=26.1.2=pyhc872135_0 pthread-stubs=0.3=h0ce48e5_1 python=3.13.14=h9631c4f_102_cp313 python_abi=3.13=4_cp313 readline=8.3=hc2a1206_0 setuptools=83.0.0=py313h06a4308_0 sqlite=3.53.2=h795bf6d_0 tk=8.6.15=h54e0aa7_0 tzdata=2026c=he532380_0 wheel=0.47.0=py313h06a4308_0 xorg-libx11=1.8.12=h9b100fa_1 xorg-libxau=1.0.12=h9b100fa_0 xorg-libxdmcp=1.1.5=h9b100fa_0 xorg-xorgproto=2024.1=h47b2149_2 xz=5.8.2=h448239c_0 zlib=1.3.2=h47b2149_0 
echo "[$(date -Is)] pip layer"
$ENV/bin/python -m pip install --no-deps --no-input --cache-dir $ROOT/pipcache \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html \
  -r $ROOT/requirements-r11.txt
echo "[$(date -Is)] check"
$ENV/bin/python -m pip check || true
$ENV/bin/python -c "import torch, torch_geometric, torch_scatter, torch_sparse, rdkit, wandb, pydantic, lmdb, pandas, sklearn; print('imports ok', torch.__version__, torch_geometric.__version__)"
echo "[$(date -Is)] DONE"
