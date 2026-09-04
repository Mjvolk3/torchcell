---
id: atgagb4dcdfjkp5er4ax4cp
title: Equivariant_cell_graph_transformer_inference_4
desc: ''
updated: 1788502078653
created: 1788502078653
---

## 2026.09.04 - Running inference_4 over four GPUs

Script: `experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer_inference_4.py`
Configs: `conf/equivariant_cell_graph_transformer_inference_4_m0{0,1,2}.yaml`
SLURM: `scripts/gh_inference_4_m{00,01,02}_shard{0,1,2,3}.slurm`

Same model, loader and streaming-Parquet machinery as the inference_1 script. Three
differences, all forced by the space being ten times larger.

### Sharding

`inference.n_shards` and `inference.shard_index` are **required** config keys, read
with `[]` rather than `.get()`. A missing key fails at startup. The alternative, a
default of one shard, is the dangerous option: an unsharded run looks identical to a
correct one for the first hour and then takes four times as long.

Shards are contiguous index ranges of the LMDB, so concatenating the per-shard Parquet
files in shard order reproduces the full prediction vector in dataset order. Output
filenames carry `_shard{i}of{n}` so four writers never collide.

### Two bugs sharding would have caused silently

Both were in code inherited from the inference_1 script and both would have produced a
file that looked fine.

1. **The gene-name lookup took the dataset and a local index.** Under sharding the
   dataset handed to the loader is a `torch.utils.data.Subset`, which has no
   `_init_lmdb_read`, and the index is a position within the shard rather than an LMDB
   key. The function now takes the underlying dataset and a global index.

2. **It swallowed every exception and returned an empty list.** Combined with (1) that
   is a silent and total failure: the `AttributeError` on the `Subset` would have been
   caught, every row would have carried an empty gene column, and nothing would have
   reported it. Errors now propagate, and a missing LMDB key raises.

The recorded `index` is likewise the global one, `shard_offset + position`, so a shard
file can be read on its own or concatenated without renumbering.

`Subset` also hides `cell_graph`, which the model constructor needs, so the constructor
reads it from the underlying dataset.

### Cost

41,877,232 triples per checkpoint at the measured 1,505 triples/s on one GilaHyper GPU
is 7.7 GPU-hours, so roughly 1.9 h per shard. Twelve jobs cover three checkpoints;
SLURM runs four at a time, so about 5.8 h wall clock for all three.

The stricter support tiers are subsets of this space, so filtering
`triple_index.parquet` on `n_supported >= 2` or `== 3` answers those without rerunning.
