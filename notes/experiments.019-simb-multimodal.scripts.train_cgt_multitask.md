---
id: pw5vim74rhfhtemexxgtrrk
title: Train_cgt_multitask
desc: ''
updated: 1784699671781
created: 1784699671781
---

## 2026.07.22 - WS13 Fig-3 multitask CGT training harness + cluster launch

Compute harness for the Fig-3 multitask Cell Graph Transformer (WS7 heads on the
WS2 Fig-3 multimodal build). Authored on GilaHyper/M1; real runs launch on IGB
(mmli/cabbi) and Delta with NO Claude Code on those machines (write here, transfer,
`sbatch` there).

### Files

- Training script:
  `experiments/019-simb-multimodal/scripts/train_cgt_multitask.py`
- Configs (`experiments/019-simb-multimodal/conf/`):
  - `default.yaml` -- Hydra base (profiler/logging).
  - `train_cgt_multitask.yaml` -- **workstation-verify** (tiny: 16-d model, 128
    subset, 1 GPU, 2 epochs). Default `@hydra.main` config; use for `dry_run=true`
    and a fast `trainer.fast_dev_run=true` smoke test.
  - `igb_mmli_train_cgt_multitask_000.yaml`, `igb_cabbi_train_cgt_multitask_000.yaml`,
    `delta_train_cgt_multitask_000.yaml` -- full 4-GPU DDP joint runs (180-d, 8
    layers, 9 heads, 600 epochs).
- SLURM (same dir): `igb_mmli_*.slurm` (`-p mmli`), `igb_cabbi_*.slurm`
  (`-p cabbi`, `expandable_segments:True`), `delta_*.slurm` (`gpuA40x4`,
  `bbub-delta-gpu`, apptainer + `/projects/bbub` binds).

### Head selection (individual baseline vs joint)

`multitask.active_heads` picks which heads train, so ONE script does both:

- individual baseline: `multitask.active_heads=[per_gene]` (or `[global]`,
  `[gene_interaction]`).
- joint: `multitask.active_heads=[gene_interaction,per_gene,global]`.

Heads: `gene_interaction` = the built-in scalar `perturbation_head` (fitness);
`per_gene` = `PerGeneHead` (expression, `[B,N]`); `global` = `GlobalHead`
(CalMorph 501-D); `per_metabolite` = `PerMetaboliteHead` (needs the metabolism
incidence graph -- only pulled in when that head is active). Loss =
`MaskedMultitaskLoss` (each head masked to genotypes carrying that phenotype;
graph-reg term added unchanged).

### Dry-run / verification

- `python .../train_cgt_multitask.py --help` and `... --cfg job` -- config OK.
- `python .../train_cgt_multitask.py dry_run=true` -- builds the model from config
  and runs ONE synthetic forward + masked loss + backward on a tiny synthetic
  `cell_graph`/`batch` (no genome/dataset/wandb/GPU). Works for every head combo
  and for the full cluster configs (`--config-name igb_mmli_... dry_run=true`).
- mypy + ruff clean on the `.py`; `bash -n` clean on all three `.slurm`.

### Head-wiring ASSUMPTIONS (validate on first real Fig-3 batch)

1. **Graph processor.** The transformer consumes per-genotype
   `perturbation_indices` batches, so the harness (re)builds the dataset with the
   `Perturbation` processor -- NOT the `SubgraphRepresentation` that
   `query_fig3.py` used for the census. First cluster run materializes this build
   under `$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig3_core`.
2. **Target/mask decode.** `MultitaskCGTTask._extract_targets_and_masks` decodes
   per-head targets + supervision masks from the COO `phenotype_values` /
   `phenotype_type_indices` / `phenotype_sample_indices` fields on `batch['gene']`,
   keyed by `multitask.head_phenotypes` (head -> phenotype-type-name list). The
   name strings (`microarray_expression`, `calmorph`, `fitness`, ...) are a
   best-effort placeholder and MUST be reconciled against the actual
   `phenotype_types` strings in a materialized batch before a production run. The
   synthetic dry-run does not touch this path.

## Transfer

What to move to each cluster before `sbatch` (code goes via `git pull`; large data
lives under that cluster's `DATA_ROOT` and is NOT git-tracked).

### 1. Code (all clusters) -- git

Land this branch, then on each cluster:

```bash
cd <cluster>/torchcell && git pull   # brings scripts + conf + queries
```

Paths per cluster (match the SLURM `cd`/python paths):

- IGB mmli & cabbi: `/home/a-m/mjvolk3/projects/torchcell`
- Delta: `/scratch/bbub/mjvolk3/torchcell`

### 2. Fig-3 dataset LMDB (under each cluster's `DATA_ROOT`)

`$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig3_core/` (the
`Perturbation`-processor build). If absent on the target cluster, the first run
builds it from the Neo4j DB + the mirrored query
(`queries/fig3_core.cql`), which needs DB reachability + the node-embedding /
genome caches below. Alternatively `rsync` a prebuilt LMDB from GilaHyper:

```bash
rsync -a $DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig3_core/ \
  <cluster>:$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig3_core/
```

### 3. Genome + node-embedding caches (under each cluster's `DATA_ROOT`)

Needed by the full training path (not the dry-run): `data/sgd/genome`, `data/go`,
`data/string`, `data/tflink`. (Fig-3 configs use learnable gene embeddings, so no
external embedding tensors are required; if a config sets
`cell_dataset.node_embeddings`, ship the matching
`data/scerevisiae/*_embedding` dirs too.)

### 4. Container

The SLURM scripts `singularity/apptainer exec` `rockylinux_9.sif` at the repo root
on each cluster (IGB: `/home/a-m/mjvolk3/projects/torchcell/rockylinux_9.sif`;
Delta: `/scratch/bbub/mjvolk3/torchcell/rockylinux_9.sif`). Ensure the image is
present + the `torchcell` conda env exists inside it.

### 5. Launch

```bash
# IGB mmli
sbatch experiments/019-simb-multimodal/scripts/igb_mmli_train_cgt_multitask_000.slurm
# IGB cabbi
sbatch experiments/019-simb-multimodal/scripts/igb_cabbi_train_cgt_multitask_000.slurm
# Delta
sbatch experiments/019-simb-multimodal/scripts/delta_train_cgt_multitask_000.slurm
```

For an individual-phenotype baseline, override on the CLI (or clone a config):
`... --config-name igb_mmli_train_cgt_multitask_000 'multitask.active_heads=[per_gene]'`.
Ensure the SLURM output dir exists on the cluster
(`.../experiments/019-simb-multimodal/slurm/output/`).
