---
id: yu2wpu0ann0h165izet00s2
title: '06'
desc: ''
updated: 1770762213945
created: 1770243892728
---

## 2026.02.04

- [x] Reduce thereshold to 0.9 instead of 0.8 which had ~500M samples on 1 gpu projected to take +200 hrs. Restarte with threshold at 0.9 and with DPP over 4 gpu for inference → `bash experiments/010-kuzmin-tmi/scripts/run_inference_3_pipeline.sh`
- [ ] Updated [[workspace.tutorial.md]]
- [ ] Investigate inference_1 vs inference_2 panel comparison plot - inference_1 dataset was 4.37M not 275M, Venn diagram label may be wrong

## 2026.02.08

- [x] Inference 3 shard merge failed with `ArrowInvalid: offset overflow` — all 4 shards (974MB each, ~465M rows) intact. Fixed `merge_parquet_shards` to cast `string` → `large_string` before `take()`. Re-run merge only: `~/miniconda3/envs/torchcell/bin/python experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer_inference_3.py --merge-only`
- [x] After merge, run gene panel selection: `sbatch experiments/010-kuzmin-tmi/scripts/gh_select_12_and_24_gene_top_triples_inference_3.slurm`
- [x] Inference-3 pipeline + final 12-gene panel graduated to durable note → [[Inference Dataset 3|experiments.010-kuzmin-tmi.inference-dataset-3]]

## 2026.06.23

- [x] Drafted the information-accounting argument for the model-construction strategy → [[Information Accounting|paper.information-accounting]] (why the model cuts between the *universe of things* and the *universe of instances*; explains learnable-embedding ≥ seq-embedding for fitness; formalizes the "embeddings only help when sequence changes" hypothesis)
  - [ ] Pin genomics constants: segregating sites $V$, per-site entropy $\bar h$ (related *cerevisiae* genomes), per-assay bits $b$, interaction-matrix rank $r$ — turns the "2–4 OOM" gap into hard Methods numbers
  - [ ] Decide Fig 1 d/e figure direction (concepts A–D + bit-budget inset are in the note); resolve scope: inside Fig 1 vs its own conceptual figure
