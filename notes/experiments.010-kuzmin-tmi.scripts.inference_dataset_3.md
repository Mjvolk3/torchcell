---
id: bh4lflt66dlfueel2mjnyji
title: Inference_dataset_3
desc: ''
updated: 1781029027453
created: 1781029027453
---

## 2026.06.09 - Build the relaxed-threshold inference-3 triple LMDB

This script produces the inference-3 LMDB dataset so the trained model can score a broader candidate pool than inference-2, generated under relaxed selection thresholds (max SMF > 1.04, max DMF > 1.08) with permissive baselines (all SMF > 0.80, all DMF > 0.80). It exists to widen the search beyond the strict inference-2 filter and test whether loosening the fitness-improvement criteria surfaces additional high-value triple interaction candidates.

### Specifics worth keeping

- Reuses `InferenceDataset` and the parallel parquet-to-LMDB loader from inference_dataset_2.py; this script is a thin driver that only changes the input/output directory and thresholds.
- Consumes `DATA_ROOT/data/torchcell/experiments/010-kuzmin-tmi/inference_3/raw/triple_combinations_list.parquet` and writes `inference_3/processed/lmdb/`.
- Allows up to 64 workers (vs 16 in inference-2), reflecting the larger candidate set produced by the relaxed filters.
