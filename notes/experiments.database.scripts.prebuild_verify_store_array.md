---
id: 1tvytmrqn3yfw4u0cipjkjk
title: Prebuild_verify_store_array
desc: ''
updated: 1792076434749
created: 1792076434749
---

## 2026.10.15 - The lane P store-build array

`experiments/database/scripts/prebuild_verify_store_array.slurm` is
`database/slurm/scripts/gilahyper_build_dataset_lmdbs_array.slurm` pointed at a PRIVATE
data root (`VERIFY_DATA_ROOT`): a tree whose `data/<aux>` and `torchcell-*` entries are
symlinks to the shared inputs and whose `data/torchcell/<slug>/raw` is a symlink to the
shared raw mirror, so a task writes only its own `processed/` and `preprocess/` and the
shared dev tree is never touched. One task per class in `CLASSES_FILE`, 16 CPU / 64 G /
8 h each (overestimated on purpose), `--retire-existing --verify`, wrapped in
`/usr/bin/time -v` for wall time and peak RSS. Job 4925 (2026-10-15, nine datasets,
`%2`): numbers and findings in [[database.kg-4-pre-build-verification]].
