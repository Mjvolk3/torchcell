---
id: anihr7934plvnxjzesmrhgf
title: Test_generation_array
desc: ''
updated: 1792055322651
created: 1792055322651
---

## 2026.10.15 - What the tests pin

`tests/torchcell/knowledge_graphs/test_generation_array.py` covers [[torchcell.knowledge_graphs.generation_array]] on synthetic adapter directories shaped as `FastCsvSink` plus `BioCypher.write_import_call` leave them: the tier table (the overestimated sizes, the default, every named adapter is a mapped adapter class, a table that names an unknown tier is refused), the order (equal to `build_adapter_map` in its own order, private only with the flag, an index outside the order is an error), the slurm array spelling (`compress_ids`), one sbatch per tier with the tier's sizes and the environment (multi-member values never inside `--export`), the dry run that never calls sbatch, the DONE protocol (missing slots, a marker naming another slot is an error, the per-tier recovery lines), and the merge: one call per label with the first header and comma-joined parts, nodes before relationships, the rewritten edge part under `merged/`, the refusal without a marker, the refusal on a non-identical duplicate node id and edge key with the key and the files named, a key repeated inside one directory left alone, byte-identical headers required, `--skip-duplicate-nodes=true` required from the BioCypher config, and `scan_part` parsing a quoted tab through the csv module.
