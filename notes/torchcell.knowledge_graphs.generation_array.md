---
id: qi08dln0ueuc03xk5b6xxgm
title: Generation_array
desc: ''
updated: 1792055272817
created: 1792055272817
---

## 2026.10.15 - The per-adapter generation array, its merge and its duplicate check

`torchcell/knowledge_graphs/generation_array.py` is the host-side half of the per-adapter KG generation (Path U, lane C of [[scratch.2026.10.15.ontology-unification-decision]]). The serial generation (`create_scerevisiae_kg_small`, one container, one loop over 170 adapters, no checkpoint) restarts from adapter 1 when adapter 124 fails; job 4901 lost 17 h 38 min that way. The array runs one slurm task per adapter ([[database.slurm.scripts.gilahyper_generate_adapter_array-slurm_docker]], the task entry [[torchcell.knowledge_graphs.generate_adapter]]), each into `biocypher-out/<ts>/adapters/<iii>_<Adapter>/` with a `DONE` marker written last, and this module turns the 170 directories back into one import.

### Order

Index `i` is position `i` of `build_adapter_map(include_private, organisms)`: the public map, then the private map under `--include-private`, narrowed to the organism set. That is the order the serial build walks, so `EnvChemgenHoepfner2014Adapter` is index 64 (job 4901's "adapter 65 of 170"). `adapter_order` is the one place that maps an index to a dataset and an adapter; the conductor writes it to `<out>/adapter_order.json` and every task and the merge read the recorded list, so the array and its merge cannot disagree on membership.

### Tiers

[[torchcell.knowledge_graphs.conf.generation_tiers.yaml]] keys adapter names to a tier (`cpus`, `mem_gb`, `hours`, `concurrency`). A slurm array carries one resource spec, so `tier_submissions` emits one `sbatch --array=<ids>%N --cpus-per-task --mem --time` per tier, with `GEN_CPUS` and `GEN_MEM_GB` in the task environment as the container's `--cpus` and `--memory`. Multi-member values (`KG_ORGANISMS`, hydra overrides) travel in the environment with `--export=ALL`, never inside `--export`'s list, which sbatch splits on commas (job 4899).

### The DONE protocol and recovery

`DoneMarker` (index, classes, dataset length, node and edge counts, wall time, peak memory, package version, build commit) is the generator's last write, after `sink.finish()` and `bc.write_import_call()`. `missing_slots` lists the slots whose directory has no marker (a marker naming another slot is an error, not a DONE), and `recovery_commands` prints the exact per-tier sbatch lines that re-run them:

```bash
python -m torchcell.knowledge_graphs.generation_array missing --out-dir /db/database/biocypher-out/<ts> \
    --out-name <ts> --torchcell-src <checkout> --build-commit <sha> --include-private --organisms scerevisiae,ecoli,pputida
```

### Merge

`merge_outputs` refuses while any directory lacks its marker (naming the indices), checks that every label's header is byte-identical across the directories carrying it, runs the duplicate check, and writes one `<out>/neo4j-admin-import-call.sh`: per label the first directory's header and the comma-joined part files of every directory carrying it, nodes before relationships. The command prefix (delimiter, quote, `--overwrite-destination`, `--skip-bad-relationships`, `--skip-duplicate-nodes`) is transplanted from a per-directory call BioCypher wrote, so the flags are the BioCypher config's; the merge refuses a config without `skip_duplicate_nodes: true`. One argument must stay under the kernel's 128 KiB single-argument limit; a label carried by all 170 directories is about 15 KB. `merge_report.json` carries the per-label counts, and `kg_manifest bootstrap --merge-report` records them in the build manifest (`GenerationRecord`).

### Duplicates

BioCypher and `FastCsvSink` deduplicate within one process; across directories content-id nodes recur (genome, environment, media, temperature, perturbation, genotype, experiment reference, interned constant). The check hashes every row of every part file (a 128-bit blake2b of the key, a 64-bit blake2b of the whole line; the key is the node id or `start\x00end` for an edge), sort-merges per label with numpy, and treats a key seen in two or more directories as a duplicate. Node rows: `--skip-duplicate-nodes=true` keeps the first row neo4j sees, and the check proves every later row is byte-identical. Edge rows: neo4j has no skip for duplicate relationships, so the later directories' rows of a duplicate key are dropped into a rewritten copy under `<out>/merged/<dir>/` that the import call references (the adapter directories are never modified). A duplicate that is not byte-identical refuses the merge with the label, the key and the files (`DuplicateConflict`); a key repeated inside one directory is what that adapter wrote and is kept. A row whose tab count differs from the header's is parsed with the csv module (a tab inside a quoted value), so the fast split is exact where it is used.

Cost, measured on two symlinked copies of the job 4901 partial (383 G, 49 labels, 124 adapters' rows; every key a duplicate, check only): see the dated section below once the run completes.

Tests: `tests/torchcell/knowledge_graphs/test_generation_array.py` (the tier sizes and that every named adapter is mapped, the order against `build_adapter_map`, the sbatch plan and the dry run, the marker protocol, the merge on synthetic directories with and without DONE, identical and non-identical duplicates for nodes and edges, the header check, the quoted-tab parse).
