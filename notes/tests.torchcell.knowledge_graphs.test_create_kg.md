---
id: 04zptjd7o7dw7z74r8dyuly
title: Test_create_kg
desc: ''
updated: 1790649266553
created: 1790649266553
---

## 2026.09.28 - The registry-driven KG build script on a fake BioCypher (Phase 9)

4 tests through the shared fakes in `tests/torchcell/knowledge_graphs/_kg_build_fakes.py` (recording BioCypher, wandb, loaders and adapters; a pinned clock; `import_build_module`, which snapshots and restores the whole environment because `create_kg.py` runs `load_dotenv()` at import). Worker split (ceil of the io fraction), the complete ordered wandb payloads, the exact BioCypher call list and the exact loader and adapter kwargs for the registry datasets, the uuid fallback for the wandb group without a SLURM job. Findings: line 195 writes the literal `biocypher-out` into `biocypher_file_name.txt` while the output directory at line 76 is built from `BIOCYPHER_OUT_PATH`, so the recorded script path does not point at the directory written (the three sibling scripts share the pattern); lines 119 to 135 construct `SCerevisiaeGenome` twice for a loader declaring both `scerevisiae_graph` and `genome`, and every loader receives `io_workers=num_workers`, the total rather than the io share. Coverage 0% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
