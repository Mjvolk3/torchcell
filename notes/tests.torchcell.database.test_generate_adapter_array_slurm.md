---
id: ms41e0admrpxr36dnftsaus
title: Test_generate_adapter_array_slurm
desc: ''
updated: 1792055339063
created: 1792055339063
---

## 2026.10.15 - What the tests pin

`tests/torchcell/database/test_generate_adapter_array_slurm.py` reads [[database.slurm.scripts.gilahyper_generate_adapter_array-slurm_docker]] and [[database.scripts.kg_generation_array_conductor]] as text and pins the contract the merge relies on: the task is one adapter in `tc-neo4j-gen-<i>` with `--cpus` and `--memory` from the tier, the generator (not the script) writes `DONE` and the script refuses to end without it, an existing marker ends the task with nothing to do, the conductor's fence is re-checked per task, the recovery command is documented at the top, and the conductor records the order and the fence and submits per tier with a dry run.
