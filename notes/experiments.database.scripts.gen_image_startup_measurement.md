---
id: s7b1jnepu8kk9xh4wo4qwfv
title: Gen_image_startup_measurement
desc: ''
updated: 1792057598853
created: 1792057598853
---

## 2026.10.15 - Container start to first generator line: install path against the image

`experiments/database/scripts/gen_image_startup_measurement.slurm` is one slurm task (16 CPU, 64 G, 2 h) that builds the generation image ([[torchcell.knowledge_graphs.generation_image]]) at `MAIN_COMMIT` and at `GEN_COMMIT` (both timed through `generation_image ensure`), then generates ONE small adapter (`ADAPTER_INDEX` 85, `IsopentenolTiterFoo2014Adapter`, 56 K of LMDB) once in the install path (the base image plus the three pip installs, as the serial build does) and `RUNS` times from the image, one container at a time. Every container line is stamped on the host with the UTC epoch, and the markers `T0_RUN_START` (host, before `docker run`), `CONTAINER_SHELL_UP`, `INSTALL_DONE` or `COMMIT_VERIFIED`, `Generating adapter` (the generator's first own line), `DONE=` and `T1_RUN_END` bound the intervals. The outputs go under `$BUILD_ROOT/database/gen-image-measure/<job>/`, never under `/db/database`; the W&B runs carry `TCDB_TAGS=gen-image-startup,<mode>`.

`experiments/database/scripts/gen_image_startup_measurement.py` reads the job's `.out` and writes `experiments/database/results/gen_image_startup_<job>.{json,md}` (the results directory is generated, not tracked): per run the shell-up, ready, startup (`Generating adapter` minus `T0_RUN_START`), `DONE` and total seconds, and the projection against two baselines, the in-job install run of the same adapter and lane C's 278 s from job 4919_64: saving per task times 170 tasks for CPU hours, over the concurrency (4) for wall clock.

Two commits because `generate_adapter` lives on lane C's branch (`kg-generation-array`, PR #1110) until it lands: the deliverable image is main's (`fcb3935c08`), the generator's is lane C's head (`068779a3c4`). Startup is a property of the container, not of the adapter or the commit, so the pairing is honest; the two build times are both reported.

### Measured: jobs 4923 and 4924 (2026-10-15, GilaHyper, 16 CPU, 64 G)

Job 4923 built both images from nothing and then failed on a bash bug in the script (a `local` line expanding a variable it was assigning, fixed in the same commit); job 4924 ran the four generations. The build times come from 4923 and the run timings from 4924; the table below is `experiments/database/results/gen_image_startup_4924.md` as the analysis script wrote it, with the two cold builds added by hand from 4923's `.out`.

| image | cold build (job 4923) | rebuild of the same commit (job 4924) |
|---|---|---|
| `tc-neo4j-gen:fcb3935c088d` (main, the deliverable) | 259 s | 0 s (BuildKit layer cache; `docker image rm` drops the tag, not the layers) |
| `tc-neo4j-gen:068779a3c431` (lane C head, the generator) | 216 s | 0 s |

| run | mode | image | shell up (s) | installed / verified (s) | startup to first generator line (s) | DONE (s) | total (s) |
|---|---|---|---|---|---|---|---|
| 0 | install | michaelvolk/tc-neo4j:5.26.28-browser.1 | 0.5 | 239.3 | 255.9 | 257.3 | 263.1 |
| 1 | image | tc-neo4j-gen:068779a3c431 | 0.6 | 0.7 | 17.3 | 19.0 | 23.8 |
| 2 | image | tc-neo4j-gen:068779a3c431 | 0.7 | 0.7 | 17.0 | 18.5 | 22.2 |
| 3 | image | tc-neo4j-gen:068779a3c431 | 0.2 | 0.2 | 13.1 | 14.4 | 18.3 |

| baseline | baseline startup (s) | image startup mean (s) | saving per task (s) | tasks | concurrency | CPU hours saved | wall hours saved |
|---|---|---|---|---|---|---|---|
| job 4924 install run (same adapter, same job) | 255.9 | 15.8 | 240.0 | 170 | 4 | 11.33 | 2.83 |
| job 4919_64 install path (lane C, 2026-10-15) | 278.0 | 15.8 | 262.2 | 170 | 4 | 12.38 | 3.10 |

Reading: the install path spends 239 s of its 256 s in the three pip installs and the image path spends 0.2 to 0.7 s in the commit check; what remains in both (13 to 17 s) is python importing torchcell, hydra and wandb before the generator's first line, which the image cannot remove. The adapter itself (`IsopentenolTiterFoo2014Adapter`, in-process) generates in 1.3 to 1.5 s in every run, so the numbers are the container's, not the adapter's. One cold image build (216 to 259 s) costs about one task's install and pays back on the second task; a rebuild of the same commit is free while the layer cache holds. Projection at concurrency 4 over the 170-task array: about 11.3 h of CPU time and 2.8 h of wall clock against the paired in-job baseline, 12.4 h and 3.1 h against lane C's 278 s. The W&B runs: project `tcdb`, tags `gen-image-startup,install` and `gen-image-startup,image`.
