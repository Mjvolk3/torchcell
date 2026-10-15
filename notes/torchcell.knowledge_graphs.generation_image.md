---
id: b3f7tm6t5w6k59iqrsxh8zl
title: Generation_image
desc: ''
updated: 1792057581083
created: 1792057581083
---

## 2026.10.15 - The prebuilt generation image, tagged at the build commit

`torchcell/knowledge_graphs/generation_image.py` owns the image a per-adapter generation task runs from. A generation container installs the torchcell commit it builds before the generator starts (`pip install --no-deps git+...@BUILD_COMMIT`, then `biocypher==0.15.2` and CaLM); lane C measured that at 278 s per container (job 4919_64, 2026-10-15), and the array runs 170 containers, 13.1 h of CPU time. The image (`database/docker/Dockerfile.tc-neo4j-gen`, [[database.docker.Dockerfile.tc-neo4j-gen]]) carries that install once, so a task starts straight into the generator.

Three rules live in the module, not in the shell scripts that call it:

1. **The tag derives from the full commit and from nothing else.** `GenerationImage(build_commit=<40-hex sha>)` gives `tc-neo4j-gen:<first 12>`; a branch, a short sha, a tag name or an uppercase sha fails validation. The Dockerfile stamps the full commit as the label `org.torchcell.build_commit`, and pip records it in torchcell's `direct_url.json`, which the build checks before the image exists. `image_commit` reads the label back; a tag whose label is another commit, or none, is a `GenerationImageMismatch`. A stale image therefore never serves another commit silently.
2. **No fallback.** `resolve_mode(ImageMode.IMAGE, ...)` with the tag absent raises `GenerationImageAbsent`, naming the tag, the commit and the build command, and installs nothing. `ImageMode.INSTALL` keeps the install path (the serial build, a one-off run) and never touches docker. `ImageMode.AUTO` is a selection made once before a run (image when the tag exists, install otherwise, printed as `RESOLVED_MODE=`); the conductor resolves it before the arrays are submitted, so every task runs under an explicit `image` or `install`.
3. **The lifecycle is the conductor's, and docker is outside slurm.** `ensure` builds before the arrays are submitted (`IMAGE_BUILD_SECONDS=` printed); `teardown_decision(merge_succeeded, tasks_without_done)` removes after a successful merge and keeps the image while any task lacks its `DONE` marker, so the re-run of the failed tasks is cheap; `remove` is the by-hand cleanup after a cancelled or timed-out job, which leaves the image in place because the daemon, not the job, owns it. `docker images tc-neo4j-gen` is the inventory.

CLI: `tag`, `check` (exit 1 with the reason when absent or mismatched), `resolve --mode <image|install|auto>`, `ensure`, `build`, `remove`, each with `--build-commit <sha>`; `--dry-run` prints every docker command prefixed `DRY-RUN` and runs none. The serial live rebuild keeps its in-container install unchanged. The array task and its conductor ([[database.scripts.kg_generation_array_conductor]], [[database.slurm.scripts.gilahyper_generate_adapter_array-slurm_docker]], lane C's PR #1110) read `GEN_IMAGE_MODE` once the three hooks land on top of that PR: the conductor runs `ensure` before its sbatch lines and passes `--gen-image-mode`, the task runs `resolve` before its container and runs from `$RUN_IMAGE` with the install lines replaced by the commit check, and the live rebuild's follow-up runs `remove` after the merge.

Measured: [[experiments.database.scripts.gen_image_startup_measurement]]. Tests: [[tests.torchcell.knowledge_graphs.test_generation_image]].

## 2026.10.15 - The three hooks are wired, and the image is measured in a real array

This module landed without a caller (check-in 08:35 CDT: "its three hooks into lane C's
array files wait on PR #1110 landing"). They are in place now: the conductor resolves the
mode and builds the tag (`GEN_IMAGE_MODE`), `generation_array --gen-image` carries the tag
into the task environment, the array task re-reads the tag's commit label and refuses
rather than installing, and the live rebuild calls `remove` after the merge succeeds,
which is `teardown_decision`'s `remove` case made concrete.

Measured in the first array to use it, job 4974, 47 tasks: **startup 11.0 s median, 11.2 s
mean, 10 to 14 s**, against this module's own 15.8 s mean on the install-versus-image
measurement and 255.9 s on the install path. Cold build 434 s on a contended node;
a same-commit rebuild was free, as designed
([[experiments.database.bacterial-generation-profile]]).
