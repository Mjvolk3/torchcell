---
id: 8ow6kngzw1uql99ko5jz1u0
title: tc-neo4j-gen
desc: ''
updated: 1792057589804
created: 1792057589804
---

## 2026.10.15 - The generation image: the serving base plus one torchcell commit

`database/docker/Dockerfile.tc-neo4j-gen` builds `tc-neo4j-gen:<sha12>` from the serving image (`michaelvolk/tc-neo4j:5.26.28-browser.1`, `ARG BASE_IMAGE`) by running, at build time, the three installs the serial generation container runs at start: `pip install --no-cache-dir --force-reinstall --no-deps git+https://github.com/Mjvolk3/torchcell.git@$BUILD_COMMIT`, `biocypher==0.15.2`, `git+https://github.com/oxpig/CaLM@main`. Then it reads the commit pip recorded in `torchcell`'s `direct_url.json` and refuses to finish unless it equals `BUILD_COMMIT`, and stamps the labels `org.torchcell.build_commit` and `org.torchcell.base_image`. Nothing else differs from the base: the same entrypoint, user and Neo4j, so a task's `docker run --entrypoint bash ... tc-neo4j-gen:<sha12>` is the serial container's line with the install lines removed.

Built, checked and removed only through `python -m torchcell.knowledge_graphs.generation_image` ([[torchcell.knowledge_graphs.generation_image]]), which derives the tag from the full commit and passes `BUILD_COMMIT` and `BASE_IMAGE` as build args with `database/docker` as the (empty) context. The dependency set is the base image's: a commit that imports a package the base lacks fails at import in the task, exactly as the install path does, and the fix is a base rebuild (`database/Dockerfile.tc-neo4j` installs `torchcell@main` for that reason).
