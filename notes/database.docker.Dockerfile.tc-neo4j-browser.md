---
id: xdgcmh9jeta4t9qo4kxzu03
title: tc-neo4j-browser
desc: ''
updated: 1789718487784
created: 1789718487784
---

## 2026.09.18 - The serving image is the base image plus the Browser seed

`michaelvolk/tc-neo4j:5.26.28-browser.1` is `michaelvolk/tc-neo4j:5.26.28` with one
layer on top: `database/browser/patch_browser_jar.py` run against the Browser jar with
`database/browser/torchcell-seed.js`. Nothing else changes (same neo4j, same python
environment), so the tag serves every container of the live rebuild: generation, build,
and serving. The base image is a 12.7 G build (torchcell from GitHub main, torch-scatter
from source); this layer builds in three seconds, so the seed can be refreshed without
rebuilding the base.

```bash
python -m torchcell.database.browser_style          # regenerate the seed from the schema
docker build -f database/docker/Dockerfile.tc-neo4j-browser \
    -t michaelvolk/tc-neo4j:5.26.28-browser.1 database
docker push michaelvolk/tc-neo4j:5.26.28-browser.1
```

Bump the `-browser.N` suffix when the seed changes and point `IMAGE` in
`gilahyper_live_rebuild-slurm_docker.slurm` and `gilahyper_increment_kg-slurm_docker.slurm`
at it. The seed re-runs in a visitor's browser only when the stylesheet's sha256 changes,
so an unchanged stylesheet in a new tag disturbs nobody's local styling.
