---
id: qi7yo95uaxo0bmrhyb2lhs4
title: Apptainer
desc: ''
updated: 1707857784926
created: 1707239794928
---

## Delta Build Database from Origin

Calling this from origin because it is from the beginning of the entire process. This includes recreating apptainer images.

```bash
apptainer build tc-neo4j.sif docker://michaelvolk/tc-neo4j:latest
```

## 2024.02.13 - Docker image Startup vs Apptainer image startup

Database is already started when starting container, so don't need to work about starting. Apparently this is due to entry point in Dockerfile.tc-neo4j but it doesn't seem to work this way with apptainer. Apptainer might not see entry point?
