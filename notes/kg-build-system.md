---
id: j5io11ow0xddrpyn9oqlf3j
title: Kg Build System
desc: ''
updated: 1791326502155
created: 1791326502155
---

## 2026.10.06 - The build, release and serving system, typeset

Paired document: `notes-tex/kg-build-system/` (`make` builds `kg-build-system.pdf`, `make diagrams` re-renders the three mermaid figures from [[kg-build-system.mermaid.build-flow]], [[kg-build-system.mermaid.serve-flow]] and [[kg-build-system.mermaid.release-lifecycle]]). It records the system after the pairing change of 2026-10-06 ([[versioning]], PR #680): the five stages and their gates, what the build host provides and what a new host must match, the four pairing gates with the current pairs, Taiga as archive versus store (Neo4j does not run a store on NFS; the Radiant fault), the deploy step, and the two-step fix for the DIVERGED verdict (a block volume on Radiant, then one `kg_release.sh deploy`). Numbers come from jobs 3297 (build), 3337 (ship), 3338 and 3339 (restore test) and are cited in the text.
