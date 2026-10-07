---
id: xild3302sfxriqcfabyv16w
title: Artifact_tier_e2e
desc: ''
updated: 1791375225202
created: 1791375225202
---

## 2026.10.07 - Phase 6 step 1: the Caudal pointer resolves locally, then over tc-data

`python experiments/036-dataset-fixes-before-kg-build/scripts/artifact_tier_e2e.py` (results in `results/artifact_tier_e2e.json`) takes the first member ref the phase 3 loader slice recorded, `tc://genomes/peter2018_1011_assemblies/allReferenceGenesWithSNPsAndIndelsInferred.tar.gz#YAL002W.fasta#AAB_YAL002W_VPS8` (sha256 `b5400b89...33f25`, 160,823,369 bytes), and runs the loop of [[plan.artifact-tier.2026.10.07]] phase 6 on GilaHyper, measured 2026-10-07:

| step | result |
|---|---|
| local resolve through the genomes manifest | `source=local`, verified, 0.293 s (sha256 of 160 MB) |
| same ref with sha256 `000...0` | `ArtifactIntegrityError` at once: the local manifest lists another sha256; nothing falls through to the remote |
| `check` with the local tier hidden, through a tc-data subprocess on a free port | resolvable, 0.002 s, no file written |
| `resolve` with the local tier hidden | `source=remote`, verified, 160,823,369 bytes into `artifact-cache/<sha256>/<basename>`, 0.33 s (488 MB/s over loopback) |
| second remote `resolve` | same path, cache hit, 0.106 s (the re-verification hash) |

The tc-data server was `torchcell.datasets.server` run as a subprocess with the local raw, genomes and objects tiers as its roots and a key minted for the run, so the bytes crossed a real socket. The `Neo4jQueryRaw` gate is not exercised here: the served KG 3.0 carries no `ArtifactRef` yet (its records predate phase 3), so the gate finds zero refs on any query until KG 4.0 is built from the rebuilt dev stores. Steps 2 and 3 of phase 6 (ship, Radiant roots) are [[scripts.ship_artifact_tiers]] and [[database.tc-data-endpoint]].
