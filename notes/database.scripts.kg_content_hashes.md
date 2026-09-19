---
id: 0kzqqr7kdsi9xfhyjbp1xk9
title: Kg_content_hashes
desc: ''
updated: 1789840382356
created: 1789840382356
---

## 2026.09.19 - Content hashes without holding the ids

One dataset at a time: `MATCH (d:Dataset {id: $name})<-[:ExperimentMemberOf]-(e:Experiment)
RETURN e.id ORDER BY e.id` through cypher-shell, the header line dropped, quotes
stripped, piped into `sha256sum`. That is `releases.content_sha256` exactly (sorted ids,
newline-joined, trailing newline); the ids are hex sha256 strings, so Neo4j's ordering
and Python's `sorted()` agree. The live rebuild runs it against the build container
after validation (all datasets), the increment runner against the served container for
the admitted datasets only. A Python version over the CSVs (`content_hashes_from_csv`)
exists for a cross-check but keeps every id in memory (about 12 GB for the 99.7 M
experiments), which a 16 G resume job cannot afford.
