---
id: uxz4ggr6wv3tr9eat4d8x6b
title: Prebuild_verify_stores
desc: ''
updated: 1792076418541
created: 1792076418541
---

## 2026.10.15 - Lane P: prove the sample stores built from the integrated branch are right

`experiments/database/scripts/prebuild_verify_stores.py` walks every record of each
store the lane built from `ontology-unify` under its private data root and reports
counts, never impressions: records against the served KG 3.0 release and against the
shared dev tree `main` built on 2026-10-13; `gene_namespace` on every perturbation
against the host the dataset implies; `assembly_set` on every genome reference or a
typed gap with reason and `looked_in`; every `GeneAdditionPerturbation` placement
resolved to the deposited contig through the construct registry (bytes re-hashed) or a
typed gap; retired experiment-type tags; a three-way content comparison (3.0 served,
main tree, branch store) on a key no proposal rewrote (gene names, phenotype value,
media, temperature, environment perturbations); the generated CSVs (rows per label,
`host_genome` on every perturbation row against the genome node's `assembly_set`,
experiment-id overlap with 3.0); and the build and generation timings off the slurm
logs. Results and the tables: [[database.kg-4-pre-build-verification]]. The companion
scripts are [[experiments.database.scripts.prebuild_verify_served_dump]] (the read-only
3.0 dump), [[experiments.database.scripts.prebuild_verify_store_array]] (the store
builds); the CSVs come from the real per-adapter array task
(`database/slurm/scripts/gilahyper_generate_adapter_array-slurm_docker.slurm`) pointed at
the private tree, because the host's biocypher (0.5.43) lacks the `_initialize_writer`
the fast CSV sink needs and generation runs only in the `tc-neo4j-gen-<i>` container.

The reader is deliberately a raw LMDB walk (`pickle` plus `resolve_interned`), not a
dataset instance: the shared main tree was written under the pre-unification schema,
whose pickles name enum classes the branch retired (`ReleasedGeneKind`), so a tolerant
unpickler returns a retired enum's value as its plain string. A content key must not
depend on any field the unification rewrote.

Results of the 2026-10-15 run (jobs 4925, 4934, 4961, 4962):
`experiments/database/results/prebuild_verify_stores.json` and
`experiments/database/results/prebuild_verify_csv_endpoint_gate.json`; the tables and
the findings are in [[database.kg-4-pre-build-verification]].
