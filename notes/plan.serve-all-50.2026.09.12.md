---
id: 9g5e2jpx7n9hea1tvf0us7h
title: Serve All 50
desc: ''
updated: 1789256176721
created: 1789256176721
---

## Context

Fifty datasets are built; 36 are served. Bloom 2019 is admissible and the other 13
(Baryshnikova 2010 and the environmental and chemogenomic block: Auesukaree 2009, Mota
2024, Vanacloig-Pedros 2022, Costanzo 2021, Hillenmeyer 2008 het and hom, Wildenhain 2015,
Hoepfner 2014, Smith 2006, Smith 2016, Lian 2019, Mormino 2022) have no adapter. The goal
is all 50 served correctly under the representation principles in
[[torchcell.knowledge_graphs.dataset-admission-loop]]: persistent entities identifiable
across datasets (genome and systematic names, compound structure identifiers, shared
`Media` objects, `Temperature`), contingent observations typed honestly, and a medium-level
join that lets aggregates form further up the environment branch. A compound without a
structure identifier is not served; the records that depend on it are dropped.

Seven read-only reviews (one per dataset group, plus one over the ontology) ran on
2026-09-12; the reports are the evidence for everything below.

## Review findings (2026-09-12)

Verdicts: no dataset is serve-as-is. Fix-then-serve: Baryshnikova, Hoepfner, Hillenmeyer
het and hom, Auesukaree, Mota, Smith 2006, Lian, Mormino, Bloom (identifiers only).
Drop-records-then-serve: Costanzo 2021, Vanacloig, Wildenhain, Smith 2016. Every unserved dev
LMDB fails L0 at 100 percent (built before `Media.is_synthetic` became required) and stores
compounds with null identifiers, so all are rebuilt.

Full-rebuild trigger from the classes: none. `kg_manifest admit` measured zero served
closure drift and zero served adapter drift for every class; the only blocks were "not in
`dataset_adapter_map`" and "unmanifested dev LMDB".

Errors found (file references in the reports):

- Hoepfner passes the experiment-tagged name (`Amitriptyline [3_50_HIP_0077]`) as the
  resolver key, so 0 of 3,112,880 records carry an InChIKey while the verifier reports
  PASS; the 2026-07-15 report on disk records L1 count `passed: false`.
- Lian 2019's medium is SED-URA/G418, not SED/G418; 62,793 CRISPRd records store the 44 nt
  donor fragment in `guide_sequence`.
- Mormino 2022's medium is SC, not SD; the 2 ug/mL ATc inducer is missing; the comparator
  is the CBL pool, not CC23.
- Auesukaree keys PPA1 to YBR011C (essential; the paper's vacuolar PPA1 is YHR026W); FEN1
  was first-matched arbitrarily.
- Mota keys 7 records to retired ORFs because the resolver accepts any systematic-looking
  token unchecked; the L4 floor of 0.90 let 0.993 through.
- Vanacloig serves the DMSO vehicle column as an inhibitor at IC30 and two QUADRIS
  commercial suspensions at arbitrary doses; 5,202 all-zero cells ship a pseudocount with
  `SE = 0.0`; its clean gap census is a false pass because unsourced values were left as
  silent Nones.
- Wildenhain carries CID and SMILES for 484,425 of 484,830 rows but only 1,029 records have
  an InChIKey because the identity table holds 33 rows.
- Baryshnikova has zero `SourcedValue`s, an unsourced 30 C where the sourced SGA rule
  puts TS-allele selection at 26 C, and `n_samples = 80` extrapolated from the array side
  onto 1,387 query-side records; it is not registered in `__init__.py`.
- Smith 2016 is not registered either; 7,410 of 14,463 records ride on vendor codes with
  no structure; 5,144 more resolve with one synonym (`nsc-180973` is tamoxifen).
- Hillenmeyer's Wayback-recovered FitDb matrices exist only in the dev build tree, with no
  raw mirror and no backup, and FitDb itself is DNS-dead.

Record drops under the molecule-identifier rule: Hoepfner 10,161 (boromycin, SMILES does
not parse) keeping 3,102,719; Hillenmeyer het 307,098 keeping 2,613,980 and hom 130,489
keeping 1,049,031 (conditional on the compound table growing by about 302 names);
Vanacloig about 16,000 keeping about 148,000 (conditional on a 45-token alias map);
Wildenhain 367 keeping 428,206; Costanzo 2021 4,390 (tunicamycin, a mixture) keeping
57,162; Smith 2016 7,410 keeping 7,053; Auesukaree, Mota, Smith 2006, Lian, Mormino,
Baryshnikova none.

The ontology coherence pass ([[torchcell.datamodels.ontology-checks]]) found four
defects: 44 of the 55 compounds in the shared media library carry no identifier; three
`base_medium` strings (`SD`, `SD_MSG`) name no library member; the adapter dereferences
the optional `Environment.temperature` unguarded; `SOTerm` is dead.

## The join problem and the rebuild decision

Media node ids are the sha256 of the whole `Media` dump. The served store has 10 media
nodes; only the two SGA selection media carry typed components. Five served datasets
(Nadal-Ribelles, Ohya 2005, Ohnuki 2018 and 2022, da Silveira 2014) share one name-only
liquid YPD node; SC, SC-URA, SM and YEPD are name-only too. A typed YPD from the shared
library hashes to a second node, so old and new datasets join on the medium name, not on
node identity. Enriching the shared library with identifiers also moves the SGA media ids
that served Costanzo and Kuzmin hang off, and the admission gate fingerprints classes,
not values, so it cannot see that.

Two honest paths: (1) incremental admission of the 14 with typed media, accepting two
YPD nodes until a later full rebuild; (2) one full rebuild of all 50 with typed media in
every loader (the last full build, job 1558, ran 21 h 22 min). Recommendation: path 2.
All work below is identical under both paths; only the final import differs. The user
decides.

## Approach

1. Shared layer, in parallel: grow the compound identity registry (about 5,600 names and
   CIDs, PubChem-resolved with provenance, plus an RDKit SMILES route with curated rows
   winning); additive schema and adapter changes (`EnvironmentResponsePhenotype.screen_id`,
   a typed `ResponseCategory` with the verbatim label kept, a `BarcodedKanMxDeletionPerturbation`
   leaf, a `crispr construct` node class and edge, projected `category` on the environment
   response node, guarded temperature dereferences); shared verifier rules (compound identity
   gate, a gap census that reaches compounds and reports silent Nones, media library
   membership, canonical gene names, SE sanity, categorical L3, retired-ORF L4) and a value
   surface (media library and compound table hashes) in the manifest gate.
2. Typed media library once the compound table lands: identifiers on every library
   compound, `base_medium` keys resolved, liquid YPD and YP glycerol, SynBase and SynH3
   minus, SC with its carbon source, YPBO/YPBM/YPBA, SED-URA with and without G418, YPD
   agar, dropout helper.
3. Per-dataset fixes fanned out (loader, raw mirror deposit with `manifest.json`, adapter
   and conf, tests, note, dev rebuild, L0-L4), editing only their own files.
4. Registration, coherence re-check (ontology tests, batch `admit`), then the import path
   the user chose.

Batch admission (`admit --dataset A,B,C`, `DATASET_CLASSES` on the runner) landed in this
worktree first so the final step is one job either way.

## 2026.09.12 - Decision: full rebuild, deferred

The user chose the full rebuild, deferred: this branch lands with NO import, more datasets are added on top of it, and one full build then serves everything at once. Consequences: the Bloom increment is not run; the 14 fixed datasets are registered and verified on the dev tree only; the batch admission path stays for later increments once the rebuilt store carries the typed media and identified compounds; the shared media library and compound table changes (which move every medium node id) are exactly what the rebuild absorbs. The next wave is scouted from the north-star candidate table (`experiments/database/scripts/build_candidate_datasets_table.py`), Albert 2018 and Jackson 2020 first.
