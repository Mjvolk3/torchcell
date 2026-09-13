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

## 2026.09.13 - Progress and the stale-store finding

Twelve of the fourteen are fixed, rebuilt, verified and registered (Costanzo 2021, Auesukaree, Mota, Baryshnikova, Bloom, Smith 2006, Smith 2016, Lian, Mormino, Vanacloig, Wildenhain, Hoepfner); Hillenmeyer het and hom are built and awaiting their verifier reports. The batch admission dry run against a copy of the production manifest BLOCKS as expected and names every served dataset: 34 through the one added `DoseBasis` member, Nadal-Ribelles also through `EnvironmentPerturbation` gaining the gap mixin, plus the composition-based node ids in the served adapter methods. That is the evidence for the single full rebuild.

The same enum change made ten of the eleven finished dev stores read `stale` (their `build_manifest.json` carries the pre-change fingerprints). The stored values are unchanged, but the verification evidence is regathered on stores whose manifest matches the schema exactly: every finished dataset is rebuilt once more, sequentially, and `run_all` is rerun. Known pre-existing failure outside this campaign: `test_cachera2023.py` asserts 4,735 records and the store holds 4,719.

## 2026.09.13 - Stale-manifest rebuild pass and re-verification

All ELEVEN of the datasets in this pass read `stale`, not ten: `crispr_magic_lian2019`
drifted on `EnvironmentPerturbation` alone, the other ten on `DoseBasis` as well
(`smf_baryshnikova2010` on `DoseBasis` alone). The two Hillenmeyer stores, added to the
pass afterwards, read `stale` on `EnvironmentPerturbation`. Freshness was read with
`check_manifest` against `load_default_surface()` on the worktree's `torchcell/datamodels/`
surface.

Each stale store had `processed/`, `preprocess/experiment_reference_index.json` and
`preprocess/build_manifest.json` moved into `<root>/deprecated-stale-manifest-2026-09-13/`,
keeping `raw/` and the rest of `preprocess/` (`dropped_records.json`, `gene_set.json`,
`block_counts.json`, `sourced_values.json`, `strain_batches.json`, and the Hoepfner
`table_s5_affected_strains.json` beside the root). The reference-index cache is read back
whenever present, so leaving it in place would have failed `post_process` on any count
change. `gene_set.json` is rewritten by the `post_process` setter, so it did not need moving.

Rebuild, then re-verification through the registered `ENVIRONMENT_RESPONSE_DATASETS` /
`FITNESS_DATASETS` / `SEGREGANT_GROWTH_DATASETS` entries with each entry's own parameters
(`stream` where set). Record counts and build wall times:

| store | records | expected | build wall (s) |
|---|---|---|---|
| crispri_mormino2022 | 12 | 12 | 0.8 |
| env_chemgen_auesukaree2009 | 525 | 525 | 0.9 |
| env_chemgen_mota2024 | 1,270 | 1,270 | 1.6 |
| smf_baryshnikova2010 | 5,993 | 5,993 | 6.7 |
| crispri_chemgen_smith2016 | 7,053 | 7,053 | 8.3 |
| env_chemgen_smith2006 | 12,747 | 12,747 | 9.7 |
| env_chemgen_costanzo2021 | 61,430 | 61,430 | 39.8 |
| env_chemgen_wildenhain2015 | 428,206 | 428,206 | 411.9 |
| bloom2019 | 530,100 | 530,100 | 1,414.5 |
| crispr_magic_lian2019 | 266,304 | 266,304 | 146.3 |
| env_chemgen_hoepfner2014 | 3,102,719 | 3,102,719 | 535.5 |
| env_chemgen_hillenmeyer2008_hom | 1,088,620 | 1,088,620 | 489.6 |
| env_chemgen_hillenmeyer2008_het | 2,698,797 | 2,698,797 | 1,015.7 |

Every rebuilt store reads `fresh`, and every record count equals its registered
`expected_count`.

Ten verifiers PASS on the rebuilt stores. Hoepfner 2014 FAILS one level:

```
[XX] L1 canonical_gene_names: 0 genes carry conflicting common-name spellings
(0 records; 0 case-only); 14 systematic names are not the genome's current name;
0 common names resolve to another gene
```

The 14 are YCL074W, YCL075W, YDR134C, YER109C, YFL056C, YIL167W, YIL170W, YIL171W,
YIR043C, YJR026W, YLL016W, YLL017W, YOL153C, YOR031W, each reported as
`non_gene_feature -> <itself>`: the resolver returns the same systematic name it was
given, and the same level's L4 companion `current_genome_genes` accepts all 5,832
measured names as genes of the current genome. The rule is being corrected in
`torchcell/verification/common.py`; the Hillenmeyer verifiers were held rather than run
against the same rule.

Two findings beyond the pass. First, `env_chemgen_hoepfner2014` was rebuilt again by
another process at 12:07 local (LMDB `data.mdb` mtime) after this pass verified it, and
the store now on disk holds 3,124,319 records against the registered 3,102,719, a
difference of 21,600. The PASS/FAIL above belongs to the 3,102,719-record store this pass
built, not to what is on disk now. Second, the verification report is written into
`preprocess/verification_report.json`, so re-running a verifier overwrites the previous
report in place; the pre-rebuild reports were not preserved by the move-aside.

## 2026.09.13 - Hoepfner ORF rule: the defect was the loader, not the L1 rule

Correction to the section above: the L1 `canonical_gene_names` rule is right and was not
changed. The 14 names it failed are `pseudogene`, `blocked_reading_frame` and
`transposable_element_gene` features that resolve to themselves with status
`non_gene_feature`; they are outside `SCerevisiaeGenome.gene_set` and outside every gene
node the graph joins on. The L4 rules accepted them because their gene universe is the R64
ORF + RNA FASTA header set, which lists those features as ORFs; that is a known looseness of
L4 (it is a containment floor, not the current-gene check) and L1 with a resolver is the
strict rule. The Hoepfner loader validated row ORFs against the same FASTA set instead of
the shared resolver, so it kept 14 non-gene rows and dropped 52 renamed merged-ORF strains
that every other chemogenomic loader keeps under the current gene. The loader now follows
the Costanzo 2021 policy (details and the census in
[[torchcell.datasets.scerevisiae.hoepfner2014]]); rebuilt to 3,124,319 records, runner and
table updated. The 3,124,319-record store the rebuild pass saw on disk was this rebuild.

Ontology check to add from this: the shared L4 gene set and the L1 resolver disagree on
non-gene ORF features by construction. A loader that filters on the FASTA set alone will
pass L4 and fail L1; the resolver is the policy. Recorded in
[[torchcell.datamodels.ontology-checks]] as a loader convention rather than a new DAG
check, since it is about gene identity, not the ontology graph.

### Closing coherence run and the final admission dry run

Test suites on the worktree at this point: datamodels, verification, adapters,
knowledge_graphs and literature 841 passed, 3 skipped, 3 xfailed (the `SOTerm` dead-code
xfail among them); datasets 194 passed, 1 failed, the pre-existing
`test_cachera2023.py::test_cachera_build_smoke` count assertion (asserts 4,735, the store
holds 4,719; not touched by this branch, reported to the user).

Batch admission of all fourteen against a COPY of the production manifest
(`scratchpad/serve50/kg_manifest.copy2.json`; the production file was compared byte for
byte afterwards and is untouched): every dev LMDB reads `fresh`; verdict BLOCKED for all
fourteen, on the same three grounds as before: 36 served datasets' schema closures moved
(`EnvironmentPerturbation` gained the gap mixin, `DoseBasis` gained a member), the served
graph class `environment perturbation` changed (the `factor` property), and adapter code
used by served datasets changed (`_environment_node`, 36 served datasets). The value
surface is not recorded in the production manifest, so it does not block. Report:
`experiments/database/results/pre-build/2026-09-13/batch_admit_report_all14.json`. This is
the evidence for the one full rebuild the user chose; no import is run on this branch.

Supported-datasets table regenerated (`build_supported_datasets_table.py --max-gb 3`,
50/50 built; `render_supported_datasets_table.py`): Hoepfner now 10,779 genotypes, 563
environments (the distinct interned conditions over 608 kept columns, in place of the
5,879 deposited sensitivity columns the row used to print), 3,124,319 records.

### Re-verification of the rebuilt Hillenmeyer stores

Both run on the stores the rebuild pass produced, with the shared resolver supplied
(`scratchpad/serve50/builds/env_chemgen_hillenmeyer2008_{hom,het}_verify_final.log`):

| store | records | L1 canonical_gene_names | verdict | wall (s) |
|---|---|---|---|---|
| env_chemgen_hillenmeyer2008_hom | 1,088,620 = 1,088,620 | 4,675 systematic names, each current in the genome | PASS | 2,183.5 |
| env_chemgen_hillenmeyer2008_het | 2,698,797 = 2,698,797 | 5,825 systematic names, each current in the genome | PASS | 5,436.6 |

The Hillenmeyer loaders already drop NON_GENE_FEATURE and RETIRED names through the
resolver, which is why the same rule that failed Hoepfner passes them unchanged.

## 2026.09.13 - Closing state: 50 of 50 built, verified and registered

The Hoepfner verifier on the final store: PASS in 6,451 s, 3,124,319 records, 5,842 genes
current in the genome, 28 merged-ORF aliases reported (verbatim block in
[[torchcell.datasets.scerevisiae.hoepfner2014]]). With the Hillenmeyer pair above that
closes the re-verification: every one of the 14 datasets this branch fixed is L0-L4 PASS
on a store built under the final schema, and `dataset_adapter_map` holds 50.

| state | count |
|---|---|
| loaders registered | 50 |
| built in the dev tree, fresh under the final schema | 50 |
| L0-L4 PASS | 50 |
| served in the production store today | 36 |

What lands: the shared media, compound-identity, node-identity, verifier and ontology-check
layers; the 14 loaders with raw mirrors, adapters, confs, tests and notes; batch
admission; the regenerated supported-datasets table; this note. What does NOT run on this
branch: any import or full rebuild (deferred until more datasets are added, per the
decision section), the Bloom increment, any write to the production manifest.

Open items carried forward, for the next sessions: the Cachera smoke-test count (4,735
asserted, 4,719 stored); a resolver for the expression, morphology, metabolite, protein
and RNA-seq verifiers; a strain-quality flag node class for Table S5-style facts;
`Environment.pre_culture_generations`, `Environment.vehicle`, `CrisprConstruct.delivery`;
Wildenhain wild-type rows; the Teyssonniere 2024 vs Muenzner 2024 de-duplication in the
candidate table.
