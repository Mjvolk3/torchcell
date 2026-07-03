---
id: 7mpok5jqher3grhi5ggbtjl
title: Supported Datasets Table Revitalized
desc: ''
updated: 1781054069727
created: 1781054069727
---

## 2026.06.09 - Revitalized Supported-Datasets Table

### Why this note exists

The prior dataset tables ([[torchcell.datasets.cell.supported_data_table]] and
`notes/scratch.2025.06.30.170146-supported_datasets.md`) marked every row with a
single checkmark. That single mark is misleading: it conflated two independent
facts, and several rows are checked that are not actually wired into the
knowledge-graph build. This note re-derives status by reading the code directly,
adds the new ingest targets that are not in the old table at all, and reframes
the whole list around the current goals.

The goals driving the rebuild:

- **Zotero is the ground-truth index.** A group library
  (<https://www.zotero.org/groups/6582362/torchcell>, group id `6582362`,
  auth = API key + group id) holds every paper whose data we support, plus
  reference-only papers we want to reason over from inside this repo.
- **Full artifact capture as backup/mirror.** For each data-backed paper we want
  the entire documentation artifact: original PDF + MinerU-OCR'd markdown + SI
  PDF + SI OCR + raw SI data files. This artifact is a *backup/mirror* to the
  existing URL-based `download()` methods, not a replacement. The current
  publisher/SGD/Box/GEO URLs stay primary; the captured artifact is the fallback
  when an upstream source goes dark.
- **DOI is the join key.** Zotero item to on-disk artifact to Pydantic dataset
  object, all joined on the unique DOI.
- **Two collections in the group library.** `torchcell-database` = papers whose
  data we actually ingest (this note's Bucket A + B). Reference-only papers live
  outside that collection but in the same group library.

### Status legend (two independent dimensions)

A dataset has two orthogonal states, and the old single checkmark hid the
difference:

- **Class** - a Python dataset class exists under
  `torchcell/datasets/scerevisiae/` and can build deep-learning-ready data.
- **KG adapter** - the class is registered in
  `torchcell/knowledge_graphs/dataset_adapter_map.py`, i.e. it actually flows
  into the Neo4j knowledge-graph build.

These are independent: a class can serve ML dataloaders without ever being
ingested into the KG. The rebuild needs both columns green for a dataset to be
"in the database."

`Zotero` column = is the paper already in the `torchcell-database` group
collection? All `no` for now; this note is the worklist to flip them to `yes`.

### Unified supported-datasets table

One table, all datasets. `Bucket` column preserves the rebuild-vs-new-ingest
distinction: **A** = already has a Python class (rebuild the DB from these);
**B** = new ingest target with no class yet. Status verified 2026.06.09 against
`torchcell/datasets/scerevisiae/*.py` and
`torchcell/knowledge_graphs/dataset_adapter_map.py`. DOIs and URLs are quoted
verbatim from the code; blank-in-source fields are marked `(none in code)`
rather than guessed; not-yet-selected Bucket B fields are marked `(to select)`.

| Dataset                 | Bucket | Class file                               |                Class                 |   KG adapter   | DOI                                   | Source URL (primary)                                                                                                            | Phenotype                                        | Zotero |
|:------------------------|:------:|:-----------------------------------------|:------------------------------------:|:--------------:|:--------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------|:------:|
| Costanzo_2016_smf       |   A    | costanzo2016.py                          |                 yes                  |      yes       | 10.1126/science.aaf1420               | `https://thecellmap.org/costanzo2016/data_files/Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip` | single-mutant fitness                            |   no   |
| Costanzo_2016_dmf       |   A    | costanzo2016.py                          |                 yes                  |      yes       | 10.1126/science.aaf1420               | (same thecellmap.org zip)                                                                                                       | double-mutant fitness                            |   no   |
| Costanzo_2016_dmi       |   A    | costanzo2016.py                          |                 yes                  |      yes       | 10.1126/science.aaf1420               | (same thecellmap.org zip)                                                                                                       | digenic interaction (epsilon)                    |   no   |
| Kuzmin_2018_smf         |   A    | kuzmin2018.py                            |                 yes                  |      yes       | 10.1126/science.aao1729               | `https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip`                             | single-mutant fitness                            |   no   |
| Kuzmin_2018_dmf         |   A    | kuzmin2018.py                            |                 yes                  |      yes       | 10.1126/science.aao1729               | (same aao1729 zip)                                                                                                              | double-mutant fitness                            |   no   |
| Kuzmin_2018_tmf         |   A    | kuzmin2018.py                            |                 yes                  |      yes       | 10.1126/science.aao1729               | (same aao1729 zip)                                                                                                              | triple-mutant fitness                            |   no   |
| Kuzmin_2018_dmi         |   A    | kuzmin2018.py                            |                 yes                  |      yes       | 10.1126/science.aao1729               | (same aao1729 zip)                                                                                                              | digenic interaction                              |   no   |
| Kuzmin_2018_tmi         |   A    | kuzmin2018.py                            |                 yes                  |      yes       | 10.1126/science.aao1729               | (same aao1729 zip)                                                                                                              | trigenic interaction (hyperedge)                 |   no   |
| Kuzmin_2020_smf         |   A    | kuzmin2020.py                            |                 yes                  |     **no**     | 10.1126/science.aaz5667               | `https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip`                                                       | single-mutant fitness                            |   no   |
| Kuzmin_2020_dmf         |   A    | kuzmin2020.py                            |                 yes                  |     **no**     | 10.1126/science.aaz5667               | (same uofi.box zip)                                                                                                             | double-mutant fitness                            |   no   |
| Kuzmin_2020_tmf         |   A    | kuzmin2020.py                            |                 yes                  |     **no**     | 10.1126/science.aaz5667               | (same uofi.box zip)                                                                                                             | triple-mutant fitness                            |   no   |
| Kuzmin_2020_dmi         |   A    | kuzmin2020.py                            |                 yes                  |     **no**     | 10.1126/science.aaz5667               | (same uofi.box zip)                                                                                                             | digenic interaction                              |   no   |
| Kuzmin_2020_tmi         |   A    | kuzmin2020.py                            |                 yes                  |     **no**     | 10.1126/science.aaz5667               | (same uofi.box zip)                                                                                                             | trigenic interaction (hyperedge)                 |   no   |
| SGD_essential           |   A    | sgd.py                                   |                 yes                  |      yes       | (dynamic, via PubMed API)             | dynamic (PubMed `get_publication_info()`)                                                                                       | gene essentiality                                |   no   |
| SynthLethalYeast        |   A    | synth_leth_db.py                         |                 yes                  |      yes       | (none in code)                        | `https://drive.google.com/uc?export=download&id=1_56ebyBatapNml8S5HlJW7Dz1l0DZZIq`                                              | synthetic lethality                              |   no   |
| SynthRescueYeast        |   A    | synth_leth_db.py                         |                 yes                  |      yes       | (none in code)                        | (same Google Drive id, shared)                                                                                                  | synthetic rescue                                 |   no   |
| scmd2_2005 (Ohya)       |   A    | ohya2005.py                              |                 yes                  |      yes       | (none in code)                        | `http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/download.php?path=mt4718data.tsv` (+ `wt122data.tsv`); Box fallbacks                  | cell morphology (CalMorph)                       |   no   |
| Baryshnikova_2010       |   A    | baryshnikovna2010.py                     | yes (commented out in `__init__.py`) |     **no**     | 10.1038/nmeth.1534 (from URL)         | `https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.1534/MediaObjects/41592_2010_BFnmeth1534_MOESM168_ESM.xls`       | single-mutant fitness                            |   no   |
| Kemmeren_2014           |   A    | kemmeren2014.py                          |                 yes                  |     **no**     | (none in code)                        | GEO: `GSE42527`, `GSE42526`, `GSE42241`, `GSE42240`, `GSE42217`, `GSE42215`                                                     | single-KO microarray expression                  |   no   |
| Sameith_2015 (sm)       |   A    | sameith2015.py                           |                 yes                  |     **no**     | 10.1186/s12915-015-0222-5 (docstring) | GEO: `GSE42536`                                                                                                                 | single-KO microarray expression                  |   no   |
| Sameith_2015 (dm)       |   A    | sameith2015.py                           |                 yes                  |     **no**     | (none in code)                        | GEO (same study)                                                                                                                | double-KO microarray expression                  |   no   |
| ODuibhir_2014           |   A    | oduibhir2014.py `[UNVERIFIED]`           |            `[UNVERIFIED]`            | `[UNVERIFIED]` | (to confirm)                          | (to confirm)                                                                                                                    | growth rate / expression                         |   no   |
| Lian_2017               |   A    | (to confirm) `[UNVERIFIED]`              |            `[UNVERIFIED]`            | `[UNVERIFIED]` | (to confirm)                          | (to confirm)                                                                                                                    | AID furfural tolerance                           |   no   |
| Wildenhain_2015         |   A    | (to confirm) `[UNVERIFIED]`              |            `[UNVERIFIED]`            | `[UNVERIFIED]` | (to confirm)                          | (to confirm)                                                                                                                    | drug tolerance fitness                           |   no   |
| FitDb                   |   A    | (to confirm) `[UNVERIFIED]`              |            `[UNVERIFIED]`            | `[UNVERIFIED]` | (to confirm)                          | (to confirm)                                                                                                                    | multi-environment fitness                        |   no   |
| scmd2_2018              |   A    | (no distinct class found) `[UNVERIFIED]` |            `[UNVERIFIED]`            | `[UNVERIFIED]` | (to confirm)                          | (to confirm)                                                                                                                    | cell morphology                                  |   no   |
| scmd2_2022              |   A    | (no distinct class found) `[UNVERIFIED]` |            `[UNVERIFIED]`            | `[UNVERIFIED]` | (to confirm)                          | (to confirm)                                                                                                                    | cell morphology                                  |   no   |
| Zelezniak_2018          |   B    | none                                     |                  no                  |       no       | (to confirm)                          | (to select)                                                                                                                     | protein abundance + metabolite conc. (kinase KO) |   no   |
| Natural-isolate phenome |   B    | none                                     |                  no                  |       no       | (to select)                           | (to select)                                                                                                                     | phenotypes across wild isolates                  |   no   |
| Pan-transcriptome       |   B    | none                                     |                  no                  |       no       | (to select)                           | (to select)                                                                                                                     | transcriptomes across natural isolates           |   no   |
| CRi-SPA                 |   B    | none                                     |                  no                  |       no       | (to select)                           | (to select)                                                                                                                     | CRISPR-Cas9 screen / strain-array phenotypes     |   no   |
| beta-carotene           |   B    | none                                     |                  no                  |       no       | (to select)                           | (to select)                                                                                                                     | production phenotypes (DBTL case study)          |   no   |

Notes on the table:

- **Kuzmin_2020 (all 5 variants): classes exist but are NOT in the adapter map.**
  The old table marked these checked. They build for ML but do not currently flow
  into the KG. Wiring these into `dataset_adapter_map.py` is rebuild work.
- **Kemmeren_2014, Sameith_2015, Baryshnikova_2010: same gap** - class present,
  no KG adapter. The expression datasets (Kemmeren/Sameith) are exactly the
  "expression from WT/KO" data the rebuild wants in the graph, so closing this
  adapter gap is on the critical path.
- **Baryshnikova_2010 is commented out** in
  `torchcell/datasets/scerevisiae/__init__.py` - effectively dormant.
- **DOIs marked `(none in code)`** (SynLethDB, Ohya/SCMD, dm-Sameith) need a
  lookup before Zotero entry - do not invent them. The paired Dendron notes did
  not contain them either.
- **Kuzmin_2018 source URL is self-hosted** in this very repo
  (`raw.githubusercontent.com/Mjvolk3/torchcell/.../aao1729_data_s1.zip`), and
  Kuzmin_2020 is on UofI Box - these are already de-facto mirrors of the Science
  SI, a useful precedent for the backup/mirror artifact model.
- **`[UNVERIFIED]` Bucket A rows** (ODuibhir_2014, Lian_2017, Wildenhain_2015,
  FitDb, scmd2_2018, scmd2_2022) were in the old table but not confirmed in this
  pass - class/adapter/DOI/URL still need reading from the code. They are kept in
  Bucket A on the assumption they have (or are intended to have) classes; demote
  to Bucket B any that turn out to be unimplemented.
- **Zelezniak is confirmed not implemented** - it lives only in planning notes,
  no class and no adapter, hence Bucket B despite being an "old" paper.
- **Natural-isolate phenome and pan-transcriptome introduce wild genomes**, which
  the current `genome` node (species + strain only) cannot represent. These two
  targets are coupled to a genome-assembly / sequence-store extension of the KG,
  not just a new dataset writer.
- **beta-carotene** currently appears only as an external case study reference
  (iBioFoundry-AI); ingesting it as a TorchCell dataset is new work.

### Verified corrections vs. the old table

- Single checkmark replaced by `Class` + `KG adapter` columns.
- Kuzmin_2020 (x5), Kemmeren_2014, Sameith_2015, Baryshnikova_2010: downgraded
  from checked to **class-yes / adapter-no**.
- Zelezniak_2018: moved from checked to Bucket B (not implemented).
- Lian_2017, Wildenhain_2015, FitDb, ODuibhir_2014: present in the old table but
  not re-verified here (not in the adapter map per the last check); carry forward
  as `[UNVERIFIED]` until their classes are confirmed.
- scmd2_2018 / scmd2_2022: flagged `[UNVERIFIED]` (no distinct class found).

### Zotero capture workflow (target, backup/mirror role)

For each Bucket A + B paper, the artifact mirrors the existing download URL:

```text
$DATA_ROOT/zotero/6582362/<citation_key>/
  paper.pdf                 # OG (publisher / Zotero attachment)
  paper.md                  # MinerU OCR
  si/
    si1.pdf  si1.md         # SI PDF + OCR
    si_data/                # raw SI data files (xls/csv/zip) - the new primitive
  manifest.json             # DOI, sha256 per file, source URLs, collection tags
```

`manifest.json` records the primary `download()` URL so a dataset writer can fall
back to `si_data/` keyed by DOI when an upstream source is down. This keeps the
existing URL-based writers primary and the artifact strictly as backup.

### Next steps

1. Resolve the `(none in code)` DOIs (SynLethDB, Ohya/SCMD, dm-Sameith, SGD's
   dynamic DOI) so every Bucket A row has a DOI for Zotero entry.
2. Confirm `[UNVERIFIED]` rows (Lian_2017, Wildenhain_2015, FitDb, ODuibhir_2014,
   scmd2_2018/2022) by reading their classes / notes.
3. Decide the citation-key scheme for the group library (iBioFoundry-AI uses
   deterministic `authorTitleWordYYYY`).
4. Seed the `torchcell-database` collection with Bucket A papers (DOIs known),
   then flip the Zotero column to `yes` as each lands.
5. Close the KG-adapter gaps for Kuzmin_2020 and the expression datasets as part
   of the rebuild.

## 2026.06.14 - Library Reality and Naming Convention

### Status: the data-backed set is now fully in the library

The Zotero group library (group `6582362`) now holds **24 papers**, and **every
data-backed dataset in the table above has a DOI-keyed paper in the `database`
collection**. So the `Zotero` column in the 2026.06.09 table reads `yes` across
the board; the worklist of seeding the collection is done.

Two corrections to the 2026.06.09 assumptions:

- The collection is named **`database`** (key `WTDHGDPE`), not
  `torchcell-database`.
- Every paper carries a DOI (the join key is intact library-wide); the
  `(none in code)` DOIs were resolved from the Zotero metadata, not the
  `download()` source.

### Database naming convention

Two distinct naming systems -- do not conflate them:

- **Zotero citation key** (Better BibTeX, auto-generated):
  `authorTitleWordsYYYY`, e.g. `costanzoGlobalGeneticInteraction2016`. Zotero
  stores this; `_resolve_citation_key` reads it; it names the on-disk **artifact
  directory** `torchcell-library/<citation_key>/`.
- **TorchCell dataset name** (our convention): `<firstauthorlastname>_<year>`,
  lowercased. When one paper / module file yields **multiple** datasets, append a
  differentiator that matches the distinguishing dataset class:
  - Costanzo / Kuzmin (fitness + interaction): differentiate by **phenotype** --
    `smf`, `dmf`, `tmf`, `dmi`, `tmi` (e.g. `costanzo_2016_dmi`,
    `kuzmin_2018_tmi`).
  - Sameith (expression): `sm` / `dm` for single- vs double-mutant
    (`sameith_2015_sm`, `sameith_2015_dm`).
  - Zelezniak (two modalities): `protein` / `metabolite`.
  - Single-dataset papers use the bare `<lastname>_<year>` (e.g.
    `baryshnikova_2010`).

### Dataset -> paper -> DOI mapping (`database` collection)

| Dataset name (convention)         | Differentiator   | Zotero citation key                            | DOI                          | Bucket | Legacy table name |
|:----------------------------------|:-----------------|:-----------------------------------------------|:-----------------------------|:------:|:------------------|
| baryshnikova_2010                 | smf              | baryshnikovaQuantitativeAnalysisFitness2010    | 10.1038/nmeth.1534           |   A    | Baryshnikova_2010 |
| costanzo_2016_smf/_dmf/_dmi       | phenotype        | costanzoGlobalGeneticInteraction2016           | 10.1126/science.aaf1420      |   A    | Costanzo_2016_*   |
| kuzmin_2018_smf/_dmf/_tmf/_dmi/_tmi | phenotype      | kuzminSystematicAnalysisComplex2018            | 10.1126/science.aao1729      |   A    | Kuzmin_2018_*     |
| kuzmin_2020_smf/_dmf/_tmf/_dmi/_tmi | phenotype      | kuzminExploringWholegenomeDuplicate2020        | 10.1126/science.aaz5667      |   A    | Kuzmin_2020_*     |
| kemmeren_2014                     | expr (sm)        | kemmerenLargeScaleGeneticPerturbations2014     | 10.1016/j.cell.2014.02.054   |   A    | Kemmeren_2014     |
| sameith_2015_sm/_dm               | sm / dm          | sameithHighresolutionGeneExpression2015        | 10.1186/s12915-015-0222-5    |   A    | Sameith_2015      |
| ohya_2005                         | morphology       | ohyaHighdimensionalLargescalePhenotyping2005   | 10.1073/pnas.0509436102      |   A    | scmd2_2005        |
| ohnuki_2018                       | morphology       | ohnukiHighdimensionalSinglecellPhenotyping2018 | 10.1371/journal.pbio.2005130 |   A    | scmd2_2018        |
| ohnuki_2022                       | morphology       | ohnukiHighthroughputPlatformYeast2022          | 10.1038/s41540-022-00212-1   |   A    | scmd2_2022        |
| oduibhir_2014                     | smf / expr       | oduibhirCellCyclePopulation2014                | 10.15252/msb.20145172        |   A    | ODuibhir_2014     |
| wang_2022_synthetic_lethality/_rescue | class        | wangSynLethDB20Webbased2022                    | 10.1093/database/baac030     |   A    | SynthLethal/Rescue |
| engel_2022                        | essentiality     | engelNewDataCollaborations2022                 | 10.1093/genetics/iyab224     |   A    | SGD_essential     |
| wildenhain_2015                   | smf drug tol.    | wildenhainPredictionSynergismChemicalGenetic2015 | 10.1016/j.cels.2015.12.003 |   A    | Wildenhain_2015   |
| lian_2017                         | AID              | lianCombinatorialMetabolicEngineering2017      | 10.1038/s41467-017-01695-x   |   A    | Lian_2017         |
| hillenmeyer_2008                  | chem-genetic fit | hillenmeyerChemicalGenomicPortrait2008         | 10.1126/science.1150021      |   A    | FitDb             |
| zelezniak_2018_protein/_metabolite | modality        | zelezniakMachineLearningPredicts2018           | 10.1016/j.cels.2018.08.001   |   B    | Zelezniak_2018    |
| caudal_2024                       | pantranscriptome | caudalPantranscriptomeRevealsLarge2024         | 10.1038/s41588-024-01769-9   |   B    | Pan-transcriptome |
| cachera_2023                      | CRISPR screen    | cacheraCRISPAHighthroughputMethod2023          | 10.1093/nar/gkad656          |   B    | CRi-SPA           |
| zaydun_2013                       | beta-carotene    | zaydnCarotenoidBasedPhenotypicScreenYeast2013  | 10.1016/j.ymben.2012.07.010  |   B    | beta-carotene     |
| khaiwal_2025                      | natural-isolate phenome | khaiwalPredictingNaturalVariation2025   | 10.1038/s44320-025-00136-y   |   B    | Natural-isolate phenome |

Legacy-name notes: `ohya_2005`/`ohnuki_2018`/`ohnuki_2022` are the convention-
correct names for the morphology rows the old table called `scmd2_2005/2018/2022`
(SCMD is the source database, not the first author); `engel_2022` is the SGD
reference paper standing in for `SGD_essential` (essentiality is sourced from SGD,
not a single experiment); `hillenmeyer_2008` is the source of `FitDb` (confirmed
via the FitDB site: "based on data in Hillenmeyer et al. (Science 2008)").

### Reference-only papers (in the library, NOT in `database`)

Kept for comparison / writing context, explorable from Claude Code, with no
dataset contract:

| Dataset name | Zotero citation key                     | DOI                       | Why reference-only        |
|:-------------|:----------------------------------------|:--------------------------|:--------------------------|
| turco_2023   | turcoGlobalAnalysisYeast2023            | 10.1126/sciadv.adg5702    | knockout phenome -- comparison |
| elsemman_2022 | elsemmanWholecellModelingYeast2022     | 10.1038/s41467-022-28467-6 | whole-cell model -- comparison |

### New in library, in `database`, but not yet in the table (to classify)

These were added to `database` (so intended as data-backed) but do not map to an
existing table row -- decide their dataset name + phenotype and add rows:

| Candidate name | Zotero citation key                     | DOI                       | Content                         |
|:---------------|:----------------------------------------|:--------------------------|:--------------------------------|
| vaishnav_2022  | vaishnavEvolutionEvolvabilityEngineering2022 | 10.1038/s41586-022-04506-6 | gene-regulatory DNA -> expression |
| zhang_2020     | zhangCombiningMechanisticMachine2020    | 10.1038/s41467-020-17910-1 | tryptophan metabolic engineering |

### Open items

1. Confirm the morphology naming: adopt convention names `ohya_2005` /
   `ohnuki_2018` / `ohnuki_2022`, or keep the `scmd2_*` database names?
2. Classify `vaishnav_2022` and `zhang_2020` (new datasets vs reference-only).
3. Per-paper **SI-data source URLs** for capture -- the one thing not in Zotero.
   Known so far: Kuzmin 2018 = Boone lab supplement
   (`boonelab.ccbr.utoronto.ca/supplement/kuzmin2017`); DRYAD `10.5061/dryad.tt367`
   exists but is gated (401). Record each paper's working SI-data URL here as we
   capture.
4. Run `capture_by_doi` over the `database` collection to populate
   `torchcell-library/<citation_key>/` for all data-backed papers.
