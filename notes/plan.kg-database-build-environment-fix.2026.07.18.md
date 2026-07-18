---
id: 1iatsnnut7xl38jhv9kvag7
title: KG Database Build Environment Fix
desc: Fix the GilaHyper KG build-environment config so the subset (and later full) KG build runs end-to-end
updated: 1784344722944
created: 1784344722944
---

# KG Database Build Environment Fix

## 2026.07.18 - Handoff plan (context ported from prior session)

### Where we are (one line)

The subset KG build now runs my code **past every import** on GilaHyper (job 963 reached
`Building KG: module=...create_scerevisiae_kg_small config=kg_small`). All code + the docker
image are DONE and LANDED. The remaining blockers are all **build-environment config** in
`/scratch/projects/torchcell/database`, resolved below using the intentional two-tree design.

### The intentional design (WHY -- do not "simplify" this away)

The KG database is deliberately built for **reproducibility**, accepting a burden now:

- **Data lives on a specific mounted drive, SEPARATED from the code.** The build tree
  (`/scratch/projects/torchcell/database/data/torchcell`) holds its OWN copy of every
  dataset LMDB -- built from scratch (or restored from backup) **inside the container**,
  then run through the adapters to build the graph. The apparent "duplicate" datasets in
  the build tree are **intentional + correct**; they should REPLACE the old ones.
- Rationale: rebuild-from-scratch-or-backup + build-in-container -> the whole graph is
  reproducible from pinned raw data, and it makes schema-change adaptivity + fast test
  builds tractable. GilaHyper builds + verifies it comes up; then it is **ported to
  Radiant** for hosting.
- **Two neo4j confs by machine:** `gh_neo4j.conf` = GilaHyper (build/verify: no public
  SSL, local ports); `neo4j.conf` = Radiant (host: HTTPS `:7473`, bolt advertised
  `torchcell-database.ncsa.illinois.edu:7687`, TLS). `directory_setup` copies the machine's
  conf -> `database/conf/neo4j.conf` in the build tree.

### DONE + LANDED (main, ~`0502020584`)

- Pickle-hang fixes (#119); CABBI metabolism adapters + L0-L4 audit (#121); general
  **subset build** -- `create_scerevisiae_kg_small` iterates the FULL `dataset_adapter_map`,
  genome-injected, capped by `subset.size` (#123); config parameterization
  (`kg_small`=1000 / `kg_full`=null via `--config-name`) + wandb node/edge monitoring (#124).
- **7 build-infra fixes:** `/metrics` mount (#125); Dockerfile py 3.11->**3.13** (#126);
  conda-forge `--override-channels` for the Anaconda ToS gate (#127); torch-scatter
  `--no-build-isolation` (#128); build-time deps from **`torchcell@main`** not stale PyPI
  (#129); apt-get **git** before @main (#130); **source `/.env`** in-container before the KG
  build (#131).
- **Docker image `michaelvolk/tc-neo4j:latest` REBUILT + healthy** on GilaHyper: py 3.13.14,
  hypernetx 2.4.0, torch-scatter, torchcell@main. Built LOCAL only -- **not pushed to Docker
  Hub**; the Radiant port needs `docker login` + `docker push`.

### Remaining blockers (all build-env config; `directory_setup` crashes -> cascade)

1. **`database/conf/gh_neo4j.conf` MISSING** (repo has only `neo4j.conf`, Radiant, 847 lines).
   `torchcell/database/directory_setup.py` copies `WORKSPACE_DIR/database/conf/gh_neo4j.conf
   -> DATA_ROOT/database/conf/neo4j.conf` and **crashes** on the missing source BEFORE it
   copies the `.env` or biocypher.
2. **`database/database.env` MISSING** (nowhere in repo/image). `directory_setup` copies
   `WORKSPACE_DIR/database/database.env -> DATA_ROOT/database/.env` -- the file that gives the
   container `DATA_ROOT` + biocypher paths.
3. **`/scratch/projects/torchcell/database/.env` is a broken root-owned DIRECTORY**
   (auto-made by `-v database/.env:/.env` when the file was absent). Remove with root (docker)
   before a real file can replace it.
4. **Path mismatch (crux).** Host `.env`: `DATA_ROOT=/scratch/projects/torchcell-scratch`
   (DEV tree), `WORKSPACE_DIR=/home/michaelvolk/Documents/projects/torchcell`. So
   `directory_setup` sets up `torchcell-scratch/database/...`. But the slurm script
   (`database/slurm/scripts/gilahyper_first_build-slurm_docker.slurm`) `cd /scratch/projects/
   torchcell` + mounts `$(pwd)/database/...` = `/scratch/projects/torchcell/database`. The
   mounted-drive build tree is `/scratch/projects/torchcell/database`.
5. **Datasets not in build tree.** LMDBs I built/verified are in DEV tree
   (`torchcell-scratch/data/torchcell`). Container reads KG-build tree
   (`/scratch/projects/torchcell/database/data/torchcell`, in-container `/var/lib/neo4j/data/
   torchcell`). Until they exist there, the subset build logs `SKIP ... no LMDB` per dataset.

Container facts (`docker exec tc-neo4j`): `NEO4J_HOME=/var/lib/neo4j`; `/var/lib/neo4j/data ->
/data`; datasets expected at `/var/lib/neo4j/data/torchcell`. => **container-side
`DATA_ROOT=/var/lib/neo4j`** (so `DATA_ROOT/data/torchcell/<ds>` = the mounted build-tree
datasets).

### Fix plan (executable next session; small landed PRs, then resubmit)

- [ ] **A. Create `database/conf/gh_neo4j.conf`** = GilaHyper variant of `neo4j.conf`: disable
      public HTTPS (`dbms.connector.https.enabled=false`), drop/relax the NCSA
      `advertised_address`, keep bolt/http on local `:7687`/`:7474`. Confirm exact
      GilaHyper-vs-Radiant knob diffs.
- [ ] **B. Create `database/database.env`** = container-side env `directory_setup` copies to
      `database/.env`. Keys the build reads (grep `create_scerevisiae_kg_small` main +
      `torchcell.graph.sgd` + BioCypher init to confirm): `DATA_ROOT=/var/lib/neo4j`,
      `WORKSPACE_DIR`, `BIOCYPHER_CONFIG_PATH`, `SCHEMA_CONFIG_PATH`, `BIOCYPHER_OUT_PATH`,
      `NEO4J_USER`, `NEO4J_PASSWORD`, `WANDB_API_KEY`. Biocypher paths derive from mounted
      `/var/lib/neo4j/biocypher`. Do NOT commit secrets -- commit a template + inject
      `WANDB_API_KEY`/`NEO4J_PASSWORD` at deploy (this is the "method of copying in the env
      file" that was lost; re-establish it).
- [ ] **C. Clear the broken `.env` dir:** `docker run --rm -v /scratch/projects/torchcell/
      database:/db alpine rm -rf /db/.env`, then let `directory_setup` (post-B) recreate it as
      a FILE (else the mount re-makes a dir).
- [ ] **D. Reconcile paths (#4):** make `directory_setup` + slurm agree on the build tree
      `/scratch/projects/torchcell` (NOT dev `torchcell-scratch`). Cleanest: export
      `DATA_ROOT=/scratch/projects/torchcell` for the `directory_setup` step in the slurm job
      (the build tree is the mounted drive), or give `directory_setup` a build-root arg.
- [ ] **E. Stage/build datasets into the build tree (the reproducibility step):** per intent,
      datasets are built (or restored from backup) INSIDE the container onto
      `/var/lib/neo4j/data/torchcell` (= `.../database/data/torchcell`). Fast SUBSET bridge
      now: rsync already-built dev-tree LMDBs `torchcell-scratch/data/torchcell/<ds>` ->
      `.../database/data/torchcell/<ds>`, REPLACING old ones. Longer term: a container step
      that rebuilds datasets from pinned raw. Build reads
      `DATA_ROOT/data/torchcell/<ds>/processed/lmdb`.
- [ ] **F. Resubmit + verify:** `sbatch --cpus-per-task=32 --mem=256G --time=12:00:00
      database/slurm/scripts/gilahyper_first_build-slurm_docker.slurm` (node 128 CPU / 513 G;
      the committed `--mem=980G` directive is STALE -- override, and fix in-file). Watch
      `.../database/slurm/output/<job>_build_database.out` for `Building KG:` -> `Instantiating
      dataset` (NOT `SKIP`) -> `Writing nodes for` -> `Finished iterating ... nodes ... edges`.
      Monitor in wandb project `tcdb`.
- [ ] **G. (Later) Port to Radiant:** `docker login` + `docker push michaelvolk/tc-neo4j:
      latest`; bring DB up on Radiant with `neo4j.conf` (HTTPS/TLS/NCSA address).

### Gotchas

- **DATA_ROOT is read at IMPORT time** by `torchcell.graph.sgd` (raises if unset); the
  full-registry import fires it before the build script's `load_dotenv`. #131 sources `/.env`
  in-container, but `/.env` must be a real FILE (blockers B/C).
- Node fits 32 CPU / 256 G for a subset build; do NOT request 64/980.
- The image is LOCAL to GilaHyper (docker uses local `:latest`); no push needed for the
  GilaHyper build itself. Re-pulling main / rebuilding the image is NOT needed unless the
  Dockerfile changes again.
- Related: [[simb2026-interning-db-rebuild]], [[abstract-biocypher-adapters]], memory
  `kg-adapter-pickle-hang-fixes`, memory `schema-impact-worktree-crash`.

## 2026.07.18 - RESULT: subset KG database BUILT + SERVING

The plan above executed. **A queryable neo4j knowledge-graph DB was built end-to-end**
(job 965 on GilaHyper): served by `tc-neo4j-readonly` (healthy).

- **DB contents:** 77,488 nodes / 179,446 edges, from **31 adapters** (216,525 nodes
  written -> dedup 77,488, since datasets share gene/genotype/environment nodes). Node
  types: Genotype 31,554 · Experiment 22,362 · PhenotypicFeature 20,841 · Perturbation
  14,145 · FitnessPhenotype 6,203 · GeneInteractionPhenotype 5,002 · MetabolitePhenotype
  3,761 · CalmorphPhenotype 3,003 · MicroarrayExpressionPhenotype 2,792 · ExperimentReference
  1,789. Multi-modal (fitness / interaction / metabolite / morphology / expression).
- Query it: `docker exec tc-neo4j-readonly bash -c 'source /.env && cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -d torchcell "MATCH (n) RETURN count(n);"'`

### Two fixes LANDED to main

- **PR #133 `8a006f1e`** build-env (this plan's A-D): `database/conf/gh_neo4j.conf` (the
  missing file that crashed `directory_setup`), `database/database.env.template`, slurm
  `DATA_ROOT=/scratch/projects/torchcell` override for `directory_setup` + `--mem 980G->256G`
  / `--cpus 64->32`.
- **PR #134 `c913291f`** `ExperimentDataset._download` returns early when `processed/lmdb`
  exists. **The big unblock** beyond this plan: PyG's `Dataset.__init__` runs `_download()`
  before `_process()` and re-fetches raw whenever `raw_file_names` are incomplete -- regardless
  of the built LMDB -- 403ing (`current.geneontology.org`, `thecellmap.org`) or colliding on
  `shutil.move`. The build read only the LMDBs, so this was pure friction.

### Runtime setup performed (host, not committed)

- Cleared stale `7474`/root build-tree subdirs via a root alpine container, then
  `directory_setup` (DATA_ROOT=/scratch/projects/torchcell) rebuilt them as michaelvolk.
- **Hardlink-staged** `cp -al torchcell-scratch/data/{torchcell,sgd,string,tflink,go} ->
  .../torchcell/database/data/` (3.2 TB, same fsid = instant; a real copy is impossible --
  1.5 TB free). Placed `go.obo` at `data/go/go.obo`. Real `database/database.env` (gitignored)
  carries the WANDB key.

### 3 datasets flagged -- USER FOLLOW-UP

1. **GeneEssentialitySgd** -- EXCLUDED. Schema drift: stored `GeneEssentialityPhenotype(is_essential=...)`
   fails `ExperimentReferenceIndex` validation against `EnvironmentResponseExperimentReference`
   (`ModelStrict`, forbids `is_essential`). Fix: rebuild the LMDB against `@main` schema, or
   reconcile the schema (schema.py -> user-territory, schema-impact gate).
2. **CaudalPanTranscriptome2024** -- EXCLUDED. Subset-perf blowup: ~1 h+ and never finished;
   its `perturbation->genotype` edges explode (each isolate ~12.5k perturbations off S288C, so
   subset=1000 isolates -> ~12.5M edges). Fix: a much smaller Caudal-specific subset, or drop it
   from `kg_small` (the config comment promises "runs in minutes").
3. **ProteomeZelezniak2018** -- PARTIAL. BioCypher dropped the `protein abundance phenotype`
   nodes: inconsistent property sets (some carry `protein_abundance_se`, some don't) and
   BioCypher requires a uniform per-class property set. Fix: the adapter should always emit
   `protein_abundance_se` (null when absent).

Excluded LMDBs were moved to `.../database/data/torchcell/<ds>/processed/lmdb.excluded_20260718`
(build tree only; the dev tree is untouched). Restore them after the fixes to re-include.

### Code follow-ups (documented, NOT auto-landed -- for review)

- **`SCerevisiaeGenome` `go.obo` provenance.** `__attrs_post_init__` unconditionally
  `download_url("http://current.geneontology.org/ontology/go.obo", go_root)` which **403s**
  (bare-urllib UA on the http->https redirect; curl works). Staged manually for now; per the
  provenance principle this should be a sha256-pinned mirror, not a live download.
- **`directory_setup` not idempotent.** The neo4j entrypoint `chown -R neo4j /data` (+ conf)
  flips bind-mount ownership to uid 7474, so a 2nd michaelvolk `directory_setup` PermissionErrors
  on `conf/neo4j.conf` (non-fatal this run -- existing valid config was reused). Fix: remove-then-copy
  in `directory_setup`. The same chown flipped **dev-tree** hardlink inodes to 7474; chowned back
  to 1000 this session.
- **kg_small Caudal cost** -- see flag #2; the subset should cap per-record genotype size, not just
  record count, or exclude Caudal from the fast config.
