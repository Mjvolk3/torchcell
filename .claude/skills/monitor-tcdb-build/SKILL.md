---
name: monitor-tcdb-build
description: Introspect a running or finished tcdb Neo4j/BioCypher KG build on GilaHyper (slurm .out + wandb + the served DB), report a concise status, and diagnose + restart on stall or failure. Use when checking on a `build_database` slurm job, polling a KG build, or driving one to a queryable DB.
---

# Monitor tcdb Build

Drive a tcdb KG database build to a verified, queryable Neo4j DB: introspect it live,
report status, and on stall/failure diagnose the cause and restart. Two structured
channels (slurm `.out` + wandb) plus the served DB — you never need to guess.

## Where everything is

- **Slurm job:** name `build_database`. Find it: `squeue -u "$USER"`.
- **Slurm log (primary channel):**
  `/scratch/projects/torchcell/database/slurm/output/<job>_build_database.out`
- **wandb (structured channel):** project `tcdb` (entity `zhao-group`). The run id is on
  the line `wandb: 🚀 View run at https://wandb.ai/zhao-group/tcdb/runs/<run_id>` near the
  top of the `.out`.
- **Build tree:** `/scratch/projects/torchcell/database` (datasets under `data/torchcell`,
  neo4j store under `data/databases`, BioCypher CSVs under `biocypher-out/<ts>`).
- **Dev tree (source LMDBs):** `/scratch/projects/torchcell-scratch/data/torchcell`.
- **Served DB:** container `tc-neo4j-readonly`, `bolt://localhost:7687`, database
  `torchcell`, auth `neo4j` / `torchcell`.
- **Submit (full build):**
  `sbatch --export=ALL,KG_CONFIG=kg_full --cpus-per-task=64 --mem=400G --time=24:00:00 database/slurm/scripts/gilahyper_first_build-slurm_docker.slurm`
  (use `KG_CONFIG=kg_small`, `--cpus 32 --mem 256G` for the fast subset test).
- Python: `~/miniconda3/envs/torchcell/bin/python`. `$DATA_ROOT` (dev) = `/scratch/projects/torchcell-scratch`.

## Step 1 — locate & read the slurm .out

```bash
squeue -u "$USER"                     # is a build_database job running?
OUT=/scratch/projects/torchcell/database/slurm/output/<job>_build_database.out
grep -nE 'Building KG:|subset: .*->|SKIP .* no LMDB|Writing (nodes|edges) for adapter|Finished iterating|Knowledge graph creation completed|Executing generated|CREATE DATABASE|count\(n\)|more or fewer properties|Error executing|Failed to create|Traceback|Container failed' "$OUT" | tail -30
stat -c '%y' "$OUT"                   # mtime: is the log still growing?
```

## Step 2 — wandb summary (structured metrics)

Get `<run_id>` from the `.out`, then:

```bash
~/miniconda3/envs/torchcell/bin/python - <<'PY'
import wandb
r = wandb.Api().run("zhao-group/tcdb/<run_id>")
s = dict(r.summary)
print("state:", r.state)
for k in ("n_adapters","total_nodes","total_edges"):
    if k in s: print(k, "=", s[k])
print("adapters written:", len([k for k in s if k.endswith("_n_nodes")]))
print("datasets instantiated:", len([k for k in s if k.endswith("_len")]))
PY
```

`r.state` is `running` / `finished` / `failed` / `crashed`. `<Adapter>_n_nodes` appearing
means that adapter finished writing — that's how you confirm a specific dataset (Caudal,
ProteomeZelezniak, …) landed. `<Dataset>_len` shows the post-subset record count (e.g. the
Costanzo caps show `100000`).

## Step 3 — assess health

- **Running fine:** job `R`, `.out` mtime recent (< ~10 min), adapter count climbing, no
  error lines. → report one line, done.
- **Slow but alive:** a big adapter (Kuzmin 632k/410k, Kemmeren/morphology big-vectors) can
  legitimately take 10–30 min with no `.out` line. Check the wandb `_len` for the current
  dataset before calling it stuck. **Do NOT use `docker exec pgrep` as a liveness probe** —
  under the 32–64 build workers it false-negatives constantly; use `.out` mtime + wandb state.
- **Stalled:** `.out` mtime stale > ~15 min AND wandb `state=running` AND no plausible
  slow-adapter reason. Inspect the current adapter; likely OOM or a pathological dataset.
- **Errored:** an error line in the `.out` (see failure modes). The build python died.
- **OOM:** `docker inspect tc-neo4j --format '{{.State.OOMKilled}}'` = `true`.
- **Done:** `.out` shows `count(n)` and `tc-neo4j-readonly` is `Up ... (healthy)`.

## Step 4 — act

- **Done:** verify the DB and regenerate the report:
  ```bash
  docker exec tc-neo4j-readonly bash -c 'source /.env; cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -d torchcell "MATCH (n) RETURN count(n);"'
  ~/miniconda3/envs/torchcell/bin/python experiments/database/scripts/neo4j_graph_contents_table.py
  ```
  Report node/edge counts + which datasets landed. If in a poll loop, **stop the loop.**
- **Errored / stalled / OOM:** diagnose from the failure-modes table, apply the fix, then
  **restart** (Step 5). If OOM, lower `--cpus-per-task`/`--mem` or add a `subset.per_dataset`
  cap for the offending dataset in `kg_full.yaml`.
- **Running fine:** one-line status, continue.

## Step 5 — restart procedure (clean rebuild)

```bash
docker stop tc-neo4j tc-neo4j-readonly; docker rm -f tc-neo4j tc-neo4j-readonly
# reset the build tree (root helper; keeps slurm/ + export/)
docker run --rm -v /scratch/projects/torchcell/database:/db alpine sh -c \
  'rm -rf /db/biocypher /db/biocypher-out /db/conf /db/data /db/.env /db/logs /db/metrics /db/plugins /db/import'
cd /home/michaelvolk/Documents/projects/torchcell && git pull --ff-only origin main   # get latest fixes
DATA_ROOT=/scratch/projects/torchcell ~/miniconda3/envs/torchcell/bin/python -m torchcell.database.directory_setup
# hardlink-stage datasets (instant, same filesystem)
cp -al /scratch/projects/torchcell-scratch/data/torchcell/. /scratch/projects/torchcell/database/data/torchcell/
for d in sgd string tflink go; do cp -al /scratch/projects/torchcell-scratch/data/$d /scratch/projects/torchcell/database/data/; done
mkdir -p /scratch/projects/torchcell/database/{biocypher-out,slurm/output}
sbatch --export=ALL,KG_CONFIG=kg_full --cpus-per-task=64 --mem=400G --time=24:00:00 \
  database/slurm/scripts/gilahyper_first_build-slurm_docker.slurm
```

## Known failure modes → fix (learned 2026-07-18/19)

| Symptom in `.out` / state | Cause | Fix |
|---|---|---|
| `PermissionError: .../conf/neo4j.conf` from `directory_setup` | neo4j entrypoint `chown -R neo4j` flipped bind-mount ownership; a 2nd michaelvolk run can't overwrite conf | Root-helper clear + rerun `directory_setup` (Step 5). Non-fatal if the copied config is already valid. |
| `HTTP Error 403` on `current.geneontology.org/.../go.obo` | genome does an unconditional external download; bare-urllib UA is blocked | Ensure `data/go/go.obo` is staged (curl once: `curl -sL -o go.obo http://current.geneontology.org/ontology/go.obo`). |
| `SKIP <Dataset>: no LMDB` | dataset not staged into the build tree | `cp -al` it from the dev tree (Step 5 staging). |
| `... extra_forbidden` / `N validation errors for ExperimentReferenceIndex` | stale LMDB **and** stale `preprocess/experiment_reference_index.json` vs current schema | Rebuild that dataset: move `processed/` AND `preprocess/experiment_reference_index.json` aside, re-instantiate the loader (inject genome/graph as its `main()` does), re-stage. |
| `... has more or fewer properties than another` (a whole node class dropped) | a node property the adapter emits isn't declared in `biocypher/config/torchcell_schema_config.yaml` | Declare it under that node class (mirror an existing `*_se: str`); land + re-`directory_setup` (the config is copied to the build tree). |
| Caudal / natural-isolate adapter runs for hours | per-isolate perturbation explosion + (pre-fix) O(P²) `genotype_id` hashing | Landed: O(P²) hoist in `cell_adapter._perturbation_to_genotype_edges` + Caudal "compact genotype" (perturbation node/edge methods removed from its adapter yaml). If a new high-perturbation dataset appears, apply the same compaction. |
| `OOMKilled=true` / stall on a big dataset | too many workers × chunk memory | Lower `--cpus-per-task`/`--mem`, or add a `subset.per_dataset: {<DatasetClass>: <n>}` cap in `kg_full.yaml`. |

## Notes

- The full-registry build is `create_scerevisiae_kg_small` (misnamed — it builds ALL of
  `dataset_adapter_map`, subset-capped by config). `kg_small`=1000/dataset, `kg_full`=full
  with `subset.per_dataset` overrides.
- wandb covers the **build phase only** (ends at `wandb.finish()`); the neo4j-admin import,
  `CREATE DATABASE`, and final **deduplicated** node count are slurm/docker steps — read them
  from the `.out`. wandb `total_nodes` is nodes *written* (pre-dedup), always ≥ the DB count.
- Related: `[[plan.kg-database-build-environment-fix.2026.07.18]]`, memory `kg-build-gilahyper-state`.
