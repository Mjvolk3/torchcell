---
id: k3frsr8rymiwvubzjhq80v6
title: Tc Data Endpoint
desc: ''
updated: 1790716976369
created: 1790716976369
---

## 2026.09.29 - Deploying tc-data on Radiant

Recipe for serving the artifact store and the raw mirror from the Radiant VM (`rocky@141.142.216.218`, the `TAIGA_HOST` of `scripts/kg_release.sh`). Not yet run on Radiant; every step below was exercised locally except the Radiant-specific ones, which follow Gotcha 4 of [[plan.data-release-program.2026.09.29]]: 40 GB root disk, no block volume, root-squashed Taiga NFS, so a Python service runs in Docker as uid 67392 with the served trees bind-mounted read-only and nothing staged on the root disk. NFS breaks Neo4j page reads, not file streaming, so the store can live on the existing Taiga mount today (Decision 7).

1. Package on GilaHyper, where the dev LMDBs live, into a local store, then ship it. `sha256sum -c SHA256SUMS` on arrival is the acceptance check, as for KG releases.

   ```bash
   STORE=/bulk/tc-data   # local store on GilaHyper (never /db)
   ~/miniconda3/envs/torchcell/bin/python scripts/package_dataset_lmdb.py \
     --dataset-dir $DATA_ROOT/data/torchcell/smf_costanzo2016 --store $STORE \
     --kg-release 2026.09.21-ab6d8c5d --kg-version 1.2
   rsync -rlpt --partial --info=progress2 $STORE/ rocky@141.142.216.218:/mnt/zhao5/mjvolk3/projects/torchcell/tc-data/
   rsync -rlpt --partial $DATA_ROOT/torchcell-raw/ rocky@141.142.216.218:/mnt/zhao5/mjvolk3/projects/torchcell/torchcell-raw/
   ssh rocky@141.142.216.218 'cd /mnt/zhao5/mjvolk3/projects/torchcell/tc-data && sha256sum -c SHA256SUMS'
   ```

   `/mnt/zhao5/mjvolk3/projects/torchcell/tc-data` is proposed as a sibling of the existing `kg-releases` directory on the same mount; `-rlpt` rather than `-a` for the same setgid-group reason recorded in `kg_release.sh`.

2. Mint a key on Radiant. Only the hash is stored; the plaintext goes to the collaborator once.

   ```bash
   mkdir -p ~/.config/torchcell && chmod 700 ~/.config/torchcell
   docker compose -f docker-compose.tc-data.yml run --rm tc-data-endpoint \
     python -m torchcell.datasets.server --gen-key <collaborator>
   # paste the printed {"<collaborator>": "<sha256hex>"} into ~/.config/torchcell/tc_data_keys.json (mode 600)
   ```

   Several collaborators are several entries in that JSON; revoke one by deleting its entry and restarting.

3. Run the container from the repo checkout on Radiant with a `.env` beside the compose file:

   ```
   TC_DATA_ROOT=/mnt/zhao5/mjvolk3/projects/torchcell/tc-data
   TC_DATA_RAW_ROOT=/mnt/zhao5/mjvolk3/projects/torchcell/torchcell-raw
   TC_DATA_KEYS_FILE=/home/rocky/.config/torchcell/tc_data_keys.json
   TC_DATA_PORT=8724
   TC_DATA_UID=67392
   ```

   ```bash
   docker compose -f docker-compose.tc-data.yml up -d --build
   curl http://127.0.0.1:8724/health
   curl -H "X-API-Key: $KEY" http://127.0.0.1:8724/datasets
   ```

   The image (`Dockerfile.tc-data`) is `python:3.13-slim` plus fastapi, uvicorn, pydantic and python-dotenv; the container runs as `TC_DATA_UID` so the root-squashed mount is readable, and all three mounts are `:ro`. The compose `HEALTHCHECK` probes `/health` through stdlib urllib.

4. Off-LAN access goes through an ssh tunnel until a reverse proxy with TLS is set up, the same stance as `tc-lit`:

   ```bash
   ssh -N -L 8724:127.0.0.1:8724 rocky@141.142.216.218
   export TC_DATA_URL=http://127.0.0.1:8724 TC_DATA_API_KEY=<plaintext>
   ```

   With those two variables set, any `ExperimentDataset` loader whose LMDB is absent downloads its artifact instead of building ([[torchcell.data.experiment_dataset]]). Swagger is at `$TC_DATA_URL/docs`; the collaborator page is `docs/source/guide/downloads.md`.

5. Deprecating an artifact: repackage the slug with `--status deprecated` (same bytes, same name, the row is replaced), rsync, and the row stays in the index while `select` skips it. Remove the file only after every checkout that could name it is gone.

## 2026.09.30 - First archives in the local store: Cooper, Mulleder, Cachera

Step 1 of the recipe above ran on GilaHyper for the three showcase group 3 stores, into `/bulk/tc-data` (`scripts/package_dataset_lmdb.py`, `--kg-release 2026.09.21-ab6d8c5d --kg-version 1.2`; the packager stamps its own `torchcell.__version__`, 1.5.0):

| slug | archive | archive sha256 | bytes | content sha256 |
|---|---|---|---:|---|
| `amino_acid_cooper2010` | `amino_acid_cooper2010-1.5.0-cabcd04a.tar.xz` | `cabcd04ae4094ed0...` | 1,083,036 | `09ddfc956c239ac4...` (equals the served release) |
| `amino_acid_mulleder2016` | `amino_acid_mulleder2016-1.5.0-633d0cf8.tar.xz` | `633d0cf88921ece4...` | 1,426,568 | `bc92df7f3359d556...` (dev store; the release serves `6c5ad75d...`, the #143 medium rebuild) |
| `betaxanthin_cachera2023` | `betaxanthin_cachera2023-1.5.0-be734791.tar.xz` | `be734791cd86c11a...` | 289,016 | `bcf528614e48defa...` (dev store; the release serves `afc2eaa4...`, the #195 renames) |

Acceptance check, run locally the same night: the server on `127.0.0.1:8799` with `TC_DATA_ROOT=/bulk/tc-data` listed the three rows (`/health` reports 3 artifacts, 19 raw keys; `/openapi.json` 200), and `DatasetClient.from_env()` selected, downloaded (sha256 verified), and unpacked each archive; opening each unpacked `processed/lmdb` gives 4,313, 4,678 and 4,719 entries, the dev-store record counts on the showcase page ([[experiments.034-showcase-datasets.scripts.amino_acid_betaxanthin]]).

Two of the three content hashes differ from the served release because the dev stores were rebuilt after it (the page's served-versus-dev table names the fields). A collaborator who downloads them gets the dev stores, not byte-for-byte what the graph serves, until the next full build re-admits them. `gene_essentiality_sgd` and `smf_costanzo2016` (group 1) are not packaged yet. Steps 2 and 3 (key mint, container on Radiant) and the rsync to Taiga have still not run.
