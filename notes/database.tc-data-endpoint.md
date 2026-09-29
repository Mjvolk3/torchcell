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
