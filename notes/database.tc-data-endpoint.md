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

## 2026.09.30 - Deployed on Radiant

Steps 1 to 3 of the recipe ran on 2026-09-30; the endpoint is live on the database host (`torchcell-database.ncsa.illinois.edu` resolves to `141.142.216.218`), container `tc-data-endpoint`, `0.0.0.0:8724`, healthy, `restart: unless-stopped`. What differed from the recipe:

- **Location.** The store and the raw mirror sit under the Taiga torchcell data tree, as siblings of the experiment outputs: `/mnt/zhao5/mjvolk3/projects/torchcell/data/torchcell/tc-data` (3 archives, `sha256sum -c SHA256SUMS` OK on arrival) and `/mnt/zhao5/mjvolk3/projects/torchcell/data/torchcell/torchcell-raw` (19 keys, 7.4 GB rsynced from `$DATA_ROOT/torchcell-raw`), not the `tc-data` sibling of `kg-releases` the recipe proposed.
- **uid.** `rocky` is uid 1000 on Radiant and writes to Taiga as `1000:555647` (`zhao5_nfs`, files mode 660 in a setgid group), so the container runs as `TC_DATA_UID=1000:555647`, not 67392. The compose file's comment now says so.
- **Build context.** The Neo4j checkout at `~/projects/torchcell` is old (`f3181714`, with local modifications under `database/`) and the repo has no `.dockerignore`, so building from a checkout would ship a 10 GB context. The image builds from `~/projects/tc-data/` (12 MB: `git archive HEAD torchcell Dockerfile.tc-data docker-compose.tc-data.yml` piped over ssh), image `tc-data-endpoint:latest`, 177 MB. Redeploying a code change is the same archive pipe plus `docker compose --env-file tc-data.conf -f docker-compose.tc-data.yml up -d --build`.
- **Config file.** The compose variables live in `~/projects/tc-data/tc-data.conf` (mode 600) and are passed with `--env-file tc-data.conf`.
- **Key mint.** `--gen-key` output through `docker compose run` starts with the network-creation lines, so the key is the fourth line of the captured output, not the second. One key, `mjvolk3`, is stored as a hash in `~/.config/torchcell/tc_data_keys.json`; the plaintext is in `~/.config/torchcell/tc_data_key.mjvolk3.txt` on Radiant (mode 600) and nowhere else.
- **Port.** 8724 answers on the host but not from outside (`curl` from GilaHyper: no route); Radiant has no `firewalld` and no `openstack` CLI, so the port is opened in the OpenStack security group of the VM, by hand, as 7473 and 7687 were. Until then the collaborator path is the ssh tunnel of step 4, which the downloads guide now documents.

Verified on the host: `/health` reports 3 artifacts and 19 raw keys, `/docs` 200, `/datasets` 401 without a key and the three rows with it, `/raw` lists 19 keys. From GilaHyper through the tunnel, `DatasetClient.from_env()` downloaded, verified and unpacked all three archives (LMDB entries 4,313, 4,678, 4,719). Group 1 (`gene_essentiality_sgd`, `smf_costanzo2016`) is still unpackaged.

## 2026.09.30 - Group 1 packaged: essentiality and SMF

Same packager, same release flags, into `/bulk/tc-data` and rsynced to the Taiga store (`sha256sum -c SHA256SUMS`: five OK):

| slug | archive | archive sha256 | bytes | content sha256 |
|---|---|---|---:|---|
| `gene_essentiality_sgd` | `gene_essentiality_sgd-1.5.0-b31114d6.tar.xz` | `b31114d648903a2a...` | 44,080 | `88ff6788bfc364f2...` |
| `smf_costanzo2016` | `smf_costanzo2016-1.5.0-d8a0f06c.tar.xz` | `d8a0f06cb534f868...` | 1,053,744 | `634eaf5ebe3e94c5...` |

The live endpoint reports 5 artifacts, all `supported`. Through the tunnel, `DatasetClient` downloaded, verified and unpacked both; the unpacked `processed/lmdb` holds 1,329 and 20,484 entries, the dev-store record counts (the 1,329 SGD records collapse to 1,140 graph nodes by content hash, as the essentiality-smf page states). Every dataset with a page is now downloadable.

## 2026.09.30 - Zero-ssh access is the user contract

The downloads guide and the README now give users one thing to set, `TC_DATA_URL=http://torchcell-database.ncsa.illinois.edu:8724`, plus a named key; no account on the host and no tunnel. The ssh tunnel in the section above is a maintainer path for the days before the OpenStack security-group rule opens port 8724 and must not appear in user-facing text. Two items remain on the maintainer side before the public URL answers: the security-group rule for 8724 (user-only, OpenStack console), and TLS in front of the endpoint, since the plain-HTTP port sends the `X-API-Key` header in clear across the internet. The simplest TLS front is a reverse proxy on the VM (Caddy or nginx with a Let's Encrypt or NCSA certificate on 443) forwarding to `127.0.0.1:8724`, after which the documented URL becomes `https://torchcell-database.ncsa.illinois.edu/data` or similar and 8724 can stay closed.

## 2026.10.01 - The `.env` route and how keys are issued, checked

Checked with a throwaway project directory holding a `.env` with dummy values (python-dotenv is a packaged dependency, `env/requirements.txt`):

| Case | `TC_DATA_URL` reaches the loader |
|---|---|
| `.env` beside the script, script only imports a loader, no `load_dotenv()` | no (`DatasetClient.from_env()` raises `KeyError: 'TC_DATA_URL'`; a loader would build from the publisher files, with no error) |
| script calls `load_dotenv()` first, run from the project directory | yes |
| same script run from another working directory | yes (python-dotenv searches upward from the calling script) |
| script in a subdirectory of the project | yes |
| `python -c` in the project directory | yes (interactive sessions search from the working directory) |

torchcell reads the two variables from the process environment (`experiment_dataset.py` `_download`, `DatasetClient.from_env`) and no module on the loader import path calls `load_dotenv()`. The README and the downloads guide now say: export the two variables, or put them in `.env` and call `load_dotenv()` before constructing a loader.

Keys are minted, one per person or group: `python -m torchcell.datasets.server --gen-key <name>` on the endpoint host prints the plaintext (given to the user once) and the `{name: sha256hex}` entry for `TC_DATA_KEYS_FILE`. The server builds its key table at startup (`DataServerConfig.from_env`), so a new key is accepted only after a restart; deleting an entry and restarting revokes it. One key exists today (`mjvolk3`). The downloads guide gained a Keys section saying this.
