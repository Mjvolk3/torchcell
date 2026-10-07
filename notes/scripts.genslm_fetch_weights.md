---
id: 2kqfvcv49qan8n6mkuogbgt
title: Genslm_fetch_weights
desc: ''
updated: 1791416310857
created: 1791416310857
---

## 2026.10.07 - Fetching the GenSLM checkpoints through Globus

GenSLM checkpoints are released only on Globus endpoint `25918ad0-2a4e-4f37-bcfc-8183b19c3150`
(the `ramanathanlab/genslm` README); the endpoint refuses unauthenticated access, and the
Globus CLI moves bytes only endpoint to endpoint. Route:

1. `$DATA_ROOT/envs/globus-cli/bin/globus login --no-local-server` (one-time, interactive:
   open the printed URL, log in with the Illinois identity, paste the code back).
2. Globus Connect Personal on GilaHyper as the destination endpoint:
   `$DATA_ROOT/envs/globusconnectpersonal/globusconnectpersonal-3.3.1/globusconnectpersonal -setup`
   (one-time, interactive, same URL-and-code dance), then `-start`, with
   `$DATA_ROOT/models/genslm` inside its accessible paths (`-restrict-paths rw<dir>` or the
   `~/.globusonline/lta/config-paths` file).
3. `python scripts/genslm_fetch_weights.py ls --path /` to find the checkpoint directories, then
   `transfer --model genslm_25M_patric --path <dir>`: submits a checksum-synced transfer, waits
   on the task, and `record`s the sha256 plus the task id into `manifest.json`.

`record` alone pins a checkpoint that arrived by any other route. The loader in
[[torchcell.models.genslm]] refuses a checkpoint without a manifest entry or with a hash that
does not match it.

Status 2026.10.07: CLI 3.43.0 and Connect Personal 3.3.1 are unpacked under `$DATA_ROOT/envs/`;
no login has been done, nothing transferred.
