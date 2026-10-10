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

## 2026.10.10 - First transfers done

The user logged in once (`globus login --no-local-server`, Illinois identity). The target
turned out to be a Globus Connect Server v5 guest collection, so `globus endpoint show`
refuses it and `globus gcs collection show` wants a consent the identity does not have; `ls`
works. Root holds `data/` and `models/{25M,250M,2.5B,25B,legacy/}` (checkpoint sizes 101 MB,
1.01 GB, 10.1 GB, 103 GB; `legacy/` is the pre-2023-05-03 namespace-bug release the README
warns about).

A private Globus Connect Personal endpoint was registered from the CLI
(`globus gcp create mapped "gilahyper-torchcell" --private`, id
`8e26aca5-c472-11f1-a924-0affd5e180af`), set up with its key, and started with
`-restrict-paths rw/scratch/projects/torchcell-scratch/models/genslm`. It is a user process,
not a service: after a reboot run `globusconnectpersonal -start ...` again from
`$DATA_ROOT/envs/globusconnectpersonal/globusconnectpersonal-3.3.1/`. `transfer` for 25M and
250M completed with checksum verification and wrote the manifest; the 2.5B is one more
`transfer --model genslm_2.5B_patric --path /models/2.5B/` away.
