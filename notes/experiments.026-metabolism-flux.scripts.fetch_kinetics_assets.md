---
id: 11yad8n4iby2b2q91sj458o
title: Fetch_kinetics_assets
desc: ''
updated: 1788409359480
created: 1788409359480
---

## 2026.09.02 - Mirroring the two assets that complete the predictor inputs

Companion to [[experiments.026-metabolism-flux.scripts.kinetics_input_audit]], which
measures what is present. This fetches what was not.

| asset | closes | size |
| --- | --- | --- |
| MetaNetX `chem_prop.tsv` | substrate SMILES, 87.5 % to 95.3 % of catalytic units | 809,687,076 B |
| AlphaFold, one PDB per GEM accession | protein 3D structure, consumed only by DeepEnzyme | ~150 MB per 500 |

Each file records source URL, retrieval method, the exact retrieval command, sha256, byte
count and retrieval time into a manifest beside it, so the mirror rebuilds and verifies
without trusting a live URL.

### Three things measured the hard way

**AlphaFold rejects a descriptive User-Agent.** The same request that returns 200 with
urllib's default agent returns 403 with `torchcell/026-metabolism-flux (research mirror)`.
So that endpoint is sent no override at all rather than a disguise. MetaNetX accepts one.

**The model version is part of the filename and it moves.** `AF-<acc>-F1-model_v4.pdb`
now 404s; the live answer is v6, created 2025-08-01. A templated version silently rots
into a wall of 404s that reads as "this protein has no structure". The file URL is
therefore resolved per accession from `api/prediction/<acc>`, which is the only place that
states the current version, and the resolved URL is what gets recorded.

**One DNS blip killed a 1,161-file run outright.** The first attempt handled `HTTPError`
only, so a transient `URLError` propagated and ended the mirror after MetaNetX. Transport
failures are now retried with backoff, and `HTTPError` deliberately is not: a 404 or a 403
is the server's actual answer, and retrying it would turn a real result into a hang. That
distinction is what keeps the absent-model count honest.

### Why this does not compete with the sweep

Both fetches are network-bound and touch no GPU, so they run alongside training. The
inference they enable is also small, 7,456 pairs, minutes on one GPU. Filling four GPUs
overnight is the flux sweep's job, not this one's.
