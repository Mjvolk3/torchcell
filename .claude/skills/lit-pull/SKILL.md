---
name: lit-pull
description: Pull OCR literature artifacts from the keyed torchcell literature endpoint and verify sha256. Use on the M1 Mac (or any client) to list citation keys, fetch a paper's manifest/markdown/SI, or download an artifact with integrity checking.
---

# Literature Pull

Client for the private keyed read-only literature endpoint served by
`torchcell.literature.server` on GilaHyper. Lists + pulls artifacts from the OCR
mirror (`$DATA_ROOT/torchcell-library/<citation_key>/`) over the LAN, verifying every
download against the manifest sha256.

## Prerequisites (set once on the client, in `.env`)

Put these two lines in your `.env` (repo root, or wherever you keep it — point at it with
`TC_LIT_ENV` if not `./.env`):

```bash
TC_LIT_URL=http://192.168.1.17:8723        # the GilaHyper server host:port
TC_LIT_API_KEY=<plaintext key from the server>   # server: python -m torchcell.literature.server --gen-key <name>
```

**`TC_LIT_API_KEY` is the PLAINTEXT key** printed by `--gen-key` (the first block), NOT the
`{name: sha256hex}` hash that goes in the server's keys file — they are different strings.

Load just those two vars into the shell (a safe partial-source that ignores the rest of the
`.env`, so unrelated lines can't break or execute):

```bash
set -a; source <(grep -E '^(TC_LIT_URL|TC_LIT_API_KEY)=' "${TC_LIT_ENV:-.env}"); set +a
```

All requests send the key as the `X-API-Key` header. A missing/unknown key returns 401. If
you get 401, confirm the key matches the server's stored hash:
`printf '%s' "$TC_LIT_API_KEY" | sha256sum` must equal the hash in the server's keys file.

## Step 1: Confirm the endpoint is up

```bash
curl -s "$TC_LIT_URL/health"           # no auth: {status, n_keys, mirror_root}
```

## Step 2: List citation keys / search

```bash
curl -s -H "X-API-Key: $TC_LIT_API_KEY" "$TC_LIT_URL/keys"
curl -s -H "X-API-Key: $TC_LIT_API_KEY" "$TC_LIT_URL/search?q=proteome"   # filename + paper.md substring
```

## Step 3: Inspect a paper

```bash
CK=messnerProteomicLandscapeGenomewide2023
curl -s -H "X-API-Key: $TC_LIT_API_KEY" "$TC_LIT_URL/keys/$CK/manifest"    # roles + sha256
curl -s -H "X-API-Key: $TC_LIT_API_KEY" "$TC_LIT_URL/keys/$CK/files"       # [{path, role, bytes, sha256}]
```

## Step 4: Download an artifact and verify integrity

The response carries `X-Artifact-SHA256`; recompute locally and compare. The manifest is
the trust anchor — never trust the header alone.

```bash
CK=messnerProteomicLandscapeGenomewide2023
REL=paper.md
curl -s -D /tmp/hdr.txt -H "X-API-Key: $TC_LIT_API_KEY" \
  "$TC_LIT_URL/keys/$CK/artifact/$REL" -o "$REL"
EXPECT=$(grep -i x-artifact-sha256 /tmp/hdr.txt | tr -d '\r' | awk '{print $2}')
GOT=$(shasum -a 256 "$REL" | awk '{print $1}')   # macOS: shasum; Linux: sha256sum
[ "$EXPECT" = "$GOT" ] && echo "OK sha256 verified" || echo "MISMATCH — do not trust bytes"
```

## Notes

- Read-only: the endpoint never mutates the mirror.
- Dynamic: newly captured papers appear in `/keys` with no server restart.
- Provenance: a manifest with `provenance_complete: false` means the paper was OCR'd before
  provenance was formalized (bytes are still sha256-pinned; upstream retrieval unknown).
- Off-LAN: tunnel over SSH (`ssh -L 8723:localhost:8723 gilahyper`) rather than exposing the
  port publicly; there is no TLS on the trusted-LAN default.
