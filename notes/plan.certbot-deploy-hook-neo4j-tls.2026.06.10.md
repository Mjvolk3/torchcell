---
id: jfi5dh8s1jk1f7krykg3hqg
title: certbot-deploy-hook-neo4j-tls
desc: ''
updated: 1781124610893
created: 1781124610893
---

## Context

The Neo4j HTTPS endpoint (`https://torchcell-database.ncsa.illinois.edu:7473`)
served an expired Let's Encrypt certificate, so browsers showed "not secure".
certbot itself was healthy -- `certbot-renew.timer` is enabled and active, and
the canonical cert under `/etc/letsencrypt/live/.../` was valid (renewed to an
ECDSA cert good through 2026-09-04). The problem was purely **deployment**: the
renewed cert was never copied into `database/certificates/https/` (the dir
bind-mounted into the `tc-neo4j` container) and the container was never
restarted to load it. Neo4j reads TLS material only at startup.

The repo already intends to solve this: `database/scripts/setup_letsencrypt.sh`
writes a deploy hook (`neo4j-reload.sh`) into
`/etc/letsencrypt/renewal-hooks/deploy/`. But the served cert still went stale,
which means on this host the hook is **absent or broken** -- certbot's most
recent successful renewal (2026-06-06) did not trigger a copy + restart (we had
to run `copy_certs.sh` by hand). This plan makes the hook reliable, removes the
duplicated cert-copy logic, and documents an install + test path that does not
burn Let's Encrypt rate limits.

## Relevant Files

| Path | Action | Purpose | Stance |
|------|--------|---------|--------|
| `database/scripts/letsencrypt-deploy-hook.sh` | NEW | Deploy hook; domain-guards on `RENEWED_DOMAINS`, then `exec`s `copy_certs.sh` | n/a |
| `database/scripts/copy_certs.sh` | REFERENCE | Single source of truth: cp -L live certs into `certificates/https`, chown rocky:neo4j, chmod 640/644, `docker restart tc-neo4j` | provisional |
| `database/scripts/setup_letsencrypt.sh` | MODIFY | Install the new hook + remove old inline `neo4j-reload.sh` duplicate | stable |
| `database/certificates/https/{private.key,public.crt}` | REFERENCE | Served cert material; git-ignored (`.gitignore:105`) -- never committed | n/a |
| `notes/database.openstack.chrome-site-insecure-fix.md` | REFERENCE | Existing manual-recovery runbook; references the hook | n/a |

## Key Design Decisions

1. **Hook delegates to `copy_certs.sh` instead of re-implementing it.** The old
   inline hook in `setup_letsencrypt.sh` duplicated the cp/chown/restart logic,
   so the manual path and the automatic path could silently diverge (and the
   domain string was hard-coded twice). One `exec copy_certs.sh` keeps a single
   source of truth.

2. **Guard on `RENEWED_DOMAINS`, not on file mtimes.** certbot runs deploy hooks
   only on actual renewal and exports `RENEWED_DOMAINS` (space-separated). A
   `case " $RENEWED_DOMAINS " in *" $DOMAIN "*)` match makes the hook a no-op for
   any unrelated lineage. This is necessary correctness logic, not a "fallback"
   (per CLAUDE.md's no-fallback rule) -- there is exactly one positive path.

3. **`setup_letsencrypt.sh` removes the legacy `neo4j-reload.sh`.** A prior run
   may have installed the divergent inline hook; `rm -f` of the old name plus
   `cp` of the maintained script to `torchcell-neo4j.sh` guarantees the broken
   one is gone and only the maintained hook remains.

4. **Absolute paths in the hook.** Deploy hooks run as root with an arbitrary
   cwd; the hook and `copy_certs.sh` both use absolute paths so cwd is
   irrelevant.

5. **No executable bit relied on in the repo.** The install step does
   `sudo chmod +x` on the installed copy; certbot requires the hook executable
   only at `/etc/letsencrypt/renewal-hooks/deploy/`.

## Approach

The hook is ~10 lines: set `DOMAIN`, bail with `exit 0` unless `DOMAIN` appears
in `RENEWED_DOMAINS`, then `exec /home/rocky/projects/torchcell/database/scripts/copy_certs.sh`.
`copy_certs.sh` is unchanged -- it already uses absolute paths, dereferences the
Let's Encrypt symlinks with `cp -L`, fixes ownership to `rocky:neo4j`, and
restarts the container.

`setup_letsencrypt.sh` no longer heredocs an inline hook; it `rm -f`s the old
`neo4j-reload.sh` and `cp`s `letsencrypt-deploy-hook.sh` to
`/etc/letsencrypt/renewal-hooks/deploy/torchcell-neo4j.sh`, then `chmod +x`.

Installation is a manual sudo step (certbot owns `/etc/letsencrypt`); it is not
run by the implementer. Out of scope: changing the renewal schedule, the cron
entry, or `build_openstack.sh`'s cert handling.

## Install + Verify (run by user; needs sudo)

```bash
# 1. Install the hook
sudo cp /home/rocky/projects/torchcell/database/scripts/letsencrypt-deploy-hook.sh \
        /etc/letsencrypt/renewal-hooks/deploy/torchcell-neo4j.sh
sudo chmod +x /etc/letsencrypt/renewal-hooks/deploy/torchcell-neo4j.sh

# 2. Remove any stale duplicate left by a prior setup run
sudo rm -f /etc/letsencrypt/renewal-hooks/deploy/neo4j-reload.sh

# 3. Confirm what is installed
sudo ls -l /etc/letsencrypt/renewal-hooks/deploy/

# 4. Test WITHOUT a real renewal (no rate-limit risk): simulate certbot's env
sudo RENEWED_DOMAINS="torchcell-database.ncsa.illinois.edu" \
     /etc/letsencrypt/renewal-hooks/deploy/torchcell-neo4j.sh

# 5. Confirm the served cert matches the live cert
echo | openssl s_client -connect localhost:7473 \
     -servername torchcell-database.ncsa.illinois.edu 2>/dev/null \
     | openssl x509 -noout -dates
```

Step 4 should print the copied-cert dates and restart `tc-neo4j`; step 5 should
report `notAfter=Sep  4 ... 2026`.

## Gotchas

1. **`certbot renew --dry-run` does NOT run deploy hooks**, and
   `--force-renewal` issues a real cert (Let's Encrypt rate limits: ~5
   duplicate certs/week). Test by invoking the hook directly with
   `RENEWED_DOMAINS` exported (step 4) -- never force-renew just to test.

2. **Hook runs as root with arbitrary cwd.** Any relative path breaks. Both
   scripts use absolute paths; keep it that way.

3. **`docker restart` during renewal is a brief read-only-DB blip.** Acceptable
   here; certbot deploy hooks fire at most a few times a year.

4. **Container-down case.** `copy_certs.sh` calls `docker restart tc-neo4j`
   unconditionally; if the container is absent the restart errors but the cert
   is still copied. Left as-is to honor the no-fallback rule -- the next
   `build_openstack.sh` / manual run picks up the already-copied cert.

5. **Never commit cert material.** `private.key`/`public.crt` are covered by
   `.gitignore:105 database/certificates/*`; verified with `git check-ignore`.

## Verification

- `bash -n` on `letsencrypt-deploy-hook.sh` and `setup_letsencrypt.sh` (both
  pass).
- `shellcheck` if available.
- Manual: the Install + Verify block above on the host.
