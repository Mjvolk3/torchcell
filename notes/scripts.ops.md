---
id: fhmx0uv44ugvfh9ozikeodu
title: Ops
desc: ''
updated: 1789840106148
created: 1789840106148
---

## 2026.09.19 - The ops panel

`make ops` (`scripts/ops.sh status`) from GilaHyper, in the shape of iBioFoundry's
`make ops`: first the served knowledge-graph releases on every host, one row per
database (`torchcell.knowledge_graphs.releases status`, columns VERSION, RELEASE,
COMMIT#, DATE, DATASETS, NODES, ALIASES, STATUS), a sync verdict between GilaHyper and
Radiant, local main as `<date>-<sha>` with its subject, and how far each served commit
lags main; then health probes: the Browser page (with the styling seed tag), tc-lit
`/health`, the merge-queue loop heartbeat (`merge_queue.py loop-status`, the same probe
iBioFoundry uses), slurm (`sinfo` answering, running and pending counts), the three
disks with free space (yellow at 90 percent), and Radiant's HTTPS port. `make
ops-health` and `make ops-releases` print one half. Read-only, exit 0 always.

Hosts come from `.env` (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`) and the
`OPS_RADIANT_*`, `OPS_TC_LIT_URL`, `OPS_BROWSER_URL` knobs. Each host's bolt probe is
bounded by `timeout` (four times `OPS_TIMEOUT_SECONDS`), so a hung host cannot freeze
the panel. First run on 2026-09-19 (before the release node existed): GilaHyper
`torchcell [default]` 51 datasets, 99,723,455 nodes, aliases `latest,pinned`, "online
(no release node)"; Radiant "(unreachable: ... java.io.IOException: Input/output
error)", its system database faulting on NFS; tc-lit 200; merge-queue heartbeat 20 s;
/db at 96 percent (the 029 build had landed there).

After the release node was written (2026-09-19 13:07 CDT) the table read:

```
HOST       DATABASE             VERSION  RELEASE              COMMIT#  DATE        DATASETS  NODES       ALIASES        STATUS
gilahyper  neo4j                -        -                    -        -           0         0           -              online
gilahyper  torchcell [default]  1.0      2026.09.17-7715ee35  5541     2026.09.15  51        99,723,456  latest,pinned  online
radiant    (dbms)               -        -                    -        -           -         -           -              faulting ({code: Neo.DatabaseError.Statement.ExecutionFailed} {message: java.io.IOException: Input/output error})

sync: DIVERGED -- gilahyper=2026.09.17-7715ee35 radiant=?
main: 2026.09.18-93e20039  (fix(008): ...)
      gilahyper: 27 behind main
      radiant: no release node
```

NODES counts the store's nodes now, one more than the release record's 99,723,455: the
`KgRelease` node itself. Both hosts are queried in one python call (`status --host`
repeated), each bounded by `OPS_TIMEOUT_SECONDS` times four, so the columns align and a
hung host costs one row.

## 2026.09.26 - Where the time went, and `make ops-fast`

Measured on GilaHyper (`/usr/bin/time`, one run each): `python -c "import torchcell.knowledge_graphs"` 10.5 s, because the package `__init__` imported `create_scerevisiae_kg_small` eagerly (BioCypher, every loader, the adapters) although the release table only needs `releases.py` and `kg_manifest.py`; the radiant bolt probe 12.9 s (the store faults, so the query runs to the error); the radiant https probe the full 5 s curl timeout; everything else under a second. Two changes: `torchcell/knowledge_graphs/__init__.py` resolves its two public names lazily (PEP 562 `__getattr__`, pinned by `tests/torchcell/knowledge_graphs/test_package_lazy_imports.py`), and `OPS_HOSTS` (default `gilahyper,radiant`) names the hosts to query and probe; a host that is not listed gets no release rows, no health section, and no sync verdict. `make ops-fast` is `OPS_HOSTS=gilahyper OPS_TIMEOUT_SECONDS=2 bash scripts/ops.sh status`. Measured after the change (same method, one run each): `import torchcell.knowledge_graphs` 0.02 s, `releases --help` 0.23 s, `make ops-fast` 0.67 s, full `make ops` 11.1 s (the remaining time is the radiant bolt probe running to its I/O fault plus the radiant https timeout; before the change the full panel took about 26 s).

## 2026.10.06 - `ops.sh sync`: the one action with an exit code

`status`, `releases` and `health` stay exit-0 reporters. `sync` (`make ops-sync`) prints the release table and then fails (exit 1) when the listed hosts serve different releases or when any host serves a store without a release node, which under the pairing rule is an unpaired store no client should read. `kg_release.sh deploy` ends with it, and a cron or CI step can gate on it. The verdict lines carry the ✓/✗ marks the health probes use.

## 2026.10.07 - The Radiant https probe measures the service, not GilaHyper's resolver

`make ops` reported `radiant https 000` while the Radiant Browser answered 200 in a browser. Measured from GilaHyper: `curl` to `https://torchcell-database.ncsa.illinois.edu:7473/` returns 200 but takes 5.07 s, and `--write-out` puts all of it in name resolution (`time_namelookup` 5.02 s, connect plus TLS plus transfer 0.05 s); with `-4` the same request takes 0.05 s. `dig A` and `dig AAAA` each answer in under 50 ms, while `getent ahosts` (glibc, what curl and python use) takes 5.02 s. glibc sends the A and AAAA queries in parallel on one socket; the configured nameserver is the router at 192.168.1.1 with no `options` line in `/etc/resolv.conf`, and a router that drops one of the two parallel replies produces exactly this signature: glibc waits its 5 s timeout, then retries sequentially and succeeds. Every off-host lookup on GilaHyper pays it; the gilahyper rows are all `localhost` and never did.

The probe's curl budget is `OPS_TIMEOUT_SECONDS` (5 s), so the stall alone exhausts it and curl prints `000`, the code for "no HTTP response", with the server up. Fix in `ops.sh`: the Radiant probe passes `-4`, so the panel measures the service. The machine-level fix is separate and needs root: `options single-request` for the resolver, set on the NetworkManager connection (`nmcli connection modify "Wired connection 1" ipv4.dns-options single-request`, then `nmcli device reapply enp37s0f0`), which also removes the 5 s from the Bolt probes, tc-data pulls and every other off-host lookup.
