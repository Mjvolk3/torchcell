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
