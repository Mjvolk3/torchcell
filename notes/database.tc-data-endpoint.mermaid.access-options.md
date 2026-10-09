---
id: ontdi34v55mh05qgpxuhdie
title: Access Options
desc: ''
updated: 1791530071209
created: 1791530071209
---

## 2026.10.09 - Three ways a collaborator reaches the tc-data tiers

The endpoint on Radiant holds one named key, `mjvolk3`, stored as a sha256 hash. The key identifies who downloads, makes revocation per person, and keeps anonymous callers at `/health` only. The data mounts are read-only. Port 8724 is plain HTTP, so the key rides in `X-API-Key` unencrypted on option A. The Caddy container `tc-proxy` on Radiant already holds a Let's Encrypt certificate for the public hostname, which is what option C uses. Measured 2026.10.09: 7473 answers 200 from GilaHyper, 8724 times out, and both ports are identical at the host level, so the security group is the only difference. Background in [[database.tc-data-endpoint]].

```mermaid
flowchart LR
  classDef user fill:#FFE6CC,stroke:#D79B00
  classDef net fill:#F8CECC,stroke:#B85450
  classDef safe fill:#E1D5E7,stroke:#9673A6
  classDef svc fill:#FFF2CC,stroke:#D6B656

  U["collaborator<br/>torchcell resolver + named key"]:::user

  subgraph A["A. open port 8724"]
    A1["internet, plain HTTP<br/>key readable in transit"]:::net
  end
  subgraph B["B. ssh tunnel, today"]
    B1["ssh to Radiant<br/>needs a Radiant account<br/>encrypted"]:::safe
  end
  subgraph C["C. tc-proxy, port 443"]
    C1["internet, HTTPS via Caddy<br/>Let's Encrypt cert already on the VM<br/>key encrypted in transit"]:::safe
  end

  SG["OpenStack security group<br/>A opens 8724, C opens 443, B opens nothing new"]:::net
  T["tc-data container<br/>checks key hash, serves read-only tiers"]:::svc
  F["Taiga files<br/>raw, genomes, objects"]:::svc

  U --> A1 --> SG
  U --> B1 --> SG
  U --> C1 --> SG
  SG --> T --> F
```

| | A. open 8724 | B. ssh tunnel | C. tc-proxy TLS |
|---|---|---|---|
| who can use it | anyone with a key | only people with a Radiant login | anyone with a key |
| key in transit | cleartext | encrypted | encrypted |
| change needed | one dashboard rule | none | Caddy route plus a dashboard rule for 443 |
| `make ops` tc-data line | green | stays red | green with the URL changed |

A collaborator cannot use B, so the choice for sharing is between A and C. C is the end state: a `handle_path /data/*` block in the Caddyfile that reverse-proxies to the tc-data container, 443 opened in the security group, and `TC_DATA_URL` set to `https://torchcell-database.ncsa.illinois.edu/data`.

Rendered: `notes/assets/pdf-output/database.tc-data-endpoint.mermaid.access-options.pdf` (`bash notes/assets/publish/scripts/mermaid_pdf.sh notes/database.tc-data-endpoint.mermaid.access-options.md`).
