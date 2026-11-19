---
id: rgcfgcxsnqakhnarwhbokl5
title: Pinned_fetch
desc: ''
updated: 1795117434060
created: 1795117434060
---

## 2026.11.19 - Source first, mirror fallback for one pinned file

`fetch_pinned` leaves a sha256-and-size-pinned file at a destination. It downloads the recorded source URL first and accepts the bytes only when they equal the pin; on a failed request, a non-200 status or changed bytes it logs the reason and calls the mirror, whose bytes are re-hashed against the same pin. `FetchOutcome` records the origin and the reason. First user: [[torchcell.sequence.genome.standard]]. Written tier-agnostic (the mirror is a callable) so network releases such as STRING can use it once they have manifests.
