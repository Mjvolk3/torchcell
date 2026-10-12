---
id: ait8focbbh6ni6esi3gwfdy
title: Manual_deposit
desc: ''
updated: 1791803755733
created: 1791803755733
---

## 2026.10.12 - One record shape for every by-hand deposit

Six bacterial keys received browser downloads on 2026-10-12 (#699, #788, #853, #691,
#910, #986). Rather than each loader hand-writing its own `DEPOSIT.md` (the Menasalvas
and Yunus deposits were written by the driver session), this module turns a typed
`ManualDeposit` into the three records CLAUDE.md asks for:

- `verify_deposit`: every file present with its recorded bytes and sha256; a file
  unpacked from a deposited archive must equal the member of that name in the archive.
  A mismatch raises with the manual recipe, the only way to make the bytes again.
- `write_deposit_records`: `DEPOSIT.md` and `SHA256SUMS.txt` beside the files, one pair
  per directory (so `sha256sum -c SHA256SUMS.txt` works in place). A record already on
  disk with other content is refused, never overwritten.
- `artifact_records` and `extend_manifest`: `RetrievalMethod.manual_browser` records whose
  `params` carry the recipe, who retrieved, where `DEPOSIT.md` sits and, for an archive
  member, the archive and its sha256; `extend_manifest` adds them without dropping or
  rewriting a held record (a disagreeing sha256 raises) and creates the manifest when
  the key has none.
- `ScriptedEquivalent`: when a scripted route was fetched and returned the same bytes
  (Elsevier's CDN for Nichols 2011, the PMC bucket for Teteneva 2024), the record says
  so, so a rebuild needs no person.

Used by `tong2020.record_carpe_deposit`, `balakrishnan2022.record_si_deposit`,
`ishii2007.record_som_deposit` and
`experiments/036-dataset-fixes-before-kg-build/scripts/manual_deposits_2026_10_12.py`.
