---
id: 82jmxn935ovpmq6nbxexyyo
title: Zotero_add_paper
desc: ''
updated: 1791795099510
created: 1791795099510
---

## 2026.10.12 - Thin wrapper, and the command line

Loads the repo `.env` (the group library's `ZOTERO_LIBRARY_ID` + `ZOTERO_API_KEY`, and `DATA_ROOT` for the raw mirror) and calls `zotero_add.main`. Policy, routes and refusals are in [[torchcell.literature.zotero_add]]; the rule itself is in the repo `CLAUDE.md`.

```bash
python scripts/zotero_add_paper.py --doi <doi> --collection database/<Organism> \
    --first-author <family name> [--title-contains <text>] \
    [--pdf <file> --pdf-source-url <url>] [--si <pdf> ... --si-source-url <url>] \
    [--attach-to <itemKey>] [--dry-run] [--raw-root <dir>] [--report <json>]
```

- `--first-author` is required and must match Crossref's first author (case- and accent-insensitive), so a mistyped DOI that resolves to another paper is refused before anything is fetched.
- `--pdf` is tried only after the publisher, the raw mirror and the PMC bucket all fail; `--si` files are attached as given. Both are recorded as `manual_browser` retrievals and need the URL the file came from.
- `--attach-to` names an existing item that holds the DOI and sits in the named collection or the `database` root; without it, a DOI the library already holds is refused.
- `--dry-run` resolves, fetches and verifies everything and writes nothing.
- Exit 0 filed (or would file), 2 refused, 3 no route yielded the article PDF (the summary prints the DOI URL to open in a browser; nothing was written).

## 2026.10.12 - First live filings

Twelve database papers were waiting on Zotero (#691, #910, #788 item 4, #911; [[database.manual-problems-ledger.2026.10]]). A read-only survey of the group library at 08:55 UTC found ten of them already filed by the owner between 08:30 and 08:37 UTC, nine of those with a stored PDF. The script made three runs:

| paper | command | result |
|---|---|---|
| Teteneva 2024, 10.1093/ismejo/wrae096 | `--collection database/Escherichia-coli --first-author Teteneva --title-contains "lake water"` | created `PUKZFNFZ`; attachment `DFFQ6A6W`, 3,972,652 bytes, route `pmc_cloud` (`PMC11188689.1/PMC11188689.1.pdf`) after both OUP `article-pdf` URLs answered 403; sha256 `d01d686125a9be5c3f9e38291b849b3e80d776464392770d9523757ce8e89d1d`, re-downloaded from Zotero and re-hashed to the same value |
| Thompson 2019 valerolactam, 10.1016/j.mec.2019.e00098 | `--collection database/Pseudomonas-putida --first-author Thompson --attach-to ZVF4KENC` | owner's item `ZVF4KENC` (database root) got attachment `65FKM8IN`, 1,194,524 bytes, route `raw_mirror` (the recorded `pmc_cloud` copy `thompsonOmicsdrivenIdentificationElimination2019/paper/PMC6838509.1.pdf`) after ScienceDirect answered 403; sha256 `ed6c14563b2b48d10f269b1b1abc624b4cbfbd634c49cbf2e19b4a1b53238b2f` |
| Nichols 2011, 10.1016/j.cell.2010.11.052 | `--collection database/Escherichia-coli --first-author Nichols --attach-to REEL3RAD` | exit 3, nothing written: ScienceDirect 403, no raw-mirror record, `PMC3060659.1` holds text and XML only |

The Teteneva run exposed a read-after-write lag: the read straight after the upload showed `DFFQ6A6W` with `md5` None and the run raised, although the upload had succeeded (a read a minute later held `a2cab4695e2ddb37dbc077c40f021800`, the MD5 of the bytes sent). `verify_stored` now re-reads a missing MD5 up to 10 times, 6 s apart; the Thompson run passed through it.

Run reports (JSON, scratch only): `thompson_valerolactam.json`, `nichols.json` under the session scratchpad `zotero-add-database-paper/`.
