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
- `--attach-to` names an existing item that holds the DOI and sits in the named collection; without it, a DOI the library already holds is refused.
- `--dry-run` resolves, fetches and verifies everything and writes nothing.
- Exit 0 filed (or would file), 2 refused, 3 no route yielded the article PDF (the summary prints the DOI URL to open in a browser; nothing was written).
