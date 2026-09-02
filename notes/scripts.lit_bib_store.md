---
id: xqx3opxdivzm6mq8gcireql
title: Lit_bib_store
desc: ''
updated: 1788387036939
created: 1788387036939
---

## 2026.09.02 - Export Every Repo-Declared Bibliography into the Mirror

`scripts/lit_bib_store.py`. Thin wrapper over [[torchcell.literature.bib_store]]: discovers
the bibliographies the repo declares, pulls each over the Zotero Web API (headless), and
writes them with a sha256 manifest into `$DATA_ROOT/torchcell-library/_bib/`, where
tc-lit serves them at `/bib` and `/bib/<name>`.

```bash
python scripts/lit_bib_store.py              # export every bibliography
python scripts/lit_bib_store.py --list       # print the discovered specs, pull nothing
python scripts/lit_bib_store.py --name paper --name eqtl-data-model
python scripts/lit_bib_store.py --dry-run    # pull + report counts, write nothing
```

Runs on the host that holds the mirror (GilaHyper). Needs `ZOTERO_LIBRARY_ID`,
`ZOTERO_USER_ID`, `ZOTERO_API_KEY`, `DATA_ROOT` in `.env`. Nightly at 03:45 from cron
(`scripts/crontab.txt`), fifteen minutes after [[scripts.lit_sync]] so a paper captured
overnight is citable the same morning. Unlike [[scripts.lit_bib]] it writes nothing
git-tracked, which is why it can run from cron. The `_bib/` directory is bind-mounted
read-only into the `tc-lit-endpoint` container along with the rest of the mirror, so a
new export is served with no restart.

Client side: [[scripts.lit_bib_pull]], or `make bib-pull` in a notes-tex document or
`paper/nature-biotech`.
