---
id: gpof70tat6ktxy3gtficrau
title: Lit_bib_pull
desc: ''
updated: 1788387044313
created: 1788387044313
---

## 2026.09.02 - Pull a Served Bibliography and Verify It

`scripts/lit_bib_pull.py`. The client half of [[torchcell.literature.bib_store]]: reads
the store manifest from `GET /bib`, downloads `GET /bib/<name>`, and writes the file only
if its sha256 matches both the manifest and the `X-Artifact-SHA256` header. The manifest
is the trust anchor. A file already at the served hash is left untouched and reported as
unchanged. Talks only to tc-lit, never to Zotero, so it runs wherever `TC_LIT_URL` and
`TC_LIT_API_KEY` are set: the Mac, GilaHyper, a collaborator's machine.

```bash
python scripts/lit_bib_pull.py --list
python scripts/lit_bib_pull.py --name paper --out paper/nature-biotech/references.bib
make -C notes-tex/eqtl-data-model bib-pull      # --name $(DOC) --out references.bib
make -C paper/nature-biotech bib-pull           # --name paper
```

`make bib` (Better BibTeX, Zotero desktop required) and `make bib-pull` (tc-lit, headless)
write the same file. They agree on citation keys and differ in field formatting, so diff
the result before committing it. The `/lit-pull` skill documents the raw curl form.
