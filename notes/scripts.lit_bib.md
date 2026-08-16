---
id: u5qqhc8ak4tq6w7turfodgr
title: Lit_bib
desc: ''
updated: 1786915857149
created: 1786915857149
---

## 2026.08.16 - CLI for the Zotero bibliography pull

Thin wrapper over [[torchcell.literature.bib]]. Pulls once from Zotero and merges into
every managed bibliography (`notes/assets/bib/bib.bib` and its publish twin) so the two
cannot drift apart the way hand-editing let them.

```bash
python scripts/lit_bib.py                                    # add new Zotero papers
python scripts/lit_bib.py --dry-run                          # report the delta, write nothing
python scripts/lit_bib.py --collection microbe-perturb-seq   # one collection only
python scripts/lit_bib.py --update-existing                  # also refresh shared entries
python scripts/lit_bib.py --verbose                          # list every added/updated key
```

Runs headless over the Zotero **Web API** — no Zotero desktop, no Better BibTeX. Exits
non-zero without touching a file if Zotero returns 0 entries (bad credentials or a
mistyped `--collection` can never blank the bibliography).

Rationale for the add-only default, the attachment-stub filter, and the invalid-key
sanitizer is in [[torchcell.literature.bib]]. Companion: [[scripts.lit_sync]] mirrors +
OCRs the PDFs; this makes the same papers citable.
