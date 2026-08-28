---
id: gl1nmtpnm4fzjc9nq6ce1il
title: Zotero_publish
desc: ''
updated: 1787703985063
created: 1787703985063
---

## 2026.08.25 - Generalized past notes-tex, so the manuscript publishes the same way

The script published only `notes-tex/<doc>/main.pdf`. It now takes any
repo-relative document directory and any PDF stem, so the Nature Biotech
manuscript gets the same versioned, comment-anchored review copy the typeset
notes have had.

```bash
make -C paper/nature-biotech publish          # editing.pdf -> torchcell/paper/nature-biotech
make -C paper/nature-biotech publish-dry      # preview
make -C paper/nature-biotech publish-list     # versions so far
make -C paper/nature-biotech publish-submission
```

### The collection path stays derived, which is the whole point

`paper/nature-biotech/editing.pdf` -> `torchcell / paper / nature-biotech`. The
script walks the repo path components and creates what is missing, so there is
still no configured mapping that can drift from where the files actually live.
The `paper` parent already existed and was empty, so only the leaf was created.

**The `torchcell / paper` collection in the PERSONAL library is ours; the `paper`
collection in the GROUP library (`W46ATS7B`) holds the papers the manuscript
cites and is what `zotero_export_bib.py` exports.** Same name, different
libraries, opposite contents. The `microbe-perturb-seq` rename hit exactly this
and nearly renamed the bibliography's source, so the docstring now says it.

### Backward compatibility was the constraint, not a nicety

`doc_key` is the repo-relative directory, and it is what the parent item is found
by. The 19 published versions of the perturb-seq document carry
`Doc Key: notes-tex/024-perturb-seq-costing`, so a bare name with no slash still
resolves to `notes-tex/<name>`. Verified before publishing anything: both
`024-perturb-seq-costing` and `notes-tex/024-perturb-seq-costing` resolve to
collection `AKUUM8DH` and parent `IW46KK3N`, and the filename is unchanged
because the stem suffix is empty for `main`.

### Two things the manuscript needed that a notes-tex document does not

- **`\title` is not in the file being built.** It lives in
  `sections/frontmatter.tex`, and it is the `sn-jnl` two-argument form
  `\title[short]{long}` with a `\cb{14/15 words}` budget macro nested inside. A
  regex cannot take that braced group: non-greedy stops at the inner brace, greedy
  runs to end of file. Extracted with a balanced-brace scan and the `\cb` stripped.
  `--tex` is an explicit flag rather than a search, because searching for `\title`
  in `paper/nature-biotech` finds two files: the real one and the stock
  `sn-article.tex` sample with its placeholder title.
- **Three authors.** The old code built the Zotero creator from
  `author.split()[0]` and `[-1]`, which on "Michael Volk, Aurosish Sharma, Huimin
  Zhao" yields the first author's given name beside the last author's surname.
  Caught in the dry run before the item was created. The parser now returns
  `(first, last)` pairs and one creator is written per author.

### Note on the first publish

v1 of the manuscript was published from this branch before it landed, so its
provenance stamp reads `publish-paper-zotero @ 77c90f25d05e-dirty`. That is
honest rather than wrong, and re-running `make publish` after landing takes the
identical-bytes path and refreshes the stamp to the landed commit without
uploading a second copy.

### 2026.08.25 - Follow-up: the paper build was not reproducible, so dedupe never fired

Publishing the manuscript exposed a gap the notes-tex documents do not have.
`SOURCE_DATE_EPOCH` lives in `notes-tex/common/Makefile.common` and the paper has
its own Makefile, which never set it. hyperref therefore wrote a wall-clock
`/CreationDate` into a compressed object stream and every build of identical
sources produced different bytes.

**Measured, not assumed.** Two rebuilds of untouched sources gave `42522d69...`
and `0c3521b8...`. That is exactly the failure this script's docstring warns
about: versioning is by content hash, so the "already published, identical bytes"
path could never fire and every `make publish` would pile up a fresh
near-identical PDF. It had already happened twice before anyone looked.

Fixed by mirroring the `Makefile.common` block into the paper's Makefile, with
the epoch taken from the newest mtime among `$(SHARED)` and the view wrappers so
`\date{\today}` still reads as the date the document was last edited. Three
forced rebuilds now agree at `b33bdcba...`.

**The first test of the fix was wrong and said it had failed.** Touching a source
to force the rebuild changes the mtime the epoch is derived from, so the PDF
legitimately changes and the test proves nothing. The rebuild has to be forced
without touching any input.

### 2026.08.25 - Publishing into a TRASHED collection, which reads as a sync failure

`find_collection` returned collections that are in the Zotero trash, so
`ensure_collection` reused one instead of creating a live one. The API reports
success, every later lookup resolves, and the document is invisible in the Zotero
client with no error anywhere.

**What it cost.** `torchcell/paper` had been trashed at some earlier point
(`deleted: True` since collection version 63086, well before today). The
manuscript was published into it, `--dry-run` and `--list` both resolved happily,
and the collection reported 0 items, which read as "empty and available" rather
than "deleted". The symptom was "I do not see `paper` under `torchcell`", which
points at the client, and it survived a round of sync-hypothesis debugging before
anyone checked the `deleted` flag. What ruled sync out was the observation that
other PDFs were syncing fine.

Fixed by skipping any collection carrying `deleted` in `find_collection`. The
collection itself was restored with a `PATCH {"deleted": false}`.

**One more is still trashed:** `torchcell/database/torchcell-swanki`
(`DFRSS7SY`). Left alone, since it is deliberate as far as anyone knows, but it
would have hit the same bug had a document ever been named `torchcell-swanki`.

**The general shape, worth remembering.** The Zotero API exposes trashed objects
and the client hides them, so any lookup that does not filter `deleted` sees a
different library than the person does. That is true of items as well as
collections: the same asymmetry is what made three same-titled manuscript items
confusing earlier in the day.

## 2026.08.27 - Documents are named for their directory, not `main`

`notes-tex/<slug>/main.tex` and `main.pdf` became `<slug>.tex` and `<slug>.pdf`
in all four documents. Four files named `main.pdf` are four files a fuzzy
file-opener cannot tell apart, and the built PDF is the one that gets opened
most. `Makefile.common` derives the name with `DOC ?= $(notdir $(CURDIR))`, so a
directory rename carries the document name with it, the same property the Zotero
collection path already had; the per-document Makefiles no longer set `DOC`.

**The one thing that had to not change is the published filename.** Attachments
are named `<doc><stem>_<timestamp>_<sha256[:8]>.pdf` where the stem is empty for
the plain build, and that emptiness used to be spelled `pdf_stem == "main"`. It
now strips either `main` or the document's own name, so every already-published
filename stays byte-identical and a collection's version history remains one
sequence instead of splitting at the rename. The default stem is the document
directory's leaf, and `--clean` means `<doc>-clean`.
