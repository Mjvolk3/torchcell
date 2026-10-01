---
id: fdmhxx654nd86ymm7jfngal
title: Bib_store
desc: ''
updated: 1788387029529
created: 1788387029529
---

## 2026.09.02 - The Bibliography Store: Named `.bib` Artifacts Served by tc-lit

`torchcell/literature/bib_store.py`. Companion of [[torchcell.literature.bib]] (the
Zotero Web API pull it reuses) and [[torchcell.literature.server]] (the `/bib` routes).
Motivation and the three pre-existing bib flows: [[tectonic-builds-and-bibliographies]].

### What it is

A bibliography treated as a mirror artifact. A host-side export
([[scripts.lit_bib_store]]) pulls each named scope over the Zotero Web API, writes
`<DATA_ROOT>/torchcell-library/_bib/<name>.bib` beside a `manifest.json`
(`BibStoreManifest` of `BibRecord`s: name, bytes, sha256, entry count, scope, origin,
`generated_at`), and tc-lit serves the directory read-through: `GET /bib` returns the
manifest, `GET /bib/<name>` streams the file with `X-Artifact-SHA256`. The client
([[scripts.lit_bib_pull]]) verifies the bytes against the manifest before writing.

### Names are discovered from the repo, not configured

`discover_bib_specs(project_root)` reads each scope from where it is already declared:

| name | scope | declared in |
|---|---|---|
| `paper` | group `paper` collection `W46ATS7B` only | `paper/nature-biotech/zotero_export_bib.py` (`DEFAULT_COLLECTION`) |
| `<notes-tex slug>` | that document's `ZOTERO_COLLECTION` paired with its `ZOTERO_PERSONAL_COLLECTION` | `notes-tex/<slug>/Makefile` |
| `library` | group library unioned with the personal `torchcell` tree | `scripts/lit_bib.py` defaults |

A notes-tex document with no `ZOTERO_COLLECTION` (`database-expansion-100`,
`w019-strain-build-list`) has no bibliography and gets no entry. The notes-tex name is the
directory name, so `make bib-pull` asks for `$(DOC)` and the join key is the same one
`zotero_publish.py` derives the Zotero output collection from.

### Invariants

- **Content-stable bytes.** The generated-file banner names the scope and entry count
  but carries no timestamp, so an unchanged Zotero collection re-exports to an identical
  sha256 (tested). `generated_at` lives only in the manifest.
- **Atomic swap.** Every spec is pulled and written to `<name>.bib.part` first; the
  served files and the manifest are renamed into place only after every pull
  succeeded. A 0-entry pull raises (never a blank bibliography), and a failure on the
  second spec leaves the first spec's previous file in place (both tested).
- **Names are path segments.** `validate_bib_name` rejects `/`, `..`, and a leading
  dot, so `/bib/{name}` can only address a file the manifest lists.
- **Service directories are not citation keys.** `_bib`, like `_sync_reports`, is
  underscore-prefixed and the server now skips such directories in `/keys` and
  `/health`. Before this change `_sync_reports` was counted as a key.

### Scope precedence, and one known divergence from the BBT route

The paired notes-tex scope reuses `fetch_paired_collection_entries`, where **personal
wins** on a shared key. `notes-tex/common/build_bib.py` (the Better BibTeX route) takes
the opposite precedence, **group wins**, and additionally reports key conflicts and
duplicate works. Keys are Zotero 7 native `citationKey`s on both routes, so the same
work resolves under the same key either way; the field formatting differs (Zotero's
translator here, BBT's export there), which is why `make bib-pull` says to diff before
committing.

### First export (2026.09.02, GilaHyper, `scripts/lit_bib_store.py`)

| name | entries |
|---|---|
| `paper` | 24 |
| `024-perturb-seq-costing` | 42 |
| `eqtl-data-model` | 2 |
| `library` | 615 (101 group, 514 from the personal tree) |

Tests: `tests/torchcell/literature/test_bib_store.py` (12; Zotero is monkeypatched, so
the export logic, the atomic swap, and the endpoint's hash header are what is exercised).

The first export differs from every committed `references.bib`, in each case because the
committed file is a snapshot of earlier curation and the collection has since changed.
Key-by-key comparison and what each collection holds today:
[[tectonic-builds-and-bibliographies]], "State of the committed bibliographies vs Zotero".

## 2026.10.01 - Keys by declaration, staging cleanup, retired files, Makefile comments (issue #529)

Previous behavior, pinned as Findings by Phase 14 of the test campaign:

- `fetch_scope_entries` decided key-or-name with `re.fullmatch(r"[A-Z0-9]{8}", ...)`, so a collection NAMED `RNASEQ01` was sent as a key;
- a failed export left the earlier specs' `.part` files in `_bib/`;
- a spec removed from the repo stopped being listed and served, but its `.bib` stayed in `_bib/` looking served;
- the Makefile pattern ended `(\S*)\s*$`, so `ZOTERO_COLLECTION := VNDH4NMX  # eQTL` read as empty and the document silently got no bibliography.

Fix:

- Every declaration in the repo is a collection KEY (`PAPER_COLLECTION_KEY`; the Makefile values `build_bib.py` hands Better BibTeX), so `BibScope.group_collection` and `user_collection` are keys by declaration. A field validator refuses any other value with `not a Zotero collection key (8 upper-case letters or digits): '<value>'; ...`. The regex only validates a declared key; it never decides. `fetch_scope_entries` always sends `collection_key`, and the paired pull passes `as_keys=True` (new keyword on [[torchcell.literature.bib]] `fetch_paired_collection_entries`). The served manifest and the repo's five specs load unchanged under the validator (checked read-only).
- A failed export unlinks every `<name>.bib.part` it staged, then re-raises.
- After a successful export, every `*.bib` or `*.bib.part` the new manifest does not serve is moved to `_bib/_retired/<generated_at>/`, with a warning naming each; nothing is deleted. This also moves the specs left out of a `--name` subset run, which the wholesale manifest replacement already stopped serving.
- `parse_makefile_collections` cuts a trailing `# comment` off the value, as make does, and refuses a value of more than one word with `ValueError` naming the Makefile and variable.

Issue #563 (nightly export failing since 2026-09-21 with `Zotero collection 'torchcell' not found`): not fixed here, but explained. `/tmp/torchcell-lit-bib-store.log` lists exactly 100 available collections (alphabetical through `w019-strain-build-list`, with `torchcell-topics` present and `torchcell` absent). `ZoteroLibrary.collection_key` calls `self.zot.collections()` without `everything(...)`, so it reads one 100-collection page; the personal library now has more than 100 collections and `torchcell` falls off that page. `collection_tree` pages correctly and then resolves its root through the unpaged `collection_key`. Hypothesis (consistent with the log, not tested against Zotero): paging `list_collections` fixes #563.

Evidence: `test_scope_collections_are_keys_by_declaration`, `test_empty_pull_message_and_no_part_file_left_by_a_failed_export`, `test_partial_failure_leaves_previous_store_intact`, `test_dropped_spec_is_unserved_and_its_file_is_moved_aside`, `test_inline_comment_after_the_value_is_cut_off`, `test_a_two_word_makefile_value_is_refused_by_name` in [[tests.torchcell.literature.test_bib_store]].
