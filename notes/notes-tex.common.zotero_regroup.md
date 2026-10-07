---
id: 82h5ga1d55a4e3tnf3sdaqh
title: Zotero_regroup
desc: ''
updated: 1791353170502
created: 1791353170502
---

## 2026.10.07 - One-shot move of the flat Zotero collections under their groups

`notes-tex/common/zotero_regroup.py` exists because `zotero_publish.py` derives a
document's Zotero collection from its repo path. When `notes-tex/` gained the group
layer (`notes-tex/<group>/<slug>/`, see [[notes-tex.common.zotero_publish]] and
`notes-tex/README.md`), every collection already published sat flat at
`torchcell/notes-tex/<slug>` with a `Doc Key: notes-tex/<slug>` marker on its parent
item. Publishing a moved document would have looked for `notes-tex/<group>/<slug>`,
found nothing, and started a second history beside the first.

What it does, per live collection directly under `torchcell/notes-tex` whose name is a
known slug:

- re-parents the collection under `torchcell/notes-tex/<group>` (created if missing);
- rewrites `Doc Key: notes-tex/<slug>` on each top-level item to the grouped path; a
  Doc Key that is not the flat path (the rendered Dendron notes published from the
  kinetics branch carry `notes/<fname>.md`) is kept as it is;
- files each item into the group collection as well, so the group reads as an index
  the way the `notes-tex` index already does.

The grouping comes from the tree (`notes_tex_groups` reads `notes-tex/<group>/<slug>/`)
plus `--assign SLUG=GROUP` for a document that exists only in an unlanded worktree. A
flat collection neither names is reported and left alone. Nothing is deleted, no
attachment is touched, `--dry-run` writes nothing, and the script is idempotent.

The publisher's `refuse_flat_legacy` is the other half: a document whose collection
still sits flat is refused with the exact `--assign` to run, never given a second
collection.

Tests: `tests/torchcell/literature/test_notes_tex_groups.py` (plan, dry run, Doc Key
rewrite, the resolver, the refusal), against the read-only `FakeZot`, so the write
path is never imitated.

The run that moved the library is recorded under the same date in
[[notes-tex.common.zotero_publish]].
