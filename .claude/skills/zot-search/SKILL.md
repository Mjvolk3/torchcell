---
name: zot-search
description: Search the whole personal Zotero library (OCR'd papers, books, supplements, web snapshots, highlights, comments, notes) through the zotlib txtai index on GilaHyper, with Claude as the reranker. Use when asked for papers, passages, quotes, figures, tables or highlights about a topic, scoped to the torchcell collection or the whole library. No API token is needed; the listwise rerank happens in-session.
---

# zot-search

Retrieval over the personal Zotero mirror at `/bulk/zotero-library` (built by the
thesis repo's `zotlib.backfill`) through the hybrid txtai index `_index/bge-large`
(bge-large-en-v1.5 + BM25, SQLite content). Two stages, exactly as iBioFoundry-AI's
protocol RAG: a bi-encoder stage fills a pool of distinct documents, then a listwise
rerank reorders the pool. Here the reranker is YOU, reading the candidates, so no LLM
API key is involved.

This is the torchcell-side entry point. zotlib is not a torchcell dependency:
`scripts/zot_search.sh` puts the thesis checkout on `PYTHONPATH`, runs its conda env,
and picks the GPU with the least used memory. GilaHyper only; the index needs the
query model resident, so it cannot be copied to a laptop (the Mac reads papers through
`/lit-pull` instead, which serves torchcell's own OCR mirror, a different tree).

## Step 1: retrieve a wide pool

Run from the repo root (any torchcell checkout or worktree on GilaHyper):

```bash
bash scripts/zot_search.sh "<query>" -k 40 --json [--scope KEY] [--kind chunk|table|figure|abstract|annotation|note] \
    [--color green] [--tag TAG] [--item-type book] [--year-min 2015] [--all-chunks]
```

- `-k 40` is the pool (iBioFoundry uses 100; 40 is enough to read). Each hit is one
  JSON line on stdout: `score, citation_key, kind, section, page_start, page_end,
  source, source_title, title, year, item_type, color_name, comment, chunk_text`.
  Model-loading chatter goes to stderr; a call takes about 25 s.
- Results are collapsed to the best record per item. Add `--all-chunks` when the user
  wants every passage of a few papers rather than breadth.
- **Scope** is a Zotero collection key, repeatable; sub-collections are included.
  No scope = whole library (2,267 items at index version 71743). torchcell keys in
  the personal library:

  | key | collection |
  |---|---|
  | `ICDCVSL6` | `torchcell` (everything below) |
  | `85U7PFYV` | `torchcell/paper` |
  | `JHLW95L2` | `torchcell/torchcell-topics` |
  | `I5JD25ER` | `torchcell/notes-tex` |
  | `3UNNFGTU` | `torchcell/database` |
  | `EHK6T28Z` | `torchcell/torchcell-queue` |

  These are personal-library keys. The GROUP library's `paper` collection
  (`W46ATS7B`, what `zotero_export_bib.py` exports) is a different library and is not
  in this index. Other keys: `/bulk/zotero-library/_state/snapshot.json` ->
  `collections` (`path` is the human name). Thesis scope is `CXY88NAM`.
- Phrase the query as the passage would be written ("heterologous pathway expression
  imposes a metabolic burden on yeast"), not as a question. Run two or three
  phrasings when the first pool looks thin; union the pools before reranking.
- The script picks the card with the least used memory; set `CUDA_VISIBLE_DEVICES`
  to override. If no checkout with `zotlib/` is found it exits 2 and says so; set
  `ZOTLIB_ROOT` to the thesis checkout that has it.

## Step 2: rerank in-session

Read the pool and order it by usefulness FOR THE USER'S ASK, not by score: a chunk that
states the mechanism beats one that mentions the term. Drop candidates that only match
on vocabulary. Prefer primary sources over reviews unless a review is asked for, and
note when a hit comes from a supplement, web snapshot or your own notes (`source` is
not `paper`). Keep the bi-encoder order among candidates you cannot distinguish.

## Step 3: quote from the mirror, never from the snippet alone

The snippet is a chunk; the quote must be verbatim from the source markdown:

```bash
grep -n -i "<distinctive phrase from the chunk>" /bulk/zotero-library/<citation_key>/paper.md
sed -n '<line-5>,<line+5>p' /bulk/zotero-library/<citation_key>/paper.md
```

Supplements are `si/<attachmentKey>.md`, web snapshots `extras/<attachmentKey>.md`,
highlights and comments `annotations.json` (fields `text`, `comment`, `color_name`,
`page_label`, `tags`). MinerU markdown can carry OCR artifacts (broken hyphenation,
`$...$` around plain numbers); quote as printed, mark any repair with brackets. Give the
page as `page_start` + 1 (content lists are 0-based) and say so once.

**Two mirrors, do not mix them.** `/bulk/zotero-library/<key>/paper.md` is zotlib's
OCR of the personal library. `$DATA_ROOT/torchcell-library/<key>/paper.md` is
torchcell's own OCR mirror, the one `tc-lit` serves and the one a loader's
`SourcedValue.quote` + sha256 must be pinned to. A quote found here that is going into
a dataset loader or a provenance record must be re-located and re-hashed in the
torchcell mirror; the same paper can OCR differently in the two pipelines.

## Step 4: answer

For each quote: the sentence(s) in quotation marks, then citation key, year, section and
page on one line. Group by theme if the user asked a question rather than for a list.
State what was NOT found (an empty pool under a scope usually means the collection is
empty or the paper is not in Zotero). End with the absolute paths of the markdown files
you quoted, one per line, last.

## Keeping it honest

- The index has a build stamp (`/bulk/zotero-library/_index/bge-large/stamp.json`,
  `library_version`). If the user added papers after that version, say the index may
  predate them; the thesis repo's nightly sync refreshes it.
- Nothing here writes to Zotero or to either mirror.
- Design and status live in the thesis repo:
  `notes/plan.zotlib-library-index.2026.10.07.md` on its `library-index` branch.
