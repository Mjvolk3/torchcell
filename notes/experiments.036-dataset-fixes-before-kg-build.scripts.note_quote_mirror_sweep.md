---
id: w0j72leftcap0fiqnzjj70w
title: Note_quote_mirror_sweep
desc: ''
updated: 1791584513446
created: 1791584513446
---

## 2026.10.09 - The hook that falsified two quotes, and the sweep that looks for more

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/note_quote_mirror_sweep.py`
Issue: #846. Related: #758 (a hash pin does not make a transcription verbatim).

### What broke

The `markdownlint-cli2 --fix` pre-commit hook rewrote two verbatim Menasalvas 2025
quotes inside `notes/torchcell.datasets.pputida.menasalvas2025.md` on PR #844. MD034
turned the bare DOI `doi:10.1126/sciadv.ady2677` into an autolink and MD037 read the `_`
pairs in Supplementary Table 4's strain string as emphasis markers and deleted the spaces
inside them. The sha256 pin on the SOURCE could not catch either, because the source is
not what changed; only a re-read of the mirror bytes did.

A note carries the same quote a loader pins in `SourcedValue.quote`, so an autofixer that
edits inline text falsifies provenance on every commit that touches such a note, and the
VS Code extension's on-save fix does the same thing outside git.

### Reproduced, then closed

Both shapes reproduce under the pinned hook version. With the old
`.markdownlint.json` and markdownlint-cli2 v0.16.0 (markdownlint v0.36.1):

```text
< the deposit at https://doi.org/10.5061/dryad.sbcc2frjq.
> the deposit at <https://doi.org/10.5061/dryad.sbcc2frjq>.
< > deletions in ΔPP_ 2428, ΔPP_ 4622, ΔPP_ 3540, and ΔPP_4373."
> > deletions in ΔPP_2428, ΔPP_ 4622, ΔPP_3540, and ΔPP_4373."
```

The fix disables every rule whose fix rewrites characters inside a line: MD004, MD011,
MD027, MD034, MD035, MD037, MD038, MD039, MD044, MD049, MD050, MD053, plus
`MD010: {code_blocks: false}` so a tab inside a fenced block stays a tab. The rule table
moved from `.markdownlint.json` to `.markdownlint.jsonc` so the reasoning sits beside the
rules, and both autofix paths read that one file.

**Check-only mode for `notes/` was measured and rejected.** The tree carries 7,580 MD009,
5,626 MD010 and 688 MD012 violations in notes the hook has never seen, because the hook
only lints STAGED files. A reporting-only hook would block every commit that touches one
of those notes on unrelated pre-existing whitespace.

**What `--fix` still does, measured over all 1,546 notes.** Running `--fix` on a copy of
the whole tree: the old config changed a non-whitespace character in 8 notes, the new one
in 0. The 38 notes the new config still touches get only a blank line added or removed
around a heading, list, fence or table, and none of those insertions lands inside a
blockquote. The rules left on move whitespace at a line's end and between blocks, and a
note joins a wrapped quote's lines with one space, so that is layout and not quoted text.

### The canary

`tests/torchcell/data/markdownlint_canary/quote_canary.md` holds one shape per disabled
rule and is inside the hook's `files` pattern, so the hook lints it on every commit that
touches it. `tests/torchcell/test_markdownlint_quote_protection.py` pins the disable list
and the canary's bytes, hermetically (no node, no network): if a rule comes back, the hook
rewrites the canary and the suite fails instead of a note being edited. Measured with the
real binary: the old config rewrites 9 of the canary's shapes, the new config reports 0
errors and leaves the file byte-identical.

### The sweep, and what it found

The sweep collects every `SourcedValue` reachable from a module under
`torchcell/datasets/`, resolves its artifact in both mirror roots
(`$DATA_ROOT/torchcell-library` for the OCR'd paper and SI,
`$DATA_ROOT/torchcell-raw` for a released data file), re-hashes it against the pinned
sha256, confirms the quote is still in those bytes, then searches all of `notes/` for the
same text. Result file:
`experiments/036-dataset-fixes-before-kg-build/results/note_quote_mirror_sweep.json`.

**Totals** (2026.10.09, after the two restores below):

| Measurement | Count |
|---|---|
| Unique `(module, key, uri, quote)` tuples checked | 1,695 |
| Loader modules they come from | 112 |
| Notes searched | 1,546 |
| Loader side: pinned sha256 matches AND quote present | 1,485 |
| Loader side: sha256 drift | 0 |
| Loader side: quote not a substring of the pinned artifact | 205 |
| Loader side: artifact not in either mirror root | 5 |
| Note side: at least one note holds the quote verbatim | 513 |
| Note side: no note repeats the quote | 1,048 |
| Note side: a near rendition the mirror DOES contain | 20 |
| Note side: the note's own prose about the quote (paraphrase) | 192 |
| **Note side: a rendition present in NO mirror artifact** | **32** |

**Attribution of the 32, which is the point of the exercise:**

| Signature | Count | What it is |
|---|---|---|
| `markdownlint_autolink` | 0 | MD034 wrapped a bare URL in `<>` |
| `markdownlint_emphasis_space` | 0 | MD037 or MD038 deleted a space beside `_` or `*` |
| `markdownlint_list_marker` | 0 | MD004 rewrote a list marker |
| `ocr_math_markup` | 16 | the note resolved MinerU LaTeX (`$1 . 5 \%$` to `1.5%`) |
| `typography_fold` | 8 | a curly quote or an en dash folded to ASCII |
| `glyph_transliteration` | 4 | `ΔPP_2675` written `dPP_2675` or `DeltaPP_2675` |
| `authored_markup` | 1 | a `**` the note author put inside a quoted span |
| `quotation_delimiter` | 2 | a `" "` join artifact inside a quoted span |
| `other` | 1 | backticks splitting one quoted span into several |

**No remaining falsification is attributable to the hook.** Every one of the three
markdownlint signatures is zero, which is the measured answer to the second half of #846:
the two quotes the fixer did rewrite were restored in PR #844, and the disabled rules plus
the canary stop a third. Detection was validated rather than assumed: re-injecting the
#846 edit into `notes/torchcell.datasets.pputida.menasalvas2025.md` and rerunning the
sweep over that one module reports it, classified `markdownlint_emphasis_space`, with the
mirror's bytes printed beside it:

```text
FALSIFIED note renditions: 1
by signature: {'markdownlint_emphasis_space': 1, ...}
  [markdownlint_emphasis_space] notes/torchcell.datasets.pputida.menasalvas2025.md
    note   : Pp TEAM-2777 ΔPP_2428 ΔPP_ 4622 ΔPP_3540 ΔPP_4373ΔPP_2074
    mirror : Pp TEAM-2777 ΔPP_ 2428 ΔPP_ 4622 ΔPP_ 3540 ΔPP_4373ΔPP_2074
```

**Two restored, both `other`, neither the hook's doing:**

- `notes/torchcell.datasets.ecoli.ishii2007.md` wrote the workbook's `"Exch"` banner as
  `\"Exch\"`, two backslashes `data/Quantitative_data.xls` does not carry.
- `notes/torchcell.datasets.ecoli.schastnaya2021.md` carried the SD3 banner with a ` / `
  separator `si/si3.md` does not print.

Both are restored to the mirror's bytes, which is why the verbatim count above reads 513
and the pre-restore run read 511.

### The worklist this leaves, and why it is not an automatic rewrite

The 30 remaining findings are human transcription choices, not autofixer damage, and
three of the classes are a note-wide convention rather than a one-line slip:

- **`ocr_math_markup` (16).** MinerU writes a formula as LaTeX with its characters spaced
  out, so the mirror reads `$1 . 5 \%$` where the note reads `1.5%`. Restoring the
  mirror's bytes would make the note unreadable, and the note states the reading.
- **`typography_fold` (8) and `quotation_delimiter` (2).** A curly quote, an en dash, or a
  `" "` join artifact. Mechanical, but the repo's own rule is American spelling and
  straight quotes in notes, so these two pull against each other.
- **`glyph_transliteration` (4).** `ΔPP_2675` written `dPP_2675` or `DeltaPP_2675`,
  applied consistently through the P. putida notes (PR #844's own commit message does
  it). Flipping one instance would make a note internally inconsistent.
- **`authored_markup` (1) and `other` (1).** A `**` and a set of backticks an author put
  inside a quoted span. The 12-panel note bolds several numbers inside the same quote, so
  this is a convention too.

Deciding whether a note quotes the mirror's OCR bytes or a cleaned reading is an owner
call, so the sweep reports them and the JSON carries each one's mirror window ready for a
restore. Rerunning the sweep after any such decision shows the count move.

### What the sweep's matcher had to learn, each from a whole-tree run

- **Character-level alignment, not word-level.** The #846 shape deletes a space, which
  merges two tokens: `ΔPP_ 2428` to `ΔPP_2428` scored 0.50 word-level and was MISSED,
  against 0.98 character-level.
- **A rendition must also differ by at most 20 characters.** Without that bound, 192 of
  233 reported renditions were the note's own framing clause aligning across the quote
  ("Supplementary Table 4's own banner says what it is: ...").
- **A candidate is checked against EVERY pinned artifact before it is called falsified.**
  Papers from one lab share Methods boilerplate: the Carruthers 2025 note's Top3 sentence
  aligns at 0.98 to the Lim 2025 quote of the same sentence and is verbatim in the
  Carruthers paper. Four mirrored papers carry that sentence.
- **MinerU's LaTeX is tested before the emphasis-space shape.** Resolving `$3 3 0 \ E .$`
  to `$330\ E.$` also deletes spaces, and was charged to MD037 until the classifier
  checked whether the difference sits inside a `$...$` span.
- **The anchor set gates the scan.** A quote present verbatim has all of its own long
  words in the note, so a note missing half of them needs no substring search. Running
  the substring search first instead cost 1,695 x 1,546 scans of a 20 kB note.

### Open, not closed

- **205 loader quotes are not substrings of their own pinned artifact.** The sha256
  matches in every case, so this is transcription on the LOADER side, not source drift:
  a quote that joins OCR fragments across a table rowspan or a page break. Pre-existing
  and outside #846; the JSON's `unauditable` list names each one.
- **5 quotes name an artifact in neither mirror root.** Also pre-existing.
- **The sweep reads `SourcedValue.quote` only.** A `retrieval_command` or a
  `*_statement` field is not covered, so the DOI inside Menasalvas's `DRYAD_MANUAL_RECIPE`
  is outside its reach; PR #844 protected that one by fencing it.
