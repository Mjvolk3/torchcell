---
id: 2z0cr4jfool8hfplws95j4v
title: strain_tables
desc: ''
updated: 1791256114766
created: 1791256114766
---

## 2026.10.04 - Why the Strain Table Is a Script and Not a Table

The twelve strains on hand are described across two documents in the source paper, and
every statistic the selection depends on is a value someone read out of a table or a
sentence. Hand-authoring that into LaTeX would make the numbers unauditable the moment
anyone asked where one came from, which is the failure the repo's provenance rule exists
to prevent. So the records are the artifact and the tables are output.

Each `Strain` carries the Supplementary Table row or main-text sentence it was read from
plus a verbatim quote, and each `Titer` records whether its number is `stated` in the
paper or `derived` by dividing a fold change. That second distinction is the reason this
is typed rather than prose: four of the twelve have no absolute titer anywhere in the
paper, only a ratio against another strain, and an untyped table would have rendered
those identically to measured values.

### Design notes

- `Evidence` is an enum rather than a bool because `not reported` is a third state, and
  it is not the same claim as `derived`. Collapsing them would assert that a strain has
  no measurement when the truth is that the paper gives a ratio.
- `Titer.fold` exists for the one strain whose only quantitative statement is a ratio and
  whose comparator also lacks an absolute, so there is nothing to divide into. Without it
  that strain rendered as `not reported`, discarding the only number the paper offers.
- `tex_escape_underscore` escapes at the render boundary so a record can hold a bacterial
  gene name as the paper writes it. `Ll_ilvD` reaching LaTeX unescaped is a math subscript
  in text mode and aborts the build.
- `titer_legend` emits a key only for the symbols a given table actually uses, so the
  genotype table carries none.

## 2026.10.05 - Verifying Against the Mirror, Not Just the Local Copy

`--verify` only ever proved that the local PDF still hashes to what was read. The document
claims more than that: it claims the values trace to hash-pinned artifacts in the
literature mirror. `--verify-mirror` makes that claim checkable by asking tc-lit what it
holds under the citation key and comparing.

An unreachable mirror is treated as a failure rather than a skip. A silent skip would let
the claim go unchecked in exactly the circumstance where it has stopped being true.

### It immediately caught a real mismatch

The first run reported DIVERGED for the Supplementary Information. The mirror numbers its
SI files in its own order, which is not the publisher's MOESM order:

| mirror path | bytes | actually is |
|---|---|---|
| `si/si1.pdf` | 3,545,488 | MOESM3, the Peer Review File |
| `si/si2.pdf` | 375,128 | MOESM2 |
| `si/si3.pdf` | 7,881,989 | MOESM1, the Supplementary Information |

The strain and plasmid tables are in `si/si3.md` for this key. Anyone pulling `si/si1.md`
expecting the Supplementary Information gets reviewer comments. The pin now names
`si/si3.pdf` and carries a comment explaining why, so the next reader does not re-derive it.
