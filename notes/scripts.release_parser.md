---
id: haco9vkn7cif3x32gxxo2xb
title: Release_parser
desc: ''
updated: 1790716461382
created: 1790716461382
---

## 2026.09.29 - Why semantic-release needs its own parser

Finding (measured on python-semantic-release 10.4.1, the installed version): the `scipy`
parser the repo used accepts `TAG: subject` and `TAG: scope: subject` only; its prefix
regex puts a colon after the tag, so the `TAG(scope): subject` form most commits carry
does not parse. On the last 30 subjects of `main` (2026-09-29, head `901d5ec31`), 23 did
not parse, among them every `FIX(...)` commit, so adding `FIX` to `patch_tags` alone
would have released nothing. The `conventional` parser reads `TAG(scope)!: subject` but
has no `major_tags`; a major bump comes only from `!` or a `BREAKING CHANGE:` paragraph
(both measured).

`scripts/release_parser.py` is the conventional parser plus `major_tags`, loaded by file
path (`commit_parser = "scripts/release_parser.py:TorchcellCommitParser"`;
`semantic_release.helpers.dynamic_import` resolves `path.py:Class` against the working
directory). Levels: `API` major; `FEAT ENH DEP DEV REV` minor; `FIX BUG BLD MAINT PERF`
patch; `DOC DOCS NOTE TST TEST STY CI REL BENCH` allowed, no bump; `!` and `BREAKING
CHANGE:` still force a major. `tests/scripts/test_release_parser.py` loads the parser the
way the config names it and pins the 30 real subjects (8 patch, 20 no release, 2 unparsed:
the `1.2.1` bump commit and `fig(008)`). Consequence: the next push to `main` with this
config releases 1.2.2, because five `FIX` commits, `PERF(ops)` and `MAINT(ci)` sit above
v1.2.1. Record: [[versioning]].
