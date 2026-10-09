# markdownlint quote canary (#846)

A fixture, not a note. Every line below is a shape the `markdownlint-cli2 --fix`
pre-commit hook used to rewrite in place, which falsifies a verbatim quote. The
hook lints this file on every commit that touches it (see the `files` pattern in
`.pre-commit-config.yaml`), and
`tests/torchcell/test_markdownlint_quote_protection.py` pins these bytes, so a
reintroduced inline autofix fails the test suite instead of editing a note.

## MD034 no-bare-urls: a bare DOI must not become an autolink

Menasalvas et al. 2025 (*Sci Adv* 11, eady2677; doi:10.1126/sciadv.ady2677) with
the deposit at https://doi.org/10.5061/dryad.sbcc2frjq.

## MD037 no-space-in-emphasis: an underscore is not an emphasis marker

> "Our two highest producer strains TEAM-3185 and TEAM-3174 contain gene
> deletions in ΔPP_ 2428, ΔPP_ 4622, ΔPP_ 3540, and ΔPP_4373."

## MD038 no-space-in-code: a padded backtick span keeps its padding

Supplementary Table 4 gives `"Pp TEAM-2777 ΔPP_ 2428 ΔPP_ 4622 ΔPP_ 3540 ΔPP_4373ΔPP_2074"`
and the OCR cell ` ΔPP_ 2428 ` carries its own padding.

## MD027 multiple-spaces-after-blockquote-symbol: a quote keeps its indent

>  "    PP_2428    PP_4622" is how the OCR laid out the table row

## MD039 no-space-in-links: a spaced link label is source bytes

> "see [ Table 4 ](S4) of the Supplementary Material"

## MD011 reversed-link-syntax: a quoted citation is not a broken link

> "see (Table 4)[S4] of the Supplementary Material"

## MD049 emphasis-style and MD050 strong-style: markers are source bytes

> "the _mvaS_ overexpression and the __PP_2074__ deletion"

## MD004 ul-style: a list marker is a source character

- "the released first bullet uses a dash"

+ "and the released second bullet uses a plus"

## MD035 hr-style: a thematic break keeps its characters

---

***

## MD053 link-reference-definitions: a definition is not dead weight

[dryad-deposit]: https://doi.org/10.5061/dryad.sbcc2frjq

## MD010 no-hard-tabs: a tab inside a fence is source bytes

```text
strain	genotype
TEAM-3185	Pp TEAM-2777 ΔPP_ 2428
```
