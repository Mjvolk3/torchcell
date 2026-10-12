---
id: 06160wrl1cf2gqb04kidmvx
title: Gene_alias_resolution
desc: ''
updated: 1791681833025
created: 1791681833025
---

## 2026.10.10 - Refusing resolution of ambiguous gene names (issue #886)

Shared by the yeast loaders that read common gene names: `xue2025`, `dasilveira2014`, `sameith2015`. Test file: `tests/torchcell/datasets/test_gene_alias_resolution.py`.

A name's candidate ORFs are the union of `genome.alias_to_systematic[name]` and the live genes whose R64 GFF `gene=` attribute (the SGD standard name) is that name. The union matters: a standard name that another gene also lists as an alias is ambiguous even when its owner does not repeat it in its own `Alias` list.

| Source name | Outcome |
|---|---|
| live systematic id (any case, padded) | itself |
| exactly one candidate ORF | that ORF |
| several candidates, a `PinnedAliasResolution` in the loader whose rule holds | the pinned ORF |
| several candidates, no pin | `GeneNameRefused`, reason `ambiguous_alias_unpinned` |
| pinned ORF not a candidate | `GeneNameRefused`, reason `pin_not_a_candidate` |
| pin's rule does not hold | `GeneNameRefused`, reason `pin_rule_contradicted` |
| no candidate at all | `GeneNameRefused`, reason `not_in_genome` |

Rules (`AliasResolutionRule`), each re-checked at build time:

- `sgd_standard_name`: the injected genome's standard-name owners of the alias are exactly `[pinned ORF]`.
- `paper_gene_list`: the paper's own released list, read by the loader at build time, pairs the alias with the pinned ORF.

A pin carries the alias, the ORF, the rule, a `Provenance` with a required `sha256`, and a verbatim quote that must contain both the alias and the ORF (case-insensitive).

`check_ambiguous_aliases(genome, pairs, pins, paper_gene_list)` is the build-time check. Each loader passes the (source name, stored systematic name) pairs it is about to write; any ambiguous source name whose stored ORF is not its pin's ORF (`stored_orf_differs_from_pin`), or that has no pin, stops the build before the store is opened. It returns an `AmbiguousAliasAudit` (pairs checked, ambiguous name -> stored ORF), which the loaders log.

The module lives at `torchcell/datasets/` rather than under `scerevisiae/` because `tests/torchcell/datasets/scerevisiae/test_raw_pins.py` refuses any `sha256`-named attribute in a `scerevisiae` module that does not verify raw files at build time, and the pin validator reads `provenance.sha256`.

Pins declared on 2026.10.10:

| Loader | Alias | Candidates in R64-4-1 | Stored ORF | Rule | Quote |
|---|---|---|---|---|---|
| `xue2025` | TFC7 | YNL039W, YOR110W | YOR110W | `sgd_standard_name` | `ID=YOR110W;Name=YOR110W;gene=TFC7;` |
| `dasilveira2014` | YPK1 | YJL093C, YKL126W, YNL307C | YKL126W | `paper_gene_list` | `YKL126W \| YPK1` (Table S4 Quant row) |
| `dasilveira2014` | SLT2 | YAL014C, YHR030C | YHR030C | `paper_gene_list` | `YHR030C \| SLT2` (Table S4 Quant row) |
| `sameith2015` | SUT2 | YMR080C, YPR009W | YPR009W | `sgd_standard_name` | `ID=YPR009W;Name=YPR009W;gene=SUT2;` |
| `sameith2015` | GAT1 | YFL021W, YKR067W | YFL021W | `sgd_standard_name` | `ID=YFL021W;Name=YFL021W;gene=GAT1;` |

GFF quotes are substrings of `saccharomyces_cerevisiae_R64-4-1_20230830.gff` (sha256 `64f61e3153083a8ef6d853721c9e83e4469cdc120883ec281e51a0df4ba390fa`, the `R64_GFF` provenance in `torchcell/datamodels/strain_background.py`); the Table S4 quotes are the `Systematic Name | Standard Name` cells of `TableS4_complete_dataset_all_lipids.xlsx` (sha256 `91409229756c132823e6e7a8dbe552d4d7451833b2ff902740f24a29bced3894`). `test_every_loader_pin_matches_its_source_bytes` re-reads both files when `DATA_ROOT` holds them and checks every quote and hash.
