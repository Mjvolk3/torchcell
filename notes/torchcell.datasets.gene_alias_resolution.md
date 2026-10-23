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

## 2026.10.12 - The #983 rule and refusal reasons are reverted (#1045)

`git revert 07c0caf1b` removes the `SGD_SYSTEMATIC_RENAME` rule, `check_pin` and the `retired_systematic_name` / `merged_into_locus` / `not_current_unpinned` refusal reasons with `torchcell/datasets/systematic_name_currency.py`. Owner, 2026-10-12: "costanzo and kuzmin etc should be reversions"; "the data is the data ... the genome is separate.. it doesn't gate anything". Their only users were #983's loaders, its measurement script and Yoshikawa 2009, which now stores every strain as released. The ambiguous-alias rule of #886 (pinned, sourced resolutions for Xue 2025, da Silveira 2014 and Sameith 2015) is unchanged. History: 07c0caf1b, and the reverted section of this note at 901e00104.

A released SYSTEMATIC name that R64 no longer carries has exactly one candidate, so first-match would resolve it whether SGD renamed the ORF or merged it into a longer one. [[torchcell.datasets.systematic_name_currency]] therefore maps such a name only through a pin of the new rule:

- `AliasResolutionRule.SGD_SYSTEMATIC_RENAME`: SGD's locus history of the pinned ORF records the name as its former systematic name, changed for nomenclature. The build re-checks the genome half (`check_pin`): the name's only candidate is the pinned ORF and the name is not a live gene. The history half is checked against the cached locus JSON bytes by the data-gated test.
- `check_pin(genome, pin, paper_gene_list=None)` is the public form of the existing pin check; it returns the pinned ORF.
- New refusal reasons: `retired_systematic_name`, `merged_into_locus` (both pinned and counted by the currency check), and `not_current_unpinned` (a non-current name with neither table entry, which stops the build).

No existing pin or resolution changes.

## 2026.10.12 - A downstream utility; no loader gates on it

Owner, 2026-10-12: "the data is the data!!!! it is what they reported ... we don't drop raw data unless they themselves pull it! ... the genome is separate.. it doesn't gate anything... we do that at modeling time if it is an issue."

The module maps a released name onto the R64 annotation for a DOWNSTREAM caller (an adapter, a modeling step, an audit), where a `GeneNameRefused` is that caller's typed answer. A loader stores each record under the names its release gives and does not call this module to decide whether a record is stored. No record carries a resolution status: comparing a released name with the annotation is a reported verification finding (`canonical_gene_names`, `current_genome_genes`, [[torchcell.verification.report]]), not a field of the data. The functions are unchanged.

- da Silveira 2014 no longer imports the module or takes a genome: it stores its table's own `Systematic Name` / `Standard Name` pair. Measured on the built store: its 127 stored pairs are exactly the 127 released pairs, so no record changes ([[torchcell.datasets.scerevisiae.dasilveira2014]]). Its two `paper_gene_list` pins (`YPK1`, `SLT2`) are gone with the call; the table's own rows are the pairing.
- `test_no_loader_outside_the_sweep_list_imports_the_resolution_modules` pins the modules that still import this module or `systematic_name_currency` at load time: Costanzo 2016, Kuzmin 2020 and Yoshikawa 2009 (held by `fix/audit-yeast-refusals-1045`, which deletes `systematic_name_currency`), Xue 2025 (held by `fix/keep-records-887-sd-zero`), Sameith 2015 (left to the sweep: its GEO titles carry common names only). The set only shrinks; a loader joining it fails the test.
- The full list of loaders that still resolve names at load time, by any route (this module, `SCerevisiaeGenome.resolve_gene_name`, the bacterial `reconcile_locus_tags` and its whole-dataset `require_resolved` threshold), is generated by `experiments/036-dataset-fixes-before-kg-build/scripts/load_time_name_resolution_inventory.py` ([[experiments.036-dataset-fixes-before-kg-build.scripts.load_time_name_resolution_inventory]]): 84 modules on this branch.

## 2026.10.23 - The read-time half: `ReleasedNameResolver`

The 10-12 rule moved resolution out of the loaders and promised it at modeling time; nothing did it, so a symbol-named strain on the released-identifier leaf reached neither a query (the gene-set clause rejects the symbol) nor a gene node (the processor matches exactly). `ReleasedNameResolver(genome)` is that modeling-time step: `resolve(name)` returns `(ORF or None, rule, detail)` with the rule `strict` (a live systematic name, or `resolve_gene_name_strict` with no pins), `sgd_standard_name` (an ambiguous name that exactly one live ORF carries as its SGD standard name, the `AliasResolutionRule.SGD_STANDARD_NAME` rule applied generically rather than through a per-name pin with a source quote, because a deletion collection names its strains by the standard name of the deleted ORF) or `refused` (the typed reason and the candidates). Names are memoized case-folded and `outcomes` counts distinct names by rule. `is_deletion_kind(perturbation_type)` names the two leaves a reader resolves. Consumer: `torchcell.data.graph_processor.Perturbation(released_name_resolver=...)`; the supported queries `amino_acid_betaxanthin` (Mulleder, Cachera blocks) and `expression_proteome_morphology` (Messner block) now admit `released_identifier` without the gene-set test so the records reach the reader at all. Measured on Cachera's 3,592 symbols (046 build): 3,511 strict, 75 standard-name, 6 refused.
