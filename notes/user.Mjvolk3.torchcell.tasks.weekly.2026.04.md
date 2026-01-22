---
id: 70iqseflyde7ujdmn2kz8l2
title: '04'
desc: ''
updated: 1769046069253
created: 1768841844455
---

## 2026.01.19

- [x] worktree merge, delete worktree, and branch
- [x] Created plan updating expression data, but did not have the worktree startup scripts so just committed plan to restart with fresh worktree startup. @Mjvolk3.torchcell.tasks.weekly.2026.04.expression-schema.wip

## 2026.01.20

- [x] Unified all `*.code-workspace` files by moving shared tasks/launch/settings to `.vscode/` (single source of truth); workspace files now only contain environment-specific config (folders, titlebar colors, mypy paths for different clusters)
- [x] Changes to SE as implications for `Deduplicator` changes [[2026.01.20 - Deduplicator as it Relates to Schema|user.mjvolk3.torchcell.tasks.future#20260120---deduplicator-as-it-relates-to-schema]]
- [x] Created SOP for extracting dataset metadata from papers with traceable citations and designed future pydantic-based evidence model [[user.Mjvolk3.torchcell.tasks.weekly.2026.04.nlp-data-enhancement]]

## 2026.01.21

- [x] Implemented [[Mermaid diagram generator|torchcell.ontology.mermaid_diagram]] for BioCypher schema visualization, mostly for exploration for now but can help tighten mapping.
- [x] Generated [[horizontal|torchcell.ontology.mermaid_diagram.horizontal]] and [[vertical|torchcell.ontology.mermaid_diagram.vertical]] Mermaid diagrams
- [x] Updated [[schema config|biocypher.config.torchcell_schema_config.yaml]] - changed experiments from `activity` to `information content entity`, dataset membership to `part of`
- [x] Documented [[implementation plan|torchcell.ontology.mermaid_diagram.wip]] and deferred Phase 2 validation
- [x] We have an issue in mapping to ontology but it can be delayed... Need more expertise before we move on this [[biocypher.config.torchcell_schema_config.yaml#20260121---flip-flopping-on-mapping]]
- [ ] [[user.Mjvolk3.torchcell.tasks.weekly.2026.04.expression-schema.wip]]
- [ ] [[fitness-interaction-n_samples|user.Mjvolk3.torchcell.tasks.weekly.2026.03.fitness-interaction-n_samples]]

- [ ] Add n to fitness interaction data
- [ ] Add examples or clarify the schema design to avoid ambiguity in property definitions.

- [ ] Do we have robust way of handling labels? Did we do by some other method?  [[Generate_calmorph_labels|scripts.generate_calmorph_labels]]

- [ ] Build db

***

Review

## 2025.12.17

- [ ] Add a check to the final gene list to see if we have any genes overlapped with the double gene expression #GH

- [ ] Kemmeren, Sameith dataset verify metadata #M1

- [ ] Follow up on the dataset outlier comparison by reporting the spearman at snapshot for very best model across the different scenarios. → From quick comparison it looks like spearman for datasets with more data are still higher. Test datasets are obviously not exactly the same. →

- [ ] Expression datasets write adapters #M1

- [ ] Start DB build over night #GH

## 2025.12.20

- [x] Is expression data from SGD web browser available in the gene json? → It is not it comes from SPELL
- [ ] Are there images associated with [yeast-gfp data](https://yeastgfp.yeastgenome.org/). Yes if needed.. Maybe they do have some more information then just categorical classification? Maybe not.

***

 Move to ideas notes

- Studying epistasis through knockouts ... maybe not feasible... too many parts.
- Do locations of proteins change depending on populations changes of different kinds of proteins. We know that proteins go to different places. Maybe you can track this probabilistically, but can you shift the distribution of other proteins?
- The GFP-tagged library is distributed by < Invitrogen >.
- The RFP-tagged strains used for the colocalization studies can be
- If we could obtain these strains

***

- [ ] Email CW
