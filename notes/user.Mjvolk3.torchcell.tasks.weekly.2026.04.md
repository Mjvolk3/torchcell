---
id: 70iqseflyde7ujdmn2kz8l2
title: '04'
desc: ''
updated: 1769477304976
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

## 2026.01.22

- [x] Fix issue associated with `.env` do copy then `sed` to replace listed env vars with proper worktree path. → This is added to [[scripts.setup-worktree]]
