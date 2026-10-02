---
id: fjztsic3otmg3ufsbu10biz
title: Test_mermaid_diagram
desc: ''
updated: 1790756753830
created: 1790756753830
---

## 2026.09.30 - Phase 11: every emitted line of the diagram

Seventeen tests, 0 to 100 percent of `torchcell/ontology/mermaid_diagram.py`. A five-node, two-edge YAML gives exactly 42 lines in order (headers, CamelCase ids, `is_a` lines, dotted data lines sorted by whole string, legend, classDefs, `class` lines); reversing the YAML key order gives an identical diagram; `{}` gives only the header, legend and styling; a seven-case `_format_node_id` table; frontmatter and mermaid extraction; `write_diagram` and `main` under `tmp_path` with exact stdout. The real schema config (read-only) renders 121 lines, 26 nodes, 13 edges, 40 data lines.

Findings: `genotype` is both a Biolink class and an auto-mapped node, so it is declared and styled twice (lines 129-143, 239-249); the default orientation "LR" is not in the docstring and any string is accepted; an empty YAML file raises `AttributeError` at line 47; labels are never escaped (lines 110, 200); the first rewrite drops the blank line after the frontmatter (lines 273 versus 354). Not a code bug: both committed notes (`torchcell.ontology.mermaid_diagram.{horizontal,vertical}`) are stale, missing `NucleicAcidEntity` and `CrisprConstruct`.

## 2026.10.01 - Real schema has 27 nodes

`test_real_schema_diagram` follows the schema config: 27 nodes (23 inherited, 4 auto-mapped), 13 edges, 40 data lines, 123 diagram lines; the added node is `interned constant` (tcdb-002), and its node and `is_a` lines are asserted.
