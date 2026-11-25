---
id: alnruwlxipbml8fvge96mff
title: Website_pages
desc: ''
updated: 1795570050758
created: 1795570050758
---

## 2026.11.24 - Website pages

Rewrites the Sphinx benchmark pages as Docusaurus MDX under `experiments/050-ecoli-benchmark/website/benchmark/ecoli/` and adds the examples page. The website branch (`website-benchmark-scaffold`) copies that directory into `website/docs/benchmark/ecoli/`.

## 2026.11.24 - Pages moved to the benchmark website only

The Sphinx benchmark pages were removed (owner: the pages belong on the benchmark website, not Read the Docs). This script now writes the website pages directly: an overview and one page per task under `experiments/050-ecoli-benchmark/website/benchmark/datasets/ecoli/`, with one web-sized figure per task (horizontal bars, one panel per split family; the lighter shade is the summed pair representation) and the best mean of each table row in bold. It replaces `results_page.py`.
