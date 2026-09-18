---
id: nqckztvnbmy87uh6q5b645y
title: Browser_style
desc: ''
updated: 1789709690166
created: 1789709690166
---

## 2026.09.18 - Why the browser paints every node one color, and the stylesheet that fixes it

Every served node carries its Biolink ancestor labels beside its class label
(`Experiment` is also `InformationContentEntity`, `Entity`, `NamedThing`; a
`FitnessPhenotype` is also `PhenotypicFeature`, `DiseaseOrPhenotypicFeature`,
`BiologicalEntity`, `NamedThing`, `Entity`). The Browser that Neo4j 5.26 serves on port
7474 (the new one, not the classic `:style` Browser: its bundle has no GRASS command and
its styling lives in a redux slice named `styling`) keeps a label priority list. Read from
the served bundle (`assets/src.DKmcIxc2.js`, `src.DbztBbaN.js`, `query.nx-view.*.js`):

- a node's color comes from its highest-priority label (`highestPriorityLabel`, a reduce
  over `node.labels`; ties fall to `labels[0]`), and priority is position in the list,
  first is highest;
- the list starts in the order the labels were first listed from the schema, so after the
  2026.09.17 swap `NamedThing` came first and painted the whole graph;
- importing a GraSS file prepends each `node.<Label>` rule to the list as it reads the
  file, so the LAST rule in the file wins; the Browser's own "Download GraSS styles"
  writes the highest-priority label last, the same convention;
- only `color`, `caption`, `diameter` (size = diameter / 2 / 0.93) and `shaft-width` are
  read from a GraSS file; border and text colors are derived from `color`, and the
  import preview says so ("... are supported at this time, so some styling may not be
  imported").

The classic Browser (the `neo4j-browser` repository) merges matching rules top to bottom
with later properties overriding, so the same file order serves both. Its manual's
sentence "only the first (closest to top) style is applied" does not match either
implementation.

`python -m torchcell.database.browser_style` writes `database/conf/torchcell.grass`:
the eleven ancestor labels first, small and gray, then one rule per node class of
`biocypher/config/torchcell_schema_config.yaml` (38 rules), colored by the ontology lane
of the paper figures through `LANE_PALETTE_INDEX` over `PLOT_PALETTE` /
`PLOT_PALETTE_FILL`: amber genotype (`Genotype`, `Perturbation`, `SegregantGenotype`,
`CrisprConstruct`), brick environment (`Environment`, `Media`, `Temperature`,
`EnvironmentPerturbation`), wheat experiment (`Experiment`, `ExperimentReference`), lilac
phenotype (every class with `is_a: phenotypic feature`, read from the schema so a new
phenotype needs no edit), steel blue provenance (`Dataset`, `Publication`, `Genome`), gray
for BioCypher's `Schema_info`. Fill is the palette fill, border the line color. Hubs
(`Experiment`, `Dataset` 65 px, `ExperimentReference` 55 px) draw larger than the 50 px
classes and the 35 px ancestors. Captions show the readable property where one exists
(`perturbed_gene_name`, `segregant_id`, `effector`, media `name`, temperature `value`,
`compound_name`, `pubmed_id`, `strain`), the node id otherwise. Edges are gray with the
relationship type as caption.

Loading: run a query that returns a graph, open the styling panel of the result, and
click "Upload GraSS styles" (the file input accepts `.grass`, `.style`, `.txt`); "Reset
styles to default" undoes it. The stylesheet lives in that browser's local storage, so
each person loads it once per browser; `browser.post_connect_cmd` in `neo4j.conf` is the
server-side alternative and was not wired (it would need the file at an allowed URL and a
serving-container restart). `--check` fails when the committed file is behind the
generator, and `tests/torchcell/database/test_browser_style.py` covers lane assignment,
rule order, the palette, and the committed file.
