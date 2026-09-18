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
each person loads it once per browser. There is no server-side route in this Browser:
`browser.post_connect_cmd` is honored (each semicolon-separated entry is dispatched as a
colon command on connect, source `post-connect-cmd`), but its `:style` command parses only
`reset` or nothing (`{type: "style", arg: "reset" | null}` in `command-thunks.*.js`); the
`:style <url>` of the classic Browser, which fetched a file from an allowed host, does not
exist here and 5.26 serves no classic Browser (`/browser/classic/` is 404). The file is on
GitHub at
`https://raw.githubusercontent.com/Mjvolk3/torchcell/main/database/conf/torchcell.grass`
for anyone to download and upload. `--check` fails when the committed file is behind the
generator, and `tests/torchcell/database/test_browser_style.py` covers lane assignment,
rule order, the palette, and the committed file.

## 2026.09.18 - Seeded from the image, so nobody uploads anything

The upload route above still works, but it is no longer needed. The Browser persists
its styling with redux-persist: the slice `styling` is registered as
`graphStyling: W9(rI, "graphStyling", qF)` in the served `src.DKmcIxc2.js`, where `W9`
is `persistReducer({key, storage: U9, version: 1, whitelist: qF, keyPrefix: ""})`,
`qF = ["nodeStyles", "relStyles", "stylingPriorityOrder"]`, and `U9 = new nw(framework)`
maps the key to `localStorage["nx.v1.nx.graphStyling"]`. On load redux-persist reads
that value (each field JSON-encoded on its own, plus `_persist`) and reconciles it over
the empty initial state. So a value planted under that key before the bundle runs IS
the styling every visitor sees, the same state "Upload GraSS styles" would leave behind.

`python -m torchcell.database.browser_style` now writes a second file,
`database/browser/torchcell-seed.js`: the stylesheet's rules converted exactly as the
importer converts them (`phe`: `color` kept, `diameter` to `size = floor(d / 2 / 0.93)`,
`caption: "{prop}"` to `[{type: "property", captionKey: "prop"}]`, and the priority
list is the rules in reverse file order, `NamedThing` last; the bare `node` and
`relationship` blocks carry no label and are skipped, so `relStyles` stays empty). The
script writes that value and records the stylesheet's sha256 under
`nx.v1.nx.torchcellGrassSha256`; while the recorded sha matches it does nothing, so a
person's own restyling survives until the stylesheet changes. Checked in node with a
fake `localStorage`: 38 labels seeded, top priority `VisualScorePhenotype`, bottom
`NamedThing`, a second run changes nothing, a sha change reseeds.

The page's CSP allows scripts only from the server itself (`script-src 'self'
cdn.segment.com canny.io`), so the seed cannot be inline and there is no static
directory beside the jar: `database/browser/patch_browser_jar.py`
([[database.browser.patch_browser_jar]]) adds the file to `neo4j-browser-2026.06.30+0.jar`
and tags `browser/index.html` ahead of the module script, inside the image build
`database/docker/Dockerfile.tc-neo4j-browser` ([[database.docker.Dockerfile.tc-neo4j-browser]]),
tag `michaelvolk/tc-neo4j:5.26.28-browser.1`. The serving container was relaunched on
it at 03:02 CDT (stop 03:02:22, torchcell online 03:03:03, 41 s of downtime, 99,723,455
nodes served after as before); `GET /browser/` carries the tag on line 16 and
`GET /browser/torchcell-seed.js` answers 200 with `Cache-Control: no-store`. The
live rebuild script defaults `IMAGE` to the new tag and, after the swap, refuses a served
page without the tag. `--check` covers both generated files; three tests cover the seed
state, the redux-persist shape, and the committed seed.

The same relaunch made `torchcell` the default and home database of the served store
(`STOP DATABASE neo4j; CALL dbms.setDefaultDatabase('torchcell'); START DATABASE neo4j`
on the system database, Enterprise; the `neo4j` database holds 0 nodes), so a Browser
session lands on torchcell without `:use torchcell`. The setting lives in the system
database inside `/db/database/data`, so it moves with the store through a swap, and the
live rebuild now sets it in the build container after `CREATE DATABASE` and asserts it
on the served store.
