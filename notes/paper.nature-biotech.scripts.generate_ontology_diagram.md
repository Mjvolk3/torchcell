---
id: hv8cjoyay4ud3dplj5x8vzo
title: Generate_ontology_diagram
desc: ''
updated: 1784508904461
created: 1784508904461
---

## 2026.07.19 - Schema/ontology diagram from the live pydantic models

A visualization of the entire torchcell schema/ontology — the typed
`genotype × environment → phenotype` experiment record — introspected directly from
the pydantic models so it can never drift from the code.

### What it is

The schema has a real shape: an **hourglass** whose waist is
`Experiment = (Genotype × Environment) → Phenotype`. On the wide left sit the deep
perturbation trees (what was changed in the cell); on the wide right sit the phenotype
trees (what was measured); underneath sit the provenance objects and the controlled
vocabularies. The layout is domain-aware (one titled, colored lane per domain) rather
than a generic graphviz hairball, so the hourglass is the first thing you see.

Colors come straight from `torchcell.utils.PLOT_PALETTE`: the six domains take the six
primary palette slots (genotype = amber, environment = brick, phenotype = lilac,
experiment = wheat, provenance = steel blue, vocabularies = gray). Green-free, warm
primaries first, exactly as the repo figure standard requires.

### Generating script

`paper/nature-biotech/scripts/generate_ontology_diagram.py`
([[paper.nature-biotech.scripts.generate_ontology_diagram]]) — run from the repo root:

```bash
python paper/nature-biotech/scripts/generate_ontology_diagram.py
```

Two reusable modules back it (so the graph and the renderer are testable on their own):

- `torchcell/paper/ontology_graph.py` — `build_ontology_graph()` reads the live models
  in `torchcell.datamodels.{schema,media,pydant}` and returns a typed `OntologyGraph`
  (pydantic) of classes, inheritance parents, own vs inherited fields, enum members,
  and composition edges. Wide discriminated unions (e.g. `Genotype.perturbations`, a
  union of 19 perturbation classes) collapse to `list[one of 19]` for width but are
  still drawn as composition edges, so nothing is lost.
- `torchcell/paper/ontology_svg.py` — lays the graph out into lanes (`build_layout`) and
  serialises three renders.

### Outputs (`$ASSET_IMAGES_DIR/schema-ontology/`)

| File | What it is | Use |
| --- | --- | --- |
| `torchcell-ontology-schematic.svg` | Legible 6-domain structural panel, all type ≥ 6 pt, 179 × ~95 mm | **the paper figure** |
| `torchcell-ontology-overview.svg` | Every class as a name-only box, hourglass topology, 179 × 74 mm (fits the 170 mm ceiling) | compact figure / SI |
| `torchcell-ontology.svg` | The full map — every field on every class, ~3850 × 3050 units | SI / link source; too tall to print as one page |
| `torchcell-ontology-explorer.html` | Self-contained pan / zoom / search / lane-filter explorer wrapping the full SVG | **the "leave a link to" artifact** |

Why three SVGs: 95 classes across 179 mm forces class labels to ~1.2 pt (5× under
Nature's 6 pt floor), so the full map cannot itself be a printed figure. The schematic
is the legible print panel; the full map is the zoomable link; the overview bridges them.

### The explorer (infinite-scroll reading)

`torchcell-ontology-explorer.html` is one self-contained file (inline SVG + CSS + JS, no
external requests). It gives the "very zoomed out, then explore" behaviour asked for:

- **Level of detail** — zoomed out you see the six domain blocks and the hourglass;
  zoom past a threshold and every class's field rows fade in.
- **Lane rail** — the legend doubles as a filter; click a domain to isolate it.
- **Search** — matches class names, field names, types, and enum values; Enter jumps to
  the first hit.
- **Detail drawer** — click any class for a catalog card: docstring, `inherits from`,
  `specialised by`, declared + inherited field counts, and `holds` (composition) chips
  that navigate the graph.
- Deep-linkable via `#ClassName`; warm-paper light theme + warm-dark theme.

`EXPLORE_URL` in the script (currently `https://torchcell.org/ontology`) is printed on
the figures as the pointer to the interactive map — update it once the explorer is
published at its final home.

### Regeneration guarantee

Because the graph is introspected at runtime, adding, renaming, or re-parenting a schema
class shows up in all four artifacts on the next run — no hand-editing. The counts on the
figure (e.g. "13 typed experiment/reference pairs") are computed from the graph, not
typed in.

![](assets/images/schema-ontology/torchcell-ontology-schematic.svg)
