---
id: oez2s69bcbhb1anxhuz75ro
title: Colors
desc: ''
updated: 1781746218556
created: 1781744092741
---

## 2026.06.17 - draw.io-aligned palette

Consistent color scheme for the [[Cell Graph Transformer mermaid|torchcell.models.equivariant_cell_graph_transformer.mermaid]] figures (the recolor target is the Type I / Type II diagram, `...mermaid` L170). The palette is built from draw.io's standard swatches so draw.io figures and mermaid diagrams stay color-consistent.

draw.io swatches are **fill / border pairs** (a pastel fill with a matched darker border). Mermaid `classDef fill:.. stroke:..` uses the same model, so colors transfer 1:1.

### Palette

![Color palette: base + alternates](assets/images/color-palette.svg)

Two tiers, listed **in order of use** (all primaries first, then secondaries): **primary** + a **secondary** that matches each color broadly but is shifted deeper. Fills are draw.io's standard pastels (secondary fill = primary fill mixed 30% toward its draw.io border). Borders are darkened from draw.io's stock strokes for stronger contrast: **primary border** = stock stroke darkened 12%, **secondary border** = darkened 27%. 14 fills + 14 borders total. Within each row the order is Orange, Red, Purple, Yellow (Base) and Blue, Green, Grey (Alt).

**Base**

| Color | Primary fill / border | Secondary fill / border |
|---|---|---|
| Orange | `#FFE6CC` / `#BD8800` | `#F3D08F` / `#9D7100` |
| Red | `#F8CECC` / `#A24A46` | `#E5A9A7` / `#863D3A` |
| Purple | `#E1D5E7` / `#846592` | `#CAB8D4` / `#6E5479` |
| Yellow | `#FFF2CC` / `#BCA04C` | `#F3E0A9` / `#9C853F` |

**Alternates**

| Color | Primary fill / border | Secondary fill / border |
|---|---|---|
| Blue | `#DAE8FC` / `#5F7DA8` | `#B9CDEA` / `#4F688B` |
| Green | `#D5E8D4` / `#729E5A` | `#BCD8B3` / `#5F834A` |
| Grey | `#F5F5F5` / `#5A5A5A` | `#CACACA` / `#4A4A4A` |

**Background**

The diagram canvas / cluster fill (set via the `%%{init ...}%%` line) is a warm beige.
Use it as the shared figure background so draw.io panels and mermaid diagrams match.

| Color | Fill / border |
|---|---|
| Background | `#F5EEDD` / `#E0D6BE` |

### Live swatch (mermaid)

The `%%{init ...}%%` line sets a beige background (overriding mermaid's default pale-yellow subgraph fill).

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph TB
  subgraph BaseP["Base · primary"]
    direction LR
    O1["Orange"]:::o1
    R1["Red"]:::r1
    P1["Purple"]:::p1
    Y1["Yellow"]:::y1
  end
  subgraph AltP["Alt · primary"]
    direction LR
    B1["Blue"]:::b1
    G1["Green"]:::g1
    Gr1["Grey"]:::gr1
  end
  subgraph BaseS["Base · secondary"]
    direction LR
    O2["Orange"]:::o2
    R2["Red"]:::r2
    P2["Purple"]:::p2
    Y2["Yellow"]:::y2
  end
  subgraph AltS["Alt · secondary"]
    direction LR
    B2["Blue"]:::b2
    G2["Green"]:::g2
    Gr2["Grey"]:::gr2
  end
  subgraph BG["Background"]
    direction LR
    BGc["Background"]:::bg
  end
  classDef bg fill:#F5EEDD,stroke:#E0D6BE,stroke-width:3px
  classDef o1 fill:#FFE6CC,stroke:#BD8800,stroke-width:3px
  classDef r1 fill:#F8CECC,stroke:#A24A46,stroke-width:3px
  classDef p1 fill:#E1D5E7,stroke:#846592,stroke-width:3px
  classDef y1 fill:#FFF2CC,stroke:#BCA04C,stroke-width:3px
  classDef b1 fill:#DAE8FC,stroke:#5F7DA8,stroke-width:3px
  classDef g1 fill:#D5E8D4,stroke:#729E5A,stroke-width:3px
  classDef gr1 fill:#F5F5F5,stroke:#5A5A5A,stroke-width:3px
  classDef o2 fill:#F3D08F,stroke:#9D7100,stroke-width:3px
  classDef r2 fill:#E5A9A7,stroke:#863D3A,stroke-width:3px
  classDef p2 fill:#CAB8D4,stroke:#6E5479,stroke-width:3px
  classDef y2 fill:#F3E0A9,stroke:#9C853F,stroke-width:3px
  classDef b2 fill:#B9CDEA,stroke:#4F688B,stroke-width:3px
  classDef g2 fill:#BCD8B3,stroke:#5F834A,stroke-width:3px
  classDef gr2 fill:#CACACA,stroke:#4A4A4A,stroke-width:3px
```

### Previous matplotlib palette (reference)

`torchcell/torchcell.mplstyle` (`axes.prop_cycle`) uses an earthier, more saturated cycle -- seen in the Random Forest r2 figure: `000000, D86E2F, 7191A9, 6B8D3A, B73C39, 34699D, 775A9F, 4A9C60, E6A65D, 52B2A8, A05B2C, 3978B5, ...`. Kept for data plots; the draw.io pastel palette above is the new standard for schematic figures.

### Notes

- The L170 mermaid has **9 classes** (input, embedding, transformer, typeI, typeII, equivariant, invariant, sparse, output). With **14** swatches (7 primary + 7 secondary) there are now more than enough distinct colors; primaries can carry the main 7 classes and secondaries cover the rest (or group a primary/secondary pair for related classes, e.g. invariant/equivariant readouts).
- **Mermaid background:** mermaid's default subgraph fill is a pale yellow (`#ffffde`); the `%%{init ...}%%` theme line sets `background`/`clusterBkg` to beige (`#F5EEDD`) to override it.
