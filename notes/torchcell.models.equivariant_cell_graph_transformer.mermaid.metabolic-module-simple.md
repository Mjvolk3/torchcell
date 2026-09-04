---
id: qiogkqpktvjgegq9xqrni1b
title: Metabolic Module Simple
desc: ''
updated: 1788530786981
created: 1788530786981
---

## 2026.09.04 - Simplified view: the main components

A five-box reading of the metabolic flux module, for orientation. The full diagram with the
equations is [[torchcell.models.equivariant_cell_graph_transformer.mermaid.metabolic-module]];
the instrument taxonomy it extends is
[[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]].

**Read the edge styles.** A solid edge is implemented and carries data in every run reported
so far. A dashed edge is built but NOT wired: the table exists, is hash-pinned, and is read
by nothing. That distinction is the single most important thing about the current state, and
the detailed diagram draws several of these edges solid when the code does not have them.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph TD
  Hpert["Perturbed gene state H_pert<br><i>from the transformer</i>"]
  Avail["Gene availability a_g<br>0 if deleted, else a sigmoid gate"]
  Enzyme["Enzyme abundance E_g<br>softmin over complex, sum over isozymes"]
  Flux["FLUX VECTOR v<br>box: v = lb + (ub - lb) * sigmoid(z)<br>capacity: |v| <= k_cat * E"]
  Out["Phenotype heads<br>betaxanthin, amino acids, growth"]

  subgraph GEM["Fixed, from yeast-GEM 9.0.2"]
    direction LR
    Stoich["Stoichiometry S<br>2806 x 4131"]
    Bounds["Bounds and media<br>direction only, no capacity"]
  end

  subgraph Tables["Parameter tables (dashed = built, read by nothing)"]
    direction LR
    Kcat["k_cat<br>OED, 4.0% measured"]
    Thermo["delta_f G<br>shipped with the GEM"]
    KcatPred["k_cat predicted<br>5 models, 95.3%"]
    ThermoNew["delta_f G recomputed<br>eQuilibrator + covariance"]
    Km["K_M<br>3 models, 93-95%"]
  end

  subgraph Soft["Soft constraints"]
    direction LR
    Mass["Mass balance<br>||Sv||"]
    Second["Second law<br>relu(v * delta_r G + eps)<br>forbids loops, no binary"]
    Diss["Gibbs dissipation"]
  end

  Hpert --> Avail --> Enzyme --> Flux
  GEM --> Flux
  Kcat --> Flux
  Thermo --> Second
  KcatPred -.->|"not wired"| Flux
  Km -.->|"not wired"| Flux
  ThermoNew -.->|"not wired"| Second
  Flux --> Soft
  Flux --> Out

  classDef learned fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef fixed fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef wired fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px
  classDef unwired fill:#FFFFFF,stroke:#999999,stroke-width:1px,stroke-dasharray:4 3
  classDef flux fill:#DAE8FC,stroke:#6C8EBF,stroke-width:3px
  classDef pen fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  classDef out fill:#FFE6CC,stroke:#A24A46,stroke-width:3px

  class Hpert,Avail,Enzyme learned
  class Stoich,Bounds fixed
  class Kcat,Thermo wired
  class KcatPred,ThermoNew,Km unwired
  class Flux flux
  class Mass,Second,Diss pen
  class Out out
```

### What the five components do

| Component | Role |
|---|---|
| **Learned** | The only part with weights. A gene's embedding becomes an availability in [0,1], folded up the gene-protein-reaction rule to an enzyme abundance. A deletion pins availability to zero; everything else is free, which is what leaves room for dosage and alleles. |
| **Fixed** | Stoichiometry and bounds, straight from yeast-GEM 9.0.2 and never learned. The bounds encode direction and on/off only: 4,129 of 4,131 reactions carry no capacity at all, which is why the enzyme constraint is the load-bearing addition. |
| **Tables** | The physical constants. Solid = consumed today, dashed = built and read by nothing. Everything measured so far ran on the OED's 4.0% kcat plus a default and on the GEM's shipped formation energies. |
| **Flux vector** | The output, and the reason the module is neither Type I nor Type II: a latent physical state rather than a phenotype. Bounds and capacity hold exactly by the sigmoid parameterization and cannot be violated. |
| **Soft constraints** | Mass balance, the second law and dissipation, each a smooth penalty rather than a hard constraint. The second law is the one that matters: because delta_r G is a difference of potentials it sums to zero around any cycle, so requiring v * delta_r G <= 0 forbids internal loops with no integer variable. |

Regenerate the figure asset (name matches this note):

```bash
bash notes/assets/publish/scripts/mermaid_pdf.sh notes/torchcell.models.equivariant_cell_graph_transformer.mermaid.metabolic-module-simple.md
```
