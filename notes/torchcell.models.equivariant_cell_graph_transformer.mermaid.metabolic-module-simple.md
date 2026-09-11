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
  Hpert["$$\begin{gathered}\text{Perturbed gene state}\ H_{\mathrm{pert}}\\\ \text{from the transformer}\end{gathered}$$"]
  Avail["$$\begin{gathered}\text{Gene availability}\\\ \gamma_g = 0\ \text{if deleted, else}\ \sigma(w^\top h_g)\end{gathered}$$"]
  Enzyme["$$\begin{gathered}\text{Enzyme abundance}\ E_j\\\ \mathrm{softmin}_\beta\ \text{over a complex (GPR AND)}\\\ \sum\ \text{over isozymes (GPR OR)}\end{gathered}$$"]
  Flux["$$\begin{gathered}\textbf{Flux vector}\ \nu \in \mathbb{R}^{4131}\\\ \nu_j = lb_j + (ub_j - lb_j)\,\sigma(z_j)\\\ |\nu_j| \le k_{cat,j}\,E_j\end{gathered}$$"]
  Out["$$\begin{gathered}\text{Phenotype heads}\\\ \text{betaxanthin, amino acids, growth}\end{gathered}$$"]

  subgraph GEM["$$\text{Fixed, from yeast-GEM 9.0.2}$$"]
    direction TB
    Stoich["$$\begin{gathered}\text{Stoichiometry}\\\ S \in \mathbb{R}^{2806 \times 4131}\end{gathered}$$"]
    GPR["$$\begin{gathered}\text{GPR rules}\\\ \texttt{(A and B) or C}\\\ \text{per reaction, in the SBML}\end{gathered}$$"]
    Bounds["$$\begin{gathered}\text{Bounds and media}\\\ lb_j,\ ub_j\text{: direction only}\end{gathered}$$"]
  end

  subgraph Wired["$$\text{Parameter tables read today}$$"]
    direction LR
    Kcat["$$\begin{gathered}k_{cat}\\\ \text{OED, 4.0\% measured}\end{gathered}$$"]
    Thermo["$$\begin{gathered}\Delta_f G\\\ \text{shipped with the GEM}\end{gathered}$$"]
  end

  subgraph Built["$$\text{Built, hash-pinned, read by nothing}$$"]
    direction LR
    KcatPred["$$\begin{gathered}k_{cat}\ \text{predicted}\\\ \text{5 models, 95.3\%}\end{gathered}$$"]
    ThermoNew["$$\begin{gathered}\Delta_f G'^{\circ},\ \Sigma\ \text{recomputed}\\\ \text{eQuilibrator + covariance}\end{gathered}$$"]
    Km["$$\begin{gathered}K_M\\\ \text{3 models, 93 to 95\%}\end{gathered}$$"]
  end

  subgraph Soft["$$\text{Soft constraints}$$"]
    direction TB
    Mass["$$\begin{gathered}\text{Mass balance}\\\ \|S\nu\|\ \text{scale-relative}\end{gathered}$$"]
    Second["$$\begin{gathered}\text{Second law}\\\ \mathrm{relu}(\nu_j \Delta_r G_j + \epsilon)\\\ \text{forbids loops, no binary}\end{gathered}$$"]
  end

  Diss["$$\begin{gathered}\text{Gibbs dissipation, readout only}\\\ g_{\mathrm{diss}} = \sum_{j \in \mathrm{exch}} \Delta_r G_j \nu_j\\\ \text{checked against}\ g_{\mathrm{lim}}\ \text{(Niebel 2019), never penalized}\end{gathered}$$"]

  Hpert --> Avail --> Enzyme --> Flux
  GPR --> Enzyme
  Stoich --> Flux
  Bounds --> Flux
  Kcat --> Flux
  Thermo --> Second
  KcatPred -.->|"$$\text{not wired}$$"| Flux
  Km -.->|"$$\text{not wired}$$"| Flux
  ThermoNew -.->|"$$\text{not wired}$$"| Second
  Flux --> Soft
  Flux -.->|"$$\text{validation}$$"| Diss
  Flux --> Out

  classDef learned fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef fixed fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef wired fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px
  classDef unwired fill:#FFFFFF,stroke:#999999,stroke-width:1px,stroke-dasharray:4 3
  classDef flux fill:#DAE8FC,stroke:#6C8EBF,stroke-width:3px
  classDef pen fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  classDef out fill:#FFE6CC,stroke:#A24A46,stroke-width:3px

  class Hpert,Avail,Enzyme learned
  class Stoich,GPR,Bounds fixed
  class Kcat,Thermo wired
  class KcatPred,ThermoNew,Km unwired
  class Flux flux
  class Mass,Second pen
  class Diss unwired
  class Out out
```

**Where the complex-versus-isozyme split comes from.** It is not inferred from EC numbers
or from any kinetic table. Each yeast-GEM reaction carries a gene-protein-reaction (GPR)
rule as a boolean string in the SBML, for example `(YKL060C and YLR044C) or YGR254W`. An
`and` is a complex whose subunits are all required, so availability folds by a softmin
(the scarcest subunit limits); an `or` is a set of isozymes that substitute for one
another, so availability folds by a sum (capacities add). `parse_catalytic_units` in
`torchcell/metabolism/constraints.py` reads cobra's `gene_reaction_rule` string for every
reaction, splits on top-level `or` into units and each unit on `and` into subunits, and
stores the result as index arrays (`CatalyticUnits`); `reaction_availability` in
`flux_layer.py` applies the softmin and the sum over those arrays. A reaction with no GPR
keeps availability 1, since an unannotated reaction is one nobody has assigned a gene to,
not one that no gene catalyzes. The softmin temperature is `softmin_beta` in
`FluxLayerConfig`.

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
