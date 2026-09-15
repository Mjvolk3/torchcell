---
id: tt3hjv3qxll0y9ru5frivsp
title: Aa Readouts
desc: ''
updated: 1789437901337
created: 1789437901337
---
## 2026.09.14 - The three amino-acid readouts

### 1. context_concentration (the control: no flux layer)

```mermaid
flowchart LR
  G["Genotype<br/>(deleted genes)"] --> B["CGT pooled backbone<br/>context vector h"]
  B --> S["Sigmoid window<br/>lo + (hi - lo) * sigmoid(base + W h)"]
  S --> C["log c<br/>2,806 species x compartments"]
  C --> M["Volume-weighted<br/>whole-cell mixture"]
  M --> P["19 amino-acid<br/>predictions"]
  P --> L["CRPS loss vs<br/>Mulleder 2016"]
  X["NO stoichiometry<br/>NO kcat table<br/>NO flux"]:::none
  classDef none fill:#FFF2CC,stroke:#D6B656,color:#000
  style B fill:#FFE6CC,stroke:#D79B00
  style S fill:#E1D5E7,stroke:#9673A6
  style C fill:#E1D5E7,stroke:#9673A6
  style M fill:#E1D5E7,stroke:#9673A6
  style P fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 2. flux_concentration (the wave-7 arm: same readout inside the anchored flux layer)

```mermaid
flowchart LR
  G["Genotype"] --> B["CGT pooled backbone<br/>context vector h"]
  B --> S["Sigmoid window"]
  S --> C["log c<br/>2,806 species x compartments"]
  C --> M["Volume-weighted<br/>whole-cell mixture"]
  M --> P["19 amino-acid<br/>predictions"]
  P --> L["CRPS loss"]
  B --> Z["latent z"]
  Z --> V["flux v<br/>box, kcat capacity,<br/>Sv = 0 projection"]
  C --> DG["delta_r G' = delta_r G0<br/>+ RT S^T log c"]
  DG --> T["second-law and<br/>dissipation penalties<br/>(scale 0.1)"]
  V --> T
  T -. "gradient only, no path to P" .-> C
  style B fill:#FFE6CC,stroke:#D79B00
  style S fill:#E1D5E7,stroke:#9673A6
  style C fill:#E1D5E7,stroke:#9673A6
  style M fill:#E1D5E7,stroke:#9673A6
  style P fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
  style Z fill:#FFF2CC,stroke:#D6B656
  style V fill:#FFF2CC,stroke:#D6B656
  style DG fill:#FFF2CC,stroke:#D6B656
  style T fill:#FFF2CC,stroke:#D6B656
```

### 3. flux_metabolite (the turnover readout: the flux is the prediction)

```mermaid
flowchart LR
  G["Genotype"] --> B["CGT pooled backbone<br/>context vector h"]
  B --> Z["latent z"]
  Z --> V["flux v<br/>box, kcat capacity,<br/>Sv = 0 projection"]
  V --> W["turnover omega_i =<br/>1/2 sum_j |S_ij v_j|<br/>on the 19 amino-acid rows"]
  W --> P["19 amino-acid<br/>predictions"]
  P --> L["CRPS loss"]
  W -. "collapses to a constant<br/>(val spread 0.000)" .-> P
  style B fill:#FFE6CC,stroke:#D79B00
  style Z fill:#FFF2CC,stroke:#D6B656
  style V fill:#FFF2CC,stroke:#D6B656
  style W fill:#E1D5E7,stroke:#9673A6
  style P fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```
