---
id: a33rvrm4mov3ukn3q5so8r3
title: Cgt Forward Pass
desc: ''
updated: 1789527927845
created: 1789527927845
---

## 2026.09.15 - Where the strain enters, and where the graphs do

The forward pass of the 010 configuration as the additive-baselines document reads it: the encoder runs once on the wildtype gene table at batch size one, so its output is the same for every record; the only strain-dependent step is the perturbation transform, which reads the three perturbed gene rows; the nine graphs enter through the layer-1 attention penalty during training and never through the prediction function. Beside it, the additive null on the same input. Rendered by `make diagrams` in `notes-tex/025-additive-baselines/` through `mermaid_pdf.sh`, then scaled from mmdc's 600 pt page to 510 pt with Ghostscript, fonts kept. Colors follow the draw.io palette: yellow = parameters, orange = computation, purple = per-record input and output, red = training-only, blue = a null on the same input.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial", "fontSize": "13px"}, "flowchart": {"nodeSpacing": 26, "rankSpacing": 34}}}%%
flowchart TB
  classDef par fill:#FFF2CC,stroke:#BCA04C,color:#1F1D1A
  classDef comp fill:#FFE6CC,stroke:#BD8800,color:#1F1D1A
  classDef rec fill:#E1D5E7,stroke:#846592,color:#1F1D1A
  classDef train fill:#F8CECC,stroke:#A24A46,color:#1F1D1A
  classDef null fill:#DAE8FC,stroke:#5B7AA6,color:#1F1D1A

  E["gene table E<br/>6,607 x 180, free"]:::par
  T["encoder T, 8 layers<br/>runs once, batch 1,<br/>no record enters"]:::comp
  Z["Z = T(E), h_CLS<br/>identical for every record"]:::comp
  S["record: perturbed set S<br/>three gene indices"]:::rec
  P["perturbation transform<br/>all genes attend to the rows Z_p, p in S<br/>residual, layer norm, feedforward"]:::comp
  R["readout<br/>pool over S, MLP on [h_CLS || z_S]"]:::comp
  Y["prediction y_hat<br/>a set function of S"]:::rec

  G["nine adjacency matrices"]:::train
  L["KL penalty on layer-1 attention<br/>gradient reaches Q, K, layer 0 and E"]:::train

  B["additive null B1<br/>beta_0 + sum of beta_g over S<br/>same input S"]:::null
  BY["prediction, no interaction term"]:::null

  E --> T --> Z --> P
  S --> P --> R --> Y
  G --> L
  L -. training only .-> T
  S --> B --> BY
```
