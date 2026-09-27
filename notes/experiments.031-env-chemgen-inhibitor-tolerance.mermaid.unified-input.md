---
id: 7hn31rskxrceyz5dmal8npj
title: Unified Input
desc: ''
updated: 1790485855374
created: 1790485855374
---

## 2026.09.27 - One input representation for five chemogenomic datasets

Every quantity in this diagram is measured, from the scripts under
`experiments/031-env-chemgen-inhibitor-tolerance/scripts/`. The five datasets differ on ploidy,
on compound panel and on dose regime, and the diagram shows which of those three a single input
representation can absorb and which it cannot. Render with
`bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.031-env-chemgen-inhibitor-tolerance.mermaid.unified-input.md`.

Colors follow the draw.io palette: gray = a served dataset, purple = a per-record input channel,
yellow = a fixed table, orange = computation, blue = an optional branch, red = what the
representation cannot carry.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial", "fontSize": "13px"}, "flowchart": {"nodeSpacing": 16, "rankSpacing": 46}}}%%
flowchart LR
  classDef src fill:#F5F5F5,stroke:#666666,color:#1F1D1A
  classDef rec fill:#E1D5E7,stroke:#846592,color:#1F1D1A
  classDef par fill:#FFF2CC,stroke:#BCA04C,color:#1F1D1A
  classDef comp fill:#FFE6CC,stroke:#BD8800,color:#1F1D1A
  classDef opt fill:#DAE8FC,stroke:#5B7AA6,color:#1F1D1A
  classDef lost fill:#F8CECC,stroke:#A24A46,color:#1F1D1A

  subgraph SRC["five served datasets, 7.48M records"]
    V["Vanacloig 2022<br/>haploid, 41 cpd<br/>IC30, no number"]:::src
    HO["Hillenmeyer HOM<br/>hom diploid, 114 cpd"]:::src
    HE["Hillenmeyer HET<br/>het diploid, 290 cpd"]:::src
    HP["Hoepfner 2014<br/>both arms, 148 cpd<br/>648,977 paired cells"]:::src
    W["Wildenhain 2015<br/>haploid, 5,170 cpd<br/>constant 20 uM"]:::src
  end

  REC["one served record<br/>genotype x environment<br/>-> phenotype"]:::comp
  V --> REC
  HO --> REC
  HE --> REC
  HP --> REC
  W --> REC

  GT["GENOTYPE channel<br/>per-gene functional dose, 6,607 genes<br/>d = copies present / copies in reference<br/>0 deleted, 0.5 het, 1 unperturbed<br/>ploidy IS the background level"]:::rec
  EN["ENVIRONMENT channel<br/>dosed compound set, 5,472 InChIKeys<br/>medium set, 37 structures<br/>dose BASIS token, not a number"]:::rec
  PH["PHYSICAL channel<br/>temperature, aerobicity, duration"]:::rec
  REC --> GT
  REC --> EN
  REC --> PH

  ME["shared molecule encoder<br/>12 registered<br/>ECFP4 count best measured"]:::par
  EN --> ME
  MET["optional metabolism branch<br/>Yeast9, 894 of 1,378 have a structure<br/>103 are also dosed compounds<br/>join on the skeleton, stereo-flat"]:::opt
  MET -.-> ME

  X["UNIFIED INPUT<br/>dose vector, compound embeddings,<br/>basis token, physical scalars"]:::comp
  GT --> X
  ME --> X
  PH --> X

  L1["the dose vector drops<br/>cassette identity, KanMX is heterologous<br/>barcode and collection<br/>ts, DAmP, CRISPRi alleles"]:::lost
  L2["dose does not pool<br/>40 of 41 Vanacloig cpd have no number<br/>Wildenhain dose is constant<br/>shared cpd differ 12.5x median"]:::lost
  L3["167 Yeast9 species have no structure<br/>tRNA, pooled lipids, proteins, biomass<br/>need a non-molecular slot"]:::lost
  X -.-> L1
  X -.-> L2
  X -.-> L3
```

**What the diagram asserts, and the evidence.**

- The genotype channel is lossless for these five on the dosage axis, because every perturbation in
  all five is either a full deletion or a one-of-two engineered copy-number variant. Verified
  against the built stores.
- Ploidy needs no separate feature because the vector's background level is the ploidy, so a
  haploid deletion at 0 against a background of 1 stays distinguishable from a homozygous diploid
  deletion at 0 against a background of 2.
- The channel is learnable, not merely expressible: Hoepfner supplies 648,977 gene-by-compound
  cells measured both heterozygous and homozygous.
- Dose enters as a basis token rather than a number, because the three regimes are not
  convertible. Hoepfner is the only dataset carrying both a molar value and an IC30 basis, and it
  shares two compounds with Vanacloig, so it cannot calibrate the panel.
- The medium and the metabolite branch share the compound encoder because 30 of 37 media
  components are Yeast9 metabolites.
