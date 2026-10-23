---
id: cnlurqew7qxo6ajzv91hauo
title: betaxanthin-omics-query
desc: ''
updated: 1792766542517
created: 1792766542517
---

## 2026.10.23

- [x] Experiment 046: the omics co-targets for the 031 betaxanthin program enter through a new KG 4.0 query (Cachera, Mulleder, Kemmeren, Messner as one raw dataset on `main`), not the hand-joined gene-keyed store [[experiments.046-betaxanthin-omics-cotarget]]
- [x] Query generator and build [[experiments.046-betaxanthin-omics-cotarget.scripts.generate_046_cql]] [[experiments.046-betaxanthin-omics-cotarget.scripts.build_046_store]] [[experiments.046-betaxanthin-omics-cotarget.scripts.gh_build_046_store]]
- [x] The fig6 deletion clause returns 1,101 of 4,759 Cachera records on KG 4.0 (symbol-named strains on the released-identifier leaf); clause extended and symbols resolved at read time [[experiments.046-betaxanthin-omics-cotarget.scripts.resolving_perturbation_046]]
- [ ] Owner to confirm the generic SGD standard-name rule for the 75 ambiguous Cachera symbols
- [ ] Trainer side: the 031 recipe on the 046 store (the 031 branch cannot read a main-schema store)
