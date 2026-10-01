---
id: ymlh1zp9sumgcisc0zm0m0j
title: Ontology_svg
desc: ''
updated: 1790872928454
created: 1790872928454
---

## 2026.10.01 - Right-side parent elbows and the schematic elbow

- A parent wholly right of its child (in the real schema `ProvenanceGapMixin` for `Phenotype`, `Environment`, `EnvironmentPerturbation`, `Compound`) is now reached from the child's right edge into the parent's left edge, turning `LANE_PAD / 2` left of the parent, with the head pointing right. Before, the elbow left the child's left edge, ran across both cards and came back to the parent's right edge. The horizontal leg can still pass under intervening cards (hidden by the opaque cards).
- The schematic elbow arrow puts its vertical leg midway between the source and the head base with radius at most half that span, so the last leg no longer runs 2.9 units backward under the head.
- Left open: the `("Genotype", "GenePerturbation")` backbone entry matches no real edge. Matching every descendant was rendered and drew twenty bold curves over the genotype lane, so the choice of the genotype spine is left to the author.
- The three committed figures under `notes/assets/images/schema-ontology/` were regenerated with `paper/nature-biotech/scripts/generate_ontology_diagram.py`. Issue #541; tests in [[tests.torchcell.paper.test_ontology_svg]].
