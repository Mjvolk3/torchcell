---
id: nix3ep27vmrjkywdsmngdhc
title: Mutant_fitness_different_p_value_visualization
desc: ''
updated: 1781029055120
created: 1781029055120
---

## 2026.06.09 - Making Fitness-Threshold Sensitivity Visible Across P-value Cutoffs

This script exists to make the consequences of P-value choice on mutant-fitness calls visible, so the panel-selection thresholds in experiment 010 are chosen deliberately rather than by default. The motivation is that which mutants count as "fit enough" shifts with the significance cutoff used to call a fitness defect, and a visualization of that sensitivity guards the downstream triple selection against an arbitrary threshold. A companion test file, `test_mutant_fitness_different_p_value_visualization.py`, accompanies it.

### Specifics worth keeping

- As of this note, the source file holds only the standard frontmatter/header comment and no executable body, so the section above documents intent from the header and its companion test file rather than implemented behavior; revisit once the logic lands.
