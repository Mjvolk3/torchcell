**Kinetic tiers, 3,728 catalytic units (kinetic_coverage_closure.json).**

| parameter | measured | homolog | predicted | default |
|---|---:|---:|---:|---:|
| k_cat | 97 | 782 | 2831 | 18 |
| K_M | 97 | 782 | 2831 | 18 |

Default units reachable with an open-source tool: 0.

**Reaction energy coverage ladder, yeast-GEM 9.0.2 (reaction_thermo_coverage_final.json).**

| rung | reactions priced | fraction | flux-carrying priced | fraction | what it would take |
|---|---:|---:|---:|---:|---|
| now | 3779 | 0.9148 | 2716 | 0.9547 | nothing; this is the table as built |
| structures_and_pka_ladders_resolved | 3792 | 0.9179 | 2724 | 0.9575 | a structure from a namespace or a curator for every species that has none, and a pKa ladder inside (0, 14) for every species whose ladder is empty |
| everything_chemically_possible_resolved | 3802 | 0.9204 | 2727 | 0.9585 | the rung above plus a decomposition or a tabulated value for every structure the group decomposer refuses |

Permanently closed reactions by sole reason: pseudo_metabolite 216, placeholder_formula 98, placeholder_structure 2.

**pKa method selection (pka_method_selection.json).**

Rule: the worst per-pH p90 of |dGf'0(predicted) - dGf'0(cache)| over the working compartment pHs, failure rate at most 0.02, evaluated on 2100 compounds at pH 5.0, 6.2, 6.6, 7.0, 7.1, 7.2, 7.4, 7.8. Selected `cascade_top3` (ensemble: unipka-simple-template then molgpka then pkasolver); worst working-pH p90 8.331 kJ/mol at pH 7.8, failure rate 0.0024, median at pH 7.2 0.067 kJ/mol.

**Betaxanthin parity, box flux head at constraint scale 0 vs pooled, six seeds (three kcat table versions).**

| seed | 2700 | 2701 | 2702 | 2703 | 2704 | 2705 | mean | sd | n | t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nocstr | 0.0747 | 0.1185 | 0.0972 | 0.0256 | 0.0225 | 0.1417 | 0.0801 | 0.0488 | | |
| pooled | 0.0717 | 0.1058 | 0.1687 | -1.2e-04 | 0.1012 | 0.1364 | 0.0973 | 0.0581 | | |
| difference | 0.0030 | 0.0127 | -0.0715 | 0.0257 | -0.0787 | 0.0053 | -0.0172 | 0.0456 | 6 | -0.93 |

**Betaxanthin parity, seeds on one kcat table.**

| seed | 2700 | 2701 | 2703 | mean | sd | n | t |
|---|---:|---:|---:|---:|---:|---:|---:|
| nocstr | 0.0747 | 0.1185 | 0.0256 | 0.0729 | 0.0465 | | |
| pooled | 0.0717 | 0.1058 | -1.2e-04 | 0.0591 | 0.0541 | | |
| difference | 0.0030 | 0.0127 | 0.0257 | 0.0138 | 0.0114 | 3 | 2.10 |

**Amino-acid parity, box flux head vs pooled (mean per-feature Spearman, 19 features).**

| seed | 2700 | 2701 | 2702 | mean | sd | n | t |
|---|---:|---:|---:|---:|---:|---:|---:|
| nocstr | 0.0275 | 5.2e-04 | 0.0103 | 0.0128 | 0.0137 | | |
| pooled | 0.0663 | 0.0320 | 0.0560 | 0.0514 | 0.0176 | | |
| difference | -0.0387 | -0.0315 | -0.0457 | -0.0386 | 0.0071 | 3 | -9.41 |

**Amino-acid, projection head vs pooled.**

| seed | 2700 | 2701 | 2702 | mean | sd | n | t |
|---|---:|---:|---:|---:|---:|---:|---:|
| proj | -0.0122 | 0.0175 | 0.0313 | 0.0122 | 0.0222 | | |
| pooled | 0.0663 | 0.0320 | 0.0560 | 0.0514 | 0.0176 | | |
| difference | -0.0785 | -0.0145 | -0.0247 | -0.0392 | 0.0344 | 3 | -1.98 |

**Projection-head cells on betaxanthin (one results JSON each; pooled at the same seed for reference).**

| arm | seed | scale | thermo source | kcat table | selected epoch | test Spearman | pooled at seed | Genesis top-100 | balance <= 1e-2 | box exact | second law exact | balance median |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flux_projection_anchored | 2700 | 0.1 | union | intermediate verified | 72 | 0.1154 | 0.0717 | 0.00 | 1.00 | 0.00 | 0.00 | 1.5e-08 |
| flux_projection_anchored | 2701 | 0.1 | union | shipped verified | 74 | 0.0199 | 0.1058 | 0.00 | 1.00 | 0.00 | 0.00 | 2.9e-09 |
| flux_projection_anchored_tdg_stochastic | 2700 | 0.1 | union_external | shipped verified | 29 | 0.0380 | 0.0717 | 1.00 | 1.00 | 1.00 | 1.00 | 2.0e-11 |
| flux_projection_anchored_tdg | 2700 | 0 | union_external | shipped verified | 60 | -0.0006 | 0.0717 | 1.00 | 1.00 | 1.00 | 1.00 | 2.8e-08 |
| flux_projection_anchored_tdg | 2701 | 0 | union_external | shipped verified | 24 | 0.0023 | 0.1058 | 0.00 | 0.00 | 1.00 | 0.99 | 0.693 |
| flux_projection_anchored_tdg | 2702 | 0 | union_external | shipped verified | 31 | -0.0625 | 0.1687 | 0.00 | 0.00 | 1.00 | 1.00 | 0.068 |
| flux_projection_anchored_tdg | 2700 | 0.1 | union_external | shipped verified | 63 | 0.0719 | 0.0717 | 0.00 | 0.00 | 1.00 | 1.00 | 0.825 |
| flux_projection_anchored_tdg | 2701 | 0.1 | union_external | shipped verified | 32 | -0.0647 | 0.1058 | 0.00 | 0.00 | 1.00 | 1.00 | 1.003 |
| flux_projection | 2700 | 0 | union | other (94, 774, 2830, 30) | 35 | 0.1170 | 0.0717 | n/a | 1.00 | 0.00 | n/a | 1.4e-08 |
| flux_projection | 2701 | 0 | union | other (94, 774, 2830, 30) | 74 | 0.1413 | 0.1058 | n/a | 1.00 | 0.00 | n/a | 9.2e-09 |
