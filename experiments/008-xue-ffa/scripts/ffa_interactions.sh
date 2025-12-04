#!/bin/bash

# multiplicative model
python experiments/008-xue-ffa/scripts/free_fatty_acid_interactions.py
python experiments/008-xue-ffa/scripts/digenic_interaction_bar_plots.py
python experiments/008-xue-ffa/scripts/trigenic_interaction_bar_plots_triple_suppression.py
python experiments/008-xue-ffa/scripts/trigenic_interaction_bar_plots_triple_suppression_relaxed.py
python experiments/008-xue-ffa/scripts/best_titers_per_ffa_with_interactions.py
python experiments/008-xue-ffa/scripts/best_titers_per_ffa_distribution_comparison.py
python experiments/008-xue-ffa/scripts/best_titers_per_ffa_cost_benefit.py

# additive model
# we have yet to write these files. we have just copied them but we need to apply the additive instead of the multiplicative model
python experiments/008-xue-ffa/scripts/additive_free_fatty_acid_interactions.py
python experiments/008-xue-ffa/scripts/additive_digenic_interaction_bar_plots.py
python experiments/008-xue-ffa/scripts/additive_trigenic_interaction_bar_plots_triple_suppression.py
python experiments/008-xue-ffa/scripts/additive_trigenic_interaction_bar_plots_triple_suppression_relaxed.py
python experiments/008-xue-ffa/scripts/additive_best_titers_per_ffa_with_interactions.py
python experiments/008-xue-ffa/scripts/additive_best_titers_per_ffa_distribution_comparison.py
python experiments/008-xue-ffa/scripts/additive_best_titers_per_ffa_cost_benefit.py

# multiplicative and additive model comparison
python experiments/008-xue-ffa/scripts/multiplicative_vs_additive_comparison.py

# Epistatic interaction models
# Log-OLS with WT-differencing (primary model)
python experiments/008-xue-ffa/scripts/log_ols_wt_differencing_epistatic_interactions.py
# GLM with log link (robustness check)
python experiments/008-xue-ffa/scripts/glm_log_link_epistatic_interactions.py
# CLR composition analysis
python experiments/008-xue-ffa/scripts/clr_composition_analysis.py
# Visualizations
python experiments/008-xue-ffa/scripts/log_ols_visualization.py
python experiments/008-xue-ffa/scripts/glm_log_link_visualization.py
# Model comparison plots
python experiments/008-xue-ffa/scripts/epistatic_models_comparison.py
# Comprehensive comparison: Multiplicative vs Additive vs GLM models
python experiments/008-xue-ffa/scripts/all_models_comparison.py
# UpSet plots comparing significant interactions across all 4 models
python experiments/008-xue-ffa/scripts/model_upset_plots.py

# Graph enrichment analysis - check if significant interactions are enriched in gene graphs
python experiments/008-xue-ffa/scripts/interaction_graph_enrichment_analysis.py
python experiments/008-xue-ffa/scripts/plot_graph_enrichment.py

# FFA metabolic network visualization pipeline
# Step 1: Identify FFA reactions in Yeast GEM (creates CSV files with reactions, genes, metabolites)
python experiments/008-xue-ffa/scripts/identify_ffa_reactions.py

# Step 2: Create FFA bipartite network (genes → reactions → metabolites) and save layout
python experiments/008-xue-ffa/scripts/create_ffa_bipartite_network.py

# Step 3: Create comprehensive visualizations with TF interaction overlays
python experiments/008-xue-ffa/scripts/create_ffa_multigraph_overlays.py

# Step 4: Create FFA-species-specific visualizations with TF interaction overlays
python experiments/008-xue-ffa/scripts/create_ffa_species_specific_overlays.py
