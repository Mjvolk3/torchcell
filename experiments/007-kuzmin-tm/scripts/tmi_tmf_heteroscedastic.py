# experiments/007-kuzmin-tm/scripts/tmi_tmf_heteroscedastic
# [[experiments.007-kuzmin-tm.scripts.tmi_tmf_heteroscedastic]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/007-kuzmin-tm/scripts/tmi_tmf_heteroscedastic
# Test file: experiments/007-kuzmin-tm/scripts/test_tmi_tmf_heteroscedastic.py


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import os.path as osp
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


def main():
    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    
    # Apply the torchcell style
    plt.style.use("/Users/michaelvolk/Documents/projects/torchcell/torchcell/torchcell.mplstyle")
    
    # Import and load the actual dataset
    from torchcell.data import Neo4jCellDataset, GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.graph_processor import SubgraphRepresentation
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset
    from torchcell.metabolism.yeast_GEM import YeastGEM
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    
    DATA_ROOT = os.getenv("DATA_ROOT")
    with open("experiments/007-kuzmin-tm/queries/001_small_build.cql", "r") as f:
        query = f.read()

    # Add Embeddings
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/007-kuzmin-tm/001-small-build"
    )

    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )
    
    # Create dataset with metabolism network
    print("Loading dataset...")
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=None,
        incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
        node_embeddings={
            "codon_frequency": codon_frequency,
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    
    # Access the label_df from the dataset
    label_df = dataset.label_df
    print(f"Dataset loaded with {len(label_df)} samples")
    
    # Check available columns
    print("Available columns:", label_df.columns.tolist())
    
    # Remove rows with NaN values in key columns
    required_cols = ['gene_interaction', 'fitness']
    label_df = label_df.dropna(subset=required_cols)
    print(f"After removing NaNs in required columns: {len(label_df)} samples")
    
    # Check for standard deviation columns
    has_fitness_std = 'fitness_std' in label_df.columns
    has_gene_interaction_std = 'gene_interaction_std' in label_df.columns
    has_gene_interaction_p_value = 'gene_interaction_p_value' in label_df.columns
    
    print(f"Has fitness_std: {has_fitness_std}")
    print(f"Has gene_interaction_std: {has_gene_interaction_std}")
    print(f"Has gene_interaction_p_value: {has_gene_interaction_p_value}")
    
    # If no std columns, compute local variance estimates from the data itself
    if not has_fitness_std and not has_gene_interaction_std:
        print("\nNo std columns found. Computing local variance estimates from binned data...")
        
        # Compute local std for fitness
        n_bins = 50  # More fine-grained binning
        fitness_bins = pd.cut(label_df['fitness'], bins=n_bins)
        fitness_binned = label_df.groupby(fitness_bins).agg({
            'fitness': ['mean', 'std', 'count']
        })
        fitness_binned.columns = ['_'.join(col).strip() for col in fitness_binned.columns]
        fitness_binned = fitness_binned[fitness_binned['fitness_count'] > 10].reset_index()
        
        # Create synthetic fitness_std based on local variance
        df_fitness_std = pd.DataFrame()
        for _, row in fitness_binned.iterrows():
            mask = (label_df['fitness'] >= row['fitness'].left) & (label_df['fitness'] < row['fitness'].right)
            temp_df = label_df[mask].copy()
            temp_df['fitness_std'] = row['fitness_std']
            df_fitness_std = pd.concat([df_fitness_std, temp_df])
        
        # Compute local std for gene interaction  
        interaction_bins = pd.cut(label_df['gene_interaction'], bins=n_bins)  # Use same n_bins=50
        interaction_binned = label_df.groupby(interaction_bins).agg({
            'gene_interaction': ['mean', 'std', 'count']
        })
        interaction_binned.columns = ['_'.join(col).strip() for col in interaction_binned.columns]
        interaction_binned = interaction_binned[interaction_binned['gene_interaction_count'] > 10].reset_index()
        
        # Create synthetic gene_interaction_std based on local variance
        df_interaction_std = pd.DataFrame()
        for _, row in interaction_binned.iterrows():
            mask = (label_df['gene_interaction'] >= row['gene_interaction'].left) & (label_df['gene_interaction'] < row['gene_interaction'].right)
            temp_df = label_df[mask].copy()
            temp_df['gene_interaction_std'] = row['gene_interaction_std']
            df_interaction_std = pd.concat([df_interaction_std, temp_df])
        
        print(f"Created synthetic fitness_std data: {len(df_fitness_std)} samples")
        print(f"Created synthetic gene_interaction_std data: {len(df_interaction_std)} samples")
        
        # Override the flags
        if len(df_fitness_std) > 0:
            has_fitness_std = True
        if len(df_interaction_std) > 0:
            has_gene_interaction_std = True
    else:
        # Filter for samples with std values
        if has_fitness_std:
            df_fitness_std = label_df.dropna(subset=['fitness', 'fitness_std'])
            print(f"Samples with fitness_std: {len(df_fitness_std)}")
        else:
            df_fitness_std = pd.DataFrame()
            
        if has_gene_interaction_std:
            df_interaction_std = label_df.dropna(subset=['gene_interaction', 'gene_interaction_std'])
            print(f"Samples with gene_interaction_std: {len(df_interaction_std)}")
        else:
            df_interaction_std = pd.DataFrame()
    
    # Create main figure for heteroscedasticity analysis
    fig = plt.figure(figsize=(22, 10))  # Slightly smaller figure
    
    # Create two rows: one for fitness, one for gene interaction
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.3, top=0.86, bottom=0.08)  # Lower top to give more space for title
    
    # ============ FITNESS HETEROSCEDASTICITY ANALYSIS ============
    if has_fitness_std and len(df_fitness_std) > 0:
        print("\n=== Analyzing Fitness Heteroscedasticity ===")
        
        # Check if we're using synthetic std
        is_synthetic = 'fitness_std' not in label_df.columns
        
        # 1. Fitness vs Fitness Std scatter plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(df_fitness_std['fitness'], df_fitness_std['fitness_std'], 
                   alpha=0.3, s=1, c='blue')
        ax1.set_xlabel('Fitness (TMF)', fontsize=12)
        ax1.set_ylabel('Fitness Standard Deviation', fontsize=12)
        title_suffix = ' (Local Variance)' if is_synthetic else ''
        ax1.set_title(f'Fitness Std vs Fitness Value{title_suffix}', pad=20, fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=11)
        
        # Add trend line
        z = np.polyfit(df_fitness_std['fitness'], df_fitness_std['fitness_std'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_fitness_std['fitness'].min(), df_fitness_std['fitness'].max(), 100)
        ax1.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.8, 
                label=f'Trend: slope={z[0]:.4f}')
        ax1.legend()
        
        # 2. Binned analysis for fitness
        ax2 = fig.add_subplot(gs[0, 1])
        n_bins = 40  # More fine-grained binning for visualization
        fitness_bins = pd.cut(df_fitness_std['fitness'], bins=n_bins)
        binned_fitness = df_fitness_std.groupby(fitness_bins).agg({
            'fitness': 'mean',
            'fitness_std': ['mean', 'std', 'count']
        })
        binned_fitness.columns = ['_'.join(col).strip() for col in binned_fitness.columns]
        binned_fitness = binned_fitness[binned_fitness['fitness_std_count'] > 5]
        
        bin_centers = [interval.mid for interval in binned_fitness.index]
        ax2.errorbar(bin_centers, binned_fitness['fitness_std_mean'], 
                    yerr=binned_fitness['fitness_std_std'],
                    fmt='o-', capsize=5, alpha=0.7, markersize=8)
        ax2.set_xlabel('Fitness (TMF) - Binned', fontsize=12)
        ax2.set_ylabel('Mean Fitness Std ± SE', fontsize=12)
        ax2.set_title('Binned Fitness Std Analysis', pad=20, fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=11)
        
        # 3. Coefficient of Variation for fitness
        ax3 = fig.add_subplot(gs[0, 2])
        cv_fitness = df_fitness_std['fitness_std'] / np.abs(df_fitness_std['fitness'])
        cv_fitness_clean = cv_fitness[np.isfinite(cv_fitness)]
        
        ax3.scatter(df_fitness_std['fitness'][np.isfinite(cv_fitness)], 
                   cv_fitness_clean, alpha=0.3, s=1)
        ax3.set_xlabel('Fitness (TMF)', fontsize=12)
        ax3.set_ylabel('Coefficient of Variation (Std/|Mean|)', fontsize=12)
        ax3.set_title('CV vs Fitness', pad=20, fontsize=13)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', labelsize=11)
        
        # Add horizontal line at CV=1 for reference
        ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='CV=1')
        ax3.legend()
        
        # 4. Statistical summary for fitness
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        
        # Calculate statistics
        corr_fitness = np.corrcoef(df_fitness_std['fitness'], df_fitness_std['fitness_std'])[0, 1]
        
        # Perform regression for heteroscedasticity test
        X_fit = df_fitness_std['fitness'].values.reshape(-1, 1)
        y_fit_std = df_fitness_std['fitness_std'].values
        
        # Perform Breusch-Pagan test
        X_const = sm.add_constant(X_fit)
        try:
            model_fit = sm.OLS(df_fitness_std['fitness_std'].values, X_const).fit()
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(model_fit.resid, X_const)
            bp_result = f"BP p={bp_pvalue:.3f}"
        except:
            bp_result = "BP: N/A"
        
        # Determine heteroscedasticity based on multiple factors
        is_hetero_fit = abs(corr_fitness) > 0.05 or abs(z[0]) > 0.00001
        
        stats_text = [
            "Fitness Heterosc.:",
            f"N = {len(df_fitness_std)}",
            f"Corr: {corr_fitness:.3f}",
            f"Slope: {z[0]:.5f}",
            f"CV μ: {cv_fitness_clean.mean():.3f}",
            bp_result,
            "",
            f"{'Heteroscedastic' if is_hetero_fit else 'Homoscedastic'}"
        ]
        
        ax4.text(0.05, 0.95, '\n'.join(stats_text), transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    else:
        # No fitness std data
        ax_no_data = fig.add_subplot(gs[0, :])
        ax_no_data.text(0.5, 0.5, 'No fitness_std data available', 
                       transform=ax_no_data.transAxes,
                       ha='center', va='center', fontsize=14)
        ax_no_data.axis('off')
    
    # ============ GENE INTERACTION HETEROSCEDASTICITY ANALYSIS ============
    if has_gene_interaction_std and len(df_interaction_std) > 0:
        print("\n=== Analyzing Gene Interaction Heteroscedasticity ===")
        
        # Check if we're using synthetic std
        is_synthetic_int = 'gene_interaction_std' not in label_df.columns
        
        # 1. Gene Interaction vs Gene Interaction Std scatter plot
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.scatter(df_interaction_std['gene_interaction'], 
                   df_interaction_std['gene_interaction_std'], 
                   alpha=0.3, s=1, c='purple')
        ax5.set_xlabel('Gene Interaction (TMI)', fontsize=12)
        ax5.set_ylabel('Gene Interaction Standard Deviation', fontsize=12)
        title_suffix_int = ' (Local Variance)' if is_synthetic_int else ''
        ax5.set_title(f'Gene Interaction Std vs Value{title_suffix_int}', pad=20, fontsize=13)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='both', labelsize=11)
        
        # Add trend line
        z_int = np.polyfit(df_interaction_std['gene_interaction'], 
                          df_interaction_std['gene_interaction_std'], 1)
        p_int = np.poly1d(z_int)
        x_line_int = np.linspace(df_interaction_std['gene_interaction'].min(), 
                                df_interaction_std['gene_interaction'].max(), 100)
        ax5.plot(x_line_int, p_int(x_line_int), 'r-', linewidth=2, alpha=0.8,
                label=f'Trend: slope={z_int[0]:.4f}')
        ax5.legend()
        
        # 2. Binned analysis for gene interaction
        ax6 = fig.add_subplot(gs[1, 1])
        n_bins = 40  # More fine-grained binning for visualization
        interaction_bins = pd.cut(df_interaction_std['gene_interaction'], bins=n_bins)
        binned_interaction = df_interaction_std.groupby(interaction_bins).agg({
            'gene_interaction': 'mean',
            'gene_interaction_std': ['mean', 'std', 'count']
        })
        binned_interaction.columns = ['_'.join(col).strip() for col in binned_interaction.columns]
        binned_interaction = binned_interaction[binned_interaction['gene_interaction_std_count'] > 5]
        
        bin_centers_int = [interval.mid for interval in binned_interaction.index]
        ax6.errorbar(bin_centers_int, binned_interaction['gene_interaction_std_mean'],
                    yerr=binned_interaction['gene_interaction_std_std'],
                    fmt='o-', capsize=5, alpha=0.7, markersize=8, color='purple')
        ax6.set_xlabel('Gene Interaction (TMI) - Binned', fontsize=12)
        ax6.set_ylabel('Mean Gene Interaction Std ± SE', fontsize=12)
        ax6.set_title('Binned Gene Interaction Std Analysis', pad=20, fontsize=13)
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='both', labelsize=11)
        
        # 3. Coefficient of Variation for gene interaction
        ax7 = fig.add_subplot(gs[1, 2])
        cv_interaction = df_interaction_std['gene_interaction_std'] / np.abs(df_interaction_std['gene_interaction'])
        cv_interaction_clean = cv_interaction[np.isfinite(cv_interaction)]
        
        ax7.scatter(df_interaction_std['gene_interaction'][np.isfinite(cv_interaction)],
                   cv_interaction_clean, alpha=0.3, s=1, color='purple')
        ax7.set_xlabel('Gene Interaction (TMI)', fontsize=12)
        ax7.set_ylabel('Coefficient of Variation (Std/|Mean|)', fontsize=12)
        ax7.set_title('CV vs Gene Interaction', pad=20, fontsize=13)
        ax7.grid(True, alpha=0.3)
        ax7.tick_params(axis='both', labelsize=11)
        
        # Add horizontal line at CV=1 for reference
        ax7.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='CV=1')
        ax7.legend()
        
        # 4. Statistical summary for gene interaction
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.axis('off')
        
        # Calculate statistics
        corr_interaction = np.corrcoef(df_interaction_std['gene_interaction'], 
                                      df_interaction_std['gene_interaction_std'])[0, 1]
        
        # Perform Breusch-Pagan test for interaction
        X_int = df_interaction_std['gene_interaction'].values.reshape(-1, 1)
        X_int_const = sm.add_constant(X_int)
        try:
            model_int = sm.OLS(df_interaction_std['gene_interaction_std'].values, X_int_const).fit()
            bp_stat_int, bp_pvalue_int, _, _ = het_breuschpagan(model_int.resid, X_int_const)
            bp_result_int = f"BP p={bp_pvalue_int:.3f}"
        except:
            bp_result_int = "BP: N/A"
        
        # Determine heteroscedasticity based on multiple factors
        is_hetero_int = abs(corr_interaction) > 0.05 or abs(z_int[0]) > 0.00001
        
        stats_text_int = [
            "Gene Int. Heterosc.:",
            f"N = {len(df_interaction_std)}",
            f"Corr: {corr_interaction:.3f}",
            f"Slope: {z_int[0]:.5f}",
            f"CV μ: {cv_interaction_clean.mean():.3f}",
            bp_result_int,
            "",
            f"{'Heteroscedastic' if is_hetero_int else 'Homoscedastic'}"
        ]
        
        ax8.text(0.05, 0.95, '\n'.join(stats_text_int), transform=ax8.transAxes,
                fontsize=11, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))
    else:
        # No gene interaction std data
        ax_no_data = fig.add_subplot(gs[1, :])
        ax_no_data.text(0.5, 0.5, 'No gene_interaction_std data available', 
                       transform=ax_no_data.transAxes,
                       ha='center', va='center', fontsize=14)
        ax_no_data.axis('off')
    
    # Overall title
    using_synthetic = 'fitness_std' not in label_df.columns or 'gene_interaction_std' not in label_df.columns
    title_text = 'Heteroscedasticity Analysis: '
    if using_synthetic:
        title_text += 'Local Variance Estimates vs Values'
    else:
        title_text += 'Std vs Values'
    title_text += ' for Fitness and Gene Interaction'
    fig.suptitle(title_text, fontsize=18, y=0.98)  # Larger main title, higher position
    
    # Save figure
    ts = timestamp()
    title = "tmi_tmf_heteroscedastic"
    save_path = osp.join(ASSET_IMAGES_DIR, f"{title}_{ts}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    # Create additional detailed analysis figures if data is available
    if (has_fitness_std and len(df_fitness_std) > 100) or (has_gene_interaction_std and len(df_interaction_std) > 100):
        fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # FITNESS ANALYSIS (Top Row)
        if has_fitness_std and len(df_fitness_std) > 100:
            # Hexbin density plot
            ax = axes[0, 0]
            hexbin = ax.hexbin(df_fitness_std['fitness'], df_fitness_std['fitness_std'],
                              gridsize=50, cmap='viridis', mincnt=1)
            plt.colorbar(hexbin, ax=ax, label='Count')
            ax.set_xlabel('Fitness (TMF)')
            ax.set_ylabel('Fitness Std')
            ax.set_title('Fitness vs Fitness Std (Density)')
            ax.grid(True, alpha=0.3)
            
            # Distribution of fitness std
            ax = axes[0, 1]
            ax.hist(df_fitness_std['fitness_std'].dropna(), bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Fitness Standard Deviation')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Fitness Std')
            ax.grid(True, alpha=0.3)
            
            # CV distribution
            ax = axes[0, 2]
            cv_fitness = df_fitness_std['fitness_std'] / np.abs(df_fitness_std['fitness'])
            cv_fitness_clean = cv_fitness[np.isfinite(cv_fitness)]
            ax.hist(cv_fitness_clean, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Coefficient of Variation')
            ax.set_ylabel('Count')
            ax.set_title('CV Distribution (Fitness)')
            ax.axvline(x=1, color='r', linestyle='--', alpha=0.5, label='CV=1')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            for ax in axes[0, :]:
                ax.text(0.5, 0.5, 'No fitness_std data', ha='center', va='center')
                ax.axis('off')
        
        # GENE INTERACTION ANALYSIS (Bottom Row)
        if has_gene_interaction_std and len(df_interaction_std) > 100:
            # Hexbin density plot
            ax = axes[1, 0]
            hexbin = ax.hexbin(df_interaction_std['gene_interaction'], 
                              df_interaction_std['gene_interaction_std'],
                              gridsize=50, cmap='plasma', mincnt=1)
            plt.colorbar(hexbin, ax=ax, label='Count')
            ax.set_xlabel('Gene Interaction (TMI)')
            ax.set_ylabel('Gene Interaction Std')
            ax.set_title('Gene Interaction vs Std (Density)')
            ax.grid(True, alpha=0.3)
            
            # Distribution of gene interaction std
            ax = axes[1, 1]
            ax.hist(df_interaction_std['gene_interaction_std'].dropna(), bins=50, 
                   alpha=0.7, edgecolor='black', color='purple')
            ax.set_xlabel('Gene Interaction Standard Deviation')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Gene Interaction Std')
            ax.grid(True, alpha=0.3)
            
            # CV distribution
            ax = axes[1, 2]
            cv_interaction = df_interaction_std['gene_interaction_std'] / np.abs(df_interaction_std['gene_interaction'])
            cv_interaction_clean = cv_interaction[np.isfinite(cv_interaction)]
            ax.hist(cv_interaction_clean, bins=50, alpha=0.7, edgecolor='black', color='purple')
            ax.set_xlabel('Coefficient of Variation')
            ax.set_ylabel('Count')
            ax.set_title('CV Distribution (Gene Interaction)')
            ax.axvline(x=1, color='r', linestyle='--', alpha=0.5, label='CV=1')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            for ax in axes[1, :]:
                ax.text(0.5, 0.5, 'No gene_interaction_std data', ha='center', va='center')
                ax.axis('off')
        
        fig2.suptitle('Detailed Standard Deviation Analysis', fontsize=14)
        plt.tight_layout()
        
        # Save second figure
        save_path2 = osp.join(ASSET_IMAGES_DIR, f"{title}_detailed_{ts}.png")
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"Detailed analysis figure saved to: {save_path2}")
    
    # Don't show, just save
    plt.close('all')


if __name__ == "__main__":
    main()