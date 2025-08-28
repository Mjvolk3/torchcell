# experiments/007-kuzmin-tm/scripts/tmi_tmf_correlation
# [[experiments.007-kuzmin-tm.scripts.tmi_tmf_correlation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/007-kuzmin-tm/scripts/tmi_tmf_correlation
# Test file: experiments/007-kuzmin-tm/scripts/test_tmi_tmf_correlation.py



import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats
import os
import os.path as osp
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    
    # Remove rows with NaN values
    label_df = label_df.dropna(subset=['gene_interaction', 'fitness'])
    print(f"After removing NaNs: {len(label_df)} samples")
    
    # Calculate correlations
    pearson_r, pearson_p = stats.pearsonr(label_df['gene_interaction'], label_df['fitness'])
    spearman_r, spearman_p = stats.spearmanr(label_df['gene_interaction'], label_df['fitness'])
    r2 = pearson_r ** 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7.4))
    
    # Create hexbin plot for density
    hexbin = ax.hexbin(
        label_df['fitness'],  # x-axis
        label_df['gene_interaction'],  # y-axis
        gridsize=150,  # Increased for smaller hexagons
        cmap='viridis',  # Other options:'plasma', 'viridis', 'inferno', 'magma', 'cividis'
        mincnt=1,
        edgecolors='none',
        linewidths=0.2
    )
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(hexbin, cax=cax)
    cbar.set_label('Count', rotation=270, labelpad=20)
    
    # Add regression line
    z = np.polyfit(label_df['fitness'], label_df['gene_interaction'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(label_df['fitness'].min(), label_df['fitness'].max(), 100)
    ax.plot(x_line, p(x_line), 'k-', linewidth=2, alpha=0.8, label='Linear fit')
    
    # Add correlation text
    text_str = f'$R^2$ = {r2:.4f}\nPearson r = {pearson_r:.4f}\nSpearman ρ = {spearman_r:.4f}'
    ax.text(0.95, 0.95, text_str, transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontsize=14)
    
    # Labels and title
    ax.set_xlabel('Fitness (TMF)')
    ax.set_ylabel('Gene Interaction (TMI)')
    ax.set_title('Correlation between Fitness and Gene Interaction')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='lower right')
    
    # Save figure
    ts = timestamp()
    title = "tmi_tmf_correlation"
    save_path = osp.join(ASSET_IMAGES_DIR, f"{title}_{ts}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
