# experiments/smf-dmf-tmf-001/node_removal_domain_overlap
# [[experiments.smf-dmf-tmf-001.node_removal_domain_overlap]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/smf-dmf-tmf-001/node_removal_domain_overlap


import os
import os.path as osp
import json
from dotenv import load_dotenv
from hashlib import sha256
from torchcell.data import Neo4jQueryRaw
import matplotlib.pyplot as plt
import numpy as np
import torchcell
import wandb

style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


def plot_mean_std_with_specified_std(
    neo4j_db, duplicate_domain, specified_std_value=0.1, save_path=None
):
    fitness_means = []
    fitness_stds = []

    for _, indices in duplicate_domain:
        fitness_values = [neo4j_db[i]["experiment"].phenotype.fitness for i in indices]
        fitness_means.append(np.mean(fitness_values))
        fitness_stds.append(np.std(fitness_values))

    # Count points above and below the specified standard deviation line
    above_std = sum(std > specified_std_value for std in fitness_stds)
    below_std = sum(std <= specified_std_value for std in fitness_stds)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.scatter(fitness_means, fitness_stds, c="#000000", alpha=0.6, zorder=1)
    plt.axhline(
        y=specified_std_value,
        color="#C51010",
        linestyle="-",
        label=f"Threshold STD: {specified_std_value}",
        zorder=0,
    )
    plt.title(
        f"Mean vs. Standard Deviation of Fitness\n"
        f"Points above STD line (excluded): {above_std}, Points below STD line (included): {below_std}"
    )
    plt.xlabel("Mean Fitness")
    plt.ylabel("Standard Deviation of Fitness")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell import datamodels

    wandb.init(mode="online", project="tcdb-explore")
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    with open((osp.join(osp.dirname(__file__), "query.cql")), "r") as f:
        query = f.read()

    neo4j_db = Neo4jQueryRaw(
        uri="bolt://localhost:7687",  # Include the database name here
        username="neo4j",
        password="torchcell",
        root_dir="data/torchcell/neo4j_query_raw_full",
        query=query,
        io_workers=10,
        num_workers=10,
        cypher_kwargs={"gene_set": list(genome.gene_set)},
    )

    duplicate_check = {}
    for i in range(len(neo4j_db)):
        perturbations = neo4j_db[i]["experiment"].genotype.perturbations
        sorted_gene_names = sorted(
            [pert.systematic_gene_name for pert in perturbations]
        )
        hash_key = sha256(str(sorted_gene_names).encode()).hexdigest()

        if hash_key not in duplicate_check:
            duplicate_check[hash_key] = []
        duplicate_check[hash_key].append(i)

    # After you've built your duplicate_check dictionary
    duplicate_domain = [(k, v) for k, v in duplicate_check.items() if len(v) > 1]
    for i, (k, v) in enumerate(duplicate_domain):
        print()

    save_path = osp.join(
        ASSET_IMAGES_DIR,
        "overlap_domain_scerevisiae_small_kg-mean_std_with_specified_std.png",
    )
    plot_mean_std_with_specified_std(
        neo4j_db, duplicate_domain, specified_std_value=0.10, save_path=save_path
    )


if __name__ == "__main__":
    main()
