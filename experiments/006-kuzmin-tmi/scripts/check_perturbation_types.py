"""Check what perturbation types exist in the 006-kuzmin-tmi dataset."""

import json
import os
import os.path as osp
from collections import defaultdict
from dotenv import load_dotenv
from torchcell.data import Neo4jCellDataset
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")

# Load the genome and gene_set
genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
genome.drop_chrmt()
genome.drop_empty_go()

print(f"Gene set size: {len(genome.gene_set)}")

# Load the query
with open("experiments/006-kuzmin-tmi/queries/001_small_build.cql", "r") as f:
    query = f.read()

# Load the dataset
dataset_root = osp.join(
    DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
)

dataset = Neo4jCellDataset(
    root=dataset_root,
    query=query,
    gene_set=genome.gene_set,
    graphs=None,
    node_embeddings=None,
    converter=None,
)

print(f"Dataset size: {len(dataset)}")

# Track perturbation types
perturbation_type_counts = defaultdict(int)
dataset_perturbation_types = defaultdict(lambda: defaultdict(int))
non_deletion_examples = []

# Access the raw LMDB data
dataset._init_lmdb_read()

with dataset.env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        idx = int(key.decode())
        data_list = json.loads(value.decode())

        for data in data_list:
            experiment = data["experiment"]
            genotype = experiment["genotype"]
            perturbations = genotype["perturbations"]
            dataset_name = experiment["dataset_name"]

            # Check each perturbation type
            for pert in perturbations:
                pert_type = pert.get("perturbation_type")
                perturbation_type_counts[pert_type] += 1
                dataset_perturbation_types[dataset_name][pert_type] += 1

                # Collect examples of non-deletion perturbations
                if pert_type != "deletion":
                    if len(non_deletion_examples) < 20:
                        non_deletion_examples.append({
                            "index": idx,
                            "dataset_name": dataset_name,
                            "perturbation_type": pert_type,
                            "gene": pert.get("systematic_gene_name"),
                            "full_perturbation": pert,
                        })

dataset.close_lmdb()

# Print summary
print(f"\n{'='*80}")
print(f"PERTURBATION TYPE SUMMARY")
print(f"{'='*80}")
print(f"\nOverall perturbation type counts:")
for pert_type, count in sorted(perturbation_type_counts.items()):
    print(f"  {pert_type}: {count:,}")

print(f"\n{'='*80}")
print(f"PERTURBATION TYPES BY DATASET")
print(f"{'='*80}")
for dataset_name in sorted(dataset_perturbation_types.keys()):
    print(f"\n{dataset_name}:")
    for pert_type, count in sorted(dataset_perturbation_types[dataset_name].items()):
        print(f"  {pert_type}: {count:,}")

# Show examples of non-deletion perturbations
if non_deletion_examples:
    print(f"\n{'='*80}")
    print(f"EXAMPLES OF NON-DELETION PERTURBATIONS")
    print(f"{'='*80}")
    for i, example in enumerate(non_deletion_examples):
        print(f"\n{i+1}. Index {example['index']} - {example['dataset_name']}")
        print(f"   Perturbation type: {example['perturbation_type']}")
        print(f"   Gene: {example['gene']}")
        print(f"   Full perturbation: {json.dumps(example['full_perturbation'], indent=4)}")
else:
    print(f"\n{'='*80}")
    print(f"✓ All perturbations are deletions!")
    print(f"{'='*80}")

# Save detailed results
output_dir = "experiments/006-kuzmin-tmi/results"
os.makedirs(output_dir, exist_ok=True)
output_file = osp.join(output_dir, "perturbation_types_analysis.json")

with open(output_file, "w") as f:
    json.dump({
        "summary": {
            "total_perturbations": sum(perturbation_type_counts.values()),
            "perturbation_type_counts": dict(perturbation_type_counts),
            "by_dataset": {
                dataset: dict(types)
                for dataset, types in dataset_perturbation_types.items()
            },
        },
        "non_deletion_examples": non_deletion_examples,
    }, f, indent=2)

print(f"\nDetailed results saved to: {output_file}")
