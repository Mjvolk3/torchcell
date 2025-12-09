"""Analyze filtering trade-offs for TmiKuzmin2018 and TmiKuzmin2020 datasets.

This script compares different filtering strategies to help decide:
1. Keep strict deletion-only filter (lose non-deletion data)
2. Relax filter to include alleles/TS alleles (gain more experiments)
3. Use different filters for different analyses

Shows experiment counts for all combinations.
"""

import json
from collections import defaultdict
from torchcell.datasets.scerevisiae.kuzmin2018 import TmiKuzmin2018Dataset
from torchcell.datasets.scerevisiae.kuzmin2020 import TmiKuzmin2020Dataset

print("="*80)
print("ANALYZING RAW DATASETS (BEFORE NEO4J FILTERING)")
print("="*80)

# Analyze TmiKuzmin2018Dataset
print("\n1. Loading TmiKuzmin2018Dataset...")
dataset_2018 = TmiKuzmin2018Dataset()
print(f"   Total experiments: {len(dataset_2018):,}")

stats_2018 = defaultdict(int)
experiment_has_pert_type_2018 = defaultdict(lambda: defaultdict(bool))

for i in range(len(dataset_2018)):
    if i % 5000 == 0:
        print(f"   Processing 2018: {i:,} / {len(dataset_2018):,}")

    item = dataset_2018[i]
    experiment = item["experiment"]
    genotype = experiment["genotype"]
    perturbations = genotype["perturbations"]

    # Track perturbation types
    for pert in perturbations:
        pert_type = pert["perturbation_type"]
        stats_2018[f"total_perturbations_{pert_type}"] += 1
        experiment_has_pert_type_2018[i][pert_type] = True

# Analyze TmiKuzmin2020Dataset
print("\n2. Loading TmiKuzmin2020Dataset...")
dataset_2020 = TmiKuzmin2020Dataset()
print(f"   Total experiments: {len(dataset_2020):,}")

stats_2020 = defaultdict(int)
experiment_has_pert_type_2020 = defaultdict(lambda: defaultdict(bool))

for i in range(len(dataset_2020)):
    if i % 10000 == 0:
        print(f"   Processing 2020: {i:,} / {len(dataset_2020):,}")

    item = dataset_2020[i]
    experiment = item["experiment"]
    genotype = experiment["genotype"]
    perturbations = genotype["perturbations"]

    for pert in perturbations:
        pert_type = pert["perturbation_type"]
        stats_2020[f"total_perturbations_{pert_type}"] += 1
        experiment_has_pert_type_2020[i][pert_type] = True

# Count experiments by perturbation type composition
print("\n3. Categorizing experiments by perturbation types...")

def categorize_experiments(experiment_has_pert_type, dataset_name):
    """Categorize experiments based on what perturbation types they contain."""
    categories = {
        "deletion_only": 0,
        "ts_allele_only": 0,
        "allele_only": 0,
        "deletion_and_ts_allele": 0,
        "deletion_and_allele": 0,
        "ts_allele_and_allele": 0,
        "all_three": 0,
    }

    for exp_id, pert_types in experiment_has_pert_type.items():
        has_deletion = pert_types.get("deletion", False)
        has_ts = pert_types.get("temperature_sensitive_allele", False)
        has_allele = pert_types.get("allele", False)

        if has_deletion and not has_ts and not has_allele:
            categories["deletion_only"] += 1
        elif has_ts and not has_deletion and not has_allele:
            categories["ts_allele_only"] += 1
        elif has_allele and not has_deletion and not has_ts:
            categories["allele_only"] += 1
        elif has_deletion and has_ts and not has_allele:
            categories["deletion_and_ts_allele"] += 1
        elif has_deletion and has_allele and not has_ts:
            categories["deletion_and_allele"] += 1
        elif has_ts and has_allele and not has_deletion:
            categories["ts_allele_and_allele"] += 1
        elif has_deletion and has_ts and has_allele:
            categories["all_three"] += 1

    return categories

cats_2018 = categorize_experiments(experiment_has_pert_type_2018, "2018")
cats_2020 = categorize_experiments(experiment_has_pert_type_2020, "2020")

# Print detailed analysis
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

print("\nTmiKuzmin2018Dataset:")
print(f"  Total experiments: {len(dataset_2018):,}")
print(f"\n  Perturbation counts:")
for key in sorted([k for k in stats_2018.keys() if k.startswith("total_perturbations_")]):
    pert_type = key.replace("total_perturbations_", "")
    count = stats_2018[key]
    total = sum(stats_2018[k] for k in stats_2018 if k.startswith("total_perturbations_"))
    pct = (count / total) * 100
    print(f"    {pert_type}: {count:,} ({pct:.2f}%)")

print(f"\n  Experiment composition:")
for cat, count in sorted(cats_2018.items()):
    pct = (count / len(dataset_2018)) * 100
    print(f"    {cat}: {count:,} ({pct:.2f}%)")

print("\n" + "-"*80)
print("\nTmiKuzmin2020Dataset:")
print(f"  Total experiments: {len(dataset_2020):,}")
print(f"\n  Perturbation counts:")
for key in sorted([k for k in stats_2020.keys() if k.startswith("total_perturbations_")]):
    pert_type = key.replace("total_perturbations_", "")
    count = stats_2020[key]
    total = sum(stats_2020[k] for k in stats_2020 if k.startswith("total_perturbations_"))
    pct = (count / total) * 100
    print(f"    {pert_type}: {count:,} ({pct:.2f}%)")

print(f"\n  Experiment composition:")
for cat, count in sorted(cats_2020.items()):
    pct = (count / len(dataset_2020)) * 100
    print(f"    {cat}: {count:,} ({pct:.2f}%)")

# Calculate filtering scenarios
print("\n" + "="*80)
print("FILTERING SCENARIOS")
print("="*80)

scenarios = {
    "deletion_only": {
        "2018": cats_2018["deletion_only"],
        "2020": cats_2020["deletion_only"],
        "description": "Only experiments with pure deletion perturbations"
    },
    "deletion_plus_mixed": {
        "2018": (cats_2018["deletion_only"] + cats_2018["deletion_and_ts_allele"] +
                 cats_2018["deletion_and_allele"] + cats_2018["all_three"]),
        "2020": (cats_2020["deletion_only"] + cats_2020["deletion_and_ts_allele"] +
                 cats_2020["deletion_and_allele"] + cats_2020["all_three"]),
        "description": "Experiments containing at least one deletion (current 006 query for 2020)"
    },
    "all_perturbations": {
        "2018": len(dataset_2018),
        "2020": len(dataset_2020),
        "description": "All experiments regardless of perturbation type"
    },
    "non_deletion_only": {
        "2018": (cats_2018["ts_allele_only"] + cats_2018["allele_only"] +
                 cats_2018["ts_allele_and_allele"]),
        "2020": (cats_2020["ts_allele_only"] + cats_2020["allele_only"] +
                 cats_2020["ts_allele_and_allele"]),
        "description": "Only non-deletion experiments (TS alleles and/or alleles)"
    },
}

for scenario_name, scenario_data in scenarios.items():
    count_2018 = scenario_data["2018"]
    count_2020 = scenario_data["2020"]
    combined = count_2018 + count_2020

    print(f"\n{scenario_name.upper().replace('_', ' ')}:")
    print(f"  {scenario_data['description']}")
    print(f"  2018: {count_2018:,} ({count_2018/len(dataset_2018)*100:.1f}% of 2018 data)")
    print(f"  2020: {count_2020:,} ({count_2020/len(dataset_2020)*100:.1f}% of 2020 data)")
    print(f"  Combined: {combined:,}")

# Show what we lose with strict deletion filter
print("\n" + "="*80)
print("IMPACT OF STRICT DELETION-ONLY FILTER")
print("="*80)

deletion_only_2018 = scenarios["deletion_only"]["2018"]
deletion_only_2020 = scenarios["deletion_only"]["2020"]
all_2018 = len(dataset_2018)
all_2020 = len(dataset_2020)

lost_2018 = all_2018 - deletion_only_2018
lost_2020 = all_2020 - deletion_only_2020

print(f"\nTmiKuzmin2018Dataset:")
print(f"  Keep (deletion only): {deletion_only_2018:,}")
print(f"  Lose (has non-deletion): {lost_2018:,} ({lost_2018/all_2018*100:.1f}%)")

print(f"\nTmiKuzmin2020Dataset:")
print(f"  Keep (deletion only): {deletion_only_2020:,}")
print(f"  Lose (has non-deletion): {lost_2020:,} ({lost_2020/all_2020*100:.1f}%)")

print(f"\nCombined:")
print(f"  Keep: {deletion_only_2018 + deletion_only_2020:,}")
print(f"  Lose: {lost_2018 + lost_2020:,}")

# Compare with current 006 query results
print("\n" + "="*80)
print("COMPARISON WITH CURRENT 006 QUERY")
print("="*80)

print(f"\nCurrent 006 query results (from Neo4j):")
print(f"  TmiKuzmin2018: 90,581 experiments")
print(f"  TmiKuzmin2020: 231,695 experiments")
print(f"  Combined: 322,276 experiments")

print(f"\nRaw dataset totals:")
print(f"  TmiKuzmin2018: {len(dataset_2018):,} experiments")
print(f"  TmiKuzmin2020: {len(dataset_2020):,} experiments")
print(f"  Combined: {len(dataset_2018) + len(dataset_2020):,} experiments")

print(f"\nFiltered out by 006 query:")
print(f"  TmiKuzmin2018: {len(dataset_2018) - 90581:,}")
print(f"  TmiKuzmin2020: {len(dataset_2020) - 231695:,}")
print(f"  Combined: {(len(dataset_2018) - 90581) + (len(dataset_2020) - 231695):,}")

# Save comprehensive results
output = {
    "dataset_2018": {
        "total_experiments": len(dataset_2018),
        "perturbation_counts": {k.replace("total_perturbations_", ""): v
                                for k, v in stats_2018.items()
                                if k.startswith("total_perturbations_")},
        "experiment_categories": cats_2018,
    },
    "dataset_2020": {
        "total_experiments": len(dataset_2020),
        "perturbation_counts": {k.replace("total_perturbations_", ""): v
                                for k, v in stats_2020.items()
                                if k.startswith("total_perturbations_")},
        "experiment_categories": cats_2020,
    },
    "filtering_scenarios": {
        name: {
            "2018": data["2018"],
            "2020": data["2020"],
            "combined": data["2018"] + data["2020"],
            "description": data["description"]
        }
        for name, data in scenarios.items()
    },
    "impact_of_deletion_only_filter": {
        "2018_kept": deletion_only_2018,
        "2018_lost": lost_2018,
        "2020_kept": deletion_only_2020,
        "2020_lost": lost_2020,
        "combined_kept": deletion_only_2018 + deletion_only_2020,
        "combined_lost": lost_2018 + lost_2020,
    }
}

output_file = "experiments/006-kuzmin-tmi/results/filtering_tradeoffs_analysis.json"
with open(output_file, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n\nResults saved to: {output_file}")
