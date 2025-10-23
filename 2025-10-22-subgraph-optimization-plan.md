SubgraphRepresentation Performance Optimization Plan

Project: TorchCell HeteroCell Model OptimizationTarget: 2x CUDA Speedup for SubgraphRepresentationDate: October
2024Current Performance: 625ms CUDA (7.5x slower than Dango baseline)Target Performance: ~300-350ms CUDA

---
Table of Contents

1. #executive-summary
2. #directory-structure
3. #key-files-reference
4. #phase-0-setup-and-baseline
5. #phase-1-optimize-bipartite-subgraph-extraction-highest-impact
6. #phase-2-optimize-mask-indexing-medium-impact
7. #phase-3-optimize-buffer-reuse-medium-impact
8. #phase-4-eliminate-redundant-device-transfers-small-impact
9. #phase-5-cache-gene-gene-edge-types-small-impact
10. #phase-6-final-verification-and-documentation
11. #expected-results-summary
12. #rollback-procedure
13. #fresh-session-startup-checklist

---
Executive Summary

This document outlines a systematic plan to optimize the SubgraphRepresentation class in TorchCell's graph processor
module. The optimization targets a 2x speedup in CUDA execution time for the HeteroCell model, which currently runs at
625ms (7.5x slower than the Dango model at 84ms).

Key Strategy: Apply 5 targeted optimizations in order of impact, validating correctness and measuring performance after
each change.

Validation Method: Equivalence testing ensures that optimized code produces identical outputs to the baseline
implementation.

Documentation: Each optimization step includes profiling, testing, and git commits for complete traceability. Our progress report for this plan is in subgraph-optimization-progress-report.md

---
Directory Structure

# Test Data and References

/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005/
├── reference_baseline.pkl              # Original unoptimized reference
├── reference_opt1_bipartite.pkl        # After Optimization 1
├── reference_opt2_mask_indexing.pkl    # After Optimization 2
├── reference_opt3_buffer_reuse.pkl     # After Optimization 3
├── reference_opt4_device_transfers.pkl # After Optimization 4
├── reference_opt5_cache_edges.pkl      # After Optimization 5 (final)
├── profiling_results/
│   ├── baseline_profile.txt            # Baseline profiling output
│   ├── opt1_profile.txt                # After Optimization 1
│   ├── opt2_profile.txt                # After Optimization 2
│   ├── opt3_profile.txt                # After Optimization 3
│   ├── opt4_profile.txt                # After Optimization 4
│   └── opt5_profile.txt                # After Optimization 5
└── metadata_*.json                     # Metadata for each step

# Test Suite

/home/michaelvolk/Documents/projects/torchcell/tests/torchcell/data/
└── test_graph_processor_equivalence.py # Equivalence test suite

# Configuration

/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/conf/
├── hetero_cell_bipartite_dango_gi.yaml # Original config (DO NOT MODIFY)
└── profiling_stable_config.yaml        # Stable config for profiling

# Scripts

/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/
├── hetero_cell_bipartite_dango_gi.py   # HeteroCell training script
├── profile_single_model.slurm          # SLURM script for profiling
├── analyze_text_profiles.py            # Profile analysis tool
└── compare_profiler_outputs.py         # Profile comparison tool

# Core Implementation

/home/michaelvolk/Documents/projects/torchcell/torchcell/
├── data/graph_processor.py             # TARGET FILE FOR OPTIMIZATION
├── scratch/
│   ├── load_batch_005.py               # Data loading utility
│   └── save_reference_baseline.py      # Reference data generator
└── models/hetero_cell_bipartite_dango_gi.py # Model using SubgraphRepresentation

# SLURM Outputs

/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/
└── *.out                                # Job outputs

# Profiler Outputs

/scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/profiler_output/
└── hetero_dango_gi_*/                   # Profile outputs by run

---
Key Files Reference

Core Files to Modify

- Primary Target: /home/michaelvolk/Documents/projects/torchcell/torchcell/data/graph_processor.py
- Class: SubgraphRepresentation (lines 31-531)
- Key Methods:
  - _process_metabolism_bipartite() (lines 364-395) - PRIMARY BOTTLENECK
  - _process_reaction_info() (lines 199-331)
  - _process_gene_interactions() (lines 178-197)

Configuration Files

- Stable Profiling Config: experiments/006-kuzmin-tmi/conf/profiling_stable_config.yaml
- Original Config: experiments/006-kuzmin-tmi/conf/hetero_cell_bipartite_dango_gi.yaml

Testing and Validation

- Equivalence Test: tests/torchcell/data/test_graph_processor_equivalence.py
- Data Loader: torchcell/scratch/load_batch_005.py
- Reference Generator: torchcell/scratch/save_reference_baseline.py

Profiling and Analysis

- SLURM Script: experiments/006-kuzmin-tmi/scripts/profile_single_model.slurm
- Profile Analyzer: experiments/006-kuzmin-tmi/scripts/analyze_text_profiles.py

---
Phase 0: Setup and Baseline

Step 0.1: Create Stable Profiling Configuration

File: experiments/006-kuzmin-tmi/conf/profiling_stable_config.yaml

cd /home/michaelvolk/Documents/projects/torchcell
cp experiments/006-kuzmin-tmi/conf/hetero_cell_bipartite_dango_gi.yaml \
    experiments/006-kuzmin-tmi/conf/profiling_stable_config.yaml

Edit profiling_stable_config.yaml:

# DO NOT MODIFY THIS FILE - Used for profiling benchmarks

# Created: [DATE]

# Purpose: Stable config for measuring SubgraphRepresentation optimizations

wandb:
project: torchcell_profiling_optimization
tags: [profiling, optimization]

cell_dataset:
graphs: [physical, regulatory]  # Only 2 graphs for faster testing
node_embeddings: [learnable]
learnable_embedding_input_channels: 64
incidence_graphs: [metabolism_bipartite]

profiler:
is_pytorch: true  # CRITICAL: Enable profiling

data_module:
is_perturbation_subset: true
perturbation_subset_size: 1000  # Small subset for profiling
batch_size: 32
num_workers: 2
pin_memory: true
prefetch: false

trainer:
max_epochs: 1  # Single epoch only
strategy: auto
num_nodes: 1
accelerator: gpu
devices: 1
precision: bf16-mixed

model:
gene_num: 6607
reaction_num: 7122
metabolite_num: 2806
hidden_channels: 64
num_layers: 3

# ... keep rest from original

Step 0.2: Create Profiling SLURM Script

File: experiments/006-kuzmin-tmi/scripts/profile_single_model.slurm

cat > experiments/006-kuzmin-tmi/scripts/profile_single_model.slurm << 'EOF'
#!/bin/bash
#SBATCH -p main
#SBATCH --mem=100g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=profile-opt
#SBATCH --time=01:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# Profile HeteroCell model with stable config

# Usage: sbatch profile_single_model.slurm

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

if [ -f /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif ]; then
    echo "Using container for profiling"

    apptainer exec --nv \
    --bind /scratch:/scratch \
    --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
    --env PYTHONUNBUFFERED=1 \
    /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc '
    
    source ~/miniconda3/bin/activate
    conda activate torchcell
    
    cd /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts
    
    echo "========================================="
    echo "Profiling HeteroCell with Stable Config"
    echo "========================================="
    python hetero_cell_bipartite_dango_gi.py --config-name profiling_stable_config
    '
else
    echo "Running without container"

    source ~/miniconda3/bin/activate
    conda activate torchcell

    cd experiments/006-kuzmin-tmi/scripts

    echo "========================================="
    echo "Profiling HeteroCell with Stable Config"
    echo "========================================="
    python hetero_cell_bipartite_dango_gi.py --config-name profiling_stable_config
fi

echo ""
echo "Profiling complete!"
echo "Profile outputs saved to: /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/profiler_output/"
EOF

chmod +x experiments/006-kuzmin-tmi/scripts/profile_single_model.slurm

Step 0.3: Create Equivalence Test Suite

File: tests/torchcell/data/test_graph_processor_equivalence.py

#!/usr/bin/env python3
"""
Equivalence test suite for SubgraphRepresentation optimizations.
Tests that optimized graph_processor produces identical outputs to baseline.
"""

import os.path as osp
import pickle
import pytest
import torch
from torchcell.scratch.load_batch_005 import load_sample_data_batch

REFERENCE_DIR = "/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005"

def compare_hetero_data(reference, current, name="data", rtol=1e-5, atol=1e-8):
    """
    Compare two HeteroData objects for structural and numerical equality.
    """
    # Compare node types
    assert set(reference.node_types) == set(current.node_types), \
        f"{name}: Node types differ.\n  Reference: {reference.node_types}\n  Current: {current.node_types}"

    # Compare edge types
    assert set(reference.edge_types) == set(current.edge_types), \
        f"{name}: Edge types differ.\n  Reference: {reference.edge_types}\n  Current: {current.edge_types}"

    # Compare node attributes
    for node_type in reference.node_types:
        ref_store = reference[node_type]
        cur_store = current[node_type]

        ref_attrs = set(ref_store.keys())
        cur_attrs = set(cur_store.keys())

        assert ref_attrs == cur_attrs, \
            f"{name}['{node_type}']: Attributes differ.\n  Reference: {ref_attrs}\n  Current: {cur_attrs}"

        for attr in ref_attrs:
            ref_val = ref_store[attr]
            cur_val = cur_store[attr]

            if torch.is_tensor(ref_val):
                assert ref_val.shape == cur_val.shape, \
                    f"{name}['{node_type}']['{attr}']: Shape mismatch.\n  Reference: {ref_val.shape}\n  Current: 
{cur_val.shape}"

                assert ref_val.dtype == cur_val.dtype, \
                    f"{name}['{node_type}']['{attr}']: Dtype mismatch"

                if ref_val.dtype in [torch.float, torch.float32, torch.float64]:
                    assert torch.allclose(ref_val, cur_val, rtol=rtol, atol=atol), \
                        f"{name}['{node_type}']['{attr}']: Values differ.\n  Max diff: {(ref_val - 
cur_val).abs().max().item()}"
                else:
                    assert torch.equal(ref_val, cur_val), \
                        f"{name}['{node_type}']['{attr}']: Values differ"
            elif isinstance(ref_val, list):
                assert len(ref_val) == len(cur_val) and ref_val == cur_val, \
                    f"{name}['{node_type}']['{attr}']: List values differ"
            else:
                assert ref_val == cur_val, \
                    f"{name}['{node_type}']['{attr}']: Value mismatch"

    # Compare edge attributes
    for edge_type in reference.edge_types:
        ref_store = reference[edge_type]
        cur_store = current[edge_type]

        ref_attrs = set(ref_store.keys())
        cur_attrs = set(cur_store.keys())

        assert ref_attrs == cur_attrs, \
            f"{name}[{edge_type}]: Attributes differ"

        for attr in ref_attrs:
            ref_val = ref_store[attr]
            cur_val = cur_store[attr]

            if torch.is_tensor(ref_val):
                assert ref_val.shape == cur_val.shape, \
                    f"{name}[{edge_type}]['{attr}']: Shape mismatch"

                assert ref_val.dtype == cur_val.dtype, \
                    f"{name}[{edge_type}]['{attr}']: Dtype mismatch"

                if ref_val.dtype in [torch.float, torch.float32, torch.float64]:
                    assert torch.allclose(ref_val, cur_val, rtol=rtol, atol=atol), \
                        f"{name}[{edge_type}]['{attr}']: Values differ"
                else:
                    assert torch.equal(ref_val, cur_val), \
                        f"{name}[{edge_type}]['{attr}']: Values differ"

if __name__ == "__main__":
    import sys

    print("="*80)
    print("SubgraphRepresentation Equivalence Test")
    print("="*80)

    # Load reference
    ref_path = osp.join(REFERENCE_DIR, "reference_baseline.pkl")
    if not osp.exists(ref_path):
        print(f"\nERROR: Reference baseline not found: {ref_path}")
        print("Run Step 0.4 to generate baseline reference data.")
        sys.exit(1)

    with open(ref_path, "rb") as f:
        ref_data = pickle.load(f)

    # Generate current
    print("\nGenerating current data...")
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=2,
        num_workers=2,
        config="hetero_cell_bipartite",
        is_dense=False
    )

    # Run tests
    try:
        print("\nTesting single instance equivalence...")
        compare_hetero_data(ref_data["single_instance"], dataset[0], "single_instance")
        print("✓ Single instance matches reference")

        print("\nTesting batch equivalence...")
        compare_hetero_data(ref_data["batch"], batch, "batch")
        print("✓ Batch matches reference")

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED - Optimizations preserve correctness!")
        print("="*80)
    except AssertionError as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        sys.exit(1)

Step 0.4: Generate Baseline Reference Data

File: torchcell/scratch/save_reference_baseline.py

#!/usr/bin/env python3
"""Generate baseline reference data BEFORE any optimizations."""

import os
import os.path as osp
import pickle
import json
from datetime import datetime
from torchcell.scratch.load_batch_005 import load_sample_data_batch

def save_baseline_reference():
    """Generate and save baseline reference data."""

    output_dir = "/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(osp.join(output_dir, "profiling_results"), exist_ok=True)

    print("="*80)
    print("Generating Baseline Reference Data (BEFORE Optimizations)")
    print("="*80)

    print("\nLoading HeteroCell configuration...")
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=2,
        num_workers=2,
        config="hetero_cell_bipartite",
        is_dense=False
    )

    single_instance = dataset[0]

    print("\n=== Single Instance Structure ===")
    print(single_instance)
    print("\n=== Batch Structure ===")
    print(batch)

    # Prepare reference data
    reference_data = {
        "single_instance": single_instance,
        "batch": batch,
        "metadata": {
            "dataset_length": len(dataset),
            "max_num_nodes": max_num_nodes,
            "input_channels": input_channels,
            "timestamp": datetime.now().isoformat(),
            "optimization_step": "baseline",

            "instance": {
                "gene_nodes": single_instance["gene"].num_nodes,
                "reaction_nodes": single_instance["reaction"].num_nodes,
                "metabolite_nodes": single_instance["metabolite"].num_nodes,
                "physical_edges": single_instance["gene", "physical", "gene"].num_edges,
                "regulatory_edges": single_instance["gene", "regulatory", "gene"].num_edges,
                "gpr_edges": single_instance["gene", "gpr", "reaction"].num_edges,
                "rmr_edges": single_instance["reaction", "rmr", "metabolite"].num_edges,
                "perturbed_genes": len(single_instance["gene"].ids_pert),
            },

            "batch": {
                "num_graphs": batch.num_graphs,
                "gene_nodes": batch["gene"].num_nodes,
                "reaction_nodes": batch["reaction"].num_nodes,
                "metabolite_nodes": batch["metabolite"].num_nodes,
                "physical_edges": batch["gene", "physical", "gene"].edge_index.size(1),
                "regulatory_edges": batch["gene", "regulatory", "gene"].edge_index.size(1),
                "gpr_edges": batch["gene", "gpr", "reaction"].hyperedge_index.size(1),
                "rmr_edges": batch["reaction", "rmr", "metabolite"].hyperedge_index.size(1),
            },
        }
    }

    # Save baseline
    output_file = osp.join(output_dir, "reference_baseline.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(reference_data, f)

    print(f"\n✓ Baseline reference saved: {output_file}")

    # Save metadata as JSON
    metadata_file = osp.join(output_dir, "metadata_baseline.json")
    with open(metadata_file, "w") as f:
        json.dump(reference_data["metadata"], f, indent=2)

    print(f"✓ Metadata saved: {metadata_file}")

    return output_file

if __name__ == "__main__":
    save_baseline_reference()

Execute:
cd /home/michaelvolk/Documents/projects/torchcell
python torchcell/scratch/save_reference_baseline.py

Step 0.5: Run Baseline Profiling

sbatch experiments/006-kuzmin-tmi/scripts/profile_single_model.slurm

Wait for completion (~10-15 minutes), then:

# Check job status

squeue -u $USER

# When complete, find and copy profile

LATEST_PROFILE=$(ls -t /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/profiler_output/hetero_dan
go_gi_*/fit-profile*.txt | head -1)

cp $LATEST_PROFILE
/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005/profiling_results/baseline_profile.txt

# Analyze baseline performance

cd /home/michaelvolk/Documents/projects/torchcell
python experiments/006-kuzmin-tmi/scripts/analyze_text_profiles.py

# Record the baseline CUDA time (should be ~625ms)

Step 0.6: Verify Baseline Test Passes

python tests/torchcell/data/test_graph_processor_equivalence.py

Expected output:
================================================================================

SubgraphRepresentation Equivalence Test
================================================================================

Generating current data...

Testing single instance equivalence...
✓ Single instance matches reference

Testing batch equivalence...
✓ Batch matches reference

================================================================================
✓ ALL TESTS PASSED - Optimizations preserve correctness!
================================================================================

Step 0.7: Commit Baseline Setup

git add experiments/006-kuzmin-tmi/conf/profiling_stable_config.yaml
git add experiments/006-kuzmin-tmi/scripts/profile_single_model.slurm
git add tests/torchcell/data/test_graph_processor_equivalence.py
git add torchcell/scratch/save_reference_baseline.py
git commit -m "Add profiling infrastructure for SubgraphRepresentation optimization

- Create stable profiling config (profiling_stable_config.yaml)
- Add SLURM script for single model profiling
- Implement equivalence test suite
- Generate baseline reference data
- Baseline CUDA time: 625ms (record actual value)

Target: 2x speedup (~312ms CUDA time)
"

---
Phase 1: Optimize Bipartite Subgraph Extraction (HIGHEST IMPACT)

Expected Impact: 40-50% speedup (reduce 244ms gather operations to ~120ms)

Step 1.1: Implement Fast Path

File to modify: torchcell/data/graph_processor.py

Current implementation (lines 364-395):
def _process_metabolism_bipartite(
    self,
    integrated_subgraph: HeteroData,
    cell_graph: HeteroData,
    reaction_info: Dict[str, Any],
) -> None:
    valid_reactions = reaction_info["valid_reactions"]
    rmr_edges = cell_graph["reaction", "rmr", "metabolite"]
    hyperedge_index = rmr_edges.hyperedge_index.to(self.device)
    stoichiometry = rmr_edges.stoichiometry.to(self.device)
    metabolite_subset = torch.arange(
        cell_graph["metabolite"].num_nodes, device=self.device
    )

    # BOTTLENECK: General bipartite_subgraph handles arbitrary subsets
    final_edge_index, final_edge_attr = bipartite_subgraph(
        (valid_reactions, metabolite_subset),
        hyperedge_index,
        edge_attr=stoichiometry,
        relabel_nodes=True,
        size=(cell_graph["reaction"].num_nodes, cell_graph["metabolite"].num_nodes),
    )
    # ... rest of function

Optimized implementation:
def _process_metabolism_bipartite(
    self,
    integrated_subgraph: HeteroData,
    cell_graph: HeteroData,
    reaction_info: Dict[str, Any],
) -> None:
    valid_reactions = reaction_info["valid_reactions"]
    rmr_edges = cell_graph["reaction", "rmr", "metabolite"]
    hyperedge_index = rmr_edges.hyperedge_index.to(self.device)
    stoichiometry = rmr_edges.stoichiometry.to(self.device)

    # OPTIMIZATION: Fast path for all-metabolite case (100% of usage)
    num_reactions = cell_graph["reaction"].num_nodes
    num_metabolites = cell_graph["metabolite"].num_nodes

    if len(valid_reactions) < num_reactions:
        # Create boolean mask for valid reactions
        reaction_mask = torch.zeros(num_reactions, dtype=torch.bool, device=self.device)
        reaction_mask[valid_reactions] = True

        # Filter edges: keep only edges from valid reactions
        edge_mask = reaction_mask[hyperedge_index[0]]
        final_edge_index = hyperedge_index[:, edge_mask].clone()
        final_edge_attr = stoichiometry[edge_mask].clone()

        # Relabel reaction indices (metabolite indices unchanged)
        reaction_map = torch.full((num_reactions,), -1, dtype=torch.long, device=self.device)
        reaction_map[valid_reactions] = torch.arange(len(valid_reactions), device=self.device)
        final_edge_index[0] = reaction_map[final_edge_index[0]]
    else:
        # All reactions valid - no filtering needed
        final_edge_index = hyperedge_index.clone()
        final_edge_attr = stoichiometry.clone()

    # Store in integrated subgraph
    edge_type = ("reaction", "rmr", "metabolite")
    integrated_subgraph[edge_type].hyperedge_index = final_edge_index
    integrated_subgraph[edge_type].stoichiometry = final_edge_attr
    integrated_subgraph[edge_type].num_edges = final_edge_index.size(1)

    integrated_subgraph["metabolite"].node_ids = cell_graph["metabolite"].node_ids
    integrated_subgraph["metabolite"].num_nodes = cell_graph["metabolite"].num_nodes
    self.masks["metabolite"]["kept"].fill_(True)
    self.masks["metabolite"]["removed"].fill_(False)

Step 1.2: Test Equivalence

python tests/torchcell/data/test_graph_processor_equivalence.py

CRITICAL: If test fails, DO NOT PROCEED. Debug and fix before continuing.

Step 1.3: Save Optimization 1 Reference

# Create and run this script

cat > /tmp/save_opt1_reference.py << 'EOF'
import os.path as osp
import shutil
from torchcell.scratch.save_reference_baseline import save_baseline_reference

# Generate new reference with optimized code

output_dir = '/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005'
save_baseline_reference()

# Rename to opt1

shutil.move(
    osp.join(output_dir, 'reference_baseline.pkl'),
    osp.join(output_dir, 'reference_opt1_bipartite.pkl')
)
shutil.move(
    osp.join(output_dir, 'metadata_baseline.json'),
    osp.join(output_dir, 'metadata_opt1_bipartite.json')
)
print('✓ Optimization 1 reference data saved')
EOF

python /tmp/save_opt1_reference.py

Step 1.4: Profile After Optimization 1

sbatch experiments/006-kuzmin-tmi/scripts/profile_single_model.slurm

# Wait for completion

squeue -u $USER

# Copy profile

LATEST_PROFILE=$(ls -t /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/profiler_output/hetero_dan
go_gi_*/fit-profile*.txt | head -1)
cp $LATEST_PROFILE
/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005/profiling_results/opt1_profile.txt

# Analyze improvement

grep "Self CUDA time total"
/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005/profiling_results/opt1_profile.txt

Step 1.5: Commit Optimization 1

git add torchcell/data/graph_processor.py
git commit -m "Optimization 1: Fast path for bipartite subgraph extraction

Replace general bipartite_subgraph() with optimized filtering for
all-metabolite case (100% of usage). Uses direct boolean masking
instead of expensive gather/scatter operations.

Performance:

- Baseline CUDA: 625ms
- After Opt1: [ACTUAL]ms
- Speedup: [ACTUAL]x

Impact on aten::gather operations reduced from 244ms to ~[ACTUAL]ms
Tests: All equivalence tests pass
"

---
Phase 2: Optimize Mask Indexing (MEDIUM IMPACT)

Step 2.1: Replace torch.isin with Boolean Masks

File: torchcell/data/graph_processor.pyLocation: Lines 296-298 in_process_reaction_info()

Current:
edge_mask = torch.isin(
    reaction_indices, torch.where[valid_with_genes_mask](0)
) & torch.isin(gene_indices, gene_info["keep_subset"])

Optimized:

# Create boolean masks for O(1) lookup instead of O(n×m) torch.isin

valid_reactions_mask = torch.zeros(max_reaction_idx, dtype=torch.bool, device=self.device)
valid_reactions_mask[torch.where[valid_with_genes_mask](0)] = True

keep_genes_mask = torch.zeros(cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device)
keep_genes_mask[gene_info["keep_subset"]] = True

# Direct indexing - O(n) instead of O(n×m)

edge_mask = valid_reactions_mask[reaction_indices] & keep_genes_mask[gene_indices]

Step 2.2-2.5: Test, Save, Profile, Commit

Follow same pattern as Phase 1:

1. Test equivalence
2. Save reference as reference_opt2_mask_indexing.pkl
3. Profile and save as opt2_profile.txt
4. Commit with performance metrics

---
Phase 3: Optimize Buffer Reuse (MEDIUM IMPACT)

Step 3.1: Add Reusable Buffers

Add to __init__():
def __init__(self) -> None:
    super().__init__()
    self.device = torch.device("cpu")
    self.masks: Dict[str, Dict[str, torch.Tensor]] = {}
    # NEW: Reusable buffers
    self._gene_map_buffer: Optional[torch.Tensor] = None
    self._reaction_map_buffer: Optional[torch.Tensor] = None

Add helper methods:
def _get_gene_map(self, num_genes: int, keep_subset: torch.Tensor) -> torch.Tensor:
    """Get or create reusable gene mapping tensor."""
    if self._gene_map_buffer is None or self._gene_map_buffer.size(0) != num_genes:
        self._gene_map_buffer = torch.full((num_genes,), -1, dtype=torch.long, device=self.device)
    else:
        self.*gene_map_buffer.fill*(-1)

    self._gene_map_buffer[keep_subset] = torch.arange(len(keep_subset), device=self.device)
    return self._gene_map_buffer

def _get_reaction_map(self, num_reactions: int, valid_reactions: torch.Tensor) -> torch.Tensor:
    """Get or create reusable reaction mapping tensor."""
    if self._reaction_map_buffer is None or self._reaction_map_buffer.size(0) != num_reactions:
        self._reaction_map_buffer = torch.full((num_reactions,), -1, dtype=torch.long, device=self.device)
    else:
        self.*reaction_map_buffer.fill*(-1)

    self._reaction_map_buffer[valid_reactions] = torch.arange(len(valid_reactions), device=self.device)
    return self._reaction_map_buffer

Replace in _process_reaction_info() (lines 301-315):

# OLD:

# gene_map = torch.full((cell_graph["gene"].num_nodes,), -1, dtype=torch.long, device=self.device)

# reaction_map = torch.full((max_reaction_idx,), -1, dtype=torch.long, device=self.device)

# NEW:

gene_map = self._get_gene_map(cell_graph["gene"].num_nodes, gene_info["keep_subset"])
reaction_map = self._get_reaction_map(max_reaction_idx, valid_reactions)

Step 3.2-3.5: Test, Save, Profile, Commit

Follow standard pattern.

---
Phase 4: Eliminate Redundant Device Transfers (SMALL IMPACT)

Step 4.1: Add Device Check Helper

def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to self.device only if needed."""
    if tensor.device != self.device:
        return tensor.to(self.device)
    return tensor

Replace all .to(self.device) calls:

# Lines 253-255, 372-373, etc.

# OLD:

hyperedge_index = rmr_edges.hyperedge_index.to(self.device)

# NEW:

hyperedge_index = self._ensure_device(rmr_edges.hyperedge_index)

Step 4.2-4.5: Test, Save, Profile, Commit

---
Phase 5: Cache Gene-Gene Edge Types (SMALL IMPACT)

Step 5.1: Pre-filter Edge Types

Add to __init__():
self._gene_gene_edge_types: Optional[List] = None

In process() method:

# Cache gene-gene edge types on first call

if self._gene_gene_edge_types is None:
    self._gene_gene_edge_types = [
        et for et in cell_graph.edge_types
        if et[0] == "gene" and et[2] == "gene"
    ]

Update _process_gene_interactions():

# Use cached list

for et in self._gene_gene_edge_types:
    # ... process edge type

Step 5.2-5.5: Test, Save, Profile, Commit

---
Phase 6: Final Verification and Documentation

Step 6.1: Create Performance Summary

# Run this script to generate summary

cat > /tmp/generate_performance_summary.py << 'EOF'
import os.path as osp
import re

results_dir = '/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005/profiling_results'

profiles = [
    ('Baseline', 'baseline_profile.txt'),
    ('Opt1: Bipartite', 'opt1_profile.txt'),
    ('Opt2: Mask Index', 'opt2_profile.txt'),
    ('Opt3: Buffer Reuse', 'opt3_profile.txt'),
    ('Opt4: Device Transfer', 'opt4_profile.txt'),
    ('Opt5: Cache Edges', 'opt5_profile.txt'),
]

print("SubgraphRepresentation Optimization Results")
print("="*80)
print(f"{'Step':<25} {'CUDA Time':<15} {'vs Baseline':<15} {'Incremental':<15}")
print("-"*80)

cuda_times = []
for name, filename in profiles:
    filepath = osp.join(results_dir, filename)
    if osp.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
            match = re.search(r'Self CUDA time total:\s*([\d.]+)(ms|s)', content)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                cuda_ms = value* 1000 if unit == 's' else value
                cuda_times.append((name, cuda_ms))

baseline = cuda_times[0][1] if cuda_times else 625.0
prev_time = baseline

for name, cuda_ms in cuda_times:
    vs_baseline = baseline / cuda_ms if cuda_ms > 0 else 1.0
    incremental = prev_time / cuda_ms if cuda_ms > 0 else 1.0
    print(f"{name:<25} {cuda_ms:>10.1f} ms {vs_baseline:>10.2f}x {incremental:>10.2f}x")
    prev_time = cuda_ms

if cuda_times:
    final_speedup = baseline / cuda_times[-1][1]
    print("-"*80)
    print(f"{'TOTAL SPEEDUP':<25} {'':<15} {final_speedup:>10.2f}x")
EOF

python /tmp/generate_performance_summary.py

Step 6.2: Create Final Documentation

File: docs/subgraph_optimization_report.md

# SubgraphRepresentation Optimization Report

## Executive Summary

Successfully optimized `SubgraphRepresentation` in `torchcell/data/graph_processor.py`
achieving [ACTUAL]x speedup in CUDA execution time for the HeteroCell model.

## Performance Results

| Optimization | CUDA Time | vs Baseline | Cumulative |
|--------------|-----------|-------------|------------|
| Baseline | 625ms | 1.00x | - |
| Opt1: Bipartite Fast Path | [ACTUAL]ms | [ACTUAL]x | [ACTUAL]x |
| Opt2: Mask Indexing | [ACTUAL]ms | [ACTUAL]x | [ACTUAL]x |
| Opt3: Buffer Reuse | [ACTUAL]ms | [ACTUAL]x | [ACTUAL]x |
| Opt4: Device Transfers | [ACTUAL]ms | [ACTUAL]x | [ACTUAL]x |
| Opt5: Cache Edge Types | [ACTUAL]ms | [ACTUAL]x | [ACTUAL]x |

**Final Result**: 625ms → [ACTUAL]ms ([ACTUAL]x speedup)
**Target**: 2.0x speedup (312ms) - [ACHIEVED/NOT ACHIEVED]

## Key Optimizations

### 1. Bipartite Subgraph Fast Path

- Replaced general `bipartite_subgraph()` with specialized implementation
- Leverages fact that all metabolites are always kept
- Impact: ~40% speedup on subgraph operations

### 2. Boolean Mask Indexing

- Replaced O(n×m) `torch.isin()` with O(n) boolean indexing
- Impact: ~10% speedup on edge filtering

### 3. Buffer Reuse

- Pre-allocate and reuse mapping tensors
- Impact: ~5% speedup on memory operations

### 4. Device Transfer Optimization

- Check device before transfers
- Impact: ~3% speedup

### 5. Edge Type Caching

- Pre-filter gene-gene edges once
- Impact: ~2% speedup

## Validation

All optimizations validated with comprehensive equivalence testing:

- ✓ Single instance data structures identical
- ✓ Batch data structures identical
- ✓ Tensor values within tolerance (rtol=1e-5, atol=1e-8)
- ✓ All metadata preserved

## Repository Changes

- **Modified**: `torchcell/data/graph_processor.py`
- **Tests**: `tests/torchcell/data/test_graph_processor_equivalence.py`
- **Config**: `experiments/006-kuzmin-tmi/conf/profiling_stable_config.yaml`
- **SLURM**: `experiments/006-kuzmin-tmi/scripts/profile_single_model.slurm`

## Reproducibility

All reference data and profiles stored at:
/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005/

To verify results:

```bash
python tests/torchcell/data/test_graph_processor_equivalence.py
sbatch experiments/006-kuzmin-tmi/scripts/profile_single_model.slurm

Commit History

- [hash] Baseline infrastructure
- [hash] Opt1: Bipartite fast path
- [hash] Opt2: Mask indexing
- [hash] Opt3: Buffer reuse
- [hash] Opt4: Device transfers
- [hash] Opt5: Cache edges
- [hash] Final documentation

### Step 6.3: Final Commit

```bash
git add docs/subgraph_optimization_report.md
git commit -m "Complete SubgraphRepresentation optimization: [ACTUAL]x speedup

Summary:
- Baseline: 625ms CUDA time
- Final: [ACTUAL]ms CUDA time
- Speedup: [ACTUAL]x
- Target 2x: [ACHIEVED/MISSED]

All optimizations validated with equivalence tests.
See docs/subgraph_optimization_report.md for full details.
"

---
Expected Results Summary

| Phase | Optimization        | Expected CUDA | Target Speedup | Actual |
|-------|---------------------|---------------|----------------|--------|
| 0     | Baseline            | 625ms         | 1.0x           | TBD    |
| 1     | Bipartite Fast Path | ~380ms        | 1.6x           | TBD    |
| 2     | Mask Indexing       | ~340ms        | 1.8x           | TBD    |
| 3     | Buffer Reuse        | ~320ms        | 1.95x          | TBD    |
| 4     | Device Transfers    | ~310ms        | 2.0x           | TBD    |
| 5     | Cache Edges         | ~305ms        | 2.05x          | TBD    |

Success Criteria: Achieve ≥2.0x speedup (≤312ms CUDA time)

---
Rollback Procedure

If any optimization fails:

1. STOP - Do not proceed to next optimization
2. Check test output for specific failure
3. If quick fix not possible, revert:
git checkout HEAD~1 torchcell/data/graph_processor.py
4. Re-run equivalence test to confirm rollback
5. Document issue before re-attempting

---
Fresh Session Startup Checklist

When starting work on this optimization:

- Read this entire document
- Verify environment:
cd /home/michaelvolk/Documents/projects/torchcell
conda activate torchcell
- Check directory structure exists:
ls /scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005/
- Verify baseline reference exists:
ls /scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005/reference_baseline.pkl
- Test baseline equivalence:
python tests/torchcell/data/test_graph_processor_equivalence.py
- Check current optimization phase from git log:
git log --oneline -10 | grep -i "optimization"
- Continue from next uncompleted phase

---
Important Notes

1. Never skip equivalence testing - This ensures correctness
2. Profile after every change - Track incremental improvements
3. Save reference data - Enables debugging if issues arise
4. Use stable config - Ensures fair performance comparisons
5. Atomic commits - Each optimization is reversible
6. Document everything - Future maintainers need context

---
Contact and Resources

- Profile Output Directory: /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/profiler_output/
- SLURM Outputs: /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/
- Test Data: /scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005/

---
Document Version: 1.0Last Updated: October 2024Target Completion: 2x CUDA Speedup for SubgraphRepresentation



