#!/usr/bin/env python3
"""
Profile the complete training step of GeneInteractionDango (Lazy Hetero) model.
Measures forward + loss + backward + optimizer to identify computational graph overhead.

Usage:
    python experiments/006-kuzmin-tmi/scripts/profile_gene_interaction_dango_full_step.py \
        --output_dir experiments/006-kuzmin-tmi/profiling_results/full_step_TIMESTAMP
"""

import argparse
import os
import os.path as osp
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.profiler import ProfilerActivity, profile, record_function

# Model and data imports
from torchcell.graph.graph import SCerevisiaeGraph, build_gene_multigraph
from torchcell.models.hetero_cell_bipartite_dango_gi_lazy import GeneInteractionDango
from torchcell.scratch.load_lazy_batch_006 import load_sample_data_batch
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.timestamp import timestamp


@dataclass
class ProfileResult:
    """Container for profiling results of a component"""
    name: str
    cpu_time_ms: float
    cuda_time_ms: float
    memory_mb: float
    num_calls: int
    flops: Optional[int] = None

    @property
    def avg_cpu_ms(self) -> float:
        return self.cpu_time_ms / max(1, self.num_calls)

    @property
    def avg_cuda_ms(self) -> float:
        return self.cuda_time_ms / max(1, self.num_calls)


class LazyHeteroFullStepProfiler:
    """Profiler for complete training step of GeneInteractionDango model"""

    def __init__(self, model: GeneInteractionDango, device: torch.device):
        self.model = model
        self.device = device
        self.results: Dict[str, ProfileResult] = {}

    def profile_full_training_step(
        self,
        cell_graph,
        batch,
        optimizer,
        loss_func,
        warmup: int = 5,
        iterations: int = 50
    ) -> Dict[str, ProfileResult]:
        """Profile the complete training step: forward + loss + backward + optimizer"""

        # Warmup runs
        print(f"Warming up for {warmup} iterations...")
        for i in range(warmup):
            optimizer.zero_grad()
            # GeneInteractionDango returns (predictions, outputs_dict)
            predictions, _ = self.model(cell_graph, batch)
            # Create dummy target
            target = torch.randn(predictions.shape[0], 1, device=self.device)
            loss = loss_func(predictions, target)
            loss.backward()
            optimizer.step()

            if (i + 1) % 2 == 0:
                print(f"  Warmup {i+1}/{warmup}")

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # Profiling runs
        print(f"\nProfiling {iterations} iterations...")
        component_times = defaultdict(lambda: {'cpu': [], 'cuda': [], 'calls': 0})
        memory_usage = []

        for iteration in range(iterations):
            # Clear cache before each iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Zero gradients
            optimizer.zero_grad()

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                # Phase 1: Forward pass
                with record_function("phase_1_forward"):
                    # GeneInteractionDango returns (predictions, outputs_dict)
                    predictions, _ = self.model(cell_graph, batch)

                # Phase 2: Loss computation
                with record_function("phase_2_loss"):
                    # Create dummy target (same shape as predictions)
                    target = torch.randn(predictions.shape[0], 1, device=self.device)
                    loss = loss_func(predictions, target)

                # Phase 3: Backward pass
                with record_function("phase_3_backward"):
                    loss.backward()

                # Phase 4: Optimizer step
                with record_function("phase_4_optimizer"):
                    optimizer.step()

            # Extract timing
            key_avgs = prof.key_averages(group_by_input_shape=False)
            for evt in key_avgs:
                if evt.key in ["phase_1_forward", "phase_2_loss", "phase_3_backward", "phase_4_optimizer"]:
                    component_times[evt.key]['cpu'].append(evt.cpu_time / 1000.0)
                    component_times[evt.key]['cuda'].append(evt.cuda_time / 1000.0 if evt.cuda_time else 0.0)
                    component_times[evt.key]['calls'] += 1

            # Get memory usage
            if torch.cuda.is_available():
                memory_usage.append(
                    torch.cuda.max_memory_allocated(self.device) / (1024**2)  # MB
                )

            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration+1}/{iterations}")

        # Convert to ProfileResult objects
        results = {}
        for name, times in component_times.items():
            results[name] = ProfileResult(
                name=name,
                cpu_time_ms=np.mean(times['cpu']) if times['cpu'] else 0.0,
                cuda_time_ms=np.mean(times['cuda']) if times['cuda'] else 0.0,
                memory_mb=np.mean(memory_usage) if memory_usage else 0.0,
                num_calls=len(times['cpu'])
            )

        return results

    def print_summary(self, results: Dict[str, ProfileResult], output_file: str, batch_size: int):
        """Print and save formatted summary of profiling results"""

        print("\n" + "="*80)
        print("LAZY HETERO COMPLETE TRAINING STEP PROFILING")
        print("="*80)

        # Calculate totals
        total_cpu = sum(r.cpu_time_ms for r in results.values())
        total_cuda = sum(r.cuda_time_ms for r in results.values())
        total_time = total_cpu + total_cuda

        # Print header
        print(f"{'Phase':<30} {'CPU (ms)':<12} {'CUDA (ms)':<12} {'Total (ms)':<12} {'% of Total':<10}")
        print("-"*80)

        # Print each phase in order
        phase_order = ["phase_1_forward", "phase_2_loss", "phase_3_backward", "phase_4_optimizer"]
        phase_names = {
            "phase_1_forward": "1. Forward Pass",
            "phase_2_loss": "2. Loss Computation",
            "phase_3_backward": "3. Backward Pass",
            "phase_4_optimizer": "4. Optimizer Step"
        }

        for phase_key in phase_order:
            if phase_key in results:
                result = results[phase_key]
                phase_total = result.cpu_time_ms + result.cuda_time_ms
                percent = (phase_total / total_time) * 100 if total_time > 0 else 0
                print(f"{phase_names[phase_key]:<30} {result.cpu_time_ms:>11.3f} {result.cuda_time_ms:>11.3f} "
                      f"{phase_total:>11.3f} {percent:>9.1f}%")

        print("-"*80)
        print(f"{'TOTAL':<30} {total_cpu:>11.3f} {total_cuda:>11.3f} {total_time:>11.3f} {'100.0':>9}%")

        # GPU utilization
        gpu_util = (total_cuda / total_time) * 100 if total_time > 0 else 0
        print(f"\nGPU Utilization: {gpu_util:.1f}% (CUDA time / Total time)")

        # Memory usage
        avg_memory = np.mean([r.memory_mb for r in results.values()])
        print(f"Average Peak Memory: {avg_memory:.1f} MB")

        # Save to file
        with open(output_file, "w") as f:
            f.write("LAZY HETERO COMPLETE TRAINING STEP PROFILING\n")
            f.write("="*80 + "\n")
            f.write(f"Model: GeneInteractionDango (Lazy Hetero)\n")
            f.write(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Batch size: {batch_size} (expands internally via expand() operation)\n")
            f.write(f"Iterations: {results[phase_order[0]].num_calls if phase_order[0] in results else 0}\n")
            f.write("="*80 + "\n\n")

            f.write("PHASE BREAKDOWN\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Phase':<30} {'CPU (ms)':<12} {'CUDA (ms)':<12} {'Total (ms)':<12} {'% of Total':<10}\n")
            f.write("-"*80 + "\n")

            for phase_key in phase_order:
                if phase_key in results:
                    result = results[phase_key]
                    phase_total = result.cpu_time_ms + result.cuda_time_ms
                    percent = (phase_total / total_time) * 100 if total_time > 0 else 0
                    f.write(f"{phase_names[phase_key]:<30} {result.cpu_time_ms:>11.3f} {result.cuda_time_ms:>11.3f} "
                           f"{phase_total:>11.3f} {percent:>9.1f}%\n")

            f.write("-"*80 + "\n")
            f.write(f"{'TOTAL':<30} {total_cpu:>11.3f} {total_cuda:>11.3f} {total_time:>11.3f} {'100.0':>9}%\n\n")

            f.write(f"GPU Utilization: {gpu_util:.1f}%\n")
            f.write(f"Average Peak Memory: {avg_memory:.1f} MB\n\n")

            f.write("COMPUTATIONAL GRAPH NOTES\n")
            f.write("-"*80 + "\n")
            f.write("This model uses expand() to create batch_size copies of the gene graph.\n")
            f.write("Each copy processes separately through convolutions, creating:\n")
            f.write("- Multiple gradient computation paths\n")
            f.write("- Gradient accumulation overhead in backward pass\n")
            f.write("- Increased optimizer workload\n")

        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Profile Lazy Hetero complete training step")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save profiling results")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for profiling")
    parser.add_argument("--iterations", type=int, default=50,
                       help="Number of profiling iterations")
    parser.add_argument("--warmup", type=int, default=5,
                       help="Number of warmup iterations")
    args = parser.parse_args()

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    dataset, batch, _, _ = load_sample_data_batch(
        batch_size=args.batch_size,
        num_workers=0,
        config="hetero_cell_bipartite",
        is_dense=False,
        use_custom_collate=True,
    )
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Build genome and graph
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Build gene multigraph with 9 graphs (physical, regulatory, tflink, 6 STRING networks)
    graph_names = [
        "physical",
        "regulatory",
        "tflink",
        "string12_0_neighborhood",
        "string12_0_fusion",
        "string12_0_cooccurence",
        "string12_0_coexpression",
        "string12_0_experimental",
        "string12_0_database",
    ]
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)

    # Initialize model
    print("\nInitializing GeneInteractionDango model...")
    hidden_channels = 64
    num_layers = 3

    # Gene encoder config (using GIN with sum aggregation)
    gene_encoder_config = {
        "encoder_type": "gin",
        "graph_aggregation_method": "sum",
        "gin_num_layers": 2,
    }

    # Local predictor disabled for clearer profiling
    local_predictor_config = {
        "use_local_predictor": False,
    }

    model = GeneInteractionDango(
        gene_num=dataset.cell_graph["gene"].num_nodes,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        gene_multigraph=gene_multigraph,
        dropout=0.0,
        norm="layer",
        activation="gelu",
        gene_encoder_config=gene_encoder_config,
        local_predictor_config=local_predictor_config,
    ).to(device)

    model.train()  # Training mode to enable gradients

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Initialize loss function
    loss_func = nn.MSELoss()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Hidden channels: {hidden_channels}")
    print(f"Num layers: {num_layers}")
    print(f"Graph names: {graph_names}")

    # Create profiler
    profiler = LazyHeteroFullStepProfiler(model, device)

    # Profile complete training step
    print("\nProfiling complete training step...")
    results = profiler.profile_full_training_step(
        cell_graph, batch, optimizer, loss_func,
        warmup=args.warmup,
        iterations=args.iterations
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save results
    timestamp_str = timestamp()
    output_file = osp.join(args.output_dir, f"profile_gene_interaction_dango_full_step_results_{timestamp_str}.txt")

    # Print summary
    profiler.print_summary(results, output_file, args.batch_size)

    print("\nProfiling complete!")


if __name__ == "__main__":
    main()
