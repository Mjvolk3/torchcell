#!/usr/bin/env python3
"""
Profile the forward pass of hetero_cell_bipartite_dango_gi_lazy model.
Identifies expensive operations and bottlenecks in the model architecture.

Usage:
    python experiments/006-kuzmin-tmi/scripts/profile_forward_pass.py

This will:
1. Load model with profile config
2. Profile individual components
3. Profile full forward pass
4. Generate detailed timing breakdowns
5. Save results and provide recommendations
"""

import os
import os.path as osp
import time
import yaml
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
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


class ModelProfiler:
    """Comprehensive profiler for GeneInteractionDango model"""

    def __init__(self, model: GeneInteractionDango, device: torch.device):
        self.model = model
        self.device = device
        self.results: Dict[str, ProfileResult] = {}

    def profile_component(
        self,
        name: str,
        func,
        *args,
        warmup: int = 3,
        iterations: int = 10,
        **kwargs
    ) -> ProfileResult:
        """Profile a specific component of the model"""

        # Warmup runs
        for _ in range(warmup):
            with torch.no_grad():
                _ = func(*args, **kwargs)

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # Profiling runs
        cpu_times = []
        cuda_times = []
        memory_usage = []

        with torch.no_grad():
            for _ in range(iterations):
                # Clear cache before each iteration
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                # Profile with torch.profiler
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                    with record_function(name):
                        _ = func(*args, **kwargs)

                # Extract timing
                key_avgs = prof.key_averages()
                for evt in key_avgs:
                    if evt.key == name:
                        cpu_times.append(evt.cpu_time / 1000.0)  # Convert to ms
                        cuda_times.append(evt.cuda_time / 1000.0 if evt.cuda_time else 0.0)
                        break

                # Get memory usage
                if torch.cuda.is_available():
                    memory_usage.append(
                        torch.cuda.max_memory_allocated(self.device) / (1024**2)  # MB
                    )

        return ProfileResult(
            name=name,
            cpu_time_ms=np.mean(cpu_times) if cpu_times else 0.0,
            cuda_time_ms=np.mean(cuda_times) if cuda_times else 0.0,
            memory_mb=np.mean(memory_usage) if memory_usage else 0.0,
            num_calls=iterations
        )

    def profile_full_forward(
        self,
        cell_graph,
        batch,
        warmup: int = 3,
        iterations: int = 10
    ) -> Dict[str, ProfileResult]:
        """Profile the full forward pass with detailed component breakdown"""

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(cell_graph, batch)

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # Detailed profiling with component breakdown
        component_times = defaultdict(lambda: {'cpu': [], 'cuda': [], 'calls': 0})

        with torch.no_grad():
            for _ in range(iterations):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                    # Profile with explicit component tracking
                    with record_function("full_forward"):
                        # Component 1: Embedding and preprocessing
                        with record_function("embedding_preprocessing"):
                            is_batch = hasattr(batch["gene"], "batch")
                            if is_batch:
                                batch_size = len(batch["gene"].ptr) - 1
                                x_gene_exp = self.model.gene_embedding.weight.expand(batch_size, -1, -1)
                                x_gene = x_gene_exp.reshape(-1, x_gene_exp.size(-1))
                            else:
                                gene_idx = torch.arange(batch["gene"].num_nodes, device=self.device)
                                x_gene = self.model.gene_embedding(gene_idx)

                            x_gene = self.model.preprocessor(x_gene)

                        x_dict = {"gene": x_gene}

                        # Component 2: Graph convolutions
                        with record_function("graph_convolutions"):
                            # Extract edge data
                            edge_index_dict = {}
                            edge_mask_dict = {}
                            for graph_name in self.model.graph_names:
                                edge_type = ("gene", graph_name, "gene")
                                if edge_type in batch.edge_types:
                                    edge_data = batch[edge_type]
                                    if hasattr(edge_data, "edge_index"):
                                        edge_index_dict[edge_type] = edge_data.edge_index.to(self.device)
                                        if hasattr(edge_data, "mask"):
                                            edge_mask_dict[edge_type] = edge_data.mask.to(self.device)
                                        else:
                                            num_edges = edge_data.edge_index.size(1)
                                            edge_mask_dict[edge_type] = torch.ones(
                                                num_edges, dtype=torch.bool, device=self.device
                                            )

                            # Apply convolutions layer by layer
                            for i, conv in enumerate(self.model.convs):
                                with record_function(f"conv_layer_{i}"):
                                    x_dict, _ = conv(x_dict, edge_index_dict, edge_mask_dict)

                        z_i = x_dict["gene"]

                        # Component 3: Global aggregation
                        with record_function("global_aggregation"):
                            if hasattr(batch["gene"], "pert_mask"):
                                batch_gene_mask = ~batch["gene"].pert_mask
                                z_i_kept = z_i[batch_gene_mask]
                                batch_idx = batch["gene"].batch[batch_gene_mask]
                            else:
                                z_i_kept = z_i
                                batch_idx = batch["gene"].batch

                            z_i_global = self.model.global_aggregator(z_i_kept, index=batch_idx)

                        # Get embeddings for perturbed genes (setup for predictors)
                        z_w = self.model.forward_single(cell_graph)
                        pert_indices = batch["gene"].perturbation_indices
                        pert_gene_embs = z_w[pert_indices]

                        # Get batch assignment
                        if hasattr(batch["gene"], "perturbation_indices_ptr"):
                            ptr = batch["gene"].perturbation_indices_ptr
                            counts = ptr[1:] - ptr[:-1]
                            batch_assign = torch.repeat_interleave(
                                torch.arange(len(counts), device=self.device),
                                counts
                            )
                        else:
                            batch_assign = None

                        # Component 4: Local predictor (only if enabled)
                        if self.model.gene_interaction_predictor is not None:
                            with record_function("local_predictor"):
                                local_interaction = self.model.gene_interaction_predictor(
                                    pert_gene_embs, batch_assign
                                )
                        else:
                            local_interaction = None

                        # Component 5: Global predictor
                        with record_function("global_predictor"):
                            # Get wildtype global
                            if hasattr(cell_graph["gene"], "pert_mask"):
                                gene_mask = ~cell_graph["gene"].pert_mask
                                z_w_kept = z_w[gene_mask]
                            else:
                                z_w_kept = z_w

                            z_w_global = self.model.global_aggregator(
                                z_w_kept,
                                index=torch.zeros(z_w_kept.size(0), device=self.device, dtype=torch.long),
                                dim_size=1
                            )

                            # Compute perturbation difference
                            batch_size = z_i_global.size(0)
                            z_w_exp = z_w_global.expand(batch_size, -1)
                            z_p_global = z_w_exp - z_i_global

                            global_interaction = self.model.global_interaction_predictor(z_p_global)

                        # Component 6: Combination
                        with record_function("combination"):
                            if global_interaction.dim() == 1:
                                global_interaction = global_interaction.unsqueeze(1)

                            if not self.model.use_local_predictor:
                                # Global-only mode
                                gene_interaction = global_interaction
                            else:
                                # Combined mode
                                if local_interaction.dim() == 1:
                                    local_interaction = local_interaction.unsqueeze(1)

                                if self.model.combination_method == "concat":
                                    gene_interaction = 0.5 * global_interaction + 0.5 * local_interaction
                                else:  # gating
                                    pred_stack = torch.cat([global_interaction, local_interaction], dim=1)
                                    gate_logits = self.model.gate_mlp(pred_stack)
                                    gate_weights = torch.nn.functional.softmax(gate_logits, dim=1)
                                    weighted_preds = pred_stack * gate_weights
                                    gene_interaction = weighted_preds.sum(dim=1, keepdim=True)

                # Parse profiling results
                key_avgs = prof.key_averages(group_by_input_shape=False)
                for evt in key_avgs:
                    if evt.key in [
                        "embedding_preprocessing", "graph_convolutions",
                        "conv_layer_0", "conv_layer_1", "conv_layer_2",
                        "global_aggregation", "local_predictor",
                        "global_predictor", "combination", "full_forward"
                    ]:
                        component_times[evt.key]['cpu'].append(evt.cpu_time / 1000.0)
                        component_times[evt.key]['cuda'].append(evt.cuda_time / 1000.0)
                        component_times[evt.key]['calls'] += 1

        # Convert to ProfileResult objects
        results = {}
        for name, times in component_times.items():
            results[name] = ProfileResult(
                name=name,
                cpu_time_ms=np.mean(times['cpu']) if times['cpu'] else 0.0,
                cuda_time_ms=np.mean(times['cuda']) if times['cuda'] else 0.0,
                memory_mb=0.0,  # Will update separately if needed
                num_calls=times['calls']
            )

        return results

    def print_summary(self, results: Dict[str, ProfileResult]):
        """Print a formatted summary of profiling results"""

        print("\n" + "="*80)
        print("FORWARD PASS PROFILING SUMMARY")
        print("="*80)

        # Sort by total time (CPU + CUDA)
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].cpu_time_ms + x[1].cuda_time_ms,
            reverse=True
        )

        # Calculate totals
        total_cpu = sum(r.cpu_time_ms for r in results.values() if "full_forward" not in r.name)
        total_cuda = sum(r.cuda_time_ms for r in results.values() if "full_forward" not in r.name)

        # Print header
        print(f"{'Component':<30} {'CPU (ms)':<12} {'CUDA (ms)':<12} {'Total (ms)':<12} {'% of Total':<10}")
        print("-"*80)

        # Print each component
        for name, result in sorted_results:
            if name == "full_forward":
                continue  # Skip the full forward as it's the sum
            total_time = result.cpu_time_ms + result.cuda_time_ms
            percent = (total_time / (total_cpu + total_cuda)) * 100 if (total_cpu + total_cuda) > 0 else 0
            print(f"{name:<30} {result.cpu_time_ms:>11.3f} {result.cuda_time_ms:>11.3f} {total_time:>11.3f} {percent:>9.1f}%")

        print("-"*80)
        print(f"{'TOTAL':<30} {total_cpu:>11.3f} {total_cuda:>11.3f} {total_cpu + total_cuda:>11.3f} {'100.0':>9}%")

        # Full forward pass comparison
        if "full_forward" in results:
            full = results["full_forward"]
            print(f"{'Full Forward (measured)':<30} {full.cpu_time_ms:>11.3f} {full.cuda_time_ms:>11.3f} "
                  f"{full.cpu_time_ms + full.cuda_time_ms:>11.3f}")

        # GPU utilization
        gpu_util = (total_cuda / (total_cpu + total_cuda)) * 100 if (total_cpu + total_cuda) > 0 else 0
        print(f"\nGPU Utilization: {gpu_util:.1f}% (CUDA time / Total time)")

        # Identify bottlenecks
        print("\n" + "="*80)
        print("BOTTLENECK ANALYSIS")
        print("="*80)

        bottlenecks = []
        for name, result in sorted_results[:3]:  # Top 3 components
            if name != "full_forward":
                total_time = result.cpu_time_ms + result.cuda_time_ms
                percent = (total_time / (total_cpu + total_cuda)) * 100
                bottlenecks.append((name, percent))

        print("Top 3 Time-Consuming Components:")
        for i, (name, percent) in enumerate(bottlenecks, 1):
            print(f"  {i}. {name}: {percent:.1f}%")

        # Recommendations
        print("\n" + "="*80)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*80)

        if gpu_util < 50:
            print("⚠️  LOW GPU UTILIZATION DETECTED")
            print("   - Consider increasing batch size")
            print("   - Reduce CPU preprocessing overhead")
            print("   - Use torch.compile() for kernel fusion")

        # Check specific components
        if "graph_convolutions" in results:
            conv_time = results["graph_convolutions"].cpu_time_ms + results["graph_convolutions"].cuda_time_ms
            conv_percent = (conv_time / (total_cpu + total_cuda)) * 100
            if conv_percent > 30:
                print(f"\n⚠️  GRAPH CONVOLUTIONS: {conv_percent:.1f}% of time")
                print("   - Consider reducing number of graphs (currently 9)")
                print("   - Simplify aggregation method (pairwise → sum/mean)")
                print("   - Reduce GIN MLP layers (currently 4)")

        if "local_predictor" in results:
            local_time = results["local_predictor"].cpu_time_ms + results["local_predictor"].cuda_time_ms
            local_percent = (local_time / (total_cpu + total_cuda)) * 100
            if local_percent > 20:
                print(f"\n⚠️  LOCAL PREDICTOR: {local_percent:.1f}% of time")
                print("   - Reduce attention heads (currently 8)")
                print("   - Reduce attention layers (currently 2)")
                print("   - Consider simpler architecture")

        # Memory recommendations
        print("\n📊 MODEL COMPLEXITY")
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   - Parameters: {param_count:,}")
        print(f"   - Hidden dim: 64")
        print(f"   - Consider reducing to 32 for ~75% parameter reduction")


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/006-kuzmin-tmi/conf"),
    config_name="hetero_cell_bipartite_dango_gi_gh_profile",
)
def main(cfg: DictConfig) -> None:
    """Main profiling function"""

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    dataset, batch, _, _ = load_sample_data_batch(
        batch_size=cfg.data_module.batch_size,
        num_workers=0,
        config="hetero_cell_bipartite",
        is_dense=False,
        use_custom_collate=True,
    )
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Build gene multigraph
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
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=cfg.cell_dataset["graphs"])

    # Initialize model
    print("\nInitializing model...")
    gene_encoder_config = OmegaConf.to_container(cfg.model.gene_encoder_config, resolve=True) if cfg.model.gene_encoder_config else {}
    local_predictor_config = OmegaConf.to_container(cfg.model.local_predictor_config, resolve=True) if hasattr(cfg.model, "local_predictor_config") and cfg.model.local_predictor_config else {}

    model = GeneInteractionDango(
        gene_num=cfg.model.gene_num,
        hidden_channels=cfg.model.hidden_channels,
        num_layers=cfg.model.num_layers,
        gene_multigraph=gene_multigraph,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
        activation=cfg.model.activation,
        gene_encoder_config=gene_encoder_config,
        local_predictor_config=local_predictor_config,
    ).to(device)

    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Hidden channels: {cfg.model.hidden_channels}")
    print(f"Num layers: {cfg.model.num_layers}")
    print(f"Aggregation method: {cfg.model.gene_encoder_config.graph_aggregation_method}")

    # Create profiler
    profiler = ModelProfiler(model, device)

    # Profile full forward pass with component breakdown
    print("\nProfiling forward pass components...")
    results = profiler.profile_full_forward(cell_graph, batch, warmup=5, iterations=20)

    # Print summary
    profiler.print_summary(results)

    # Save results to file
    timestamp_str = timestamp()
    output_file = f"profile_results_{timestamp_str}.txt"
    print(f"\nSaving detailed results to: {output_file}")

    with open(output_file, "w") as f:
        f.write("Forward Pass Profiling Results\n")
        f.write("="*80 + "\n")
        f.write(f"Model: GeneInteractionDango\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch size: {cfg.data_module.batch_size}\n")
        f.write("="*80 + "\n\n")

        # Add full YAML model configuration
        f.write("FULL MODEL CONFIGURATION (YAML)\n")
        f.write("-"*80 + "\n")
        model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
        f.write(yaml.dump(model_config_dict, default_flow_style=False, sort_keys=False))
        f.write("\n")

        # Add configuration details summary
        f.write("CONFIGURATION SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Config name: {cfg._name if hasattr(cfg, '_name') else 'hetero_cell_bipartite_dango_gi_gh_profile'}\n")
        f.write(f"Hidden channels: {cfg.model.hidden_channels}\n")
        f.write(f"Num layers: {cfg.model.num_layers}\n")
        f.write(f"Activation: {cfg.model.activation}\n")
        f.write(f"Norm: {cfg.model.norm}\n")
        f.write(f"Dropout: {cfg.model.dropout}\n")
        f.write(f"\nGene Encoder Config:\n")
        f.write(f"  Encoder type: {cfg.model.gene_encoder_config.encoder_type}\n")
        f.write(f"  Graph aggregation: {cfg.model.gene_encoder_config.graph_aggregation_method}\n")
        f.write(f"  GIN layers: {cfg.model.gene_encoder_config.gin_num_layers}\n")
        f.write(f"  GIN hidden dim: {cfg.model.gene_encoder_config.gin_hidden_dim or cfg.model.hidden_channels}\n")
        if cfg.model.gene_encoder_config.graph_aggregation_method == "pairwise_interaction":
            f.write(f"  Pairwise layers: {cfg.model.gene_encoder_config.graph_aggregation_config.pairwise_num_layers}\n")
            f.write(f"  Pairwise hidden dim: {cfg.model.gene_encoder_config.graph_aggregation_config.pairwise_hidden_dim}\n")
        f.write(f"\nLocal Predictor Config:\n")
        use_local = cfg.model.local_predictor_config.get("use_local_predictor", True)
        f.write(f"  Enabled: {use_local}\n")
        if use_local:
            f.write(f"  Attention layers: {cfg.model.local_predictor_config.num_attention_layers}\n")
            f.write(f"  Attention heads: {cfg.model.local_predictor_config.num_heads}\n")
            f.write(f"  Combination method: {cfg.model.local_predictor_config.combination_method}\n")
        else:
            f.write(f"  Mode: Global-only (local predictor disabled)\n")
        f.write(f"\nGraphs used ({len(cfg.cell_dataset.graphs)}):\n")
        for graph_name in cfg.cell_dataset.graphs:
            f.write(f"  - {graph_name}\n")
        f.write("="*80 + "\n\n")

        # Add formatted summary table
        f.write("FORWARD PASS PROFILING SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"{'Component':<30} {'CPU (ms)':<12} {'CUDA (ms)':<12} {'Total (ms)':<12} {'% of Total':<10}\n")
        f.write("-"*80 + "\n")

        # Sort and calculate totals (excluding full_forward)
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if k != "full_forward"],
            key=lambda x: x[1].cpu_time_ms + x[1].cuda_time_ms,
            reverse=True
        )
        total_cpu = sum(r.cpu_time_ms for k, r in results.items() if k != "full_forward")
        total_cuda = sum(r.cuda_time_ms for k, r in results.items() if k != "full_forward")

        # Write each component
        for name, result in sorted_results:
            total_time = result.cpu_time_ms + result.cuda_time_ms
            percent = (total_time / (total_cpu + total_cuda)) * 100 if (total_cpu + total_cuda) > 0 else 0
            f.write(f"{name:<30} {result.cpu_time_ms:>11.3f} {result.cuda_time_ms:>11.3f} "
                   f"{total_time:>11.3f} {percent:>9.1f}%\n")

        f.write("-"*80 + "\n")
        f.write(f"{'TOTAL':<30} {total_cpu:>11.3f} {total_cuda:>11.3f} "
               f"{total_cpu + total_cuda:>11.3f} {'100.0':>9}%\n")

        if "full_forward" in results:
            full = results["full_forward"]
            f.write(f"{'Full Forward (measured)':<30} {full.cpu_time_ms:>11.3f} {full.cuda_time_ms:>11.3f} "
                   f"{full.cpu_time_ms + full.cuda_time_ms:>11.3f}\n")

        gpu_util = (total_cuda / (total_cpu + total_cuda)) * 100 if (total_cpu + total_cuda) > 0 else 0
        f.write(f"\nGPU Utilization: {gpu_util:.1f}% (CUDA time / Total time)\n")
        f.write("="*80 + "\n\n")

        # Add raw timing results
        f.write("RAW PROFILING RESULTS\n")
        f.write("-"*80 + "\n")
        for name, result in sorted(results.items(), key=lambda x: x[1].cpu_time_ms + x[1].cuda_time_ms, reverse=True):
            f.write(f"{name}:\n")
            f.write(f"  CPU time: {result.cpu_time_ms:.3f} ms\n")
            f.write(f"  CUDA time: {result.cuda_time_ms:.3f} ms\n")
            f.write(f"  Total: {result.cpu_time_ms + result.cuda_time_ms:.3f} ms\n")
            f.write(f"  Calls: {result.num_calls}\n\n")

    print("\nProfiling complete!")


if __name__ == "__main__":
    main()