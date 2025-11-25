#!/usr/bin/env python3
"""
Profile the forward pass of Dango model with STRING12.0 networks.
Identifies expensive operations and bottlenecks for comparison with GeneInteractionDango.

Usage:
    python experiments/006-kuzmin-tmi/scripts/profile_dango_forward_pass.py \\
        --config-name dango_kuzmin2018_tmi_string12_0_profile

This will:
1. Load Dango model with STRING12.0 networks
2. Profile individual components (DangoPreTrain, MetaEmbedding, HyperSAGNN)
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
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.profiler import ProfilerActivity, profile, record_function

# Model and data imports
from torchcell.graph.graph import SCerevisiaeGraph, build_gene_multigraph
from torchcell.models.dango import Dango
from torchcell.scratch.load_batch_005 import load_sample_data_batch
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


class DangoProfiler:
    """Comprehensive profiler for Dango model"""

    def __init__(self, model: Dango, device: torch.device):
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
                    with record_function("dango_full_forward"):
                        # Component 1: DangoPreTrain
                        with record_function("dango_pretrain_total"):
                            # Get gene node indices
                            gene_data = cell_graph["gene"]
                            num_nodes = gene_data.num_nodes
                            node_indices = torch.arange(num_nodes, device=self.device)

                            # Initial embedding (shared across networks)
                            with record_function("dango_pretrain_embedding"):
                                x_init = self.model.pretrain_model.gene_embedding(node_indices)

                            # Process each network separately
                            embeddings = {}
                            for edge_type in self.model.pretrain_model.edge_types:
                                edge_key = ("gene", edge_type, "gene")

                                if edge_key in cell_graph.edge_types:
                                    edge_index = cell_graph[edge_key].edge_index

                                    # First SAGEConv layer
                                    with record_function(f"dango_pretrain_{edge_type}_layer1"):
                                        h1 = self.model.pretrain_model.layer1_convs[edge_type](x_init, edge_index)
                                        h1 = F.relu(h1)

                                    # Second SAGEConv layer
                                    with record_function(f"dango_pretrain_{edge_type}_layer2"):
                                        h2 = self.model.pretrain_model.layer2_convs[edge_type](h1, edge_index)
                                        h2 = F.relu(h2)

                                    # Reconstruction layer
                                    with record_function(f"dango_pretrain_{edge_type}_reconstruction"):
                                        recon = self.model.pretrain_model.recon_layers[edge_type](h2)

                                    embeddings[edge_type] = h2
                                else:
                                    embeddings[edge_type] = torch.zeros_like(x_init)

                        # Component 2: MetaEmbedding
                        with record_function("dango_meta_embedding_total"):
                            # Stack embeddings
                            with record_function("dango_meta_embedding_stack"):
                                embeddings_list = list(embeddings.values())
                                stacked_embeddings = torch.stack(embeddings_list, dim=1)
                                num_nodes_meta, num_networks, hidden_channels = stacked_embeddings.shape
                                reshaped_embeddings = stacked_embeddings.view(-1, hidden_channels)

                            # MLP attention
                            with record_function("dango_meta_embedding_mlp"):
                                attention_scores = self.model.meta_embedding.attention_mlp(reshaped_embeddings)
                                attention_scores = attention_scores.view(num_nodes_meta, num_networks)
                                attention_weights = F.softmax(attention_scores, dim=1)

                            # Weighted sum
                            with record_function("dango_meta_embedding_combine"):
                                attention_weights_expanded = attention_weights.unsqueeze(-1)
                                integrated_embeddings = (stacked_embeddings * attention_weights_expanded).sum(dim=1)

                        # Component 3: HyperSAGNN
                        with record_function("dango_hyper_sagnn_total"):
                            # Get perturbed embeddings
                            perturbed_embeddings = integrated_embeddings[batch["gene"].perturbation_indices]
                            batch_indices = batch["gene"].perturbation_indices_batch

                            # Static embedding
                            with record_function("dango_hyper_sagnn_static"):
                                static_embeddings = self.model.hyper_sagnn.static_embedding(perturbed_embeddings)

                            # Create attention masks
                            with record_function("dango_hyper_sagnn_mask"):
                                total_nodes = perturbed_embeddings.size(0)
                                same_set_mask = batch_indices.unsqueeze(-1) == batch_indices.unsqueeze(0)
                                self_mask = torch.eye(total_nodes, dtype=torch.bool, device=self.device)
                                valid_attention_mask = same_set_mask & ~self_mask

                            # First attention layer
                            with record_function("dango_hyper_sagnn_attention_layer1"):
                                Q1 = self.model.hyper_sagnn.Q1(perturbed_embeddings)
                                K1 = self.model.hyper_sagnn.K1(perturbed_embeddings)
                                V1 = self.model.hyper_sagnn.V1(perturbed_embeddings)

                                # Reshape for multi-head
                                num_heads = self.model.hyper_sagnn.num_heads
                                head_dim = self.model.hyper_sagnn.head_dim
                                Q1 = Q1.view(total_nodes, num_heads, head_dim).permute(1, 0, 2)
                                K1 = K1.view(total_nodes, num_heads, head_dim).permute(1, 0, 2)
                                V1 = V1.view(total_nodes, num_heads, head_dim).permute(1, 0, 2)

                                # Attention computation
                                attention1 = torch.matmul(Q1, K1.transpose(-2, -1)) / (head_dim**0.5)
                                expanded_mask = valid_attention_mask.unsqueeze(0).expand(num_heads, -1, -1)
                                attention1.masked_fill_(~expanded_mask, -float("inf"))
                                attention_weights1 = F.softmax(attention1, dim=-1)
                                attention_weights1 = torch.nan_to_num(attention_weights1, nan=0.0)

                                out1 = torch.matmul(attention_weights1, V1)
                                out1 = out1.permute(1, 0, 2).contiguous().view(total_nodes, -1)
                                out1 = self.model.hyper_sagnn.O1(out1)
                                dynamic_embeddings = self.model.hyper_sagnn.beta1 * out1 + perturbed_embeddings

                            # Second attention layer
                            with record_function("dango_hyper_sagnn_attention_layer2"):
                                Q2 = self.model.hyper_sagnn.Q2(dynamic_embeddings)
                                K2 = self.model.hyper_sagnn.K2(dynamic_embeddings)
                                V2 = self.model.hyper_sagnn.V2(dynamic_embeddings)

                                Q2 = Q2.view(total_nodes, num_heads, head_dim).permute(1, 0, 2)
                                K2 = K2.view(total_nodes, num_heads, head_dim).permute(1, 0, 2)
                                V2 = V2.view(total_nodes, num_heads, head_dim).permute(1, 0, 2)

                                attention2 = torch.matmul(Q2, K2.transpose(-2, -1)) / (head_dim**0.5)
                                attention2.masked_fill_(~expanded_mask, -float("inf"))
                                attention_weights2 = F.softmax(attention2, dim=-1)
                                attention_weights2 = torch.nan_to_num(attention_weights2, nan=0.0)

                                out2 = torch.matmul(attention_weights2, V2)
                                out2 = out2.permute(1, 0, 2).contiguous().view(total_nodes, -1)
                                out2 = self.model.hyper_sagnn.O2(out2)
                                dynamic_embeddings = self.model.hyper_sagnn.beta2 * out2 + dynamic_embeddings

                            # Prediction
                            with record_function("dango_hyper_sagnn_prediction"):
                                from torch_scatter import scatter_mean
                                squared_diff = (dynamic_embeddings - static_embeddings) ** 2
                                node_scores = self.model.hyper_sagnn.prediction_layer(squared_diff).squeeze(-1)

                                unique_batches = torch.unique(batch_indices)
                                num_batches = len(unique_batches)
                                interaction_scores = scatter_mean(
                                    node_scores, batch_indices, dim=0, dim_size=num_batches
                                )

                # Parse profiling results
                key_avgs = prof.key_averages(group_by_input_shape=False)
                for evt in key_avgs:
                    # Track all component times
                    if any(keyword in evt.key for keyword in [
                        "dango_pretrain", "dango_meta", "dango_hyper", "dango_full"
                    ]):
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
        print("DANGO FORWARD PASS PROFILING SUMMARY")
        print("="*80)

        # Sort by total time (CPU + CUDA)
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].cpu_time_ms + x[1].cuda_time_ms,
            reverse=True
        )

        # Calculate totals (excluding full_forward)
        total_cpu = sum(r.cpu_time_ms for k, r in results.items() if "full_forward" not in k)
        total_cuda = sum(r.cuda_time_ms for k, r in results.items() if "full_forward" not in k)

        # Print header
        print(f"{'Component':<40} {'CPU (ms)':<12} {'CUDA (ms)':<12} {'Total (ms)':<12} {'% of Total':<10}")
        print("-"*90)

        # Print each component
        for name, result in sorted_results:
            if name == "dango_full_forward":
                continue  # Skip the full forward as it's the sum
            total_time = result.cpu_time_ms + result.cuda_time_ms
            percent = (total_time / (total_cpu + total_cuda)) * 100 if (total_cpu + total_cuda) > 0 else 0

            # Indent sub-components
            display_name = name
            if "_layer" in name or "_reconstruction" in name or "embedding" in name.split("_")[-1]:
                display_name = "  " + name

            print(f"{display_name:<40} {result.cpu_time_ms:>11.3f} {result.cuda_time_ms:>11.3f} {total_time:>11.3f} {percent:>9.1f}%")

        print("-"*90)
        print(f"{'TOTAL':<40} {total_cpu:>11.3f} {total_cuda:>11.3f} {total_cpu + total_cuda:>11.3f} {'100.0':>9}%")

        # Full forward pass comparison
        if "dango_full_forward" in results:
            full = results["dango_full_forward"]
            print(f"{'Full Forward (measured)':<40} {full.cpu_time_ms:>11.3f} {full.cuda_time_ms:>11.3f} "
                  f"{full.cpu_time_ms + full.cuda_time_ms:>11.3f}")

        # GPU utilization
        gpu_util = (total_cuda / (total_cpu + total_cuda)) * 100 if (total_cpu + total_cuda) > 0 else 0
        print(f"\nGPU Utilization: {gpu_util:.1f}% (CUDA time / Total time)")

        # Identify bottlenecks
        print("\n" + "="*80)
        print("BOTTLENECK ANALYSIS")
        print("="*80)

        bottlenecks = []
        for name, result in sorted_results[:5]:  # Top 5 components
            if name != "dango_full_forward":
                total_time = result.cpu_time_ms + result.cuda_time_ms
                percent = (total_time / (total_cpu + total_cuda)) * 100
                bottlenecks.append((name, percent))

        print("Top 5 Time-Consuming Components:")
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
        if "dango_pretrain_total" in results:
            pretrain_time = results["dango_pretrain_total"].cpu_time_ms + results["dango_pretrain_total"].cuda_time_ms
            pretrain_percent = (pretrain_time / (total_cpu + total_cuda)) * 100
            if pretrain_percent > 40:
                print(f"\n⚠️  DANGO PRETRAIN: {pretrain_percent:.1f}% of time")
                print("   - Processing 6 STRING12.0 networks with SAGEConv")
                print("   - Consider reducing number of networks")
                print("   - Try simpler convolution (GCN vs SAGEConv)")

        if "dango_hyper_sagnn_total" in results:
            sagnn_time = results["dango_hyper_sagnn_total"].cpu_time_ms + results["dango_hyper_sagnn_total"].cuda_time_ms
            sagnn_percent = (sagnn_time / (total_cpu + total_cuda)) * 100
            if sagnn_percent > 30:
                print(f"\n⚠️  HYPER SAGNN: {sagnn_percent:.1f}% of time")
                print("   - Reduce attention heads (currently 4)")
                print("   - Consider single attention layer instead of 2")
                print("   - Attention mask computation may be expensive")

        # Memory recommendations
        print("\n📊 MODEL COMPLEXITY")
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   - Parameters: {param_count:,}")
        print(f"   - Hidden dim: {self.model.hidden_channels}")


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/006-kuzmin-tmi/conf"),
    config_name="dango_kuzmin2018_tmi_string12_0_profile",
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
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=cfg.data_module.batch_size,
        num_workers=0,
        config="dango_string12_0",
        is_dense=False,
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

    # Get edge types from config
    edge_types = cfg.cell_dataset["graphs"]
    print(f"\nUsing {len(edge_types)} STRING12.0 networks:")
    for edge_type in edge_types:
        print(f"  - {edge_type}")

    # Initialize model
    print("\nInitializing Dango model...")
    model = Dango(
        gene_num=max_num_nodes,
        edge_types=edge_types,
        hidden_channels=cfg.model.hidden_channels,
        num_heads=cfg.model.num_heads,
    ).to(device)

    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Hidden channels: {cfg.model.hidden_channels}")
    print(f"Num attention heads: {cfg.model.num_heads}")

    # Create profiler
    profiler = DangoProfiler(model, device)

    # Profile full forward pass with component breakdown
    print("\nProfiling forward pass components...")
    results = profiler.profile_full_forward(cell_graph, batch, warmup=5, iterations=20)

    # Print summary
    profiler.print_summary(results)

    # Save results to file
    timestamp_str = timestamp()
    output_file = f"profile_dango_results_{timestamp_str}.txt"
    print(f"\nSaving detailed results to: {output_file}")

    with open(output_file, "w") as f:
        f.write("Dango Forward Pass Profiling Results\n")
        f.write("="*80 + "\n")
        f.write(f"Model: Dango\n")
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
        f.write(f"Config name: {cfg._name if hasattr(cfg, '_name') else 'dango_kuzmin2018_tmi_string12_0_profile'}\n")
        f.write(f"Hidden channels: {cfg.model.hidden_channels}\n")
        f.write(f"Num attention heads: {cfg.model.num_heads}\n")
        f.write(f"\nGraphs used ({len(edge_types)}):\n")
        for edge_type in edge_types:
            f.write(f"  - {edge_type}\n")
        f.write("="*80 + "\n\n")

        # Add formatted summary table
        f.write("FORWARD PASS PROFILING SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"{'Component':<40} {'CPU (ms)':<12} {'CUDA (ms)':<12} {'Total (ms)':<12} {'% of Total':<10}\n")
        f.write("-"*90 + "\n")

        # Sort and calculate totals (excluding full_forward)
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if k != "dango_full_forward"],
            key=lambda x: x[1].cpu_time_ms + x[1].cuda_time_ms,
            reverse=True
        )
        total_cpu = sum(r.cpu_time_ms for k, r in results.items() if k != "dango_full_forward")
        total_cuda = sum(r.cuda_time_ms for k, r in results.items() if k != "dango_full_forward")

        # Write each component
        for name, result in sorted_results:
            total_time = result.cpu_time_ms + result.cuda_time_ms
            percent = (total_time / (total_cpu + total_cuda)) * 100 if (total_cpu + total_cuda) > 0 else 0

            # Indent sub-components
            display_name = name
            if "_layer" in name or "_reconstruction" in name or "embedding" in name.split("_")[-1]:
                display_name = "  " + name

            f.write(f"{display_name:<40} {result.cpu_time_ms:>11.3f} {result.cuda_time_ms:>11.3f} "
                   f"{total_time:>11.3f} {percent:>9.1f}%\n")

        f.write("-"*90 + "\n")
        f.write(f"{'TOTAL':<40} {total_cpu:>11.3f} {total_cuda:>11.3f} "
               f"{total_cpu + total_cuda:>11.3f} {'100.0':>9}%\n")

        if "dango_full_forward" in results:
            full = results["dango_full_forward"]
            f.write(f"{'Full Forward (measured)':<40} {full.cpu_time_ms:>11.3f} {full.cuda_time_ms:>11.3f} "
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
    print(f"\nCompare with GeneInteractionDango profiling results to see architectural differences:")
    print(f"  - Dango: SAGEConv (6 networks) vs GeneInteractionDango: GIN (9 graphs)")
    print(f"  - Dango: Meta-embedding vs GeneInteractionDango: Pairwise interaction")
    print(f"  - Dango: HyperSAGNN vs GeneInteractionDango: Local/Global predictors")


if __name__ == "__main__":
    main()
