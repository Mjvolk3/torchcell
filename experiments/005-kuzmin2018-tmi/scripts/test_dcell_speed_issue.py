#!/usr/bin/env python
# experiments/005-kuzmin2018-tmi/scripts/test_dcell_speed_issue.py
# Profiling script for DCell model to identify performance bottlenecks

import os
import os.path as osp
import torch
import torch.nn as nn
import time
from torch.profiler import profile, record_function, ProfilerActivity
from dotenv import load_dotenv
import argparse
from torchcell.scratch.load_batch_005 import load_sample_data_batch
from torchcell.models.dcell import DCellModel
from torchcell.timestamp import timestamp

# Load environment variables
load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", "assets/images")


def profile_forward_pass(
    model,
    cell_graph,
    batch,
    num_repeats=10,
    use_cuda=True,
    detailed=False,
    save_trace=True,
):
    """
    Profile the forward pass of the model with detailed metrics.

    Args:
        model: The DCellModel to profile
        cell_graph: The cell graph data
        batch: A batch of data
        num_repeats: Number of times to repeat the forward pass
        use_cuda: Whether to use CUDA events for timing
        detailed: Whether to collect detailed metrics
        save_trace: Whether to save trace file for Chrome profiler

    Returns:
        Profiling results
    """
    print("\n" + "=" * 80)
    print("PROFILING DCELL FORWARD PASS")
    print("=" * 80)

    # Get device from model or input data
    try:
        device = next(model.parameters()).device
    except StopIteration:
        # If model has no parameters yet, get device from inputs
        device = cell_graph.device if hasattr(cell_graph, "device") else torch.device("cpu")

    # Basic profiling with CUDA event timing
    if use_cuda and torch.cuda.is_available():
        # First, warmup pass
        model(cell_graph, batch)
        torch.cuda.synchronize()

        # Measure with CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Time entire forward pass
        start_event.record()
        for _ in range(num_repeats):
            with torch.no_grad():
                outputs = model(cell_graph, batch)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = (
            start_event.elapsed_time(end_event) / 1000 / num_repeats
        )  # Convert to seconds
        print(f"Average forward pass time (CUDA events): {elapsed_time:.6f} seconds")

    # Detailed PyTorch profiling
    if detailed:
        print("\nDetailed profiling with PyTorch profiler:")

        with profile(
            activities=(
                [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                if torch.cuda.is_available()
                else [ProfilerActivity.CPU]
            ),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=num_repeats),
            on_trace_ready=(
                torch.profiler.tensorboard_trace_handler(
                    f"./runs/dcell_profile_{timestamp()}"
                )
                if save_trace
                else None
            ),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as prof:
            for _ in range(num_repeats + 2):  # +2 for wait and warmup
                with record_function("forward_pass"):
                    with torch.no_grad():
                        outputs = model(cell_graph, batch)
                prof.step()

        # Print summary
        print(
            prof.key_averages().table(
                sort_by=(
                    "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
                )
            )
        )

        # Save trace for Chrome view
        if save_trace:
            trace_file = f"dcell_profile_{timestamp()}.json"
            prof.export_chrome_trace(trace_file)
            print(f"\nTrace saved to {trace_file}")
            print(
                "Open Chrome browser, navigate to chrome://tracing and load this file to view detailed profile"
            )

        return prof

    # More detailed profiling of DCell components
    # Profile the stratum processing specifically which is a key optimization target
    print("\nProfiling individual stratum processing:")

    # Access stratum info from model
    strata = model.dcell.sorted_strata
    stratum_to_systems = model.dcell.stratum_to_systems

    stratum_times = {}
    for stratum in strata:
        num_subsystems = len(stratum_to_systems[stratum])

        # Time this stratum
        start_time = time.time()
        if use_cuda and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        # Simulate processing just this stratum to isolate timing
        for _ in range(num_repeats):
            with torch.no_grad():
                # Sample a few subsystems from this stratum
                sample_size = min(5, num_subsystems)
                for i in range(sample_size):
                    term_id, subsystem = stratum_to_systems[stratum][i]
                    # Create dummy input for this subsystem
                    input_size = subsystem.layers[0].weight.size(1)
                    dummy_input = torch.randn(
                        batch.num_graphs, input_size, device=device
                    )
                    # Process through subsystem
                    _ = subsystem(dummy_input)

        if use_cuda and torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            elapsed = end_event.elapsed_time(start_event) / 1000  # to seconds
            # Scale up to estimate full stratum time
            estimated_full_stratum_time = elapsed * (num_subsystems / sample_size)
            stratum_times[stratum] = estimated_full_stratum_time / num_repeats
        else:
            elapsed = time.time() - start_time
            estimated_full_stratum_time = elapsed * (num_subsystems / sample_size)
            stratum_times[stratum] = estimated_full_stratum_time / num_repeats

    # Print stratum timings
    print("\nEstimated time per stratum (scaled from sample):")
    total_stratum_time = sum(stratum_times.values())
    for stratum, elapsed in sorted(stratum_times.items()):
        subsystems_count = len(stratum_to_systems[stratum])
        print(
            f"  Stratum {stratum}: {elapsed:.6f}s ({subsystems_count} subsystems, "
            f"{elapsed/total_stratum_time*100:.1f}% of total)"
        )

    print(f"Total estimated stratum processing time: {total_stratum_time:.6f}s")
    if use_cuda and torch.cuda.is_available():
        print(f"Actual forward pass time: {elapsed_time:.6f}s")
        print(
            f"Stratum processing percentage: {total_stratum_time/elapsed_time*100:.1f}%"
        )

    # Memory usage statistics
    if torch.cuda.is_available():
        print("\nGPU Memory statistics:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

        # Try to get more detailed memory usage by tensor
        largest_tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.device.type == "cuda":
                    largest_tensors.append(
                        (obj.shape, obj.element_size() * obj.numel())
                    )
            except:
                pass

        largest_tensors.sort(key=lambda x: x[1], reverse=True)
        if largest_tensors:
            print("\nLargest CUDA tensors:")
            for i, (shape, size) in enumerate(largest_tensors[:5]):
                print(f"  {i+1}. Shape: {shape}, Size: {size / 1024**2:.2f} MB")

    return {
        "elapsed_time": (
            elapsed_time if use_cuda and torch.cuda.is_available() else None
        ),
        "stratum_times": stratum_times,
        "total_stratum_time": total_stratum_time,
    }


def analyze_batch_size_impact(model, cell_graph, batch_sizes=[8, 16, 32, 64, 128]):
    """
    Test how batch size affects performance to identify optimal batch size.
    """
    print("\n" + "=" * 80)
    print("ANALYZING BATCH SIZE IMPACT")
    print("=" * 80)

    results = {}

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        try:
            # Load data with this batch size
            dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
                batch_size=batch_size, num_workers=0, config="dcell", is_dense=False
            )

            # Move data to model device
            device = next(model.parameters()).device
            cell_graph = dataset.cell_graph.to(device)
            batch = batch.to(device)

            # Warm-up run
            with torch.no_grad():
                _ = model(cell_graph, batch)
                torch.cuda.synchronize() if torch.cuda.is_available() else None

            # Timing run
            start_time = time.time()
            with torch.no_grad():
                _ = model(cell_graph, batch)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start_time

            # Record results
            results[batch_size] = {
                "time": elapsed,
                "samples_per_second": batch_size / elapsed,
            }

            print(
                f"Time: {elapsed:.4f}s, Throughput: {batch_size / elapsed:.2f} samples/second"
            )

            # Record memory usage
            if torch.cuda.is_available():
                results[batch_size]["memory"] = torch.cuda.memory_allocated() / 1024**2
                print(f"GPU Memory: {results[batch_size]['memory']:.1f} MB")

        except Exception as e:
            print(f"Error with batch size {batch_size}: {e}")
            results[batch_size] = {"error": str(e)}

    # Find optimal batch size
    valid_results = {bs: data for bs, data in results.items() if "time" in data}
    if valid_results:
        optimal_bs = max(
            valid_results.items(), key=lambda x: x[1]["samples_per_second"]
        )[0]
        print(f"\nOptimal batch size: {optimal_bs}")
        print(
            f"Best throughput: {valid_results[optimal_bs]['samples_per_second']:.2f} samples/second"
        )

    return results


def analyze_model_configuration(batch_size=32, num_repeats=3):
    """
    Test different model configurations to identify optimal parameters.
    """
    print("\n" + "=" * 80)
    print("ANALYZING MODEL CONFIGURATIONS")
    print("=" * 80)

    # Parameters to test
    norm_types = ["batch", "layer", "none"]
    subsystem_layers = [1, 2]
    activations = [nn.Tanh(), nn.ReLU()]

    # Load data once
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=batch_size, num_workers=0, config="dcell", is_dense=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell_graph = dataset.cell_graph.to(device)
    batch_data = batch.to(device)

    results = {}

    for norm_type in norm_types:
        for subsystem_num_layers in subsystem_layers:
            for activation in activations:
                config_name = f"{norm_type}_{subsystem_num_layers}layers_{activation.__class__.__name__}"
                print(f"\nTesting configuration: {config_name}")

                try:
                    # Initialize model with these parameters
                    model_params = {
                        "gene_num": max_num_nodes,
                        "subsystem_output_min": 20,
                        "subsystem_output_max_mult": 0.3,
                        "output_size": 1,
                        "norm_type": norm_type,
                        "norm_before_act": False,
                        "subsystem_num_layers": subsystem_num_layers,
                        "activation": activation,
                        "init_range": 0.001,
                    }

                    # Create model and move to device
                    model = DCellModel(**model_params).to(device)

                    # Initialize with a forward pass
                    with torch.no_grad():
                        _ = model(cell_graph, batch_data)
                        torch.cuda.synchronize() if torch.cuda.is_available() else None

                    # Time multiple passes
                    start_time = time.time()
                    for _ in range(num_repeats):
                        with torch.no_grad():
                            _ = model(cell_graph, batch_data)
                            (
                                torch.cuda.synchronize()
                                if torch.cuda.is_available()
                                else None
                            )
                    elapsed = (time.time() - start_time) / num_repeats

                    # Get parameter count
                    param_count = sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )

                    # Record results
                    results[config_name] = {
                        "time": elapsed,
                        "params": param_count,
                        "throughput": batch_size / elapsed,
                    }

                    print(f"Average time: {elapsed:.4f}s")
                    print(f"Parameters: {param_count:,}")
                    print(f"Throughput: {batch_size / elapsed:.2f} samples/second")

                except Exception as e:
                    print(f"Error with configuration {config_name}: {e}")
                    results[config_name] = {"error": str(e)}

    # Find optimal configuration
    valid_results = {name: data for name, data in results.items() if "time" in data}
    if valid_results:
        optimal_config = min(valid_results.items(), key=lambda x: x[1]["time"])[0]
        print(f"\nOptimal configuration: {optimal_config}")
        print(f"Best time: {valid_results[optimal_config]['time']:.4f}s")
        print(
            f"Best throughput: {valid_results[optimal_config]['throughput']:.2f} samples/second"
        )

    return results


def add_dcell_profiling_hooks(model):
    """Add profiling hooks to key methods in the DCell model."""
    
    # Store original methods to call them from the wrapped methods
    original_dcell_forward = model.dcell.forward
    
    # Create a logger for profiling info
    from collections import defaultdict
    import time
    profile_logger = defaultdict(list)
    
    # Wrap the DCell forward method to time each step
    def profiled_dcell_forward(self, cell_graph, batch):
        # Time the overall forward pass
        full_start = time.time()
        
        # Time initialization if needed
        if not self.initialized:
            init_start = time.time()
            self._initialize_from_cell_graph(cell_graph)
            init_time = time.time() - init_start
            profile_logger["init_time"].append(init_time)
            print(f"DCell initialization time: {init_time:.4f}s")
        
        # Time strata processing
        strata_times = {}
        output_time = 0
        
        # Get term_gene_states preparation time
        prep_start = time.time()
        term_indices_set = torch.unique(batch["gene_ontology"].mutant_state[:, 0]).tolist()
        term_gene_states = {}
        
        for term_idx in term_indices_set:
            term_idx = term_idx if isinstance(term_idx, int) else term_idx.item()
            
            # Get genes for this term
            genes = cell_graph["gene_ontology"].term_to_gene_dict.get(term_idx, [])
            num_genes = max(1, len(genes))
            
            # Create gene state tensor
            term_gene_states[term_idx] = torch.ones(
                (batch.num_graphs, num_genes), 
                dtype=torch.float, 
                device=cell_graph.device
            )
        
        term_prep_time = time.time() - prep_start
        profile_logger["term_prep_time"].append(term_prep_time)
        
        # Process strata with timing
        strata_start = time.time()
        for stratum in self.sorted_strata:
            stratum_start = time.time()
            
            # Code here that would process just this stratum
            # We're not actually doing the computation here, just for profiling estimation
            
            stratum_time = time.time() - stratum_start
            strata_times[stratum] = stratum_time
        
        total_strata_time = time.time() - strata_start
        profile_logger["total_strata_time"].append(total_strata_time)
        
        # Now actually call the original method
        root_output, outputs = original_dcell_forward(self, cell_graph, batch)
        
        # Record full forward time
        full_time = time.time() - full_start
        profile_logger["full_forward_time"].append(full_time)
        
        # Print detailed timing info
        print(f"\nDCell Forward Pass Profile:")
        print(f"  Full forward time: {full_time:.6f}s")
        if "init_time" in profile_logger:
            print(f"  Initialization time: {profile_logger['init_time'][-1]:.6f}s")
        print(f"  Term prep time: {term_prep_time:.6f}s")
        print(f"  Strata processing time (estimate): {total_strata_time:.6f}s")
        
        if stratum in profile_logger["strata_times"]:
            for s, t in sorted(profile_logger["strata_times"][-1].items()):
                print(f"    Stratum {s}: {t:.6f}s")
        
        return root_output, outputs
    
    # Replace the methods with our profiled versions
    model.dcell.forward = profiled_dcell_forward.__get__(model.dcell, model.dcell.__class__)
    
    # Return the logger for external access
    return profile_logger


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Profile DCell model performance")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use")
    parser.add_argument(
        "--num_repeats", type=int, default=10, help="Number of repeats for timing"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Run detailed profiling"
    )
    parser.add_argument(
        "--analyze_batch", action="store_true", help="Analyze impact of batch size"
    )
    parser.add_argument(
        "--analyze_config",
        action="store_true",
        help="Analyze different model configurations",
    )
    parser.add_argument(
        "--save_trace", action="store_true", help="Save trace file for Chrome profiler"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Use torch.compile() for acceleration (requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--profile_dcell_internal",
        action="store_true",
        help="Add detailed profiling hooks inside DCell model itself",
    )
    parser.add_argument(
        "--fast_debug",
        action="store_true",
        help="Run a faster version with fewer iterations for debug purposes",
    )
    args = parser.parse_args()

    # Use CUDA if available and requested
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Import gc here to avoid circular import if not needed
    if use_cuda:
        import gc

    # Load data
    print("\nLoading test data...")
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=args.batch_size, num_workers=0, config="dcell", is_dense=False
    )

    # Move data to device
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Initialize model
    print("Initializing model...")
    model_params = {
        "gene_num": max_num_nodes,
        "subsystem_output_min": 20,
        "subsystem_output_max_mult": 0.3,
        "output_size": 1,
        "norm_type": "batch",
        "norm_before_act": False,
        "subsystem_num_layers": 1,
        "activation": nn.Tanh(),
        "init_range": 0.001,
    }
    model = DCellModel(**model_params).to(device)
    
    # Run a first forward pass to initialize the model structure
    print("Running initial forward pass to initialize model structure...")
    with torch.no_grad():
        try:
            _ = model(cell_graph, batch)
            initialized = True
        except Exception as e:
            print(f"Error during model initialization: {e}")
            initialized = False
    
    if initialized:
        # Print model statistics
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {param_count:,} parameters")
        
        # Add profiling hooks if requested
        if args.profile_dcell_internal:
            print("Adding internal profiling hooks to DCell model...")
            profile_logger = add_dcell_profiling_hooks(model)
        
        # Add custom optimizations for testing
        if hasattr(torch, 'compile') and args.use_compile:
            try:
                print("Using torch.compile() for acceleration")
                model = torch.compile(model, mode="reduce-overhead")
                print("Model successfully compiled")
            except Exception as e:
                print(f"Warning: torch.compile() failed with error: {e}")
                print("Continuing with uncompiled model")
    else:
        print("Model initialization failed. Skipping profiling.")
        return

    if args.analyze_batch:
        # Analyze impact of batch size
        batch_results = analyze_batch_size_impact(
            model, cell_graph, [8, 16, 32, 64, 128, 256, 512] if not args.fast_debug else [32, 128]
        )

    if args.analyze_config:
        # Analyze impact of model configuration
        config_results = analyze_model_configuration(args.batch_size, args.num_repeats)

    # Run detailed profiling
    num_repeats = 3 if args.fast_debug else args.num_repeats
    
    # If we're using PyTorch 2.0+ and on CUDA, let's also create an optimized version for comparison
    if hasattr(torch, 'compile') and use_cuda and not args.use_compile:
        try:
            print("\nCreating compiled model for comparison...")
            compiled_model = torch.compile(model, mode="reduce-overhead")
            
            print("Testing uncompiled model first:")
            prof_results = profile_forward_pass(
                model,
                cell_graph,
                batch,
                num_repeats=num_repeats,
                use_cuda=use_cuda,
                detailed=args.detailed,
                save_trace=args.save_trace,
            )
            
            print("\nTesting compiled model:")
            compiled_prof_results = profile_forward_pass(
                compiled_model,
                cell_graph,
                batch,
                num_repeats=num_repeats,
                use_cuda=use_cuda,
                detailed=False,  # Avoid detailed profiling with compiled model
                save_trace=False,
            )
            
            print("\nCompiled vs Uncompiled model speedup comparison:")
            if "elapsed_time" in prof_results and "elapsed_time" in compiled_prof_results:
                speedup = prof_results["elapsed_time"] / compiled_prof_results["elapsed_time"]
                print(f"  Speedup from torch.compile(): {speedup:.2f}x")
        except Exception as e:
            print(f"Error testing compiled model: {e}")
            prof_results = profile_forward_pass(
                model,
                cell_graph,
                batch,
                num_repeats=num_repeats,
                use_cuda=use_cuda,
                detailed=args.detailed,
                save_trace=args.save_trace,
            )
    else:
        # Standard profiling
        prof_results = profile_forward_pass(
            model,
            cell_graph,
            batch,
            num_repeats=num_repeats,
            use_cuda=use_cuda,
            detailed=args.detailed,
            save_trace=args.save_trace,
        )

    print("\nProfiling complete!")


if __name__ == "__main__":
    main()
