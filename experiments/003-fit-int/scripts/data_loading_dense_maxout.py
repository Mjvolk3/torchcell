import os
import os.path as osp
import hydra
from omegaconf import DictConfig
import torch
import gc
import time
from dotenv import load_dotenv
import psutil

from torchcell.scratch.load_batch import load_sample_data_batch


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/003-fit-int/conf"),
    config_name="hetero_cell_nsa",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()

    # Get batch size from config (you can override this via command line)
    batch_size = cfg.data_module.batch_size

    # Determine device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator.lower() == "gpu"
        else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Testing batch size: {batch_size}")

    # Record initial memory
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    if device.type == "cuda":
        initial_gpu_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)
        print(f"Initial GPU memory allocated: {initial_gpu_memory:.2f} MB")

    try:
        # Load data
        start_time = time.time()
        dataset, batch, _, _ = load_sample_data_batch(
            batch_size=batch_size,
            num_workers=cfg.data_module.num_workers,
            metabolism_graph="metabolism_bipartite",
            is_dense=True,
        )
        load_time = time.time() - start_time
        print(f"Data loading time: {load_time:.2f} seconds")

        # Move to device
        move_start_time = time.time()
        cell_graph = dataset.cell_graph.to(device)
        batch = batch.to(device)
        move_time = time.time() - move_start_time
        print(f"Time to move data to device: {move_time:.2f} seconds")

        # Print basic information about the data
        print(f"\nData Information:")
        print(f"Cell graph node types: {cell_graph.node_types}")
        print(f"Batch node types: {batch.node_types}")

        # Print node information
        print("\nNode Counts:")
        for node_type in batch.node_types:
            if hasattr(batch[node_type], "num_nodes"):
                print(f"  {node_type}: {batch[node_type].num_nodes} nodes")

        # Memory after loading
        current_memory = get_memory_usage()
        print(f"\nMemory after loading: {current_memory:.2f} MB")
        print(f"Memory increase: {current_memory - initial_memory:.2f} MB")

        if device.type == "cuda":
            current_gpu_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)
            print(f"GPU memory allocated: {current_gpu_memory:.2f} MB")
            print(
                f"GPU memory increase: {current_gpu_memory - initial_gpu_memory:.2f} MB"
            )
            print(
                f"GPU memory reserved: {torch.cuda.memory_reserved(device)/(1024*1024):.2f} MB"
            )

        # Iterate through node features
        print("\nIterating through data...")
        for node_type in batch.node_types:
            if hasattr(batch[node_type], "x") and batch[node_type].x is not None:
                x = batch[node_type].x
                print(f"  {node_type} features: shape={x.shape}, dtype={x.dtype}")

        # Iterate through edge indices
        for edge_type in batch.edge_types:
            if (
                hasattr(batch[edge_type], "edge_index")
                and batch[edge_type].edge_index is not None
            ):
                edge_index = batch[edge_type].edge_index
                print(
                    f"  {edge_type} edge_index: shape={edge_index.shape}, dtype={edge_index.dtype}"
                )

    except Exception as e:
        print(f"Error: {e}")

    # Final memory usage
    final_memory = get_memory_usage()
    print(f"\nFinal memory usage: {final_memory:.2f} MB")
    print(f"Total memory increase: {final_memory - initial_memory:.2f} MB")

    if device.type == "cuda":
        final_gpu_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)
        print(f"Final GPU memory allocated: {final_gpu_memory:.2f} MB")
        print(
            f"Total GPU memory increase: {final_gpu_memory - initial_gpu_memory:.2f} MB"
        )


if __name__ == "__main__":
    main()
