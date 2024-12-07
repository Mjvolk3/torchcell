import concurrent.futures
import time
import numpy as np
import multiprocessing
import os
import psutil

def set_cpu_affinity(cpu_list):
    """Pin process to specific CPUs"""
    os.sched_setaffinity(0, cpu_list)

def get_numa_cpu_mapping():
    """Get CPU to NUMA node mapping"""
    numa_mapping = {}
    cpu_info = psutil.Process().cpu_affinity()
    cpus_per_node = len(cpu_info) // 4  # 4 NUMA nodes
    
    for node in range(4):
        start_cpu = node * cpus_per_node
        numa_mapping[node] = list(range(start_cpu, start_cpu + cpus_per_node))
    
    return numa_mapping

def matrix_multiply_numa(args):
    """NUMA-aware matrix multiplication"""
    size, numa_cpus = args
    set_cpu_affinity(numa_cpus)
    
    # Allocate matrices on local NUMA node
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    np.dot(A, B)

def benchmark(matrix_size: int) -> float:
    """Run NUMA-aware benchmark"""
    start_time = time.time()
    numa_mapping = get_numa_cpu_mapping()
    
    # Create one process per NUMA node, with 32 multiplications per node
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        args = [(matrix_size, cpus) for cpus in numa_mapping.values()] * 32  # Multiply by 32 to match original workload
        executor.map(matrix_multiply_numa, args)
    
    return time.time() - start_time

def run_benchmark(repeats: int, matrix_size: int) -> list:
    """Run multiple benchmarks"""
    print(f"Matrix size: {matrix_size}")
    print(f"repeats: {repeatsa}")
    times = [benchmark(matrix_size) for _ in range(repeats)]
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Raw Data: {times}")
    print(f"Mean Time: {mean_time:.2f} seconds")
    print(f"Standard Deviation: {std_time:.2f} seconds")
    
    return times

if __name__ == "__main__":
    run_benchmark(repeats=5, matrix_size=4000)