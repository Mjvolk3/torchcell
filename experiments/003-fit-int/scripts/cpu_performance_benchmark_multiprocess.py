import time
import numpy as np
import concurrent.futures
import os
from typing import List, Dict
from tqdm import tqdm
def matrix_multiply(size: int) -> None:
    """Single matrix multiplication"""
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    np.dot(A, B)

def run_parallel_benchmark(matrix_size: int, num_multiplies: int, num_tasks: int) -> float:
    """Run matrix multiplications in parallel"""
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_tasks) as executor:
        # Each process gets equal share of multiplications
        executor.map(matrix_multiply, [matrix_size] * num_multiplies)
    
    return time.time() - start_time

def run_benchmarks(
    matrix_size: int,
    num_multiplies: int,
    num_repeats: int
) -> Dict:
    """Run complete benchmark suite with repeats"""
    # Get SLURM config or default to all CPUs
    num_tasks = int(os.getenv('SLURM_NTASKS', '1')) * int(os.getenv('SLURM_CPUS_PER_TASK', os.cpu_count()))
    
    print("Benchmark Configuration:")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Number of matrix multiplications: {num_multiplies}")
    print(f"Number of parallel tasks: {num_tasks}")
    print(f"Number of repeats: {num_repeats}")
    
    print("\nHardware Configuration:")
    print(f"NUMA nodes (tasks): {os.getenv('SLURM_NTASKS', '1')}")
    print(f"CPUs per task: {os.getenv('SLURM_CPUS_PER_TASK', 'Not Set')}")
    print(f"OMP threads: {os.getenv('OMP_NUM_THREADS', 'Not Set')}")
    
    print("\nNumPy Configuration:")
    np.show_config()
    
    # Run benchmark multiple times
    times = []
    for i in tqdm(range(num_repeats)):
        print(f"\nRunning repeat {i+1}/{num_repeats}")
        duration = run_parallel_benchmark(matrix_size, num_multiplies, num_tasks)
        times.append(duration)
        
    results = {
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'config': {
            'matrix_size': matrix_size,
            'num_multiplies': num_multiplies,
            'num_tasks': num_tasks,
            'num_repeats': num_repeats
        }
    }
    
    print("\nResults:")
    print(f"Raw times: {times}")
    print(f"Mean time: {results['mean']:.2f} seconds")
    print(f"Std deviation: {results['std']:.2f} seconds")
    print(f"Multiplies per second: {num_multiplies/results['mean']:.2f}")
    
    return results

if __name__ == "__main__":
    # Example usage
    results = run_benchmarks(
        matrix_size=1000,      # Size of each matrix
        num_multiplies=128*10,   # Total number of multiplications
        num_repeats=5          # Number of times to repeat entire benchmark
    )