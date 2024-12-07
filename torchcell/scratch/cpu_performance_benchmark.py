import concurrent.futures
import time
import numpy as np
import multiprocessing as mp
import os

def get_cpu_count():
    """Get the number of CPUs allocated by SLURM or fall back to system CPU count."""
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus is not None:
        return int(slurm_cpus)
    return mp.cpu_count()

def matrix_multiply(size_seed: tuple) -> None:
    """Performs matrix multiplication of two random matrices with a set seed."""
    size, seed = size_seed
    np.random.seed(seed)
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    np.dot(A, B)  # Perform the matrix multiplication

def benchmark(num_jobs: int, matrix_size: int, base_seed: int) -> float:
    """Run the benchmark with a specified number of jobs and matrix size."""
    start_time = time.time()
    
    # Generate seeds for each process
    seeds = [base_seed + i for i in range(num_jobs)]
    
    # Use allocated cores with ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
        executor.map(matrix_multiply, [(matrix_size, seed) for seed in seeds])
    
    end_time = time.time()
    return end_time - start_time

def run_benchmark(repeats: int, matrix_size: int, base_seed: int) -> list:
    """Run the benchmark multiple times and return the results."""
    cpu_count = get_cpu_count()
    print(f"Available CPUs (from SLURM or system): {cpu_count}")
    times = []
    
    for i in range(repeats):
        print(f"Running benchmark {i+1}/{repeats}")
        duration = benchmark(cpu_count, matrix_size, base_seed + i * cpu_count)
        times.append(duration)
    
    return times

if __name__ == "__main__":
    # Running benchmark 5 times with matrix size 4000
    repeats = 5
    matrix_size = 4000
    base_seed = 42  # Set a base seed for reproducibility
    
    print(f"Matrix multiplication benchmark")
    print(f"Repeats: {repeats}")
    print(f"Matrix size: {matrix_size}")
    print(f"Base random seed: {base_seed}")
    
    results = run_benchmark(repeats, matrix_size, base_seed)
    
    # Round results to two decimal places
    rounded_results = [round(t, 2) for t in results]
    mean_time = round(np.mean(results), 2)
    std_time = round(np.std(results), 2)
    
    print("\nResults:")
    print(f"Raw timing data: {rounded_results} seconds")  # Rounded list
    print(f"Mean time: {mean_time} seconds")              # Rounded mean
    print(f"Standard deviation: {std_time} seconds")     # Rounded std
