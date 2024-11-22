import concurrent.futures
import time
import numpy as np
import multiprocessing


def matrix_multiply(size: int) -> None:
    """Performs matrix multiplication of two random matrices."""
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    np.dot(A, B)  # Perform the matrix multiplication


def benchmark(num_jobs: int, matrix_size: int) -> float:
    """Run the benchmark with a specified number of jobs and matrix size."""
    start_time = time.time()

    # Use all cores with ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
        executor.map(matrix_multiply, [matrix_size] * num_jobs)

    end_time = time.time()
    return end_time - start_time


def run_benchmark(repeats: int, matrix_size: int) -> list:
    """Run the benchmark multiple times and return the results."""
    cpu_count = multiprocessing.cpu_count()
    times = []

    for _ in range(repeats):
        duration = benchmark(cpu_count, matrix_size)
        times.append(duration)

    return times


# Running benchmark 5 times with matrix size 1000
repeats = 1
matrix_size = 100
print(f"repeat: {repeats}, matrix_size: {matrix_size}")

results = run_benchmark(repeats, matrix_size)

# Calculate mean and standard deviation
mean_time = np.mean(results)
std_time = np.std(results)

print(f"Raw Data: {results}")
print(f"Mean Time: {mean_time:.2f} seconds")
print(f"Standard Deviation: {std_time:.2f} seconds")
