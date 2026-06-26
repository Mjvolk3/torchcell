"""CLI to delete a sharded subset of files in a directory across parallel tasks."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor


def delete_file(file_path):
    """Delete a single file if it exists, otherwise report it as missing."""
    if os.path.exists(file_path):  # Check if file exists
        os.remove(file_path)
    else:
        print(f"File {file_path} does not exist.")


def delete_files(files):
    """Delete the given files concurrently using a thread pool."""
    with ThreadPoolExecutor() as executor:
        list(
            executor.map(delete_file, files),
            total=len(files),
            desc="Deleting files",
            unit="file",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete a subset of files.")
    parser.add_argument(
        "path", help="Path to the directory containing files to delete."
    )
    parser.add_argument(
        "--task-id", type=int, help="ID of this task (starting from 1)."
    )
    parser.add_argument("--total-tasks", type=int, help="Total number of tasks.")
    args = parser.parse_args()

    folder_path = args.path
    task_id = args.task_id
    total_tasks = args.total_tasks

    if not os.path.exists(folder_path):
        print(f"The directory {folder_path} does not exist.")
        exit(1)

    all_files = []
    for root, _, files in os.walk(folder_path):
        all_files.extend([os.path.join(root, f) for f in files])

    # Divide files among tasks
    my_files = [f for i, f in enumerate(all_files) if i % total_tasks == (task_id - 1)]

    delete_files(my_files)
