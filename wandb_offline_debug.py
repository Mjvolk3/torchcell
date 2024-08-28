#wandb_offline_debug.py
import wandb
import random
import json
import os
import time


def run_experiment(mode):
    # Initialize wandb
    run = wandb.init(project="wandb-offline-debug", mode=mode)

    # Log some random data
    for step in range(100):
        data = {f"metric_{i}": random.random() for i in range(5)}
        wandb.log(data)

    # Print the summary
    print(f"\nMode: {mode}")
    print("Summary:")
    summary_dict = wandb.run.summary._as_dict()
    print(json.dumps(summary_dict, indent=2))

    wandb.finish()
    return run.dir


def print_summary_contents(run_dir):
    summary_file_path = os.path.join(run_dir, "wandb-summary.json")
    if os.path.exists(summary_file_path):
        print(f"\nContents of {summary_file_path}:")
        with open(summary_file_path, "r") as f:
            print(f.read())
    else:
        print(f"\nSummary file not found at {summary_file_path}")


if __name__ == "__main__":
    # Run in online mode
    online_run_dir = run_experiment("online")

    # Run in offline mode
    offline_run_dir = run_experiment("offline")

    # Print contents of both summary files after runs finish
    print("\n--- Final Summary Contents ---")
    print_summary_contents(online_run_dir)
    print_summary_contents(offline_run_dir)
