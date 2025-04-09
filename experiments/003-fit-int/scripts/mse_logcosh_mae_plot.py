# experiments/003-fit-int/scripts/mse_logcosh_mae_plot
# [[experiments.003-fit-int.scripts.mse_logcosh_mae_plot]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/003-fit-int/scripts/mse_logcosh_mae_plot
# Test file: experiments/003-fit-int/scripts/test_mse_logcosh_mae_plot.py


import numpy as np
import matplotlib.pyplot as plt
import time
import os
import os.path as osp


def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2


def mae(y_true, y_pred):
    return np.abs(y_true - y_pred)


def logcosh(y_true, y_pred):
    return np.log(np.cosh(y_true - y_pred))


def create_loss_comparison_plot(output_dir: str) -> str:
    # Generate data
    y_true = 0
    y_pred = np.linspace(-5, 5, 1000)

    # Calculate losses
    mse_loss = mse(y_true, y_pred)
    mae_loss = mae(y_true, y_pred)
    logcosh_loss = logcosh(y_true, y_pred)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_pred, mse_loss, label="MSE", linewidth=2)
    plt.plot(y_pred, mae_loss, label="MAE", linewidth=2)
    plt.plot(y_pred, logcosh_loss, label="LogCosh", linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlabel("Prediction Error (y_pred - y_true)", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Comparison of MSE, MAE, and LogCosh Loss Functions", fontsize=14)
    plt.tight_layout()

    # Save the figure
    output_path = osp.join(output_dir, "loss_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    import os.path as osp

    # Load environment variables
    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    if not ASSET_IMAGES_DIR:
        print(
            "Warning: ASSET_IMAGES_DIR environment variable not set. Using current directory."
        )
        ASSET_IMAGES_DIR = "."

    # Create directory if it doesn't exist
    os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

    start_time = time.time()
    print(f"Starting loss function comparison... {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create and save the plot
    output_path = create_loss_comparison_plot(ASSET_IMAGES_DIR)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Analysis complete! Total time: {(time.time() - start_time):.2f} seconds")
    print(f"Loss comparison plot saved to: {output_path}")
    print(f"{'='*80}")
