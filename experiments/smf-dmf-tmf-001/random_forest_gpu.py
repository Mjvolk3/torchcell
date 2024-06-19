import os
import os.path as osp
import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
from cuml.dask.ensemble import RandomForestRegressor as cuRF
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, progress
import dask.array as da
from sklearn.model_selection import KFold
from cuml.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import hashlib
import json
import uuid
import warnings
from torchcell.viz import fitness
from dotenv import load_dotenv
from torchcell.utils import format_scientific_notation
from scipy.stats import ConstantInputWarning
from wandb_osh.hooks import TriggerWandbSyncHook
import socket
import torchcell


import os
import multiprocessing as mp


style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


@hydra.main(version_base=None, config_path="conf", config_name="random-forest-gpu")
def main(cfg: DictConfig) -> None:
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
    hostname = socket.gethostname()
    hostname_slurm_job_id = f"{hostname}-{slurm_job_id}"
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{hostname_slurm_job_id}_{hashed_cfg}"
    experiment_dir = osp.join(
        DATA_ROOT, "wandb-experiments", str(hostname_slurm_job_id)
    )
    os.makedirs(experiment_dir, exist_ok=True)
    wandb.init(
        mode="offline",
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=group,
        tags=wandb_cfg["wandb"]["tags"],
        dir=experiment_dir,
    )

    max_size_str = format_scientific_notation(
        float(wandb.config.cell_dataset["max_size"])
    )
    is_pert = wandb.config.cell_dataset["is_pert"]
    aggregation = wandb.config.cell_dataset["aggregation"]
    node_embeddings = "_".join(wandb.config.cell_dataset["node_embeddings"])
    is_pert_str = "_pert" if is_pert else ""

    dataset_path = osp.join(
        DATA_ROOT,
        "data/torchcell/experiments/smf-dmf-tmf-traditional-ml",
        node_embeddings,
        f"{aggregation}{is_pert_str}_{max_size_str}",
    )

    n_estimators = wandb.config.random_forest["n_estimators"]
    max_depth = wandb.config.random_forest["max_depth"]
    min_samples_split = wandb.config.random_forest["min_samples_split"]

    cluster = LocalCUDACluster(n_workers=1)
    client = Client(cluster)

    for split in ["all", "train", "val", "test"]:
        X = cp.asarray(np.load(osp.join(dataset_path, split, "X.npy")))
        y = cp.asarray(np.load(osp.join(dataset_path, split, "y.npy")))

        # Scatter data ahead of time
        X = client.scatter(X)
        y = client.scatter(y)

        # Compute the data to ensure it's fully loaded
        X = X.result()
        y = y.result()

        X = da.from_array(X, chunks=(X.shape[0], X.shape[1]))
        y = da.from_array(y, chunks=(y.shape[0],))

        if (
            split == "all" and wandb.config.is_cross_validated
        ):  # Check if cross-validation is enabled
            # Perform 5-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            model = cuRF(
                client=client,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                ignore_empty_partitions=True,
                n_streams=1,  # Set n_streams to 1 for reproducibility
            )

            cv_table = wandb.Table(
                columns=["Fold", "MSE", "MAE", "R2", "Pearson", "Spearman", "Data Path"]
            )

            for fold, (train_index, val_index) in enumerate(kf.split(X), start=1):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                model.fit(X_train, y_train)

                y_pred_train = model.predict(X_train)
                y_pred_val = model.predict(X_val)

                # Convert Dask arrays to NumPy arrays
                y_train_np = y_train.compute().get()
                y_pred_train_np = y_pred_train.compute().get()
                y_val_np = y_val.compute().get()
                y_pred_val_np = y_pred_val.compute().get()

                mse_train = mean_squared_error(y_train_np, y_pred_train_np)
                mae_train = mean_absolute_error(y_train_np, y_pred_train_np)
                r2_train = r2_score(y_train_np, y_pred_train_np)
                pearson_train, _ = pearsonr(y_train_np, y_pred_train_np)
                spearman_train, _ = spearmanr(y_train_np, y_pred_train_np)

                mse_val = mean_squared_error(y_val_np, y_pred_val_np)
                mae_val = mean_absolute_error(y_val_np, y_pred_val_np)
                r2_val = r2_score(y_val_np, y_pred_val_np)
                pearson_val, _ = pearsonr(y_val_np, y_pred_val_np)
                spearman_val, _ = spearmanr(y_val_np, y_pred_val_np)

                data_path = osp.join(
                    "torchcell",
                    *(
                        dataset_path.split(os.sep)[
                            dataset_path.split(os.sep).index("torchcell") + 1 :
                        ]
                    ),
                    split,
                )
                cv_table.add_data(
                    fold,
                    float(mse_val),
                    float(mae_val),
                    float(r2_val),
                    float(pearson_val),
                    float(spearman_val),
                    data_path,
                )

                wandb.log(
                    {
                        f"fold_{fold}_train_mse": float(mse_train),
                        f"fold_{fold}_train_mae": float(mae_train),
                        f"fold_{fold}_train_r2": float(r2_train),
                        f"fold_{fold}_train_pearson": float(pearson_train),
                        f"fold_{fold}_train_spearman": float(spearman_train),
                        f"fold_{fold}_val_mse": float(mse_val),
                        f"fold_{fold}_val_mae": float(mae_val),
                        f"fold_{fold}_val_r2": float(r2_val),
                        f"fold_{fold}_val_pearson": float(pearson_val),
                        f"fold_{fold}_val_spearman": float(spearman_val),
                    }
                )

            wandb.log(
                {
                    f"{node_embeddings}_cross_validation_table": cv_table,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                }
            )

        else:
            # Use the train, val, and test splits for final evaluation
            X_train = client.scatter(
                cp.asarray(np.load(osp.join(dataset_path, "train", "X.npy")))
            )
            y_train = client.scatter(
                cp.asarray(np.load(osp.join(dataset_path, "train", "y.npy")))
            )
            X_val = client.scatter(
                cp.asarray(np.load(osp.join(dataset_path, "val", "X.npy")))
            )
            y_val = client.scatter(
                cp.asarray(np.load(osp.join(dataset_path, "val", "y.npy")))
            )
            X_test = client.scatter(
                cp.asarray(np.load(osp.join(dataset_path, "test", "X.npy")))
            )
            y_test = client.scatter(
                cp.asarray(np.load(osp.join(dataset_path, "test", "y.npy")))
            )

            # Compute the data to ensure it's fully loaded
            X_train = X_train.result()
            y_train = y_train.result()
            X_val = X_val.result()
            y_val = y_val.result()
            X_test = X_test.result()
            y_test = y_test.result()

            X_train = da.from_array(
                X_train, chunks=(X_train.shape[0], X_train.shape[1])
            )
            y_train = da.from_array(y_train, chunks=(y_train.shape[0],))
            X_val = da.from_array(X_val, chunks=(X_val.shape[0], X_val.shape[1]))
            y_val = da.from_array(y_val, chunks=(y_val.shape[0],))
            X_test = da.from_array(X_test, chunks=(X_test.shape[0], X_test.shape[1]))
            y_test = da.from_array(y_test, chunks=(y_test.shape[0],))

            model = cuRF(
                client=client,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                ignore_empty_partitions=True,
                n_streams=1,  # Set n_streams to 1 for reproducibility
            )

            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)

            # Convert Dask arrays to NumPy arrays
            y_train_np = y_train.compute().get()
            y_pred_train_np = y_pred_train.compute().get()
            y_val_np = y_val.compute().get()
            y_pred_val_np = y_pred_val.compute().get()
            y_test_np = y_test.compute().get()
            y_pred_test_np = y_pred_test.compute().get()

            mse_train = mean_squared_error(y_train_np, y_pred_train_np)
            mae_train = mean_absolute_error(y_train_np, y_pred_train_np)
            r2_train = r2_score(y_train_np, y_pred_train_np)
            pearson_train, _ = pearsonr(y_train_np, y_pred_train_np)
            spearman_train, _ = spearmanr(y_train_np, y_pred_train_np)

            mse_val = mean_squared_error(y_val_np, y_pred_val_np)
            mae_val = mean_absolute_error(y_val_np, y_pred_val_np)
            r2_val = r2_score(y_val_np, y_pred_val_np)
            pearson_val, _ = pearsonr(y_val_np, y_pred_val_np)
            spearman_val, _ = spearmanr(y_val_np, y_pred_val_np)

            mse_test = mean_squared_error(y_test_np, y_pred_test_np)
            mae_test = mean_absolute_error(y_test_np, y_pred_test_np)
            r2_test = r2_score(y_test_np, y_pred_test_np)
            pearson_test, _ = pearsonr(y_test_np, y_pred_test_np)
            spearman_test, _ = spearmanr(y_test_np, y_pred_test_np)

            wandb.log(
                {
                    "train_mse": float(mse_train),
                    "train_mae": float(mae_train),
                    "train_r2": float(r2_train),
                    "train_pearson": float(pearson_train),
                    "train_spearman": float(spearman_train),
                    "val_mse": float(mse_val),
                    "val_mae": float(mae_val),
                    "val_r2": float(r2_val),
                    "val_pearson": float(pearson_val),
                    "val_spearman": float(spearman_val),
                    "test_mse": float(mse_test),
                    "test_mae": float(mae_test),
                    "test_r2": float(r2_test),
                    "test_pearson": float(pearson_test),
                    "test_spearman": float(spearman_test),
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                }
            )

            # Create fitness boxplot for test predictions
            fig = fitness.box_plot(y_test_np, y_pred_test_np)
            wandb.log({f"test_predictions_fitness_boxplot": wandb.Image(fig)})
            plt.close(fig)

    wandb.finish()
    client.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted by user.")
        wandb.finish()
