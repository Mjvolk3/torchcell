import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# trigger_sync = TriggerWandbSyncHook()
style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


def get_n_jobs():
    if "SLURM_JOB_ID" in os.environ:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        return int(slurm_cpus) if slurm_cpus is not None else 1
    else:
        return -1


@hydra.main(version_base=None, config_path="conf", config_name="random-forest")
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

    n_jobs = get_n_jobs()

    for split in ["all", "train", "val", "test"]:
        X = np.load(osp.join(dataset_path, split, "X.npy"))
        y = np.load(osp.join(dataset_path, split, "y.npy"))

        if (
            split == "all" and wandb.config.is_cross_validated
        ):  # Check if cross-validation is enabled
            # Perform 5-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                n_jobs=n_jobs,
            )

            cv_table = wandb.Table(
                columns=["Fold", "MSE", "MAE", "R2", "Pearson", "Spearman", "Data Path"]
            )

            for fold, (train_index, val_index) in enumerate(kf.split(X), start=1):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                model.fit(X_train, y_train)
                num_params = sum(tree.tree_.node_count for tree in model.estimators_)
                wandb.log({"num_params": num_params})

                y_pred_train = model.predict(X_train)
                y_pred_val = model.predict(X_val)

                mse_train = mean_squared_error(y_train, y_pred_train)
                mae_train = mean_absolute_error(y_train, y_pred_train)
                r2_train = r2_score(y_train, y_pred_train)
                pearson_train, _ = pearsonr(y_train, y_pred_train)
                spearman_train, _ = spearmanr(y_train, y_pred_train)

                mse_val = mean_squared_error(y_val, y_pred_val)
                mae_val = mean_absolute_error(y_val, y_pred_val)
                r2_val = r2_score(y_val, y_pred_val)
                pearson_val, _ = pearsonr(y_val, y_pred_val)
                spearman_val, _ = spearmanr(y_val, y_pred_val)

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
                    fold, mse_val, mae_val, r2_val, pearson_val, spearman_val, data_path
                )

                wandb.log(
                    {
                        f"fold_{fold}_train_mse": mse_train,
                        f"fold_{fold}_train_mae": mae_train,
                        f"fold_{fold}_train_r2": r2_train,
                        f"fold_{fold}_train_pearson": pearson_train,
                        f"fold_{fold}_train_spearman": spearman_train,
                        f"fold_{fold}_val_mse": mse_val,
                        f"fold_{fold}_val_mae": mae_val,
                        f"fold_{fold}_val_r2": r2_val,
                        f"fold_{fold}_val_pearson": pearson_val,
                        f"fold_{fold}_val_spearman": spearman_val,
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
            X_train = np.load(osp.join(dataset_path, "train", "X.npy"))
            y_train = np.load(osp.join(dataset_path, "train", "y.npy"))
            X_val = np.load(osp.join(dataset_path, "val", "X.npy"))
            y_val = np.load(osp.join(dataset_path, "val", "y.npy"))
            X_test = np.load(osp.join(dataset_path, "test", "X.npy"))
            y_test = np.load(osp.join(dataset_path, "test", "y.npy"))

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
            )

            model.fit(X_train, y_train)
            num_params = sum(tree.tree_.node_count for tree in model.estimators_)
            wandb.log({"num_params": num_params})

            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)

            mse_train = mean_squared_error(y_train, y_pred_train)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            r2_train = r2_score(y_train, y_pred_train)
            pearson_train, _ = pearsonr(y_train, y_pred_train)
            spearman_train, _ = spearmanr(y_train, y_pred_train)

            mse_val = mean_squared_error(y_val, y_pred_val)
            mae_val = mean_absolute_error(y_val, y_pred_val)
            r2_val = r2_score(y_val, y_pred_val)
            pearson_val, _ = pearsonr(y_val, y_pred_val)
            spearman_val, _ = spearmanr(y_val, y_pred_val)

            mse_test = mean_squared_error(y_test, y_pred_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            r2_test = r2_score(y_test, y_pred_test)
            pearson_test, _ = pearsonr(y_test, y_pred_test)
            spearman_test, _ = spearmanr(y_test, y_pred_test)

            wandb.log(
                {
                    "train_mse": mse_train,
                    "train_mae": mae_train,
                    "train_r2": r2_train,
                    "train_pearson": pearson_train,
                    "train_spearman": spearman_train,
                    "val_mse": mse_val,
                    "val_mae": mae_val,
                    "val_r2": r2_val,
                    "val_pearson": pearson_val,
                    "val_spearman": spearman_val,
                    "test_mse": mse_test,
                    "test_mae": mae_test,
                    "test_r2": r2_test,
                    "test_pearson": pearson_test,
                    "test_spearman": spearman_test,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                }
            )

            # Create fitness boxplot for test predictions
            fig = fitness.box_plot(y_test, y_pred_test)
            wandb.log({f"test_predictions_fitness_boxplot": wandb.Image(fig)})
            plt.close(fig)
            # trigger_sync()
    wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted by user.")
        wandb.finish()
