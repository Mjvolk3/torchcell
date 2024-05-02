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

import torchcell

style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
trigger_sync = TriggerWandbSyncHook()

def get_n_jobs():
    if "SLURM_JOB_ID" in os.environ:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        return int(slurm_cpus) if slurm_cpus is not None else 1
    else:
        return -1


@hydra.main(version_base=None, config_path="conf", config_name="random-forest")
def main(cfg: DictConfig) -> None:
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{slurm_job_id}_{hashed_cfg}"
    wandb.init(
        mode=wandb_cfg["wandb"].get("mode", "online"),
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=group,
        tags=wandb_cfg["wandb"]["tags"],
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

        if split == "all":
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
                y_pred = model.predict(X_val)

                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                pearson, _ = pearsonr(y_val, y_pred)
                spearman, _ = spearmanr(y_val, y_pred)

                data_path = osp.join(
                    "torchcell",
                    *(
                        dataset_path.split(os.sep)[
                            dataset_path.split(os.sep).index("torchcell") + 1 :
                        ]
                    ),
                    split,
                )
                cv_table.add_data(fold, mse, mae, r2, pearson, spearman, data_path)

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

            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)

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
                    "val_mse": mse_val,
                    "val_mae": mae_val,
                    "val_r2": r2_val,
                    "test_mse": mse_test,
                    "test_mae": mae_test,
                    "test_r2": r2_test,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                }
            )

            if not np.isnan(pearson_val):
                wandb.log({"val_pearson": pearson_val})
            else:
                wandb.log({"val_pearson": None})

            if not np.isnan(spearman_val):
                wandb.log({"val_spearman": spearman_val})
            else:
                wandb.log({"val_spearman": None})

            if not np.isnan(pearson_test):
                wandb.log({"test_pearson": pearson_test})
            else:
                wandb.log({"test_pearson": None})

            if not np.isnan(spearman_test):
                wandb.log({"test_spearman": spearman_test})
            else:
                wandb.log({"test_spearman": None})

            # Create fitness boxplot for test predictions
            fig = fitness.box_plot(y_test, y_pred_test)
            wandb.log({f"test_predictions_fitness_boxplot": wandb.Image(fig)})
            plt.close(fig)
            trigger_sync() 
    wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted by user.")
        wandb.finish()
