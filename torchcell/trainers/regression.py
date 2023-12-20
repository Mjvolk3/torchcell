# torchcell/trainers/regression.py
# [[torchcell.trainers.regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/trainers/regression.py
# Test file: torchcell/trainers/test_regression.py

import math

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)
from tqdm import tqdm

import wandb
from torchcell.losses import WeightedMSELoss
from torchcell.viz import fitness, genetic_interaction_score

import torchcell
style_file_path = osp.join(osp.dirname(torchcell.__file__), 'torchcell.mplstyle')
plt.style.use(style_file_path)

class RegressionTask(L.LightningModule):
    """LightningModule for training models on graph-based regression datasets."""

    def __init__(
        self,
        models: dict[str, nn.Module],
        wt: Data,
        target: str,
        wt_train_per_epoch: float = 10,
        boxplot_every_n_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        loss: str = "mse",
        batch_size: int = None,
        train_wt_node_loss: bool = False,
        train_epoch_size: int = None,
        clip_grad_norm: bool = False,
        clip_grad_norm_max_norm: float = 0.1,
        order_penalty: bool = False,
        lambda_order: float = 1.0,
        train_wt_diff: bool = True,
        **kwargs,
    ):
        super().__init__()

        # target for training
        self.target = target

        # Lightning settings, doing this for WT embedding
        self.automatic_optimization = False

        self.model_ds = models["deep_set"]
        self.model_lin = models["mlp_ref_set"]
        self.wt = wt
        self.wt_train_per_epoch = wt_train_per_epoch
        self.is_wt_init = False
        self.wt_nodes_hat, self.wt_set_hat, self.wt_global_hat = None, None, None

        # clip grad norm
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_norm_max_norm = clip_grad_norm_max_norm

        # order loss
        self.order_penalty = order_penalty
        self.lambda_order = lambda_order

        # train wt diff
        self.train_wt_diff = train_wt_diff
        if self.train_wt_diff:
            self.x_name = "x"
            self.x_batch_name = "batch"
        else:
            self.x_name = "x_pert"
            self.x_batch_name = "x_pert_batch"

        # loss
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "weighted_mse":
            mean_value = kwargs.get("fitness_mean_value")
            penalty = kwargs.get("penalty", 1.0)
            self.loss = WeightedMSELoss(mean_value=mean_value, penalty=penalty)
        elif loss == "mae":
            self.loss = nn.L1Loss()
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid."
                "Currently, supports 'mse' or 'mae' loss."
            )
        self.loss_node = nn.MSELoss()

        # Node Loss Regularization to Unit Vector
        self.train_wt_node_loss = train_wt_node_loss

        # train epoch size for wt frequency
        self.train_epoch_size = train_epoch_size

        # optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # batch_size
        self.batch_size = batch_size

        self.train_metrics = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        # Separate attributes for Pearson and Spearman correlation coefficients
        self.pearson_corr = PearsonCorrCoef()
        self.spearman_corr = SpearmanCorrCoef()

        # Used in end for whisker plot
        self.boxplot_every_n_epochs = boxplot_every_n_epochs
        self.true_values = []
        self.predictions = []

        # wandb model artifact logging
        self.last_logged_best_step = None

    def setup(self, stage=None):
        self.model_ds = self.model_ds.to(self.device)
        self.model_lin = self.model_lin.to(self.device)

    def forward(self, x, batch):
        inst_nodes_hat, inst_set_hat = self.model_ds(x, batch)
        # This case is only for sanity checking
        if self.wt_set_hat is None and self.train_wt_diff:
            self.wt_set_hat = torch.ones_like(inst_set_hat)
        if self.train_wt_diff:
            y_set_hat = self.wt_set_hat.mean(dim=0) - inst_set_hat
        else:
            y_set_hat = inst_set_hat
        y_hat = self.model_lin(y_set_hat)
        return y_hat

    def on_train_start(self):
        # Calculate the model size (number of parameters)
        parameter_size = sum(p.numel() for p in self.parameters())
        # Log it using wandb
        self.log("model/parameters_size", parameter_size)

    def train_wt(self):
        # CHECK on definition of global_step - refresh with epoch?
        if self.global_step == 0 and not self.is_wt_init:
            wt_batch = Batch.from_data_list([self.wt] * self.batch_size).to(self.device)
            self.wt_nodes_hat, self.wt_set_hat = self.model_ds(
                wt_batch.x, wt_batch.batch
            )

            self.is_wt_init = True
        if (self.global_step == 0) or self.global_step % math.ceil(
            (self.train_epoch_size + 1) / self.wt_train_per_epoch
        ) == 0:
            wt_y_hats = []
            loss_wts = []

            progress_bar = tqdm(desc="Processing", position=0, leave=True)

            wt_y_hat_mean = np.nan
            while True:
                # Global Loss
                # set up optimizer
                opt = self.optimizers()
                opt.zero_grad()

                wt_batch = Batch.from_data_list([self.wt] * self.batch_size).to(
                    self.device
                )
                self.wt_y_hat = self(wt_batch.x, wt_batch.batch)
                loss_wt = self.loss(self.wt_y_hat, wt_batch[self.target])
                self.log("wt loss", loss_wt)
                self.log("wt mean", self.wt_y_hat.detach().mean())
                self.log("wt batch fitness mean", wt_batch[self.target].mean())
                # get updated wt reference
                if self.train_wt_node_loss:
                    self.wt_nodes_hat, self.wt_set_hat = self.model_ds(
                        wt_batch.x, wt_batch.batch
                    )
                    # Node Loss
                    loss_nodes = self.loss_node(
                        self.wt_nodes_hat, torch.ones_like(self.wt_nodes_hat)
                    )
                    self.log("wt loss_nodes", loss_nodes)
                    self.manual_backward(loss_wt + loss_nodes)
                else:
                    self.manual_backward(loss_wt)

                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.parameters(), max_norm=self.clip_grad_norm_max_norm
                    )
                # finish optimization
                opt.step()
                opt.zero_grad()

                # Get updated wt reference
                self.model_ds.eval()
                with torch.no_grad():
                    self.wt_nodes_hat, self.wt_set_hat = self.model_ds(
                        wt_batch.x, wt_batch.batch
                    )
                self.model_ds.train()
                ###
                wt_y_hat_mean = self.wt_y_hat.mean().cpu().detach().numpy()
                wt_y_hats.append(wt_y_hat_mean)
                loss_wts.append(loss_wt.cpu().detach().numpy())
                progress_bar.update(1)
                if 0.99 < wt_y_hat_mean < 1.01 or self.current_epoch < 10:
                    # if self.current_epoch >= 2:
                    #     plt.plot(wt_y_hats)
                    #     plt.show()
                    #     plt.plot(loss_wts)
                    #     plt.show()
                    break
            progress_bar.close()

    def ordering_penalty(self, y_hat, y):
        # Step 1: Obtain the indices that would sort y
        _, sorted_indices = torch.sort(y, descending=False)

        # Step 2: Use these indices to rearrange y_hat
        y_hat_sorted = torch.gather(y_hat, 0, sorted_indices)

        # Step 3: Compute the penalty based on rearranged y_hat
        diffs = y_hat_sorted[1:] - y_hat_sorted[:-1]
        violations = torch.clamp(diffs, max=0)
        penalty = -torch.sum(violations)

        return penalty

    def training_step(self, batch, batch_idx):
        # Train on wt reference
        if self.train_wt_diff:
            self.train_wt()
        # Extract the batch vector
        x, y, batch_vector = (
            batch[self.x_name],
            batch[self.target],
            batch[self.x_batch_name],
        )
        # Pass the batch vector to the forward method
        y_hat = self(x, batch_vector)

        opt = self.optimizers()
        opt.zero_grad()
        if self.order_penalty:
            order_penalty_value = self.ordering_penalty(y_hat, y)
            # Hyperparameter to adjust the weight of ordering penalty
            order_penalty_loss = self.lambda_order * order_penalty_value
            self.log("order_penalty_loss", order_penalty_loss)
            loss = self.loss(y, y_hat) + self.lambda_order * order_penalty_loss
        else:
            loss = self.loss(y_hat, y)

        self.manual_backward(loss)  # error on this line
        if self.clip_grad_norm:
            nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=self.clip_grad_norm_max_norm
            )
        opt.step()
        opt.zero_grad()
        # logging
        batch_size = batch_vector[-1].item() + 1
        self.log("train_loss", loss, batch_size=batch_size, sync_dist=True)
        self.train_metrics(y_hat, y)
        # Logging the correlation coefficients
        self.log(
            "train_pearson",
            self.pearson_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train_spearman",
            self.spearman_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        # Extract the batch vector
        x, y, batch_vector = (
            batch[self.x_name],
            batch[self.target],
            batch[self.x_batch_name],
        )
        y_hat = self(x, batch_vector)
        loss = self.loss(y_hat, y)
        batch_size = batch_vector[-1].item() + 1
        self.log("val_loss", loss, batch_size=batch_size, sync_dist=True)
        self.val_metrics(y_hat, y)
        # Logging the correlation coefficients
        self.log(
            "val_pearson",
            self.pearson_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val_spearman",
            self.spearman_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.true_values.append(y.detach())
        self.predictions.append(y_hat.detach())

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()

        # Skip plotting during sanity check
        if self.trainer.sanity_checking or (
            self.current_epoch % self.boxplot_every_n_epochs != 0
        ):
            return

        # Convert lists to tensors
        true_values = torch.cat(self.true_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)

        if self.target == "fitness":
            fig = fitness.box_plot(true_values, predictions)
        elif self.target == "genetic_interaction_score":
            fig = genetic_interaction_score.box_plot(true_values, predictions)
        wandb.log({"binned_values_box_plot": wandb.Image(fig)})
        plt.close(fig)
        # Clear the stored values for the next epoch
        self.true_values = []
        self.predictions = []

        current_global_step = self.global_step
        if (
            self.trainer.checkpoint_callback.best_model_path
            and current_global_step != self.last_logged_best_step
        ):
            # Save model as a W&B artifact
            artifact = wandb.Artifact(
                name=f"model-global_step-{current_global_step}",
                type="model",
                description=f"Model on validation epoch end step - {current_global_step}",
                metadata=dict(self.hparams),
            )
            artifact.add_file(self.trainer.checkpoint_callback.best_model_path)
            wandb.log_artifact(artifact)
            self.last_logged_best_step = (
                current_global_step  # update the last logged step
            )

    def test_step(self, batch, batch_idx):
        # Extract the batch vector
        x, y, batch_vector = (
            batch[self.x_name],
            batch[self.target],
            batch[self.x_batch_name],
        )
        y_hat = self(x, batch_vector)
        loss = self.loss(y_hat, y)
        batch_size = batch_vector[-1].item() + 1
        self.log("test_loss", loss, batch_size=batch_size, sync_dist=True)
        self.test_metrics(y_hat, y)
        # Logging the correlation coefficients
        self.log(
            "test_pearson",
            self.pearson_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "test_spearman",
            self.spearman_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        params = list(self.model_ds.parameters()) + list(self.model_lin.parameters())
        optimizer = torch.optim.Adam(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
