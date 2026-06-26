"""Minimal LightningModule scaffold for cell-data regression training."""

import lightning as L
import torch
import torch.nn as nn

from torchcell.datamodules import CellDataModule


class SimpleModel(L.LightningModule):
    """Placeholder linear LightningModule scaffold for MSE regression."""

    def __init__(self) -> None:
        """Initialize the linear layer placeholder."""
        super().__init__()
        self.linear = nn.Linear(
            in_features=..., out_features=...
        )  # Define your model's architecture here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the linear projection of the input."""
        return self.linear(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute and return the MSE training loss for one batch."""
        # Define the training loop
        x, y = batch
        y_pred = self(x)
        loss = nn.MSELoss()(y_pred, y)  # Example loss for regression task
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Compute and log the MSE validation loss for one batch."""
        x, y = batch
        y_pred = self(x)
        val_loss = nn.MSELoss()(y_pred, y)
        self.log("val_loss", val_loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return an Adam optimizer over the module parameters."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    data_module = CellDataModule(batch_size=64)
    model = SimpleModel()

    trainer = L.Trainer(max_epochs=10, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, datamodule=data_module)
