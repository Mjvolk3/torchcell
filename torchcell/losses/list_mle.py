# torchcell/losses/list_mle
# [[torchcell.losses.list_mle]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/list_mle
# Test file: tests/torchcell/losses/test_list_mle.py

import torch
import torch.nn as nn


class ListMLELoss(nn.Module):
    def __init__(self):
        super(ListMLELoss, self).__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the ListMLE loss given predictions and true values.

        Parameters:
        - y_pred: A tensor of predictions from the model, shape (batch_size, list_size).
        - y_true: A tensor of true scores or relevance, shape (batch_size, list_size).

        Returns:
        - loss: The ListMLE loss for the batch.
        """
        if y_pred.dim() == 1 and y_true.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        # Ensure predictions and targets are float type
        y_pred = y_pred.float()
        y_true = y_true.float()

        # Sort true values in descending order and get indices
        _, true_rank = y_true.sort(descending=True, dim=1)

        # Gather predictions according to the true ranking
        y_pred_sorted = y_pred.gather(1, true_rank)

        # Compute the cumulative sum of the sorted predictions using log_softmax
        y_pred_sorted_log_softmax = torch.log_softmax(y_pred_sorted, dim=1)

        # Correctly compute the loss as the negative sum of the log softmax values across all items
        loss = -y_pred_sorted_log_softmax.sum(dim=1).mean()

        return loss


def main():
    # Example usage
    y_pred = torch.tensor([0.2, 0.8, 0.1]).unsqueeze(0)
    y_true = torch.tensor([0.1, 0.7, 0.2]).unsqueeze(0)
    loss_fn = ListMLELoss()
    loss = loss_fn(y_pred, y_true)

    print(f"ListMLE Loss: {loss.item()}")


if __name__ == "__main__":
    main()
