import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class SupCRLoss(BaseMetricLossFunction):
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)

    def compute_loss(self, embeddings, labels, indices_tuple):
        mat = self.distance(embeddings)
        if not self.distance.is_inverted:
            mat = -mat

        pos_mask, neg_mask = lmu.get_all_pairs_indices(labels)

        if pos_mask.bool().any() and neg_mask.bool().any():
            mat = mat / self.temperature

            mat_max, _ = mat.max(dim=1, keepdim=True)
            mat = mat - mat_max.detach()  # for numerical stability

            denominator = lmu.logsumexp(
                mat, keep_mask=(pos_mask + neg_mask).bool(), add_one=False, dim=1
            )

            log_prob = mat - denominator

            mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
                pos_mask.sum(dim=1) + c_f.small_val(mat.dtype)
            )

            return {
                "loss": {
                    "losses": -mean_log_prob_pos,
                    "indices": c_f.torch_arange_from_size(mat),
                    "reduction_type": "element",
                }
            }
        return self.zero_losses()

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_default_distance(self):
        return CosineSimilarity()


# Main function
if __name__ == "__main__":
    from torchcell.models import DeepSet

    # Simulated data
    batch_size = 32
    input_dim = 128
    hidden_channels = 64
    output_dim = 32
    num_node_layers = 2
    num_set_layers = 2

    # Create a deep_set model
    model = DeepSet(
        in_channels=input_dim,
        hidden_channels=hidden_channels,
        out_channels=output_dim,
        num_node_layers=num_node_layers,
        num_set_layers=num_set_layers,
    )

    # Generate random input data and labels
    input_data = torch.randn(batch_size, input_dim)
    batch = torch.tensor([0] * 16 + [1] * 16)  # Simulated batch indices
    labels = torch.randn(batch_size)

    # Forward pass through the deep_set model
    node_embeddings, set_embeddings = model(input_data, batch)
    print("set_embedding shape:", set_embeddings.shape)

    # Compute SupCR loss
    loss_func = SupCRLoss()
    loss = loss_func(set_embeddings, labels, None)

    # Backward pass
    loss["loss"]["losses"].mean().backward()

    print("SupCR Loss:", loss)
