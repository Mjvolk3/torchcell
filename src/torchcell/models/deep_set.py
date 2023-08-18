import torch
import torch.nn as nn
from torch_scatter import scatter_add


class DeepSet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        instance_layers: list[int],
        set_layers: list[int],
        output_activation: str = None,  # Default to no activation
    ):
        super().__init__()

        # Instance Layers
        instance_modules = []
        in_features = input_dim
        for out_features in instance_layers:
            instance_modules.append(nn.Linear(in_features, out_features))
            instance_modules.append(nn.ReLU())
            in_features = out_features
        self.instance_layers = nn.Sequential(*instance_modules)

        # Set Layers
        set_modules = []
        for out_features in set_layers:
            set_modules.append(nn.Linear(in_features, out_features))
            set_modules.append(nn.ReLU())
            in_features = out_features
        self.set_layers = nn.Sequential(*set_modules)

        # Output layer
        self.output_layer = nn.Linear(in_features, 1)

        # Output activation
        if output_activation == "relu":
            self.output_activation = nn.ReLU()
        elif output_activation == "softplus":
            self.output_activation = nn.Softplus()
        else:
            self.output_activation = None  # No activation if not specified

    def forward(self, x, batch):
        # x should be of shape [num_nodes, input_dim]
        # batch is the batch vector provided by PyG
        x_transformed = self.instance_layers(x)

        # Sum over nodes belonging to the same graph using scatter_add
        x_summed = scatter_add(x_transformed, batch, dim=0)

        x_processed = self.set_layers(x_summed)
        out = self.output_layer(x_processed)

        if self.output_activation:
            out = self.output_activation(out)

        return out.squeeze(-1)


def main():
    # Example usage:
    input_dim = 10
    instance_layers = [64, 32]
    set_layers = [16, 8]
    model = DeepSet(
        input_dim, instance_layers, set_layers
    )  # No output activation specified
    x = torch.rand(
        5, 20, input_dim
    )  # 5 sets, each with 20 instances, and 10 features per instance
    output = model(x)
    print(output)


if __name__ == "__main__":
    main()
