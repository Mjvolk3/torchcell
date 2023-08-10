import torch
import torch.nn as nn


class DeepSet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        instance_layers: list[int],
        set_layers: list[int],
    ):
        super(DeepSet, self).__init__()

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

    def forward(self, x):
        # x should be of shape (batch_size, num_instances, input_dim)
        x_transformed = self.instance_layers(x)
        x_summed = x_transformed.sum(dim=1)
        x_processed = self.set_layers(x_summed)
        out = self.output_layer(x_processed)
        return out


def main():
    # Example usage:
    input_dim = 10
    instance_layers = [64, 32]
    set_layers = [16, 8]
    model = DeepSet(input_dim, instance_layers, set_layers)
    x = torch.rand(
        5, 20, input_dim
    )  # 5 sets, each with 20 instances, and 10 features per instance
    output = model(x)
    print(output)


if __name__ == "__main__":
    main()
