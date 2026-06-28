"""Set Transformer aggregation (SAB/ISAB encoders + PMA pooling) for PyG."""

import torch
from torch import Tensor
from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.aggr.utils import (
    InducedSetAttentionBlock,
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)


class SetTransformerAggregation(Aggregation):  # type: ignore[misc]  # Aggregation is untyped (Any) in PyG stubs
    """Permutation-invariant set aggregation via Set Transformer encoders and PMA."""

    def __init__(
        self,
        channels: int,
        num_seed_points: int = 1,
        num_encoder_blocks: int = 2,
        num_decoder_blocks: int = 1,
        heads: int = 4,
        concat: bool = True,
        layer_norm: bool = False,
        dropout: float = 0.0,
        num_induced_points: int = 32,
        use_isab: bool = True,
    ):
        """Build the encoder blocks (SAB or ISAB), the PMA pooler, and decoders."""
        super().__init__()
        self.channels = channels
        self.num_seed_points = num_seed_points
        self.heads = heads
        self.concat = concat
        self.layer_norm = layer_norm
        self.dropout = dropout

        # Create encoder blocks based on use_isab flag
        self.encoders = torch.nn.ModuleList()
        for _ in range(num_encoder_blocks):
            if use_isab:
                self.encoders.append(
                    InducedSetAttentionBlock(
                        channels, num_induced_points, heads, layer_norm, dropout
                    )
                )
            else:
                self.encoders.append(
                    SetAttentionBlock(channels, heads, layer_norm, dropout)
                )

        self.pma = PoolingByMultiheadAttention(
            channels, num_seed_points, heads, layer_norm, dropout
        )

        self.decoders = torch.nn.ModuleList(
            [
                SetAttentionBlock(channels, heads, layer_norm, dropout)
                for _ in range(num_decoder_blocks)
            ]
        )

    def reset_parameters(self) -> None:
        """Reset parameters of all encoder, PMA, and decoder blocks."""
        for encoder in self.encoders:
            encoder.reset_parameters()  # type: ignore[operator]  # ModuleList yields untyped Module; block has reset_parameters
        self.pma.reset_parameters()
        for decoder in self.decoders:
            decoder.reset_parameters()  # type: ignore[operator]  # ModuleList yields untyped Module; block has reset_parameters

    @disable_dynamic_shapes(required_args=["dim_size", "max_num_elements"])  # type: ignore[untyped-decorator]  # PyG decorator is untyped (Any)
    def forward(
        self,
        x: Tensor,
        index: Tensor | None = None,
        ptr: Tensor | None = None,
        dim_size: int | None = None,
        dim: int = -2,
        max_num_elements: int | None = None,
    ) -> Tensor:
        """Aggregate grouped elements into per-set seed-point representations."""
        x, mask = self.to_dense_batch(
            x, index, ptr, dim_size, dim, max_num_elements=max_num_elements
        )

        for encoder in self.encoders:
            x = encoder(x, mask)

        x = self.pma(x, mask)

        # After PMA, x has shape [batch_size, num_seed_points, channels]
        # No mask needed for decoders since all seed points are valid
        for decoder in self.decoders:
            x = decoder(x)  # No mask parameter

        x = x.nan_to_num()
        return x.flatten(1, 2) if self.concat else x.mean(dim=1)


def main() -> None:
    """Smoke-test the SAB and ISAB variants with a forward/backward pass."""
    import torch
    from torch.optim import Adam

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Sample parameters
    channels = 16
    num_nodes = 10
    batch_size = 4

    # Create sample input
    x = torch.randn(batch_size * num_nodes, channels)  # Features
    index = torch.arange(batch_size).repeat_interleave(num_nodes)  # Batch indices

    # Test both SAB and ISAB variants
    for use_isab in [False, True]:
        print(f"\nTesting {'ISAB' if use_isab else 'SAB'} variant:")

        # Initialize model
        model = SetTransformerAggregation(
            channels=channels,
            num_seed_points=2,  # Output 2 vectors per set
            num_encoder_blocks=2,
            num_decoder_blocks=1,
            heads=4,
            concat=True,
            layer_norm=True,
            dropout=0.1,
            use_isab=use_isab,
            num_induced_points=4 if use_isab else 32,
        )

        optimizer = Adam(model.parameters(), lr=0.01)

        # Forward pass
        print("Input shape:", x.shape)
        out = model(x, index)
        print("Output shape:", out.shape)

        # Check if output makes sense
        # Should be [batch_size, num_seed_points * channels]
        expected_shape = (batch_size, 2 * channels)
        assert out.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {out.shape}"
        )

        # Test backprop
        loss = out.sum()
        loss.backward()

        # Check if gradients are computed
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                print(f"Warning: {name} has no gradient")
            else:
                print(f"{name}: grad shape {grad.shape}, grad mean {grad.mean():.3e}")

        # Test optimizer step
        optimizer.step()
        print("Optimizer step completed successfully")


if __name__ == "__main__":
    main()
