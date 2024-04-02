from typing import Dict

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torchcell.models import DeepSet

CONFIG_NAME = "deep_set-sweep_03"


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(version_base=None, config_path="conf", config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    # Convert the Hydra configuration to a plain dictionary
    # This step ensures that the configuration is serializable by wandb
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Ensure 'command' is excluded since it's likely not relevant for wandb logging
    cfg_dict.pop("command", None)  # Remove 'command' key if present

    print(f"Configuration: {cfg_dict}")

    # Initialize wandb project with the serialized configuration
    wandb.init(project="torchcell_smf-dmf-tmf-001", config=cfg_dict, tags=[CONFIG_NAME])

    # Extract parameter ranges directly from the configuration
    models_cfg = cfg_dict["parameters"]["models"]["parameters"]["graph"]["parameters"]
    hidden_channels_range = models_cfg["hidden_channels"]["values"]
    out_channels_range = models_cfg["out_channels"]["values"]
    num_node_layers_range = models_cfg["num_node_layers"]["values"]
    num_set_layers_range = models_cfg["num_set_layers"]["values"]

    # Iterate over all parameter combinations
    for hidden_channels in hidden_channels_range:
        for out_channels in out_channels_range:
            for num_node_layers in num_node_layers_range:
                for num_set_layers in num_set_layers_range:
                    model_params = {
                        "hidden_channels": hidden_channels,
                        "out_channels": out_channels,
                        "num_node_layers": num_node_layers,
                        "num_set_layers": num_set_layers,
                    }

                    # Create and configure the DeepSet model with the specified parameters
                    model = DeepSet(
                        in_channels=10,
                        hidden_channels=hidden_channels,
                        out_channels=out_channels,
                        num_node_layers=num_node_layers,
                        num_set_layers=num_set_layers,
                        norm=models_cfg["norm"]["values"][0],
                        activation=models_cfg["activation"]["values"][0],
                        skip_node=models_cfg["skip_node"]["values"][0],
                        skip_set=models_cfg["skip_set"]["values"][0],
                    )

                    # Log model parameters and its size to wandb
                    num_params = count_parameters(model)
                    wandb.log({**model_params, "model_size": num_params})


if __name__ == "__main__":
    main()
