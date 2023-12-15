# torchcell/models/act.py
# [[torchcell.models.act]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/act.py
# Test file: torchcell/models/test_act.py

from torch import nn

act_register = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),  # TODO add params, look at graph gym
    "tanh": nn.Tanh(),
}
