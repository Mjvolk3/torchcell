# src/torchcell/models/act.py
# [[src.torchcell.models.act]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/models/act.py
# Test file: src/torchcell/models/test_act.py

from torch import nn

act_register = {"relu": nn.ReLU(), "gelu": nn.GELU(), "sigmoid": nn.Sigmoid()}
