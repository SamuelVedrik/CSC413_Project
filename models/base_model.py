import torch.nn as nn
import torch.nn.functional as F
import torch
from abc import ABC, abstractmethod
from models.convblock import ConvBlock

class BaseModel(ABC, nn.Module):
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x, *args):
        pass
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def build_conv_layers(self, layer_opts):
        """
        layer_opts: A list of dictionary with the options to pass for each block.
        """

        layers = [ConvBlock(**kwargs) for kwargs in layer_opts]
        return nn.Sequential(*layers)