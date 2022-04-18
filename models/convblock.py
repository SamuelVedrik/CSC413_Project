import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, maxpool_kernel=2):
        
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                    in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=kernel_size, 
                      padding=padding, 
                      stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(maxpool_kernel)
        )
    
    def forward(self, x):
        return self.net(x)

    