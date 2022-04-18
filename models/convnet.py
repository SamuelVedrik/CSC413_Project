import torch.nn as nn
from models.base_model import BaseModel

class ConvNet(BaseModel):
    def __init__(self, layer_opts, output_size):
        super().__init__()
        
        final_layer_size = layer_opts[-1]["out_channels"]
        self.conv_net = self.build_conv_layers(layer_opts)
        self.flatten = nn.Flatten()
        
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))
        self.fc = nn.Linear(in_features = final_layer_size * 10 * 10, out_features=output_size)
    
    def forward(self, x):
        x = self.conv_net(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.fc(x)
