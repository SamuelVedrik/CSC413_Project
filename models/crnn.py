import torch.nn as nn
import torch.nn.functional as F
import torch
from models.base_model import BaseModel

class CRNN(BaseModel):
    def __init__(self, layer_opts, gru_hidden_size, output_size):
        super().__init__()
        
        self.conv_net = self.build_conv_layers(layer_opts)
        final_layer_size = layer_opts[-1]["out_channels"]
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 8))
        # N x 15 x 256 (Batch, sequence_len, feature_size)
        self.gru = nn.GRU(input_size=final_layer_size, hidden_size=gru_hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features=gru_hidden_size, out_features=output_size)
    
    def forward(self, x):
        
        x = self.conv_net(x)
        x = self.avgpool(x)
        # N x 256 x 1 x 15
        x = x.squeeze(2)
        # N x 256 x 15
        x = x.permute(0, 2, 1)
        # N x 15 x 256
        out, _ = self.gru(x)
        out = out[:, -1, :] # Use the last sequence
        return self.fc(out)
        