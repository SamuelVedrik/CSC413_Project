import torch.nn as nn
import torch.nn.functional as F
import torch
from models.base_model import BaseModel

class MCCRNN(BaseModel):
    def __init__(self, column1_opts, column2_opts, combined_opts, gru_hidden_size, output_size):
        super().__init__()
        
        self.conv_net1 = self.build_conv_layers(column1_opts)
        self.conv_net2 = self.build_conv_layers(column2_opts)
        self.conv_net3 = self.build_conv_layers(combined_opts)
        final_layer_size = combined_opts[-1]["out_channels"]
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 8))
        # N x 15 x 256 (Batch, sequence_len, feature_size)
        self.gru = nn.GRU(input_size=final_layer_size, hidden_size=gru_hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features=gru_hidden_size, out_features=output_size)
    
    def forward(self, x):
        
        x1 = self.conv_net1(x)
        x2 = self.conv_net2(x)
        x_comb = torch.cat([x1, x2], dim=1)
        x_comb = self.conv_net3(x_comb)
        out = self.avgpool(x_comb)
        # N x 256 x 1 x 15
        out = out.squeeze(2)
        # N x 256 x 15
        out = out.permute(0, 2, 1)
        # N x 15 x 256
        out, _ = self.gru(out)
        out = out[:, -1, :] # Use the last sequence
        return self.fc(out)