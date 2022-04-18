import torch.nn as nn
import torch.nn.functional as F
import torch

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 1 X (W x H)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        # 64 X (W x H)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # 64 X (W X H)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CRNN(nn.Module):
    def __init__(self, in_channels, output):
        super().__init__()
        
        self.layer1 = Layer(in_channels, 64)
        self.layer2 = Layer(64, 64)
        self.layer3 = Layer(64, 128)
        self.layer4 = Layer(128, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 15))
        # N x 15 x 256 (Batch, sequence_len, feature_size)
        self.gru = nn.GRU(input_size=256, hidden_size=30, batch_first=True)
        self.fc = nn.Linear(in_features=30, out_features=output)
    
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # N x 256 x 1 x 15
        x = x.squeeze(2)
        # N x 256 x 15
        x = x.permute(0, 2, 1)
        # N x 15 x 256
        out, h_n = self.gru(x)
        out = out[:, -1, :] # Use the last sequence
        return self.fc(out)
    
    @property
    def device(self):
        return next(self.parameters()).device
        