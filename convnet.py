import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from dataset import build_datasets, GTZANDataset
from torch.utils.data import DataLoader
from tqdm.tqdm import tqdm


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
        x = self.conv1(x)
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class Net(nn.Module):
    def __init__(self, input_in_channels, output):
        super().__init__()
        # 1 x 128 x 276
        self.layer1 = Layer(input_in_channels, 64)
        # 64 x 128//2 x 276//2 
        self.layer2 = Layer(64, 64)
        # 64 x 128//4 x 276//4
        self.layer3 = Layer(64, 128)
        # 128 x 128//8 x 276//8
        self.layer4 = Layer(128, 256)
        # 256 x 128//16 x 276//16
        self.flat = nn.Flatten()

        # Alternative 
        # self.avgpool = nn.AdaptiveAvgPool2d((10, 10))
        # self.fc = nn.Linear(in_features = 256 * 10 * 10, out_features=output)
        
        self.fc = nn.Linear(256 * (128//16) * (276//16), output)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flat(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    transforms = MelSpectrogram(sample_rate = 22050, n_fft=800, n_mels=128)
    train_dataset, valid_dataset, test_dataset = build_datasets(root="genres",
                                                                num_seconds_per_sample=5,
                                                                transforms=transforms)


    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    # BCELossWithLogits (Multi Label classification)

    # SGD, Adam, AdamW
    optimizer = nn.optim.Adam(net.parameters(), lr=1e-3)

    # weight = weight - (lr * gradient) 
    EPOCHS = 10
    
    for epoch in range(EPOCHS):
        running_loss = 0
        for batch in tqdm(train_dataloader):
            X, y = batch
            y_pred = net(X)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {running_loss / len(train_dataloader.dataset)}")