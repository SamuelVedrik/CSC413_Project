from models.convnet import ConvNet
import torch
from torch import nn
from dataset import build_datasets, GTZANDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.train_utils import training_loop, validation_loop

if __name__ == "__main__":
    mel_opts= dict(n_fft=800, n_mels=128)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    train_dataset, valid_dataset, test_dataset = build_datasets(root="genres",
                                                                num_seconds_per_sample=5,
                                                                mel_opts=mel_opts)
    

    print(f"Dataset Sizes: Train {len(train_dataset)} | Validation {len(valid_dataset)} | Test {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    validation_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=32)

    net = ConvNet(1, len(train_dataset.classes))
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    EPOCHS = 50
    
    train_losses, train_accs, val_losses, val_accs  = [], [], [], []
    for epoch in range(EPOCHS):
        train_loss, train_acc = training_loop(net, train_dataloader, criterion, optimizer, epoch, verbose=True)
        val_loss, val_acc = validation_loop(net, validation_dataloader, criterion, epoch, verbose=True)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    
    