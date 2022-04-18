from models.convnet import ConvNet
import torch
from torch import nn
from dataset.dataset import build_datasets, get_normalizer
from torch.utils.data import DataLoader
from models.crnn import CRNN
from utils.train_utils import training_loop, validation_loop
from utils.visual_utils import plot_accuracies, plot_losses
import os

# Options for the model type

# ==== CRNN ====
ModelClass = CRNN
MODEL_OPTS = dict(
    layer_opts=[
        dict(in_channels=1, out_channels=64),
        dict(in_channels=64, out_channels=64),
        dict(in_channels=64, out_channels=128),
        dict(in_channels=128, out_channels=256),
        ],
    gru_hidden_size=30,
    output_size=10
)

if __name__ == "__main__":
    mel_opts= dict(n_fft=800, n_mels=128)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    train_dataset, valid_dataset, test_dataset = build_datasets(root="genres",
                                                                num_seconds_per_sample=5,
                                                                mel_opts=mel_opts)
    
    
    normalizer = get_normalizer()
    print(f"Dataset Sizes: Train {len(train_dataset)} | Validation {len(valid_dataset)} | Test {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    validation_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=32)

    net = ModelClass(**MODEL_OPTS)
    net = net.to(device)
    
    print(f"Model Type: {ModelClass.__name__}| Num parameters: {net.num_parameters():,}")
    
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    EPOCHS = 15
    
    # Make a results folder
    
    if not os.path.exists("results/"):
        os.mkdir("results/")
    train_losses, train_accs, val_losses, val_accs  = [], [], [], []
    for epoch in range(EPOCHS):
        train_loss, train_acc = training_loop(net, train_dataloader, normalizer, criterion, optimizer, epoch, verbose=True)
        val_loss, val_acc = validation_loop(net, validation_dataloader, normalizer, criterion, epoch, verbose=True)
        
        # Saving the best model so far
        if max(*val_acc, val_acc) == val_acc:
            torch.save(net.state_dict(), f"results/{ModelClass.__name__}.pth")
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    test_loss, test_acc = validation_loop(net, train_dataloader, normalizer, criterion, epoch=None, test=True, verbose=True)
    plot_accuracies(train_accs, val_accs, f"results/{ModelClass.__name__}_accuracies.png")
    plot_losses(train_losses, val_accs,f"results/{ModelClass.__name__}_losses.png" )
    
    