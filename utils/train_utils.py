import torch
from tqdm.auto import tqdm


def get_correct(pred, target):
    predictions = pred.argmax(axis=1)
    return (target == predictions).sum()

    
def validation_loop(model, validation_dataloader, criterion, epoch, verbose=False):
    device = model.device
    dataset_size = len(validation_dataloader.dataset)
    running_loss = 0
    running_accuracy = 0
    model.eval()
    # For computational efficiency, we turn off gradients during validation
    with torch.no_grad():
        for spectrograms, target in tqdm(validation_dataloader, desc=f"Validation Epoch {epoch}"):
            spectrograms = spectrograms.to(device)
            target = target.to(device)
            pred = model(spectrograms)
            loss = criterion(pred, target)
            correct = get_correct(pred, target)
            running_loss += loss.item() * target.shape[0]
            running_accuracy += correct
    
    avg_loss = running_loss / dataset_size
    avg_acc = running_accuracy / dataset_size
    if verbose:
        print(f"Epoch {epoch} validation loss: {avg_loss:.3f} | validation acc {avg_acc:.3f}")
            
    return avg_loss, avg_acc
    

def training_loop(model, train_dataloader, criterion, optimizer, epoch, verbose=False):
    device = model.device
    dataset_size = len(train_dataloader.dataset)
    running_loss = 0
    running_accuracy = 0
    model.train()
    for spectrograms, target in tqdm(train_dataloader):
        spectrograms = spectrograms.to(device)
        target = target.to(device)
        pred = model(spectrograms)
        loss = criterion(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += (loss.item() * target.shape[0])
        running_accuracy += get_correct(pred, target)
    
    avg_loss = running_loss / dataset_size
    avg_acc = running_accuracy / dataset_size
    if verbose:
        print(f"Epoch {epoch} training loss: {avg_loss:.3f} | training acc {avg_acc:.3f}")

    return avg_loss, avg_acc