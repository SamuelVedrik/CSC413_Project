import argparse
from model_opts import OPTIONS
import torch
from dataset.dataset import build_datasets, get_normalizer
from torch.utils.data import DataLoader
from utils.train_utils import representation_loop
from sklearn.manifold import TSNE
from utils.visual_utils import plot_representations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument("model", choices=["convnet", "crnn", "mccrnn"], help="model to choose from")
    args = parser.parse_args()
    ModelClass, model_opts = OPTIONS[args.model]["model_class"], OPTIONS[args.model]["model_opts"]
    mel_opts= dict(n_fft=800, n_mels=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    
    train_dataset, valid_dataset, test_dataset = build_datasets(root="genres",
                                                                num_seconds_per_sample=5,
                                                                mel_opts=mel_opts)
    
    
    normalizer = get_normalizer(train_dataset)
    print(f"Dataset Sizes: Train {len(train_dataset)} | Validation {len(valid_dataset)} | Test {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=32, num_workers=4)
    
    net = ModelClass(**model_opts)
    net = net.to(device)
    net.load_state_dict(torch.load(f"results/{ModelClass.__name__}.pth"))
    
    logits, targets = representation_loop(net, test_dataloader, normalizer)
    logits_small = TSNE(n_components=2).fit_transform(logits)
    plot_representations(logits_small, targets, test_dataset.classes, ModelClass.__name__, f"results/{ModelClass.__name__}_representations.png")
    
