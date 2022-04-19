import argparse
from model_opts import OPTIONS
import torch
from dataset.dataset import build_datasets, get_normalizer
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from utils.train_utils import inference_loop
from utils.visual_utils import plot_confusion

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
    
    preds, targets = inference_loop(net, test_dataloader, normalizer)
    confusion = confusion_matrix(targets, preds, normalize="all")
    precision = precision_score(targets, preds, average=None)
    recall = recall_score(targets, preds, average=None)
    f1 = f1_score(targets, preds, average=None)
    
    plot_confusion(confusion, test_dataset.classes, ModelClass.__name__, f"results/{ModelClass.__name__}_confusion.png")
    
    for class_, idx in test_dataset.class_to_idx.items():
        print(f"{class_}: Precision: {precision[idx]:.4f} | Recall: {recall[idx]:.4f} | F1: {f1[idx]:.4f}")
    
    print("=== Aggregate Statistics === ")
    print(f"Precision: {precision.mean():.4f} | Recall: {recall.mean():.4f} | F1: {f1.mean():.4f}")
    
