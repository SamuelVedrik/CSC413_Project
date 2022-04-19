import argparse
from model_opts import OPTIONS
import torch
from dataset.dataset import build_datasets, get_normalizer
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from utils.train_utils import inference_loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument("model", choices=["convnet", "crnn", "mccrnn"], help="model to choose from")
    args = parser.parse_args()
    ModelClass, model_opts = OPTIONS[args.model]["model_class"], OPTIONS[args.model]["model_opts"]
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
    confusion = confusion_matrix(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    print(confusion)
    print(precision)
    print(recall)
    
