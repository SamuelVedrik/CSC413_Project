import torchaudio
import os
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


def build_datasets(root="genres", num_seconds_per_sample=5, transforms=None):
    classes = [
        genre for genre in os.listdir(root) if os.path.isdir(os.path.join(root, genre))
    ]
    data = []
    for genre in classes:
        path_name = os.path.join(root, genre)
        for filename in os.listdir(path_name):
            data.append([os.path.join(path_name, filename), genre])

    data = pd.DataFrame(data, columns=["filename", "class"])

    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["class"]
    )
    test_data, valid_data = train_test_split(
        test_data, test_size=0.5, random_state=42, stratify=test_data["class"]
    )

    return (
        GTZANDataset(train_data, num_seconds_per_sample, transforms),
        GTZANDataset(valid_data, num_seconds_per_sample, transforms),
        GTZANDataset(test_data, num_seconds_per_sample, transforms),
    )


class GTZANDataset(Dataset):
    def __init__(self, files_df, num_seconds_per_sample, transforms):
        self.files_df = files_df
        self.n = num_seconds_per_sample
        
        # Each audio sample is 30 seconds long. This is the number of samples created from one audio file
        # Given that each sample is n seconds long.
        self.samples_per_file = 30 // num_seconds_per_sample
        self.transforms = transforms
        
        self.classes = files_df["class"].unique()
        self.class_to_idx = {class_: idx for idx, class_ in enumerate(self.classes)}
        
    def __getitem__(self, idx):
        file_idx, split = idx // self.samples_per_file, idx % self.samples_per_file
        path, class_ = self.files.iloc[file_idx]
        audio, sample_rate = torchaudio.load(path)
        audio = audio[0, split * self.n * sample_rate:(split+1) * self.n * sample_rate]
        if self.transforms:
            audio = self.transforms(audio)
        return audio, self.class_to_idx[class_]

    def __len__(self):
        return self.files_df.shape[0] * self.samples_per_file