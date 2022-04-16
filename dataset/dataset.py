import os
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import librosa
import torch


def build_datasets(root="genres", num_seconds_per_sample=5, mel_opts=None):
    
    if mel_opts is None:
        mel_opts = dict(n_fft=2048, n_mels=128)
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
        GTZANDataset(train_data, num_seconds_per_sample, mel_opts),
        GTZANDataset(valid_data, num_seconds_per_sample, mel_opts),
        GTZANDataset(test_data, num_seconds_per_sample, mel_opts),
    )

def get_normalizer():
    """
    Returns a function that normalizes data.
    """
    # The mean and std values are calculated as below: 
    # vals = torch.cat([train_dataset[i][0].flatten() for i in range(len(train_dataset))])
    # mean = vals.mean()
    # std = vals.std()
    mean = -21.5775
    std = 16.8580
    normalizer = lambda x: (x - mean) / (std)
    return normalizer

class GTZANDataset(Dataset):
    def __init__(self, files_df, num_seconds_per_sample, spectrogram_opts):
        self.files_df = files_df
        self.n = num_seconds_per_sample

        # Each audio sample is 30 seconds long. This is the number of samples created from one audio file
        # given that each sample is n seconds long.
        # We discard the last one in case it does not contain n seconds long of audio.
        self.samples_per_file = math.ceil(30 // num_seconds_per_sample) - 1

        self.classes = files_df["class"].unique()
        self.class_to_idx = {class_: idx for idx, class_ in enumerate(self.classes)}
        
        self.spectrogram_opts = spectrogram_opts

    def __getitem__(self, idx):
        file_idx, split = idx // self.samples_per_file, idx % self.samples_per_file
        path, class_ = self.files_df.iloc[file_idx]

        audio, sample_rate = librosa.load(path, duration=30.0)

        audio = audio[
            split * self.n * sample_rate : (split + 1) * self.n * sample_rate
        ]
        melspectrogram = self.convert_to_melspectrogram(audio, sample_rate, **self.spectrogram_opts)
        melspectrogram = torch.FloatTensor(melspectrogram).unsqueeze(0)
        
        return melspectrogram, self.class_to_idx[class_]

        
    def __len__(self):
        return self.files_df.shape[0] * self.samples_per_file

    @staticmethod
    def convert_to_melspectrogram(audio, sample_rate, n_fft, n_mels):
        
        hop_length = n_fft // 2
        melspectrogram = librosa.power_to_db(
            librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        )
        
        return melspectrogram