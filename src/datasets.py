import torch
from torch.utils.data import Dataset

import os
import numpy as np


class MaestroPianorollDataset(Dataset):
    """
    Dataset class for the MAESTRO dataset.
    """

    def __init__(self, data_path: str, nbars: int = 2, resolution: int = 8):
        self.data_path = data_path
        self.bar_length = nbars * resolution
        self.dataset: list[np.ndarray] = self.get_dataset()

    def get_dataset(self) -> list[np.ndarray]:
        """
        Loads all dataset into memory and splits songs into nbar chunks.
        """
        dataset = []

        for file_path in os.listdir(self.data_path):
            pianoroll = np.load(os.path.join(self.data_path, file_path))
            n = pianoroll.shape[0] // self.bar_length
            for i in range(n):
                start = i * self.bar_length
                end = start + self.bar_length
                dataset.append(pianoroll[start:end])

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx], dtype=torch.float32)
