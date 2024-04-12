import torch
from torch.utils.data import Dataset

import os
import numpy as np
from tqdm import tqdm


class PianorollDataset(Dataset):
    """
    Dataset class for the MAESTRO dataset.
    """

    def __init__(self, data_path: str, n_notes: int = 16):
        self.data_path = data_path
        self.n_notes = n_notes
        self.dataset: list[np.ndarray] = self.get_dataset()

    def get_dataset(self) -> list[np.ndarray]:
        """
        Loads all dataset into memory and splits songs into nbar chunks.
        """
        dataset = []

        for file_path in tqdm(os.listdir(self.data_path)):
            pianoroll: np.ndarray = np.load(os.path.join(self.data_path, file_path))

            n = pianoroll.shape[0] // self.n_notes
            dataset.extend(
                pianoroll[i * self.n_notes : (i + 1) * self.n_notes] for i in range(n)
            )

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx], dtype=torch.float32)


class PianorollGanCNNDataset(Dataset):
    """
    Dataset class for the MAESTRO dataset for the CNN Gan model.
    """

    def __init__(self, data_path: str, n_notes: int = 16):
        self.data_path = data_path
        self.n_notes = n_notes
        self.dataset: list[np.ndarray] = self.get_dataset()

    def get_dataset(self) -> list[np.ndarray]:
        """
        Loads all dataset into memory and splits songs into nbar chunks.
        """
        dataset = []

        for file_path in tqdm(os.listdir(self.data_path)):
            pianoroll: np.ndarray = np.load(os.path.join(self.data_path, file_path))

            n = pianoroll.shape[0] // self.n_notes

            track = tuple(
                pianoroll[i * self.n_notes : (i + 1) * self.n_notes] for i in range(n)
            )

            dataset.extend(zip(track[1:], track))

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.dataset[idx][0], dtype=torch.float32), torch.tensor(
            self.dataset[idx][1], dtype=torch.float32
        )
