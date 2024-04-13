import torch
from torch.utils.data import Dataset

import os
import numpy as np
from tqdm import tqdm


class BasePianorollDataset(Dataset):
    """
    Base Dataset class for the MAESTRO dataset.
    """

    def __init__(self, data_path: str, n_notes: int = 16):
        self.data_path = data_path
        self.n_notes = n_notes


class PianorollDataset(BasePianorollDataset):
    """
    Base Dataset class for the MAESTRO dataset.
    """

    def __init__(self, data_path: str, n_notes: int = 16):
        super().__init__(data_path, n_notes)
        self.dataset = self.get_dataset()

    def get_dataset(self) -> list[np.ndarray]:
        """
        Loads all dataset into memory and splits songs into nbar chunks.
        """
        dataset: list[np.ndarray] = []

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


class PianorollGanCNNDataset(BasePianorollDataset):
    """
    Dataset class for the MAESTRO dataset for the CNN Gan model.
    """

    def __init__(self, data_path: str, n_notes: int = 16):
        super().__init__(data_path, n_notes)
        self.pitch_dim = np.load(
            os.path.join(data_path, os.listdir(data_path)[0])
        ).shape[1]

        self.silence = np.zeros((self.n_notes, self.pitch_dim), dtype=np.float32)
        self.dataset = self.get_dataset()

    def get_dataset(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Loads all dataset into memory and splits songs into nbar chunks.
        """
        dataset: list[tuple[np.ndarray, np.ndarray]] = []

        for file_path in tqdm(os.listdir(self.data_path)):
            pianoroll: np.ndarray = np.load(os.path.join(self.data_path, file_path))

            n = pianoroll.shape[0] // self.n_notes

            track = [self.silence] + [
                pianoroll[i * self.n_notes : (i + 1) * self.n_notes] for i in range(n)
            ]

            # Group the current and previous bar (real, prev)
            dataset.extend(zip(track[1:], track))

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.dataset[idx][0], dtype=torch.float32), torch.tensor(
            self.dataset[idx][1], dtype=torch.float32
        )
