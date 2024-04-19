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
        return torch.tensor(self.dataset[idx], dtype=torch.double), torch.tensor(
            self.dataset[idx], dtype=torch.double
        )


class PianorollDiskDataset(BasePianorollDataset):
    """
    Piano roll dataset loaded into a single numpy array.
    """

    def __init__(self, data_path: str, n_notes: int = 16):
        super().__init__(data_path, n_notes)
        self.index_to_file_and_idx = self.get_index_to_file_and_idx()

    def get_index_to_file_and_idx(self) -> list[tuple[str, int]]:
        index_to_file_and_idx: list[tuple[str, int]] = []

        for file_path in tqdm(os.listdir(self.data_path), desc="Indexing dataset"):
            # Read shape of npy without loading it using mmap_mode
            with open(os.path.join(self.data_path, file_path), "rb") as f:
                major, minor = np.lib.format.read_magic(f)
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
                n = shape[0] // self.n_notes

            index_to_file_and_idx.extend((file_path, i) for i in range(n))

        return index_to_file_and_idx

    def __len__(self):
        return len(self.index_to_file_and_idx)

    def __getitem__(self, idx):
        file_path, i = self.index_to_file_and_idx[idx]
        pianoroll: np.ndarray = np.load(os.path.join(self.data_path, file_path))

        return (
            torch.tensor(
                pianoroll[i * self.n_notes : (i + 1) * self.n_notes],
                dtype=torch.double,
            ),
            torch.tensor(
                pianoroll[i * self.n_notes : (i + 1) * self.n_notes],
                dtype=torch.double,
            ),
        )


class PianorollGanCNNDataset(BasePianorollDataset):
    """
    Dataset class for the MAESTRO dataset for the CNN Gan model.
    """

    def __init__(self, data_path: str, n_notes: int = 16):
        super().__init__(data_path, n_notes)
        self.dataset = self.get_dataset()

    def get_dataset(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Loads all dataset into memory and splits songs into nbar chunks.
        """
        dataset: list[tuple[np.ndarray, np.ndarray]] = []

        for file_path in tqdm(os.listdir(self.data_path)):
            pianoroll: np.ndarray = np.load(os.path.join(self.data_path, file_path))

            n = pianoroll.shape[0] // self.n_notes

            track = tuple(
                pianoroll[i * self.n_notes : (i + 1) * self.n_notes] for i in range(n)
            )

            # Group the current and previous bar (real, prev)
            dataset.extend(map(np.array, zip(track[1:], track)))

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.dataset[idx][0][0], dtype=torch.float32), torch.tensor(
            self.dataset[idx][1], dtype=torch.float32
        )


class PitchDataset(Dataset):
    """
    Dataset class for the MAESTRO dataset for the CNN Gan model.

    Args:
        data_path (str): Path to the dataset.
        percentage_notes (float): Percentage of notes to keep.
        np_seed (int): Seed for the random number generator.
    """

    def __init__(self, data_path: str, n_notes_per_song: int = 10, np_seed: int = None):
        self.data_path = data_path
        self.n_notes_per_song = n_notes_per_song
        if np_seed:
            np.random.seed(np_seed)
        self.dataset = self.get_dataset()

    def get_dataset(self) -> np.ndarray:
        """
        Loads all dataset into memory keeping only a percentage of notes of each song.
        """
        n_files = len(os.listdir(self.data_path))

        dataset: np.ndarray = np.zeros(
            (n_files * self.n_notes_per_song, 88), dtype=np.float32
        )

        for i, file_path in enumerate(tqdm(os.listdir(self.data_path))):
            pianoroll: np.ndarray = np.load(os.path.join(self.data_path, file_path))

            # Choose n_notes_per_file notes randomly
            if pianoroll.shape[0] < self.n_notes_per_song:
                continue

            indices = np.random.choice(
                range(pianoroll.shape[0]),
                self.n_notes_per_song,
            )

            dataset[i * self.n_notes_per_song : (i + 1) * self.n_notes_per_song] = (
                pianoroll[indices]
            )

        # Number of zero vectors
        n_zeros = np.sum(np.all(dataset == 0, axis=1))
        print(f"Number of zero examples: {n_zeros}")

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.dataset[idx], dtype=torch.double), torch.tensor(
            self.dataset[idx], dtype=torch.double
        )
