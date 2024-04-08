# deep learning libraries
import torch
from torch.utils.data import Dataset, DataLoader


# own libraries
from utils import set_seed


# other libraries
import pypianoroll as ppr
import numpy as np
import os


set_seed(42)


class PianoRollDataset(Dataset):
    """
    This class represents a dataset of piano roll representations of songs.

    Attributes:
        dataset (torch.Tensor): The binary representation of the songs.
    """

    dataset: torch.Tensor

    def __init__(self, dataset: list[np.ndarray], bar_length: int = 16) -> None:
        """
        Initializes the PianoRollDataset.

        Args:
            dataset (list[np.ndarray]): The binary representation of the songs.
            It has the shape (n_songs, n_timesteps (different for each song), n_pitches).
            bar_length (int): The length of a bar in beats.
        """

        self.bar_length = bar_length

        self.silence = torch.zeros((self.bar_length, dataset[0].shape[1]))

        self.dataset = [bar for track in dataset for bar in self.process_bars(track)]

    def process_bars(self, track: np.ndarray) -> list:
        """
        Process a track splitting it into bars.

        Args:
            track (np.ndarray): The piano roll representation of the song.

        Returns:
            list: The piano roll representation of the song split into bars.
        """

        n_bars = track.shape[0] // self.bar_length
        new_track = torch.concatenate(
            self.silence
            + [
                torch.tensor(track[i * self.bar_length : (i + 1) * self.bar_length])
                for i in range(n_bars)
            ]
        )

        return list(zip(new_track, new_track[1:]))

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The number of tracks in the dataset.
        """

        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns an element from the dataset based on the index.

        Args:
            index (int): The index of the element.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The former and input bars of the song.
        """

        return self.dataset[index]


def load_data(
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    train_p: float = 0.7,
    val_p: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    This method returns Dataloaders of the chosen dataset.

    Args:
        batch_size (int): The size of the batch.
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Whether to drop the last batch if it's smaller than the batch size.
        num_workers (int): The number of worker threads for data loading.
        train_p (float): The percentage of data used for training.
        val_p (float): The percentage of data used for validation.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: The train, validation, and test dataloaders.
    """

    # Transform and load the data
    files = transform_data()

    # separate the data into train, validation and test
    files = list(np.random.permutation(files))

    n_files = len(files)
    n_train = int(n_files * train_p)
    n_val = int(n_files * val_p)

    df_train = files[:n_train]
    df_val = files[n_train : n_train + n_val]
    df_test = files[n_train + n_val :]

    # Create datasets
    train_dataset = PianoRollDataset(df_train)
    val_dataset = PianoRollDataset(df_val)
    test_dataset = PianoRollDataset(df_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def transform_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This method transforms the data to be used in the model.

    Returns:
        train data.
        val data.
        test data.
    """

    n_files = len(os.listdir("data/midis"))

    if not os.path.exists("data/npy_files"):

        os.makedirs("data/npy_files")

        for i, file in enumerate(os.listdir("data/midis")):
            if file.endswith(".mid"):
                try:
                    track = ppr.read("data/midis/" + file)

                except Exception as e:
                    print(f"{file} could not be read")
                    print(e)
                    continue

                if len(track.tracks) != 1:
                    print(f"{file} has more than one track")
                    continue

                track = track.binarize()

                np.save(f"data/npy_files/{file[:-4]}.npy", track.tracks[0].pianoroll)

            else:
                print(f"{file} is not a midi file")

            if i % 100 == 0:
                print(f"{i}/{n_files} files processed")

    files: list[np.ndarray] = []

    for file in enumerate(os.listdir("data/train")):
        files.append(np.load("data/npy_files/" + file))

    return files


if __name__ == "__main__":
    files = transform_data(0.7, 0.15)

    for i in range(20):
        track = files[i]
        print(f"Track {i+1}:")
        print(track)
        print()

    print(f"n files: {len(files)}")
