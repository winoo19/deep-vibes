# deep learning libraries
from torch.utils.data import DataLoader, Dataset

# own libraries
from src.datasets import PianorollDataset, PianorollGanCNNDataset

# other libraries
import pypianoroll as ppr
import numpy as np
import os
import requests
import zipfile
import shutil
from tqdm import tqdm


DATA_PATH = "data"
RESOLUTION = 8


def download_data() -> None:
    """
    Load the MAESTRO MIDI dataset from the web.
    """

    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"

    response = requests.get(url)

    temp_path = os.path.join(DATA_PATH, "temp")
    zip_file_path = os.path.join(temp_path, "maestro-v3.0.0-midi.zip")

    # Create temp folder
    os.makedirs(temp_path, exist_ok=True)

    with open(zip_file_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(temp_path)

    # Create midi folder
    os.makedirs(os.path.join(DATA_PATH, "midi"), exist_ok=True)

    # Loop all folders and move files to data folder
    for root, _, files in os.walk(temp_path):
        for file in files:
            if file.endswith(".midi"):
                os.rename(
                    os.path.join(root, file), os.path.join(DATA_PATH, "midi", file)
                )

    # Remove temp folder
    shutil.rmtree(temp_path)


def transform_data() -> None:
    """
    This method transforms the data from midi to piano roll and saves it as a numpy file.
    """

    midi_path = os.path.join(DATA_PATH, "midi")
    npy_path = os.path.join(DATA_PATH, "npy")

    if not os.path.exists(midi_path):
        print("Data is not downloaded. Downloading data...")
        download_data()

    os.makedirs(npy_path, exist_ok=True)

    pbar = tqdm(os.listdir(midi_path))

    for i, file in enumerate(pbar):
        track = ppr.read(os.path.join(midi_path, file), resolution=RESOLUTION)

        # Keep only piano keys
        pianoroll = track.tracks[0].pianoroll[:, 21:109]
        np.save(os.path.join(npy_path, f"pianoroll_{i}.npy"), pianoroll)

    print("Data successfully transformed!")


def load_data(
    dataset: Dataset,
    n_notes: int = 16,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    This method creates a dataloader for the dataset.
    """
    npy_path = os.path.join(DATA_PATH, "npy")

    train_dataset = dataset(data_path=npy_path, n_notes=n_notes)

    data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return data_loader


if __name__ == "__main__":
    # download_data()
    # transform_data()
    data_loader = load_data(PianorollDataset, n_notes=16, batch_size=64)

    print("Dataset length:", len(data_loader.dataset))

    for i, data in enumerate(data_loader):
        print("Item:", i, data.shape)
        print(data[0])
        print("Non zero elements:", np.count_nonzero(data[0]))
        if i == 0:
            break
