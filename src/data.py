# deep learning libraries
from torch.utils.data import DataLoader, Dataset

# own libraries
from src.datasets import PianorollDataset, PianorollGanCNNDataset
from src.midi import (
    midi2pianoroll,
    pianoroll2matrix,
    trim_silence,
)

# other libraries
import numpy as np
import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


DATA_FOLDER = "data"
FS = 16


def download_data() -> None:
    """
    Load the MAESTRO MIDI dataset from the web.
    """

    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"

    response = requests.get(url)

    temp_path = os.path.join(DATA_FOLDER, "temp")
    zip_file_path = os.path.join(temp_path, "maestro-v3.0.0-midi.zip")

    # Create temp folder
    os.makedirs(temp_path, exist_ok=True)

    with open(zip_file_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(temp_path)

    # Create midi folder
    os.makedirs(os.path.join(DATA_FOLDER, "midi"), exist_ok=True)

    # Loop all folders and move files to data folder
    for root, _, files in os.walk(temp_path):
        for file in files:
            if file.endswith(".midi"):
                os.rename(
                    os.path.join(root, file), os.path.join(DATA_FOLDER, "midi", file)
                )

    # Remove temp folder
    shutil.rmtree(temp_path)


def setup_data():
    """
    This method sets up the data folder.
    """

    if os.path.exists(os.path.join(DATA_FOLDER, "midi")):
        print("Data already extracted!")
        return

    # Check if src/data.py data/surname_checked_midis_v1.2.zip exists
    zip_file_path = os.path.join(DATA_FOLDER, "surname_checked_midis_v1.2.zip")
    if not os.path.exists(zip_file_path):
        print("Zip is missing!")
        return

    # Extract all midi files into the data folder
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(DATA_FOLDER)

    print("Data successfully extracted!")

    # Create midi folder
    os.makedirs(os.path.join(DATA_FOLDER, "midi"), exist_ok=True)

    # Loop all folders and move files to data folder
    for root, _, files in os.walk(os.path.join(DATA_FOLDER, "surname_checked_midis")):
        for file in files:
            if file.endswith(".mid"):
                output_file = os.path.join(
                    DATA_FOLDER, "midi", file.replace(".mid", ".midi")
                )
                os.rename(os.path.join(root, file), output_file)

    print("Data moved to midi folder!")


def get_most_common_composers(n_most_common: int = 10) -> tuple:
    """
    This method returns the 10 most common composers.
    """

    midi_path = os.path.join(DATA_FOLDER, "midi")

    composer_names = [
        fn.split(",")[1] + " " + fn.split(",")[0]
        for fn in os.listdir(midi_path)
        if fn.endswith(".midi") and "," in fn
    ]

    # x axis is unique composers, y axis is number of compositions
    composers, counts = np.unique(composer_names, return_counts=True)

    # Sort by counts
    indices = np.argsort(-counts)
    top_composers = composers[indices][:n_most_common]
    top_counts = counts[indices][:n_most_common]

    return top_composers, top_counts


def explore_data() -> None:
    """
    This method explores the data.
    """

    composers, counts = get_most_common_composers()

    plt.figure(figsize=(10, 7))
    sns.barplot(x=composers[:10], y=counts[:10], palette="viridis")
    plt.xticks(rotation=15)
    plt.title("10 most common composers")
    plt.show()


def transform_data() -> None:
    """
    This method transforms the data from midi to piano roll and saves it as a numpy file.
    """

    midi_path = os.path.join(DATA_FOLDER, "midi")
    npy_path = os.path.join(DATA_FOLDER, "npy")

    if os.path.exists(npy_path):
        print("Data already transformed!")
        return

    os.makedirs(npy_path, exist_ok=True)

    most_common_composers, _ = get_most_common_composers(n_most_common=10)

    # Skip if composer is not in most common composers
    valid_paths = [
        path
        for path in os.listdir(midi_path)
        if any(
            all(name in path for name in composer.split())
            for composer in most_common_composers
        )
        and path.endswith(".midi")
    ]

    pbar = tqdm(valid_paths, desc="Transforming data")
    for i, file in enumerate(pbar):
        pianoroll = midi2pianoroll(os.path.join(midi_path, file), fs=FS)
        matrix = pianoroll2matrix(pianoroll)
        matrix = trim_silence(matrix)

        np.save(os.path.join(npy_path, f"pianoroll_{i}.npy"), matrix)

    print("Data successfully transformed!")


def load_data(
    dataset_type: type[PianorollDataset],
    n_notes: int = 16,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    This method creates a dataloader for the dataset.
    """
    npy_path = os.path.join(DATA_FOLDER, "npy")

    train_dataset = dataset_type(npy_path, n_notes=n_notes)

    data_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return data_loader


if __name__ == "__main__":
    # download_data()
    # setup_data()
    # explore_data()
    transform_data()
    print("Nº songs:", len(os.listdir("data/npy")))

    # dataset = PianorollDataset
    # data_loader = load_data(dataset)
    # print(len(data_loader))

    # dataset = PianorollGanCNNDataset
    # data_loader = load_data(dataset)
    # print(len(data_loader))
    # input("Press Enter to continue...")
