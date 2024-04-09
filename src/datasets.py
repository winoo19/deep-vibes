import os
from torch.utils.data import Dataset
import numpy as np


class MaestroPianorollDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, idx):
        file_path = os.listdir(self.data_path)[idx]
        pianoroll = np.load(os.path.join(self.data_path, file_path))
        return pianoroll
