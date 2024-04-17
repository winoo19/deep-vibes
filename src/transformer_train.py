import torch
from src.transformer import MyModel
from src.utils import parameters_to_double, set_seed
from src.data import load_data
from src.datasets import PianorollDataset, PianorollGanCNNDataset

from tqdm.auto import tqdm

# Set the seed
set_seed(42)
torch.set_num_threads(8)

DATA_PATH: str = "data"

device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def main() -> None:
    """
    This function is the main program for the training.
    """
