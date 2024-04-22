import torch
from datetime import datetime
import matplotlib.pyplot as plt


try:
    from src.transformer import TransformerDecoder, CustomBCELoss
    from src.utils import parameters_to_double, set_seed, load_model
    from src.data import DATA_PATH
    from src.datasets import PianorollTransformerDataset
    from src.train_functions_transformer import train_loop_transformer
except:
    from transformer import TransformerDecoder, CustomBCELoss
    from utils import parameters_to_double, set_seed
    from data import DATA_PATH
    from datasets import PianorollTransformerDataset

from torch.utils.data import Dataset, DataLoader, random_split


# Set the seed
set_seed(42)

device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
# device = torch.device("cpu")
print(f"Device: {device}")


def get_random_sample() -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function returns a random sample from the data.

    Returns:
        image: random image.
    """
    pianoroll_dataset = PianorollTransformerDataset(DATA_PATH, n_notes=160)
    train_loader: DataLoader = DataLoader(
        pianoroll_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    idx: int = torch.randint(0, len(train_loader), (1,))
    for i, (x, target) in enumerate(train_loader):
        if i == idx:
            return x, target

    return


@torch.no_grad()
def test_generate(pitch_dim: int):
    """
    This function performs inference of the model.

    Args:
        model: model to train.
        device: device for running operations

    Returns:
        output: output of the model.
    """

    # Load the model
    model_name: str = "transformer_2024-04-21_00-13-24/autosaves/model_val_loss_0.050"
    model_state_dict: TransformerDecoder = load_model(model_name).state_dict()

    model: TransformerDecoder = TransformerDecoder(
        pitch_dim, num_heads=8, hidden_dim=512, num_layers=5, ctx_size=160
    )
    model.load_state_dict(model_state_dict)
    model.to(device)
    parameters_to_double(model)
    model.eval()

    seq_length: int = 13

    # Generate from scratch
    idx = torch.randint(35, 55, (1,))
    x = torch.zeros(1, 160, pitch_dim).to(device)
    x[:, :seq_length, idx] = 0.7
    matrix = model.generate(x, seq_length)
    plt.figure(figsize=(10, 10))
    plt.title("Generated from scratch")
    plt.imshow(
        matrix[0, 1:].cpu().detach().numpy().T,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="gray",
    )
    plt.plot()

    # Generate from random sample, with the first seq_length context
    _, target = get_random_sample()
    x = torch.zeros(1, 160, pitch_dim).to(device)
    x[:, :seq_length, :] = target[9, :seq_length, :]
    x = x.to(device)
    matrix = model.generate(x, seq_length)
    plt.figure(figsize=(10, 10))
    plt.title("Generated from random sample")
    plt.imshow(
        matrix[0, 1:].cpu().detach().numpy().T,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="gray",
    )
    plt.plot()

    top_values, top_indices = torch.topk(matrix, k=10, dim=-1)
    matrix = torch.zeros_like(matrix)
    matrix.scatter_(-1, top_indices, top_values)

    # Compare predictions
    x, target = get_random_sample()
    x = x.to(device)
    target = target.to(device)
    matrix = model(x)

    # Keep 10 greatest numbers and set the rest to 0 of dimension 1
    matrix[matrix < 0.05] = 0.0

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(
        matrix[0, 1:].cpu().detach().numpy().T,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="gray",
    )
    plt.title("Prediction")
    plt.subplot(2, 1, 2)
    plt.imshow(
        target[0, :-1].cpu().detach().numpy().T,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="gray",
    )
    plt.title("Target")
    plt.show()


def main() -> None:
    """
    This function is the main program for the training.
    """
    # Parameters
    batch_size: int = 64
    n_notes: int = 160

    # Load the data
    pianoroll_dataset = PianorollTransformerDataset(DATA_PATH, n_notes=n_notes)

    print(f"Dataset size: {len(pianoroll_dataset)}")

    # split train and validation
    train_size = int(0.8 * len(pianoroll_dataset))
    val_size = len(pianoroll_dataset) - train_size
    train_data: Dataset
    val_data: Dataset
    train_data, val_data = random_split(pianoroll_dataset, [train_size, val_size])

    train_loader: DataLoader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    val_loader: DataLoader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    # print(f"Inputs size: {train_loader[0].shape}")

    # Initialize the model
    model: TransformerDecoder = TransformerDecoder(
        88, num_heads=8, hidden_dim=512, num_layers=6, ctx_size=n_notes
    ).to(device)
    parameters_to_double(model)

    # Initialize the loss
    loss: torch.nn.Module = CustomBCELoss(torch.nn.BCEWithLogitsLoss())

    # Initialize the optimizer
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Initialize the learning rate scheduler
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1
    )

    # Initialize the tensorboard writer
    writer: torch.utils.tensorboard.SummaryWriter = (
        torch.utils.tensorboard.SummaryWriter("runs/transformer")
    )

    # Hyperparameters
    n_epochs: int = 50
    patience: int = 5
    model_name: str = f"transformer_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Train the model
    train_loop_transformer(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        lr_scheduler,
        writer,
        n_epochs,
        patience,
        model_name,
    )


if __name__ == "__main__":
    # main()
    test_generate(88)
