import torch
from datetime import datetime

try:
    from src.transformer import TransformerDecoder, CustomBCELoss
    from src.utils import parameters_to_double, set_seed, load_model
    from src.data import load_data, DATA_PATH
    from src.datasets import PianorollTransformerDataset
    from src.train_functions_transformer import train_loop_transformer
except:
    from transformer import TransformerDecoder, CustomBCELoss
    from utils import parameters_to_double, set_seed
    from data import load_data, DATA_PATH
    from datasets import PianorollTransformerDataset

from torch.utils.data import Dataset, DataLoader, random_split


# Set the seed
set_seed(42)

# device: torch.device = (
#     torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# )
device = torch.device("cpu")
print(f"Device: {device}")


@torch.no_grad()
def inference(pitch_dim: int):
    """
    This function performs inference of the model.

    Args:
        model: model to train.
        device: device for running operations

    Returns:
        output: output of the model.
    """
    model: TransformerDecoder = load_model("model_val_loss_0.047").to(device)
    model.eval()

    x = torch.zeros(1, 160, pitch_dim).to(device)

    for seq_length in range(pitch_dim):
        x = model.inference(x, seq_length)

    return x


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
    loss: torch.nn.Module = CustomBCELoss(torch.nn.BCELoss())

    # Initialize the optimizer
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Initialize the learning rate scheduler
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 0.99**epoch
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
    main()
