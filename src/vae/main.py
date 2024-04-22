# Torch
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

# Own modules
from src.data import DATA_PATH
from src.vae.train_functions import train_loop
from src.vae.train_functions import plot_generate, plot_original_reconstructed
from src.vae.models import CNNVAE, VAELoss, GammaScheduler
from src.utils import set_seed, save_model, parameters_to_double, load_model
from src.datasets import BinaryPianorollDataset

# Other
from typing import TypedDict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cProfile

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


class HyperParams(TypedDict):
    epochs: int
    patience: int
    lr: float
    lr_step_size: int
    lr_gamma: float
    weight_decay: float
    batch_size: int
    embed_size: int
    gamma: float
    gamma_zero_epochs: int
    gamma_min_exponent: int
    gamma_max_exponent: int
    gamma_real_epochs: int


def main() -> None:
    """
    This function is the main program for training.
    """

    # TODO
    hyperparams: HyperParams = {
        "epochs": 500,
        "patience": 10,
        "lr": 5e-3,
        "lr_step_size": 5,
        "lr_gamma": 0.75,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "embed_size": 1024,
        "gamma": 0.0,
        "gamma_zero_epochs": 1,
        "gamma_min_exponent": -4,
        "gamma_max_exponent": -2,
        "gamma_real_epochs": 1,
    }

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    pitch_dataset = BinaryPianorollDataset(DATA_PATH, n_notes=16 * 5)
    print(f"Data loaded. Number of samples: {len(pitch_dataset)}")

    # split train and validation
    train_size = int(0.8 * len(pitch_dataset))
    val_size = len(pitch_dataset) - train_size
    train_data: Dataset
    val_data: Dataset
    train_data, val_data = random_split(pitch_dataset, [train_size, val_size])

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")

    train_loader: DataLoader = DataLoader(
        train_data,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    val_loader: DataLoader = DataLoader(
        val_data,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    print("Data shape: ", train_loader.dataset[0][0].shape)

    # define name and writer
    timestamp: str = datetime.now().strftime("%Y%m%dT%H%M%S")
    model_name: str = f"model_vae_{timestamp}"
    writer: SummaryWriter = SummaryWriter(f"runs/{model_name}")

    # define model
    model: torch.nn.Module = CNNVAE(
        n_notes=16 * 5,
        n_features=88,
        embed_size=hyperparams["embed_size"],
    ).to(device)
    parameters_to_double(model)

    # define loss and optimizer
    loss: torch.nn.Module = VAELoss(gamma=hyperparams["gamma"])
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
    )
    lr_scheduler: LRScheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=hyperparams["lr_step_size"], gamma=hyperparams["lr_gamma"]
    )
    gamma_scheduler: GammaScheduler = GammaScheduler(
        loss=loss,
        zero_epochs=hyperparams["gamma_zero_epochs"],
        min_exponent=hyperparams["gamma_min_exponent"],
        max_exponent=hyperparams["gamma_max_exponent"],
        real_epochs=hyperparams["gamma_real_epochs"],
    )

    # train loop
    start_time: float = datetime.now().timestamp()
    try:
        train_loop(
            model,
            train_loader,
            val_loader,
            loss,
            optimizer,
            lr_scheduler,
            gamma_scheduler,
            writer,
            hyperparams["epochs"],
            hyperparams["patience"],
            model_name,
        )
    except KeyboardInterrupt:
        print("Training interrupted, testing model and saving...")

    # total training time
    end_time: float = datetime.now().timestamp()
    print(f"Total training time: {timedelta(seconds=int(end_time - start_time))}")

    # save model
    save_model(model, f"model_final", f"models/{model_name}")

    # Save hyperparameters
    with open(f"models/{model_name}/hyperparameters.txt", "w") as file:
        for key, value in hyperparams.items():
            file.write(f"{key}: {value}\n")

    return None


def get_results():
    """
    Loads a trained model and generate some samples.
    """
    model_state_dict = load_model(
        "model_vae_20240421T002345/autosaves/model_val_loss_0.939"
    ).state_dict()

    model = CNNVAE(
        n_notes=16 * 5,
        n_features=88,
        embed_size=1024,
    )
    model.load_state_dict(model_state_dict)
    parameters_to_double(model)
    model.to(device)
    model.eval()

    print("Model loaded!:")
    print(model)

    # Generate
    timestamp: str = datetime.now().strftime("%Y%m%dT%H%M%S")
    for i in range(5):
        plot_generate(
            model,
            f"images2/",
            f"generated_pianoroll_{timestamp}_{i}.png",
            show=False,
        )

    # Load a random sample
    dataset = BinaryPianorollDataset(DATA_PATH, n_notes=16 * 5)
    for _ in range(5):
        idx = torch.randint(0, len(dataset), (1,)).item()
        sample = dataset[idx][0]

        # Plot original and reconstructed
        plot_original_reconstructed(
            model, sample, f"images2/", f"original_reconstructed_{idx}.png", show=False
        )


def test_initialization():
    """
    Test initialization of the model.
    """
    model = CNNVAE(
        n_notes=16 * 5,
        n_features=88,
        embed_size=200,
    )
    model.to(device)
    model.eval()

    print("Model created!:")
    print(model)

    x = torch.randn(1, 16 * 5, 88).to(device)  # [batch_size, n_notes, 88]
    x_hat, mean, std = model(x)

    print(x_hat.shape)
    print(f"Mean: {mean.mean():.4f} +- {mean.std():.4f}")
    print(f"Std: {std.mean():.4f} +- {std.std():.4f}")

    loss = VAELoss(gamma=1)

    _, rec_loss, kl_loss = loss((x_hat, mean, std), x)

    print(f"Rec loss: {rec_loss.item()}")
    print(f"KL loss: {kl_loss.item()}")


if __name__ == "__main__":
    main()
    # get_results()
