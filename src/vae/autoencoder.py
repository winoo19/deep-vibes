import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

# own libraries
from src.data import DATA_PATH
from src.utils import set_seed, save_model, parameters_to_double, load_model
from src.datasets import PitchDataset
from src.vae.models import Autoencoder, AELoss
from src.vae.train_functions_ae import train_loop

# other libraries
from typing import TypedDict
from datetime import datetime, timedelta

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
    ecoder_hidden_size: int
    decoder_hidden_size: int
    embed_size: int


def main() -> None:
    """
    This function is the main program for training.
    """

    # TODO
    hyperparams: HyperParams = {
        "epochs": 500,
        "patience": 10,
        "lr": 8e-4,
        "lr_step_size": 5,
        "lr_gamma": 0.2,
        "weight_decay": 3e-3,
        "batch_size": 128,
        "encoder_hidden_size": 512,
        "decoder_hidden_size": 512,
        "embed_size": 16,
    }

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    pitch_dataset = PitchDataset(DATA_PATH, n_notes_per_song=50, np_seed=42)
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

    # define name and writer
    timestamp: str = datetime.now().strftime("%Y%m%dT%H%M%S")
    model_name: str = f"model_ae_{timestamp}"
    writer: SummaryWriter = SummaryWriter(f"runs/{model_name}")

    # define model
    model: torch.nn.Module = Autoencoder(
        input_size=88,
        embed_size=hyperparams["embed_size"],
        encoder_hidden_size=hyperparams["encoder_hidden_size"],
        decoder_hidden_size=hyperparams["decoder_hidden_size"],
    ).to(device)
    parameters_to_double(model)

    # define loss and optimizer
    loss: torch.nn.Module = AELoss(torch.nn.BCELoss())
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
    )
    lr_scheduler: LRScheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=hyperparams["lr_step_size"], gamma=hyperparams["lr_gamma"]
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


def test():
    # Load model
    model = load_model("model_ae_20240422T195628/autosaves/model_val_loss_0.042").to(
        device
    )
    model.eval()

    # C chord medium velocity
    C = torch.zeros(88, dtype=torch.double).to(device)
    C[24] = 0.6
    C[28] = 0.6
    C[31] = 0.6

    print("C chord medium velocity:")
    output, encoded = model(C)
    C_decoded = output.cpu().detach().numpy().round(2)
    print(C[[24, 28, 31]].cpu().detach().numpy().round(2))
    print(C_decoded[[24, 28, 31]])
    print(C_decoded.sum() - C_decoded[[24, 28, 31]].sum())
    print(encoded.cpu().detach().numpy().round(2))

    # G chord piano
    G = torch.zeros(88, dtype=torch.double).to(device)
    G[31] = 0.2
    G[38] = 0.2
    G[43] = 0.2
    G[47] = 0.2
    G[50] = 0.2

    print("\nG chord piano:")
    output, encoded = model(G)
    G_decoded = output.cpu().detach().numpy().round(2)
    print(G[[31, 38, 43, 47, 50]].cpu().detach().numpy().round(2))
    print(G_decoded[[31, 38, 43, 47, 50]])
    print(G_decoded.sum() - G_decoded[[31, 38, 43, 47, 50]].sum())
    print(encoded.cpu().detach().numpy().round(2))

    # D chord forte
    D = torch.zeros(88, dtype=torch.double).to(device)
    D[26] = 0.9
    D[31] = 0.9
    D[38] = 0.9
    D[43] = 0.9

    print("\nD chord forte:")
    output, encoded = model(D)
    D_decoded = output.cpu().detach().numpy().round(2)
    print(D[[26, 31, 38, 43]].cpu().detach().numpy().round(2))
    print(D_decoded[[26, 31, 38, 43]])
    print(D_decoded.sum() - D_decoded[[26, 31, 38, 43]].sum())
    print(encoded.cpu().detach().numpy().round(2))

    # Silence
    S = torch.zeros(88, dtype=torch.double).to(device)

    print("\nSilence:")
    output, encoded = model(S)
    S_decoded = output.cpu().detach().numpy().round(2)
    print(S.sum().item())
    print(S_decoded.sum())
    print(encoded.cpu().detach().numpy().round(2))


if __name__ == "__main__":
    # main()
    test()
