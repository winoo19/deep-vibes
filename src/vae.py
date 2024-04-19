import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import DATA_PATH
from src.train_functions import train_loop
from src.utils import set_seed, save_model, parameters_to_double, load_model
from src.datasets import PianorollDataset

from typing import TypedDict
from datetime import datetime, timedelta
import cProfile

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
set_seed(42)


class Sigmoid(torch.nn.Module):
    """
    Sigmoid activation function with temperature.
    """

    def __init__(self, temperature: float = 1.0, trainable: bool = False):
        super(Sigmoid, self).__init__()
        self.trainable = trainable
        if trainable:
            self.temperature = torch.nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature

    def forward(self, x):
        return torch.sigmoid(x / self.temperature)


class VAE(torch.nn.Module):
    """
    Autoencoder model.

    Args:
        input_size (int): Size of the input tensor.
        encoder_hidden_size (int): Size of the hidden layer in the encoder.
        decoder_hidden_size (int): Size of the hidden layer in the decoder.
    """

    def __init__(
        self,
        input_size: int,
        embed_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        # Encoder
        self.encoder = torch.nn.GRU(
            input_size=input_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_layers,
            batch_first=True,
        )
        self.mean_linear = torch.nn.Linear(encoder_hidden_size, embed_size)
        self.logvar_linear = torch.nn.Linear(encoder_hidden_size, embed_size)

        # Decoder
        self.decoder = torch.nn.GRU(
            input_size=embed_size,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_layers,
            batch_first=True,
        )
        self.output_linear = torch.nn.Linear(decoder_hidden_size, input_size)
        self.sigmoid = Sigmoid(temperature=temperature, trainable=False)

        self.input_size = input_size
        self.embed_size = embed_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor. Dimensions: [batch_size, n_notes, 88]

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Tuple with the output, mean, std and logvar tensors.
        """
        # Encode
        _, hn = self.encoder(x)  # [1, batch_size, encoder_hidden_size]
        hn = hn.squeeze(0)  # [batch_size, encoder_hidden_size]
        mean = self.mean_linear(hn)  # [batch_size, embed_size]
        logvar = self.logvar_linear(hn)  # [batch_size, embed_size]

        # Reparametrization trick
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)  # [batch_size, embed_size]

        # Decode
        z_rep = z.unsqueeze(1).repeat(
            1, x.shape[1], 1
        )  # [batch_size, n_notes, embed_size]
        outputs, _ = self.decoder(z_rep)  # [batch_size, n_notes, decoder_hidden_size]
        x_hat = self.sigmoid(self.output_linear(outputs))  # [batch_size, n_notes, 88]

        return x_hat, mean, std, logvar

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                Tuple with the mean tensor and the logvar tensor.
        """

        h1 = self.encoder(x)
        mean = self.mean_linear(h1)
        logvar = self.logvar_linear(h1)

        return mean, logvar

    def generate(self, n_notes: int) -> torch.Tensor:
        """
        Decode the latent vector.

        Args:
            z (torch.Tensor): Latent vector.
        """
        z = torch.randn(1, self.embed_size).to(device)
        z_rep = z.unsqueeze(1).repeat(
            1, n_notes, 1
        )  # [batch_size, n_notes, embed_size]
        outputs, _ = self.decoder(z_rep)
        x_hat = self.sigmoid(self.output_linear(outputs))

        return x_hat


class VAELoss(torch.nn.Module):
    def __init__(self, gamma: float = 1e-3):
        super(VAELoss, self).__init__()
        self.bce_loss = torch.nn.BCELoss(reduction="mean")
        self.gamma = gamma

    def forward(self, x: tuple[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tuple[torch.Tensor]): Tuple with the output tensor, mean tensor and var tensor.
            y (torch.Tensor): Target tensor.
        """
        reconstruction_loss = self.bce_loss(x[0], y)
        mean, std, logvar = x[1], x[2], x[3]
        kl_divergence = 0.5 * torch.sum(mean**2 + std**2 - logvar - 1)

        return reconstruction_loss + self.gamma * kl_divergence


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
        "lr": 1e-3,
        "lr_step_size": 5,
        "lr_gamma": 0.2,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "embed_size": 128,
        "encoder_hidden_size": 512,
        "decoder_hidden_size": 512,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "temperature": 2.0,
        "balancing_gamma": 1e-3,
    }

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    pitch_dataset = PianorollDataset(DATA_PATH, n_notes=16 * 5)
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
    model: torch.nn.Module = VAE(
        input_size=88,
        embed_size=hyperparams["embed_size"],
        encoder_hidden_size=hyperparams["encoder_hidden_size"],
        decoder_hidden_size=hyperparams["decoder_hidden_size"],
        encoder_layers=hyperparams["encoder_layers"],
        decoder_layers=hyperparams["decoder_layers"],
        temperature=hyperparams["temperature"],
    ).to(device)
    parameters_to_double(model)

    # define loss and optimizer
    loss: torch.nn.Module = VAELoss(gamma=hyperparams["balancing_gamma"])
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


def validate() -> None:
    """
    Load model and data and predict the validation set.

    """

    # load model
    model_state_dict = load_model("model_val_loss_93738.182").state_dict()

    # load data
    pitch_dataset = PianorollDataset(DATA_PATH, n_notes=16 * 5)
    print(f"Data loaded. Number of samples: {len(pitch_dataset)}")

    # split train and validation
    train_size = int(0.8 * len(pitch_dataset))
    val_size = len(pitch_dataset) - train_size
    train_data: Dataset
    val_data: Dataset
    train_data, val_data = random_split(pitch_dataset, [train_size, val_size])

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")

    val_loader: DataLoader = DataLoader(
        val_data,
        batch_size=128,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    # define model
    model: torch.nn.Module = VAE(
        input_size=88,
        embed_size=16,
        encoder_hidden_size=512,
        decoder_hidden_size=512,
    )
    model.load_state_dict(model_state_dict)
    model.to(device)
    parameters_to_double(model)
    model.eval()

    # define loss
    loss: torch.nn.Module = VAELoss(gamma=1e-3)

    # validate
    val_loss = 0
    for x, y in tqdm(val_loader):
        x, y = x.to(device), y.to(device)
        x_hat, mean, std, logvar = model(x)
        loss_val = loss((x_hat, mean, std, logvar), y)
        val_loss += loss_val.item()

    val_loss /= len(val_loader)
    print(f"Validation loss: {val_loss}")

    return None


def test():
    model_state_dict = load_model("model_val_loss_93738.182").state_dict()

    input_size = 88
    embed_size = 16
    encoder_hidden_size = 512
    decoder_hidden_size = 512

    model = VAE(input_size, embed_size, encoder_hidden_size, decoder_hidden_size)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    print(model)

    # Generate
    n_notes = 16
    x_hat = model.generate(n_notes)

    print(x_hat.shape)

    matrix = x_hat.squeeze().detach().cpu().numpy().T

    print(matrix.shape, matrix.min(), matrix.max(), matrix.dtype)

    # Plot heatmat
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, aspect="auto", cmap="gray")
    plt.show()


if __name__ == "__main__":
    # main()
    # validate()
    # test()

    cProfile.run("main()", "profile2")

    import pstats
    from pstats import SortKey

    p = pstats.Stats("profile2")
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(50)
