# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Own modules
from src.utils import save_model
from src.vae.models import GammaScheduler
from src.midi import matrix2pianoroll, pianoroll2midi

# other libraries
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def plot_generate(
    model: torch.nn.Module, folder: str, filename: str, show: bool = False
) -> None:
    """
    Plot the generated pianoroll.

    Args:
        model: model to generate the pianoroll.
    """

    # Generate
    x_hat = model.generate()
    matrix = x_hat.squeeze().detach().cpu().numpy()
    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min()) > 0.3
    matrix = matrix.astype(float)

    # Plot heatmat
    fig = plt.figure()
    plt.imshow(matrix.T, aspect="auto", cmap="gray")
    plt.title("Generated pianoroll")

    # Create folder if it does not exist
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)

    if show:
        fig.savefig(os.path.join(folder, filename))
        plt.show()
    else:
        fig.savefig(os.path.join(folder, filename))
        plt.close()

    # Convert to midi
    pianoroll = matrix2pianoroll(matrix)
    pianoroll2midi(pianoroll, fs=16, save_path=f"{folder}/{filename}.mid")

    return None


def plot_original_reconstructed(
    model: torch.nn.Module,
    sample: torch.Tensor,
    folder: str,
    filename: str,
    show: bool = True,
) -> None:
    """
    Plot the original and reconstructed pianoroll.

    Args:
        model: model to reconstruct the pianoroll.
        sample: original pianoroll.
    """

    x_hat, _, _, _ = model(sample.unsqueeze(0).to(device))
    x_hat = torch.sigmoid(x_hat)
    matrix = x_hat.squeeze().detach().cpu().numpy()
    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min()) > 0.3
    matrix = matrix.astype(float)

    # Plot original and reconstructed heatmat
    _, axs = plt.subplots(2, 1)
    axs[0].imshow(sample.T, aspect="auto", cmap="gray")
    axs[1].imshow(matrix.T, aspect="auto", cmap="gray")
    plt.suptitle("Original vs reconstructed pianoroll")

    # Create folder if it does not exist
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)

    if show:
        plt.savefig(os.path.join(folder, filename))
        plt.show()
    else:
        plt.savefig(os.path.join(folder, filename))
        plt.close()

    # Save midis
    pianoroll = matrix2pianoroll(matrix)
    pianoroll2midi(pianoroll, fs=16, save_path=f"{folder}/{filename}_rec.mid")

    original_pianoroll = sample.numpy()
    pianoroll = matrix2pianoroll(original_pianoroll)
    pianoroll2midi(pianoroll, fs=16, save_path=f"{folder}/{filename}_orig.mid")

    return None


def train_loop(
    model: torch.nn.Module,
    train_data: DataLoader,
    val_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LRScheduler,
    gamma_scheduler: GammaScheduler,
    writer: SummaryWriter,
    epochs: int,
    patience: int,
    model_name: str,
) -> None:
    """
    This function is the main training loop.

    Args:
        model: pytorch model.
        train_data: train dataloader.
        val_data: validation dataloader.
        loss: loss function.
        optimizer: optimizer object.
        lr_scheduler: learning rate scheduler.
        writer: tensorboard writer.
        epochs: number of epochs to train.
        model_name: name of the model.
    """

    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0
    for epoch in range(epochs):
        # call train step
        model.train()
        train_step(model, train_data, loss, optimizer, writer, epoch, device)

        # call val step
        model.eval()
        val_loss, val_rec_loss, val_kl_loss, latent_means, latent_stds = val_step(
            model, val_data, loss, writer, epoch, device
        )

        # Update learning rate
        lr_scheduler.step()

        # Update gamma
        gamma_scheduler.step()

        # save results
        plot_generate(
            model, f"models/{model_name}/images/", f"generated_{epoch}", show=False
        )
        random_idx = np.random.randint(0, len(val_data.dataset))
        random_sample = val_data.dataset[random_idx][0]
        plot_original_reconstructed(
            model,
            random_sample,
            f"models/{model_name}/images/",
            f"orig_rec_{epoch}",
            show=False,
        )

        # save best model if losses improves
        real_loss = val_rec_loss + val_kl_loss
        if real_loss <= best_val_loss - 1e-4:
            best_val_loss = real_loss
            epochs_without_improvement = 0
            save_model(
                model,
                f"model_val_loss_{best_val_loss if real_loss < 1e6 else np.inf:.3f}",
                f"models/{model_name}/autosaves",
            )
            print(
                f"\nEpoch {epoch}: "
                f"val_loss = {val_loss:.4f} "
                f"val_rec_loss = {val_rec_loss:.4f}. "
                f"val_kl_loss = {val_kl_loss:.4f}. "
                "Improved! Model saved."
            )
            print(
                f"Latent mean: {np.mean(latent_means):.4f} +- {np.std(latent_means):.4f}"
            )
            print(
                f"Latent  std: {np.mean(latent_stds):.4f} +- {np.std(latent_stds):.4f}"
            )
            # print("Gamma: ", loss.gamma)
        else:
            print(
                f"\nEpoch {epoch}: val_loss = {val_loss:.4f} "
                f"val_rec_loss = {val_rec_loss:.4f} "
                f"val_kl_loss = {val_kl_loss:.4f}"
            )
            print(
                f"Latent mean: {np.mean(latent_means):.4f} +- {np.std(latent_means):.4f}"
            )
            print(
                f"Latent  std: {np.mean(latent_stds):.4f} +- {np.std(latent_stds):.4f}"
            )
            # print("Gamma: ", loss.gamma)

        # early stopping
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}"
            )
            break

    return None


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    # TODO
    total_loss: float = 0.0
    total_rec_loss: float = 0.0
    total_kl_loss: float = 0.0

    pbar = tqdm(train_data, desc=f"Epoch {epoch}")
    for step, (input_batch, target_batch) in enumerate(pbar, 1):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        # Forward pass
        output_batch = model(input_batch)
        # Compute loss
        loss_batch, rec_batch, kl_batch = loss(output_batch, target_batch)
        # Resets gradients
        optimizer.zero_grad()
        # Backward pass
        loss_batch.backward()
        # Update parameters
        optimizer.step()

        # compute metrics
        total_loss += loss_batch.item()
        total_rec_loss += rec_batch.item()
        total_kl_loss += kl_batch.item()

        kl_str = f"{total_kl_loss / step if total_kl_loss / step < 1e6 else np.inf:.4f}"
        pbar.set_postfix(
            {
                "loss": f"{total_loss / step:.4f}",
                "rec_loss": f"{total_rec_loss / step:.4f}",
                "kl_loss": kl_str,
            }
        )

    # write on tensorboard
    writer.add_scalar("train/loss", total_loss / len(train_data), epoch)


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> tuple[float, float, float, list[float], list[float]]:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """
    total_loss: float = 0.0
    rec_loss: float = 0.0
    kl_loss: float = 0.0

    latent_means: list[float] = []
    latent_stds: list[float] = []

    for input_batch, target_batch in val_data:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        # Forward pass
        output_batch = model(input_batch)
        # Compute loss
        loss_batch, rec_batch, kl_batch = loss(output_batch, target_batch)
        # compute metrics
        total_loss += loss_batch.item()
        rec_loss += rec_batch.item()
        kl_loss += kl_batch.item()

        latent_means.append(output_batch[1].mean().item())
        latent_stds.append(output_batch[2].mean().item())

    total_loss /= len(val_data)
    rec_loss /= len(val_data)
    kl_loss /= len(val_data)

    # write on tensorboard
    writer.add_scalar("val/loss", total_loss, epoch)

    return total_loss, rec_loss, kl_loss, latent_means, latent_stds
