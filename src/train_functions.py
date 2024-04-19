# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

from src.utils import save_model

# other libraries
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")


def train_loop(
    model: torch.nn.Module,
    train_data: DataLoader,
    val_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LRScheduler,
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
        val_loss = val_step(model, val_data, loss, writer, epoch, device)

        # Update learning rate
        lr_scheduler.step()

        # save best model if losses improves
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_model(
                model,
                f"model_val_loss_{best_val_loss:.3f}",
                f"models/{model_name}/autosaves",
            )
            print(
                f"\nEpoch {epoch}: "
                f"val_loss = {val_loss:.4f}. "
                "Improved! Model saved."
            )
        else:
            print(f"\nEpoch {epoch}: val_loss = {val_loss:.4f}.")

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
    losses: list[float] = []

    pbar = tqdm(train_data, desc=f"Epoch {epoch}")
    for input_batch, target_batch in pbar:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        # Forward pass
        output_batch = model(input_batch)
        # Compute loss
        loss_batch = loss(output_batch, target_batch)
        # Resets gradients
        optimizer.zero_grad()
        # Backward pass
        loss_batch.backward()
        # Update parameters
        optimizer.step()

        # compute metrics
        losses.append(loss_batch.item())

        pbar.set_postfix(
            {
                "loss": f"{np.mean(losses):.4f}",
            }
        )

    # write on tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch)


@torch.enable_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> float:
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

    for input_batch, target_batch in val_data:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        # Forward pass
        output_batch = model(input_batch)
        # Compute loss
        loss_batch = loss(output_batch, target_batch)
        # compute metrics
        total_loss += loss_batch.item()

    total_loss /= len(val_data)

    # write on tensorboard
    writer.add_scalar("val/loss", total_loss, epoch)

    return total_loss
